"""Mapping-driven Shader Profiler evidence queries (not a validated Nsight dialect).

Only stdlib is required. Layouts are tied to exact export bytes. All capture,
scope and symbol metadata is declared evidence, never inferred from filenames.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sys

VERSION = 1
METRICS = ("self_samples", "inclusive_samples", "dependency_samples", "live_registers")
TEXT_FIELDS = ("module", "entry", "file", "line", "code")
LIMITATIONS = [
    "Mapping-driven import; native Nsight CSV dialect is not yet validated.",
    "Capture identity, scope and source correlation are caller-declared, not verified.",
    "Samples are not GPU time; source and IL representations are not additive.",
    "Live registers are not allocated registers or occupancy.",
    "Exported rows do not establish complete source or instruction coverage.",
]


def fail(message):
    raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def records(data, delimiter=",", encoding="utf-8-sig"):
    if delimiter not in (",", "\t", ";"):
        fail("Unsupported delimiter")
    if not data or len(data) > 256 * 1024 * 1024:
        fail("Export must contain 1..256 MiB of data")
    if encoding not in ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"):
        fail("Unsupported encoding; transcode explicitly and retain the original")
    csv.field_size_limit(16 * 1024 * 1024)
    reader = csv.reader(io.StringIO(data.decode(encoding), newline=""),
                        delimiter=delimiter, strict=True)
    result, previous = [], 0
    for number, cells in enumerate(reader, 1):
        result.append({"record": number, "physical_lines": [previous + 1, reader.line_num],
                       "cells": cells})
        previous = reader.line_num
    return result


def inspect_export(data, delimiter, encoding, start, count):
    if start < 1 or count < 1 or count > 100:
        fail("Use start >= 1 and count in 1..100")
    rows = records(data, delimiter, encoding)
    widths = {}
    for row in rows:
        width = str(len(row["cells"]))
        widths[width] = widths.get(width, 0) + 1
    return {"sha256": sha(data), "record_count": len(rows), "width_histogram": widths,
            "preview": rows[start - 1:start - 1 + count],
            "note": "Record numbers are 1-based; physical lines can span quoted newlines."}


def number(value):
    if value == "":
        return None
    if not re.fullmatch(r"[0-9]+", value):
        fail("Expected exact nonnegative integer or empty cell, got " + repr(value))
    return int(value)


def validate_context(context):
    if not isinstance(context, dict):
        fail("Context must be an object")
    required = ("artifact_sha256", "nsight_version", "export_id", "selection")
    if any(not context.get(key) for key in required):
        fail("Context requires artifact_sha256, nsight_version, export_id, selection")
    if not re.fullmatch(r"[a-f0-9]{64}", context["artifact_sha256"]):
        fail("artifact_sha256 must be a lowercase SHA256")
    selection = context["selection"]
    if not isinstance(selection, dict) or any(not selection.get(k) for k in
                                               ("id", "requested_scope", "achieved_scope")):
        fail("selection requires id, requested_scope and achieved_scope")
    if selection["achieved_scope"] not in ("dispatch", "shader-in-window", "marker-range", "whole-device"):
        fail("Unknown achieved_scope")
    if selection["requested_scope"] != selection["achieved_scope"]:
        fail("Requested/achieved scope mismatch; narrow selection or record a separate case")


def normalize(data, layout, context):
    validate_context(context)
    if not isinstance(layout, dict):
        fail("Layout must be an object")
    if layout.get("version") != VERSION or layout.get("raw_sha256") != sha(data):
        fail("Layout version or raw SHA256 mismatch")
    rows = records(data, layout.get("delimiter", ","), layout.get("encoding", "utf-8-sig"))
    sections = layout.get("sections")
    if not isinstance(sections, list) or not sections:
        fail("At least one explicitly bounded table section is required")
    used, unique = set(), {}
    for section in sections:
        if not isinstance(section, dict):
            fail("Section must be an object")
        header_record, end = section["header_record"], section["end_record"]
        if type(header_record) is not int or type(end) is not int or not 1 <= header_record < end <= len(rows):
            fail("Invalid section bounds")
        covered = set(range(header_record, end + 1))
        if used & covered:
            fail("Sections overlap")
        used.update(covered)
        header = rows[header_record - 1]["cells"]
        if not header or section["header"] != header:
            fail("Exact header mismatch")
        rep = section["representation"]
        if rep not in ("il", "source", "summary"):
            fail("representation must be il, source or summary")
        columns = section["columns"]
        allowed = set(TEXT_FIELDS + METRICS)
        if not isinstance(columns, dict) or set(columns) - allowed or not {"module", "entry"} <= columns.keys():
            fail("Unknown columns or missing module/entry mapping")
        identity = section.get("identity_columns")
        if not isinstance(identity, list) or not identity:
            fail("identity_columns must define a stable row identity (include callsite when applicable)")
        stall_columns = section.get("stalls", [])
        if not isinstance(stall_columns, list) or any(
                not isinstance(pair, dict) or set(pair) != {"name", "samples"} for pair in stall_columns):
            fail("stalls must contain name/samples column pairs")
        stall_coverage = section.get("stall_coverage")
        if stall_columns and stall_coverage not in ("top-k-lower-bound", "complete"):
            fail("Stall columns require explicit stall_coverage")
        indexes = list(columns.values()) + identity + [i for pair in stall_columns for i in pair.values()]
        if any(type(index) is not int or not 0 <= index < len(header) for index in indexes):
            fail("Column index out of bounds")
        # Columns may have duplicate names. Positional mapping retains their identity.
        for raw in rows[header_record:end]:
            cells = raw["cells"]
            if not cells:  # Empty records in a table have no metrics.
                continue
            if len(cells) != len(header):
                fail("Row width mismatch at record " + str(raw["record"]))
            item = {name: cells[columns[name]] if name in columns else None for name in TEXT_FIELDS}
            if not item["module"]:
                fail("Missing module identity at record " + str(raw["record"]))
            # Empty entry is explicit unknown, never invented from a filename.
            item["entry"] = item["entry"] or None
            item.update({name: number(cells[columns[name]]) if name in columns else None for name in METRICS})
            row_identity = tuple(cells[index] for index in identity)
            if not any(row_identity):
                fail("Empty row identity")
            stalls = []
            for pair in stall_columns:
                name, samples = cells[pair["name"]], number(cells[pair["samples"]])
                if not name and samples is not None:
                    fail("Stall samples have no reason name")
                if name:
                    stalls.append({"name": name, "samples": samples})
            item.update(representation=rep, identity=list(row_identity),
                        stalls=stalls, stall_coverage=stall_coverage)
            key = (item["module"], item["entry"], rep, row_identity)
            evidence = {"record": raw["record"], "physical_lines": raw["physical_lines"],
                        "header_record": header_record, "columns": columns}
            if key in unique:
                previous = {k: v for k, v in unique[key].items() if k != "evidence"}
                if previous != item:
                    fail("Conflicting duplicate identity at record " + str(raw["record"]))
                unique[key]["evidence"].append(evidence)
            else:
                item["evidence"] = [evidence]
                unique[key] = item
    if not unique:
        fail("No mapped data rows")
    unmapped = [r for r in rows if r["record"] not in used and any(r["cells"])]
    return {"version": VERSION, "kind": "metallic.nsight.source", "raw_sha256": sha(data),
            "context": context, "metadata_trust": "caller-declared",
            "native_dialect_validated": False, "limitations": LIMITATIONS,
            "unmapped_nonempty_records": len(unmapped),
            "unmapped_preview": [r["record"] for r in unmapped[:20]],
            "rows": list(unique.values())}


def import_bundle(raw_path, layout_path, context_path, output):
    data = Path(raw_path).read_bytes()
    layout_data, context_data = Path(layout_path).read_bytes(), Path(context_path).read_bytes()
    result = normalize(data, json.loads(layout_data.decode("utf-8-sig")),
                       json.loads(context_data.decode("utf-8-sig")))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw.csv").write_bytes(data)
    (output / "layout.json").write_bytes(layout_data)
    (output / "context.json").write_bytes(context_data)
    write_json(output / "analysis.json", result)
    write_json(output / "manifest.json", {
        "version": VERSION, "kind": "metallic.nsight.source.bundle",
        "files": {name: sha((output / name).read_bytes())
                  for name in ("raw.csv", "layout.json", "context.json", "analysis.json")},
        "parser_sha256": sha(Path(__file__).read_bytes())})
    return {"bundle": str(output.resolve()), "rows": len(result["rows"]),
            "native_dialect_validated": False}


def load_bundle(directory):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    names = ("raw.csv", "layout.json", "context.json", "analysis.json")
    if not isinstance(manifest, dict):
        fail("Manifest must be an object")
    if (manifest.get("version") != VERSION or manifest.get("kind") != "metallic.nsight.source.bundle"
            or set(manifest.get("files", {})) != set(names)):
        fail("Invalid bundle manifest")
    for name in names:
        if sha((directory / name).read_bytes()) != manifest["files"][name]:
            fail("Bundle file changed: " + name)
    if manifest.get("parser_sha256") != sha(Path(__file__).read_bytes()):
        fail("Parser changed; re-import original evidence to a new bundle")
    analysis = read_json(directory / "analysis.json")
    regenerated = normalize((directory / "raw.csv").read_bytes(), read_json(directory / "layout.json"),
                            read_json(directory / "context.json"))
    if regenerated != analysis:
        fail("Derived analysis does not match raw evidence")
    return analysis


def shaders(result):
    groups = {}
    for row in result["rows"]:
        key = (row["module"], row["entry"])
        groups.setdefault(key, []).append(row)
    output = []
    for (module, entry), rows in groups.items():
        il = [r for r in rows if r["representation"] == "il"]
        known = [r["self_samples"] for r in il if r["self_samples"] is not None]
        output.append({"module": module, "entry": entry, "rows": len(rows),
                       "il_self_samples": sum(known) if known else None,
                       "il_self_complete_for_mapped_rows": bool(il) and len(known) == len(il),
                       "source_rows_available": any(r["representation"] == "source" for r in rows),
                       "identity_complete": entry is not None})
    return output


def select(result, module, entry=None):
    matches = [s for s in shaders(result) if s["module"] == module and
               (entry is None or s["entry"] == entry)]
    if len(matches) != 1:
        fail("Shader selection is missing or ambiguous; specify exact module and entry")
    if not matches[0]["identity_complete"]:
        fail("Selected shader has no entry identity; capture/mapping evidence is incomplete")
    selected = matches[0]
    return [r for r in result["rows"] if (r["module"], r["entry"]) ==
            (selected["module"], selected["entry"])]


def hotspots(result, module, entry, representation, metric, top):
    if top < 1 or top > 1000:
        fail("top must be in 1..1000")
    rows = [r for r in select(result, module, entry) if r["representation"] == representation]
    if not rows or not any(r[metric] is not None for r in rows):
        fail("Requested representation/metric is unavailable")
    ranked = sorted((r for r in rows if r[metric] is not None), key=lambda r: -r[metric])
    return {"representation": representation, "metric": metric, "unit": "samples" if metric != "live_registers" else "registers",
            "missing_metric_rows": sum(r[metric] is None for r in rows), "rows": ranked[:top],
            "note": "No cross-representation total; inclusive/dependency values are not additive."}


def source_context(result, module, entry, record, radius):
    if not 0 <= radius <= 100:
        fail("radius must be in 0..100")
    rows = select(result, module, entry)
    anchors = [r for r in rows if any(e["record"] == record for e in r["evidence"])]
    if len(anchors) != 1:
        fail("Record is not part of this unique shader selection")
    anchor = anchors[0]
    if not anchor["file"] or not anchor["line"] or not anchor["line"].isdigit():
        fail("Source location unavailable; no current-worktree fallback")
    if anchor["representation"] != "source":
        fail("Use an exported source row; IL/source mapping is not inferred")
    nearby = [r for r in rows if r["representation"] == "source" and
              r["file"] == anchor["file"] and r["line"] and r["line"].isdigit() and
              abs(int(r["line"]) - int(anchor["line"])) <= radius]
    if not any(r["code"] for r in nearby):
        fail("Exported source text unavailable")
    return {"file": anchor["file"], "anchor_line": int(anchor["line"]),
            "rows": sorted(nearby, key=lambda r: int(r["line"])),
            "note": "Only exported source rows; gaps and unobserved lines are not filled."}


def compare_repeats(results):
    if len(results) < 3:
        fail("At least three independently declared exports are required")
    ids = [r["context"]["export_id"] for r in results]
    if len(ids) != len(set(ids)):
        fail("Repeated export_id; re-importing one export is not three exports")
    def normalized(result):
        context = {key: result["context"][key] for key in
                   ("artifact_sha256", "nsight_version", "selection")}
        rows = [{k: v for k, v in row.items() if k != "evidence"} for row in result["rows"]]
        return {"context": context, "rows": sorted(rows, key=lambda r: json.dumps(r, sort_keys=True)),
                "unmapped_nonempty_records": result["unmapped_nonempty_records"]}
    values = [normalized(r) for r in results]
    equal = all(v == values[0] for v in values[1:])
    return {"structured_results_equal": equal, "declared_exports": len(results),
            "automation_verified": False,
            "note": "Equality does not verify independent UI exports or capture/selection metadata."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("raw")
    inspect.add_argument("--delimiter", choices=("comma", "tab", "semicolon"), default="comma")
    inspect.add_argument("--encoding", default="utf-8-sig")
    inspect.add_argument("--start", type=int, default=1)
    inspect.add_argument("--count", type=int, default=12)
    ingest = commands.add_parser("import")
    for name in ("raw", "layout", "context", "output"):
        ingest.add_argument("--" + name, required=True)
    for command in ("verify", "shaders", "hotspots", "source"):
        sub = commands.add_parser(command)
        sub.add_argument("bundle")
        if command in ("hotspots", "source"):
            sub.add_argument("--module", required=True)
            sub.add_argument("--entry")
        if command == "hotspots":
            sub.add_argument("--representation", choices=("il", "source", "summary"), default="il")
            sub.add_argument("--metric", choices=METRICS, default="self_samples")
            sub.add_argument("--top", type=int, default=10)
        if command == "source":
            sub.add_argument("--record", type=int, required=True)
            sub.add_argument("--radius", type=int, default=3)
    repeat = commands.add_parser("compare-repeats")
    repeat.add_argument("bundles", nargs="+")
    args = parser.parse_args()
    try:
        if args.command == "inspect":
            value = inspect_export(Path(args.raw).read_bytes(),
                                   {"comma": ",", "tab": "\t", "semicolon": ";"}[args.delimiter],
                                   args.encoding, args.start, args.count)
        elif args.command == "import":
            value = import_bundle(args.raw, args.layout, args.context, args.output)
        elif args.command == "compare-repeats":
            value = compare_repeats([load_bundle(p) for p in args.bundles])
        else:
            result = load_bundle(args.bundle)
            if args.command == "verify":
                value = {"integrity": "passed", "native_dialect_validated": False}
            elif args.command == "shaders":
                value = {"shaders": shaders(result)}
            elif args.command == "hotspots":
                value = hotspots(result, args.module, args.entry, args.representation, args.metric, args.top)
            else:
                value = source_context(result, args.module, args.entry, args.record, args.radius)
            value.update(raw_sha256=result["raw_sha256"], context=result["context"],
                         metadata_trust=result["metadata_trust"],
                         native_dialect_validated=result["native_dialect_validated"],
                         unmapped_nonempty_records=result["unmapped_nonempty_records"],
                         limitations=result["limitations"])
        print(json.dumps(value, indent=2, ensure_ascii=False))
        return 1 if value.get("structured_results_equal") is False else 0
    except (ValueError, KeyError, TypeError, OSError, csv.Error) as error:
        print(json.dumps({"error": str(error), "command": args.command}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
