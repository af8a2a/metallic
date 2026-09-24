"""Parser for the exact Source/IL CSV dialect observed in profiledata.csv.

This parses exported bytes, not the Nsight UI or its private capture format.
"""
import re

DIALECT = "nsight-source-il-observed-v1"
IL_HEADER = ["#", "Source", "Samples", "Top Stall #1 (Type)", "Top Stall #1 (Samples)",
             "Top Stall #2 (Type)", "Top Stall #2 (Samples)", "Top Stall #3 (Type)",
             "Top Stall #3 (Samples)", "Avg. Warp Latency", "Instruction Mix",
             "Dependency-Attributed Samples", "Cooperative Vector Fusion", "Live Registers"]
SOURCE_HEADER = IL_HEADER[:2] + ["Total Samples"] + IL_HEADER[2:]


def parse(records, raw_hash, context, number):
    if not isinstance(context, dict):
        raise ValueError("Context must be an object")
    headers, tables, annotations = [], [], []
    current, header = None, None
    for raw in records:
        cells = raw["cells"]
        if not cells:
            continue
        if cells[0] == "#":
            if cells not in (SOURCE_HEADER, IL_HEADER):
                raise ValueError("Unknown native header at record " + str(raw["record"]))
            header = cells
            headers.append(raw["record"])
            current = None
            continue
        if header is None or len(cells) < 2:
            raise ValueError("Data without native header")
        # Nsight file/module markers have fewer trailing empty cells than data.
        marker = not cells[0] and cells[1] and not any(cells[2:])
        module_match = re.fullmatch(r"(.+\.spv) \(([0-9a-fA-F]+)\)", cells[1]) if marker else None
        source_marker = marker and header == SOURCE_HEADER and not cells[1].startswith("//")
        if module_match or source_marker:
            if (module_match is not None) != (header == IL_HEADER):
                raise ValueError("Module marker under source header")
            current = {"representation": "il" if module_match else "source", "name": cells[1],
                       "marker_record": raw["record"], "header_record": headers[-1],
                       "rows": [], "entries": [], "source_references": []}
            tables.append(current)
            continue
        if current is None:
            raise ValueError("Data without file/module marker")
        if not cells[0]:
            if current["representation"] != "il":
                raise ValueError("Unexpected unnumbered source row")
            reference = re.fullmatch(r"// (.+):([0-9]+)", cells[1])
            if reference and not any(cells[2:]):
                current["source_references"].append({"file": reference[1], "line": int(reference[2]),
                                                      "record": raw["record"]})
            elif len(cells) != len(header):
                raise ValueError("Unknown annotation shape at record " + str(raw["record"]))
            else:
                # Validate metrics even for mirrors; never include them in totals.
                for name in ("Samples", "Dependency-Attributed Samples", "Live Registers"):
                    number(cells[header.index(name)])
            annotations.append({"module": current["name"], "record": raw["record"],
                                "physical_lines": raw["physical_lines"], "cells": cells,
                                "kind": "source-reference" if reference else "source-mirror"})
            continue
        if not cells[0].isascii() or not cells[0].isdigit() or len(cells) != len(header):
            raise ValueError("Invalid numbered row at record " + str(raw["record"]))
        columns = {"line": 0, "code": 1, "self_samples": header.index("Samples"),
                   "dependency_samples": header.index("Dependency-Attributed Samples"),
                   "live_registers": header.index("Live Registers")}
        item = {"module": current["name"] if current["representation"] == "il" else None,
                "entry": None, "file": current["name"], "line": cells[0], "code": cells[1],
                "representation": current["representation"], "identity": [cells[0]],
                "self_samples": number(cells[columns["self_samples"]]),
                "dependency_samples": number(cells[columns["dependency_samples"]]),
                "live_registers": number(cells[columns["live_registers"]]),
                "inclusive_samples": None,
                "total_samples": number(cells[2]) if header == SOURCE_HEADER else None,
                "stalls": [], "stall_coverage": "top-k-lower-bound",
                "raw_metrics": dict(zip(header[2:], cells[2:])),
                "evidence": [{"record": raw["record"], "physical_lines": raw["physical_lines"],
                              "header_record": current["header_record"], "columns": columns,
                              "marker_record": current["marker_record"]}]}
        for i in (1, 2, 3):
            name, count = cells[header.index(f"Top Stall #{i} (Type)")], number(cells[header.index(f"Top Stall #{i} (Samples)")])
            if (not name and count is not None) or (name and count is None):
                raise ValueError("Incomplete stall pair at record " + str(raw["record"]))
            if name:
                item["stalls"].append({"name": name, "samples": count})
        if item["self_samples"] is not None and sum(s["samples"] for s in item["stalls"]) > item["self_samples"]:
            raise ValueError("Top stalls exceed Samples")
        entry = re.match(r'^OpEntryPoint\s+(\S+)\s+(%\S+)\s+"([^"]+)"(?:\s|$)', cells[1])
        if entry and current["representation"] == "il":
            value = {"execution_model": entry[1], "symbol": entry[2], "name": entry[3],
                     "record": raw["record"]}
            if not any(all(e[k] == value[k] for k in ("execution_model", "symbol", "name"))
                       for e in current["entries"]):
                current["entries"].append(value)
        current["rows"].append(item)
    modules = [t for t in tables if t["representation"] == "il"]
    if not modules or not any(t["rows"] for t in modules):
        raise ValueError("Native export requires a nonempty SPIR-V table")
    # Deduplicate repeated module tables when establishing source attribution.
    module_names = {m["name"] for m in modules}
    unique, source_associations = {}, []
    for table in tables:
        candidates = [m for m in modules if any(
            ref["file"].replace("\\", "/").rsplit("/", 1)[-1] == table["name"]
            for ref in m["source_references"])]
        signatures = {(e["name"], e["symbol"], e["execution_model"])
                      for m in candidates for e in m["entries"]}
        reference_paths = {ref["file"] for m in candidates for ref in m["source_references"]
                           if ref["file"].replace("\\", "/").rsplit("/", 1)[-1] == table["name"]}
        associated = table if table["representation"] == "il" else (
            candidates[0] if len(module_names) == 1 and candidates and len(signatures) == 1
            and len(reference_paths) == 1 else None)
        entry = associated["entries"][0] if associated and len(associated["entries"]) == 1 else None
        if table["representation"] == "source":
            source_associations.append({"file": table["name"],
                                        "module": associated["name"] if associated else None,
                                        "basis": "single-exported-module-and-source-reference" if associated else "unresolved",
                                        "ui_selection_verified": False})
        for item in table["rows"]:
            if associated:
                item["module"] = associated["name"]
            item["entry"] = entry["name"] if entry else None
            for ev in item["evidence"]:
                ev["entry_record"] = entry["record"] if entry else None
            key = (item["module"], item["entry"], item["representation"], item["file"], item["line"])
            if key in unique:
                previous = {k: v for k, v in unique[key].items() if k != "evidence"}
                if previous != {k: v for k, v in item.items() if k != "evidence"}:
                    raise ValueError("Conflicting native row identity")
                unique[key]["evidence"].extend(item["evidence"])
            else:
                unique[key] = item
    rows = list(unique.values())
    return {"version": 1, "kind": "metallic.nsight.source", "raw_sha256": raw_hash,
            "dialect": DIALECT, "native_dialect_validated": True,
            "validation_scope": "observed CSV headers, markers, numeric rows and SPIR-V entry declarations only",
            "context": context, "metadata_trust": "export-content; capture context unverified",
            "unmapped_nonempty_records": 0, "unmapped_preview": [],
            "headers": headers,
            "modules": [{"module": m["name"], "marker_record": m["marker_record"],
                         "entry_points": m["entries"], "source_references": m["source_references"]} for m in modules],
            "source_associations": source_associations, "annotations": annotations, "rows": rows,
            "limitations": [
                "Observed CSV dialect only; no automatic UI export or capture/selection verification.",
                "Total Samples retained separately; no inferred inclusive/self equality.",
                "Blank sample cells remain unknown; sums cover explicitly reported Samples only.",
                "Source mirrors and source/IL representations are not additive.",
                "Top-three stall counts are lower bounds, not GPU time or recoverable speedup.",
                "IL line numbers are textual SPIR-V lines, including embedded debug source; not SASS PCs.",
                "Live Registers are not allocated registers or occupancy.",
                "Source association uses export-internal evidence, not current repository source."
            ]}
