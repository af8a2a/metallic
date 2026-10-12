"""Summarize sealed compute experiments without treating GPU frames as replicates.

Creates independent-process measurements, distributions and paired ratios. Full
pipeline load includes application-cache preparation and native binary loading;
device initialization, SPIR-V compilation/modules and cache saving are excluded.
Pipeline executable diagnostics are optional and remain separate from timings.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
import re
import statistics

import RunComputePipelineExperiments as experiments


METRICS = ("pipeline_load_and_create_wall_ms", "pipeline_cache_preparation_wall_ms",
           "pipeline_wall_ms", "native_create_ms", "driver_ms", "native_binary_import_ms",
           "native_binary_export_ms", "cache_lock_wait_ms", "shader_cache_load_wall_ms",
           "shader_module_wall_ms")
STAT = re.compile(r"\[PipelineStatistics\] entry=(\S+) spirv=([0-9a-f]{16}) executable=(.*?) subgroup=(\d+) (.*?)=(.*?) \((.*)\)$", re.MULTILINE)
BINDING = re.compile(r"\[PipelineStatisticsBinding\] cacheKey=([0-9a-f]{16}) inputSpirvFnv1a64=(\d+) deviceSpirvFnv1a64=(\d+) shader=(\S+) entry=(\S+)$", re.MULTILINE)


def finite(value, *, positive=False):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError("Expected a finite nonnegative measurement")
    return float(value)


def distribution(values):
    values = [finite(value) for value in values]
    if not values:
        return None
    return dict(n=len(values), min=min(values), median=statistics.median(values),
                mean=statistics.mean(values), max=max(values))


def native_rows(root, summary):
    rows = []
    for run in summary["runs"]:
        report = experiments.read(run["report"])
        sample = experiments.driver.single_sample(report, native=True)
        row = dict(experiment=str(root), command="compile", block=run["block"],
                   arm=run["arm"]["name"], optimization=run["arm"]["optimization"],
                   backend=run["arm"]["backend"], disable_driver_optimization=run["arm"]["disable_driver_optimization"],
                   imported=run["imported"], case="", report=run["report"],
                   cache_kind=run.get("cache_kind", "native-binary" if run["imported"] else "none"),
                   shader_request=json.dumps(sample.get("request"), sort_keys=True),
                   effective_descriptor_mode=sample.get("effective_descriptor_mode"),
                   build_configuration=report.get("build_configuration"),
                   validation_active=report.get("device", {}).get("validation_active"),
                   device_spirv_bytes=sample["device_spirv_bytes"],
                   device_spirv_fnv1a64=sample["device_spirv_fnv1a64"],
                   native_binary_data_bytes=sample["pipeline"].get("native_binary_data_bytes", 0))
        row.update({f"spirv_{name}": value for name, value in sample["device_spirv_statistics"].items()
                    if not isinstance(value, list)})
        for metric in METRICS:
            value = sample["compile_wall_ms"] if metric == "shader_cache_load_wall_ms" else \
                    sample.get(metric, sample["pipeline"].get(metric))
            row[metric] = None if value is None else finite(value)
        if row["pipeline_load_and_create_wall_ms"] is not None:
            total = row["pipeline_load_and_create_wall_ms"]
            prep = row["pipeline_cache_preparation_wall_ms"]
            if prep is None or not math.isclose(total, prep + row["pipeline_wall_ms"], abs_tol=1e-4):
                raise ValueError("Full pipeline load scope does not include its recorded cache preparation")
            if row["native_create_ms"] > total + 1e-4:
                raise ValueError("Native duration exceeds the measured full pipeline load scope")
            row["native_fraction"] = row["native_create_ms"] / total if total else None
            separately_measured = row["native_create_ms"] + (row["native_binary_import_ms"] or 0) + (row["native_binary_export_ms"] or 0)
            other = row["pipeline_wall_ms"] - separately_measured
            if other < -1e-3:
                raise ValueError("Native/import/export durations overlap or exceed the pipeline wall scope")
            row["other_pipeline_wall_ms"] = max(0, other)
        else:
            row["native_fraction"] = None
            row["other_pipeline_wall_ms"] = None
        rows.append(row)
    return rows


def dispatch_rows(root, summary):
    rows = []
    for run in summary["runs"]:
        if not run.get("qualified") or not all(case["qualified"] for case in run["correctness"]):
            raise ValueError("GPU timing did not pass the declared correctness gate")
        cache_enabled = summary["native_binary_cache"] and run["arm"]["backend"] != "pipeline"
        for case, metrics in run["timings"].items():
            rows.append(dict(experiment=str(root), command="dispatch", block=run["block"],
                             arm=run["arm"]["name"], optimization=run["arm"]["optimization"],
                             backend=run["arm"]["backend"], disable_driver_optimization=run["arm"]["disable_driver_optimization"],
                             imported=False, native_binary_cache_enabled=cache_enabled,
                             cache_kind="runtime-configured-native-cache" if cache_enabled else "none",
                             case=case, report=run["report"],
                             correctness_max_absolute_error=max(readback["max_absolute_error"]
                                 for result in run["correctness"] if result["case"] == case for readback in result["readbacks"]),
                             **{metric: finite(value["median"], positive=True) for metric, value in metrics.items()},
                             timing_frame_count=metrics["dispatch_scope_ms"]["count"]))
    return rows


def group_rows(rows):
    groups = {}
    for row in rows:
        key = (row["experiment"], row["command"], row["arm"], row["imported"], row["cache_kind"], row["case"])
        group = groups.setdefault(key, [])
        if any(other["block"] == row["block"] for other in group):
            raise ValueError("Duplicate process in an arm/block; refusing to pool measurements")
        group.append(row)
    result = []
    for key, group in groups.items():
        identity = dict(zip(("experiment", "command", "arm", "imported", "cache_kind", "case"), key))
        fields = METRICS + ("native_fraction", "graph_ms", "pass_ms", "dispatch_scope_ms", "correctness_max_absolute_error")
        metrics = {field: distribution([row[field] for row in group if row.get(field) is not None]) for field in fields}
        result.append(dict(**identity, metrics={field: value for field, value in metrics.items() if value is not None}))
    return result


def paired_ratios(rows):
    """Candidate/baseline for matched independent-process blocks, not frames."""
    result = []
    for group in group_rows(rows):
        candidates = [row for row in rows if all(row[field] == group[field]
                      for field in ("experiment", "command", "arm", "imported", "cache_kind", "case"))]
        baseline_arm = group["arm"] if group["command"] == "compile" and group["imported"] else "O1-pipeline-normal"
        metric = "pipeline_load_and_create_wall_ms" if group["command"] == "compile" else "dispatch_scope_ms"
        baseline = [row for row in rows if row["experiment"] == group["experiment"] and row["command"] == group["command"]
                    and row["arm"] == baseline_arm and row["case"] == group["case"] and not row["imported"]]
        if not baseline or (group["arm"] == baseline_arm and not group["imported"]):
            continue
        by_block = {row["block"]: row for row in baseline}
        pairs = []
        for candidate in candidates:
            reference = by_block.get(candidate["block"])
            if reference is None or candidate.get(metric) is None or reference.get(metric) is None:
                continue
            # A source/import comparison must use exactly the same device code.
            if group["command"] == "compile" and group["imported"] and \
               candidate["device_spirv_fnv1a64"] != reference["device_spirv_fnv1a64"]:
                raise ValueError("Source/import pair has different device SPIR-V")
            pairs.append(dict(block=candidate["block"], candidate=candidate[metric], baseline=reference[metric],
                              ratio=finite(candidate[metric]) / finite(reference[metric], positive=True)))
        if pairs:
            result.append(dict(**{field: group[field] for field in ("experiment", "command", "arm", "imported", "cache_kind", "case")},
                               baseline_arm=baseline_arm, metric=metric, unit="candidate/baseline; below 1 is faster",
                               pairs=pairs, ratios=distribution([pair["ratio"] for pair in pairs])))
    return result


def statistics_log(label, path):
    log = Path(path).read_text(encoding="utf-8", errors="replace")
    if "[PipelineStatistics] enabled=true" not in log or re.search(r"\[PipelineStatistics\].*unavailable", log):
        raise ValueError("Pipeline executable statistics are unavailable or incomplete")
    bindings = {}
    for match in BINDING.finditer(log):
        key, source, device, shader, entry = match.groups()
        value = dict(input_spirv_fnv1a64=source, device_spirv_fnv1a64=device, shader=shader, entry=entry)
        if key in bindings and bindings[key] != value:
            raise ValueError("Conflicting executable-to-shader identity")
        bindings[key] = value
    records = {}
    for match in STAT.finditer(log):
        entry, key, executable, subgroup, name, value, description = match.groups()
        if key not in bindings or bindings[key]["entry"] != entry:
            raise ValueError("Pipeline executable diagnostic has no matching shader binding")
        record = dict(cache_key=key, **bindings[key], executable=executable, subgroup=int(subgroup),
                      name=name, value=value, description=description)
        identity = (key, executable, name)
        if identity in records and records[identity] != record:
            raise ValueError("Conflicting repeated driver statistic")
        records[identity] = record
    if not records:
        raise ValueError("No driver executable statistics")
    return dict(label=label, log=str(Path(path).resolve()), timing_eligible=False,
                caveat="Driver-defined statistics. Local Memory Size is not assumed to mean spills.",
                records=list(records.values()))


def write_csv(path, rows):
    columns = sorted({field for row in rows for field in row})
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def plot(path, groups, metric, title):
    selected = [group for group in groups if metric in group["metrics"]]
    width, row_height, left, right = 1100, 45, 440, 170
    height = max(180, 115 + row_height * len(selected))
    scale = max((group["metrics"][metric]["max"] for group in selected), default=1) or 1
    span = width - left - right
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="#fff"/>',
                '<g font-family="Segoe UI, sans-serif" font-size="13" fill="#172b4d">',
                f'<text x="24" y="30" font-size="20">{html.escape(title)}</text>',
                '<text x="24" y="53">One measurement per independent process; bars: median, whiskers: min–max. No confidence intervals.</text>']
    for index, group in enumerate(selected):
        values = group["metrics"][metric]
        y = 90 + index * row_height
        source = Path(group["experiment"]).name
        mode = f" / {group['cache_kind']} import" if group["imported"] else " / source"
        label = f"{group['arm']}{mode}" + (f" / {group['case']}" if group["case"] else "")
        short_label = label if len(label) <= 58 else label[:55] + "..."
        elements += [f'<text x="24" y="{y + 5}"><title>{html.escape(label)}</title>{html.escape(short_label)}</text>',
                     f'<rect x="{left}" y="{y - 11}" width="{values["median"] / scale * span:.3f}" height="22" fill="#306ab3"/>',
                     f'<path d="M{left + values["min"] / scale * span:.3f} {y} H{left + values["max"] / scale * span:.3f}" stroke="#142843" stroke-width="2"/>',
                     f'<text x="{width - right + 12}" y="{y + 5}">{values["median"]:.3f} ms (n={values["n"]})</text>',
                     f'<text x="24" y="{y + 21}" font-size="10" fill="#5f6b7a">{html.escape(source)}</text>']
    elements += [f'<text x="{left}" y="{height - 24}">0</text>',
                 f'<text x="{width - right - 45}" y="{height - 24}">{scale:.3f} ms</text>', '</g></svg>']
    Path(path).write_text("\n".join(elements), encoding="utf-8")


def cost_breakdown(rows):
    components = ("pipeline_cache_preparation_wall_ms", "native_binary_import_ms", "native_create_ms",
                  "native_binary_export_ms", "other_pipeline_wall_ms")
    result = []
    for group in group_rows(rows):
        if group["command"] != "compile" or "pipeline_load_and_create_wall_ms" not in group["metrics"]:
            continue
        matching = [row for row in rows if all(row[field] == group[field] for field in
                    ("experiment", "command", "arm", "imported", "cache_kind", "case"))]
        result.append(dict(**{field: group[field] for field in ("experiment", "arm", "imported", "cache_kind")},
                           n=len(matching), mean_total_ms=statistics.mean(row["pipeline_load_and_create_wall_ms"] for row in matching),
                           components={metric: statistics.mean(row[metric] or 0 for row in matching) for metric in components}))
    return result


def plot_costs(path, breakdown, imported):
    selected = [row for row in breakdown if row["imported"] == imported]
    colors = ("#8f63c4", "#efae4b", "#306ab3", "#65a77a", "#c5cdd8")
    names = ("App cache preparation", "Binary import preparation", "Native creation", "Binary export", "Other pipeline wall")
    width, left, span, height = 1100, 400, 530, max(180, 135 + 45 * len(selected))
    maximum = max((row["mean_total_ms"] for row in selected), default=1) or 1
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
                '<rect width="100%" height="100%" fill="white"/>', '<g font-family="Segoe UI, sans-serif" font-size="12" fill="#172b4d">',
                f'<text x="24" y="28" font-size="20">{"Import" if imported else "Source"} load cost components</text>',
                '<text x="24" y="50">Mean nonoverlapping wall components, one sample/process. Driver duration is contained in native creation.</text>']
    for index, (name, color) in enumerate(zip(names, colors)):
        x = 24 + index * 214
        elements += [f'<rect x="{x}" y="66" width="12" height="12" fill="{color}"/>',
                     f'<text x="{x + 17}" y="76">{name}</text>']
    for index, row in enumerate(selected):
        y, x = 103 + index * 45, float(left)
        label = row["arm"] + (f" / {row['cache_kind']}" if imported else "")
        elements += [f'<text x="24" y="{y + 5}">{html.escape(label)}</text>',
                     f'<text x="24" y="{y + 20}" font-size="10" fill="#5f6b7a">{html.escape(Path(row["experiment"]).name)}</text>']
        for (metric, value), color in zip(row["components"].items(), colors):
            part_width = value / maximum * span
            elements.append(f'<rect x="{x:.3f}" y="{y - 10}" width="{part_width:.3f}" height="22" fill="{color}"><title>{metric}: {value:.6f} ms</title></rect>')
            x += part_width
        elements.append(f'<text x="950" y="{y + 5}">{row["mean_total_ms"]:.3f} ms (n={row["n"]})</text>')
    elements += ['</g></svg>']
    Path(path).write_text("\n".join(elements), encoding="utf-8")


def analyze(inputs, output, statistics_logs=()):
    output = Path(output).resolve()
    roots = [Path(root).resolve(strict=True) for root in inputs]
    if len(set(roots)) != len(roots):
        raise ValueError("Duplicate experiment directory")
    if any(output == root or root in output.parents for root in roots):
        raise ValueError("Analysis output must be outside sealed experiment directories")
    rows, sources, warnings, frontend = [], [], [], []
    for root in roots:
        summary = experiments.verify(root)
        sources.append(dict(directory=str(root), manifest_sha256=experiments.digest(root / "Manifest.json"),
                            command=summary["command"], scope=summary["scope"],
                            device=summary.get("device", summary.get("devices")),
                            workload=summary.get("workload"), caveats=summary.get("caveats", []),
                            validation=summary.get("validation"), shaders=summary.get("shaders"),
                            device_shaders=summary.get("device_shaders")))
        rows += native_rows(root, summary) if summary["command"] == "compile" else dispatch_rows(root, summary)
        for prefill in summary.get("excluded_runs", []):
            if prefill.get("condition") != "SPIR-V prefill":
                continue
            report = experiments.read(prefill["report"])
            sample = experiments.driver.single_sample(report, native=False)
            frontend.append(dict(experiment=str(root), optimization=prefill["optimization"], n=1,
                                 scope="Excluded one-time SPIR-V prefill; includes shader cache bookkeeping, excludes module and native creation.",
                                 compile_wall_ms=finite(prefill["compile_wall_ms"]),
                                 compilation_phases=sample.get("compilation_phases"), report=prefill["report"],
                                 cache_hit=sample.get("shader_cache_hit")))
    if any(row["command"] == "compile" and row["pipeline_load_and_create_wall_ms"] is None for row in rows):
        warnings.append("Legacy reports omit cache preparation. Native API timings are retained; full load comparisons are unavailable for those runs.")
    groups = group_rows(rows)
    breakdown = cost_breakdown(rows)
    result = dict(schema_version=1, statistical_unit="One independent-process sample (GPU: process median); experiments are not pooled.",
                  scope="Full pipeline load excludes device/Slang/module initialization and cache saving. Driver time is inside native time.",
                  sources=sources, warnings=warnings, measurements=rows, distributions=groups, paired_ratios=paired_ratios(rows),
                  frontend_prefill_diagnostics=frontend,
                  cost_breakdown=dict(scope="Means of nonoverlapping measured wall components; independent-process distributions retain medians.", groups=breakdown),
                  executable_diagnostics=[statistics_log(label, path) for label, path in statistics_logs])
    output.mkdir(parents=True, exist_ok=True)
    experiments.save(output / "Analysis.json", result)
    write_csv(output / "Measurements.csv", rows)
    plot(output / "NativeCreate.svg", [group for group in groups if not group["imported"]], "native_create_ms", "Source native creation")
    plot(output / "FullPipelineLoad.svg", [group for group in groups if not group["imported"]], "pipeline_load_and_create_wall_ms", "Source: cache preparation + pipeline creation")
    plot(output / "ImportedPipelineLoad.svg", [group for group in groups if group["imported"]], "pipeline_load_and_create_wall_ms", "Import: cache preparation + pipeline creation")
    plot(output / "GPUDispatch.svg", groups, "dispatch_scope_ms", "Production path trace shading GPU scope")
    plot_costs(output / "SourceCostBreakdown.svg", breakdown, False)
    plot_costs(output / "ImportCostBreakdown.svg", breakdown, True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--statistics-log", action="append", default=[], metavar="LABEL=PATH")
    args = parser.parse_args()
    logs = []
    for specification in args.statistics_log:
        if "=" not in specification:
            parser.error("--statistics-log expects LABEL=PATH")
        label, path = specification.split("=", 1)
        if not label or not path:
            parser.error("--statistics-log expects a nonempty label and path")
        logs.append((label, Path(path)))
    result = analyze(args.inputs, args.output_dir, logs)
    print(f"Verified {len(result['sources'])} experiments; wrote {len(result['measurements'])} independent-process measurements to {args.output_dir}")


if __name__ == "__main__":
    main()
