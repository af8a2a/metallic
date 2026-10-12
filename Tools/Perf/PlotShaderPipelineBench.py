"""Export PSO sample distributions from MetallicShaderPipelineBench reports."""

import argparse
import csv
from html import escape
import json
import math
from pathlib import Path
import statistics


LOG_FLOOR = 1e-6


def plot_value(value, logarithmic):
    # Native durations have nanosecond precision. Clamp zero to one nanosecond
    # (1e-6 ms) for display only; raw CSV/sample values remain unchanged.
    return max(value, LOG_FLOOR) if logarithmic else value


def plot_bounds(values, logarithmic):
    if logarithmic:
        low = math.floor(math.log10(min(plot_value(value, True) for value in values)))
        high = math.ceil(math.log10(max(plot_value(value, True) for value in values)))
        if low == high:
            high += 1
        return 10.0 ** low, 10.0 ** high
    return 0, max(.001, max(values)) * 1.05


def sample_rows(report, path):
    configuration = report.get("configuration", {})
    for index, sample in enumerate(report.get("samples", [])):
        pipeline = sample.get("pipeline")
        if sample.get("status") != "passed" or not pipeline:
            continue
        native_ms = pipeline.get("native_create_ms")
        if not isinstance(native_ms, (int, float)) or not math.isfinite(native_ms) or native_ms < 0:
            raise ValueError(f"invalid native_create_ms: {path} sample {index}")
        request = sample.get("request", {})
        stages = sample.get("shader_stages", [])
        batch_index = sample.get("pipeline_batch")
        batches = report.get("pipeline_batches", [])
        batch = batches[batch_index] if isinstance(batch_index, int) and 0 <= batch_index < len(batches) else {}
        modules = " + ".join(dict.fromkeys(stage.get("module", "") for stage in stages)) if stages else request.get("module", sample.get("recipe_name", "graphics recipe"))
        entries = " + ".join(f'{stage.get("stage", "")}: {stage.get("source_entry", stage.get("entry", ""))}' for stage in stages) if stages else request.get("entry", sample.get("spirv_entry", ""))
        pipeline_type = sample.get("pipeline_type", pipeline.get("type", "compute" if sample.get("execution_model") == 5 else "graphics"))
        hit = pipeline.get("native_cache_hit")
        yield dict(report=str(path), sample=index, mode=configuration.get("mode", "serial"),
                   build_configuration=report.get("build_configuration", "unknown"),
                   driver_cache_control=report.get("environment", {}).get("METALLIC_VK_INTERNAL_PIPELINE_CACHE") or "default",
                   workers=configuration.get("workers", configuration.get("pipeline_workers", 1)),
                   actual_workers=batch.get("actual_workers", 1), pipeline_worker=sample.get("pipeline_worker", 0),
                   iteration=sample.get("iteration", 0), pipeline_replica=sample.get("pipeline_replica", 0),
                   cache_lock_scope=configuration.get("cache_lock_scope", "unspecified"),
                   shader_module=modules, entry_point=entries,
                   shader_stages=json.dumps(stages, separators=(",", ":")),
                   pipeline_type=pipeline_type, spirv_bytes=sample.get("spirv_bytes", 0),
                   spirv_fnv1a64=sample.get("spirv_fnv1a64", ""),
                   slang_optimization=sample.get("effective_slang_optimization", configuration.get("slang_optimization", "unspecified")),
                   disable_driver_optimization=configuration.get("disable_driver_optimization", False),
                   compute_backend=pipeline.get("compute_backend", "pipeline"),
                   native_binary_cache_hit=pipeline.get("native_binary_cache_hit", False),
                   native_binary_import_ms=pipeline.get("native_binary_import_ms"),
                   native_binary_export_ms=pipeline.get("native_binary_export_ms"),
                   spirv_instruction_count=sample.get("spirv_statistics", {}).get("instruction_count"),
                   spirv_function_count=sample.get("spirv_statistics", {}).get("function_count"),
                   spirv_basic_block_count=sample.get("spirv_statistics", {}).get("basic_block_count"),
                   spirv_max_function_instructions=sample.get("spirv_statistics", {}).get("max_function_instructions"),
                   native_create_ms=native_ms, driver_ms=pipeline.get("driver_ms"),
                   cache_lock_wait_ms=pipeline.get("cache_lock_wait_ms"),
                   pipeline_wall_ms=sample.get("pipeline_wall_ms"),
                   application_pipeline_cache_hit="unknown" if hit is None else "hit" if hit else "miss")


def variant_label(row):
    return (f'{row.get("slang_optimization", "unspecified")}, {row.get("compute_backend", "pipeline")}, '
            f'driver {"disable-opt" if row.get("disable_driver_optimization") else "normal"}, '
            f'{"binary import" if row.get("native_binary_cache_hit") else "source"}')


def svg_distribution(rows, output, top, log_scale=False):
    """Dependency-free export when matplotlib is not installed."""
    colors = dict(hit="#237a57", miss="#c1582c", unknown="#637c94")
    groups = {}
    for row in rows:
        groups.setdefault((row["pipeline_type"], row["application_pipeline_cache_hit"]), []).append(row)

    def text(x, y, value, size=13, anchor="start"):
        return f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}">{escape(str(value))}</text>'

    def marker(x, y, kind, color, label="", opacity=1):
        attributes = f'fill="{color}" opacity="{opacity}" data-pipeline-type="{escape(kind)}"'
        title = f'<title>{escape(label)}</title>' if label else ""
        if kind == "compute":
            return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" {attributes}>{title}</circle>'
        return f'<polygon points="{x:.2f},{y - 5:.2f} {x - 5:.2f},{y + 4:.2f} {x + 5:.2f},{y + 4:.2f}" {attributes}>{title}</polygon>'

    def transform(value, logarithmic):
        return math.log10(plot_value(value, True)) if logarithmic else value

    def ticks(low, high, logarithmic):
        if logarithmic:
            return [(10.0 ** exponent, f"1e{exponent}") for exponent in range(
                round(math.log10(low)), round(math.log10(high)) + 1)]
        return [(low + (high - low) * tick / 5, f"{low + (high - low) * tick / 5:.3g}") for tick in range(6)]

    def plot_frame(title, xlabel, ylabel, xmax, ymax, xmin=0, ymin=0, log_x=False, log_y=False, width=1100, height=620):
        left, right, top_margin, bottom = 95, 300, 70, 80
        plot_width, plot_height = width - left - right, height - top_margin - bottom
        parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                 '<rect width="100%" height="100%" fill="white"/>',
                 '<g font-family="Arial,sans-serif" fill="#243244">', text(left, 30, title, 20),
                 text(left, 52, "Feedback describes application cache reuse; internal driver-cache hits are unobserved.", 12)]
        xlow, xhigh = transform(xmin, log_x), transform(xmax, log_x)
        ylow, yhigh = transform(ymin, log_y), transform(ymax, log_y)
        point = lambda x, y: (left + plot_width * (transform(x, log_x) - xlow) / (xhigh - xlow),
                             top_margin + plot_height * (1 - (transform(y, log_y) - ylow) / (yhigh - ylow)))
        for value, label in ticks(xmin, xmax, log_x):
            x, _ = point(value, ymin)
            parts.append(f'<path d="M {x} {top_margin} V {top_margin + plot_height}" stroke="#e4e9ef" fill="none"/>')
            parts.append(text(x, top_margin + plot_height + 23, label, anchor="middle"))
        for value, label in ticks(ymin, ymax, log_y):
            _, y = point(xmin, value)
            parts.append(f'<path d="M {left} {y} H {left + plot_width}" stroke="#e4e9ef" fill="none"/>')
            parts.append(text(left - 10, y + 4, label, anchor="end"))
        parts.extend((text(left + plot_width / 2, height - 25, xlabel, anchor="middle"),
                      f'<text transform="translate(20 {top_margin + plot_height / 2}) rotate(-90)" font-size="13" text-anchor="middle">{escape(ylabel)}</text>'))
        if log_x or log_y:
            parts.append(text(left, height - 7, "Base-10 log axes; display floor 1e-6 ms (and 1e-6 KiB for size). CSV retains raw values.", 11))
        return parts, point, (left + plot_width + 20, top_margin + 20)

    def legends(parts, position):
        for index, (kind, hit) in enumerate(sorted(groups)):
            x, y = position[0], position[1] + index * 25
            parts.extend((marker(x, y - 4, kind, colors[hit]), text(x + 12, y, f"{kind}, cache {hit}", 12)))

    xmin, xmax = plot_bounds([row["spirv_bytes"] / 1024 for row in rows], log_scale)
    ymin, ymax = plot_bounds([row["native_create_ms"] for row in rows], log_scale)
    parts, point, legend = plot_frame("SPIR-V size versus native pipeline creation", "SPIR-V size (KiB)", "Native creation (ms)",
        xmax, ymax, xmin=xmin, ymin=ymin, log_x=log_scale, log_y=log_scale)
    for row in rows:
        x, y = point(row["spirv_bytes"] / 1024, row["native_create_ms"])
        label = f'{row["shader_module"]}.{row["entry_point"]}: {row["native_create_ms"]:.4g} ms ({row["pipeline_type"]}; {variant_label(row)})'
        hit = row["application_pipeline_cache_hit"]
        parts.append(marker(x, y, row["pipeline_type"], colors[hit], label, .7))
    legends(parts, legend)
    parts.append('</g></svg>')
    (output / "PipelineSizeLatency.svg").write_text("\n".join(parts), encoding="utf-8")
    parts, point, legend = plot_frame("Native pipeline creation latency distribution", "Native creation (ms)", "Cumulative fraction",
        ymax, 1, xmin=ymin, log_x=log_scale)
    for (kind, hit), group in sorted(groups.items()):
        coordinates = [point(ymin, 0)]
        for index, value in enumerate(sorted(row["native_create_ms"] for row in group)):
            coordinates.extend((point(value, index / len(group)), point(value, (index + 1) / len(group))))
        coordinates.append(point(ymax, 1))
        dash = ' stroke-dasharray="6 4"' if kind != "compute" else ""
        parts.append(f'<polyline points="' + " ".join(f"{x:.2f},{y:.2f}" for x, y in coordinates) + f'" fill="none" stroke="{colors[hit]}" stroke-width="2"{dash}/>')
        # Mark endpoints of each step with the same pipeline-type symbols as
        # scatter and legend. Deduplicate equal durations before marking.
        for value in sorted({row["native_create_ms"] for row in group}):
            fraction = sum(row["native_create_ms"] <= value for row in group) / len(group)
            x, y = point(value, fraction)
            parts.append(marker(x, y, kind, colors[hit]))
    legends(parts, legend)
    parts.append('</g></svg>')
    (output / "PipelineLatencyCDF.svg").write_text("\n".join(parts), encoding="utf-8")
    module_groups = {}
    for row in rows:
        module_groups.setdefault((row["shader_module"], row["entry_point"], row["pipeline_type"], variant_label(row)), []).append(row["native_create_ms"])
    module_groups = sorted(module_groups.items(), key=lambda item: max(item[1]), reverse=True)[:top]
    height, left, width = max(220, 120 + 32 * len(module_groups)), 490, 1100
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="Arial,sans-serif" fill="#243244">', text(20, 30, "Slowest module/entry groups; range and median", 20)]
    scale = lambda value: left + 560 * (transform(value, log_scale) - transform(ymin, log_scale)) / (transform(ymax, log_scale) - transform(ymin, log_scale))
    for index, ((module, entry, kind, variant), values) in enumerate(module_groups):
        y = 75 + index * 32
        label = f"{module}.{entry} ({kind}; {variant})"
        parts.append(text(left - 12, y + 4, label if len(label) < 64 else "..." + label[-60:], 12, "end"))
        parts.append(f'<line x1="{scale(min(values))}" x2="{scale(max(values))}" y1="{y}" y2="{y}" stroke="#637c94" stroke-width="3"/>')
        parts.append(marker(scale(statistics.median(values)), y, kind, "#243244", f"{label}; n={len(values)}"))
    for value, label in ticks(ymin, ymax, log_scale):
        x = scale(value)
        parts.append(f'<path d="M {x:.2f} 60 V {height - 60}" fill="none" stroke="#e4e9ef"/>')
        parts.append(text(x, height - 40, label, 11, "middle"))
    parts.append(text(left, height - 20, "Native creation (ms); circle = compute, triangle = graphics", 12))
    if log_scale:
        parts.append(text(left, height - 5, "Base-10 log axis; display floor 1e-6 ms. CSV retains raw values.", 11))
    parts.append('</g></svg>')
    (output / "PipelineModules.svg").write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+", help="report.json files or experiment/report directories")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top", type=int, default=20, help="maximum module/entry groups in latency plot")
    parser.add_argument("--log-scale", action="store_true", help="use base-10 log size/latency axes; display floor 1e-6 ms (1 ns), raw CSV unchanged")
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be positive")
    files = set()
    for path in args.reports:
        path = path.resolve(strict=True)
        if path.is_dir() and (path / "summary.json").exists():
            summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
            if summary.get("runs") and all("report" in run for run in summary["runs"]):
                # Helper manifests list measured processes, excluding SPIR-V
                # prefill and driver priming from plotted timing samples.
                files.update(Path(run["report"]).resolve(strict=True) for run in summary["runs"])
                continue
        files.update(path.rglob("report.json") if path.is_dir() else (path,))
    rows = []
    for path in sorted(files):
        report = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(sample_rows(report, path))
    if not rows:
        parser.error("no successful native pipeline samples")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "Samples.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Keep evidence export usable on a minimal Python installation.
    try:
        import matplotlib
    except ImportError:
        svg_distribution(rows, output, args.top, args.log_scale)
        print(output)
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    colors = dict(hit="#237a57", miss="#c1582c", unknown="#637c94")
    types = sorted({row["pipeline_type"] for row in rows})
    for pipeline_type in types:
        for hit in colors:
            group = [row for row in rows if row["pipeline_type"] == pipeline_type and row["application_pipeline_cache_hit"] == hit]
            if not group:
                continue
            marker = "o" if pipeline_type == "compute" else "^"
            axes[0].scatter([plot_value(row["spirv_bytes"] / 1024, args.log_scale) for row in group],
                            [plot_value(row["native_create_ms"], args.log_scale) for row in group],
                            label=f"{pipeline_type}, application cache {hit}", color=colors[hit],
                            marker=marker, alpha=.7, s=23)
            ordered = sorted(plot_value(row["native_create_ms"], args.log_scale) for row in group)
            axes[1].step(ordered, [(i + 1) / len(ordered) for i in range(len(ordered))], where="post",
                         label=f"{pipeline_type}, application cache {hit}", color=colors[hit], marker=marker, markersize=3,
                         linestyle="-" if pipeline_type == "compute" else "--")
    axes[0].set(xlabel="SPIR-V size (KiB)", ylabel="Native pipeline creation (ms)", title="Size versus native creation time")
    axes[1].set(xlabel="Native pipeline creation (ms)", ylabel="Cumulative fraction", title="Latency distribution")
    if args.log_scale:
        axes[0].set_xscale("log", base=10)
        axes[0].set_yscale("log", base=10)
        axes[1].set_xscale("log", base=10)
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    note = "Vulkan feedback identifies application pipeline-cache reuse; internal driver-cache hits are unobserved."
    if args.log_scale:
        note += "\nBase-10 log axes; display floor 1e-6 ms (size floor 1e-6 KiB). CSV retains raw values."
    fig.suptitle(note, fontsize=10)
    fig.savefig(output / "PipelineDistribution.png", dpi=180)
    plt.close(fig)
    groups = {}
    for row in rows:
        key = (row["shader_module"], row["entry_point"], row["pipeline_type"], variant_label(row))
        groups.setdefault(key, []).append(row["native_create_ms"])
    groups = sorted(groups.items(), key=lambda item: max(item[1]), reverse=True)[:args.top]
    fig, axis = plt.subplots(figsize=(11, max(4, len(groups) * .35)), layout="constrained")
    axis.boxplot([[plot_value(value, args.log_scale) for value in values] for _, values in reversed(groups)], vert=False,
                 tick_labels=[f"{module}.{entry} ({kind}; {variant})" for (module, entry, kind, variant), _ in reversed(groups)],
                 showfliers=True)
    axis.set(xlabel="Native pipeline creation (ms)", title=f"Slowest {len(groups)} module/entry groups by maximum observed latency")
    if args.log_scale:
        axis.set_xscale("log", base=10)
        fig.suptitle("Base-10 log axis; display floor 1e-6 ms (1 ns). CSV retains raw values.", fontsize=10)
    axis.grid(axis="x", alpha=.2)
    fig.savefig(output / "PipelineModules.png", dpi=180)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
