"""Source-compiled Fresnel A/B with fixed inline/trusted production parameters.

Only LobeUtils.slang may change, through a journaled transaction. The old
inline-root runner's process, recipe, readback and timing checks are reused.
Two correctness processes precede three AB/BA/AB process pairs. A failed image
gate stops the schedule; no timing result is manufactured from partial runs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid

sys.dont_write_bytecode = True
import MeasurePathTraceInlineRoot as base
from ExperimentRunner import process_alive

experiment, dispatch, driver, rendering = base.experiment, base.dispatch, base.driver, base.rendering
ROOT = base.ROOT
TARGET = ROOT / "Shaders/Modules/OpenPBR/LobeUtils.slang"
ARMS = ("reference", "candidate")
NAME = "openpbr-fresnel-source"
FIXED = dict(inlineComputeRoot=True, trustedSceneParameters=True)
HELPERS = (Path(__file__).name, "MeasurePathTraceInlineRoot.py", "MeasurePathTraceBackendDispatch.py",
           "RunComputePipelineExperiments.py", "RunShaderPipelineDriver.py", "ValidatePathTraceBackendRendering.py",
           "ExperimentRunner.py", "WorkloadCase.py")
require = base.require


def sha(data):
    return hashlib.sha256(data).hexdigest()


def schedule():
    rows = copy.deepcopy(base.schedule(base.TRUSTED_EXPERIMENT))
    for row in rows:
        row["arm"] = ARMS[("checked", "trusted").index(row["arm"])]
        row["label"] = ("correctness-" if row["phase"] == "correctness" else f"block-{row['block']:02}-") + row["arm"]
    return rows


def validate_sources(reference, candidate, head):
    normalize = lambda data: data.replace(b"\r\n", b"\n")
    require(normalize(reference) == normalize(head), "Reference differs from git HEAD beyond line endings")
    left, right = normalize(reference), normalize(candidate)
    require(left != right, "Fresnel candidate is a semantic/textual no-op")
    start = b"public float openPBRFresnel("
    end = b"// This is an RGB version of the Fresnel function."
    require(all(data.count(start) == 1 and data.count(end) == 1 for data in (left, right)),
            "Ambiguous Fresnel function boundary")
    require(left[:left.index(start)] == right[:right.index(start)] and
            left[left.index(end):] == right[right.index(end):], "Candidate changes code outside scalar Fresnel")
    require(all(data.index(start) < data.index(end) for data in (left, right)), "Invalid Fresnel source boundaries")


def atomic_replace(path, data, expected):
    """Check cooperative ownership immediately before replacing the single file."""
    require(path.read_bytes() == expected, "Concurrent shader edit: refusing to overwrite")
    temporary = path.with_name(path.name + ".fresnel-" + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        require(path.read_bytes() == expected, "Concurrent shader edit before atomic replacement")
        os.replace(temporary, path)
        require(path.read_bytes() == data, "Shader changed immediately after replacement")
    finally:
        temporary.unlink(missing_ok=True)


class SourceTransaction:
    def __init__(self, output, original, sources, *, target=TARGET):
        self.output, self.target, self.original, self.sources = output, target, original, sources
        self.current = original
        self.journal = dict(target=str(target), owner_pid=os.getpid(), state="prepared", original_sha256=sha(original),
                            source_sha256={arm: sha(data) for arm, data in sources.items()}, events=[])
        self.save()

    def save(self):
        experiment.save(self.output / "SourceTransaction.json", self.journal)

    def install(self, arm):
        data = self.sources[arm]
        require(self.target.read_bytes() == self.current, "Concurrent shader edit: refusing source arm installation")
        self.journal.update(state="installing", pending_sha256=sha(data), current_sha256=sha(self.current))
        self.save()  # Recovery knows both possible states before the atomic mutation.
        if data != self.current:
            atomic_replace(self.target, data, self.current)
        self.current = data
        self.journal.update(state="installed", current_sha256=sha(data))
        self.journal["events"].append(dict(arm=arm, sha256=sha(data), time=time.time()))
        self.save()

    def restore(self):
        try:
            require(self.target.read_bytes() == self.current, "Concurrent shader edit: original retained; refusing restoration")
            if self.current != self.original:
                atomic_replace(self.target, self.original, self.current)
            self.current = self.original
            self.journal.update(state="restored", current_sha256=sha(self.original))
        except BaseException as error:
            self.journal.update(state="restore-conflict", restore_error=str(error))
            raise
        finally:
            self.save()


def prepare_workload(cases, output):
    # Use the established fixture construction unchanged, then select its
    # trusted configuration for BOTH source arms. Unused configs stay archived.
    _, profiles, inputs, absent = base.prepare_workload(cases, output / "prepared", base.TRUSTED_EXPERIMENT)
    selected = profiles["trusted"]
    configuration = selected["configuration"]
    validate_workload(configuration)
    return configuration, selected["file"], inputs, absent


def validate_workload(configuration):
    dispatch.validate_workload(configuration)
    require(configuration["frames"] == 56 and configuration["warmupFrames"] == 16 and configuration["timingFrames"] == 32,
            "Changed fixed frame windows")
    for case in configuration["cases"]:
        properties = base.path_trace_node(experiment.read(Path(case["graph"])))["properties"]
        require(all(properties.get(key) is value for key, value in FIXED.items()), "Root/trusted state changed")
        require(case["width"] == 512 and case["height"] == 512 and properties["samples"] == 2 and
                properties["maxDepth"] == 12 and properties["cacheMode"] == "off", "Fixed workload changed")


def comparable_report(raw, configuration):
    # Existing validation checks require trusted=true, root=true and 24-byte
    # layout. Keep the actual graph unchanged: source bytes are the treatment.
    base.comparable_report(raw, configuration, configuration, "trusted", base.TRUSTED_EXPERIMENT)
    return copy.deepcopy(raw)


def runtime_identity(destination, configuration, validation):
    return base.runtime_identity(destination, configuration, "trusted", validation, base.TRUSTED_EXPERIMENT)


def validate_pair(targets):
    translated = {("checked" if arm == "reference" else "trusted"): value for arm, value in targets.items()}
    base.validate_target_pair(translated, base.TRUSTED_EXPERIMENT)


def summarize(runs):
    translated = copy.deepcopy(runs)
    for item in translated:
        item["arm"] = "checked" if item["arm"] == "reference" else "trusted"
    return base.summarize(translated, base.TRUSTED_EXPERIMENT)


def expected_inputs(inputs, sources, arm):
    result = dict(inputs)
    result[str(TARGET.resolve())] = sources[arm]["sha256"]
    return result


def inspect_run(output, summary, item, baseline, identities):
    destination = output / item["label"]
    configuration = summary["workload"]
    raw = experiment.read(destination / "MaterialBaseline.json")
    require(raw.get("validation") is item["validation"], "Runtime validation policy changed")
    normalized = comparable_report(raw, configuration)
    comparable = destination / "ComparableReport.json"
    identity = runtime_identity(destination, configuration, item["validation"])
    require(identities.setdefault("device", identity["device"]) == identity["device"] and
            identities.setdefault("ordinary", identity["ordinary"]) == identity["ordinary"], "Device or non-target shader changed")
    targets = identities.setdefault("targets", {})
    require(targets.setdefault(item["arm"], identity["targets"]) == identity["targets"], "Target shader changed within one arm")
    validate_pair(targets)
    return normalized, identity, comparable, baseline or comparable


def run(args, output, summary):
    reference_path, candidate_path = args.reference.resolve(strict=True), args.candidate.resolve(strict=True)
    sources = dict(reference=reference_path.read_bytes(), candidate=candidate_path.read_bytes())
    original = TARGET.read_bytes()
    require(original == sources["candidate"], "Entry source must exactly equal the declared candidate; refusing to discard edits")
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    head = subprocess.check_output(["git", "show", f"{git_head}:{TARGET.relative_to(ROOT).as_posix()}"], cwd=ROOT)
    validate_sources(sources["reference"], sources["candidate"], head)
    archive = output / "Sources"
    archive.mkdir()
    for arm, data in sources.items():
        (archive / f"LobeUtils.{arm}.slang").write_bytes(data)
    (archive / "LobeUtils.entry.slang").write_bytes(original)
    (archive / "LobeUtils.HEAD.slang").write_bytes(head)
    for name in HELPERS:
        shutil.copyfile(Path(__file__).with_name(name), archive / name)
    (output / "prepared").mkdir()
    configuration, cases_file, workload_inputs, absent = prepare_workload(args.cases.resolve(strict=True), output)
    executable = args.executable.resolve(strict=True)
    cpp = [p for p in (ROOT / "Source/Runtime/Render").rglob("*") if p.suffix in (".cpp", ".h")]
    inputs = experiment.input_identity(executable, workload_inputs + cpp + list(archive.iterdir()) +
        [Path(__file__).with_name(name) for name in HELPERS] + [reference_path, candidate_path], production=True)
    experiment.save(output / "Inputs.json", inputs)
    experiment.save(output / "AbsentInputs.json", absent)
    inventory = experiment.gpu_inventory()
    require(inventory, "Need an observed NVIDIA GPU/driver inventory")
    source_records = {arm: dict(file=f"Sources/LobeUtils.{arm}.slang", sha256=sha(data)) for arm, data in sources.items()}
    summary.update(workload=configuration, cases_file=cases_file, executable=str(executable), sources=source_records,
        git_head=git_head, git_head_source_sha256=sha(head), entry_source_sha256=sha(original), gpu_inventory=inventory,
        cache_policy="Per-process empty application PSO directory; shared source-aware Slang cache and uncontrolled driver cache; 16 warmup frames.")
    checks, baseline, identities = [], None, {}
    transaction = SourceTransaction(output, original, sources)
    try:
        for row in schedule():
            transaction.install(row["arm"])
            arm_inputs = expected_inputs(inputs, source_records, row["arm"])
            dispatch.snapshot_check(arm_inputs, absent, inventory, "before-" + row["label"], checks)
            experiment.save(output / "SnapshotChecks.json", checks)
            destination = output / row["label"]
            destination.mkdir()
            environment = base.runtime_environment(cases_file, destination)
            command = [str(executable), "--gtest_filter=RHIRendering.material_phase0_baseline",
                       "--rhi-validation" if row["validation"] else "--rhi-no-validation", "--output-dir", str(destination)]
            experiment.save(destination / "Invocation.json", dict(command=command, timeout_seconds=args.timeout,
                environment={key: value for key, value in environment.items() if key.startswith(("METALLIC_", "VK_"))}))
            item = dict(**row, source_sha256=source_records[row["arm"]]["sha256"])
            summary["runs"].append(item)
            experiment.save(output / "summary.json", summary)
            print(f"{row['label']}: validation={'on' if row['validation'] else 'off'}", flush=True)
            try:
                base.execute(command, environment, destination, args.timeout)
            finally:
                if (destination / "process.json").is_file():
                    process = experiment.read(destination / "process.json")
                    item.update(process=process, execution_id=process["execution_id"])
                    experiment.save(output / "summary.json", summary)
            normalized, identity, comparable, baseline = inspect_run(output, summary, item, baseline, identities)
            experiment.save(comparable, normalized)
            comparison = experiment.compare_readbacks(baseline, comparable, configuration, **base.TOLERANCE)
            item.update(identity=identity, correctness=comparison)
            experiment.save(output / "summary.json", summary)
            require(all(case["qualified"] for case in comparison), "HDR/temporal/guide outputs exceeded the unchanged declared tolerance")
            if row["phase"] == "timing":
                item["timings"] = dispatch.verify_timing_rows(normalized, configuration)
            dispatch.snapshot_check(arm_inputs, absent, inventory, "after-" + row["label"], checks)
            experiment.save(output / "SnapshotChecks.json", checks)
            experiment.save(output / "summary.json", summary)
        summary.update(correctness_qualified=True, timing_summary=summarize(summary["runs"]), status="completed")
    finally:
        transaction.restore()
        summary["restoration"] = transaction.journal["state"]


def verify(output, current_inputs=False):
    output = output.resolve(strict=True)
    dispatch.validate_seal(output)
    summary = experiment.read(output / "summary.json")
    require(summary["schema_version"] == 1 and summary["experiment"] == NAME and summary["fixed_properties"] == FIXED and
            summary["tolerance"] == base.TOLERANCE and summary["schedule"] == schedule(), "Experiment contract changed")
    require(summary["status"] in ("completed", "failed"), "Unfinished experiment")
    if "sources" not in summary:
        require(summary["status"] == "failed" and not summary["runs"], "Missing source identity")
        return summary
    require(all(summary["sources"][arm]["file"] == f"Sources/LobeUtils.{arm}.slang" for arm in ARMS),
            "Source snapshot path changed")
    sources = {arm: (output / summary["sources"][arm]["file"]).read_bytes() for arm in ARMS}
    head = (output / "Sources/LobeUtils.HEAD.slang").read_bytes()
    validate_sources(sources["reference"], sources["candidate"], head)
    require(all(sha(sources[arm]) == summary["sources"][arm]["sha256"] for arm in ARMS) and
            sha(head) == summary["git_head_source_sha256"] and
            sha((output / "Sources/LobeUtils.entry.slang").read_bytes()) == summary["entry_source_sha256"] == sha(sources["candidate"]),
            "Archived source identity changed")
    configuration = summary["workload"]
    require(experiment.read(Path(summary["cases_file"])) == configuration, "Case configuration changed")
    validate_workload(configuration)
    inputs, absent = experiment.read(output / "Inputs.json"), experiment.read(output / "AbsentInputs.json")
    require(inputs[str(TARGET.resolve())] == summary["entry_source_sha256"], "Entry source input changed")
    require(len(summary["runs"]) <= len(schedule()), "Unexpected extra GPU process")
    checks_path = output / "SnapshotChecks.json"
    checks = experiment.read(checks_path) if checks_path.is_file() else []
    expected_labels = [prefix + row["label"] for row in schedule() for prefix in ("before-", "after-")]
    require([check["label"] for check in checks] == expected_labels[:len(checks)], "Source snapshot order changed")
    for check in checks:
        row = next(row for row in schedule() if check["label"] in ("before-" + row["label"], "after-" + row["label"]))
        require(check["inputs"] == expected_inputs(inputs, summary["sources"], row["arm"]) and
                check["absent_inputs"] == absent and check["gpu_inventory"] == summary["gpu_inventory"], "Input/device changed during a process")
    baseline, identities, previous_end, execution_ids = None, {}, None, set()
    for item, row in zip(summary["runs"], schedule()):
        require(all(item[key] == value for key, value in row.items()) and item["source_sha256"] == summary["sources"][row["arm"]]["sha256"],
                "Execution order/source treatment changed")
        destination = output / item["label"]
        require(any(check["label"] == "before-" + item["label"] for check in checks), "Process lacks source preflight")
        if "process" not in item:
            require(summary["status"] == "failed" and item is summary["runs"][-1], "Missing process evidence")
            continue
        process = experiment.read(destination / "process.json")
        require(process == item["process"] and process["execution_id"] == item["execution_id"] and
                process["execution_id"] not in execution_ids and (previous_end is None or process["started"] >= previous_end),
                "Process evidence changed or processes overlap")
        expected_command = [summary["executable"], "--gtest_filter=RHIRendering.material_phase0_baseline",
                            "--rhi-validation" if row["validation"] else "--rhi-no-validation", "--output-dir", str(destination)]
        environment = base.runtime_environment(summary["cases_file"], destination, live=False)
        require(process["command"] == expected_command and process["environment"] ==
                {key: value for key, value in environment.items() if key.startswith("METALLIC_")}, "Compiler/process environment changed")
        previous_end = process["ended"]
        execution_ids.add(process["execution_id"])
        if "identity" not in item:
            require(summary["status"] == "failed" and item is summary["runs"][-1], "Missing runtime shader evidence")
            continue
        dispatch.validate_process(process, item["execution_id"])
        normalized, identity, comparable, baseline = inspect_run(output, summary, item, baseline, identities)
        require(experiment.read(comparable) == normalized and identity == item["identity"], "Runtime identity/report changed")
        comparison = experiment.compare_readbacks(baseline, comparable, configuration, **base.TOLERANCE)
        require(comparison == item["correctness"], "Readback comparison changed")
        qualified = all(case["qualified"] for case in comparison)
        if not qualified:
            require(summary["status"] == "failed" and item is summary["runs"][-1] and "timings" not in item,
                    "Failed correctness gate entered performance statistics")
        if "timings" in item:
            require(row["phase"] == "timing" and qualified and
                    dispatch.verify_timing_rows(normalized, configuration) == item["timings"], "Timing evidence changed")
        elif summary["status"] == "completed":
            require(row["phase"] == "correctness", "Missing ordinary timing result")
    journal = experiment.read(output / "SourceTransaction.json")
    require(journal["target"] == str(TARGET) and journal["original_sha256"] == summary["entry_source_sha256"] and
            journal["source_sha256"] == {arm: summary["sources"][arm]["sha256"] for arm in ARMS}, "Restoration target changed")
    require(len(journal["events"]) <= len(schedule()) and len(journal["events"]) >= len(summary["runs"]),
            "Missing or excessive source installations")
    for event, row in zip(journal["events"], schedule()):
        require(event["arm"] == row["arm"] and event["sha256"] == summary["sources"][row["arm"]]["sha256"],
                "Source installation order changed")
    if summary["status"] == "completed":
        require(len(summary["runs"]) == len(schedule()) and len(checks) == len(expected_labels) and
                journal["state"] == summary["restoration"] == "restored" and summary.get("correctness_qualified") is True and
                journal["current_sha256"] == summary["entry_source_sha256"] and
                summary["timing_summary"] == summarize(summary["runs"]), "Incomplete qualified experiment")
    else:
        require("timing_summary" not in summary and summary.get("correctness_qualified") is not True,
                "Failed experiment contains qualified performance summary")
    if current_inputs:
        experiment.check_snapshots(inputs)
        rendering.check_absences(absent)
        require(experiment.gpu_inventory() == summary["gpu_inventory"], "Current GPU inventory changed")
    return summary


def report(output, destination):
    """Read sealed evidence only; failed gates may plot errors, never timings."""
    output, destination = output.resolve(strict=True), destination.resolve()
    require(not destination.is_relative_to(output), "Reports must be outside sealed experiment evidence")
    summary = verify(output)
    destination.mkdir(parents=True, exist_ok=True)
    comparisons = [dict(run=item["label"], **case) for item in summary["runs"]
                   for case in item.get("correctness", [])]
    data = dict(evidence=str(output), summary_sha256=experiment.digest(output / "summary.json"),
                status=summary["status"], decision="inconclusive", error=summary.get("error"),
                scope=summary["scope"], tolerance=summary["tolerance"], fixed_properties=FIXED,
                comparisons=comparisons, timing_summary=summary.get("timing_summary"), python=sys.version)
    images = []
    if comparisons:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        data["matplotlib"] = matplotlib.__version__
        last = summary["runs"][-1]
        rows = last.get("correctness", [])
        if rows:
            fig, ax = plt.subplots(figsize=(8, 3.7), layout="constrained")
            values = [max(capture["max_absolute_error"] for capture in row["readbacks"]) for row in rows]
            ax.barh([row["case"] for row in rows], values,
                    color=["#347f69" if row["qualified"] else "#b34141" for row in rows])
            ax.set_xlabel("Maximum absolute component error across the 16 readbacks")
            ax.set_title(last["label"] + " — unchanged per-component image gate")
            for index, value in enumerate(values):
                ax.annotate(f" {value:.7g}", (value, index), va="center")
            ax.set_xlim(0, max(values, default=0) * 1.22 or 1)
            fig.savefig(destination / "readback_errors.png", dpi=150)
            plt.close(fig)
            images.append("readback_errors.png")
    if summary["status"] == "completed":
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
        for ax, case in zip(axes, base.CASES):
            blocks = summary["timing_summary"]["paired"][case]["dispatch_scope_ms"]["blocks"]
            ax.plot([1, 2, 3], [row["baseline_ms"] for row in blocks], "o-", label="reference")
            ax.plot([1, 2, 3], [row["candidate_ms"] for row in blocks], "o-", label="candidate")
            ax.set(title=case, xlabel="Independent process pair (AB / BA / AB)", ylabel="Path trace shading median (ms)", xticks=[1, 2, 3])
            ax.legend()
        fig.suptitle("n = 3 exploratory pairs; no confirmation stage or established production gain")
        fig.savefig(destination / "shading_pairs.png", dpi=150)
        plt.close(fig)
        images.append("shading_pairs.png")
    experiment.save(destination / "data.json", data)
    rows_html = []
    for row in comparisons:
        errors = row["readbacks"]
        rows_html.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in
            (row["run"], row["case"], len(errors), row["qualified"], max(item["max_absolute_error"] for item in errors))) + "</tr>")
    status = ("Completed: n = 3 independent exploratory pairs; no confirmation stage."
              if summary["status"] == "completed" else
              "FAILED / INCOMPLETE: no qualified production timing conclusion. The schedule stopped; missing timings are not zero.")
    document = f"""<!doctype html><meta charset="utf-8"><title>OpenPBR Fresnel A/B</title>
<style>body{{font:16px system-ui;max-width:1150px;margin:36px auto;padding:0 24px}}img{{max-width:100%}}td,th{{padding:6px;border-bottom:1px solid #ccc;text-align:left}}.status{{font-weight:700;color:#922}}</style>
<h1>OpenPBR Fresnel source A/B</h1><p class="status">{html.escape(status)}</p>
<p>Both arms: inline root 24 bytes, trusted scene parameters; unchanged 512×512 / 2 samples / depth 12 workload.
Validation-on correctness precedes ordinary timing. The fixed threshold is |candidate − reference| ≤ 2e−5 + 2e−5 × |reference| for every component of every output.
Monte Carlo divergence does not waive this gate. Reference self-comparison is not independent correctness evidence.</p>
<p>{html.escape(summary.get('error', ''))}</p><p>{html.escape(summary['scope'])}</p>
<p><a href="{(output / 'summary.json').as_uri()}">Raw summary</a> · <a href="{(output / 'SourceTransaction.json').as_uri()}">Source transaction</a> · <a href="data.json">Report data</a></p>
{''.join(f'<img src="{name}" alt="{name}">' for name in images)}
<table><tr><th>Run</th><th>Case</th><th>Readbacks</th><th>Qualified</th><th>Max absolute error</th></tr>{''.join(rows_html)}</table>
<p>No new GPU work was performed by this report. A function microbenchmark, if run separately, has a different scope and cannot rescue a failed production image gate.</p>"""
    (destination / "Report.html").write_text(document, encoding="utf-8")
    script = "import sys\nfrom pathlib import Path\n" + f"sys.path.insert(0, {str(Path(__file__).parent)!r})\n" + \
             "from MeasureOpenPBRFresnel import report\n" + f"report(Path({str(output)!r}), Path(__file__).resolve().parent)\n"
    (destination / "plot_report.py").write_text(script, encoding="utf-8")
    return dict(report=str(destination / "Report.html"), status=summary["status"], images=images)


def recover(output, receipt):
    output = output.resolve(strict=True)
    journal = experiment.read(output / "SourceTransaction.json")
    require(journal["target"] == str(TARGET) and not process_alive(journal["owner_pid"]), "Wrong target or runner still alive")
    original = (output / "Sources/LobeUtils.entry.slang").read_bytes()
    require(sha(original) == journal["original_sha256"], "Recovery original is damaged")
    current = TARGET.read_bytes()
    require(sha(current) in {journal["original_sha256"], *journal["source_sha256"].values()},
            "Concurrent source edit; refusing recovery overwrite")
    # The shared process helper writes its receipt on exit. If its parent was
    # forcibly killed before that point, child liveness cannot be established.
    # Refuse automatic source recovery instead of guessing that child is gone.
    for invocation in output.glob("*/Invocation.json"):
        require((invocation.parent / "process.json").is_file(),
                "Missing child completion receipt; inspect the interrupted child before manual recovery")
    for path in output.glob("*/process.json"):
        process = experiment.read(path)
        require(not process.get("pid") or not process_alive(process["pid"]), "Recorded child process is still alive")
    lock = ROOT / "build/compute-pipeline-experiment.lock"
    if lock.exists():
        owner = experiment.read(lock)
        require(Path(owner["output"]).resolve() == output and not process_alive(owner["pid"]), "Lock has another/live owner")
    receipt = receipt.resolve()
    require(not receipt.is_relative_to(output) and not receipt.exists(), "Recovery receipt must be new and outside sealed evidence")
    if current != original:
        atomic_replace(TARGET, original, current)
    experiment.save(receipt, dict(state="restored", source=str(TARGET), sha256=sha(original), evidence=str(output), time=time.time()))
    if lock.exists():
        lock.unlink()
    return dict(state="restored", receipt=str(receipt))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    live = sub.add_parser("run")
    for name in ("executable", "cases", "reference", "candidate", "output-dir"):
        live.add_argument("--" + name, type=Path, required=True)
    live.add_argument("--timeout", type=float, default=1200)
    check = sub.add_parser("verify")
    check.add_argument("output", type=Path)
    check.add_argument("--current-inputs", action="store_true")
    plot = sub.add_parser("report")
    plot.add_argument("output", type=Path)
    plot.add_argument("--output-dir", type=Path, required=True)
    restore = sub.add_parser("recover")
    restore.add_argument("output", type=Path)
    restore.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify":
        summary = verify(args.output, args.current_inputs)
        print(json.dumps({key: summary.get(key) for key in ("status", "error", "restoration", "timing_summary")}, indent=2))
        return
    if args.command == "recover":
        print(json.dumps(recover(args.output, args.receipt), indent=2))
        return
    if args.command == "report":
        print(json.dumps(report(args.output, args.output_dir), indent=2))
        return
    require(dispatch.positive_finite(args.timeout), "Timeout must be finite and positive")
    for name in experiment.LOCK_NAMES:
        require(not (ROOT / "build" / name).exists(), "Existing GPU experiment lock; inspect owner first")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    lock = ROOT / "build/compute-pipeline-experiment.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    summary = dict(schema_version=1, experiment=NAME, status="running", decision="inconclusive", runs=[],
        schedule=schedule(), fixed_properties=FIXED, tolerance=base.TOLERANCE, scope=dispatch.MEASUREMENT_SCOPE,
        caveats=["Three exploratory independent process pairs, no confirmation stage.",
                 "Only the declared scalar Fresnel source may change; original entry bytes are restored.",
                 "Strict deterministic image gates can reject Monte Carlo path divergence; a rejection establishes no production gain.",
                 "Cooperative locks do not prove system-wide GPU exclusivity; clocks are unaltered."])
    with lock.open("x", encoding="utf-8") as stream:
        json.dump(dict(pid=os.getpid(), output=str(output), experiment=NAME), stream)
    failed = False
    try:
        experiment.save(output / "summary.json", summary)
        run(args, output, summary)
    except BaseException as error:
        failed = True
        summary.pop("timing_summary", None)
        summary.pop("correctness_qualified", None)
        summary.update(status="failed", error=str(error), failure_type=type(error).__name__)
        (output / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        experiment.save(output / "summary.json", summary)
        experiment.seal(output)
        lock.unlink()
    verify(output)
    print(output / "summary.json")
    if failed:
        parser.exit(1, "Experiment failed; retained all available evidence and stopped the schedule.\n")


if __name__ == "__main__":
    main()
