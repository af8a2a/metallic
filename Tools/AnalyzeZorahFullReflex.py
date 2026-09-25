"""Attribute frame-begin wall time without summing independent percentiles.

Usage: python Tools/AnalyzeZorahFullReflex.py <capture/run1> [<capture/run1> ...]
Driver reports are cached (normally refreshed every 60 frames), not current-frame GPU timings.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from AnalyzeZorahFullRoam import distribution, frame_stats

PHASES = ("mutexWaitMs", "tokenMs", "optionsMs", "sleepMs", "statusMs", "markerMs")


def analyze(root):
    capture = json.loads((root / "Capture.json").read_text(encoding="utf-8-sig"))
    rows = [json.loads(line) for line in (root / "Frames.jsonl").read_text(encoding="utf-8-sig").splitlines()]
    if capture["status"] != "capture_complete" or len(rows) != capture["frames"] or not rows:
        raise ValueError("Incomplete capture")
    definitions = {s["id"]: s["path"] for s in capture["scopes"]}
    for row in rows:
        begin = row["streamlineBegin"]
        if not begin["active"] or any(not math.isfinite(begin[k]) or begin[k] < 0 for k in (*PHASES, "totalMs")):
            raise ValueError("Invalid frame-begin timings")
        if sum(begin[k] for k in PHASES) > begin["totalMs"] + .001:
            raise ValueError("Phase time exceeds constructor time")
        row["cpu"] = {definitions[s["id"]]: s["cpuMs"] for s in row["scopes"]}
        row["frontWaitMs"] = row["cpu"].get("Frame/Wait Frame Slot Before Input", 0) + begin["sleepMs"]
        row["outsideFrontWaitMs"] = row["frameMs"] - row["frontWaitMs"]
        row["gpu"] = {definitions[s["id"]]: s["gpuMs"] for s in row["scopes"] if s["gpuMs"] is not None}

    def summarize(selected):
        if not selected:
            return None
        phases = {key: distribution([r["streamlineBegin"][key] for r in selected]) for key in (*PHASES, "totalMs")}
        total = sum(r["streamlineBegin"]["totalMs"] for r in selected)
        return dict(count=len(selected), phases=phases,
                    frameSlotPlusSleepMs=distribution([r["frontWaitMs"] for r in selected]),
                    outsideFrontWaitMs=distribution([r["outsideFrontWaitMs"] for r in selected]),
                    sleepShare=sum(r["streamlineBegin"]["sleepMs"] for r in selected)/total if total else None,
                    nonSleepMs=distribution([r["streamlineBegin"]["totalMs"]-r["streamlineBegin"]["sleepMs"] for r in selected]),
                    **frame_stats(selected))

    slow = sorted(rows, key=lambda r: r["streamlineBegin"]["totalMs"], reverse=True)[:math.ceil(len(rows)*.05)]
    stages = defaultdict(list)
    for row in rows:
        stages[row["stage"]].append(row)
    scope_paths = [path for path in definitions.values() if any(path.endswith("/" + token) for token in
        ("Wait Frame Slot Before Input", "Wait Slot Completion", "Acquire Swapchain Image", "Present",
         "RenderGraph GPU envelope", "Streamline frame begin / Reflex pacing", "Wait External", "Wait Submitted"))]
    scopes = {}
    for path in scope_paths:
        scopes[path] = {kind: distribution([r[kind][path] for r in rows if path in r[kind]]) for kind in ("cpu", "gpu")}
    reports = {r["streamlineBegin"]["reportFrameId"]: r["streamlineBegin"] for r in rows if r["streamlineBegin"]["reportAvailable"]}
    # Reports are refreshed periodically and then repeated in frame exports.
    # Deduplicate by driver frame ID; never pair cached timestamps with the
    # current application frame or infer exact GPU idle time across report gaps.
    raw_reports = [r["driverTiming"] for r in reports.values() if "driverTiming" in r]

    def interval(start, end):
        return distribution([(r[end] - r[start]) / 1000.0 for r in raw_reports
                             if r[start] and r[end] >= r[start]])

    # The bundled Vulkan backends synthesize gpuActiveRenderTimeUs from the
    # start/end span; it is not a hardware busy-time counter.
    driver_timing = dict(
        activeFieldIsHardwareBusyTime=False,
        simulationMs=interval("simulationStartUs", "simulationEndUs"),
        renderSubmitMs=interval("renderSubmitStartUs", "renderSubmitEndUs"),
        presentMs=interval("presentStartUs", "presentEndUs"),
        simulationToGpuStartMs=interval("simulationStartUs", "gpuRenderStartUs"),
        gpuActiveMs=distribution([r["gpuActiveRenderTimeUs"] / 1000.0 for r in raw_reports
                                  if r["gpuActiveRenderTimeUs"] > 0]),
        gpuFrameMs=distribution([r["gpuFrameTimeUs"] / 1000.0 for r in raw_reports
                                 if r["gpuFrameTimeUs"] > 0]))
    result = dict(source=str(root), conditions={k: capture[k] for k in
        ("outputExtent", "renderExtent", "hidden", "vsync", "frameSlots", "config")},
        effectiveModes=sorted({r["streamlineBegin"]["effectiveMode"] for r in rows}),
        frameLimitUs=sorted({r["streamlineBegin"]["frameLimitUs"] for r in rows}),
        sleepCalls=sum(r["streamlineBegin"]["sleepCalled"] for r in rows),
        optionsUpdates=sum(r["streamlineBegin"]["optionsUpdated"] for r in rows),
        suspendedFrames=sum(r["streamlineBegin"]["suspended"] for r in rows),
        all=summarize(rows), slowestFivePercent=summarize(slow),
        refreshFrames=summarize([r for r in rows if r["streamlineBegin"]["statusRefreshed"]]),
        stages={k: summarize(v) for k, v in stages.items()}, scopes=scopes,
        cachedDriverReports={"uniqueReports": len(reports), "timing": driver_timing,
            "renderLatencyMs": distribution([r["renderLatencyMs"] for r in reports.values()]),
            "gpuRenderMs": distribution([r["gpuRenderMs"] for r in reports.values()])},
        slowest=[{k: r[k] for k in ("frame", "seconds", "stage", "frameMs", "streamlineBegin", "frontWaitMs", "cpu", "gpu")} for r in slow[:10]])
    (root / "ReflexSummary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("source", "effectiveModes", "all", "slowestFivePercent", "cachedDriverReports")}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path, nargs="+")
    for root in parser.parse_args().captures:
        analyze(root)
