"""Correlate xperf CSV exports with a Full benchmark (all times in ETW us).

Export --markers with the Metallic.Pacing provider; --cpu is an optional
unfiltered dumper export with -stacktimeshifting (a short -range is sufficient).
--gpu is an optional DxgKrnl-only export. HAGS packet submission/completion
notifications are NOT hardware execution start/end timestamps.
"""
import argparse
import bisect
import collections
import csv
import json
from pathlib import Path

from AnalyzeZorahFullRoam import distribution


def events(path, prefixes):
    # xperf may include ANSI module paths in otherwise ASCII exports. Numeric
    # event fields used here are ASCII; replace undecodable display-name bytes.
    with path.open(encoding="utf-8-sig", errors="replace") as stream:
        for line in stream:
            if not line.lstrip().startswith(prefixes):
                continue
            row = [value.strip() for value in next(csv.reader([line], skipinitialspace=True))]
            if len(row) > 1 and row[1].isdigit():
                yield row


def analyze(args):
    markers = list(events(args.markers, ("Metallic.Pacing/FramePhase/",)))
    phases = collections.defaultdict(dict)
    bounds = {r[9]: int(r[1]) for r in markers if r[9] in ("CaptureBegin", "CaptureEnd")}
    starts = [r for r in markers if r[9] == "SleepBegin"]
    if not starts:
        raise ValueError("No Sleep markers; enable METALLIC_PACING_TRACE=1 before launch")
    process, tid = starts[0][2:4]
    if any(r[2:4] != [process, tid] for r in starts):
        raise ValueError("Export must contain one workload process/main thread")
    for r in markers:
        if r[9] in ("SleepBegin", "SleepEnd", "PCLMarker"):
            phases[int(r[10])][r[9] + (":" + r[11] if r[9] == "PCLMarker" else "")] = int(r[1])
    frames = [json.loads(line) for line in (args.run / "Frames.jsonl").read_text(encoding="utf-8-sig").splitlines()]
    # Cache refreshes are sparse: match the report's ID, never the exporting frame.
    reports = {r["streamlineBegin"]["reportFrameId"]: r["streamlineBegin"]["driverTiming"]
               for r in frames if r["streamlineBegin"]["reportAvailable"]}
    submits = sorted(int(r[1]) for r in markers if r[9] == "QueueSubmitBegin" and int(r[11]) == args.graphics_family)
    correlations = []
    for frame, report in sorted(reports.items()):
        p = phases[frame]
        if not all(key in p for key in ("SleepBegin", "SleepEnd", "PCLMarker:0", "PCLMarker:2", "PCLMarker:3")):
            continue
        if not bounds.get("CaptureBegin", 0) <= p["SleepBegin"] <= bounds.get("CaptureEnd", float("inf")):
            continue
        # Re-anchor EACH report using its own SimulationStart; a single offset
        # drifts by ~0.6 ms across this 30 s capture. Marker call overhead remains.
        offset = report["simulationStartUs"] - p["PCLMarker:0"]
        gpu_start = report["gpuRenderStartUs"] - offset
        gpu_end = report["gpuRenderEndUs"] - offset
        if report["gpuFrameTimeUs"] <= 0:
            continue
        previous_end = gpu_end - report["gpuFrameTimeUs"]
        index = bisect.bisect_left(submits, p["PCLMarker:2"])
        first = submits[index] if index < len(submits) and submits[index] <= p["PCLMarker:3"] else None
        correlations.append(dict(frame=frame, sleepBeginUs=p["SleepBegin"], sleepEndUs=p["SleepEnd"],
            submitBeginUs=p["PCLMarker:2"], firstGraphicsSubmitUs=first, submitEndUs=p["PCLMarker:3"],
            gpuStartUs=gpu_start, gpuEndUs=gpu_end, previousGpuEndUs=previous_end,
            signedEnvelopeGapMs=(gpu_start - previous_end) / 1000,
            anchorOffsetUs=offset))
    result = dict(process=process, mainTid=tid, captureBoundsUs=bounds,
        markerCoverageUs=[min(int(r[1]) for r in markers), max(int(r[1]) for r in markers)],
        driverCorrelations=correlations,
        signedEnvelopeGapMs=distribution([r["signedEnvelopeGapMs"] for r in correlations]))

    if args.cpu:
        switches, ready, stacks = [], [], collections.defaultdict(list)
        for r in events(args.cpu, ("CSwitch,", "ReadyThread,", "Stack,")):
            if r[0] == "CSwitch" and (r[3] == tid or r[9] == tid):
                switches.append(r)
            elif r[0] == "ReadyThread" and r[5] == tid:
                ready.append(r)
            elif r[0] == "Stack" and r[2] == tid:
                stacks[int(r[1])].append(r[-1])
        if not switches:
            raise ValueError("CPU export has no workload thread switches")
        ready.sort(key=lambda r: int(r[1]))
        ready_times = [int(r[1]) for r in ready]
        spans, previous = [], None
        for r in sorted(switches, key=lambda r: int(r[1])):
            time = int(r[1])
            if r[3] == tid and previous:
                start, state, reason = previous
                i = bisect.bisect_left(ready_times, start)
                wake = ready[i] if i < len(ready) and ready_times[i] <= time else None
                wake_time = int(wake[1]) if wake else time
                if state != "Waiting":
                    wake_time = start
                spans.append(dict(beginUs=start, readyUs=wake_time, endUs=time, state=state, reason=reason,
                    waker=([wake[2], wake[3]] if wake else None), stack=stacks[time]))
                previous = None
            if r[9] == tid:
                previous = (time, r[12], r[13])
        coverage = [max(bounds.get("CaptureBegin", 0), min(int(r[1]) for r in switches)),
                    min(bounds.get("CaptureEnd", float("inf")), max(int(r[1]) for r in switches))]
        sleeps = []
        for frame, p in sorted(phases.items()):
            if "SleepBegin" not in p or "SleepEnd" not in p:
                continue
            a, b = p["SleepBegin"], p["SleepEnd"]
            if a < coverage[0] or b > coverage[1]:
                continue
            intersect = lambda x, y: max(0, min(y, b) - max(x, a)) / 1000
            waits = [s for s in spans if s["beginUs"] < b and s["endUs"] > a]
            blocked = sum(intersect(s["beginUs"], s["readyUs"]) for s in waits)
            runnable = sum(intersect(s["readyUs"], s["endUs"]) for s in waits)
            sleeps.append(dict(frame=frame, beginUs=a, endUs=b, sleepMs=(b-a)/1000,
                blockedMs=blocked, readyMs=runnable, runningMs=(b-a)/1000-blocked-runnable, waits=waits))
        long_sleeps = [s for s in sleeps if s["sleepMs"] > 20]
        result["cpu"] = dict(coverageUs=coverage, sleeps=sleeps,
            longSleepSummary={k: distribution([s[k] for s in long_sleeps])
                              for k in ("sleepMs", "blockedMs", "readyMs", "runningMs")},
            note="Ready time includes preemption. Missing ReadyThread is conservatively counted blocked; inspect waits/stacks. Running is a residual, not sampled CPU work.")

    if args.gpu:
        dma = list(events(args.gpu, ("Microsoft-Windows-DxgKrnl/DmaPacket/win:Info,",)))
        # These two payload layouts are HAGS submit and completion notifications.
        contexts = collections.Counter(r[9] for r in dma if len(r) == 14 and r[2] == process)
        if contexts:
            primary = contexts.most_common(1)[0][0]
            primary_events = [r for r in dma if r[9] == primary and len(r) in (11, 14)]
            result["gpu"] = dict(primaryContext=primary, contextSubmissionCounts=dict(contexts),
                coverageUs=[min(int(r[1]) for r in dma), max(int(r[1]) for r in dma)],
                packets=[dict(timeUs=int(r[1]), kind="submit" if len(r) == 14 else "complete",
                              fence=r[10], process=r[2], tid=r[3]) for r in primary_events],
                note="Most frequently submitted workload HAGS context; other queues/processes may remain active. Notification times are not GPU hardware execution times.")
        else:
            result["gpu"] = dict(note="No recognized workload HAGS notifications; do not infer GPU idle.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--markers", type=Path, required=True)
    parser.add_argument("--cpu", type=Path)
    parser.add_argument("--gpu", type=Path)
    parser.add_argument("--graphics-family", type=int, default=0,
                        help="Graphics queue family from this device's setup (5070 Ti capture: 0)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(analyze(args), indent=2), encoding="utf-8")
