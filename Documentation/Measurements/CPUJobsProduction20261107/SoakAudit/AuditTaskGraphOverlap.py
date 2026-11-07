"""Count actual DAG records against recorded measured-frame clock envelopes.

python -B AuditTaskGraphOverlap.py <soak root> <new output JSON>
Only first/last frame JSON is decoded; TaskGraph streams are small and read once.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


def main():
    root, output = Path(sys.argv[1]), Path(sys.argv[2])
    if output.exists():
        raise ValueError("Use a new output JSON")
    results = []
    for line in (root / "Raw.jsonl").read_text(encoding="utf-8-sig").splitlines():
        row = json.loads(line)
        paths = {key: root / value.replace("\\", "/") for key, value in row["artifacts"].items()}
        with paths["frames"].open("rb") as stream:
            first_line = stream.readline()
            last_line = first_line
            for frame_line in stream:
                if frame_line.strip():
                    last_line = frame_line
        first, last = json.loads(first_line), json.loads(last_line)
        start = first["originSteadyNs"]
        end = last["originSteadyNs"] + last["frameMs"] * 1e6
        all_names, overlaps, contained = Counter(), Counter(), Counter()
        digest = hashlib.sha256()
        with paths["taskGraphs"].open("rb") as stream:
            for graph_line in stream:
                digest.update(graph_line)
                graph = json.loads(graph_line)
                name = graph["name"]
                all_names[name] += 1
                if graph["submittedNs"] < end and graph["completedNs"] > start:
                    overlaps[name] += 1
                if start <= graph["submittedNs"] <= graph["completedNs"] <= end:
                    contained[name] += 1
        actual = digest.hexdigest()
        if actual != row["artifactSha256"]["taskGraphs"]:
            raise ValueError("TaskGraph artifact SHA mismatch")
        results.append({"runId": row["runId"], "candidate": row["candidate"], "firstFrame": first["frame"],
            "lastFrame": last["frame"], "startSteadyNs": start, "endSteadyNs": end,
            "graphNamesAll": dict(all_names), "graphNamesOverlappingMeasuredEnvelope": dict(overlaps),
            "graphNamesContainedInMeasuredEnvelope": dict(contained), "taskGraphsSHA256": actual})
    report = {"schema": 1, "scriptSHA256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "definition": "Overlap uses actual DAG submit/completion clocks against first measured frame origin through last measured frame end; names/counts do not imply a CPU critical path.",
        "runtimeIdentityGuard": "Source/Editor/EditorCPUJobBenchmark.cpp:408 checks every captured draw for viewport output width/height and stable graph generation; thrown violations prevent passed status. Per-frame extent fields are not exported.",
        "processes": results}
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
