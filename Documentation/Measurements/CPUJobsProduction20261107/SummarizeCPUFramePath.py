"""Summarize existing non-overlapping coordinator intervals; launches no workload."""
import collections
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
output = {"definition": "Arithmetic means of disjoint wall intervals over each process's same measured frames; includes waits, not OS CPU running time", "processes": []}
for line in (root / "Raw.jsonl").read_text(encoding="utf-8").splitlines():
    process = json.loads(line)
    if process["phase"] != "measured":
        continue
    path = root / process["artifacts"]["frames"]
    totals = collections.defaultdict(float)
    count = 0
    with path.open(encoding="utf-8") as frames:
        for row in frames:
            frame = json.loads(row)
            for segment in frame["cpuFramePathSegmentsMs"]:
                totals[segment["name"]] += segment["endMs"] - segment["startMs"]
            count += 1
    output["processes"].append({"runId": process["runId"], "candidate": process["candidate"],
        "frames": count, "frameMeanMs": process["frameMs"]["mean"],
        "segmentMeanMs": {name: value / count for name, value in totals.items()},
        "framesSHA256": process["artifactSha256"]["frames"]})
output["scriptSHA256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
Path(sys.argv[2]).write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
for process in output["processes"]:
    print(process["runId"], {key: round(value, 4) for key, value in process["segmentMeanMs"].items()})
