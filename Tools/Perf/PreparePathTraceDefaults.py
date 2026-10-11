"""Prepare the three PT budget comparisons at the registered user camera/size."""
import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINE = Path(__file__).with_name("PathTraceBaseline.ABeautifulGame.json")


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def prepare(output, baseline):
    settings = json.loads(baseline.read_text(encoding="utf-8"))
    source = ROOT / "Pipelines/Samples/pathtracing_abeautiful_game_openpbr_dlss_rr.metallic_graph.json"
    template = json.loads(source.read_text(encoding="utf-8"))
    node = next(n for n in template["nodes"] if n["name"] == "PathTrace")
    scene = ROOT / node["properties"]["path"]
    model = json.loads(scene.read_text(encoding="utf-8"))
    assets = {baseline, Path(__file__).resolve(), source, scene,
              ROOT / settings["environment"]["path"]}
    for item in model.get("buffers", []) + model.get("images", []):
        uri = item.get("uri", "")
        if uri and not uri.startswith("data:"):
            assets.add(scene.parent / uri)
    sidecar = scene.with_suffix(".metallic_scene.json")
    if sidecar.is_file():
        assets.add(sidecar)
    for path in assets:
        if not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=False)
    save(output / "Baseline.json", settings)
    cases = []
    for name, samples in (("Previous2Rpp", 2), ("OneRppReference", 1), ("DefaultOneRppSharc", None)):
        current = copy.deepcopy(node)
        props = current["properties"]
        props.update(camera=settings["camera"], accumulate=False,
                     outputLinear=True, exportDenoiserGuides=True)
        # Exercise real runtime defaults in the default arm, while keeping the
        # uncached references explicit and independent of future sample edits.
        for key in list(props):
            if key in ("samples", "maxDepth", "cacheMode") or key.startswith("sharc."):
                del props[key]
        if samples is not None:
            props.update(samples=samples, maxDepth=12, cacheMode="off")
        graph = dict(version=1, name=name, nodes=[current], edges=[], outputs=["PathTrace.color"])
        save(output / (name + ".json"), graph)
        cases.append(dict(id=name, graph=name + ".json", scene=None, output="PathTrace.color",
                          width=settings["width"], height=settings["height"],
                          environment=settings["environment"],
                          requiredTiming=["PathTrace", "SHaRC query" if samples is None else "Path trace shading"]))
    assets.add(output / "Baseline.json")
    save(output / "Cases.json", dict(version=2, frames=settings["frames"],
         warmupFrames=settings["warmupFrames"], timingFrames=settings["timingFrames"],
         aaRelativeRangeLimit=0.1, cases=cases, assetFiles=[str(p) for p in sorted(assets)]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    args = parser.parse_args()
    prepare(args.output.resolve(), args.baseline.resolve())
