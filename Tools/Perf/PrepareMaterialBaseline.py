"""Prepare explicit production material workloads without modifying lookdev assets."""
import argparse
import copy
from pathlib import Path
from MaterialBaseline import ROOT, read, save


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    studio = ROOT / "build/MaterialValidation/WhiteStudio02"
    catalog = read(studio / "Catalog.json")
    specs = []

    def add(identifier, graph, scene, target, **extra):
        name = identifier + ".json"
        graph["outputs"] = [target]
        save(output / name, graph)
        specs.append(dict(id=identifier, graph=name, scene=scene, output=target,
                          width=512, height=512, requiredTiming=[target.split('.')[0], None], **extra))

    def surface(identifier, original, scene, **extra):
        for mode in ("Deferred", "PT"):
            graph = copy.deepcopy(original)
            names = {"Deferred", "VBuffer"} if mode == "Deferred" else {"Reference"}
            graph["nodes"] = [n for n in graph["nodes"] if n["name"] in names]
            graph["edges"] = [e for e in graph["edges"] if e["src"].split('.')[0] in names and e["dst"].split('.')[0] in names]
            for node in graph["nodes"]:
                p = node["properties"]
                if node["name"] == "Deferred":
                    p.update(samples=64, materialBinning=True, halfPrecision=True,
                             exportUpscalerGuides=False, stochasticTextureFiltering=False,
                             debugDisableShadows=False, debugDisableTransmission=False, debugView="final")
                if node["name"] == "Reference":
                    p.update(samples=4, maxDepth=12, accumulate=True, outputLinear=True)
            add(identifier + "-" + mode, graph, scene,
                "Deferred.color" if mode == "Deferred" else "Reference.color",
                group="Surface", backend=mode, **extra)

    for sample in catalog["samples"]:
        identifier = sample["id"].removeprefix("studio-white-")
        if identifier in ("overview", "chart"):
            continue
        surface(identifier, read(Path(sample["graphPath"])), sample["scenePath"])

    source = next(s for s in catalog["samples"] if s["id"] == "studio-white-M01_NeutralDielectric-uniform")
    slab = read(ROOT / "Asset/Materials/SlabLayer.materialdef")
    for name in ("SlabSingle", "SlabMix", "SlabLayer"):
        definition = copy.deepcopy(slab)
        closure = definition["surfaceProgram"]["closure"]
        if name == "SlabSingle":
            definition["surfaceProgram"]["closure"] = closure["a"]
        elif name == "SlabMix":
            closure["op"] = "mix"
            closure["weight"] = 0.5
        save(output / (name + ".materialdef"), definition)
        save(output / (name + ".material"), dict(type="Metallic.MaterialInstance", version=1,
            definitionVersion=1, definition="asset://" + name + ".materialdef"))
        surface(name, read(Path(source["graphPath"])), source["scenePath"],
                materialAsset="asset://" + name + ".material", materialRoot=str(output.resolve()))

    graph = read(ROOT / "LookDev/MaterialSystem/Fiber/Chiang.metallic_graph.json")
    add("RTXCRChiang-PT", graph, None, "PathTrace.color", group="Fiber", backend="PT",
        environment=dict(path="External/RTXCR-Assets/EnvironmentMaps/studio_small_09_1k.hdr", intensity=1.5, rotationDegrees=25.0))
    graph = read(ROOT / "Pipelines/Samples/native_strands.metallic_graph.json")
    graph["nodes"] = [n for n in graph["nodes"] if n["name"] in ("Strands", "Fiber")]
    graph["edges"] = [e for e in graph["edges"] if e["dst"].startswith("Fiber.")]
    add("NativeStrands", graph, None, "Fiber.color", group="NativeStrands", backend="Layers8", requireEnvironment=False)
    roots = [studio, ROOT / "build/MaterialValidation/PainterLookDev", ROOT / "Asset/Materials",
             ROOT / "Asset/Strands", ROOT / "External/RTXCR-Assets/Claire"]
    suffixes = {".gltf", ".glb", ".bin", ".png", ".jpg", ".jpeg", ".tga", ".hdr", ".exr", ".json", ".material", ".materialdef", ".slang"}
    assets = {p.resolve() for root in roots for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes and not p.name.endswith(".meshlets.bin")}
    assets.add((ROOT / "External/RTXCR-Assets/EnvironmentMaps/studio_small_09_1k.hdr").resolve())
    assets.update(p.resolve() for p in output.glob("*.material*"))
    save(output / "Cases.json", dict(version=2, frames=97, warmupFrames=32, timingFrames=64,
        aaRelativeRangeLimit=0.10, cases=specs, assetFiles=sorted(str(p) for p in assets)))
    print(f"Prepared {len(specs)} cases; {len(assets)} asset files; {sum(p.stat().st_size for p in assets)/2**30:.2f} GiB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    prepare(parser.parse_args().output.resolve())
