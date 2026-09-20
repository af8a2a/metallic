"""Inspect Zorah texture residency plans using metadata only; no GPU or payload decode."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

from AuditZorahFull import ktx_header


def first_mip(info, cap):
    mip = 0
    while mip + 1 < info["levels"] and max(info["width"] >> mip, info["height"] >> mip) > cap:
        mip += 1
    return mip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("Asset/ZorahFull/zorah_textured_public.v1.gltf"))
    parser.add_argument("--output", type=Path, default=Path("build-release/zorah-texture-residency/analysis.json"))
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    raw = source.read_bytes()
    scene = json.loads(raw)
    paths = [source.parent / image["uri"] for image in scene["images"]]
    with ThreadPoolExecutor(max_workers=4) as workers:
        images = list(workers.map(ktx_header, paths))

    slots = defaultdict(set)
    masked_images = set()
    material_images = []
    for material in scene["materials"]:
        used = set()

        def visit(value, key=""):
            if isinstance(value, dict):
                if key.endswith("Texture") and "index" in value:
                    image = scene["textures"][value["index"]]["source"]
                    slots[key].add(image)
                    used.add(image)
                    if material.get("alphaMode") == "MASK" and key == "baseColorTexture":
                        masked_images.add(image)
                for name, child in value.items():
                    visit(child, name)
            elif isinstance(value, list):
                for child in value:
                    visit(child, key)

        visit(material)
        material_images.append(used)

    active_materials = set()
    visited = set()
    pending = list(scene["scenes"][scene.get("scene", 0)]["nodes"])
    while pending:
        node_index = pending.pop()
        if node_index in visited:
            continue
        visited.add(node_index)
        node = scene["nodes"][node_index]
        pending.extend(node.get("children", []))
        if "mesh" in node:
            for primitive in scene["meshes"][node["mesh"]]["primitives"]:
                if "material" in primitive:
                    active_materials.add(primitive["material"])
    active_images = set().union(*(material_images[i] for i in active_materials))

    def summarize(bases):
        payload = sum(sum(info["gpuMipBytes"][base:]) for info, base in zip(images, bases))
        return {"payloadBytes": payload, "payloadMiB": payload / 2**20,
                "mipCount": sum(info["levels"] - base for info, base in zip(images, bases)),
                "maxDimensionHistogram": dict(Counter(str(max(1, info["width"] >> base, info["height"] >> base))
                    for info, base in zip(images, bases)))}

    caps = {str(cap): summarize([first_mip(info, cap) for info in images]) for cap in (64, 128, 256, 512)}
    mixed = summarize([first_mip(info, 512 if i in masked_images else 128) for i, info in enumerate(images)])
    mixed256 = summarize([first_mip(info, 512 if i in masked_images else 256) for i, info in enumerate(images)])
    reference_plans = {}
    for budget_mib in (512, 1024, 4096):
        bases = [0] * len(images)
        total = sum(sum(info["gpuMipBytes"]) for info in images)
        while total > budget_mib * 2**20:
            dropped = False
            for i, info in enumerate(images):
                if bases[i] + 1 == info["levels"]:
                    continue
                total -= info["gpuMipBytes"][bases[i]]
                bases[i] += 1
                dropped = True
                if total <= budget_mib * 2**20:
                    break
            if not dropped:
                break
        reference_plans[str(budget_mib)] = summarize(bases)

    report = {"source": str(source), "gltfSha256": hashlib.sha256(raw).hexdigest(),
              "scope": "Header-only decoded BC payload estimates. Excludes GPU alignment, allocation overhead, staging and fallback. No visibility or GPU residency measurement.",
              "images": len(images), "materials": len(material_images),
              "activeMaterials": len(active_materials), "activeImages": len(active_images),
              "unreferencedImageCount": len(images) - len(active_images),
              "uniqueImagesBySlot": {slot: len(indices) for slot, indices in slots.items()},
              "maskedBaseColorImages": len(masked_images), "uniformCaps": caps,
              "base128MaskBaseColor512": mixed,
              "base256MaskBaseColor512": mixed256,
              "referenceRoundRobinPlans": reference_plans}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
