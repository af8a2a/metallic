"""Generate small, reproducible ZorahFull glTF probes without copying payloads.

Keeps ancestor transforms, GPU instance attributes and all material JSON. Only
instance TRS ranges are read to produce independent expected world matrices.
Geometry and KTX2 payloads stay in the source asset. LargestMesh is metadata-only
by default; cooking it is an explicit later-stage memory stress test.
"""

import argparse
import copy
import hashlib
import json
import math
import shlex
import struct
from pathlib import Path
from urllib.parse import unquote


EXT = "EXT_mesh_gpu_instancing"


def write_json(path, value):
    text = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")


def texture_infos(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key.endswith("Texture") and isinstance(child, dict) and "index" in child:
                yield child
            else:
                yield from texture_infos(child)
    elif isinstance(value, list):
        for child in value:
            yield from texture_infos(child)


def identity():
    return [float(i // 4 == i % 4) for i in range(16)]


def multiply(a, b):
    # glTF matrices are column-major, including the independent reference data.
    return [sum(a[k * 4 + row] * b[col * 4 + k] for k in range(4))
            for col in range(4) for row in range(4)]


def matrix(node):
    if "matrix" in node:
        return node["matrix"]
    x, y, z, w = node.get("rotation", [0, 0, 0, 1])
    length = math.sqrt(x*x + y*y + z*z + w*w)
    x, y, z, w = [v / length for v in (x, y, z, w)]
    result = [1-2*(y*y+z*z), 2*(x*y+z*w), 2*(x*z-y*w), 0,
              2*(x*y-z*w), 1-2*(x*x+z*z), 2*(y*z+x*w), 0,
              2*(x*z+y*w), 2*(y*z-x*w), 1-2*(x*x+y*y), 0,
              *node.get("translation", [0, 0, 0]), 1]
    for col, scale in enumerate(node.get("scale", [1, 1, 1])):
        for row in range(3):
            result[col * 4 + row] *= scale
    return result


def instance_transforms(scene, source_dir, node):
    attributes = node.get("extensions", {}).get(EXT, {}).get("attributes")
    if attributes is None:
        return [identity()]
    decoded = {}
    for semantic, index in attributes.items():
        accessor = scene["accessors"][index]
        view = scene["bufferViews"][accessor["bufferView"]]
        if accessor["componentType"] != 5126 or "extensions" in view or "sparse" in accessor:
            raise ValueError("ZorahFull reference requires plain float instance accessors")
        components = {"VEC3": 3, "VEC4": 4}[accessor["type"]]
        stride = view.get("byteStride", 4 * components)
        count = accessor["count"]
        with (source_dir / unquote(scene["buffers"][view["buffer"]]["uri"])).open("rb") as file:
            file.seek(view.get("byteOffset", 0) + accessor.get("byteOffset", 0))
            data = file.read((count - 1) * stride + components * 4)
        decoded[semantic.lower()] = [struct.unpack_from("<" + "f" * components, data, i * stride)
                                      for i in range(count)]
    counts = {len(v) for v in decoded.values()}
    if len(counts) != 1:
        raise ValueError("Mismatched instance counts")
    return [matrix({key: values[i] for key, values in decoded.items()}) for i in range(counts.pop())]


def make_probe(scene, source, meshes, mirrored=False):
    parents = {}
    for index, node in enumerate(scene["nodes"]):
        for child in node.get("children", []):
            if child in parents:
                raise ValueError("Source is not a single-parent hierarchy")
            parents[child] = index
    selected = [next(i for i, node in enumerate(scene["nodes"]) if node.get("mesh") == mesh) for mesh in meshes]
    keep = set(selected)
    for index in selected:
        while index in parents:
            index = parents[index]
            keep.add(index)
    ids = {"nodes": sorted(keep), "meshes": sorted(meshes)}
    mesh_map = {old: new for new, old in enumerate(ids["meshes"])}
    node_map = {old: new for new, old in enumerate(ids["nodes"])}
    nodes = [copy.deepcopy(scene["nodes"][old]) for old in ids["nodes"]]
    for old, node in zip(ids["nodes"], nodes):
        node.pop("camera", None)
        if old in selected:
            node["mesh"] = mesh_map[node["mesh"]]
        else:
            node.pop("mesh", None)
            node.get("extensions", {}).pop(EXT, None)
        # Probes select geometry/materials; no unrelated scene light references.
        node.get("extensions", {}).pop("KHR_lights_punctual", None)
        node["children"] = [node_map[c] for c in node.get("children", []) if c in keep]
    roots = [node_map[i] for i in ids["nodes"] if parents.get(i) not in keep]
    if mirrored:
        nodes.append({"name": "Z1 mirrored nonuniform parent", "scale": [-1, 2, 0.5], "children": roots})
        roots = [len(nodes) - 1]
    probe = {"asset": copy.deepcopy(scene["asset"]), "scene": 0,
             "scenes": [{"name": "ZorahFull Z1 probe", "nodes": roots}], "nodes": nodes,
             "meshes": [copy.deepcopy(scene["meshes"][i]) for i in ids["meshes"]]}
    primitives = [p for m in probe["meshes"] for p in m["primitives"]]
    accessor_ids = {i for p in primitives for i in p["attributes"].values()}
    accessor_ids.update(p["indices"] for p in primitives if "indices" in p)
    for node in nodes:
        accessor_ids.update(node.get("extensions", {}).get(EXT, {}).get("attributes", {}).values())
    ids["accessors"] = sorted(accessor_ids)
    ids["materials"] = sorted({p["material"] for p in primitives if "material" in p})
    probe["materials"] = [copy.deepcopy(scene["materials"][i]) for i in ids["materials"]]
    ids["textures"] = sorted({info["index"] for mat in probe["materials"] for info in texture_infos(mat)})
    ids["images"] = sorted({scene["textures"][i]["source"] for i in ids["textures"]})
    ids["samplers"] = sorted({scene["textures"][i]["sampler"] for i in ids["textures"] if "sampler" in scene["textures"][i]})
    ids["bufferViews"] = sorted({scene["accessors"][i]["bufferView"] for i in ids["accessors"]})
    buffer_ids = {scene["bufferViews"][i]["buffer"] for i in ids["bufferViews"]}
    buffer_ids.update(scene["bufferViews"][i]["extensions"]["EXT_meshopt_compression"]["buffer"]
                      for i in ids["bufferViews"] if "EXT_meshopt_compression" in scene["bufferViews"][i].get("extensions", {}))
    ids["buffers"] = sorted(buffer_ids)
    maps = {key: {old: new for new, old in enumerate(values)} for key, values in ids.items()}
    for key in ("accessors", "bufferViews", "buffers", "textures", "images", "samplers"):
        probe[key] = [copy.deepcopy(scene[key][i]) for i in ids[key]]
    for p in primitives:
        p["attributes"] = {key: maps["accessors"][i] for key, i in p["attributes"].items()}
        for key, target in (("indices", "accessors"), ("material", "materials")):
            if key in p:
                p[key] = maps[target][p[key]]
    for node in nodes:
        attrs = node.get("extensions", {}).get(EXT, {}).get("attributes", {})
        for key, index in attrs.items():
            attrs[key] = maps["accessors"][index]
    for accessor in probe["accessors"]:
        if "sparse" in accessor:
            raise ValueError("Sparse source accessors require explicit probe remapping")
        accessor["bufferView"] = maps["bufferViews"][accessor["bufferView"]]
    for view in probe["bufferViews"]:
        view["buffer"] = maps["buffers"][view["buffer"]]
        compressed = view.get("extensions", {}).get("EXT_meshopt_compression")
        if compressed:
            compressed["buffer"] = maps["buffers"][compressed["buffer"]]
    for entry in probe["buffers"] + probe["images"]:
        if "uri" in entry:
            entry["uri"] = (source.parent / unquote(entry["uri"])).resolve(strict=True).as_posix()
    for material in probe["materials"]:
        for info in texture_infos(material):
            info["index"] = maps["textures"][info["index"]]
    for texture in probe["textures"]:
        if texture.get("extensions"):
            raise ValueError("Texture source extensions require explicit probe remapping")
        texture["source"] = maps["images"][texture["source"]]
        if "sampler" in texture:
            texture["sampler"] = maps["samplers"][texture["sampler"]]
    used = set()
    def collect_extensions(value):
        if isinstance(value, dict):
            used.update(value.get("extensions", {}).keys())
            for child in value.values():
                collect_extensions(child)
        elif isinstance(value, list):
            for child in value:
                collect_extensions(child)
    collect_extensions(probe)
    probe["extensionsUsed"] = sorted(used)
    probe["extensionsRequired"] = [e for e in scene.get("extensionsRequired", []) if e in used]

    # Reference values use ORIGINAL accessors/nodes, independent of index remapping.
    expected = []
    def visit(old, parent_world):
        world = multiply(parent_world, matrix(scene["nodes"][old]))
        if old in selected:
            mesh = scene["nodes"][old]["mesh"]
            for instance_index, local in enumerate(instance_transforms(scene, source.parent, scene["nodes"][old])):
                for primitive_index, p in enumerate(scene["meshes"][mesh]["primitives"]):
                    expected.append({"sourceNode": old, "sourceInstance": instance_index,
                                     "sourceMesh": mesh, "sourcePrimitive": primitive_index,
                                     "sourceMaterial": p.get("material", -1),
                                     "material": maps["materials"].get(p.get("material"), -1),
                                     "worldMatrix": multiply(world, local)})
        for child in scene["nodes"][old].get("children", []):
            if child in keep:
                visit(child, world)
    for root in ids["nodes"]:
        if parents.get(root) not in keep:
            visit(root, matrix({"scale": [-1, 2, 0.5]}) if mirrored else identity())
    source_triangles = sum(scene["accessors"][p.get("indices", p["attributes"]["POSITION"])]["count"] // 3
                           for m in meshes for p in scene["meshes"][m]["primitives"])
    return probe, {"sourceIndices": ids, "sourceTriangles": source_triangles,
                   "primitiveInstances": len(expected), "expectedInstances": expected,
                   "syntheticParentScale": [-1, 2, 0.5] if mirrored else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("Asset/ZorahFull/zorah_textured_public.v1.gltf"))
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    data = source.read_bytes()
    scene = json.loads(data)
    if len(scene["meshes"]) != 2812 or len(scene["materials"]) != 1514:
        raise ValueError("Probe selection requires the audited ZorahFull v1 asset")
    args.directory.mkdir(parents=True, exist_ok=True)
    definitions = [
        ("StoneUdim", [12], "Stone, authored tangent, normal map and explicit UDIM image", False),
        ("InstancingNoTangent", [49], "GPU instancing and normal map without authored tangent", False),
        ("MaskedLeaves", [1312], "MASK, double-sided grass and alpha cutoff", False),
        ("TextureTransformBc4", [398], "Texture transform, BC4 specular and masked foliage", False),
        ("Glass", [334], "Transmission and IOR, source material 424", False),
        ("Blend", [294], "BLEND, source material 395", False),
        ("Unlit", [109], "Unlit sphere without normals, source material 1513", False),
        ("SharedGeometry", [12, 844], "Identical geometry accessors with different materials", False),
        ("MirroredInstances", [49], "Instancing under an added mirrored, nonuniform parent", True),
        ("LargestMesh", [2322], "32M-triangle mesh, isolated later-stage memory stress probe", False),
    ]
    probes = []
    for name, meshes, purpose, mirrored in definitions:
        probe, info = make_probe(scene, source, meshes, mirrored)
        destination = (args.directory / (name + ".gltf")).resolve()
        write_json(destination, probe)
        info.update(name=name, path=destination.as_posix(), purpose=purpose,
                    cookRecommended=name != "LargestMesh", materials=len(probe["materials"]),
                    images=len(probe["images"]), textures=len(probe["textures"]))
        probes.append(info)
    cfg_path = source.with_suffix(".cfg")
    cfg = cfg_path.read_text(encoding="utf-8")
    manifest = {"version": 1, "source": source.as_posix(), "sourceSha256": hashlib.sha256(data).hexdigest(),
                "cfg": {"path": cfg_path.as_posix(), "sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
                        "text": cfg, "arguments": [shlex.split(line) for line in cfg.splitlines()[1:] if line.strip()],
                        "skipMeshesApplied": False,
                        "note": "Source cfg retained for later same-camera comparison; these probes keep their selected source nodes."},
                "matrixLayout": "column-major", "payloadPolicy": "external references; no geometry/image payload copied or decoded",
                "probes": probes}
    write_json(args.directory / "probes.json", manifest)
    print(json.dumps([{key: p[key] for key in ("name", "sourceTriangles", "primitiveInstances", "materials", "images", "cookRecommended")}
                      for p in probes], indent=2))


if __name__ == "__main__":
    main()
