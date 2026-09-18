"""Read glTF metadata and KTX2 headers without loading geometry or texture payloads."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import struct


def ktx_header(path):
    with path.open("rb") as f:
        header = f.read(80)
        if len(header) != 80 or header[:12] != b"\xabKTX 20\xbb\r\n\x1a\n":
            raise ValueError(f"Invalid KTX2 header: {path}")
        h = struct.unpack("<13I2Q", header[12:])
        levels = [struct.unpack("<3Q", f.read(24)) for _ in range(max(1, h[7]))]
        size = path.stat().st_size
        if any(offset + length > size for offset, length, _ in levels):
            raise ValueError(f"Invalid KTX2 level range: {path}")
        if h[0] not in (139, 141, 146) or h[4] != 0 or h[5] != 0 or h[6] != 1:
            raise ValueError(f"Unexpected format or texture topology: {path}")
        block_bytes = 8 if h[0] == 139 else 16
        for mip, (_, _, decoded_bytes) in enumerate(levels):
            expected = ((max(1, h[2] >> mip) + 3) // 4) * ((max(1, h[3] >> mip) + 3) // 4) * block_bytes
            if decoded_bytes != expected:
                raise ValueError(f"Unexpected BC mip byte count: {path}, mip {mip}")
        f.seek(h[11])
        kvd = f.read(h[12])
        kv = {}
        offset = 0
        while offset < len(kvd):
            length = struct.unpack_from("<I", kvd, offset)[0]
            offset += 4
            item = kvd[offset:offset + length]
            key, _, value = item.partition(b"\0")
            kv[key.decode("utf-8", errors="replace")] = value.rstrip(b"\0").decode("utf-8", errors="replace")
            offset += (length + 3) & ~3
        return {"width": h[2], "height": h[3], "vkFormat": h[0], "levels": len(levels),
                "supercompression": h[8], "diskBytes": size,
                "gpuMipBytes": [n for _, _, n in levels], "metadata": kv}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("Asset/ZorahFull/zorah_textured_public.v1.gltf"))
    parser.add_argument("--output", type=Path, default=Path("Documentation/ZorahFullAssetAudit.json"))
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    raw = source.read_bytes()
    scene = json.loads(raw)
    accessors, meshes, nodes = (scene[k] for k in ("accessors", "meshes", "nodes"))
    attributes, modes = Counter(), Counter()
    mesh_triangles, mesh_vertices, largest = [], [], []
    geometry_signatures, geometry_material_signatures = {}, {}
    for index, mesh in enumerate(meshes):
        triangles = vertices = 0
        for primitive in mesh["primitives"]:
            attributes.update(primitive["attributes"].keys())
            mode = primitive.get("mode", 4)
            modes[str(mode)] += 1
            vertex_count = accessors[primitive["attributes"]["POSITION"]]["count"]
            vertices += vertex_count
            if mode == 4:
                primitive_triangles = accessors[primitive["indices"]]["count"] // 3 if "indices" in primitive else vertex_count // 3
                triangles += primitive_triangles
                signature = (tuple(sorted(primitive["attributes"].items())), primitive.get("indices"), mode)
                geometry_signatures.setdefault(signature, primitive_triangles)
                geometry_material_signatures.setdefault((signature, primitive.get("material")), primitive_triangles)
        mesh_triangles.append(triangles)
        mesh_vertices.append(vertices)
        largest.append({"mesh": index, "name": mesh.get("name"), "triangles": triangles, "vertices": vertices})
    instances = primitive_instances = triangles_instanced = extension_nodes = 0
    visited = set()
    stack = list(scene["scenes"][scene.get("scene", 0)]["nodes"])
    skip = re.compile(r"SM_Courtyard_Kite_Frame_A01_01.*|SM_Kite_VAT.*")
    skipped_instances = skipped_triangles = 0
    while stack:
        index = stack.pop()
        if index in visited:
            raise ValueError("Repeated node or cycle in selected scene")
        visited.add(index)
        node = nodes[index]
        stack.extend(node.get("children", []))
        if "mesh" not in node:
            continue
        mesh = node["mesh"]
        ext = node.get("extensions", {}).get("EXT_mesh_gpu_instancing")
        count = 1
        if ext:
            extension_nodes += 1
            counts = {accessors[a]["count"] for a in ext["attributes"].values()}
            if len(counts) != 1:
                raise ValueError("Inconsistent instance accessor counts")
            count = counts.pop()
        instances += count
        primitive_instances += count * len(meshes[mesh]["primitives"])
        triangles_instanced += count * mesh_triangles[mesh]
        if skip.fullmatch(meshes[mesh].get("name", "")):
            skipped_instances += count
            skipped_triangles += count * mesh_triangles[mesh]
    missing = []
    buffer_paths = {source.parent / b["uri"] for b in scene["buffers"] if b.get("uri")}
    image_paths = {source.parent / i["uri"] for i in scene["images"] if i.get("uri")}
    for path in sorted(buffer_paths | image_paths):
        if not path.is_file():
            missing.append(str(path))
    textures = {str(p.relative_to(source.parent)): ktx_header(p) for p in sorted(image_paths) if p.is_file()}
    caps = {}
    for cap in (128, 256, 512, 1024, 2048, 4096):
        total = 0
        for t in textures.values():
            first = 0
            while first + 1 < t["levels"] and max(t["width"] >> first, t["height"] >> first) > cap:
                first += 1
            total += sum(t["gpuMipBytes"][first:])
        caps[str(cap)] = total
    materials = scene["materials"]
    material_extensions = Counter(e for m in materials for e in m.get("extensions", {}))
    texture_slots, texcoords, transforms = Counter(), Counter(), []
    def inspect_material(value, material, key=""):
        if isinstance(value, dict):
            if key.endswith("Texture") and "index" in value:
                texture_slots[key] += 1
                transform = value.get("extensions", {}).get("KHR_texture_transform", {})
                texcoords[str(transform.get("texCoord", value.get("texCoord", 0)))] += 1
                if transform:
                    transforms.append({"material": material, "slot": key, **transform})
            for k, child in value.items():
                inspect_material(child, material, k)
    for index, material in enumerate(materials):
        inspect_material(material, index)
    all_texture_files = list((source.parent / "textures").glob("*.ktx2"))
    report = {
        "source": str(source), "gltfSha256": hashlib.sha256(raw).hexdigest(),
        "scope": "Metadata, file sizes, KTX2 headers/level indices only; no geometry decode, image decode or render.",
        "counts": {k: len(scene.get(k, [])) for k in ("meshes", "nodes", "materials", "images", "textures", "buffers", "skins", "animations")},
        "geometry": {"primitiveCount": sum(modes.values()), "primitiveAttributes": dict(attributes), "primitiveModes": dict(modes),
                     "meshPrimitiveTriangles": sum(mesh_triangles), "primitiveVertices": sum(mesh_vertices),
                     "exactAccessorGeometrySignatures": len(geometry_signatures),
                     "exactAccessorGeometryTriangles": sum(geometry_signatures.values()),
                     "exactAccessorGeometryMaterialSignatures": len(geometry_material_signatures),
                     "exactAccessorGeometryMaterialTriangles": sum(geometry_material_signatures.values()),
                     "selectedSceneNodes": len(visited), "nodesWithChildren": sum(bool(n.get("children")) for n in nodes),
                     "gpuInstancingNodes": extension_nodes, "meshInstances": instances, "primitiveInstances": primitive_instances,
                     "instancedTriangles": triangles_instanced, "cfgSkippedMeshInstances": skipped_instances,
                     "cfgSkippedTriangles": skipped_triangles, "largestMeshes": sorted(largest, key=lambda m: m["triangles"], reverse=True)[:8]},
        "extensionsRequired": scene.get("extensionsRequired"), "extensionsUsed": scene.get("extensionsUsed"),
        "materials": {"alphaModes": dict(Counter(m.get("alphaMode", "OPAQUE") for m in materials)),
                      "doubleSided": sum(bool(m.get("doubleSided")) for m in materials), "extensions": dict(material_extensions),
                      "textureSlots": dict(texture_slots), "texcoordSets": dict(texcoords), "textureTransformCount": len(transforms),
                      "textureTransformExamples": transforms[:8]},
        "storage": {"referencedBufferFiles": len(buffer_paths), "referencedBufferBytes": sum(p.stat().st_size for p in buffer_paths if p.is_file()),
                    "referencedTextureFiles": len(textures), "allTextureFiles": len(all_texture_files),
                    "referencedTextureDiskBytes": sum(t["diskBytes"] for t in textures.values()),
                    "textureGpuFullMipBytes": sum(sum(t["gpuMipBytes"]) for t in textures.values()),
                    "textureGpuBytesWithMaxDimension": caps,
                    "textureFormats": dict(Counter(str(t["vkFormat"]) for t in textures.values())),
                    "supercompression": dict(Counter(str(t["supercompression"]) for t in textures.values())),
                    "dimensions": dict(Counter(f'{t["width"]}x{t["height"]}' for t in textures.values())),
                    "ktxSwizzles": dict(Counter(t["metadata"].get("KTXswizzle", "<absent>") for t in textures.values())),
                    "udimNamedReferencedTextures": sum(bool(re.search(r"\.1\d{3}\.ktx2$", p)) for p in textures),
                    "mimeTypes": dict(Counter(i.get("mimeType", "<absent>") for i in scene["images"])),
                    "missingFiles": missing},
        "textureExamples": dict(list(textures.items())[:3]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "geometry": report["geometry"], "materials": report["materials"], "storage": report["storage"]}, indent=2))


if __name__ == "__main__":
    main()
