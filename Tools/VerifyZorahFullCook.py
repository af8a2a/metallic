"""Verify Full Zorah cook coverage against glTF metadata without reading payloads.

Checks exact accessor/material geometry deduplication, per-geometry LOD0 triangle
counts, hierarchical scene and GPU-instance counts, terminal roots, and completed
file evidence. Binary payload/topology validation is required from the cooker;
this script does not independently decode geometry or instance transforms.
Missing evidence produces an incomplete report and a nonzero exit status.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


class MissingEvidence(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, f"Invalid {label}: {value!r}")
    return value


def indexed(values, index, label):
    integer(index, label)
    require(index < len(values), f"Out-of-range {label}: {index}")
    return values[index]


def geometry_key(primitive, share_materials=False):
    # Matches gltfGeometryKey, including defaults used by tinygltf. All
    # attributes participate, even those not currently encoded in meshlets.
    key = {"mode": primitive.get("mode", 4), "indices": primitive.get("indices", -1),
           "attributes": primitive.get("attributes", {}), "targets": primitive.get("targets", [])}
    if not share_materials:
        key["material"] = primitive.get("material", -1)
    return json.dumps(key, sort_keys=True)


def source_coverage(root, share_materials=False):
    accessors, meshes, nodes = (root[key] for key in ("accessors", "meshes", "nodes"))
    geometries, keys, mesh_geometry_ids = [], {}, []
    source_primitive = 0
    for mesh_index, mesh in enumerate(meshes):
        geometry_ids = []
        for primitive_index, primitive in enumerate(mesh["primitives"]):
            label = f"mesh {mesh_index}, primitive {primitive_index}"
            require(primitive.get("mode", 4) == 4, f"{label}: expected triangle geometry")
            require(not primitive.get("targets"), f"{label}: morph targets are not supported by the static cooker")
            attributes = primitive["attributes"]
            for semantic, accessor_id in attributes.items():
                indexed(accessors, accessor_id, f"{label} {semantic} accessor")
            position = indexed(accessors, attributes["POSITION"], f"{label} POSITION")
            vertices = integer(position["count"], f"{label} vertex count", 1)
            require(position["type"] == "VEC3", f"{label}: POSITION must be VEC3")
            if primitive.get("indices", -1) >= 0:
                indices = indexed(accessors, primitive["indices"], f"{label} indices")
                require(indices["type"] == "SCALAR", f"{label}: indices must be SCALAR")
                index_count = integer(indices["count"], f"{label} index count", 1)
            else:
                index_count = vertices
            require(index_count % 3 == 0, f"{label}: incomplete triangle")
            key = geometry_key(primitive, share_materials)
            if key not in keys:
                keys[key] = len(geometries)
                geometries.append({"primitive": len(geometries), "sourcePrimitive": source_primitive,
                                   "mesh": mesh_index, "meshPrimitive": primitive_index,
                                   "sourceAliases": [], "vertices": vertices,
                                   "lod0Triangles": index_count // 3,
                                   "material": max(primitive.get("material", -1), 0), "instances": 0,
                                   "instanceMaterials": {}})
            geometry_id = keys[key]
            geometries[geometry_id]["sourceAliases"].append(source_primitive)
            geometry_ids.append(geometry_id)
            source_primitive += 1
        mesh_geometry_ids.append(geometry_ids)

    def instance_count(node, node_id):
        extension = node.get("extensions", {}).get("EXT_mesh_gpu_instancing")
        if extension is None:
            return 1
        require("mesh" in node and "skin" not in node and "weights" not in node,
                f"Node {node_id}: GPU instancing requires a static mesh")
        attributes = extension["attributes"]
        require(isinstance(attributes, dict) and attributes, f"Node {node_id}: empty instance attributes")
        counts = set()
        for semantic, accessor_id in attributes.items():
            require(semantic in ("TRANSLATION", "ROTATION", "SCALE"),
                    f"Node {node_id}: unsupported instance attribute {semantic}")
            accessor = indexed(accessors, accessor_id, f"node {node_id} instance accessor")
            require(accessor["type"] == ("VEC4" if semantic == "ROTATION" else "VEC3"),
                    f"Node {node_id}: invalid {semantic} accessor type")
            counts.add(integer(accessor["count"], f"node {node_id} instance count", 1))
        require(len(counts) == 1, f"Node {node_id}: inconsistent instance accessor counts")
        return counts.pop()

    # The builder expands GPU instances into mesh-only children and appends the
    # original children once. Instance multiplicity never propagates to them.
    # Count every traversal occurrence (including shared nodes), detect cycles.
    scene_id = root.get("scene", 0)
    if scene_id == -1:
        scene_id = 0
    scene = indexed(root["scenes"], scene_id, "default scene")
    stack = [(node, False) for node in reversed(scene.get("nodes", []))]
    active, visited = set(), set()
    node_visits = extension_visits = mesh_instances = 0
    while stack:
        node_id, leaving = stack.pop()
        if leaving:
            active.remove(node_id)
            continue
        node = indexed(nodes, node_id, "scene node")
        require(node_id not in active, f"Cycle in selected scene at node {node_id}")
        active.add(node_id)
        visited.add(node_id)
        node_visits += 1
        stack.append((node_id, True))
        count = instance_count(node, node_id)
        if "EXT_mesh_gpu_instancing" in node.get("extensions", {}):
            extension_visits += 1
        if "mesh" in node:
            geometry_ids = indexed(mesh_geometry_ids, node["mesh"], f"node {node_id} mesh")
            mesh_instances += count
            source_mesh = meshes[node["mesh"]]
            for primitive_index, geometry_id in enumerate(geometry_ids):
                geometries[geometry_id]["instances"] += count
                material = str(max(source_mesh["primitives"][primitive_index].get("material", -1), 0))
                histogram = geometries[geometry_id]["instanceMaterials"]
                histogram[material] = histogram.get(material, 0) + count
        stack.extend((child, False) for child in reversed(node.get("children", [])))

    return {"sourcePrimitiveCount": source_primitive, "geometryCount": len(geometries),
            "materialIndependentGeometry": share_materials,
            "instanceCount": sum(g["instances"] for g in geometries),
            "sourceTriangles": sum(g["lod0Triangles"] for g in geometries),
            "instancedSourceTriangles": sum(g["lod0Triangles"] * g["instances"] for g in geometries),
            "selectedScene": scene_id, "selectedSceneNodes": len(visited), "nodeVisits": node_visits,
            "gpuInstancingNodeVisits": extension_visits, "meshInstances": mesh_instances,
            "geometryMappings": geometries}


def verify_metadata(coverage, report):
    expected = coverage["geometryMappings"]
    count = len(expected)
    require(report["primitives"] == count, "Cooked primitive count differs from source deduplication")
    geometries = report["geometries"]
    require(len(geometries) == count, "Geometry table coverage mismatch")
    require(Counter(g["primitive"] for g in geometries) == Counter(range(count)),
            "Duplicate, missing or invalid cooked primitive IDs")
    by_id = {g["primitive"]: g for g in geometries}
    for primitive_id, geometry in by_id.items():
        source = expected[primitive_id]
        for field in ("sourcePrimitive", "instances"):
            require(geometry[field] == source[field], f"Primitive {primitive_id}: {field} mismatch")
        if coverage["materialIndependentGeometry"]:
            require(geometry["instanceMaterials"] == source["instanceMaterials"],
                    f"Primitive {primitive_id}: instance material bindings mismatch")
        require(0 < geometry["terminalGroups"] <= geometry["groups"],
                f"Primitive {primitive_id}: empty/invalid terminal groups")
        require(geometry["levels"] > 0, f"Primitive {primitive_id}: no LOD levels")
    require(report["instances"] == coverage["instanceCount"], "Total instance count mismatch")
    require(report["levels"]["0"]["triangles"] == coverage["sourceTriangles"],
            "Total LOD0 triangles differ from deduplicated source")
    require(sum(g["groups"] for g in geometries) == report["groups"], "Group total mismatch")
    pages = integer(report["pages"], "page count", 1)
    terminal_pages = report["terminalPages"]
    require(terminal_pages and len(set(terminal_pages)) == len(terminal_pages) == report["terminalPageCount"],
            "Empty or duplicate terminal page set")
    require(all(type(page) is int and 0 <= page < pages for page in terminal_pages),
            "Terminal page index out of range")
    require(sum(g["terminalGroups"] for g in geometries) == len(terminal_pages),
            "Terminal group/page total mismatch")

    audit = report["rootCutAudit"]
    roots = audit["primitivesByRootBytes"]
    if {r["primitive"] for r in roots} != set(range(count)):
        raise MissingEvidence("Per-geometry root audit/LOD0 counts are incomplete (including uninstantiated geometries)")
    require(len(roots) == count, "Duplicate primitive in root cut audit")
    for row in roots:
        primitive_id = row["primitive"]
        source = expected[primitive_id]
        for field in ("sourcePrimitive", "instances", "lod0Triangles", "material"):
            require(row[field] == source[field], f"Root audit primitive {primitive_id}: {field} mismatch")
        require(row["rootPages"] == by_id[primitive_id]["terminalGroups"],
                f"Primitive {primitive_id}: root page count mismatch")
        for field in ("rootPages", "rootClusters", "rootTriangles", "rootBytes"):
            require(row[field] > 0, f"Primitive {primitive_id}: empty {field}")
    require(sum(r["rootPages"] for r in roots) == audit["rootPages"] == len(terminal_pages),
            "Root audit page total mismatch")
    require(sum(r["rootClusters"] for r in roots) == audit["rootClusters"], "Root cluster total mismatch")
    require(sum(r["rootTriangles"] for r in roots) == audit["rootTriangles"], "Root triangle total mismatch")
    require(sum(r["rootBytes"] for r in roots) == audit["rootBytes"], "Root byte total mismatch")
    require(sum(r["rootPages"] * r["instances"] for r in roots) ==
            audit["rootInstanceGroups"] == report["terminalInstanceGroups"], "Instanced root group total mismatch")
    require(sum(r["rootClusters"] * r["instances"] for r in roots) ==
            audit["rootInstanceClusters"] == report["terminalInstanceClusters"], "Instanced root cluster total mismatch")


def reference_size_comparison(actual_bytes, reference_bytes):
    require(reference_bytes > 0, "Reference cache is empty")
    minimum = (reference_bytes * 95 + 99) // 100
    maximum = reference_bytes * 105 // 100
    return {"referenceBytes": reference_bytes, "actualBytes": actual_bytes,
            "minimumBytes": minimum, "maximumBytes": maximum,
            "differencePercent": (actual_bytes / reference_bytes - 1) * 100,
            "withinFivePercent": minimum <= actual_bytes <= maximum}


def verify(source, manifest, expected_geometries=5408, expected_instances=43068, cook_revision=4,
           reference_cache=None):
    source = Path(source).resolve(strict=True)
    manifest = Path(manifest).resolve(strict=True)
    source_bytes, manifest_bytes = source.read_bytes(), manifest.read_bytes()
    root, report = json.loads(source_bytes), json.loads(manifest_bytes)
    coverage = source_coverage(root, share_materials=cook_revision >= 4)
    result = {"status": "pending", "source": str(source), "manifest": str(manifest),
              "sourceGltfSha256": hashlib.sha256(source_bytes).hexdigest(),
              "manifestSha256": hashlib.sha256(manifest_bytes).hexdigest(),
              "scope": "glTF metadata dedup representatives, LOD0 counts and aggregate per-geometry instances; "
                       "no payload, attribute, transform-value or image decoding",
              "expectedCookRevision": cook_revision, **coverage,
              "checks": {}, "failures": [], "missingEvidence": []}

    def check(name, operation):
        try:
            operation()
            result["checks"][name] = "passed"
        except (MissingEvidence, KeyError) as error:
            result["checks"][name] = "incomplete"
            message = f"Missing required report field {error}" if isinstance(error, KeyError) else str(error)
            result["missingEvidence"].append(f"{name}: {message}")
        except (ValueError, TypeError, OSError) as error:
            result["checks"][name] = "failed"
            result["failures"].append(f"{name}: {error}")

    def check_source():
        if not report.get("source"):
            raise MissingEvidence("Cook report does not bind a source path")
        require(Path(report["source"]).resolve() == source, "Manifest source path mismatch")

    def check_payloads():
        if report.get("payloadValidation") != "all-pages":
            raise MissingEvidence("Cook report does not attest all-page payload validation")

    def check_file():
        asset = Path(report["asset"]).resolve(strict=True)
        require(asset.is_file(), "Cook output is not a regular file")
        require(asset.stat().st_size == integer(report["fileBytes"], "fileBytes", 1), "Cook output file size mismatch")
        require(not Path(str(asset) + ".partial").exists(), "Partial checkpoint still exists")
        result["asset"] = str(asset)
        result["fileBytes"] = asset.stat().st_size
        result["meshoptTemporaryCacheRemoved"] = not Path(str(asset) + ".meshopt-cache").exists()

    check("expectedGeometryCount", lambda: require(coverage["geometryCount"] == expected_geometries,
                                                  f"Expected {expected_geometries} unique geometries; found {coverage['geometryCount']}"))
    check("expectedInstanceCount", lambda: require(coverage["instanceCount"] == expected_instances,
                                                  f"Expected {expected_instances} instances; found {coverage['instanceCount']}"))
    check("completedCook", lambda: require(report["status"] == "complete", "Cook report is not complete"))
    check("cookRevision", lambda: require(report["cookRevision"] == cook_revision, "Cook revision mismatch"))
    check("sourceBinding", check_source)
    check("allPagePayloadValidation", check_payloads)
    check("geometryAndInstanceCoverage", lambda: verify_metadata(coverage, report))
    check("completedFile", check_file)
    if reference_cache is not None:
        def check_reference_size():
            reference = Path(reference_cache).resolve(strict=True)
            comparison = reference_size_comparison(Path(report["asset"]).stat().st_size, reference.stat().st_size)
            result["referenceCacheSize"] = {"path": str(reference), **comparison}
            require(comparison["withinFivePercent"], "Cache size differs from reference by more than five percent")
        check("referenceCacheSizeWithinFivePercent", check_reference_size)
    result["status"] = "failed" if result["failures"] else "incomplete" if result["missingEvidence"] else "verified"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("Asset/ZorahFull/zorah_textured_public.v1.gltf"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-geometries", type=int, default=5408)
    parser.add_argument("--expected-instances", type=int, default=43068)
    parser.add_argument("--cook-revision", type=int, default=4)
    parser.add_argument("--reference-cache", type=Path, help="Require file size within plus or minus five percent")
    args = parser.parse_args()
    try:
        result = verify(args.source, args.manifest, args.expected_geometries, args.expected_instances,
                        args.cook_revision, args.reference_cache)
    except (ValueError, KeyError, TypeError, OSError) as error:
        result = {"status": "failed", "source": str(args.source), "manifest": str(args.manifest),
                  "failures": [str(error)], "missingEvidence": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{result['status']}: geometries={result.get('geometryCount', 'unknown')}, "
          f"instances={result.get('instanceCount', 'unknown')}; report={args.output}")
    for message in result["failures"] + result["missingEvidence"]:
        print(message, file=sys.stderr)
    return 0 if result["status"] == "verified" else 2 if result["status"] == "incomplete" else 1


if __name__ == "__main__":
    sys.exit(main())
