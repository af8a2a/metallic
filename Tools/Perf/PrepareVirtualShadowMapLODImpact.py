"""Prepare resident Bunny shadow LOD GPU replays without changing source assets."""
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import struct

from VirtualShadowMapAcceptance import ROOT, digest, read, save


FIXTURE_ROOT = ROOT / "build/vsm-shadow-hierarchy/Run2/virtual_shadow_map_shadow_lod/VirtualShadowMapLOD"
SCENE_PATH = FIXTURE_ROOT / "Fixture/scene.gltf"
STREAM_PATH = FIXTURE_ROOT / "Bunny.meshstream.bin"
GRAPH_PATH = ROOT / "Pipelines/Samples/realtime_lighting_vsm.metallic_graph.json"
CAMERA = dict(eye=[1.5, 0, 8], center=[1.5, 0, 0], up=[0, 1, 0],
              projection="orthographic", orthoHeight=8, fovDegrees=60, znear=.01, zfar=20)


def inspect_assets():
    """Read actual glTF counts and the version 9 stream metadata, without loading a GPU."""
    scene = read(SCENE_PATH)
    files = [SCENE_PATH, STREAM_PATH]
    for entry in scene.get("buffers", []) + scene.get("images", []):
        uri = entry.get("uri", "")
        if uri and not uri.startswith("data:"):
            files.append((SCENE_PATH.parent / uri).resolve())
    assets = dict(files=[dict(path=path.as_posix(), sha256=digest(path))
                         for path in sorted(set(files))])
    primitives = []
    for mesh_index, mesh in enumerate(scene["meshes"]):
        for primitive_index, primitive in enumerate(mesh["primitives"]):
            if primitive.get("mode", 4) != 4:
                raise ValueError("The shadow LOD fixture must contain triangle primitives")
            positions = scene["accessors"][primitive["attributes"]["POSITION"]]
            count = scene["accessors"][primitive["indices"]]["count"] if "indices" in primitive else positions["count"]
            if count % 3:
                raise ValueError("Incomplete source triangle in the shadow LOD fixture")
            primitives.append(dict(mesh=mesh_index, primitive=primitive_index,
                                   vertexCount=positions["count"], triangleCount=count // 3))

    size = STREAM_PATH.stat().st_size
    with STREAM_PATH.open("rb") as stream:
        header = stream.read(216)
        if len(header) != 216 or header[:8] != b"MTLMSTRM":
            raise ValueError("Unsupported meshstream header")
        version, endian = struct.unpack_from("<II", header, 8)
        if version != 9 or endian != 0x01020304 or struct.unpack_from("<Q", header, 16)[0] != size:
            raise ValueError("Expected a complete little-endian version 9 meshstream asset")
        primitive_count, instance_count, geometry_count = struct.unpack_from("<III", header, 56)
        lod_count, group_count, _, node_count, page_count = struct.unpack_from("<IIIII", header, 72)
        lod_offset, group_offset = struct.unpack_from("<QQ", header, 144)
        if lod_offset + lod_count * 28 > size or group_offset + group_count * 44 > size:
            raise ValueError("Meshstream LOD metadata lies outside the file")
        stream.seek(lod_offset)
        levels = [struct.unpack("<IIIIIff", stream.read(28)) for _ in range(lod_count)]
        stream.seek(group_offset)
        groups = [struct.unpack("<IIIIfffffII", stream.read(44)) for _ in range(group_count)]
    group_levels = Counter(group[2] for group in groups)
    terminal_groups = sum(bool(group[10] & 1) for group in groups)
    if len(group_levels) < 2 or group_count <= terminal_groups:
        raise ValueError("The GPU LOD comparison requires a genuine multi-level stream asset")
    metrics = dict(
        sourcePrimitives=primitives,
        authoredPrimitiveTriangleCount=sum(primitive["triangleCount"] for primitive in primitives),
        stream=dict(version=version, byteSize=size, primitiveCount=primitive_count,
                    instanceCount=instance_count, geometryCount=geometry_count, pageCount=page_count,
                    groupCount=group_count, terminalGroupCount=terminal_groups, nodeCount=node_count,
                    lodLevelRecordCount=lod_count, maximumLOD=max(group_levels),
                    groupCountByLOD={str(level): count for level, count in sorted(group_levels.items())},
                    finestClusterCount=sum(level[4] for level in levels if level[1] == 0),
                    levels=[dict(primitive=level[0], level=level[1], pageCount=level[3], clusterCount=level[4])
                            for level in levels]))
    return assets, metrics


def base_graph():
    graph = read(GRAPH_PATH)
    removed = {"DLSSSR", "DLSSNR"}
    graph["nodes"] = [node for node in graph["nodes"] if node["name"] not in removed]
    graph["edges"] = [edge for edge in graph["edges"] if edge["src"].split(".")[0] not in removed
                      and edge["dst"].split(".")[0] not in removed]
    next_id = max(edge["id"] for edge in graph["edges"]) + 1
    graph["edges"] += [dict(id=next_id, src="Deferred.color", dst="AutoExposure.source"),
                       dict(id=next_id + 1, src="AutoExposure.color", dst="FinalBlit.source")]
    graph["name"] = "Resident Bunny Shadow LOD GPU Impact"
    graph["view"] = dict(camera=deepcopy(CAMERA), temporalJitter=False)
    for node in graph["nodes"]:
        properties = node["properties"]
        if node["name"] in ("VBuffer", "Deferred"):
            properties.update(path=SCENE_PATH.as_posix(), sceneBinding="world",
                              streamAssetPath=STREAM_PATH.as_posix(), autoBuildStreamAsset=False,
                              enableClas=False, enableClusterRtx=False,
                              maxResidentPages=128, maxLockedFallbackPages=128,
                              maxPageUploadsPerFrame=128, pageLoadConcurrency=0)
        if node["name"] == "VBuffer":
            properties.update(initialLoad=True, autoLod=False, lodLevel=0, hybridRaster=False,
                              instanceFrustumCull=False, instanceHzbCull=False,
                              meshletFrustumCull=False, meshletNormalConeCull=False)
        if node["name"] == "Deferred":
            properties["exportUpscalerGuides"] = False
        if node["name"] == "Shadows":
            properties.update(enabled=True, pageSize=128, virtualPagesPerAxis=16, clipmapCount=4,
                              maxWorkItems=1048576, firstClipmapExtent=4., physicalPageBudget=256,
                              shadowBias=.002, lightCapacity=2, filter="hard", cacheLayering=False,
                              onePassProjection=True, smrtTemporalEnabled=False,
                              shadowLODErrorTexels=1., benchmarkGPUProfile=True,
                              benchmarkCPUProfile=False, benchmarkBuildSubstages=False,
                              compactOutputs=True, debugOutput=False)
    return graph


def prepare(args):
    assets, metrics = inspect_assets()
    graph_template = base_graph()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifests = []
    for mode in ("Diagnostic", "Timing"):
        diagnostic = mode == "Diagnostic"
        for workload in ("Redraw", "Warm"):
            for arm in ("Off", "Flat", "Hierarchy"):
                name = f"{mode}-{workload}-{arm}"
                graph = deepcopy(graph_template)
                shadows = next(node["properties"] for node in graph["nodes"] if node["name"] == "Shadows")
                shadows.update(cacheEnabled=workload == "Warm", benchmarkGPUWork=diagnostic,
                               shadowLOD=arm != "Off", shadowLODHierarchy=arm == "Hierarchy")
                save(output / f"{name}.json", graph)
                # A fully loaded world scene retains resident shadow geometry. Its VBuffer
                # uses the same prebuilt stream asset in all arms at fixed finest LOD.
                case = dict(id=f"BunnyResident{workload}", graph=f"{name}.json",
                            scene=SCENE_PATH.as_posix(), metadataOnly=False,
                            enableStreamline=False, synchronizationValidation=diagnostic,
                            requireNoRayCapabilities=True, temporalJitter=False,
                            width=768, height=512, output="FinalBlit.color", comparisonOutput="Deferred.color",
                            camera=deepcopy(CAMERA), requireShadow=True,
                            lighting=dict(sun=dict(direction=[2, 0, -1], illuminance=3., angularRadius=.00465)),
                            environment=dict(enabled=False), exposureEV100=0,
                            captureFrames=[32, 47] if diagnostic else [], comparisonFrames=[],
                            comparisonStartFrame=16, freezeStreamingFrame=16,
                            events=[dict(frame=16, cameraCut=True)])
                manifest_name = f"{name}Cases.json"
                save(output / manifest_name,
                     dict(version=1, mode=mode.lower(), frames=48 if diagnostic else 96,
                          warmupFrames=32, cases=[case]))
                manifests.append(manifest_name)
    save(output / "Assets.json", assets)
    save(output / "Preparation.json", dict(
        protocol="virtual-shadow-map-lod-impact-preparation-v1", fixtureMetrics=metrics,
        templateGraph=dict(path=GRAPH_PATH.as_posix(), sha256=digest(GRAPH_PATH)),
        scene=SCENE_PATH.as_posix(), streamAsset=STREAM_PATH.as_posix(),
        shadowBase="Resident full geometry: metadataOnly=false, sceneBinding=world",
        rasterGeometry="Streamed prebuilt geometry with autoLod=false, lodLevel=0; frozen at frame 16",
        workloadScope="Controlled multi-LOD Bunny plus receiver fixture; not a production-scene performance claim",
        redrawCondition="cacheEnabled=false every frame; warmup still excludes shader and initial residency setup",
        warmCondition="cacheEnabled=true and fixed camera; measures reused shadow pages",
        sourceAssetsModified=False, gpuExecutionPerformed=False))
    save(output / "Catalog.json", dict(caseManifests=manifests,
                                      execution="One case manifest per independent process"))
    print(f"Prepared {len(manifests)} cases in {output}")
    print(f"Source triangles: {metrics['authoredPrimitiveTriangleCount']}; "
          f"stream groups: {metrics['stream']['groupCount']}; "
          f"maximum LOD: {metrics['stream']['maximumLOD']}; "
          f"finest clusters: {metrics['stream']['finestClusterCount']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing directories are rejected")
    prepare(parser.parse_args())
