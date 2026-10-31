"""Independent authored-Bunny CPU ray oracle for sealed Smart-bias diagnostics.

No GPU work is launched. Camera rays first identify the nearest authored
triangle, then directional visibility is traced from that CPU hit. GPU depth is
used only to verify agreement with the receiver, never to construct the oracle.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

import numba
from numba import njit
import numpy as np

import VirtualShadowMapAcceptance as acceptance
import VirtualShadowMapLODImpact as impact
import VirtualShadowMapLODSmartBias as experiment
import VirtualShadowMapLODSmartBiasQuality as quality


EPSILONS = (0., 1e-6, 1e-5, 1e-4)
TRIANGLE_EPSILON = 1e-7
DEPTH_TOLERANCE = 2e-5


def authored_triangles(scene_path, discarded_position_bits=0):
    document = acceptance.read(scene_path)
    buffers = [(scene_path.parent / entry["uri"]).read_bytes() for entry in document["buffers"]]

    def accessor(index):
        value = document["accessors"][index]
        if "sparse" in value or value.get("normalized", False):
            raise ValueError("This fixed fixture oracle expects nonsparse, unnormalized accessors")
        view = document["bufferViews"][value["bufferView"]]
        dtype = np.dtype({5121: "u1", 5123: "<u2", 5125: "<u4", 5126: "<f4"}[value["componentType"]])
        components = {"SCALAR": 1, "VEC3": 3}[value["type"]]
        stride = view.get("byteStride", components * dtype.itemsize)
        start = view.get("byteOffset", 0) + value.get("byteOffset", 0)
        return np.ndarray((value["count"], components), dtype=dtype,
                          buffer=buffers[view["buffer"]], offset=start,
                          strides=(stride, dtype.itemsize)).copy()

    def local_matrix(node):
        if "matrix" in node:
            return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
        x, y, z, w = node.get("rotation", [0, 0, 0, 1])
        rotation = np.array([[1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
                             [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
                             [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]])
        result = np.eye(4)
        result[:3, :3] = rotation @ np.diag(node.get("scale", [1, 1, 1]))
        result[:3, 3] = node.get("translation", [0, 0, 0])
        return result

    triangles, owners, instances = [], [], []

    def visit(index, parent):
        node = document["nodes"][index]
        world = parent @ local_matrix(node)
        if "skin" in node:
            raise ValueError("Skinned geometry is outside the fixed Bunny oracle")
        if "mesh" in node:
            mesh = node["mesh"]
            for primitive_index, primitive in enumerate(document["meshes"][mesh]["primitives"]):
                if primitive.get("mode", 4) != 4 or "targets" in primitive:
                    raise ValueError("Expected static authored triangles")
                vertices = accessor(primitive["attributes"]["POSITION"])
                if discarded_position_bits:
                    # Independent diagnostic of the fixed reference codec's
                    # position quantization, not the authored shadow oracle.
                    vertices.view(np.uint32)[:] &= np.uint32(0xffffffff << discarded_position_bits & 0xffffffff)
                vertices = vertices.astype(np.float64)
                vertices = vertices @ world[:3, :3].T + world[:3, 3]
                indices = accessor(primitive["indices"]).ravel() if "indices" in primitive else np.arange(len(vertices))
                if len(indices) % 3:
                    raise ValueError("Incomplete authored triangle")
                transformed = vertices[indices.reshape(-1, 3)]
                triangles.append(transformed)
                owners.append(np.full(len(transformed), mesh, dtype=np.int32))
                instances.append(dict(node=index, mesh=mesh, primitive=primitive_index,
                                      triangles=len(transformed), worldMatrix=world.tolist()))
        for child in node.get("children", []):
            visit(child, world)

    for root in document["scenes"][document.get("scene", 0)]["nodes"]:
        visit(root, np.eye(4))
    vertices = np.ascontiguousarray(np.concatenate(triangles))
    return vertices, np.concatenate(owners), instances


def build_bvh(triangles):
    lows, highs = triangles.min(axis=1), triangles.max(axis=1)
    centroids = triangles.mean(axis=1)
    order = np.arange(len(triangles), dtype=np.int32)
    bounds, nodes, max_depth = [], [], 0

    def build(first, count, depth):
        nonlocal max_depth
        max_depth = max(max_depth, depth)
        selected = order[first:first + count]
        low, high = lows[selected].min(axis=0), highs[selected].max(axis=0)
        index = len(nodes)
        bounds.append(np.concatenate((low, high)))
        nodes.append([first, count, -1, -1])
        if count > 8:
            axis = int(np.argmax(high - low))
            middle = count // 2
            order[first:first + count] = selected[np.argpartition(centroids[selected, axis], middle)]
            left = build(first, middle, depth + 1)
            right = build(first + middle, count - middle, depth + 1)
            nodes[index] = [first, 0, left, right]
        return index

    build(0, len(triangles), 0)
    if max_depth >= 63:
        raise ValueError("BVH exceeds the independently bounded traversal stack")
    return (np.ascontiguousarray(triangles[:, 0]),
            np.ascontiguousarray(triangles[:, 1] - triangles[:, 0]),
            np.ascontiguousarray(triangles[:, 2] - triangles[:, 0]),
            np.array(bounds), np.array(nodes, dtype=np.int32), order), max_depth


@njit(cache=False)
def triangle_distance(origin, direction, a, e1, e2, minimum):
    p = np.cross(direction, e2)
    determinant = np.dot(e1, p)
    if abs(determinant) < 1e-12:
        return 1e30
    t = origin - a
    u = np.dot(t, p) / determinant
    q = np.cross(t, e1)
    v = np.dot(direction, q) / determinant
    distance = np.dot(e2, q) / determinant
    if u >= -1e-12 and v >= -1e-12 and u + v <= 1. + 1e-12 and distance > minimum:
        return distance
    return 1e30


@njit(cache=False)
def trace(origin, direction, minimum, maximum, a, e1, e2, bounds, nodes, order, any_hit=False):
    stack = np.empty(64, dtype=np.int32)
    stack[0] = 0
    size, nearest, triangle = 1, maximum, -1
    while size:
        size -= 1
        index = stack[size]
        near, far, intersects = minimum, nearest, True
        for axis in range(3):
            if abs(direction[axis]) < 1e-12:
                if origin[axis] < bounds[index, axis] - 1e-9 or origin[axis] > bounds[index, axis + 3] + 1e-9:
                    intersects = False
            else:
                v0 = (bounds[index, axis] - 1e-9 - origin[axis]) / direction[axis]
                v1 = (bounds[index, axis + 3] + 1e-9 - origin[axis]) / direction[axis]
                near, far = max(near, min(v0, v1)), min(far, max(v0, v1))
        if not intersects or far < near:
            continue
        first, count, left, right = nodes[index]
        if count:
            for position in range(first, first + count):
                candidate = order[position]
                distance = triangle_distance(origin, direction, a[candidate], e1[candidate], e2[candidate], minimum)
                if distance < nearest:
                    nearest, triangle = distance, candidate
                    if any_hit:
                        return nearest, triangle
        else:
            stack[size], stack[size + 1] = left, right
            size += 2
    return nearest, triangle


@njit(cache=False)
def brute(origin, direction, minimum, maximum, a, e1, e2):
    nearest, triangle = maximum, -1
    for candidate in range(len(a)):
        distance = triangle_distance(origin, direction, a[candidate], e1[candidate], e2[candidate], minimum)
        if distance < nearest:
            nearest, triangle = distance, candidate
    return nearest, triangle


@njit(cache=False)
def camera_hits(origins, directions, near, far, a, e1, e2, bounds, nodes, order):
    distances = np.empty(len(origins))
    triangles = np.empty(len(origins), dtype=np.int32)
    for index in range(len(origins)):
        distances[index], triangles[index] = trace(origins[index], directions[index], near, far,
                                                  a, e1, e2, bounds, nodes, order)
    return distances, triangles


@njit(cache=False)
def shadow_rays(positions, direction, epsilon, a, e1, e2, bounds, nodes, order):
    result = np.empty(len(positions), dtype=np.uint8)
    for index in range(len(positions)):
        origin = positions[index] + direction * epsilon
        _, triangle = trace(origin, direction, TRIANGLE_EPSILON, 1e30, a, e1, e2, bounds, nodes, order, True)
        result[index] = 255 if triangle < 0 else 0
    return result


def camera_rays(camera, xs, ys, width, height):
    if camera["projection"] != "orthographic":
        raise ValueError("This protocol currently verifies orthographic Bunny views only")
    eye = np.array(camera["eye"], dtype=np.float64)
    forward = np.array(camera["center"], dtype=np.float64) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, camera["up"])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    half_height = camera["orthoHeight"] * .5
    origins = eye + np.outer((2 * (xs + .5) / width - 1) * width / height * half_height, right)
    origins += np.outer((1 - 2 * (ys + .5) / height) * half_height, up)
    return np.ascontiguousarray(origins), np.tile(forward, (len(xs), 1))


def evaluate(source, output):
    started = time.perf_counter()
    verified = experiment.verify(source)
    if verified["phase"] != "diagnostic":
        raise ValueError("CPU oracle requires a sealed diagnostic package")
    impact.validate_assets(verified["assets"])
    records = verified["records"]
    samples = quality.read_samples(records)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(protocol="virtual-shadow-map-lod-smart-bias-authored-oracle-v1",
        diagnostic=str(source), diagnosticManifestSHA256=acceptance.digest(source / "Manifest.json"),
        oracleScriptSHA256=acceptance.digest(Path(__file__)), numpy=np.__version__, numba=numba.__version__,
        contract=dict(originOffsetDirection="toLight, never surface normal", originOffsetsMetres=EPSILONS,
                      triangleMinimumDistanceMetres=TRIANGLE_EPSILON, depthTolerance=DEPTH_TOLERANCE,
                      cameraPixelCenter="x+.5,y+.5; no jitter", geometry="authored glTF world triangles, double precision",
                      cpuWorkOnly=True), geometryChecks=[], rows=[])
    scene = Path(records[0]["normalized"]["cases"][0]["scene"])
    triangles, owners, instances = authored_triangles(scene)
    bvh, max_depth = build_bvh(triangles)
    report["geometry"] = dict(scene=str(scene), triangles=len(triangles), instances=instances,
                              bvhNodes=len(bvh[4]), bvhDepth=max_depth)
    print(f"Authored geometry: {len(triangles)} triangles, BVH {len(bvh[4])} nodes, depth {max_depth}", flush=True)
    oracle_cache, plot_data, quantized_bvh = {}, [], None
    for workload in verified["workloads"]:
        case = next(r for r in records if r["workload"] == workload)["normalized"]["cases"][0]
        camera = case["camera"]
        direction = -np.array(case["lighting"]["sun"]["direction"], dtype=np.float64)
        direction /= np.linalg.norm(direction)
        for frame in (32, 47):
            reference = samples[workload, "Off", frame]
            masks = quality.masks_for(reference, camera)
            bunny = masks["bunny"]
            height, width = bunny.shape
            ys, xs = np.where(bunny)
            origins, directions = camera_rays(camera, xs, ys, width, height)
            key = (str(camera), tuple(direction), bunny.tobytes(), reference["depth"].tobytes())
            if key not in oracle_cache:
                distances, hit_ids = camera_hits(origins, directions, camera["znear"], camera["zfar"], *bvh)
                if np.any(hit_ids < 0) or np.any(owners[hit_ids] != 0):
                    raise ValueError("GPU Bunny ROI does not correspond to authored Bunny camera hits")
                positions = origins + directions * distances[:, None]
                normals = np.cross(bvh[1][hit_ids], bvh[2][hit_ids])
                normals /= np.linalg.norm(normals, axis=1)[:, None]
                ndotl = normals @ direction
                cpu_depth = (camera["zfar"] - distances) / (camera["zfar"] - camera["znear"])
                residual = np.abs(cpu_depth - reference["depth"][ys, xs])
                check = dict(workload=workload, frame=frame, pixels=len(xs),
                             depthErrors=int(np.count_nonzero(residual > DEPTH_TOLERANCE)),
                             maximumDepthError=float(residual.max()), p99DepthError=float(np.percentile(residual, 99)),
                             worldDistanceErrorMaximum=float(residual.max() * (camera["zfar"] - camera["znear"])))
                disagreement = np.flatnonzero(residual > DEPTH_TOLERANCE)
                check["depthMismatchPixels"] = [dict(x=int(xs[i]), y=int(ys[i]), triangle=int(hit_ids[i]),
                    cpuDepth=float(cpu_depth[i]), gpuDepth=float(reference["depth"][ys[i], xs[i]]),
                    gpuVisibility=int(reference["visibility"][ys[i], xs[i]]), position=positions[i].tolist(),
                    normal=normals[i].tolist(), cameraCosine=float(np.dot(normals[i], -directions[i])),
                    armShadowVisibility={arm: int(samples[workload, arm, frame]["shadow"][ys[i], xs[i]])
                                         for arm in ("Off", "Legacy", "Smart")})
                    for i in disagreement]
                if len(disagreement):
                    if quantized_bvh is None:
                        quantized_geometry, _, _ = authored_triangles(scene, 7)
                        quantized_bvh, _ = build_bvh(quantized_geometry)
                    quantized_distances, quantized_ids = camera_hits(origins, directions, camera["znear"], camera["zfar"], *quantized_bvh)
                    quantized_depth = (camera["zfar"] - quantized_distances) / (camera["zfar"] - camera["znear"])
                    quantized_residual = np.abs(quantized_depth - reference["depth"][ys, xs])
                    check["referenceCodecCameraControl"] = dict(discardedPositionBits=7,
                        source="MeshletStreamReferenceCodec.h default; quantizedBits clears low IEEE754 bits before instance transform",
                        depthErrors=int(np.count_nonzero((quantized_ids < 0) | (quantized_residual > DEPTH_TOLERANCE))),
                        maximumDepthError=float(quantized_residual.max()), p99DepthError=float(np.percentile(quantized_residual, 99)),
                        authoredMismatchResolved=int(np.count_nonzero(quantized_residual[disagreement] <= DEPTH_TOLERANCE)))
                # Retain receiver evidence even when validation stops the oracle.
                np.savez_compressed(output / f"{workload}-{frame}-CameraValidation.npz", mask=bunny,
                    ys=ys, xs=xs, triangleIDs=hit_ids, positions=positions, normals=normals,
                    cpuDepth=cpu_depth, gpuDepth=reference["depth"][ys, xs], residual=residual)
                subset = np.flatnonzero((xs % 31 == 0) & (ys % 29 == 0))
                if len(subset) < 8:
                    raise ValueError("Insufficient deterministic brute-force cross-check samples")
                camera_errors = 0
                for index in subset:
                    distance, triangle = brute(origins[index], directions[index], camera["znear"], camera["zfar"], *bvh[:3])
                    camera_errors += triangle < 0 or abs(distance - distances[index]) > 1e-9
                check.update(bruteForceCameraSamples=len(subset), bruteForceCameraErrors=int(camera_errors))
                oracle = {}
                shadow_errors = 0
                for epsilon in EPSILONS:
                    values = shadow_rays(positions, direction, epsilon, *bvh)
                    for index in subset:
                        _, triangle = brute(positions[index] + direction * epsilon, direction,
                                            TRIANGLE_EPSILON, 1e30, *bvh[:3])
                        shadow_errors += (triangle < 0) != (values[index] == 255)
                    image = np.full((height, width), 127, dtype=np.uint8)
                    image[ys, xs] = values
                    oracle[epsilon] = image
                check.update(bruteForceShadowSamples=len(subset) * len(EPSILONS), bruteForceShadowErrors=int(shadow_errors))
                report["geometryChecks"].append(check)
                acceptance.save(output / "GeometryValidation.json", {"geometry": report["geometry"], "checks": report["geometryChecks"]})
                print(f"{workload} frame {frame}: {len(xs)} camera hits, max depth error {residual.max():.9g}, brute camera/shadow errors {camera_errors}/{shadow_errors}", flush=True)
                if camera_errors or shadow_errors:
                    raise ValueError("Independent BVH/brute-force validation failed")
                normal_image = np.zeros((height, width))
                normal_image[ys, xs] = ndotl
                qualified = np.zeros((height, width), dtype=bool)
                qualified[ys, xs] = residual <= DEPTH_TOLERANCE
                oracle_cache[key] = oracle, normal_image, qualified
                np.savez_compressed(output / f"{workload}-{frame}-Oracle.npz", mask=bunny, ys=ys, xs=xs,
                                    cameraTriangleIDs=hit_ids, cameraPositions=positions, cameraDepth=cpu_depth,
                                    gpuDepth=reference["depth"][ys, xs], ndotl=ndotl,
                                    originOffsets=np.array(EPSILONS), visibility=np.stack([oracle[e] for e in EPSILONS]))
            oracle, normal_image, qualified = oracle_cache[key]
            stable = bunny & np.logical_and.reduce([oracle[e] == oracle[1e-5] for e in EPSILONS])
            for epsilon, image in oracle.items():
                regions = dict(bunny=bunny, qualifiedBunny=qualified, unresolvedBunny=bunny & ~qualified,
                    lit=qualified & (image == 255), umbra=qualified & (image == 0),
                    litInterior=quality.erode(qualified & (image == 255)),
                    umbraInterior=quality.erode(qualified & (image == 0)), stableEpsilon=qualified & stable)
                for name, low, high in (("backFacing", -1.001, 0), ("grazing", 0, .15),
                                         ("shallow", .15, .35), ("oblique", .35, .65), ("facing", .65, 1.001)):
                    regions["ndotl_" + name] = qualified & (normal_image >= low) & (normal_image < high)
                for arm in ("Off", "Legacy", "Smart"):
                    candidate = samples[workload, arm, frame]["shadow"]
                    metrics = {name: quality.changes(image, candidate, mask) for name, mask in regions.items()}
                    report["rows"].append(dict(workload=workload, frame=frame, arm=arm, epsilonMetres=epsilon,
                        epsilonSensitivePixels=int(np.count_nonzero(bunny & ~stable)), metrics=metrics,
                        qualifiedPixels=int(qualified.sum()), unresolvedPixels=int(np.count_nonzero(bunny & ~qualified)),
                        unresolvedDarkPixels=int(np.count_nonzero(bunny & ~qualified & (candidate == 0))),
                        unresolvedLitPixels=int(np.count_nonzero(bunny & ~qualified & (candidate == 255))),
                        bunnyInternalEdges=quality.edge_displacement(image, candidate, quality.erode(qualified))))
            if frame == 47:
                plot_data.append((workload, oracle[1e-5], qualified, {arm: samples[workload, arm, frame]["shadow"] for arm in ("Off", "Legacy", "Smart")}))
    report["elapsedSeconds"] = time.perf_counter() - started
    report["geometryValidationPassed"] = all(c["depthErrors"] == 0 for c in report["geometryChecks"])
    report["qualifiedReceiverAnalysisAvailable"] = True
    report["qualityScope"] = "Independent visibility counts, not a tuned acceptance threshold. Preserve all epsilon and raw/interior results. Plane quality uses the separate existing oracle."
    acceptance.save(output / "Oracle.json", report)
    plot(plot_data, output)
    write_report(report, output)
    acceptance.save(output / "Manifest.json", {str(p.relative_to(output)): acceptance.digest(p)
                                               for p in sorted(output.rglob("*")) if p.is_file() and p.name != "Manifest.json"})
    print(output / "Report.md", flush=True)


def plot(samples, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for workload, oracle, bunny, arms in samples:
        ys, xs = np.where(bunny)
        crop = (slice(max(0, ys.min() - 4), ys.max() + 5), slice(max(0, xs.min() - 4), xs.max() + 5))
        figure, axes = plt.subplots(2, 4, figsize=(13, 8), constrained_layout=True)
        axes[0, 0].imshow(np.where(bunny, oracle, 127)[crop], cmap="gray", vmin=0, vmax=255)
        axes[0, 0].set_title("Authored CPU oracle")
        axes[1, 0].text(.02, .5, "Red: false shadow\nCyan: lost true occlusion\n\nOutside qualified ROI is gray\nOrigin offset: 10 micrometres", transform=axes[1, 0].transAxes)
        for column, (arm, image) in enumerate(arms.items(), 1):
            shown = np.where(bunny, image, 127)
            axes[0, column].imshow(shown[crop], cmap="gray", vmin=0, vmax=255)
            axes[0, column].set_title(arm)
            difference = np.zeros((*image.shape, 3), dtype=np.uint8)
            difference[bunny & (oracle == 255) & (image == 0)] = (255, 60, 60)
            difference[bunny & (oracle == 0) & (image == 255)] = (0, 230, 255)
            axes[1, column].imshow(difference[crop])
            axes[1, column].set_title(arm + " vs authored oracle")
        for axis in axes.flat:
            axis.set_axis_off()
        figure.savefig(output / f"{workload}-AuthoredOracle.png", dpi=140)
        plt.close(figure)


def write_report(report, output):
    lines = ["# Independent authored Bunny visibility", "", "CPU nearest-hit camera rays were compared with GPU receiver depth and verified against a deterministic brute-force subset. Depth discrepancies above 2e-5 remain unresolved and are excluded only from the explicitly qualified metrics. Their positions, normals and all three arm values are retained in GeometryValidation.json. Directional shadow rays start at the CPU-authored triangle hit; Off images do not supply ground truth.", "",
             "| State | Arm | Offset (m) | False shadow | Lost true occlusion | Lit-interior false shadow | Umbra-interior leaks | Epsilon-sensitive pixels |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["rows"]:
        if row["frame"] != 47:
            continue
        m = row["metrics"]
        lines.append(f"| {row['workload']} | {row['arm']} | {row['epsilonMetres']:g} | {m['qualifiedBunny']['extraDark']} | {m['qualifiedBunny']['lostOcclusion']} | {m['litInterior']['extraDark']} / {m['litInterior']['pixels']} | {m['umbraInterior']['lostOcclusion']} / {m['umbraInterior']['pixels']} | {row['epsilonSensitivePixels']} |")
    lines += ["", "Raw, interior and all origin-offset results are retained. These measurements do not declare a newly tuned acceptance threshold. Complex Bunny edge matching is diagnostic; it must not inherit the flat-receiver edge tolerance without qualification.", ""]
    for workload in dict.fromkeys(row["workload"] for row in report["rows"]):
        lines.append(f"![{workload} authored oracle]({workload}-AuthoredOracle.png)")
    (output / "Report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostic", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    evaluate(args.diagnostic.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
