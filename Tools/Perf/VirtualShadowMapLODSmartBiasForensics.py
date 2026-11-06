"""Offline geometric explanation of Smart-bias changes against an authored oracle.

This does not fit rejection thresholds or launch a GPU. It retains every newly
lost true-shadow pixel and compares the same measurements on corrected acne.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
from numba import njit

import VirtualShadowMapAcceptance as acceptance
import VirtualShadowMapLODSmartBias as experiment
import VirtualShadowMapLODSmartBiasOracle as oracle
import VirtualShadowMapLODSmartBiasQuality as quality


@njit(cache=False)
def nearest_mask_distances(xs, ys, boundary_x, boundary_y):
    result = np.full(len(xs), np.inf)
    for i in range(len(xs)):
        for j in range(len(boundary_x)):
            distance = (xs[i] - boundary_x[j]) ** 2 + (ys[i] - boundary_y[j]) ** 2
            if distance < result[i]:
                result[i] = distance
        result[i] = math.sqrt(result[i])
    return result


def verified_manifest(directory):
    manifest = acceptance.read(directory / "Manifest.json")
    actual = {str(p.relative_to(directory)): acceptance.digest(p) for p in sorted(directory.rglob("*"))
              if p.is_file() and p.name != "Manifest.json"}
    if actual != manifest:
        raise ValueError("Oracle artifact file/hash set changed")
    return acceptance.digest(directory / "Manifest.json")


def camera_basis(camera):
    eye = np.array(camera["eye"], dtype=np.float64)
    forward = np.array(camera["center"], dtype=np.float64) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, camera["up"])
    right /= np.linalg.norm(right)
    return eye, forward, right, np.cross(right, forward)


def project(position, basis, camera, width, height):
    eye, forward, right, up = basis
    delta = position - eye
    half_height = camera["orthoHeight"] * .5
    screen = np.array([(np.dot(delta, right) / (half_height * width / height) + 1) * width * .5,
                       (1 - np.dot(delta, up) / half_height) * height * .5])
    return screen, float(np.dot(delta, forward))


def receiver_images(camera, depth):
    height, width = depth.shape
    ys, xs = np.mgrid[:height, :width]
    origins, directions = oracle.camera_rays(camera, xs.ravel(), ys.ravel(), width, height)
    linear_depth = camera["zfar"] - depth.astype(float) * (camera["zfar"] - camera["znear"])
    positions = (origins + directions * linear_depth.reshape(-1, 1)).reshape(height, width, 3)
    valid = depth > 0
    normals = np.zeros_like(positions)
    center = positions[1:-1, 1:-1]
    z = linear_depth[1:-1, 1:-1]
    has_left, has_right = valid[1:-1, :-2], valid[1:-1, 2:]
    has_top, has_bottom = valid[:-2, 1:-1], valid[2:, 1:-1]
    use_left = has_left & (~has_right | (np.abs(linear_depth[1:-1, :-2] - z) < np.abs(linear_depth[1:-1, 2:] - z)))
    use_top = has_top & (~has_bottom | (np.abs(linear_depth[:-2, 1:-1] - z) < np.abs(linear_depth[2:, 1:-1] - z)))
    dx = np.where(use_left[..., None], center - positions[1:-1, :-2],
                  np.where(has_right[..., None], positions[1:-1, 2:] - center, 0))
    dy = np.where(use_top[..., None], center - positions[:-2, 1:-1],
                  np.where(has_bottom[..., None], positions[2:, 1:-1] - center, 0))
    normal = np.cross(dy, dx)
    length = np.linalg.norm(normal, axis=2)
    normals[1:-1, 1:-1] = normal / np.where(length > 1e-10, length, 1)[..., None]
    normals[~valid] = 0
    return positions, normals, linear_depth, valid


def boundary_masks(depth, normals, valid, cpu_visibility, bunny, pixel_world_size):
    depth_edges = np.zeros_like(valid)
    normal_edges = np.zeros_like(valid)
    shadow_edges = np.zeros_like(valid)
    for a, b in (((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                 ((slice(None, -1), slice(None)), (slice(1, None), slice(None)))):
        pair = valid[a] & valid[b]
        depth_jump = pair & (np.abs(depth[a] - depth[b]) > pixel_world_size)
        normal_jump = pair & (np.sum(normals[a] * normals[b], axis=2) < math.cos(math.radians(45)))
        shadow_jump = bunny[a] & bunny[b] & (cpu_visibility[a] != cpu_visibility[b])
        depth_edges[a] |= depth_jump
        depth_edges[b] |= depth_jump
        normal_edges[a] |= normal_jump
        normal_edges[b] |= normal_jump
        shadow_edges[a] |= shadow_jump
        shadow_edges[b] |= shadow_jump
    return dict(depthJump=depth_edges, normalJump=normal_edges, trueShadowEdge=shadow_edges)


def percentiles(values):
    data = np.array([v for v in values if v is not None], dtype=float)
    return dict(count=len(data), minimum=float(data.min()), p10=float(np.percentile(data, 10)),
                median=float(np.median(data)), p90=float(np.percentile(data, 90)), maximum=float(data.max())) if len(data) else dict(count=0)


def normal_sign_study(positions, normals, valid, to_light, rows):
    """Use only the sign of locally reconstructed light-facing normals."""
    height, width = valid.shape
    results = []
    for row in rows:
        x, y = row["x"], row["y"]
        quadrants = {}
        for horizontal, hx in (("Left", -1), ("Right", 1)):
            for vertical, vy in (("Top", -1), ("Bottom", 1)):
                if not (0 <= x + hx < width and 0 <= y + vy < height
                        and valid[y, x + hx] and valid[y + vy, x]):
                    continue
                dx = (positions[y, x + hx] - positions[y, x]) * hx
                dy = (positions[y + vy, x] - positions[y, x]) * vy
                cross = np.cross(dy, dx)
                length = np.linalg.norm(cross)
                quadrants[horizontal + vertical] = float(np.dot(cross, to_light) / length) if length > 1e-10 else 0.
        neighbors = {}
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            if 0 <= x + dx < width and 0 <= y + dy < height and valid[y + dy, x + dx]:
                neighbors[f"{dx},{dy}"] = float(np.dot(normals[y + dy, x + dx], to_light))
        quad_min, quad_max = min(quadrants.values()), max(quadrants.values())
        neighbor4_min = min(v for k, v in neighbors.items() if k in ("-1,0", "1,0", "0,-1", "0,1"))
        neighbor8_min = min(neighbors.values())
        results.append(dict(category=row["category"], x=x, y=y, authoredNdotL=row["authoredNdotL"],
            originalNdotL=row["gpuNormalNdotL"], quadrantNdotL=quadrants, quadrantMin=quad_min, quadrantMax=quad_max,
            quadrantCrossesZero=quad_min <= 0 < quad_max, neighborNdotL=neighbors,
            neighbor4Min=neighbor4_min, neighbor8Min=neighbor8_min,
            rejection=dict(quadrants=quad_min <= 0, neighbor4=neighbor4_min <= 0, neighbor8=neighbor8_min <= 0,
                           quadrantsOrNeighbor4=quad_min <= 0 or neighbor4_min <= 0,
                           quadrantsOrNeighbor8=quad_min <= 0 or neighbor8_min <= 0)))
    summary = {}
    for method in results[0]["rejection"]:
        leaks = [r for r in results if r["category"] == "newTrueLeak"]
        acne = [r for r in results if r["category"] == "correctedAcne"]
        summary[method] = dict(preventedTrueLeaks=sum(r["rejection"][method] for r in leaks),
            remainingTrueLeaks=sum(not r["rejection"][method] for r in leaks),
            retainedAcneCorrections=sum(not r["rejection"][method] for r in acne),
            lostAcneCorrections=sum(r["rejection"][method] for r in acne))
    unresolved = [r for r in results if r["category"] == "newTrueLeak" and r["authoredNdotL"] <= 0 and r["quadrantMin"] > 0]
    return dict(contract="All tests use NdotL <= 0 only. No fitted angular threshold. Invalid depth neighbors are omitted; degenerate cross products use shader's zero-normal convention.",
                summary=summary, trueBackFacingButAllQuadrantsFront=unresolved, pixels=results)


def geometry_normal_sign_study(camera, scene, to_light, rows, width, height):
    coordinates = sorted({(r["x"] + dx, r["y"] + dy) for r in rows
                          for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                          if 0 <= r["x"] + dx < width and 0 <= r["y"] + dy < height})
    xs = np.array([p[0] for p in coordinates])
    ys = np.array([p[1] for p in coordinates])
    geometry, owners, _ = oracle.authored_triangles(scene)
    bvh, _ = oracle.build_bvh(geometry)
    origins, directions = oracle.camera_rays(camera, xs, ys, width, height)
    distances, triangles = oracle.camera_hits(origins, directions, camera["znear"], camera["zfar"], *bvh)
    hits = {}
    for index, xy in enumerate(coordinates):
        triangle = triangles[index]
        if triangle < 0:
            continue
        normal = np.cross(bvh[1][triangle], bvh[2][triangle])
        length = np.linalg.norm(normal)
        normal = normal / length if length > 1e-10 else normal * 0.
        hits[xy] = dict(triangle=int(triangle), mesh=int(owners[triangle]), normal=normal.tolist(),
                       ndotl=float(np.dot(normal, to_light)), cameraDistance=float(distances[index]))
    results = []
    for row in rows:
        x, y = row["x"], row["y"]
        center = hits[x, y]
        if center["triangle"] != row["receiverTriangle"] or abs(center["ndotl"] - row["authoredNdotL"]) > 1e-10:
            raise ValueError("Continuous CPU camera query changed the qualified receiver")
        neighbors = {f"{dx},{dy}": hits[x + dx, y + dy] for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                     if (dx or dy) and (x + dx, y + dy) in hits}
        n4 = min([center["ndotl"]] + [v["ndotl"] for k, v in neighbors.items() if k in ("-1,0", "1,0", "0,-1", "0,1")])
        n8 = min([center["ndotl"]] + [v["ndotl"] for v in neighbors.values()])
        results.append(dict(category=row["category"], x=x, y=y, center=center, neighbors=neighbors,
            centerAnd4Min=n4, centerAnd8Min=n8, rejection=dict(center=center["ndotl"] <= 0,
                centerAnd4=n4 <= 0, centerAnd8=n8 <= 0)))
    summary = {}
    leaks = [r for r in results if r["category"] == "newTrueLeak"]
    acne = [r for r in results if r["category"] == "correctedAcne"]
    for method in results[0]["rejection"]:
        summary[method] = dict(preventedTrueLeaks=sum(r["rejection"][method] for r in leaks),
            remainingTrueLeaks=sum(not r["rejection"][method] for r in leaks),
            retainedAcneCorrections=sum(not r["rejection"][method] for r in acne),
            lostAcneCorrections=sum(r["rejection"][method] for r in acne))
    return dict(contract="Independent continuous camera nearest-hit authored triangle normals at pixel centers; zero sign threshold only. All missing camera hits omitted. Center is cross-checked against the qualified forensic receiver.",
        queriedPixels=len(coordinates), cameraHits=len(hits), summary=summary,
        remainingCenterFrontFacing=[r for r in leaks if not r["rejection"]["center"]],
        hiddenBlockerPixel=next(r for r in leaks if (r["x"], r["y"]) == (295, 353)), pixels=results)


def segment_pixels(start, end, width, height):
    delta = end - start
    times = [0., 1.]
    for axis in range(2):
        if abs(delta[axis]) < 1e-12:
            continue
        for boundary in range(math.floor(min(start[axis], end[axis])) + 1,
                              math.ceil(max(start[axis], end[axis]))):
            t = (boundary - start[axis]) / delta[axis]
            if 0 < t < 1:
                times.append(t)
    times = sorted(set(times))
    pixels = {tuple(np.floor(start + delta * ((a + b) * .5)).astype(int)) for a, b in zip(times, times[1:])}
    pixels.add(tuple(np.floor(start).astype(int)))
    pixels.add(tuple(np.floor(end).astype(int)))
    return sorted((int(x), int(y)) for x, y in pixels if 0 <= x < width and 0 <= y < height)


def segment_study(camera, scene, to_light, rows, width, height, normal_report):
    normal_manifest = verified_manifest(normal_report)
    normal_data = acceptance.read(normal_report / "NormalSigns.json")
    normals = {(r["x"], r["y"]): r for r in normal_data["pixels"]}
    geometry, _, instances = oracle.authored_triangles(scene)
    if len({r["mesh"] for r in instances}) != len(instances):
        raise ValueError("This fixed-fixture mesh-owner shortcut requires one instance per mesh")
    # In this immutable fixture every mesh has exactly one instance. The node
    # is retained explicitly instead of conflating a material seam with owner.
    mesh_owner = {r["mesh"]: [r["mesh"], r["node"]] for r in instances}
    bvh, _ = oracle.build_bvh(geometry)
    basis = camera_basis(camera)
    prepared, all_coordinates = [], set()
    for row in rows:
        normal = normals[row["x"], row["y"]]
        neighbors = [normal["neighbors"].get(k) for k in ("-1,0", "1,0", "0,-1", "0,1")]
        owner = mesh_owner[normal["center"]["mesh"]]
        footprint_reject = any(n is None or mesh_owner[n["mesh"]] != owner for n in neighbors)
        entry = dict(category=row["category"], x=row["x"], y=row["y"], normalRejected=normal["rejection"]["centerAnd4"],
                     footprintRejected=footprint_reject, centerOwnerMeshNode=owner,
                     fourNeighborOwners=[mesh_owner[n["mesh"]] if n is not None else None for n in neighbors],
                     receiverPosition=row["receiverPosition"], directPixels=[], expandedPixels=[],
                     directIntersection=None, expandedIntersection=None)
        if not entry["normalRejected"] and not footprint_reject:
            start = np.array(row["receiverPosition"]) + to_light * 1e-5
            end = np.array(row["receiverPosition"]) + to_light * .12
            start_screen, _ = project(start, basis, camera, width, height)
            end_screen, _ = project(end, basis, camera, width, height)
            pixels = segment_pixels(start_screen, end_screen, width, height)
            expanded = sorted({(x + dx, y + dy) for x, y in pixels for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                               if 0 <= x + dx < width and 0 <= y + dy < height})
            entry.update(directPixels=pixels, expandedPixels=expanded, screenSegment=[start_screen.tolist(), end_screen.tolist()])
            all_coordinates.update(expanded)
        prepared.append(entry)
    coordinates = sorted(all_coordinates)
    xs, ys = np.array([p[0] for p in coordinates]), np.array([p[1] for p in coordinates])
    origins, directions = oracle.camera_rays(camera, xs, ys, width, height)
    _, triangles = oracle.camera_hits(origins, directions, camera["znear"], camera["zfar"], *bvh)
    hits = {xy: int(triangles[i]) for i, xy in enumerate(coordinates)}
    for row in prepared:
        if row["normalRejected"] or row["footprintRejected"]:
            continue
        start = np.array(row["receiverPosition"]) + to_light * 1e-5
        for key, pixels in (("directIntersection", row["directPixels"]), ("expandedIntersection", row["expandedPixels"])):
            candidates = sorted({hits[tuple(pixel)] for pixel in pixels if hits[tuple(pixel)] >= 0})
            nearest, nearest_triangle = .12 - 1e-5, -1
            for triangle in candidates:
                distance = oracle.triangle_distance(start, to_light, *[part[triangle] for part in bvh[:3]], oracle.TRIANGLE_EPSILON)
                if distance <= nearest:
                    nearest, nearest_triangle = distance, triangle
            row[key] = dict(candidateTriangleCount=len(candidates), blocked=nearest_triangle >= 0,
                            triangle=nearest_triangle, distanceMetres=float(nearest + 1e-5) if nearest_triangle >= 0 else None)
    stages = {
        "centerAnd4": lambda r: r["normalRejected"],
        "centerAnd4SameOwnerFootprint": lambda r: r["normalRejected"] or r["footprintRejected"],
        "withSegmentTriangles": lambda r: r["normalRejected"] or r["footprintRejected"] or r["directIntersection"]["blocked"],
        "withOnePixelExpandedTriangles": lambda r: r["normalRejected"] or r["footprintRejected"] or r["expandedIntersection"]["blocked"],
    }
    summary = {}
    for name, rejects in stages.items():
        leak = [r for r in prepared if r["category"] == "newTrueLeak"]
        acne = [r for r in prepared if r["category"] == "correctedAcne"]
        summary[name] = dict(preventedTrueLeaks=sum(rejects(r) for r in leak),
            remainingTrueLeaks=sum(not rejects(r) for r in leak),
            retainedAcneCorrections=sum(not rejects(r) for r in acne),
            lostAcneCorrections=sum(rejects(r) for r in acne))
    return dict(contract="Center+4 continuous camera-hit geometric normals, zero sign threshold; reject any missing/different mesh-instance 4-neighbor. Remaining rays test primary triangles from all traversed 0.12m projected segment cells, optionally expanded by exactly one pixel. Whole 3D segment intersection is tested, not endpoint sampling. No fitted thresholds.",
        normalReportManifestSHA256=normal_manifest, triangleOwnerInstances=instances, queriedPixels=len(coordinates),
        summary=summary, remainingAfterFootprint=[r for r in prepared if r["category"] == "newTrueLeak" and not r["normalRejected"] and not r["footprintRejected"]],
        hiddenBlockerPixel=next(r for r in prepared if (r["x"], r["y"]) == (295, 353)), pixels=prepared)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostic", type=Path)
    parser.add_argument("oracle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reuse-forensics", type=Path,
                        help="Read a verified previous pixel report and run only the local normal-sign study")
    parser.add_argument("--geometry-normal-study", action="store_true",
                        help="Use independent authored camera-hit normals instead of depth finite differences")
    parser.add_argument("--segment-study", type=Path,
                        help="Use a verified geometry-normal study for owner-footprint and exact visible-triangle segment checks")
    args = parser.parse_args()
    started = time.perf_counter()
    source, evidence, output = args.diagnostic.resolve(), args.oracle.resolve(), args.output.resolve()
    manifest_hash = verified_manifest(evidence)
    oracle_report = acceptance.read(evidence / "Oracle.json")
    if oracle_report["diagnosticManifestSHA256"] != acceptance.digest(source / "Manifest.json"):
        raise ValueError("The oracle is from another GPU diagnostic")
    value = experiment.verify(source)
    samples = quality.read_samples(value["records"])
    records = value["records"]
    case = next(r for r in records if r["workload"] == "Redraw")["normalized"]["cases"][0]
    camera = case["camera"]
    basis = camera_basis(camera)
    to_light = -np.array(case["lighting"]["sun"]["direction"], dtype=np.float64)
    to_light /= np.linalg.norm(to_light)
    image = samples["Redraw", "Off", 47]
    data = np.load(evidence / "Redraw-32-Oracle.npz")
    receivers = np.load(evidence / "Redraw-32-CameraValidation.npz")
    cpu_visibility = data["visibility"][int(np.flatnonzero(data["originOffsets"] == 1e-5)[0])]
    qualified = np.zeros_like(data["mask"])
    qualified[receivers["ys"], receivers["xs"]] = receivers["residual"] <= oracle.DEPTH_TOLERANCE
    changed = qualified & (samples["Redraw", "Legacy", 47]["shadow"] == 0) & (samples["Redraw", "Smart", 47]["shadow"] == 255)
    categories = dict(newTrueLeak=changed & (cpu_visibility == 0), correctedAcne=changed & (cpu_visibility == 255))
    if (int(categories["newTrueLeak"].sum()), int(categories["correctedAcne"].sum())) != (63, 1359):
        raise ValueError("This forensic protocol is locked to Diagnostic2's 63 leaks and 1,359 corrections")
    output.mkdir(parents=True, exist_ok=False)
    if args.reuse_forensics is not None:
        previous = args.reuse_forensics.resolve()
        previous_hash = verified_manifest(previous)
        receipt = acceptance.read(previous / "Summary.json")
        if receipt["diagnosticManifestSHA256"] != oracle_report["diagnosticManifestSHA256"]:
            raise ValueError("Previous pixel report used another diagnostic")
        rows = acceptance.read(previous / "newTrueLeak-Pixels.json") + acceptance.read(previous / "correctedAcne-Pixels.json")
        if args.segment_study is not None:
            study = segment_study(camera, Path(case["scene"]), to_light, rows,
                                  image["depth"].shape[1], image["depth"].shape[0], args.segment_study.resolve())
        elif args.geometry_normal_study:
            study = geometry_normal_sign_study(camera, Path(case["scene"]), to_light, rows,
                                               image["depth"].shape[1], image["depth"].shape[0])
        else:
            gpu_positions, gpu_normals, _, valid = receiver_images(camera, image["depth"])
            study = normal_sign_study(gpu_positions, gpu_normals, valid, to_light, rows)
        study.update(sourceForensics=str(previous), sourceForensicsManifestSHA256=previous_hash,
                     scriptSHA256=acceptance.digest(Path(__file__)), elapsedSeconds=time.perf_counter() - started)
        acceptance.save(output / "NormalSigns.json", study)
        acceptance.save(output / "Manifest.json", {str(p.relative_to(output)): acceptance.digest(p) for p in sorted(output.rglob("*"))
                                                   if p.is_file() and p.name != "Manifest.json"})
        print(json.dumps(study["summary"], indent=2), flush=True)
        return
    triangles, _, _ = oracle.authored_triangles(Path(case["scene"]))
    bvh, _ = oracle.build_bvh(triangles)
    gpu_positions, gpu_normals, linear_depth, valid = receiver_images(camera, image["depth"])
    height, width = valid.shape
    pixel_size = camera["orthoHeight"] / height
    boundaries = boundary_masks(linear_depth, gpu_normals, valid, cpu_visibility, data["mask"], pixel_size)
    receiver_index = np.full(valid.shape, -1, dtype=np.int32)
    receiver_index[receivers["ys"], receivers["xs"]] = np.arange(len(receivers["xs"]))
    rows = []
    for category, mask in categories.items():
        ys, xs = np.where(mask)
        distances = {}
        for name, boundary in boundaries.items():
            by, bx = np.where(boundary)
            distances[name] = nearest_mask_distances(xs, ys, bx, by)
        for i, (x, y) in enumerate(zip(xs, ys)):
            index = receiver_index[y, x]
            position, normal = receivers["positions"][index], receivers["normals"][index]
            distance, triangle = oracle.trace(position + to_light * 1e-5, to_light,
                oracle.TRIANGLE_EPSILON, 1e30, *bvh)
            row = dict(category=category, x=int(x), y=int(y), receiverTriangle=int(receivers["triangleIDs"][index]),
                receiverPosition=position.tolist(), authoredNormal=normal.tolist(), gpuDepthNormal=gpu_normals[y, x].tolist(),
                authoredNdotL=float(np.dot(normal, to_light)), gpuNormalNdotL=float(np.dot(gpu_normals[y, x], to_light)),
                normalDifferenceDegrees=math.degrees(math.acos(float(np.clip(np.dot(normal, gpu_normals[y, x]), -1, 1)))),
                receiverCameraCosine=float(np.dot(normal, -basis[1])),
                boundaryDistancePixels={name: float(values[i]) for name, values in distances.items()},
                nearestTrueBlockerDistanceMetres=None, blocker=None)
            if triangle >= 0:
                distance += 1e-5
                blocker_position = position + to_light * distance
                blocker_normal = np.cross(bvh[1][triangle], bvh[2][triangle])
                blocker_normal /= np.linalg.norm(blocker_normal)
                screen, blocker_depth = project(blocker_position, basis, camera, width, height)
                bx, by = int(math.floor(screen[0])), int(math.floor(screen[1]))
                in_view = 0 <= bx < width and 0 <= by < height
                continuous_origin = blocker_position - basis[1] * blocker_depth
                camera_distance, camera_triangle = oracle.trace(continuous_origin, basis[1],
                    camera["znear"], camera["zfar"], *bvh)
                visible = camera_triangle >= 0 and abs(camera_distance - blocker_depth) <= 1e-5
                blocker = dict(triangle=int(triangle), position=blocker_position.tolist(), normal=blocker_normal.tolist(),
                    normalLightCosine=float(np.dot(blocker_normal, to_light)), cameraCosine=float(np.dot(blocker_normal, -basis[1])),
                    continuousScreenPixelCenter=(screen - .5).tolist(), inViewport=in_view,
                    authoredCameraVisible=bool(visible), cameraFirstTriangle=int(camera_triangle),
                    cameraOcclusionGapMetres=float(blocker_depth - camera_distance),
                    viewDepthMetres=blocker_depth, projectedRayLengthPixels=float(np.linalg.norm(screen - [x + .5, y + .5])))
                if in_view:
                    sample_position = gpu_positions[by, bx]
                    denominator = row["gpuNormalNdotL"]
                    blocker.update(samplePixel=[bx, by], gpuDepthValid=bool(valid[by, bx]),
                        sampledDepth=float(image["depth"][by, bx]), sampledLinearDepthMetres=float(linear_depth[by, bx]),
                        pointMinusSampleDepthMetres=float(blocker_depth - linear_depth[by, bx]),
                        samplePosition=sample_position.tolist(), sampleNormal=gpu_normals[by, bx].tolist(),
                        receiverPlaneDistanceMetres=float(np.dot(gpu_normals[y, x], sample_position - gpu_positions[y, x]) / denominator)
                            if abs(denominator) > 1e-5 else None,
                        depthNeighborMinMetres=float(linear_depth[max(0, by-1):by+2, max(0, bx-1):bx+2].min()),
                        depthNeighborMaxMetres=float(linear_depth[max(0, by-1):by+2, max(0, bx-1):bx+2].max()))
                row.update(nearestTrueBlockerDistanceMetres=float(distance), blocker=blocker)
            elif category == "newTrueLeak":
                raise ValueError("A reported true-shadow pixel has no authored blocker")
            rows.append(row)
        acceptance.save(output / f"{category}-Pixels.json", rows[-len(xs):])
        print(f"{category}: {len(xs)} pixels analyzed", flush=True)
    summary = {}
    for category in categories:
        chosen = [r for r in rows if r["category"] == category]
        ys, xs = np.where(categories[category])
        summary[category] = dict(pixels=len(chosen), boundingBox=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            blockerDistanceMetres=percentiles([r["nearestTrueBlockerDistanceMetres"] for r in chosen]),
            authoredNdotL=percentiles([r["authoredNdotL"] for r in chosen]), gpuNormalNdotL=percentiles([r["gpuNormalNdotL"] for r in chosen]),
            normalDifferenceDegrees=percentiles([r["normalDifferenceDegrees"] for r in chosen]),
            boundaries={name: percentiles([r["boundaryDistancePixels"][name] for r in chosen]) for name in boundaries},
            blockerCameraVisible=sum(r["blocker"] is not None and r["blocker"]["authoredCameraVisible"] for r in chosen),
            blockerCameraHidden=sum(r["blocker"] is not None and not r["blocker"]["authoredCameraVisible"] for r in chosen),
            blockerProjectedRayPixels=percentiles([r["blocker"]["projectedRayLengthPixels"] for r in chosen if r["blocker"] is not None]))
    report = dict(protocol="virtual-shadow-map-lod-smart-bias-forensics-v1", summary=summary,
        diagnostic=str(source), diagnosticManifestSHA256=acceptance.digest(source / "Manifest.json"),
        oracle=str(evidence), oracleManifestSHA256=manifest_hash, scriptSHA256=acceptance.digest(Path(__file__)),
        contract=dict(workload="Redraw", frame=47, epsilonMetres=1e-5, qualifiedDepthTolerance=oracle.DEPTH_TOLERANCE,
            worldPixelSizeMetres=pixel_size, depthBoundary="Adjacent valid pixel linear depths differ by more than one screen pixel's world size",
            normalBoundary="Adjacent reconstructed depth normals differ by more than 45 degrees",
            shadowBoundary="Adjacent authored visibility differs within Bunny ROI",
            boundaryScope="Descriptive distributions only; no fitted shader threshold or pixel whitelist"),
        elapsedSeconds=time.perf_counter() - started)
    acceptance.save(output / "Summary.json", report)
    plot(image["shadow"], data["mask"], categories, rows, output)
    lines = ["# Smart bias geometric forensics", "", "Fixed Diagnostic2, Redraw frame 47. Authored camera-depth-qualified pixels only. All per-pixel values remain in the two JSON files; these descriptive measurements do not prescribe thresholds.", "",
             "```json", json.dumps(summary, indent=2), "```", "", "![Locations](PixelLocations.png)"]
    (output / "Report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    acceptance.save(output / "Manifest.json", {str(p.relative_to(output)): acceptance.digest(p) for p in sorted(output.rglob("*"))
                                               if p.is_file() and p.name != "Manifest.json"})
    print(output / "Report.md", flush=True)


def plot(reference, bunny, categories, rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ys, xs = np.where(bunny)
    figure, axes = plt.subplots(1, 2, figsize=(13, 7), constrained_layout=True)
    for axis in axes:
        axis.imshow(np.where(bunny, reference, 127), cmap="gray", vmin=0, vmax=255)
        axis.set_xlim(xs.min()-5, xs.max()+5)
        axis.set_ylim(ys.max()+5, ys.min()-5)
        axis.set_aspect("equal")
    for category, color in (("correctedAcne", "lime"), ("newTrueLeak", "red")):
        ys, xs = np.where(categories[category])
        axes[0].scatter(xs, ys, s=7 if category == "correctedAcne" else 20, color=color, label=category)
    axes[0].legend()
    axes[0].set_title("Qualified changed pixels")
    for visible, color in ((True, "cyan"), (False, "magenta")):
        selected = [r for r in rows if r["blocker"] is not None and r["blocker"]["authoredCameraVisible"] == visible]
        axes[1].scatter([r["x"] for r in selected], [r["y"] for r in selected], color=color, s=20,
                        label="Blocker camera-visible" if visible else "Blocker camera-hidden")
        for row in selected:
            point = row["blocker"]["continuousScreenPixelCenter"]
            axes[1].plot([row["x"], point[0]], [row["y"], point[1]], color=color, alpha=.5)
    axes[1].legend()
    axes[1].set_title("True leaks and nearest blocker projections")
    figure.savefig(output / "PixelLocations.png", dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
