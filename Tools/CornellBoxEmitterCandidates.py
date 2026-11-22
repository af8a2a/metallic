"""Review the emitter-colored rays that G's triangle oracle left unresolved."""

from collections import Counter
import math

from CheckCornellBoxDetailRays import (
    capture_gdf_voxel_size, EMITTER_RGB_TOLERANCE, FLOAT, FLOAT4, GDF_HIT, HALF4,
    INVALID_PROBE, MIN_TRACE_DISTANCE, PROBE_SLOTS, PROBE_TILE_EDGE, RAY_COUNT,
    RAY_EDGE, SCREEN_ACCEPTED, SURFACE_BIAS, TRACE_RECORD, add, camera_axes, dot,
    first_hit, floor_screen_origin, floor_screen_pixel, multiply, nearest_surfaces, normalize,
    primary_hit, read_buffer, subtract,
)
from MeasureCornellBoxContactVisibility import (
    SCREEN_HIT_CLASS, SCREEN_HIT_DISTANCE_TOLERANCE, channel_max_error,
    emitter_filter_support, read_pixel,
)
from MeasureCornellBoxGI import read_pixels


def trace_flags(trace, ray_capture, tile, slot, lane):
    x, y, width, height = ray_capture["tiles"]
    tx, ty = tile[0] - x, tile[1] - y
    if 0 <= tx < width and 0 <= ty < height:
        index = ((ty * width + tx) * PROBE_SLOTS + slot) * RAY_COUNT + lane
        return read_pixel(TRACE_RECORD, trace, index)[-1]
    return None


def screen_texel(position, manifest):
    projection = floor_screen_pixel(position, manifest)
    edge_coords = tuple(coord + 0.5 for coord in projection)
    pixel = tuple(min(max(math.floor(coord), 0), size - 1) for coord, size in zip(edge_coords, manifest["resolution"]))
    return pixel, projection, min(abs(coord - round(coord)) for coord in edge_coords)


def hit_after_starting_tile(triangles, origin, direction, hit, manifest):
    # TraceHZB steps out of the full-resolution starting tile before depth testing.
    start_pixel = screen_texel(origin, manifest)[0]
    skipped = []
    while hit is not None and screen_texel(hit["position"], manifest)[0] == start_pixel:
        skipped.append(hit)
        hit = first_hit(triangles, origin, direction, math.nextafter(hit["distance"], math.inf),
                        manifest["view"]["camera"]["zfar"])
    return start_pixel, skipped, hit


def infer_screen_sample(manifest, triangles, receiver, origin, direction, distance, rgb, previous_image, hzb):
    # Accepted Screen rays store the unencoded distance from the unbiased receiver
    # to the projected ray endpoint. Solve that distance along the biased ray.
    offset = subtract(origin, receiver)
    along = dot(offset, direction)
    radicand = distance * distance - dot(offset, offset) + along * along
    if radicand < 0.0:
        return None
    ray_parameter = math.sqrt(radicand) - along
    if ray_parameter < 0.0:
        return None
    world = add(origin, multiply(direction, ray_parameter))
    extent = manifest["resolution"]
    camera = manifest["view"]["camera"]
    pixel, projection, edge_clearance = screen_texel(world, manifest)
    index = pixel[1] * extent[0] + pixel[0]
    previous_layout, previous_data = previous_image
    previous_rgb = read_pixel(previous_layout, previous_data, index)[:3]
    camera_hit = primary_hit(triangles, pixel, extent, camera)
    device_z, = read_pixel(FLOAT, hzb, index)
    near, far = camera["znear"], camera["zfar"]
    hzb_depth = near * far / (near + device_z * (far - near)) if device_z > 0.0 else None
    # primary_hit uses a ray with a forward component of one, so its parameter
    # measures camera-space depth here, rather than world-space Euclidean distance.
    triangle_depth_error = abs(hzb_depth - camera_hit["distance"]) if hzb_depth is not None and camera_hit else None
    endpoint_view_depth = dot(subtract(world, camera["eye"]), camera_axes(camera)[0])
    depth_gap = endpoint_view_depth - hzb_depth if hzb_depth is not None else None
    relative_thickness = manifest.get("screen_trace_relative_depth_thickness", 0.02)
    depth_accepts = hzb_depth is not None and depth_gap < relative_thickness * hzb_depth
    color_matches = channel_max_error(rgb, previous_rgb) <= EMITTER_RGB_TOLERANCE
    corroborated_emitter_texel = (color_matches and camera_hit is not None and camera_hit["emissive"]
                                  and triangle_depth_error is not None and triangle_depth_error <= 1.0e-4
                                  and depth_accepts and edge_clearance > 1.0e-3)
    return {"world": world, "ray_parameter_m": ray_parameter, "projected_pixel_center_coords": projection,
            "point_texel": pixel, "point_texel_edge_clearance_px": edge_clearance,
            "previous_rgb": previous_rgb, "previous_rgb_matches_ray": color_matches,
            "texel_camera_triangle_first_hit": camera_hit, "current_hzb_depth_m": hzb_depth,
            "current_hzb_triangle_depth_error_m": triangle_depth_error,
            "endpoint_depth_minus_hzb_depth_m": depth_gap,
            "full_resolution_depth_thickness_accepts": depth_accepts,
            "emitter_texel_corroborated": corroborated_emitter_texel}


def review_unverified_emitters(manifest, root, record, previous_capture, replay, triangles, reference_rgb):
    """Return unresolved candidates keyed by (lane index, global probe index)."""
    capture, ray_capture = record["capture"], record["ray_capture"]
    width, height = manifest["resolution"]
    voxel_size = capture_gdf_voxel_size(manifest)
    probe_size = ((width + PROBE_TILE_EDGE - 1) // PROBE_TILE_EDGE, (height + PROBE_TILE_EDGE - 1) // PROBE_TILE_EDGE)
    support = emitter_filter_support(replay.taps_by_pixel, probe_size)
    distances = read_buffer(root, ray_capture["distance"], FLOAT, len(replay.probe_pixels) * RAY_COUNT)
    trace = read_buffer(root, ray_capture["trace"], TRACE_RECORD, ray_capture["tiles"][2] * ray_capture["tiles"][3]
                        * PROBE_SLOTS * RAY_COUNT)
    hzb = (root / ray_capture["closest_hzb"]).read_bytes()
    if len(hzb) < width * height * FLOAT.size:
        raise ValueError("The closest HZB readback is missing its full-resolution depth plane")
    previous_image = read_pixels(root, previous_capture["realtime"], capture["frame"] - 1, width * height)
    candidates = {}
    reasons = Counter()
    for probe, (x, y) in enumerate(replay.probe_pixels):
        tile, slot = divmod(probe, PROBE_SLOTS)
        if x == INVALID_PROBE or tile not in support:
            continue
        tile_xy = (tile % probe_size[0], tile // probe_size[0])
        receiver = read_pixel(FLOAT4, replay.layers["position"], y * width + x)[:3]
        normal = normalize(read_pixel(HALF4, replay.layers["normal"], y * width + x)[:3])
        screen_origin = floor_screen_origin(receiver, normal, manifest)
        normal_bias = multiply(normal, SURFACE_BIAS)
        for lane, direction in enumerate(replay.directions):
            address = probe * RAY_COUNT + lane
            raw = read_pixel(FLOAT4, replay.layers["raw_radiance"], address)
            rgb, trace_class = raw[:3], raw[3]
            if trace_class not in (GDF_HIT, SCREEN_HIT_CLASS) or channel_max_error(rgb, reference_rgb) > EMITTER_RGB_TOLERANCE:
                continue
            if probe in replay.sources.get(lane, {}):
                continue
            if trace_class == SCREEN_HIT_CLASS:
                origin, tolerance = screen_origin, SCREEN_HIT_DISTANCE_TOLERANCE
            else:
                origin = add(add(receiver, multiply(direction, SURFACE_BIAS)), normal_bias)
                tolerance = voxel_size
            hit = first_hit(triangles, origin, direction, MIN_TRACE_DISTANCE, manifest["view"]["camera"]["zfar"])
            distance, = read_pixel(FLOAT, distances, address)
            error = distance - hit["distance"] if hit else None
            if hit and hit["emissive"] and abs(error) <= tolerance:
                raise ValueError("An unresolved candidate matches the G emitter verification rule")
            reasons[hit["object"] if hit else "miss"] += 1
            sampled = None
            gdf_hit = None
            parameter_error = error
            classification = "emitter_triangle_distance_mismatch" if hit and hit["emissive"] else "non_emitter_or_miss"
            flags = trace_flags(trace, ray_capture, tile_xy, slot, lane)
            if trace_class == SCREEN_HIT_CLASS:
                if flags is not None and not flags & SCREEN_ACCEPTED:
                    raise ValueError("Screen ray class and captured trace flags disagree")
                sampled = infer_screen_sample(manifest, triangles, receiver, origin, direction, distance, rgb, previous_image, hzb)
                parameter_error = sampled["ray_parameter_m"] - hit["distance"] if sampled and hit else None
                if sampled and sampled["emitter_texel_corroborated"] and (not hit or not hit["emissive"]):
                    start_pixel, skipped, after = hit_after_starting_tile(triangles, origin, direction, hit, manifest)
                    sampled["starting_texel"] = start_pixel
                    sampled["triangles_in_starting_texel"] = skipped
                    sampled["triangle_first_hit_after_starting_texel"] = after
                    classification = "screen_emitter_texel_without_emitter_first_hit"
                    if skipped and after and skipped[-1]["front_face"] and not after["front_face"] and after["node_index"] == skipped[-1]["node_index"]:
                        # A starting-tile entry can leave a back-face exit from
                        # the same source volume as the next triangle crossing.
                        after_volume = first_hit(triangles, origin, direction, math.nextafter(after["distance"], math.inf),
                                                 manifest["view"]["camera"]["zfar"])
                        sampled["triangle_first_hit_after_starting_volume_exit"] = after_volume
                        if (after_volume and after_volume["emissive"] and
                                abs(sampled["ray_parameter_m"] - after_volume["distance"]) <= tolerance):
                            classification = "screen_emitter_after_starting_tile_skip"
            else:
                world = add(origin, multiply(direction, distance))
                surfaces = nearest_surfaces(world, direction, triangles)
                gdf_hit = {"world": world, "nearest_triangle_surfaces": dict(list(surfaces.items())[:3])}
            candidates[lane, probe] = {"probe_index": probe, "probe_pixel": (x, y), "probe_tile": tile_xy, "slot": slot,
                                       "lane": (lane % RAY_EDGE, lane // RAY_EDGE), "rgb": rgb, "trace_class": trace_class,
                                       "trace_flags": flags, "trace_distance_m": distance, "ray_origin": origin,
                                       "triangle_first_hit": hit, "distance_minus_triangle_parameter_m": error,
                                       "endpoint_parameter_minus_triangle_m": parameter_error,
                                       "inferred_screen_sample": sampled, "reconstructed_gdf_hit": gdf_hit,
                                       "classification": classification}
    if reasons != replay.emitter_scan["unverified_samples_by_triangle_hit"]:
        raise ValueError("The candidate review disagrees with G's unresolved emitter counts")
    return candidates
