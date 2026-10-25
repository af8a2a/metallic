"""Strict semantic decoder for the current streamed MiniZorah diagnostic ABI."""
import hashlib
import json
from pathlib import Path

import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def rows_hash(values):
    return digest(b"".join(sorted(row.tobytes() for row in values)))


def safe_file(directory, name):
    path = (directory / name).resolve()
    require(path.is_relative_to(directory.resolve()), "Capture path escapes directory")
    return path


def decode(directory, state):
    directory = Path(directory)
    metadata = read(directory / "Buffers.json")
    data, meta = {}, {}
    for item in metadata:
        raw = safe_file(directory, item["file"]).read_bytes()
        require(len(raw) == item["bytes"], "Buffer length mismatch")
        name = item["resource"].rsplit(".", 1)[-1]
        require(name not in data, "Duplicate buffer")
        data[name], meta[name] = raw, item
    require(len(data) == 14, "Expected fourteen diagnostic buffers")
    u32 = lambda name: np.frombuffer(data[name], "<u4")
    active, rt, request, demand = (u32(k) for k in ("activeHeader", "rtRepresentationHeader", "requestHeader", "demandStats"))
    require(active[3] == rt[3] == 0 and all(request[i] == 0 for i in (5, 6, 7)), "Overflow/invalid request")
    require(demand[17] == 1, "Distributed demand was not enabled")
    demand_metadata = meta["demandStats"]["metadata"]
    require(demand_metadata["distributedDemandActive"] and demand[17] == 1, "Distributed demand metadata/counter mismatch")
    shader = demand_metadata["demandShaderIdentity"]
    require(shader["entryPoint"] == "streamDistributedDemandMain" and shader["spirvHash"], "Missing bound demand shader identity")
    pages, clas = u32("pageTable").reshape(-1, 2), u32("clasPageTable").reshape(-1, 2)
    groups = u32("activeGroups").reshape(-1, 28)[:int(active[0])].copy()
    require(len(groups) == active[0] and np.all(groups[:, 1] < len(pages)), "Invalid active range")
    require(np.array_equal(groups[:, 0], pages[groups[:, 1], 0] & np.uint32(0xfffffff8)), "Active page mapping mismatch")
    groups[:, 0] = 0  # Physical residency address only; preserve all semantic cut fields.
    m = meta["rtRepresentationState"]["metadata"]
    primitive_count, instance_count = m["scenePrimitiveCount"], m["sceneInstanceCount"]
    offset, storage = m["rtInstanceStateWordOffset"], m["blasStorageAddress"]
    instances = u32("instances").reshape(-1, 24)[:instance_count]
    tlas = u32("tlasInstances").reshape(-1, 16)[:instance_count]
    owners = u32("rtRepresentations").reshape(-1, 16)
    selected = u32("rtRepresentationState")[offset + 2:offset + instance_count * 4:4]
    fallback = np.frombuffer(data["fallbackBlasAddresses"], "<u8")
    require(len(instances) == len(tlas) == len(selected) == instance_count, "Missing instance range")
    require(len(owners) >= primitive_count * 3 and len(fallback) >= primitive_count, "Missing owner range")
    require(np.all(instances[:, 0] < primitive_count), "Invalid primitive")
    valid = selected != np.uint32(0xffffffff)
    require(np.all(selected[valid] < primitive_count * 3), "Invalid owner")
    info = np.zeros((instance_count, 16), dtype=np.uint32)
    info[valid] = owners[selected[valid]]
    ready = ((info[:, 5] & 2) != 0) & ((info[:, 5] & 5) == 0) & (info[:, 2] != 0) & (info[:, 3] == info[:, 2]) & (info[:, 15] == 1) & (info[:, 11] != 0)
    expected = fallback[instances[:, 0]].copy()
    expected[ready] = np.uint64(storage) + info[ready, 10].astype(np.uint64)
    actual = tlas[:, 14].astype(np.uint64) | (tlas[:, 15].astype(np.uint64) << np.uint64(32))
    require(np.array_equal(actual, expected), "TLAS physical pointer does not match selected owner/fallback")
    require(np.all(selected[ready] // 3 == instances[ready, 0]), "TLAS owner belongs to another primitive")
    effective = np.column_stack((tlas[:, :14], ready.astype(np.uint32), np.where(ready, selected, instances[:, 0])))
    texture = state["diagnosticTextureSemanticIdentity"]
    require(texture and texture["schema"] == "published-texture-source-mips-v1" and all(s["uploaded"] for s in texture["slots"]), "Incomplete semantic texture source")
    requests = u32("loadRequests")[:min(int(request[2]), int(request[0]))]
    semantics = dict(pageStates=digest((pages[:, 0] & 7).tobytes()), clasStates=digest((clas[:, 0] >> 30).tobytes()),
                     activeCut=rows_hash(groups), orderedActiveCut=digest(groups.tobytes()),
                     rtCut=rows_hash(u32("rtRepresentationGroups").reshape(-1, 4)[:int(rt[0])]),
                     orderedRTCut=digest(u32("rtRepresentationGroups").reshape(-1, 4)[:int(rt[0])].tobytes()),
                     effectiveTlas=digest(effective.tobytes()), instances=digest(instances.tobytes()),
                     loadRequests=digest(np.sort(requests).tobytes()), activeCount=int(active[0]), rtCount=int(rt[0]),
                     requestCount=int(request[2]), textureSemantic=canonical(texture))
    counts = {key: int(demand[index]) for key, index in dict(processedTasks=1, visitedNodes=2, testedGroups=3,
              nonemptyTasks=6, workerSlots=7, seededTasks=16, distributed=17, allGroupTests=18).items()}
    return {"semantics": semantics, "demand": counts, "demandShaderIdentity": shader, "actualTlasPointersValidated": True}


def images(directory):
    directory = Path(directory)
    result = {}
    for item in read(directory / "Images.json"):
        kind, channels = item["format"].rsplit("x", 1)
        dtype = {"float16": "<f2", "float32": "<f4", "unorm8": "u1", "srgb8": "u1", "uint32": "<u4"}[kind]
        raw = safe_file(directory, item["file"]).read_bytes()
        array = np.frombuffer(raw, dtype)
        require(array.size == item["width"] * item["height"] * int(channels), "Image size mismatch")
        require(np.isfinite(array).all(), "Nonfinite image")
        result[item["resource"]] = {"metadata": item, "sha256": digest(raw), "array": array}
    require({"Deferred.color", "FinalBlit.color", "ReflectionQuality.color", "Shadows.shadow", "VBuffer.depth", "VBuffer.visibility"} <= result.keys(), "Missing quality image")
    require(result["Deferred.color"]["metadata"]["format"] == "float32x4", "HDR correctness requires RGBA32F")
    return result


def compare_images(reference, candidate):
    require(reference.keys() == candidate.keys(), "Image resource sets differ")
    result = {}
    for name, a in reference.items():
        b = candidate[name]
        require(a["metadata"] == b["metadata"], "Image layout changed")
        delta = b["array"].astype(np.float64) - a["array"].astype(np.float64)
        result[name] = {"exact": a["sha256"] == b["sha256"], "maxAbs": float(np.abs(delta).max()),
                        "rmse": float(np.sqrt(np.mean(delta * delta))), "differentValues": int(np.count_nonzero(delta))}
    return result
