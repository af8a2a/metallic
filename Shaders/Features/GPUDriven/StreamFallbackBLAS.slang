module StreamFallbackBLAS;

import ShaderCore;
using Metallic;

struct FallbackPush
{
    uint metadata;
    uint pages;
    uint clasPages;
    uint clasAddresses;
    uint references;
    uint infos;
    uint destinations;
    uint count;
    uint selected;
    uint published;
    uint primitiveCount;
    uint phase;
    uint64_t referenceAddress;
    uint64_t storageAddress;
};

groupshared uint outputIndex;

[shader("compute")]
[numthreads(64, 1, 1)]
void streamFallbackBLASMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex, uniform FallbackPush push)
{
    RWStructuredBuffer<uint> count = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.count));
    if (push.phase == 0u) {
        if (lane == 0u) { count[0] = 0u; }
        return;
    }
    uint index = group.x + group.y * 65535u;
    if (index >= push.primitiveCount) { return; }
    StructuredBuffer<uint4> metadata = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.metadata));
    RWStructuredBuffer<uint> selected = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.selected));
    RWStructuredBuffer<uint2> published = resolveUniform(ResourceHandle<RWStructuredBuffer<uint2>>(push.published));
    if (push.phase == 2u) {
        if (lane == 0u && index < count[0]) {
            uint source = selected[index];
            uint4 a = metadata[source * 2u];
            uint4 b = metadata[source * 2u + 1u];
            uint64_t address = push.storageAddress + (uint64_t(b.y) << 32u | b.x);
            published[a.x] = uint2(uint(address), uint(address >> 32u));
        }
        return;
    }
    uint4 a = metadata[index * 2u]; // primitive, reference count, reference offset lo/hi
    uint4 b = metadata[index * 2u + 1u]; // storage offset lo/hi, first page, page count
    StructuredBuffer<uint2> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint2>>(push.pages));
    StructuredBuffer<uint2> clasPages = resolveUniform(ResourceHandle<StructuredBuffer<uint2>>(push.clasPages));
    if (lane == 0u) {
        outputIndex = 0xffffffffu;
        bool ready = all(published[a.x] == uint2(0u, 0u));
        for (uint i = 0u; i < b.w && ready; ++i) {
            ready = (clasPages[pages[b.z + i].x].x >> 30u) == 1u;
        }
        if (ready) {
            InterlockedAdd(count[0], 1u, outputIndex);
            selected[outputIndex] = index;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (outputIndex == 0xffffffffu) { return; }
    StructuredBuffer<uint2> addresses = resolveUniform(ResourceHandle<StructuredBuffer<uint2>>(push.clasAddresses));
    RWStructuredBuffer<uint2> references = resolveUniform(ResourceHandle<RWStructuredBuffer<uint2>>(push.references));
    uint offset = a.z; // CPU initialization validates the reference table fits 32-bit indexing.
    for (uint i = 0u; i < b.w; ++i) {
        uint2 page = pages[b.z + i];
        uint source = clasPages[page.x].x & 0x3fffffffu;
        for (uint cluster = lane; cluster < page.y; cluster += 64u) {
            references[offset + cluster] = addresses[source + cluster];
        }
        offset += page.y;
    }
    if (lane == 0u) {
        RWStructuredBuffer<uint4> infos = resolveUniform(ResourceHandle<RWStructuredBuffer<uint4>>(push.infos));
        RWStructuredBuffer<uint2> destinations = resolveUniform(ResourceHandle<RWStructuredBuffer<uint2>>(push.destinations));
        uint64_t reference = push.referenceAddress + (uint64_t(a.w) << 32u | a.z) * 8u;
        uint64_t destination = push.storageAddress + (uint64_t(b.y) << 32u | b.x);
        infos[outputIndex] = uint4(a.y, 8u, uint(reference), uint(reference >> 32u));
        destinations[outputIndex] = uint2(uint(destination), uint(destination >> 32u));
    }
}
