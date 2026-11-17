#language slang 2026
implementing VirtualShadowMaps;

uint vsmBatchJobCount(ShadowBatchJob job)
{
    if (job.countBuffer == 0xffffffffu) { return job.groupCount; }
    StructuredBuffer<uint> counts = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(job.countBuffer));
    return counts[job.countWord];
}

ShadowBatchJob vsmBatchJob(ShadowPush batchPush, uint index)
{
    StructuredBuffer<ShadowBatchJob> jobs = resolveUniform(ResourceHandle<StructuredBuffer<ShadowBatchJob>>(batchPush.physicalPool));
    return jobs[batchPush.poolFace + index];
}

// A row owns exactly one face. Both the guard and resource handles remain
// group-uniform, so every participating lane reaches the original barriers.
bool vsmBatchComputePush(inout uint3 group, ShadowPush batchPush, out ShadowPush push)
{
    ShadowBatchJob job = vsmBatchJob(batchPush, group.y);
    uint localGroup = group.x + batchPush.workBase;
    push = job.push;
    if (localGroup >= vsmBatchJobCount(job)) { return false; }
    group = uint3(localGroup, 0u, 0u);
    return true;
}

groupshared uint gVsmBatchMaxGroups[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void vsmBatchPrepareMain(uint lane : SV_GroupIndex, uniform VirtualShadowMapBatchPreparePush push)
{
    StructuredBuffer<ShadowBatchJob> jobs = resolveUniform(ResourceHandle<StructuredBuffer<ShadowBatchJob>>(push.jobs));
    RWStructuredBuffer<uint> outputs = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.outputs));
    uint maximum = 0u;
    for (uint index = lane; index < push.jobCount; index += 256u) {
        ShadowBatchJob job = jobs[push.firstJob + index];
        uint count = job.groupCount;
        if (job.countBuffer != 0xffffffffu) {
            // Preparation assigns different faces to lanes of the same wave.
            StructuredBuffer<uint> counts = resolveNonUniform(ResourceHandle<StructuredBuffer<uint>>(job.countBuffer));
            count = counts[job.countWord];
        }
        if (push.mode == 1u) {
            uint offset = push.outputWord + index * 3u;
            outputs[offset] = count;
            outputs[offset + 1u] = 1u;
            outputs[offset + 2u] = 1u;
        } else {
            maximum = max(maximum, count);
        }
    }
    if (push.mode == 1u) { return; }
    gVsmBatchMaxGroups[lane] = maximum;
    for (uint stride = 128u; stride != 0u; stride >>= 1u) {
        GroupMemoryBarrierWithGroupSync();
        if (lane < stride) {
            gVsmBatchMaxGroups[lane] = max(gVsmBatchMaxGroups[lane], gVsmBatchMaxGroups[lane + stride]);
        }
    }
    if (lane == 0u) {
        uint remaining = gVsmBatchMaxGroups[0] - min(gVsmBatchMaxGroups[0], push.groupBase);
        outputs[push.outputWord] = min(remaining, 65535u);
        outputs[push.outputWord + 1u] = push.jobCount;
        outputs[push.outputWord + 2u] = 1u;
    }
}
