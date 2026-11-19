#language slang 2026
implementing VirtualShadowMaps;

#ifndef VSM_COMPACT_SHADOW_VIEWS
#define VSM_COMPACT_SHADOW_VIEWS 0
#endif

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
    uint row = group.y;
#if VSM_COMPACT_SHADOW_VIEWS
    if ((batchPush.rasterCoarse & kVSMBatchCompactRows) != 0u) {
        StructuredBuffer<uint> rows = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(batchPush.arguments));
        row = rows[batchPush.geometry + row];
    }
#endif
    ShadowBatchJob job = vsmBatchJob(batchPush, row);
    uint localGroup = group.x + batchPush.workBase;
    push = job.push;
    if (localGroup >= vsmBatchJobCount(job)) { return false; }
    group = uint3(localGroup, 0u, 0u);
    return true;
}

groupshared uint gVsmBatchMaxGroups[256];
groupshared uint gVsmBatchCompactRowCount;

[shader("compute")]
[numthreads(256, 1, 1)]
void vsmBatchPrepareMain(uint lane : SV_GroupIndex, uniform VirtualShadowMapBatchPreparePush push)
{
    StructuredBuffer<ShadowBatchJob> jobs = resolveUniform(ResourceHandle<StructuredBuffer<ShadowBatchJob>>(push.jobs));
    RWStructuredBuffer<uint> outputs = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.outputs));
    bool compact = push.mode == kVSMBatchCompactBuild;
    if (compact) {
        if (lane == 0u) { gVsmBatchCompactRowCount = 0u; }
        GroupMemoryBarrierWithGroupSync();
    }
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
            if (compact) {
                if (count <= push.groupBase) { continue; }
                StructuredBuffer<uint> views = resolveNonUniform(ResourceHandle<StructuredBuffer<uint>>(job.push.sampleReserved));
                if (views[kVSMActiveFineViewMask] == 0u) { continue; }
                uint row;
                InterlockedAdd(gVsmBatchCompactRowCount, 1u, row);
                outputs[push.outputWord + 3u + row] = index;
                if (push.collectBuildRows != 0u) {
                    RWStructuredBuffer<uint> diagnostics = resolveNonUniform(ResourceHandle<RWStructuredBuffer<uint>>(job.push.sampleReserved));
                    InterlockedAdd(diagnostics[kVSMActiveInitialBuildRows], 1u);
                }
            }
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
        outputs[push.outputWord + 1u] = compact ? gVsmBatchCompactRowCount : push.jobCount;
        outputs[push.outputWord + 2u] = 1u;
    }
}
