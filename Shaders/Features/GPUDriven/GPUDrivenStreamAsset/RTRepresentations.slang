#language slang 2026
implementing GPUDrivenStreamAsset;

import ShaderCore;
using Metallic;

import GPUDriven;
using Metallic.GPUDriven;

// RT representation selection, dependency validation and allocation publication.
void buildStreamRTRepresentations(uint item, uint groupRow, GPUDrivenStreamAssetPush push,
    GPUDrivenStreamAssetParams params)
{
    if ((push.activeBuildPhase >= 35u && push.activeBuildPhase <= 37u) || push.activeBuildPhase == 41u) {
        sortStreamRTOwners(item + groupRow * (65535u * 64u), push, params);
        return;
    }
    if (push.activeBuildPhase == 38u || push.activeBuildPhase == 40u) {
        item += groupRow * (65535u * 64u);
    }
    uint lane = item & 63u;
    if (push.activeBuildPhase == 22u || push.activeBuildPhase == 25u || push.activeBuildPhase == 27u) {
        item = (item >> 6u) + groupRow * 65535u;
    }
    RWStructuredBuffer<uint> state = resolveUniform(
        ResourceHandle<RWStructuredBuffer<uint>>(params.rtRepresentationStateBuffer));
    RWStructuredBuffer<uint> cutCache = resolveUniform(
        ResourceHandle<RWStructuredBuffer<uint>>(params.rtRepresentationCacheBuffer));
    StructuredBuffer<GPUDrivenStreamAssetInstance> instances = resolveUniform(
        ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetInstance>>(push.instanceBuffer));
    StructuredBuffer<GPUDrivenStreamAssetPrimitive> primitives = resolveUniform(
        ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetPrimitive>>(push.primitiveBuffer));
    StructuredBuffer<GPUDrivenStreamAssetGroup> groups = resolveUniform(
        ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetGroup>>(push.groupBuffer));
    StructuredBuffer<uint> topology = resolveUniform(
        ResourceHandle<StructuredBuffer<uint>>(params.lodTopologyBuffer));
    StructuredBuffer<uint> lodState = resolveUniform(
        ResourceHandle<StructuredBuffer<uint>>(params.lodStateBuffer));
    RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTable = resolveUniform(
        ResourceHandle<RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry>>(push.pageTableBuffer));
    RWStructuredBuffer<GPUDrivenStreamAssetActiveHeader> rtHeader = resolveUniform(
        ResourceHandle<RWStructuredBuffer<GPUDrivenStreamAssetActiveHeader>>(params.rtRepresentationHeaderBuffer));
    RWStructuredBuffer<GPUDrivenStreamAssetBLASHeader> header = resolveUniform(
        ResourceHandle<RWStructuredBuffer<GPUDrivenStreamAssetBLASHeader>>(push.blasHeaderBuffer));

    // Primitive words: representative, shared maximum LOD, cached LOD,
    // consumer counts (merged/shared/cached), group count, group offset,
    // persistent semantic revisions (three), published cache LOD + frame flags.
    // Instance words: conservative minimum LOD, actual cut maximum LOD,
    // desired representation, reserved. Histograms use four words per LOD.
    if (push.activeBuildPhase == 20u) {
        if (item < params.sceneLodLevelCount) {
            uint base = streamRTHistogramBase(params) + item * 4u;
            state[base] = 0u; state[base + 1u] = 0u;
            state[base + 2u] = 0xffffffffu; state[base + 3u] = 0u;
        }
        if (item < params.scenePrimitiveCount) {
            uint base = streamRTPrimitiveBase(params, item);
            for (uint mode = 0u; mode < 3u; ++mode) {
                uint importance = streamRTOwnerImportanceBase(params, item * 3u + mode);
                state[importance] = 0u;
                state[importance + 1u] = 0x7f7fffffu;
                state[importance + 2u] = 0u;
                state[importance + 3u] = 0u;
                state[importance + 4u] = 0u;
                state[importance + 5u] = 0u;
            }
            // Each primitive owns 16 cache words: merged epochs/counts (4),
            // discrete epochs/counts (4), discrete LOD, shared owner/counts (3),
            // shared valid, current shared difference/count, frame reuse bits.
            uint cacheBase = item * 16u;
            if ((push.traversalPhase & 3u) != 0u) {
                for (uint word = 0u; word < 16u; ++word) { cutCache[cacheBase + word] = 0u; }
            } else {
                cutCache[cacheBase + 13u] = 0u;
                cutCache[cacheBase + 14u] = 0u;
                cutCache[cacheBase + 15u] = 0u;
            }
            for (uint word = 0u; word < 8u; ++word) { state[base + word] = 0u; }
            state[base] = 0xffffffffu; state[base + 1u] = 0xffffffffu;
            state[base + 2u] = 0xffffffffu;
            state[base + 11u] &= kRTCachedLevelMask;
            if ((push.traversalPhase & 1u) != 0u) {
                state[base + 8u] = 0u; state[base + 9u] = 0u; state[base + 10u] = 0u;
                state[base + 11u] = 0u;
            }
        }
        if (item == 0u) {
            uint priorityControl = streamRTOwnerPriorityControlBase(params);
            state[priorityControl] = (push.traversalPhase & 1u) != 0u ? 1u : 0u;
            state[priorityControl + 1u] = 0u;
            state[priorityControl + 2u] = 0xffffffffu;
            // Feedback can lag the captured GPU state. Retain the latest
            // mismatch through hits so its input transition stays inspectable.
            if ((push.traversalPhase & 1u) != 0u) {
                state[priorityControl + 3u] = 0u;
                state[priorityControl + 4u] = 0u;
                state[priorityControl + 5u] = 0u;
                state[priorityControl + 6u] = 0xffffffffu;
            }
            GPUDrivenStreamAssetActiveHeader value = (GPUDrivenStreamAssetActiveHeader)0;
            value.activeGroupCapacity = params.rtRepresentationGroupCapacity;
            value.maxActiveGroupClusters = 32u; value.frameIndex = params.frameIndex;
            rtHeader[0] = value;
            header[0].representationCount = 0u;
            header[0].sharingInstanceCount = 0u;
            header[0].mergedInstanceCount = 0u;
            header[0].cachedInstanceCount = 0u;
            header[0].representationGroupCount = 0u;
            header[0].representationGroupOverflow = 0u;
            header[0].cacheDependencyInvalidations = 0u;
        }
        return;
    }
    if (push.activeBuildPhase == 21u) {
        if (item >= params.sceneInstanceCount) { return; }
        GPUDrivenStreamAssetInstance instance = instances[item];
        uint output = streamRTInstanceBase(params, item);
        state[output] = 0u; state[output + 1u] = 0xffffffffu;
        state[output + 2u] = 0xffffffffu; state[output + 3u] = 0u;
        if (instance.visible == 0u || instance.primitiveIndex >= params.scenePrimitiveCount) { return; }
        GPUDrivenStreamAssetPrimitive primitive = primitives[instance.primitiveIndex];
        if (primitive.lodLevelCount == 0u) { return; }
        StructuredBuffer<float4> metadata = resolveUniform(
            ResourceHandle<StructuredBuffer<float4>>(params.rtRepresentationMetadataBuffer));
        float3 forward = safeNormalize(params.center.xyz - params.eye.xyz, float3(0, 0, -1));
        float4 eye = float4(params.eye.xyz, params.clipOrtho.x);
        float4 direction = float4(forward, params.upProjection.w);
        float4 projection = float4(params.viewport.z, tan(params.viewport.w * 0.5),
            params.clipOrtho.z, params.lodPixelError);
        float threshold = params.lodPixelError * (streamRayLodEnabled(params)
            ? 1.0 - clamp(params.rtLodHysteresis, 0.0, 0.9) : 1.0);
        uint minimum = 0u;
        if (params.enableGpuLodSelection == 0u && params.selectedLodLevel != kNoDebugLODOverride) {
            minimum = min(params.selectedLodLevel, primitive.lodLevelCount - 1u);
        } else {
            // An enclosing sphere and maximum error bound every group's metric
            // at this level, including shear, the raster guard and hysteresis.
            // Never advance across an uncertain/non-finite upper bound.
            for (uint level = 0u; level + 1u < primitive.lodLevelCount; ++level) {
                uint metadataIndex = (primitive.lodLevelOffset + level) * 2u;
                float4 sphere = metadata[metadataIndex];
                float3 extent = abs(sphere.xyz) + sphere.www;
                float3 magnitude = abs(instance.world0.xyz) * extent.x + abs(instance.world1.xyz) * extent.y +
                    abs(instance.world2.xyz) * extent.z + abs(instance.world3.xyz) + abs(params.eye.xyz);
                float padding = dot(magnitude, float3(1, 1, 1)) * (32.0 * 1.1920928955078125e-7);
                float upper = streamLodMetric(sphere, metadata[metadataIndex + 1u].x,
                    instance, eye, direction, projection, params, true, padding);
                if (!(upper < threshold - max(abs(threshold) * 1e-4, 1e-5))) { break; }
                minimum = level + 1u;
            }
        }
        bool hasCut = lodState[item * 4u + 3u] != 0u && lodState[item * 4u] != 0u;
        state[output] = minimum;
        state[output + 1u] = hasCut ? 0u : 0xffffffffu;
        return;
    }
    if (push.activeBuildPhase == 28u) {
        StructuredBuffer<GPUDrivenStreamAssetActiveHeader> rasterHeader = resolveUniform(
            ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetActiveHeader>>(push.activeHeaderBuffer));
        uint count = min(rasterHeader[0].activeGroupCount, rasterHeader[0].activeGroupCapacity);
        if (item >= count) { return; }
        StructuredBuffer<GPUDrivenStreamAssetActiveGroup> rasterGroups = resolveUniform(
            ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetActiveGroup>>(push.activeGroupBuffer));
        GPUDrivenStreamAssetActiveGroup group = rasterGroups[item];
        if (group.instanceIndex < params.sceneInstanceCount && group.clusterSelectionMask != 0u) {
            InterlockedMax(state[streamRTInstanceBase(params, group.instanceIndex) + 1u], group.lodLevel);
            uint primitiveIndex = group.primitiveIndex;
            uint cacheBase = primitiveIndex * 16u;
            if (cutCache[cacheBase + 12u] != 0u && cutCache[cacheBase + 9u] == group.instanceIndex) {
                GPUDrivenStreamAssetPrimitive primitive = primitives[primitiveIndex];
                uint groupIndex = primitive.groupOffset + group.pageIndex - primitive.pageOffset;
                if (state[params.sceneGroupCount + groupIndex] != group.clusterSelectionMask) {
                    InterlockedOr(cutCache[cacheBase + 13u], 1u);
                }
                // The frontier emits each group at most once. Equality of the
                // count plus every current mask also detects removed groups.
                InterlockedAdd(cutCache[cacheBase + 14u], 1u);
            }
        }
        return;
    }
    if (push.activeBuildPhase == 29u) {
        if (item >= params.sceneInstanceCount) { return; }
        GPUDrivenStreamAssetInstance instance = instances[item];
        if (instance.visible == 0u || instance.primitiveIndex >= params.scenePrimitiveCount) { return; }
        GPUDrivenStreamAssetPrimitive primitive = primitives[instance.primitiveIndex];
        if (primitive.lodLevelCount == 0u) { return; }
        uint output = streamRTInstanceBase(params, item);
        uint minimum = state[output], maximum = state[output + 1u];
        uint histogram = streamRTHistogramBase(params) + (primitive.lodLevelOffset + minimum) * 4u;
        InterlockedAdd(state[histogram], 1u);
        if (maximum != 0xffffffffu && minimum < primitive.lodLevelCount - 1u) {
            histogram = streamRTHistogramBase(params) + (primitive.lodLevelOffset + maximum) * 4u;
            InterlockedAdd(state[histogram + 1u], 1u);
            InterlockedMin(state[histogram + 2u], item);
        }
        return;
    }
    if (push.activeBuildPhase == 22u) {
        if (item >= params.scenePrimitiveCount) { return; }
        GPUDrivenStreamAssetPrimitive primitive = primitives[item];
        uint base = streamRTPrimitiveBase(params, item);
        if (lane == 0u) {
            uint covered = 0u, bestCovered = 0u;
            for (uint remaining = primitive.lodLevelCount; remaining != 0u; --remaining) {
                uint level = remaining - 1u;
                uint histogram = streamRTHistogramBase(params) + (primitive.lodLevelOffset + level) * 4u;
                covered += state[histogram];
                if (state[histogram + 1u] != 0u && covered > bestCovered && covered > 1u) {
                    state[base] = state[histogram + 2u];
                    state[base + 1u] = level; bestCovered = covered;
                }
            }
            g_RTCacheMissing = covered == 0u && (state[base + 11u] & kRTCachedLevelMask) == 0u
                ? 0x80000000u : 0u;
        }
        GroupMemoryBarrierWithGroupSync();
        if (g_RTCacheMissing == 0x80000000u) { return; }
        uint firstCacheLevel = primitive.lodLevelCount > 3u ? primitive.lodLevelCount - 3u : 0u;
        uint missing = 0u;
        for (uint local = lane; local < primitive.groupCount; local += 64u) {
            GPUDrivenStreamAssetGroup group = groups[primitive.groupOffset + local];
            if (group.lodLevel < firstCacheLevel && (group.flags & 1u) == 0u) { continue; }
            if (groupPageDrawable(pageTable, params, group)) { continue; }
            for (uint level = firstCacheLevel; level + 1u < primitive.lodLevelCount; ++level) {
                if (group.lodLevel >= level || (group.flags & 1u) != 0u) {
                    missing |= 1u << (level - firstCacheLevel);
                }
            }
        }
        InterlockedOr(g_RTCacheMissing, missing);
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0u) {
            for (uint level = firstCacheLevel; level + 1u < primitive.lodLevelCount; ++level) {
                if ((g_RTCacheMissing & (1u << (level - firstCacheLevel))) == 0u) {
                    state[base + 2u] = level;
                    break;
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0u) {
            // The fourth histogram word is classification scratch until phase
            // 25 clears it. Decide before phase 23
            // atomically aggregates consumers, so no instance races the choice.
            uint canonicalMerged = 0u;
            uint threshold = min(primitive.lodLevelCount - 1u,
                min(state[base + 1u], state[base + 2u]));
            uint mergedConsumers = 0u;
            for (uint level = 0u; level < threshold; ++level) {
                uint histogram = streamRTHistogramBase(params) + (primitive.lodLevelOffset + level) * 4u;
                mergedConsumers += state[histogram];
            }
            // The representative is allowed its own cut even if its LOD
            // interval spans the sharing threshold; it is not merged demand.
            uint representative = state[base];
            if (representative != 0xffffffffu &&
                state[streamRTInstanceBase(params, representative)] < threshold) {
                --mergedConsumers;
            }
            canonicalMerged = mergedConsumers != 0u ? 1u : 0u;
            state[streamRTHistogramBase(params) + primitive.lodLevelOffset * 4u + 3u] = canonicalMerged;
        }
        return;
    }
    if (push.activeBuildPhase == 23u) {
        if (item >= params.sceneInstanceCount) { return; }
        GPUDrivenStreamAssetInstance instance = instances[item];
        if (instance.visible == 0u || instance.primitiveIndex >= params.scenePrimitiveCount) { return; }
        GPUDrivenStreamAssetPrimitive primitive = primitives[instance.primitiveIndex];
        uint output = streamRTInstanceBase(params, item);
        uint base = streamRTPrimitiveBase(params, instance.primitiveIndex);
        uint minimum = state[output];
        if (primitive.lodLevelCount == 0u || minimum >= primitive.lodLevelCount - 1u) { return; }
        uint mode = minimum >= state[base + 2u] ? 2u : minimum >= state[base + 1u] ? 1u : 0u;
        // The representative itself retains its own view cut even when its
        // interval spans several levels. Other users satisfy the conservative test.
        if (mode == 0u && item == state[base]) { mode = 1u; }
        // A demanded highest-drawable cut can serve every consumer of the same
        // primitive. Charge and prioritize that owner once rather than retaining
        // redundant shared/cache builds while other primitives use terminal cuts.
        if (state[streamRTHistogramBase(params) + primitive.lodLevelOffset * 4u + 3u] != 0u) {
            mode = 0u;
        }
        uint owner = instance.primitiveIndex * 3u + mode;
        state[output + 2u] = owner;
        InterlockedAdd(state[base + 3u + mode], 1u);
        streamAccumulateRTOwnerImportance(state, owner, instance, params);
        return;
    }
    if (push.activeBuildPhase == 25u) {
        if (item >= params.scenePrimitiveCount) { return; }
        GPUDrivenStreamAssetPrimitive primitive = primitives[item];
        uint base = streamRTPrimitiveBase(params, item);
        StructuredBuffer<GPUDrivenStreamAssetLODLevel> levels = resolveUniform(
            ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetLODLevel>>(push.lodLevelBuffer));
        uint total = 0u, references = 0u;
        uint cachedReferences = 0u;
        uint cacheBase = item * 16u;
        bool scratchWritten = false;
        // Mode 1 is temporary reachability scratch until merged/cache masks
        // have been computed. Shared masks are copied last. Readers never race
        // a writer replacing an active bit with its complementary cluster mask.
        for (uint step = 0u; step < 3u; ++step) {
            uint mode = step == 0u ? 0u : step == 1u ? 2u : 1u;
            uint stateBase = mode * params.sceneGroupCount + primitive.groupOffset;
            uint scratchBase = params.sceneGroupCount + primitive.groupOffset;
            if (state[base + 3u + mode] == 0u) {
                if (mode != 2u || (state[base + 11u] & kRTCachedLevelMask) == 0u) { continue; }
                StructuredBuffer<GPUDrivenStreamAssetInstanceBLAS> representations = resolveUniform(
                    ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetInstanceBLAS>>(params.rtRepresentationBuffer));
                if (representations[item * 3u + 2u].cachedValid != 1u) { continue; }
                uint cachedLevel = (state[base + 11u] & kRTCachedLevelMask) - 1u;
                bool unchanged = cutCache[cacheBase + 4u] == push.pageTableMutationSerial &&
                    cutCache[cacheBase + 5u] == push.clasPublicationRevision &&
                    cutCache[cacheBase + 8u] == cachedLevel + 1u;
                if (!unchanged) {
                    if (lane == 0u) { g_RTCacheMissing = 0u; }
                    GroupMemoryBarrierWithGroupSync();
                    uint missing = 0u;
                    for (uint local = lane; local < primitive.groupCount; local += 64u) {
                        GPUDrivenStreamAssetGroup group = groups[primitive.groupOffset + local];
                        if ((group.lodLevel >= cachedLevel || (group.flags & 1u) != 0u) &&
                            !groupPageDrawable(pageTable, params, group)) { missing = 1u; }
                    }
                    InterlockedOr(g_RTCacheMissing, missing);
                    GroupMemoryBarrierWithGroupSync();
                    if (g_RTCacheMissing != 0u) { continue; }
                }
                if (lane == 0u) {
                    state[base + 2u] = cachedLevel;
                    state[base + 11u] |= kRTIdleCache;
                }
                AllMemoryBarrierWithGroupSync();
            }
            uint count = 0u, clusters = 0u;
            uint epochBase = cacheBase + (mode == 2u ? 4u : 0u);
            // Residency/CLAS publication epochs cover every drawable dependency;
            // metadata and masks are published by the same submitted dispatch.
            // Shared cuts instead compare this frame's real frontier exactly.
            // A merged/discrete miss overwrites shared scratch, forcing a copy
            // even when the representative's semantic cut itself is unchanged.
            bool reuse = mode == 1u ? !scratchWritten && cutCache[cacheBase + 12u] != 0u &&
                    cutCache[cacheBase + 9u] == state[base] && cutCache[cacheBase + 13u] == 0u &&
                    cutCache[cacheBase + 14u] == cutCache[cacheBase + 10u]
                : cutCache[epochBase] != 0u && cutCache[epochBase] == push.pageTableMutationSerial &&
                    cutCache[epochBase + 1u] == push.clasPublicationRevision &&
                    (mode == 0u || cutCache[cacheBase + 8u] == state[base + 2u] + 1u);
            uint2 totals;
            if (reuse) {
                totals = mode == 1u ? uint2(cutCache[cacheBase + 10u], cutCache[cacheBase + 11u])
                    : uint2(cutCache[epochBase + 2u], cutCache[epochBase + 3u]);
                if (lane == 0u) { cutCache[cacheBase + 15u] |= 1u << mode; }
            } else {
                if (mode == 1u) {
                    uint source = topology[params.lodInstanceOffsetsOffset + state[base]];
                    for (uint local = lane; local < primitive.groupCount; local += 64u) {
                        uint mask = lodState[source + local * 2u + 1u];
                        state[stateBase + local] = mask;
                        count += mask != 0u ? 1u : 0u;
                        clusters += countbits(mask);
                    }
                } else {
                    scratchWritten = true;
                    if (lane == 0u) { g_RTCacheMissing = 0u; }
                    GroupMemoryBarrierWithGroupSync();
                    // Stream assets own one page per group, in identical per-level
                    // order. Parents are strictly coarser, so one device barrier
                    // per LOD level makes all parents visible to the next level.
                    for (uint remaining = primitive.lodLevelCount; remaining != 0u; --remaining) {
                        GPUDrivenStreamAssetLODLevel level = levels[primitive.lodLevelOffset + remaining - 1u];
                        uint first = level.pageOffset - primitive.pageOffset;
                        for (uint child = lane; child < level.pageCount; child += 64u) {
                            uint local = first + child;
                            GPUDrivenStreamAssetGroup group = groups[primitive.groupOffset + local];
                            bool terminal = (group.flags & 1u) != 0u;
                            bool active = mode == 0u || terminal || group.lodLevel >= state[base + 2u];
                            active = active && groupPageDrawable(pageTable, params, group);
                            if (terminal && !active) { InterlockedOr(g_RTCacheMissing, 1u); }
                            for (uint parent = 0u; parent < group.parentCount; ++parent) {
                                active = active && state[params.sceneGroupCount + topology[group.parentOffset + parent]] != 0u;
                            }
                            state[scratchBase + local] = active ? 1u : 0u;
                        }
                        AllMemoryBarrierWithGroupSync();
                    }
                    for (uint local = lane; local < primitive.groupCount; local += 64u) {
                        GPUDrivenStreamAssetGroup group = groups[primitive.groupOffset + local];
                        uint mask = 0u;
                        if (g_RTCacheMissing == 0u && state[scratchBase + local] != 0u) {
                            for (uint cluster = 0u; cluster < group.clusterCount; ++cluster) {
                                uint refined = topology[group.clusterRefinedOffset + cluster];
                                if (refined == kInvalidClusterIndex || state[params.sceneGroupCount + refined] == 0u) {
                                    mask |= 1u << cluster;
                                }
                            }
                        }
                        state[stateBase + local] = mask;
                        count += mask != 0u ? 1u : 0u;
                        clusters += countbits(mask);
                    }
                }
                AllMemoryBarrierWithGroupSync();
                scanRTRepresentationCounts(uint4(count, clusters, 0u, 0u), lane);
                totals = g_StreamBlasStoragePrefix[63].xy;
                GroupMemoryBarrierWithGroupSync();
                if (lane == 0u) {
                    if (mode == 1u) {
                        cutCache[cacheBase + 9u] = state[base];
                        cutCache[cacheBase + 10u] = totals.x;
                        cutCache[cacheBase + 11u] = totals.y;
                        cutCache[cacheBase + 12u] = 1u;
                    } else {
                        cutCache[epochBase] = push.pageTableMutationSerial;
                        cutCache[epochBase + 1u] = push.clasPublicationRevision;
                        cutCache[epochBase + 2u] = totals.x;
                        cutCache[epochBase + 3u] = totals.y;
                        if (mode == 2u) { cutCache[cacheBase + 8u] = state[base + 2u] + 1u; }
                    }
                }
            }
            if (lane == 0u && totals.x != 0u) { InterlockedAdd(header[0].representationCount, 1u); }
            total += totals.x;
            references += totals.y;
            // The persistent GPU allocator admits using actual encoded bytes.
            // Its idle-cache eviction policy must not charge power-of-two bounds.
            if (mode == 2u) { cachedReferences = totals.y; }
        }
        if (lane == 0u) {
            if (scratchWritten && state[base + 4u] == 0u) { cutCache[cacheBase + 12u] = 0u; }
            state[base + 6u] = total;
            uint scratch = streamRTHistogramBase(params) + primitive.lodLevelOffset * 4u;
            state[scratch] = references; state[scratch + 1u] = 0u;
            state[scratch + 2u] = cachedReferences; state[scratch + 3u] = 0u;
        }
        return;
    }
    if (push.activeBuildPhase >= 38u && push.activeBuildPhase <= 40u) {
        RWStructuredBuffer<uint4> scan = resolveUniform(
            ResourceHandle<RWStructuredBuffer<uint4>>(push.blasHeaderBuffer));
        uint scratch = kStreamBLASHeaderWords / 4u + params.rtRepresentationGroupCapacity;
        if (push.activeBuildPhase == 39u) {
            if (item != 0u) { return; }
            uint prefix = 0u;
            for (uint block = 0u; block < (params.rtRepresentationCount + 63u) / 64u; ++block) {
                uint count = scan[scratch + block].x;
                scan[scratch + block] = uint4(prefix, 0u, 0u, 0u);
                prefix += min(count, 0xffffffffu - prefix);
            }
            return;
        }
        uint owner = 0u;
        uint base = 0u;
        uint count = 0u;
        bool first = false;
        if (item < params.rtRepresentationCount) {
            uint order = streamRTOwnerOrderBase(params);
            owner = state[order + item];
            uint primitive = owner / 3u;
            uint ranks = order + params.rtRepresentationCount * 2u + primitive * 3u;
            first = item == min(state[ranks], min(state[ranks + 1u], state[ranks + 2u]));
            base = streamRTPrimitiveBase(params, primitive);
            if (first) {
                count = state[base + 6u];
                if (push.activeBuildPhase == 38u && header[0].representationPadding != 0u &&
                    (state[base + 11u] & kRTIdleCache) != 0u) {
                    StructuredBuffer<GPUDrivenStreamAssetInstanceBLAS> representations = resolveUniform(
                        ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetInstanceBLAS>>(params.rtRepresentationBuffer));
                    count -= representations[primitive * 3u + 2u].previousGroupCount;
                    state[base + 11u] |= kRTEvictIdleCache;
                    InterlockedAdd(header[0].representationCount, 0xffffffffu);
                    state[base + 6u] = count;
                }
            }
        }
        if (push.activeBuildPhase == 38u) {
            // Only the primitive's highest-priority owner reserves its complete
            // group range. Impossible single requests cannot block smaller ones.
            uint demand = count <= params.rtRepresentationGroupCapacity ? count : 0u;
            uint4 before = scanRTRepresentationCounts(uint4(demand, 0u, 0u, 0u), lane);
            if (first) { state[base + 7u] = before.x; }
            if (lane == 63u) { scan[scratch + item / 64u] = g_StreamBlasStoragePrefix[63]; }
        } else if (first && count != 0u) {
            uint prefix = scan[scratch + item / 64u].x;
            prefix += min(state[base + 7u], 0xffffffffu - prefix);
            bool fits = prefix <= params.rtRepresentationGroupCapacity &&
                count <= params.rtRepresentationGroupCapacity - prefix;
            if (!fits) {
                state[base + 11u] |= kRTGroupOverflow;
                state[base + 6u] = 0u;
                InterlockedAdd(header[0].representationGroupOverflow, 1u);
            }
        }
        return;
    }
    if (push.activeBuildPhase == 31u || push.activeBuildPhase == 32u) {
        RWStructuredBuffer<uint4> scan = resolveUniform(
            ResourceHandle<RWStructuredBuffer<uint4>>(push.blasHeaderBuffer));
        uint4 demand = 0u;
        if (item < params.scenePrimitiveCount) {
            uint base = streamRTPrimitiveBase(params, item);
            if (push.activeBuildPhase == 31u) {
                uint scratch = streamRTHistogramBase(params) + primitives[item].lodLevelOffset * 4u;
                demand = uint4(state[scratch], state[scratch + 1u], state[base + 6u], 0u);
            } else {
                uint count = state[base + 6u];
                state[base + 6u] = count;
                demand.x = count;
            }
        }
        uint4 before = scanRTRepresentationCounts(demand, lane);
        if (push.activeBuildPhase == 32u && item < params.scenePrimitiveCount) {
            state[streamRTPrimitiveBase(params, item) + 7u] = before.x;
        }
        if (lane == 63u) { scan[kStreamBLASHeaderWords / 4u + params.rtRepresentationGroupCapacity + item / 64u] = g_StreamBlasStoragePrefix[63]; }
        return;
    }
    if (push.activeBuildPhase == 26u || push.activeBuildPhase == 33u) {
        if (item != 0u) { return; }
        RWStructuredBuffer<uint4> scan = resolveUniform(
            ResourceHandle<RWStructuredBuffer<uint4>>(push.blasHeaderBuffer));
        uint4 prefix = 0u;
        for (uint block = 0u; block < (params.scenePrimitiveCount + 63u) / 64u; ++block) {
            uint index = kStreamBLASHeaderWords / 4u + params.rtRepresentationGroupCapacity + block;
            uint4 count = scan[index];
            if (push.activeBuildPhase == 33u) { scan[index] = prefix; }
            prefix += min(count, 0xffffffffu - prefix);
        }
        if (push.activeBuildPhase == 26u) {
            // Representation planning precedes the BLAS header reset. A real
            // allocation rejection from the last submitted frame may evict
            // unreferenced discrete caches without predicting bucket sizes.
            bool actualStoragePressure = (push.traversalPhase & 1u) == 0u && header[0].storageRejected != 0u;
            bool evictIdle = prefix.z > params.rtRepresentationGroupCapacity || actualStoragePressure;
            header[0].representationPadding = evictIdle ? 1u : 0u;
        }
        return;
    }
    if (push.activeBuildPhase == 34u) {
        if (item >= params.scenePrimitiveCount) { return; }
        StructuredBuffer<uint4> scan = resolveUniform(
            ResourceHandle<StructuredBuffer<uint4>>(push.blasHeaderBuffer));
        uint base = streamRTPrimitiveBase(params, item);
        uint count = state[base + 6u];
        uint offset = scan[kStreamBLASHeaderWords / 4u + params.rtRepresentationGroupCapacity + item / 64u].x;
        offset += min(state[base + 7u], 0xffffffffu - offset);
        state[base + 7u] = offset;
        if (count == 0u) { return; }
        bool fits = offset <= params.rtRepresentationGroupCapacity && count <= params.rtRepresentationGroupCapacity - offset;
        if (!fits) {
            state[base + 11u] |= kRTGroupOverflow;
            InterlockedAdd(header[0].representationGroupOverflow, 1u);
        } else {
            InterlockedMax(rtHeader[0].activeGroupCount, offset + count);
            InterlockedMax(header[0].representationGroupCount, offset + count);
        }
        return;
    }
    if (push.activeBuildPhase == 27u) {
        if (item >= params.scenePrimitiveCount) { return; }
        GPUDrivenStreamAssetPrimitive primitive = primitives[item];
        uint base = streamRTPrimitiveBase(params, item);
        if ((state[base + 11u] & kRTGroupOverflow) != 0u) { return; }
        RWStructuredBuffer<uint4> output = resolveUniform(
            ResourceHandle<RWStructuredBuffer<uint4>>(params.rtRepresentationGroupBuffer));
        uint offset = state[base + 7u];
        for (uint mode = 0u; mode < 3u; ++mode) {
            bool retainedIdle = mode == 2u && (state[base + 11u] & kRTIdleCache) != 0u;
            if (state[base + 3u + mode] == 0u && !retainedIdle) { continue; }
            if (mode == 2u && (state[base + 11u] & kRTEvictIdleCache) != 0u) { continue; }
            for (uint first = 0u; first < primitive.groupCount; first += 64u) {
                uint local = first + lane;
                uint mask = local < primitive.groupCount
                    ? state[mode * params.sceneGroupCount + primitive.groupOffset + local] : 0u;
                uint added;
                uint before = streamCooperativePrefix(mask != 0u ? 1u : 0u, lane, added);
                if (mask != 0u) {
                    GPUDrivenStreamAssetGroup group = groups[primitive.groupOffset + local];
                    output[offset + before] = uint4(group.pageIndex, mask, group.clusterCount, item * 3u + mode);
                }
                offset += added;
            }
        }
        return;
    }
    if (push.activeBuildPhase == 24u) {
        if (item >= params.sceneInstanceCount) { return; }
        RWStructuredBuffer<GPUDrivenStreamAssetInstanceBLAS> instanceHistory = resolveUniform(
            ResourceHandle<RWStructuredBuffer<GPUDrivenStreamAssetInstanceBLAS>>(push.instanceBlasBuffer));
        StructuredBuffer<GPUDrivenStreamAssetInstanceBLAS> representations = resolveUniform(
            ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetInstanceBLAS>>(params.rtRepresentationBuffer));
        StructuredBuffer<uint2> fallback = resolveUniform(
            ResourceHandle<StructuredBuffer<uint2>>(push.fallbackBlasAddressBuffer));
        GPUDrivenStreamAssetInstance instance = instances[item];
        GPUDrivenStreamAssetInstanceBLAS history = instanceHistory[item];
        if ((push.traversalPhase & 1u) != 0u) { history = (GPUDrivenStreamAssetInstanceBLAS)0; }
        uint selected = state[streamRTInstanceBase(params, item) + 2u];
        bool dynamic = false;
        if (selected != 0xffffffffu) {
            dynamic = streamBlasDynamicReady(representations[selected]);
            if (!dynamic) {
                // A successful highest-drawable cut can satisfy every instance
                // of this primitive, even when its shared/cache slot was denied.
                uint merged = instance.primitiveIndex * 3u;
                if (streamBlasDynamicReady(representations[merged])) {
                    selected = merged;
                    dynamic = true;
                }
            }
            if (!dynamic) { selected = 0xffffffffu; }
        }
        // TLAS consumes this resolved selection, exactly matching semantic
        // publication and the successful (not merely requested) reuse counters.
        state[streamRTInstanceBase(params, item) + 2u] = selected;
        if (dynamic) {
            if (selected % 3u == 0u) { InterlockedAdd(header[0].mergedInstanceCount, 1u); }
            else if (selected % 3u == 1u) { InterlockedAdd(header[0].sharingInstanceCount, 1u); }
            else { InterlockedAdd(header[0].cachedInstanceCount, 1u); }
        }
        uint revision = dynamic ? state[streamRTPrimitiveBase(params, selected / 3u) + 8u + selected % 3u] : 0u;
        uint mode = instance.visible == 0u ? 0u : dynamic ? kBLASInstancePublishedDynamic :
            instance.primitiveIndex < params.scenePrimitiveCount && any(fallback[instance.primitiveIndex] != uint2(0u))
                ? kBLASInstancePublishedFallback : 0u;
        if (mode != (history.flags & kBLASInstancePublishedMask) ||
            (mode == kBLASInstancePublishedDynamic && (history.previousFirstGroup != selected || history.previousGroupCount != revision))) {
            InterlockedAdd(header[0].semanticChangeCount, 1u);
        }
        history.flags = mode;
        history.previousFirstGroup = dynamic ? selected : 0xffffffffu;
        history.previousGroupCount = revision;
        instanceHistory[item] = history;
    }
}

