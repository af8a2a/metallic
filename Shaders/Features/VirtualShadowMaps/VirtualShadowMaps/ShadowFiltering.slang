#language slang 2026
implementing VirtualShadowMaps;

// Implementation units preprocess independently; retain this unit's permutation defaults.
#ifndef VSM_DIRECTIONAL_PROJECTION
#define VSM_DIRECTIONAL_PROJECTION 0
#endif
#ifndef VSM_ADAPTIVE_RAY_COUNT
#define VSM_ADAPTIVE_RAY_COUNT 0
#endif
#ifndef VSM_PROJECTION_PAGE_SHIFT
#define VSM_PROJECTION_PAGE_SHIFT -1
#endif

float vsmRayDepth(float depth, ShadowProjection projection, uint2 virtualTexel, float3 receiver, uint pageSize)
{
#if VSM_DIRECTIONAL_PROJECTION
    return (depth - dot(projection.rowZ, float4(receiver, 1.0))) / length(projection.rowZ.xyz);
#else
    if (projection.metric.w < 0.5) {
        return (depth - dot(projection.rowZ, float4(receiver, 1.0))) / length(projection.rowZ.xyz);
    }
    float2 ndc = (float2(virtualTexel) + 0.5) *
        (2.0 / float(projection.page.y * pageSize)) - 1.0;
    float3 forward = normalize(projection.rowW.xyz);
    float3 texelRay = forward + projection.metric.x *
        (normalize(projection.rowX.xyz) * ndc.x + normalize(projection.rowY.xyz) * ndc.y);
    float scale = dot(projection.rowZ.xyz, forward);
    float offset = dot(projection.rowZ, float4(projection.originNear.xyz, 1.0));
    // Inverse radial depth gives every cube face the same continuous history
    // coordinate. Larger values, as for directional depth, are toward the light.
    return (depth - scale) / (offset * length(texelRay));
#endif
}

float vsmFilterVisibility(float3 position, float3 normal, uint receiverLevel, uint2 pixel,
    StructuredBuffer<ShadowProjection> projections, ShadowPush push, inout uint3 filterCounts, bool useGroupCache = false)
{
    ShadowProjection receiverProj = vsmGetCachedProjection(projections, receiverLevel, useGroupCache);
    if (push.filterMode == 1u) {
        float sum = 0.0;
        for (int y = -1; y <= 1; ++y) {
            for (int x = -1; x <= 1; ++x) {
                float3 tap = vsmPCFPosition(position, normal, receiverProj, push, int2(x, y));
                ShadowLookup sample = vsmFilterLookup(tap, push, receiverLevel, filterCounts);
                float compare = sample.valid ? vsmComparisonDepth(tap, normal, sample.projected,
                    sample.virtualTexel, vsmLookupProjection(sample), push.pageSize) : 1.0;
                sum += !sample.valid || sample.storedDepth == 0.0 || compare >= sample.storedDepth ? 1.0 : 0.0;
            }
        }
        return sum / 9.0;
    }
    float visibility = 0.0;
    float stepRandom = vsmSampleRandom(pixel, 2u, push.samplePhase);
    float invSamplesPerRay = 1.0 / float(push.samplesPerRay);
    float invRayCount = 1.0 / float(push.rayCount);
#if VSM_ADAPTIVE_RAY_COUNT
    uint tracedRays = push.rayCount;
#endif
    // A ray keeps its receiver's resolution. An angular footprint shrinks to
    // zero at contact and would request entire fine clipmaps, starving the
    // receiver pages and forcing nearly all samples onto the coarse map.
    ShadowProjection proj0 = vsmGetCachedProjection(projections, 0u, useGroupCache);
#if VSM_DIRECTIONAL_PROJECTION
    float3 toLight = normalize(proj0.rowZ.xyz);
    float receiverTexelSize = receiverProj.metric.y;
#else
    float3 toLight = push.lightType == 0u ? normalize(proj0.rowZ.xyz) :
        normalize(proj0.originNear.xyz - position);
    float receiverTexelSize = vsmWorldTexelSize(position, receiverProj, push.pageSize, push.lightType);
#endif
    float3 right = normalize(cross(abs(toLight.y) < 0.95 ? float3(0, 1, 0) : float3(1, 0, 0), toLight));
    float3 up = cross(toLight, right);
    float tanAngularRadius = tan(push.angularRadius);
#if VSM_DIRECTIONAL_PROJECTION
    float receiverExtent = receiverProj.metric.x;
    float3 normX = normalize(proj0.rowX.xyz);
    float3 normY = normalize(proj0.rowY.xyz);
    float biasZ = proj0.metric.z;
    float receiverInvResolution = 2.0 / float(push.virtualPagesPerAxis * push.pageSize);
    float coarseInvResolution = 2.0 / float(push.pageSize);
    float denominator = dot(normal, toLight);
    float invDenom = abs(denominator) > 1e-5 ? length(proj0.rowZ.xyz) / denominator : 0.0;
    float2 normalSlope = float2(dot(normal, normX), dot(normal, normY)) * invDenom;
    float2 receiverScaledSlope = normalSlope * receiverExtent;
    StructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.pages));
    Texture2D<float> atlas = resolveUniform(ResourceHandle<Texture2D<float>>(push.atlas));
    VSMDirectionalRayBase rayBase = vsmCreateDirectionalRayBase(position, receiverProj, pages, push, receiverLevel);
    VSMDirectionalRayCache rayCache;
    rayCache.lastPage = 0xffffffffu;
    rayCache.lastPhysicalCoord = uint2(0u);
    rayCache.lastPageHit = false;
    float2 texelSlope = -receiverScaledSlope * receiverInvResolution;
    float2 baseOffset = float2(0.5 * receiverInvResolution - 1.0) - rayBase.comparisonOrigin.xy;
    float baseSlopeCorrection = -dot(receiverScaledSlope, baseOffset) + biasZ;
#endif
    for (uint ray = 0u; ray < push.rayCount; ++ray) {
        vsmWorkAdd(push, kVSMWorkProjectionRays, 1u);
        float2 disk = vsmDiskSample(ray, invRayCount, pixel, push.samplePhase);
        float3 rayOffset = right * disk.x + up * disk.y;
#if VSM_DIRECTIONAL_PROJECTION
        float3 endpoint = vsmRayEndpointDirectional(position, toLight, rayOffset, tanAngularRadius, push.rayLength);
#else
        float3 endpoint = (push.lightType == 0u) ?
            vsmRayEndpointDirectional(position, toLight, rayOffset, tanAngularRadius, push.rayLength) :
            vsmRayEndpoint(position, proj0, push, toLight, rayOffset);
#endif
        float2 ditherOffset = float2(vsmSampleRandom(pixel, 3u + 2u * ray, push.samplePhase),
            vsmSampleRandom(pixel, 4u + 2u * ray, push.samplePhase)) - 0.5;
        float3 dither = (right * ditherOffset.x + up * ditherOffset.y) * receiverTexelSize;
#if VSM_DIRECTIONAL_PROJECTION
        VSMDirectionalRay dirRay;
        float3 direction = endpoint - position;
        dirRay.lookupOrigin = vsmProjectPosition(position + dither, receiverProj);
        dirRay.step = float3(dot(receiverProj.rowX.xyz, direction), dot(receiverProj.rowY.xyz, direction), dot(receiverProj.rowZ.xyz, direction));
        float raySlopeZ = dirRay.step.z + dot(receiverScaledSlope, dirRay.step.xy);
        float maxSlope = 0.05;
#else
        float maxSlope = push.lightType == 0u ? 0.05 : 0.05 * abs(
            1.0 / length(endpoint - proj0.originNear.xyz) -
            1.0 / length(position - proj0.originNear.xyz));
#endif
        float historyDepth = 0.0, historyTime = 0.0, historySlope = 0.0, previousReference = 0.0;
        bool hasHistory = false;
        bool blocked = false;
        // Trace back toward the receiver, including t=0. Depth discontinuities
        // can expose only the front layer: continue the preceding surface with
        // bounded slope through that gap instead of inventing an intersection.
#if VSM_DIRECTIONAL_PROJECTION
        for (uint step = 0u; step <= push.samplesPerRay; ++step) {
            float time = vsmRaySampleTime(step, push, stepRandom, invSamplesPerRay);
            ++filterCounts.x;
            float pZ = fma(time, dirRay.step.z, dirRay.lookupOrigin.z);
            if (pZ < 0.0 || pZ > 1.0) {
                rayCache.lastPage = 0xffffffffu;
                rayCache.lastPageHit = false;
                continue;
            }
            float2 pXY = fma(float2(time), dirRay.step.xy, dirRay.lookupOrigin.xy);
            float3 p = float3(pXY, pZ);

            float compare;
            float reference;
            float storedDepth;
            bool sampleValid = false;

            if (rayBase.allowClipmaps && all(abs(pXY) < rayBase.receiverMarginMax)) {
                uint2 texel = uint2(fma(pXY, float2(rayBase.halfReceiverResolution), float2(rayBase.halfReceiverResolution)));
#if VSM_PROJECTION_PAGE_SHIFT >= 0
                uint2 pageCoord = texel >> VSM_PROJECTION_PAGE_SHIFT;
                uint2 localTexel = vsmPageLocalTexel(texel, push.pageSize);
                uint pageOffset = (pageCoord.y << rayBase.virtualPagesPerAxisShift) | pageCoord.x;
#else
                bool isPow2 = rayBase.powerOfTwoLayout;
                uint2 pageCoord;
                uint2 localTexel;
                uint pageOffset;
                uint pageSizeShift = rayBase.pageSizeShift;
                uint pageSizeMask = rayBase.pageSizeMask;
                uint vpaShift = rayBase.virtualPagesPerAxisShift;
                if (isPow2) {
                    pageCoord = texel >> pageSizeShift;
                    localTexel = texel & pageSizeMask;
                    pageOffset = (pageCoord.y << vpaShift) | pageCoord.x;
                } else {
                    pageCoord = texel / push.pageSize;
                    localTexel = texel % push.pageSize;
                    pageOffset = pageCoord.y * push.virtualPagesPerAxis + pageCoord.x;
                }
#endif
                uint page = rayBase.receiverPageOffset + pageOffset;
                vsmWorkAdd(push, kVSMWorkProjectionMappingLoads, 1u);
                uint2 physical;
                bool isFastFine = false;
                bool isFastAncestor = false;
                uint ancestorLevel = 0u;
                uint2 ancestorTexel = uint2(0u);
                float2 ancestorTranslation = float2(0.0);
                float ancestorExtent = 0.0;

                if (page == rayCache.lastPage && rayCache.lastPageHit) {
                    physical = rayCache.lastPhysicalCoord;
                    isFastFine = true;
                } else {
                    uint4 mapping = pages[rayBase.receiverMappingOffset + pageOffset];
                    if (mapping.w == 1u) {
                        physical = vsmMappingPhysicalCoord(mapping.y);
                        rayCache.lastPhysicalCoord = physical;
                        rayCache.lastPage = page;
                        rayCache.lastPageHit = true;
                        isFastFine = true;
                    } else {
                        rayCache.lastPage = page;
                        rayCache.lastPageHit = false;
                        ancestorLevel = vsmMappingLevel(mapping.y);
                        if (mapping.w != 0u && ancestorLevel < push.clipmapCount) {
                            physical = vsmMappingPhysicalCoord(mapping.y);
                            float3 ancestorP = p;
                            ancestorTexel = texel;
                            vsmMapDirectionalAncestor(mapping, rayBase.sourceTranslation, ancestorP, ancestorTexel, projections, push, rayBase.receiverLevel, ancestorTranslation, ancestorExtent, useGroupCache);
                            isFastAncestor = true;
                        } else {
                            ShadowLookup sample = vsmDirectionalRayLookupSlow(p, texel, page, mapping, true, time, dirRay, rayBase, projections, pages, atlas, push, filterCounts, rayCache, useGroupCache);
                            if (sample.valid) {
                                sampleValid = true;
                                storedDepth = sample.storedDepth;
                                float3 stepOffset = time * dirRay.step;
                                vsmDirectionalSampleCompareAndReference(sample, time, stepOffset, dirRay,
                                    rayBase, projections, push, receiverInvResolution, coarseInvResolution,
                                    receiverScaledSlope, normalSlope, biasZ, useGroupCache, compare, reference);
                            }
                        }
                    }
                }

                if (isFastFine) {
#if VSM_PROJECTION_PAGE_SHIFT >= 0
                    uint2 atlasTexel = vsmAtlasTexel(physical, localTexel, push.pageSize);
#else
                    uint2 atlasTexel;
                    if (isPow2) {
                        atlasTexel = (physical << pageSizeShift) | localTexel;
                    } else {
                        atlasTexel = physical * push.pageSize + localTexel;
                    }
#endif
                    storedDepth = atlas.Load(int3(atlasTexel, 0));
                    vsmWorkAdd(push, kVSMWorkProjectionAtlasLoads, 1u);

                    float relDepth = fma(time, raySlopeZ, dot(texelSlope, float2(texel)) + baseSlopeCorrection);
                    compare = rayBase.comparisonOrigin.z + relDepth;
                    reference = relDepth * rayBase.invDepthScale;
                    sampleValid = true;
                } else if (isFastAncestor) {
#if VSM_PROJECTION_PAGE_SHIFT >= 0
                    uint2 atlasTexel = vsmAtlasTexel(physical, vsmPageLocalTexel(ancestorTexel, push.pageSize), push.pageSize);
#else
                    uint2 atlasTexel;
                    if (isPow2) {
                        atlasTexel = (physical << pageSizeShift) | (ancestorTexel & pageSizeMask);
                    } else {
                        atlasTexel = vsmAtlasTexelFromVirtual(physical, ancestorTexel, push.pageSize);
                    }
#endif
                    storedDepth = atlas.Load(int3(atlasTexel, 0));
                    vsmWorkAdd(push, kVSMWorkProjectionAtlasLoads, 1u);

                    float scale = asfloat((127u - (ancestorLevel - rayBase.receiverLevel)) << 23u);
                    float2 translation = ancestorTranslation - rayBase.sourceTranslation * scale;
                    float2 ancestorOrigin = fma(rayBase.comparisonOrigin.xy, float2(scale), translation);
                    float2 ancestorSampleNDC = fma(float2(ancestorTexel), float2(receiverInvResolution), float2(0.5 * receiverInvResolution - 1.0));
                    float2 ancestorOffset = (ancestorSampleNDC - ancestorOrigin) * ancestorExtent;
                    float ancestorCorrection = -dot(normalSlope, ancestorOffset);
                    float relDepth = fma(time, raySlopeZ, ancestorCorrection + biasZ);
                    compare = rayBase.comparisonOrigin.z + relDepth;
                    reference = relDepth * rayBase.invDepthScale;
                    sampleValid = true;
                }
            } else {
                ShadowLookup sample = vsmDirectionalRayLookupSlow(p, uint2(0u), 0u, uint4(0u), false, time, dirRay, rayBase, projections, pages, atlas, push, filterCounts, rayCache, useGroupCache);
                if (sample.valid) {
                    sampleValid = true;
                    storedDepth = sample.storedDepth;
                    float3 stepOffset = time * dirRay.step;
                    vsmDirectionalSampleCompareAndReference(sample, time, stepOffset, dirRay,
                        rayBase, projections, push, receiverInvResolution, coarseInvResolution,
                        receiverScaledSlope, normalSlope, biasZ, useGroupCache, compare, reference);
                }
            }

            if (!sampleValid) { continue; }

            float diff = (storedDepth - compare) * rayBase.invDepthScale;
            if (!hasHistory) {
                blocked = diff > 0.0;
                if (blocked) { break; }
                historyDepth = reference + diff;
                historyTime = time;
                hasHistory = true;
            } else {
                float tolerance = abs(reference - previousReference) * 1.05;
                if (diff <= tolerance) {
                    blocked = diff > 0.0;
                    float depth = reference + diff;
                    if (depth != historyDepth) {
                        float timeDelta = time - historyTime;
                        historySlope = clamp((depth - historyDepth) / timeDelta, -maxSlope, maxSlope);
                        historyDepth = depth;
                        historyTime = time;
                    }
                } else {
                    float timeDelta = time - historyTime;
                    float predictedSeparation = (historyDepth + historySlope * timeDelta) - reference;
                    blocked = predictedSeparation > 0.0 && predictedSeparation <= tolerance;
                }
            }
            if (blocked) { break; }
            previousReference = reference;
        }
#else
        for (uint step = 0u; step <= push.samplesPerRay; ++step) {
            float time = vsmRaySampleTime(step, push, stepRandom, invSamplesPerRay);
            float3 point = vsmRayPosition(position, endpoint, time, proj0, push);
            ShadowLookup sample = vsmFilterLookup(point + dither, push, receiverLevel, filterCounts);
            if (!sample.valid) { continue; }
            ShadowProjection projection = vsmLookupProjection(sample);
            float compare = vsmComparisonDepth(point, normal, vsmProjectPosition(point, projection),
                sample.virtualTexel, projection, push.pageSize);
            float reference = vsmRayDepth(compare, projection, sample.virtualTexel, position, push.pageSize);
            float depth = push.lightType != 0u && sample.storedDepth == 0.0 ? 0.0 :
                vsmRayDepth(sample.storedDepth, projection, sample.virtualTexel, position, push.pageSize);
            if (!hasHistory) {
                // This ordinary comparison also retains blockers beyond the
                // bounded search; there is no separate hard-shadow tail.
                blocked = depth > reference;
                historyDepth = depth; historyTime = time;
                hasHistory = true;
            } else {
                float tolerance = abs(reference - previousReference) * 1.05;
                float reconstructed = depth;
                if (depth - reference > tolerance) {
                    reconstructed = historyDepth + historySlope * (time - historyTime);
                } else if (depth != historyDepth) {
                    historySlope = clamp((depth - historyDepth) / (time - historyTime), -maxSlope, maxSlope);
                    historyDepth = depth; historyTime = time;
                }
                float separation = reconstructed - reference;
                blocked = separation > 0.0 && separation <= tolerance;
            }
            if (blocked) { break; }
            previousReference = reference;
        }
#endif
        visibility += blocked ? 0.0 : 1.0;
#if VSM_ADAPTIVE_RAY_COUNT
        if (ray + 1u < push.rayCount) {
            // Vote only among this light's active receivers. This heuristic
            // assumes full light after one miss, or umbra after two hits in
            // every participating lane; it is not the fixed-ray estimator.
            bool keepTracing = ray == 0u ? blocked : (visibility > 0.0);
            if (!WaveActiveAnyTrue(keepTracing)) {
                tracedRays = ray + 1u;
                vsmWorkAdd(push, ray == 0u ? kVSMWorkProjectionAdaptiveLitReceivers :
                    kVSMWorkProjectionAdaptiveUmbraReceivers, 1u);
                vsmWorkAdd(push, kVSMWorkProjectionAdaptiveSavedRays, push.rayCount - tracedRays);
                break;
            }
        }
#endif
    }
#if VSM_ADAPTIVE_RAY_COUNT
    return visibility / float(tracedRays);
#else
    return visibility / float(push.rayCount);
#endif
}

