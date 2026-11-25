#language slang 2026
module ClusteredSurfaceLighting;

import Core;
import Lighting;
import OpenPBRSurface;
import RectLightLTC;
import ScenePathMath;
import ScenePathTypes;
import SceneSurfaceMaterial;
import SurfaceLighting;

namespace Metallic.Lighting
{
using Metallic.SceneOpenPBR;
using Metallic.ScenePath;

public interface IClusteredLightVisibility
{
    public float3 evaluate(SurfaceLightSample light);
};

public struct ClusteredSurfaceLights<TVisibility : IClusteredLightVisibility> : ISurfaceLightProvider
{
    public TVisibility shadow;
    public uint debugFlags;
    public float3 position;
    public float3 normal;
    public float3 view;
    public float3 diffuseColor;
    public float3 specularColor;
    public float roughness;
    public bool useRectLights;
    public uint4 gridCounts;
    public ClusterLightGridLookup lookup;
    public bool rectLTCPrepared;
    public RectLightLTCReceiver rectReceiver;

    public uint count()
    {
        return (debugFlags & kOpenPBRDebugDisableDirectLighting) != 0u
            ? 0u : kCelestialLightCount + lookup.count + gridCounts.z;
    }

    [mutating]
    public SurfaceLightSample sample(uint index)
    {
        SurfaceLightSample result = {};
        PunctualLightSample light;
        if (index < kCelestialLightCount) {
            result.source = celestialSourceIdentity(index);
            let celestial = resolveBuffer<StructuredBuffer<GPUCelestialLight>>(
                getSceneResourceParameterField<ResourceHandle<ByteAddressBuffer>>(kSceneEnvironmentOffset));
            light = sampleCelestialLightAtPosition(celestial, index, position);
        } else {
            index -= kCelestialLightCount;
            let candidates = resolveBuffer<StructuredBuffer<uint>>(
                getSceneResourceParameterField<ResourceHandle<ByteAddressBuffer>>(kSceneLightGridCandidatesOffset));
            let indices = resolveBuffer<StructuredBuffer<uint>>(
                getSceneResourceParameterField<ResourceHandle<ByteAddressBuffer>>(kSceneLightGridIndicesOffset));
            result.source = index < lookup.count
                ? clusterLightGridLocalLightIndex(lookup, indices, candidates, index)
                : candidates[gridCounts.x + (index - lookup.count)];
            // Rectangle payloads follow the stable punctual slots in the source ID domain.
            if (result.source >= gridCounts.y) {
                if (!useRectLights) { return result; }
                let rect = getResourceParameterField<SceneRealtimeResourceParameters, RectLightResourceParameters>(
                    kSceneRectLightsOffset);
                let rectangle = resolveBuffer<StructuredBuffer<GPURectLight>>(rect.lights)[result.source - gridCounts.y];
                float3 center = rectangle.positionRange.xyz - position;
                float attenuation = rectLightAttenuation(rectangle, center);
                if (attenuation == 0.0) { return result; }
                if (!rectLTCPrepared) {
                    let samplerState = resolveUniform(
                        getSceneResourceParameterField<SamplerHandle>(kSceneOpenPBRLUTSamplerOffset));
                    rectReceiver = prepareRectLightLTC(normal, view, diffuseColor, specularColor, roughness,
                        resolveUniform(rect.matrixLUT), resolveUniform(rect.amplitudeLUT), samplerState);
                    rectLTCPrepared = true;
                }
                result.integrated = true;
                result.integratedRadiance = evaluateRectLightLTC(rectReceiver, rectangle, center, attenuation);
                return result;
            }
            let punctual = resolveBuffer<StructuredBuffer<GPUPunctualLight>>(
                getSceneResourceParameterField<ResourceHandle<ByteAddressBuffer>>(kSceneLightGridLightsOffset));
            light = samplePunctualLightData(punctual[result.source], position);
        }
        result.direction = light.direction;
        result.illuminance = light.illuminance;
        result.distance = light.distance;
        result.inverseSourcePdf = any(light.illuminance > 0.0) ? 1.0 : 0.0;
        return result;
    }

    public float3 visibility(SurfaceLightSample light)
    {
        return shadow.evaluate(light);
    }
};

public ClusteredSurfaceLights<TVisibility> makeClusteredSurfaceLights<TVisibility : IClusteredLightVisibility>(
    TVisibility shadow, HitInfo hit, float3 viewDirection, float2 pixel, ScenePathTracePush push,
    SceneSurfaceSummary summary, bool useRectLights)
{
    ClusteredSurfaceLights<TVisibility> lights = {};
    lights.shadow = shadow;
    lights.debugFlags = push.debugFlags;
    lights.position = hit.position;
    lights.normal = hit.normal;
    lights.view = viewDirection;
    lights.diffuseColor = summary.diffuseColor;
    lights.specularColor = summary.specularReflectance;
    lights.roughness = summary.roughness;
    lights.useRectLights = useRectLights;
    if ((push.debugFlags & kOpenPBRDebugDisableDirectLighting) != 0u) { return lights; }
    let grid = resolveBuffer<StructuredBuffer<ClusterLightGridParams>>(
        getSceneResourceParameterField<ResourceHandle<ByteAddressBuffer>>(kSceneLightGridParamsOffset))[0];
    lights.gridCounts = grid.counts;
    if (grid.counts.x > 0u) {
        let cells = resolveBuffer<StructuredBuffer<ClusterLightGridCell>>(
            getSceneResourceParameterField<ResourceHandle<ByteAddressBuffer>>(kSceneLightGridCellsOffset));
        lights.lookup = clusterLightGridLookup(grid, cells, uint2(pixel), cameraLinearDepth(hit.position, push));
    }
    return lights;
}

} // namespace Metallic.Lighting
