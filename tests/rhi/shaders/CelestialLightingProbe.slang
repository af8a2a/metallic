import Core;
using Metallic;
import Lighting;
using Metallic.Lighting;

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    let resources = getResourceParameters<SceneResourceParameters>();
    let lights = resolveBuffer<StructuredBuffer<GPUCelestialLight>>(resources.environment.celestialLights);
    let output = resolveBuffer<RWStructuredBuffer<float4>>(resources.probeOutput);
    float3 sum = float3(0.0);
    for (uint slot = 0u; slot < kCelestialLightCount; ++slot) {
        GPUCelestialLight light = lights[slot];
        output[slot * 3u] = float4(sampleCelestialLight(lights, slot).direction, light.angularRadius);
        output[slot * 3u + 1u] = float4(light.irradiance, asfloat(light.flags));
        output[slot * 3u + 2u] = float4(light.diskRadiance, light.shadowImportance);
        // Missing optional atmosphere bindings must preserve the HDRI-mode
        // celestial value without resolving the invalid descriptor sentinel.
        sum += sampleCelestialLightAtPosition(lights, slot, float3(0.0)).illuminance;
    }
    output[6] = float4(sum, 0.0);
    output[7] = float4(asfloat(celestialSourceIdentity(0u)), asfloat(celestialSourceIdentity(1u)),
        celestialLightCastsShadow(lights, 0u) ? 1.0 : 0.0,
        celestialLightCastsShadow(lights, 1u) ? 1.0 : 0.0);
}
