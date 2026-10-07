import Core;
import Atmosphere;
using Metallic;

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    let resources = getResourceParameters<SceneResourceParameters>();
    let environment = resources.environment;
    let parameters = resolveBuffer<StructuredBuffer<GPUAtmosphereParameters>>(environment.atmosphereParameters)[0];
    let output = resolveBuffer<RWStructuredBuffer<float4>>(resources.probeOutput);
    output[0] = parameters.observerPlanetBottom;
    output[1] = parameters.observerWorldTop;
    output[2] = parameters.rayleighScaleHeight;
    output[3] = parameters.mieScatteringScaleHeight;
    output[4] = parameters.mieExtinctionAnisotropy;
    output[5] = parameters.ozoneCenter;
    output[6] = parameters.groundOzoneWidth;
    output[7] = parameters.sunDirectionRadius;
    output[8] = parameters.sunIrradianceEnabled;
    output[9] = parameters.moonDirectionRadius;
    output[10] = parameters.moonIrradianceEnabled;
    output[11] = parameters.settings;
    uint width, height;
    resolveUniform(environment.transmittance).GetDimensions(width, height);
    output[12] = float4(width, height, 0.0, 0.0);
    resolveUniform(environment.multiScattering).GetDimensions(width, height);
    output[13] = float4(width, height, 0.0, 0.0);
    resolveUniform(environment.skyView).GetDimensions(width, height);
    output[14] = float4(width, height, 0.0, 0.0);
    let aerial = resolveBuffer<StructuredBuffer<float4>>(environment.aerialPerspective);
    let primary = resolveBuffer<StructuredBuffer<float4>>(environment.primaryAerial);
    output[15] = aerial[0];
    output[16] = aerial[1];
    output[17] = primary[0];
    output[18] = primary[1];
}
