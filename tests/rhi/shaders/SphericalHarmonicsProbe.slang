#include "../../../Shaders/Libraries/Resources/ComputeResources.slang"

#include "../../../Shaders/Libraries/Math/SphericalHarmonics.slang"

#define gOutput METALLIC_RESOURCE(RWStructuredBuffer<float4>, 0)
#define gInput METALLIC_RESOURCE(StructuredBuffer<float4>, 1)

[shader("compute")]
[numthreads(1, 1, 1)]
void sphericalHarmonicsProbeMain()
{
    float3 direction = gInput[0].xyz;
    float3 normal = gInput[1].xyz;
    float3 radiance = gInput[2].xyz;
    float3 constantRadiance = gInput[3].xyz;
    SH::L2RGB sh = SH::projectOntoL2(direction, radiance);
    SH::storeL2RGB(gOutput, 0u, sh);
    gOutput[9] = float4(SH::evaluate(sh, normal), 0);
    gOutput[10] = float4(SH::calculateIrradiance(sh, normal), 0);
    SH::L1 scalar = SH::projectOntoL1(direction, 1.0);
    gOutput[11] = float4(SH::evaluate(scalar, normal), 0, 0, 0);
    gOutput[12] = float4(SH::calculateIrradiance(SH::projectOntoL1(direction, radiance), normal), 0);
    gOutput[13] = float4(SH::dotProduct(sh, sh), 0);

    SH::L2RGB arithmetic = SH::L2RGB::zero();
    arithmetic += sh;
    arithmetic = arithmetic + sh;
    arithmetic -= sh;
    arithmetic = 2.0 * arithmetic;
    arithmetic *= 2.0;
    arithmetic /= 4.0;
    arithmetic *= float3(2.0, 3.0, 4.0);
    arithmetic /= float3(2.0, 3.0, 4.0);
    arithmetic = (arithmetic - sh * 0.5) / 0.5;
    gOutput[14] = float4(SH::evaluate(arithmetic, normal), 0);
    gOutput[15] = float4(SH::evaluate(SH::lerp(sh, SH::projectOntoL2(normal, constantRadiance), 0.3), normal), 0);

    SH::L2RGB constantSH = SH::L2RGB::zero();
    constantSH.c[0] = constantRadiance * sqrt(4.0 * SH::kPi);
    gOutput[16] = float4(SH::calculateIrradiance(constantSH, normal), 0);
    gOutput[17] = float4(SH::toIrradiance(constantSH).evaluate(normal), 0);
    gOutput[18] = float4(SH::loadIrradianceL2RGB(gInput, 5u).evaluate(normal), 0);
    gOutput[19] = float4(SH::evaluate(constantSH, normal), 0);

    // Fixed proper rotation around multiple axes, using row-vector convention.
    const float3x3 rotation = float3x3(0.36, -0.48, 0.8, 0.8, 0.6, 0.0, -0.48, 0.64, 0.6);
    SH::L2RGB rotated = SH::rotate(sh, rotation);
    gOutput[20] = float4(SH::evaluate(rotated, mul(normal, rotation)) - SH::evaluate(sh, normal), 0);
    gOutput[21] = float4(SH::evaluate(SH::rotate(scalar, rotation), mul(normal, rotation)) - SH::evaluate(scalar, normal), 0, 0, 0);
    SH::L2RGB restored = SH::rotate(rotated, transpose(rotation));
    float3 rotationError = 0;
    SH::L2RGB projectedRotation = SH::projectOntoL2(mul(direction, rotation), radiance);
    for (int i = 0; i < SH::L2RGB::kCoefficientCount; ++i) {
        rotationError = max(rotationError, max(abs(restored.c[i] - sh.c[i]), abs(rotated.c[i] - projectedRotation.c[i])));
    }
    gOutput[22] = float4(rotationError, 0);
    gOutput[23] = float4(SH::evaluate(SH::convolveWithZH(sh, float3(1, 2, 3)), normal), 0);
    gOutput[24] = float4(SH::evaluate(SH::toRGB(SH::truncateToL1(SH::projectOntoL2(direction, 1.0))), normal), 0);
    gOutput[25] = float4(SH::evaluate(SH::L1RGB::zero(), normal), 0);
    gOutput[26] = float4(SH::evaluate(SH::convolveWithZH(scalar, float2(1, 2)), normal), 0, 0, 0);
    SH::storeL2RGB(gOutput, 32u, SH::loadL2RGB(gInput, 5u));
}

// Compile-only coverage: half coefficients use the same API without requiring
// every GPU test device to expose native fp16 shader arithmetic.
[shader("compute")]
[numthreads(1, 1, 1)]
void sphericalHarmonicsHalfCompileMain()
{
    float3 direction = normalize(gInput[0].xyz);
    SH::L2Of<half3> rgb = SH::L2Of<half3>::zero();
    rgb += SH::projectOntoL2(direction, half3(gInput[2].xyz));
    rgb *= 2.0;
    rgb /= half3(2.0);
    rgb = SH::lerp(rgb, rgb + rgb, 0.5);
    rgb -= rgb * 0.25;
    rgb = SH::rotate(rgb, float3x3(0,1,0, -1,0,0, 0,0,1));
    SH::L1Of<half> scalar = SH::projectOntoL1(direction, half(1.0));
    SH::L2Of<half> scalarL2 = SH::projectOntoL2(direction, half(1.0));
    scalar = SH::rotate(scalar, float3x3(1,0,0, 0,1,0, 0,0,1));
    scalar = SH::convolveWithZH(scalar, float2(1, 2));
    rgb = SH::convolveWithZH(rgb, float3(1, 2, 3));
    gOutput[0] = float4(float3(SH::calculateIrradiance(rgb, direction)), 0);
    gOutput[1] = float4(float3(SH::evaluate(SH::toRGB(scalar), direction)), 0);
    gOutput[2] = float4(float3(SH::dotProduct(rgb, SH::toRGB(scalarL2))), 0);
    gOutput[3] = float4(float3(SH::evaluate(SH::truncateToL1(rgb), direction)), 0);
}
