import Core;
import NRDEncoding;
using Metallic;
#include "../TestResourceParameters.h"
#include "../../../Shaders/Interop/MaterialTextureDecode.hlsli"
#define output resolveBuffer<RWStructuredBuffer<float4>>(getResourceParameters<BatchBarrierProbeResources>().output)
[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    float3 red = WorkingColor::fromLinearRec709(float3(1, 0, 0));
    output[0] = float4(red, WorkingColor::luminance(red));
    output[1] = float4(WorkingColor::toLinearRec709(red), 1);
    output[2] = float4(materialTextureColor(float3(0.5, 0.2, 0.8), 0), 1);
    output[3] = float4(materialTextureColor(float3(0.5, 0.2, 0.8), 4), 1);
    output[4] = float4(materialTextureColor(float3(0.5, 0.2, 0.8), 9), 1);
    output[5] = float4(materialTextureModulate(WorkingColor::fromLinearRec709(float3(0.8, 0.3, 0.7)), float3(0.5, 0.2, 0.8), 0), 1);
    output[6] = float4(WorkingColor::toACEScg(WorkingColor::fromLinearRec709(float3(0.18))), 1);
    output[7] = float4(WorkingColor::fromLinearRec709(WorkingColor::toLinearRec709(float3(4, -0.2, 10))), 1);
#ifdef METALLIC_TEST_NRD_ADAPTER
    output[9] = float4(materialTextureModulate(WorkingColor::fromLinearRec709(float3(0.8, 0.3, 0.7)), float3(0.5, 0.2, 0.8), 9), 1);
    output[8] = Metallic.Interop.NRDEncoding::packNrdRadianceHitDistance(WorkingColor::fromLinearRec709(float3(0.8, 0.3, 0.1)), 2.0, true);
#endif
}
