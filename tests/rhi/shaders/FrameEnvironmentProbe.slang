import Core;
using Metallic;
#include "../TestResourceParameters.h"

#define environmentTexture resolveUniform(getResourceParameters<FrameEnvironmentProbeResources>().environment)
#define result resolveBuffer<RWStructuredBuffer<float4>>(getResourceParameters<FrameEnvironmentProbeResources>().output)

[shader("compute")]
[numthreads(1, 1, 1)]
void readEnvironment(uint3 thread : SV_DispatchThreadID)
{
    result[0] = environmentTexture.Load(int3(0, 0, 0));
}
