import Core;
using Metallic;

[shader("compute")]
[numthreads(1, 1, 1)]
void viewConstantsProbeMain()
{
    getResource<RWStructuredBuffer<ViewConstants>>(0)[0] =
        getResource<StructuredBuffer<ViewConstants>>(1)[0];
}
