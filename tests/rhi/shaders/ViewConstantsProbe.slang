#include "../ViewProbeParameters.h"

[[vk::push_constant]] ConstantBuffer<ViewProbeParameters> gProbe;

[shader("compute")]
[numthreads(1, 1, 1)]
void viewConstantsProbeMain()
{
    if (!gProbe.output.contains(0u) || !gProbe.view.contains(0u)) { return; }
    gProbe.output.data[0] = gProbe.view.data[0];
}
