module ProgramBinningProbe;

import MaterialBinningProbe;

[shader("compute")]
[numthreads(32, 1, 1)]
void materialIndirectProbeMain(uint3 group : SV_GroupID)
{
    dispatchMaterialIndirectProbe(group);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void materialReadbackResetMain(uint3 id : SV_DispatchThreadID)
{
    dispatchMaterialReadbackReset(id);
}
