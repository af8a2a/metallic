import Core;
using Metallic;
#include "../TestResourceParameters.h"

[shader("compute")]
[numthreads(64, 1, 1)]
void materialAssetCopyMain(uint3 id : SV_DispatchThreadID)
{
    ByteAddressBuffer source = resolveBuffer<ByteAddressBuffer>(getResourceParameters<MaterialRuntimeProbeResources>().materials);
    RWByteAddressBuffer output = resolveBuffer<RWByteAddressBuffer>(getResourceParameters<MaterialRuntimeProbeResources>().output);
    uint size;
    source.GetDimensions(size);
    if (id.x < size / 4u) { output.Store(id.x * 4u, source.Load(id.x * 4u)); }
}
