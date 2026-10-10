import Core;
using Metallic;
#include "../TestResourceParameters.h"
[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    let resources = getResourceParameters<NamedResourceProbeResources>();
    let input = typedBufferSpan<uint>(resources.input);
    resolveBuffer<RWByteAddressBuffer>(resources.output).Store(0, input.load(0) + getConstants<uint>());
    resolveBuffer<RWByteAddressBuffer>(resources.output).Store(4, input.load(input.count));
}
