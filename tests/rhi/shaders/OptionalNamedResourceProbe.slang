import Core;
using Metallic;

struct OptionalNamedResources
{
    ResourceHandle<RWByteAddressBuffer> output;
    ResourceHandle<ByteAddressBuffer> input;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    let resources = getResourceParameters<OptionalNamedResources>();
    let output = resolveUniform(resources.output);
    bool present = resources.input.index != 0xffffffffu;
    uint value = 0u;
    if (present) { value = resolveUniform(resources.input).Load(0); }
    output.Store(0, resources.input.index);
    output.Store(4, value);
    output.Store(8, present ? 1u : 0u);
}
