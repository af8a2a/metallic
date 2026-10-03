import Core;
using Metallic;
struct NamedResources
{
    ResourceHandle<RWByteAddressBuffer> output;
    BufferSpan<uint> input;
};
[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    let resources = getResourceParameters<NamedResources>();
    let input = typedBufferSpan<uint>(resources.input);
    resolveUniform(resources.output).Store(0, input.load(0) + getConstants<uint>());
    resolveUniform(resources.output).Store(4, input.load(input.count));
}
