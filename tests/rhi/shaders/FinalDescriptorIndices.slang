// All values are final indices in the currently bound descriptor heap.
struct Push
{
    uint inputBuffer;
    uint cookie;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uniform Push push)
{
    ByteAddressBuffer input = DescriptorHandle<ByteAddressBuffer>(uint2(push.inputBuffer, 0));
    uint outputIndex = input.Load(0);
    RWByteAddressBuffer output = DescriptorHandle<RWByteAddressBuffer>(uint2(outputIndex, 0));
    output.Store4(0, uint4(input.Load(4), push.cookie, push.inputBuffer, outputIndex));
}
