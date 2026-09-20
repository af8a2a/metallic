import Core;
using Metallic;
struct Push { uint slot; uint wantedMip; uint samples; uint unused; };
[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    Push push = getConstants<Push>();
    RWStructuredBuffer<uint> feedback = getResource<RWStructuredBuffer<uint>>(0);
    uint count, stride;
    feedback.GetDimensions(count, stride);
    uint offset = push.slot * 8;
    if (offset + 7 < count && feedback[offset] != 0 && push.samples != 0) {
        uint ignored;
        InterlockedMin(feedback[offset + 4], push.wantedMip, ignored);
        InterlockedAdd(feedback[offset + 5], push.samples, ignored);
    }
}
