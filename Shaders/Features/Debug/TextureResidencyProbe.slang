#include "../../../Source/Runtime/Render/Core/TextureResidencyProbeParameters.h"
using Metallic;
[[vk::push_constant]] ConstantBuffer<TextureResidencyProbeParameters> push;
[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    uint count = push.feedback.length();
    if (push.slot >= count / 8u) { return; }
    uint* feedback = push.feedback.data;
    uint offset = push.slot * 8;
    if (offset + 7 < count && feedback[offset] != 0 && push.samples != 0) {
        uint ignored;
        InterlockedMin(feedback[offset + 4], push.wantedMip, ignored);
        InterlockedAdd(feedback[offset + 5], push.samples, ignored);
    }
}
