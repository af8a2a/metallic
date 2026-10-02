// P0 fixture: the selector is explicit; unrelated invocations do not print.
#include "../ShaderDiagnosticParameters.h"
[[vk::push_constant]] ConstantBuffer<ShaderDiagnosticParameters> gDiagnostic;

#ifndef PRINTF_RECORD_COUNT
#define PRINTF_RECORD_COUNT 1
#endif

[shader("compute")]
[numthreads(32, 1, 1)]
void ordinaryMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    if (group.x == 1 && lane == 3) {
        for (uint seq = 0; seq < PRINTF_RECORD_COUNT; ++seq) {
            printf("MTP0 ordinary seq=%u group=%u lane=%u value=%u\n", seq, group.x, lane, 73u);
        }
    }
}

[shader("compute")]
[numthreads(32, 1, 1)]
void heapMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    if (group.x == 1 && lane == 3) {
        ShaderDiagnosticParameters push = gDiagnostic;
        ByteAddressBuffer input = resolveDescriptor(push.input);
        uint2 outputHandle = input.Load2(0);
        uint value = input.Load(8);
        RWByteAddressBuffer output = resolveDescriptor(DescriptorHandle<RWByteAddressBuffer>(outputHandle));
        output.Store4(0, uint4(value + 1, push.cookie, value, outputHandle.x));
        for (uint seq = 0; seq < PRINTF_RECORD_COUNT; ++seq) {
            printf("MTP0 heap seq=%u group=%u lane=%u value=%u cookie=%u\n", seq, group.x, lane, value, push.cookie);
        }
    }
}
