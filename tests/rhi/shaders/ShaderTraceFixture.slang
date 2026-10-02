import ShaderTrace;
#include "../ShaderDiagnosticParameters.h"
[[vk::push_constant]] ConstantBuffer<ShaderDiagnosticParameters> gDiagnostic;

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
        ShaderTraceIdentity id;
        id.session = uint2(TRACE_SESSION_LO, TRACE_SESSION_HI);
        id.run = uint2(TRACE_RUN_LO, TRACE_RUN_HI);
        id.dispatch = uint2(TRACE_DISPATCH_LO, TRACE_DISPATCH_HI);
        id.site = 1;
        id.group = group;
        id.localIndex = lane;
        shaderTraceBegin(id);
        uint evaluated = TRACE_SCENARIO == 2 ? 0 : TRACE_SCENARIO == 4 ? 15 : 1;
        uint matched = TRACE_SCENARIO == 1 ? 0 : evaluated;
        uint emitted = 0;
        for (uint i = 0; i < matched; ++i) {
            if (emitted < TRACE_MAX_RECORDS - 2) {
                shaderTraceWords5(id, 1, emitted + 1,
                    uint4(asuint(asfloat(0x80000000u)), asuint(asfloat(0x7fc12345u)), 1, 0x20000000u), value);
                ++emitted;
            }
            // Fixture computation continues even when the logging quota is exhausted.
        }
        if (TRACE_SCENARIO != 3) {
            shaderTraceWords5(id, 2, emitted + 1, uint4(evaluated, matched, emitted, 0), matched > emitted ? 1 : 0);
        }
    }
}
