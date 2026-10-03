import Core;
using Metallic;
#include "../TestResourceParameters.h"

[shader("compute")]
[numthreads(1, 1, 1)]
void viewConstantsProbeMain()
{
    let resources = getResourceParameters<ViewConstantsProbeResources>();
    let output = resolveBuffer<RWStructuredBuffer<ViewConstants>>(resources.output);
    let input = resolveBuffer<StructuredBuffer<ViewConstants>>(resources.input);
    // Keep typed field offsets observable. A whole nested-aggregate assignment
    // crashes the NVIDIA native-heap pipeline compiler for this probe.
    output[0].current.eye = input[0].current.eye;
    output[0].current.center = input[0].current.center;
    output[0].current.upProjection = input[0].current.upProjection;
    output[0].current.viewport = input[0].current.viewport;
    output[0].current.clipOrtho = input[0].current.clipOrtho;
    output[0].previous.eye = input[0].previous.eye;
    output[0].previous.center = input[0].previous.center;
    output[0].previous.upProjection = input[0].previous.upProjection;
    output[0].previous.viewport = input[0].previous.viewport;
    output[0].previous.clipOrtho = input[0].previous.clipOrtho;
    output[0].jitter = input[0].jitter;
    output[0].frame = input[0].frame;
    output[0].outputSize = input[0].outputSize;
}
