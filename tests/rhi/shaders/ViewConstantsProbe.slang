#include "../../../Shaders/Libraries/Resources/ComputeResources.slang"
#include "../../../Shaders/Libraries/Camera/ViewConstants.slang"

[shader("compute")]
[numthreads(1, 1, 1)]
void viewConstantsProbeMain()
{
    METALLIC_RESOURCE(RWStructuredBuffer<ViewConstants>, 0)[0] =
        METALLIC_RESOURCE(StructuredBuffer<ViewConstants>, 1)[0];
}
