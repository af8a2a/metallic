#ifndef METALLIC_MATERIAL_BINNING_COMMON
#define METALLIC_MATERIAL_BINNING_COMMON

static const uint kMaterialBinGroupSize = 64u;
// Vulkan's guaranteed minimum maxComputeWorkGroupCount[0]. Large bins spill
// into Y; the resolve uses the same row stride, with no CPU count readback.
static const uint kMaterialBinMaxGroupsX = 65535u;

struct MaterialBin
{
    uint offset;
    uint count;
};

#endif // METALLIC_MATERIAL_BINNING_COMMON
