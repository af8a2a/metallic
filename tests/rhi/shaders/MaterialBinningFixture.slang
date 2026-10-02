#include "../MaterialFixtureParameters.h"

[[vk::push_constant]] ConstantBuffer<MaterialFixtureParameters> gFixture;

[shader("compute")]
[numthreads(64, 1, 1)]
void materialFixtureMain(uint3 id : SV_DispatchThreadID)
{
    MaterialFixtureParameters push = gFixture;
    if (!push.records.contains(259u) || !push.instances.contains(259u) ||
        !push.materials.contains(256u) || !push.shading.contains(256u)) { return; }
    uint i = id.y * kMaterialBinMaxGroupsX * 64u + id.x;
    if (i < 260u) {
        VisibleClusterRecord record = {0u, i, 0u, 0u};
        if (i == 257u) { record.flags = 1u << 28u; }
        if (i == 258u) { record.instanceIndex = 10000u; }
        push.records.data[i] = record;
        GPUDrivenPreviewInstance instance = {};
        instance.identity.y = i == 259u ? 10000u : (i * 17u) % 257u;
        push.instances.data[i] = instance;
    }
    if (i < 257u) {
        GPUDrivenPreviewMaterial material = {};
        material.identity.z = i == 0u ? 10000u : (i * 73u) % 257u;
        push.materials.data[i] = material;
        // Change executable features while keeping source material IDs stable.
        uint type = 1u + (i + push.bin) % 4u;
        PathTraceMaterial shading = {};
        shading.params.x = type == 1u ? 0.0 : (type == 2u ? 1.0 : 0.5);
        shading.glassParams.x = type == 4u ? 1.0 : 0.0;
        shading.metallicRoughnessTexture.textureIndex = 0xffffffffu;
        shading.metallicRoughnessTexture.ntcTextureSetIndex = 0xffffffffu;
        if (i == 1u || i == 2u) {
            shading.params.x = 1.0;
            shading.glassParams.x = 0.0;
            if (i == 1u) { shading.metallicRoughnessTexture.textureIndex = 7u; }
            else { shading.metallicRoughnessTexture.ntcTextureSetIndex = 0u; }
        }
        push.shading.data[i] = shading;
    }
    if (i < push.width * push.height) {
        uint kind = i % 262u;
        // A dense tile list crosses Vulkan's guaranteed 65535-group X limit.
        if (push.width > 4096u) { kind = 0u; }
        uint record = kind < 257u ? kind : kind - 2u;
        uint visibility = ((record + 1u) << 7u) | (i % 127u);
        if (kind == 257u || push.width == 8u) { visibility = 0u; }
        if (kind == 258u) { visibility = 10001u << 7u; }
        resolveDescriptor(push.visibility)[uint2(i % push.width, i / push.width)] = visibility;
    }
}

