#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <cgltf.h>

#include "material_loader.h"
#include <iostream>
#include <filesystem>
#include <algorithm>

static MTL::Texture* createTextureFromImage(MTL::Device* device, MTL::CommandQueue* commandQueue,
                                            const std::string& imagePath) {
    int w, h, ch;
    unsigned char* pixels = stbi_load(imagePath.c_str(), &w, &h, &ch, 4);
    if (!pixels) {
        std::cerr << "Failed to load image: " << imagePath << std::endl;
        return nullptr;
    }

    // Compute mip levels
    uint32_t mipLevels = 1;
    {
        int dim = std::max(w, h);
        while (dim > 1) { dim >>= 1; mipLevels++; }
    }

    auto* texDesc = MTL::TextureDescriptor::texture2DDescriptor(
        MTL::PixelFormatRGBA8Unorm, w, h, true);
    texDesc->setMipmapLevelCount(mipLevels);
    texDesc->setStorageMode(MTL::StorageModeShared);
    texDesc->setUsage(MTL::TextureUsageShaderRead);

    MTL::Texture* texture = device->newTexture(texDesc);
    if (!texture) {
        stbi_image_free(pixels);
        return nullptr;
    }

    // Upload base mip level
    texture->replaceRegion(MTL::Region(0, 0, 0, w, h, 1), 0, pixels, w * 4);
    stbi_image_free(pixels);

    // Generate mipmaps via blit encoder
    MTL::CommandBuffer* cmdBuf = commandQueue->commandBuffer();
    MTL::BlitCommandEncoder* blit = cmdBuf->blitCommandEncoder();
    blit->generateMipmaps(texture);
    blit->endEncoding();
    cmdBuf->commit();
    cmdBuf->waitUntilCompleted();

    return texture;
}
bool loadGLTFMaterials(MTL::Device* device, MTL::CommandQueue* commandQueue,
                       const std::string& gltfPath, LoadedMaterials& out) {
    cgltf_options options = {};
    cgltf_data* data = nullptr;

    cgltf_result result = cgltf_parse_file(&options, gltfPath.c_str(), &data);
    if (result != cgltf_result_success) {
        std::cerr << "Failed to parse glTF for materials: " << gltfPath << std::endl;
        return false;
    }

    result = cgltf_load_buffers(&options, data, gltfPath.c_str());
    if (result != cgltf_result_success) {
        std::cerr << "Failed to load glTF buffers for materials" << std::endl;
        cgltf_free(data);
        return false;
    }

    // Resolve base directory for image paths
    std::filesystem::path basePath = std::filesystem::path(gltfPath).parent_path();

    // Load all images as textures
    const cgltf_size textureCount = std::min<cgltf_size>(data->images_count, MAX_SCENE_TEXTURES);
    if (data->images_count > textureCount) {
        std::cerr << "Warning: scene has " << data->images_count
                  << " images, but only first " << textureCount
                  << " are bound (MAX_SCENE_TEXTURES)." << std::endl;
    }

    out.textures.resize(textureCount, nullptr);
    for (cgltf_size i = 0; i < textureCount; i++) {
        const cgltf_image& img = data->images[i];
        if (!img.uri) continue;
        std::string fullPath = (basePath / img.uri).string();
        out.textures[i] = createTextureFromImage(device, commandQueue, fullPath);
        if (!out.textures[i]) {
            std::cerr << "Warning: failed to load texture " << i << ": " << fullPath << std::endl;
        }
    }

    // Create 1x1 white placeholder for any null texture slots
    MTL::Texture* placeholder = nullptr;
    for (auto*& tex : out.textures) {
        if (!tex) {
            if (!placeholder) {
                auto* desc = MTL::TextureDescriptor::texture2DDescriptor(
                    MTL::PixelFormatRGBA8Unorm, 1, 1, false);
                desc->setStorageMode(MTL::StorageModeShared);
                desc->setUsage(MTL::TextureUsageShaderRead);
                placeholder = device->newTexture(desc);
                uint32_t white = 0xFFFFFFFF;
                placeholder->replaceRegion(MTL::Region(0, 0, 0, 1, 1, 1), 0, &white, 4);
            }
            tex = placeholder->retain();
        }
    }
    if (placeholder) placeholder->release();

    std::cout << "Loaded " << out.textures.size() << " textures" << std::endl;

    auto resolveTextureIndex = [&](const cgltf_texture* tex) -> uint32_t {
        if (!tex || !tex->image)
            return INVALID_TEXTURE_INDEX;

        const uint32_t imageIndex = static_cast<uint32_t>(tex->image - data->images);
        if (imageIndex >= out.textures.size())
            return INVALID_TEXTURE_INDEX;

        return imageIndex;
    };

    // Build material array
    std::vector<GPUMaterial> materials(data->materials_count);
    for (cgltf_size i = 0; i < data->materials_count; i++) {
        const cgltf_material& mat = data->materials[i];
        GPUMaterial& gpu = materials[i];

        gpu.baseColorTexIndex = INVALID_TEXTURE_INDEX;
        gpu.normalTexIndex = INVALID_TEXTURE_INDEX;
        gpu.metallicRoughnessTexIndex = INVALID_TEXTURE_INDEX;

        // Base color texture
        if (mat.has_pbr_metallic_roughness) {
            const auto& pbr = mat.pbr_metallic_roughness;
            gpu.baseColorTexIndex = resolveTextureIndex(pbr.base_color_texture.texture);
            gpu.baseColorFactor[0] = pbr.base_color_factor[0];
            gpu.baseColorFactor[1] = pbr.base_color_factor[1];
            gpu.baseColorFactor[2] = pbr.base_color_factor[2];
            gpu.baseColorFactor[3] = pbr.base_color_factor[3];
            gpu.metallicFactor = pbr.metallic_factor;
            gpu.roughnessFactor = pbr.roughness_factor;

            // Metallic-roughness texture
            gpu.metallicRoughnessTexIndex = resolveTextureIndex(pbr.metallic_roughness_texture.texture);
        } else {
            gpu.baseColorFactor[0] = 1.0f;
            gpu.baseColorFactor[1] = 1.0f;
            gpu.baseColorFactor[2] = 1.0f;
            gpu.baseColorFactor[3] = 1.0f;
            gpu.metallicFactor = 1.0f;
            gpu.roughnessFactor = 1.0f;
        }

        // Normal texture
        gpu.normalTexIndex = resolveTextureIndex(mat.normal_texture.texture);

        // Alpha mode
        gpu.alphaMode = (mat.alpha_mode == cgltf_alpha_mode_mask) ? 1 : 0;
        gpu.alphaCutoff = mat.alpha_cutoff;
        gpu._pad = 0.0f;
    }

    out.materialCount = static_cast<uint32_t>(materials.size());
    out.materialBuffer = device->newBuffer(
        materials.data(), materials.size() * sizeof(GPUMaterial),
        MTL::ResourceStorageModeShared);

    std::cout << "Loaded " << out.materialCount << " materials" << std::endl;

    // Create shared sampler (LINEAR + mipmap + REPEAT)
    auto* samplerDesc = MTL::SamplerDescriptor::alloc()->init();
    samplerDesc->setMinFilter(MTL::SamplerMinMagFilterLinear);
    samplerDesc->setMagFilter(MTL::SamplerMinMagFilterLinear);
    samplerDesc->setMipFilter(MTL::SamplerMipFilterLinear);
    samplerDesc->setSAddressMode(MTL::SamplerAddressModeRepeat);
    samplerDesc->setTAddressMode(MTL::SamplerAddressModeRepeat);
    out.sampler = device->newSamplerState(samplerDesc);
    samplerDesc->release();

    cgltf_free(data);
    return true;
}
