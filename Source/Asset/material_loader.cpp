#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <cgltf.h>

#include "material_loader.h"
#include "rhi_resource_utils.h"

#include <spdlog/spdlog.h>
#include <filesystem>
#include <algorithm>

static RhiTextureHandle createTextureFromImage(const RhiDevice& device,
                                               const RhiCommandQueue& commandQueue,
                                               const std::string& imagePath) {
    int w, h, ch;
    unsigned char* pixels = stbi_load(imagePath.c_str(), &w, &h, &ch, 4);
    if (!pixels) {
        spdlog::warn("Failed to load image: {}", imagePath);
        return {};
    }

    // Compute mip levels
    uint32_t mipLevels = 1;
    {
        int dim = std::max(w, h);
        while (dim > 1) { dim >>= 1; mipLevels++; }
    }

    RhiTextureHandle texture = rhiCreateTexture2D(device,
                                                  static_cast<uint32_t>(w),
                                                  static_cast<uint32_t>(h),
                                                  RhiFormat::RGBA8Unorm,
                                                  true,
                                                  mipLevels,
                                                  RhiTextureStorageMode::Shared,
                                                  RhiTextureUsage::ShaderRead);
    if (!texture.nativeHandle()) {
        stbi_image_free(pixels);
        return {};
    }

    // Upload base mip level
    rhiUploadTexture2D(texture,
                       static_cast<uint32_t>(w),
                       static_cast<uint32_t>(h),
                       pixels,
                       static_cast<size_t>(w) * 4u);
    stbi_image_free(pixels);

    rhiGenerateMipmaps(commandQueue, texture);

    return texture;
}

bool loadGLTFMaterials(const RhiDevice& device, const RhiCommandQueue& commandQueue,
                       const std::string& gltfPath, LoadedMaterials& out) {
    cgltf_options options = {};
    cgltf_data* data = nullptr;

    cgltf_result result = cgltf_parse_file(&options, gltfPath.c_str(), &data);
    if (result != cgltf_result_success) {
        spdlog::error("Failed to parse glTF for materials: {}", gltfPath);
        return false;
    }

    result = cgltf_load_buffers(&options, data, gltfPath.c_str());
    if (result != cgltf_result_success) {
        spdlog::error("Failed to load glTF buffers for materials");
        cgltf_free(data);
        return false;
    }

    // Resolve base directory for image paths
    std::filesystem::path basePath = std::filesystem::path(gltfPath).parent_path();

    // Load all images as textures
    const cgltf_size textureCount = std::min<cgltf_size>(data->images_count, MAX_SCENE_TEXTURES);
    if (data->images_count > textureCount) {
        spdlog::warn("Scene has {} images, but only first {} are bound (MAX_SCENE_TEXTURES)",
                     data->images_count, textureCount);
    }

    out.textures.resize(textureCount);
    for (cgltf_size i = 0; i < textureCount; i++) {
        const cgltf_image& img = data->images[i];
        if (!img.uri) continue;
        std::string fullPath = (basePath / img.uri).string();
        out.textures[i] = createTextureFromImage(device, commandQueue, fullPath);
        if (!out.textures[i].nativeHandle()) {
            spdlog::warn("Failed to load texture {}: {}", i, fullPath);
        }
    }

    // Create 1x1 white placeholder for any null texture slots
    RhiTextureHandle placeholder;
    for (auto& tex : out.textures) {
        if (!tex.nativeHandle()) {
            if (!placeholder.nativeHandle()) {
                placeholder = rhiCreateTexture2D(device,
                                                 1,
                                                 1,
                                                 RhiFormat::RGBA8Unorm,
                                                 false,
                                                 1,
                                                 RhiTextureStorageMode::Shared,
                                                 RhiTextureUsage::ShaderRead);
                uint32_t white = 0xFFFFFFFF;
                rhiUploadTexture2D(placeholder, 1, 1, &white, 4);
            }
            tex = rhiRetainTexture(placeholder);
        }
    }
    rhiReleaseHandle(placeholder);

    spdlog::info("Loaded {} textures", out.textures.size());

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
    out.materialBuffer = rhiCreateSharedBuffer(
        device, materials.data(), materials.size() * sizeof(GPUMaterial), "Materials");

    spdlog::info("Loaded {} materials", out.materialCount);

    // Create shared sampler (LINEAR + mipmap + REPEAT)
    RhiSamplerDesc samplerDesc;
    samplerDesc.minFilter = RhiSamplerFilterMode::Linear;
    samplerDesc.magFilter = RhiSamplerFilterMode::Linear;
    samplerDesc.mipFilter = RhiSamplerMipFilterMode::Linear;
    samplerDesc.addressModeS = RhiSamplerAddressMode::Repeat;
    samplerDesc.addressModeT = RhiSamplerAddressMode::Repeat;
    out.sampler = rhiCreateSampler(device, samplerDesc);
    out.textureViews.clear();
    out.textureViews.reserve(out.textures.size());
    for (const auto& texture : out.textures) {
        out.textureViews.push_back(&texture);
    }

    cgltf_free(data);
    return true;
}
