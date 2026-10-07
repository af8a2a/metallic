#include "Runtime/Render/RayTracing/SceneCoverage.h"

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <utility>

namespace metallic::render {
namespace {

constexpr uint64_t kMaxCoverageSnapshotBytes = 256ull * 1024 * 1024;
constexpr uint64_t kMaxCoverageImageTexels = 16ull * 1024 * 1024;

bool validImageDimensions(uint32_t width, uint32_t height, uint64_t budget)
{
    return width != 0 && height != 0 &&
        uint64_t(width) + 1 <= kMaxCoverageImageTexels / (uint64_t(height) + 1) &&
        uint64_t(width) * height * 4 <= budget;
}

bool validImageSnapshot(const scene::RenderImage::Mip& image, uint64_t budget)
{
    return validImageDimensions(image.width, image.height, budget) &&
        image.pixels.size() >= uint64_t(image.width) * image.height * 4 && image.pixels.size() <= budget;
}

bool usesAlpha(const scene::RenderMaterial& material)
{
    return material.alphaMode == "MASK" || material.alphaMode == "BLEND";
}

// Preserve the material loader's fallback when decoding is unavailable. Neural
// textures are deliberately excluded: their reconstructed alpha can differ.
bool loadAlphaImage(const scene::Scene& scene, int32_t textureIndex, scene::RenderImage::Mip& decoded,
    const scene::RenderImage::Mip*& mip, uint64_t budget)
{
    mip = nullptr;
    if (textureIndex < 0 || size_t(textureIndex) >= scene.textures().size()) {
        return true;
    }
    const auto& texture = scene.textures()[textureIndex];
    if (texture.hasNeuralSource()) {
        return false;
    }
    if (texture.imageIndex < 0 || size_t(texture.imageIndex) >= scene.images().size()) {
        return true;
    }
    const auto& image = scene.images()[texture.imageIndex];
    if (!image.decodedMips.empty()) {
        if (!validImageSnapshot(image.decodedMips.front(), budget)) {
            return false;
        }
        mip = &image.decodedMips.front();
        return true;
    }
    if (image.decodeAttempted || image.channelComposition.has_value()) {
        return false;
    }
    int width = 0, height = 0, channels = 0;
    stbi_uc* pixels = nullptr;
    if (!image.encodedData.empty() && image.encodedData.size() <= size_t(std::numeric_limits<int>::max())) {
        if (!stbi_info_from_memory(image.encodedData.data(), int(image.encodedData.size()), &width, &height, &channels) ||
            width <= 0 || height <= 0 || !validImageDimensions(uint32_t(width), uint32_t(height), budget)) {
            return false;
        }
        pixels = stbi_load_from_memory(image.encodedData.data(), int(image.encodedData.size()), &width, &height, &channels, 4);
    } else if (!image.uri.empty() && !image.uri.starts_with("data:")) {
        std::filesystem::path path = image.uri;
        if (path.is_relative()) {
            path = scene.filename().parent_path() / path;
        }
        if (!stbi_info(path.string().c_str(), &width, &height, &channels) || width <= 0 || height <= 0 ||
            !validImageDimensions(uint32_t(width), uint32_t(height), budget)) {
            return false;
        }
        pixels = stbi_load(path.string().c_str(), &width, &height, &channels, 4);
    }
    if (pixels == nullptr || width <= 0 || height <= 0 ||
        !validImageDimensions(uint32_t(width), uint32_t(height), budget)) {
        stbi_image_free(pixels);
        return false;
    }
    decoded.width = uint32_t(width);
    decoded.height = uint32_t(height);
    decoded.pixels.assign(pixels, pixels + size_t(width) * height * 4);
    stbi_image_free(pixels);
    mip = &decoded;
    return true;
}

} // namespace

SceneCoverageInput::SceneCoverageInput(const scene::RenderMaterial& material,
    std::shared_ptr<const scene::RenderImage::Mip> image,
    const scene::RenderPrimitive& primitive)
    : image_(std::move(image))
{
    const auto& transform = material.baseColorTexture.uvTransform;
    if (!usesAlpha(material) || !material.valueProgram.empty() || primitive.mode != 4 ||
        !std::isfinite(material.baseColorFactor.w) || !std::isfinite(material.alphaCutoff) ||
        std::any_of(transform.begin(), transform.end(), [](float value) { return !std::isfinite(value); })) {
        return;
    }
    if (image_ != nullptr && !validImageSnapshot(*image_, kMaxCoverageSnapshotBytes)) {
        return;
    }
    const size_t count = (primitive.indices.empty() ? primitive.positions.size() : primitive.indices.size()) / 3;
    const uint64_t imageBytes = image_ != nullptr ? image_->pixels.size() : 0;
    if (count == 0 || count > std::numeric_limits<uint32_t>::max() ||
        count > (kMaxCoverageSnapshotBytes - imageBytes) / sizeof(RayTracingCoverageTriangle)) {
        return;
    }
    triangles_.resize(count);
    for (size_t triangle = 0; triangle < count; ++triangle) {
        for (uint32_t vertex = 0; vertex < 3; ++vertex) {
            const size_t index = primitive.indices.empty() ? triangle * 3 + vertex : primitive.indices[triangle * 3 + vertex];
            if (index >= primitive.positions.size()) {
                triangles_.clear();
                return;
            }
            const float2 source = index < primitive.texcoords0.size() ? primitive.texcoords0[index] : float2{0.0f, 0.0f};
            triangles_[triangle].uv[vertex] = {
                transform[0] * source.x + transform[1] * source.y + transform[2],
                transform[3] * source.x + transform[4] * source.y + transform[5],
            };
        }
    }
    desc_.mode = material.alphaMode == "BLEND" ? RayTracingCoverageMode::Blend : RayTracingCoverageMode::Mask;
    desc_.alphaFactor = material.baseColorFactor.w;
    desc_.alphaCutoff = material.alphaCutoff;
    desc_.triangles = triangles_;
    if (image_ != nullptr) {
        desc_.width = image_->width;
        desc_.height = image_->height;
        desc_.pixelsRGBA8 = image_->pixels;
    }
}

std::vector<ScenePrimitiveCoverage> scenePrimitiveCoverage(const scene::Scene& scene)
{
    std::vector<ScenePrimitiveCoverage> result(scene.renderPrimitives().size());
    for (const auto& node : scene.renderNodes()) {
        if (node.renderPrimitiveIndex < 0 || size_t(node.renderPrimitiveIndex) >= result.size()) {
            continue;
        }
        const int32_t material = node.materialIndex >= 0 && size_t(node.materialIndex) < scene.materials().size() ? node.materialIndex : 0;
        auto& coverage = result[node.renderPrimitiveIndex];
        if (coverage.materialIndex == -2) {
            coverage.materialIndex = material;
        } else if (coverage.materialIndex != material) {
            coverage.materialIndex = -1;
        }
        if (size_t(material) < scene.materials().size()) {
            coverage.usesAlpha |= usesAlpha(scene.materials()[material]);
        }
    }
    return result;
}

std::vector<std::unique_ptr<const SceneCoverageInput>> makeSceneCoverageInputs(
    const scene::Scene& scene, std::span<const ScenePrimitiveCoverage> coverage)
{
    std::vector<std::unique_ptr<const SceneCoverageInput>> result(scene.renderPrimitives().size());
    if (coverage.size() != result.size()) {
        return result;
    }
    std::map<int32_t, std::vector<uint32_t>> groups;
    for (uint32_t index = 0; index < scene.renderPrimitives().size(); ++index) {
        const int32_t material = coverage[index].materialIndex;
        if (material >= 0 && size_t(material) < scene.materials().size() && usesAlpha(scene.materials()[material])) {
            groups[material].push_back(index);
        }
    }
    uint64_t snapshotBytes = 0;
    for (const auto& [materialIndex, primitives] : groups) {
        const auto& material = scene.materials()[materialIndex];
        // Live Coverage must keep regular triangle candidates. A snapshot of its
        // current parameters cannot establish immutable opaque/transparent areas.
        const auto& transform = material.baseColorTexture.uvTransform;
        if (!material.valueProgram.empty() || !std::isfinite(material.baseColorFactor.w) ||
            !std::isfinite(material.alphaCutoff) ||
            std::any_of(transform.begin(), transform.end(), [](float value) { return !std::isfinite(value); })) {
            continue;
        }
        const uint64_t remainingBytes = kMaxCoverageSnapshotBytes - snapshotBytes;
        uint64_t smallestTriangleBytes = remainingBytes + 1;
        for (uint32_t index : primitives) {
            const auto& primitive = scene.renderPrimitives()[index];
            const size_t count = (primitive.indices.empty() ? primitive.positions.size() : primitive.indices.size()) / 3;
            if (primitive.mode == 4 && count != 0 && count <= std::numeric_limits<uint32_t>::max()) {
                smallestTriangleBytes = std::min(smallestTriangleBytes, uint64_t(count) * sizeof(RayTracingCoverageTriangle));
            }
        }
        if (smallestTriangleBytes > remainingBytes) {
            continue;
        }
        scene::RenderImage::Mip decoded;
        const scene::RenderImage::Mip* image = nullptr;
        if (!loadAlphaImage(scene, material.baseColorTexture.textureIndex, decoded, image,
            remainingBytes - smallestTriangleBytes)) {
            continue;
        }
        std::shared_ptr<const scene::RenderImage::Mip> snapshot;
        const uint64_t imageBytes = image != nullptr ? image->pixels.size() : 0;
        bool retainedImage = false;
        for (uint32_t index : primitives) {
            const auto& primitive = scene.renderPrimitives()[index];
            const size_t count = (primitive.indices.empty() ? primitive.positions.size() : primitive.indices.size()) / 3;
            if (primitive.mode != 4 || count == 0 || count > std::numeric_limits<uint32_t>::max()) {
                continue;
            }
            const uint64_t inputBytes = uint64_t(count) * sizeof(RayTracingCoverageTriangle) +
                (retainedImage ? 0 : imageBytes);
            // Reject before allocating either the shared alpha copy or triangle
            // UVs. Budget exhaustion retains ordinary live coverage evaluation.
            if (inputBytes > kMaxCoverageSnapshotBytes - snapshotBytes) {
                continue;
            }
            if (image != nullptr && snapshot == nullptr) {
                snapshot = image == &decoded
                    ? std::make_shared<scene::RenderImage::Mip>(std::move(decoded))
                    : std::make_shared<scene::RenderImage::Mip>(*image);
            }
            auto input = std::make_unique<SceneCoverageInput>(material, snapshot, primitive);
            if (input->valid()) {
                snapshotBytes += inputBytes;
                retainedImage = true;
                result[index] = std::move(input);
            }
        }
    }
    return result;
}

} // namespace metallic::render
