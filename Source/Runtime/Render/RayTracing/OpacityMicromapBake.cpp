#include "Runtime/Render/RayTracing/OpacityMicromapBake.h"

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

namespace metallic::render {
namespace {

// Adapted from Khronos Vulkan's VK_KHR_opacity_micromap Reference Code,
// Copyright 2022-2026 The Khronos Group Inc., CC-BY-4.0.
// https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_opacity_micromap.html
// Used only for strictly interior microtriangle centroids, so no edge clamping is needed.
uint32_t microtriangleIndex(float u, float v, uint32_t level)
{
    const uint32_t scale = 1u << level;
    const float fu = u * float(scale), fv = v * float(scale);
    const uint32_t iu = uint32_t(fu), iv = uint32_t(fv);
    uint32_t iw = ~(iu + iv);
    if ((fu - float(iu)) + (fv - float(iv)) >= 1.0f && iu + iv < scale - 1u) {
        --iw;
    }
    uint32_t b0 = ~(iu ^ iw) & (scale - 1u);
    const uint32_t t = (iu ^ iv) & b0;
    uint32_t f = t;
    f ^= f >> 1u;
    f ^= f >> 2u;
    f ^= f >> 4u;
    f ^= f >> 8u;
    uint32_t b1 = ((f ^ iu) & ~b0) | t;
    const auto interleave = [](uint32_t bits) {
        bits = (bits | (bits << 8u)) & 0x00ff00ffu;
        bits = (bits | (bits << 4u)) & 0x0f0f0f0fu;
        bits = (bits | (bits << 2u)) & 0x33333333u;
        return (bits | (bits << 1u)) & 0x55555555u;
    };
    return interleave(b0) | (interleave(b1) << 1u);
}

bool usesAlpha(const scene::RenderMaterial& material)
{
    return material.alphaMode == "MASK" || material.alphaMode == "BLEND";
}

// Preserve the material loader's fallback when decoding is unavailable. Neural
// textures are deliberately excluded: their reconstructed alpha can differ.
bool loadAlphaImage(const scene::Scene& scene, int32_t textureIndex, scene::RenderImage::Mip& decoded,
    const scene::RenderImage::Mip*& mip)
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
        mip = &image.decodedMips.front();
        return true;
    }
    if (image.decodeAttempted || image.channelComposition.has_value()) {
        return false;
    }
    int width = 0, height = 0, channels = 0;
    stbi_uc* pixels = nullptr;
    if (!image.encodedData.empty() && image.encodedData.size() <= size_t(std::numeric_limits<int>::max())) {
        pixels = stbi_load_from_memory(image.encodedData.data(), int(image.encodedData.size()), &width, &height, &channels, 4);
    } else if (!image.uri.empty() && !image.uri.starts_with("data:")) {
        std::filesystem::path path = image.uri;
        if (path.is_relative()) {
            path = scene.filename().parent_path() / path;
        }
        pixels = stbi_load(path.string().c_str(), &width, &height, &channels, 4);
    }
    if (pixels == nullptr || width <= 0 || height <= 0) {
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

OpacityMicromapBaker::OpacityMicromapBaker(const scene::RenderMaterial& material, const scene::RenderImage::Mip* image)
    : textureInfo_(material.baseColorTexture)
{
    if (!usesAlpha(material) || !std::isfinite(material.baseColorFactor.w) || !std::isfinite(material.alphaCutoff) ||
        std::any_of(textureInfo_.uvTransform.begin(), textureInfo_.uvTransform.end(), [](float v) { return !std::isfinite(v); })) {
        return;
    }
    if (image != nullptr) {
        width_ = image->width;
        height_ = image->height;
        // Bound CPU bake memory to 128 MiB per material (two coverage tables).
        if (width_ == 0 || height_ == 0 || (uint64_t(width_) + 1) * (uint64_t(height_) + 1) > 16ull * 1024 * 1024 ||
            image->pixels.size() < uint64_t(width_) * height_ * 4) {
            return;
        }
    }
    const size_t stride = size_t(width_) + 1;
    opaquePrefix_.resize(stride * (size_t(height_) + 1), 0);
    unknownPrefix_.resize(opaquePrefix_.size(), 0);
    for (uint32_t y = 0; y < height_; ++y) {
        uint32_t rowOpaque = 0, rowUnknown = 0;
        for (uint32_t x = 0; x < width_; ++x) {
            const float textureAlpha = image != nullptr ? float(image->pixels[(size_t(y) * width_ + x) * 4 + 3]) / 255.0f : 1.0f;
            const float alpha = std::clamp(material.baseColorFactor.w * textureAlpha, 0.0f, 1.0f);
            const bool opaque = material.alphaMode == "MASK" ? alpha >= std::clamp(material.alphaCutoff, 0.0f, 1.0f) : alpha >= 1.0f;
            rowOpaque += opaque;
            rowUnknown += material.alphaMode == "BLEND" && alpha > 0.0f && alpha < 1.0f;
            const size_t index = (size_t(y) + 1) * stride + x + 1;
            opaquePrefix_[index] = opaquePrefix_[index - stride] + rowOpaque;
            unknownPrefix_[index] = unknownPrefix_[index - stride] + rowUnknown;
        }
    }
    valid_ = true;
}

uint8_t OpacityMicromapBaker::classify(const std::array<float2, 3>& uv) const
{
    if (std::any_of(uv.begin(), uv.end(), [](const float2& value) {
            return !std::isfinite(value.x) || !std::isfinite(value.y);
        })) {
        return 3;
    }
    double minX = std::min({double(uv[0].x), double(uv[1].x), double(uv[2].x)});
    double maxX = std::max({double(uv[0].x), double(uv[1].x), double(uv[2].x)});
    double minY = std::min({double(uv[0].y), double(uv[1].y), double(uv[2].y)});
    double maxY = std::max({double(uv[0].y), double(uv[1].y), double(uv[2].y)});
    if (!std::isfinite(minX + maxX + minY + maxY) ||
        std::max({std::abs(minX), std::abs(maxX), std::abs(minY), std::abs(maxY)}) > 1048576.0) {
        return 3;
    }
    // Include round-off at microtriangle/texel boundaries, including wrap seams.
    const double pad = 0.000004 * (1.0 + std::max({std::abs(minX), std::abs(maxX), std::abs(minY), std::abs(maxY)}));
    const auto intervals = [pad](double low, double high, uint32_t dimension) {
        std::array<std::array<uint32_t, 2>, 2> ranges{};
        int64_t begin = int64_t(std::floor((low - pad) * dimension));
        int64_t end = int64_t(std::floor((high + pad) * dimension)) + 1;
        if (end - begin >= dimension) {
            ranges[0] = {0, dimension};
        } else {
            const uint32_t first = uint32_t((begin % dimension + dimension) % dimension);
            const uint32_t last = first + uint32_t(end - begin);
            ranges[0] = {first, std::min(last, dimension)};
            if (last > dimension) {
                ranges[1] = {0, last - dimension};
            }
        }
        return ranges;
    };
    const auto xs = intervals(minX, maxX, width_), ys = intervals(minY, maxY, height_);
    uint64_t area = 0, opaque = 0, unknown = 0;
    const size_t stride = size_t(width_) + 1;
    for (const auto& x : xs) {
        for (const auto& y : ys) {
            if (x[0] == x[1] || y[0] == y[1]) {
                continue;
            }
            const auto sum = [&](const std::vector<uint32_t>& prefix) -> uint32_t {
                return prefix[size_t(y[1]) * stride + x[1]] - prefix[size_t(y[0]) * stride + x[1]] -
                    prefix[size_t(y[1]) * stride + x[0]] + prefix[size_t(y[0]) * stride + x[0]];
            };
            area += uint64_t(x[1] - x[0]) * (y[1] - y[0]);
            opaque += sum(opaquePrefix_);
            unknown += sum(unknownPrefix_);
        }
    }
    return unknown != 0 ? 3 : opaque == 0 ? 0 : opaque == area ? 1 : 3;
}

bool OpacityMicromapBaker::bake(const scene::RenderPrimitive& primitive, uint32_t level, BakedOpacityMicromap& output) const
{
    output = {};
    const size_t count = (primitive.indices.empty() ? primitive.positions.size() : primitive.indices.size()) / 3;
    if (!valid_ || level > 8 || count == 0 || count > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    const uint32_t microCount = 1u << (2 * level), scale = 1u << level;
    // Avoid an unbounded scene bake; callers retain normal alpha traversal on failure.
    constexpr size_t kMaxBakeBytes = 256ull * 1024 * 1024;
    if (count > kMaxBakeBytes / (sizeof(OpacityMicromapTriangle) + std::max(1u, microCount / 4))) {
        return false;
    }
    std::vector<uint32_t> usageCounts(level + 1, 0);
    output.triangles.reserve(count);
    for (size_t triangle = 0; triangle < count; ++triangle) {
        std::array<float2, 3> uv;
        for (uint32_t vertex = 0; vertex < 3; ++vertex) {
            const size_t index = primitive.indices.empty() ? triangle * 3 + vertex : primitive.indices[triangle * 3 + vertex];
            if (index >= primitive.positions.size()) {
                output = {};
                return false;
            }
            const float2 source = index < primitive.texcoords0.size() ? primitive.texcoords0[index] : float2{0.0f, 0.0f};
            const auto& t = textureInfo_.uvTransform;
            uv[vertex] = {t[0] * source.x + t[1] * source.y + t[2], t[3] * source.x + t[4] * source.y + t[5]};
        }
        const uint8_t wholeState = classify(uv);
        const uint32_t actualLevel = wholeState < 2 ? 0 : level;
        const uint32_t offset = uint32_t(output.data.size());
        const uint32_t byteCount = std::max(1u, (1u << (actualLevel * 2)) / 4);
        output.triangles.push_back({offset, uint16_t(actualLevel), OpacityMicromapFormat::FourState});
        ++usageCounts[actualLevel];
        output.data.resize(offset + byteCount, 0);
        if (actualLevel == 0) {
            output.data[offset] = wholeState;
            ++output.stateCounts[wholeState];
            continue;
        }
        const auto store = [&](std::array<float2, 3> bary) {
            std::array<float2, 3> microUv;
            float2 centroid{0.0f, 0.0f};
            for (size_t i = 0; i < 3; ++i) {
                const float u = bary[i].x / float(scale), v = bary[i].y / float(scale);
                centroid.x += u / 3.0f;
                centroid.y += v / 3.0f;
                microUv[i] = {uv[0].x * (1.0f - u - v) + uv[1].x * u + uv[2].x * v,
                    uv[0].y * (1.0f - u - v) + uv[1].y * u + uv[2].y * v};
            }
            const uint32_t index = microtriangleIndex(centroid.x, centroid.y, level);
            const uint8_t state = classify(microUv);
            output.data[offset + index / 4] |= uint8_t(state << ((index % 4) * 2));
            ++output.stateCounts[state];
        };
        for (uint32_t y = 0; y < scale; ++y) {
            for (uint32_t x = 0; x < scale - y; ++x) {
                store({float2{float(x), float(y)}, float2{float(x + 1), float(y)}, float2{float(x), float(y + 1)}});
                if (x + y + 1 < scale) {
                    store({float2{float(x + 1), float(y)}, float2{float(x + 1), float(y + 1)}, float2{float(x), float(y + 1)}});
                }
            }
        }
    }
    for (uint32_t i = 0; i <= level; ++i) {
        if (usageCounts[i] != 0) {
            output.usages.push_back({usageCounts[i], i, OpacityMicromapFormat::FourState});
        }
    }
    return true;
}

std::vector<ScenePrimitiveOpacity> scenePrimitiveOpacity(const scene::Scene& scene)
{
    std::vector<ScenePrimitiveOpacity> result(scene.renderPrimitives().size());
    for (const auto& node : scene.renderNodes()) {
        if (node.renderPrimitiveIndex < 0 || size_t(node.renderPrimitiveIndex) >= result.size()) {
            continue;
        }
        const int32_t material = node.materialIndex >= 0 && size_t(node.materialIndex) < scene.materials().size() ? node.materialIndex : 0;
        auto& opacity = result[node.renderPrimitiveIndex];
        if (opacity.materialIndex == -2) {
            opacity.materialIndex = material;
        } else if (opacity.materialIndex != material) {
            opacity.materialIndex = -1;
        }
        if (size_t(material) < scene.materials().size()) {
            opacity.usesAlpha |= usesAlpha(scene.materials()[material]);
        }
    }
    return result;
}

std::vector<BakedOpacityMicromap> bakeSceneOpacityMicromaps(
    const scene::Scene& scene, std::span<const ScenePrimitiveOpacity> opacity, uint32_t level)
{
    std::vector<BakedOpacityMicromap> result(scene.renderPrimitives().size());
    if (opacity.size() != result.size()) {
        return result;
    }
    std::map<int32_t, std::vector<uint32_t>> groups;
    for (uint32_t i = 0; i < scene.renderPrimitives().size(); ++i) {
        const int32_t material = opacity[i].materialIndex;
        if (material >= 0 && size_t(material) < scene.materials().size() && usesAlpha(scene.materials()[material])) {
            groups[material].push_back(i);
        }
    }
    size_t totalBytes = 0;
    for (const auto& [materialIndex, primitives] : groups) {
        const auto& material = scene.materials()[materialIndex];
        scene::RenderImage::Mip decoded;
        const scene::RenderImage::Mip* image = nullptr;
        if (!loadAlphaImage(scene, material.baseColorTexture.textureIndex, decoded, image)) {
            continue;
        }
        OpacityMicromapBaker baker(material, image);
        for (uint32_t index : primitives) {
            const auto& primitive = scene.renderPrimitives()[index];
            auto& baked = result[index];
            if (primitive.mode != 4 || !baker.bake(primitive, level, baked)) {
                continue;
            }
            const size_t bytes = baked.data.size() + baked.triangles.size() * sizeof(OpacityMicromapTriangle);
            if (baked.stateCounts[0] + baked.stateCounts[1] == 0 || totalBytes + bytes > 256ull * 1024 * 1024) {
                baked = {};
                continue;
            }
            totalBytes += bytes;
        }
    }
    return result;
}

} // namespace metallic::render
