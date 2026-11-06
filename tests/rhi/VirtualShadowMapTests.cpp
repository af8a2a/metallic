#include "RHITest.h"
#include "harness/Fixtures.h"
#include "Runtime/Render/Shadow/VirtualShadowMapPrototype.h"
#include "Runtime/Scene/scene.h"

#include <json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace metallic::tests {
namespace {

using namespace render;
using Triangle = VirtualShadowMapPrototypeTriangle;
using Page = VirtualShadowMapPrototypePage;
using Receiver = VirtualShadowMapPrototypeReceiver;
using Query = VirtualShadowMapPrototypeQuery;
using Status = VirtualShadowMapPrototypeQueryStatus;
using Projection = VirtualShadowMapPrototypeProjection;
using Point = std::array<float, 3>;

void require(bool value, const std::string& message)
{
    if (!value) { throw std::runtime_error(message); }
}

template<typename T>
void require(const Result<T>& value, const std::string& message)
{
    require(bool(value), message + (value ? "" : ": " + std::string(resultToString(value))));
}

Triangle triangle(Point a, Point b, Point c, uint32_t material = 0)
{
    Triangle result;
    result.materialIndex = material;
    const std::array points{a, b, c};
    for (size_t i = 0; i < 3; ++i) {
        result.vertices[i].position = {points[i][0], points[i][1], points[i][2], 1};
        result.vertices[i].uv = {(points[i][0] + 1) * .5f, (points[i][1] + 1) * .5f, 0, 0};
    }
    return result;
}

void rectangle(std::vector<Triangle>& triangles, float left, float bottom, float right, float top,
    float z, uint32_t material, bool reverse = false)
{
    auto a = triangle({left, bottom, z}, {right, bottom, z}, {left, top, z}, material);
    auto b = triangle({right, bottom, z}, {right, top, z}, {left, top, z}, material);
    if (reverse) { std::swap(a.vertices[1], a.vertices[2]); std::swap(b.vertices[1], b.vertices[2]); }
    triangles.push_back(a);
    triangles.push_back(b);
}

Point project(const Projection& projection, const std::array<float, 4>& position)
{
    Point result{};
    for (size_t row = 0; row < result.size(); ++row) {
        for (size_t column = 0; column < position.size(); ++column) {
            result[row] += projection.rows[row * 4 + column] * position[column];
        }
    }
    return result;
}

enum class Coverage { Material, CustomUvAlpha, ConstantTexture };

struct Reference
{
    float depth = 0;
    bool stable = true;
};

// Independent light-ray / world-triangle intersection reference. It does not
// construct page worklists, clip polygons, or reproduce physical atlas routing.
Reference referenceDepth(std::span<const Triangle> triangles, std::span<const scene::RenderMaterial> materials,
    std::span<const Coverage> coverage, const Projection& projection, float x, float y)
{
    Reference result;
    for (const auto& primitive : triangles) {
        const auto a = project(projection, primitive.vertices[0].position);
        const auto b = project(projection, primitive.vertices[1].position);
        const auto c = project(projection, primitive.vertices[2].position);
        const double area = double(b[0] - a[0]) * (c[1] - a[1]) - double(b[1] - a[1]) * (c[0] - a[0]);
        const auto& material = materials[primitive.materialIndex];
        // Positive-height Vulkan viewport with the runtime's CounterClockwise
        // front face makes negative clip-XY area front-facing.
        if (area == 0 || (!material.doubleSided && area > 0)) { continue; }
        const double wb = (double(x - a[0]) * (c[1] - a[1]) - double(y - a[1]) * (c[0] - a[0])) / area;
        const double wc = (double(b[0] - a[0]) * (y - a[1]) - double(b[1] - a[1]) * (x - a[0])) / area;
        const double wa = 1 - wb - wc;
        if (wa < -1e-5 || wb < -1e-5 || wc < -1e-5) { continue; }
        if (std::min({std::abs(wa), std::abs(wb), std::abs(wc)}) < 1e-5) { result.stable = false; }
        const float z = float(wa * a[2] + wb * b[2] + wc * c[2]);
        if (z < 0 || z > 1) { continue; }
        float alpha = material.baseColorFactor.w;
        if (coverage[primitive.materialIndex] == Coverage::ConstantTexture) { alpha *= .25f; }
        if (coverage[primitive.materialIndex] == Coverage::CustomUvAlpha) {
            const float u = float(wa * primitive.vertices[0].uv[0] + wb * primitive.vertices[1].uv[0] + wc * primitive.vertices[2].uv[0]);
            alpha *= u + material.valueParameters[0];
        }
        if (material.alphaMode == "MASK") {
            if (std::abs(alpha - material.alphaCutoff) < 1e-5f) { result.stable = false; }
            if (alpha < material.alphaCutoff) { continue; }
        }
        result.depth = std::max(result.depth, z);
    }
    return result;
}

struct Snapshot
{
    std::vector<float> atlas;
    std::vector<Query> queries;
    std::array<uint32_t, 8> statistics{};
};

std::unique_ptr<Buffer> readbackBuffer(Device& device, size_t bytes)
{
    auto buffer = device.createBuffer({.size = bytes, .usage = BufferUsageBits::TransferDestination,
        .memoryLocation = MemoryLocation::HostReadback});
    require(buffer, "Readback allocation");
    return std::move(*buffer);
}

void readback(Buffer& buffer, void* destination, size_t bytes)
{
    buffer.invalidate();
    const void* mapped = buffer.map();
    require(mapped != nullptr, "Readback map");
    std::memcpy(destination, mapped, bytes);
    buffer.unmap();
}

Snapshot renderAndRead(Device& device, VirtualShadowMapPrototype& prototype, size_t receiverCount,
    TextureLayout previousLayout, bool clearAll)
{
    const auto& desc = prototype.desc();
    const uint32_t width = desc.physicalPagesX * desc.pageSize;
    const uint32_t height = desc.physicalPagesY * desc.pageSize;
    Snapshot result;
    result.atlas.resize(size_t(width) * height);
    result.queries.resize(receiverCount);
    auto atlasReadback = readbackBuffer(device, result.atlas.size() * sizeof(float));
    auto queryReadback = readbackBuffer(device, receiverCount * sizeof(Query));
    auto statisticsReadback = readbackBuffer(device, sizeof(result.statistics));
    bench::GPUCommands gpu(*device.getQueue(QueueType::Graphics));
    require(gpu.initialize(device), "Commands initialize");
    require(prototype.record(*gpu.commands, previousLayout, clearAll), "VSM record");
    const TextureBarrierDesc beforeAtlas{.texture = &prototype.atlas(), .oldLayout = TextureLayout::ShaderRead,
        .newLayout = TextureLayout::TransferSource, .before = {PipelineStageBits::AllCommands, AccessBits::ShaderRead},
        .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}};
    const BufferBarrierDesc beforeBuffers[]{
        {.buffer = &prototype.receiverResults(), .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
            .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}},
        {.buffer = &prototype.statistics(), .before = {PipelineStageBits::AllCommands, AccessBits::ShaderRead | AccessBits::ShaderWrite},
            .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}}};
    require(gpu.commands->synchronize({.textures = {&beforeAtlas, 1}, .buffers = beforeBuffers}), "Readback barriers");
    require(atlasReadback->slice().and_then([&](const auto& slice) {
        return gpu.commands->copyTextureToBuffer({.texture = &prototype.atlas(), .buffer = slice, .width = width, .height = height});
    }), "Atlas copy");
    const auto copyBuffer = [&](Buffer& source, Buffer& destination) {
        auto src = source.slice();
        auto dst = destination.slice();
        require(src, "Source slice"); require(dst, "Destination slice");
        require(gpu.commands->copyBuffer(*src, *dst), "Result copy");
    };
    copyBuffer(prototype.receiverResults(), *queryReadback);
    copyBuffer(prototype.statistics(), *statisticsReadback);
    const TextureBarrierDesc afterAtlas{.texture = &prototype.atlas(), .oldLayout = TextureLayout::TransferSource,
        .newLayout = TextureLayout::ShaderRead, .before = {PipelineStageBits::Transfer, AccessBits::TransferRead},
        .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead}};
    require(gpu.commands->synchronize({.textures = {&afterAtlas, 1}}), "Restore atlas read layout");
    require(gpu.submitAndWait(), "Submit and wait");
    readback(*atlasReadback, result.atlas.data(), result.atlas.size() * sizeof(float));
    readback(*queryReadback, result.queries.data(), result.queries.size() * sizeof(Query));
    readback(*statisticsReadback, result.statistics.data(), sizeof(result.statistics));
    return result;
}

std::vector<Receiver> receivers(uint32_t width, uint32_t height)
{
    std::vector<Receiver> result;
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            result.push_back({{(float(x) + .5f) / width * 2 - 1, (float(y) + .5f) / height * 2 - 1, .2f, 0}});
        }
    }
    result.push_back({{1.01f, 0, .2f, 0}});
    result.push_back({{0, -1.01f, .2f, 0}});
    result.push_back({{0, 0, 1.01f, 0}});
    result.push_back({{0, 0, -.01f, 0}});
    result.push_back({{0, 0, .95f, 0}});
    return result;
}

void saveImages(RHITestContext& context, const std::string& name, const Snapshot& snapshot,
    const VirtualShadowMapPrototypeDesc& desc)
{
    const uint32_t atlasWidth = desc.physicalPagesX * desc.pageSize;
    const uint32_t atlasHeight = desc.physicalPagesY * desc.pageSize;
    std::vector<uint8_t> pixels(snapshot.atlas.size() * 4);
    for (size_t i = 0; i < snapshot.atlas.size(); ++i) {
        const auto value = uint8_t(std::clamp(snapshot.atlas[i], 0.f, 1.f) * 255);
        pixels[i * 4] = pixels[i * 4 + 1] = pixels[i * 4 + 2] = value;
        pixels[i * 4 + 3] = 255;
    }
    std::string log;
    require(saveRgba8Png(context.outputDirectory / (name + "-atlas.png"), pixels.data(), atlasWidth, atlasHeight, log), log);
    const uint32_t width = desc.virtualPagesX * desc.pageSize, height = desc.virtualPagesY * desc.pageSize;
    pixels.resize(size_t(width) * height * 4);
    for (size_t i = 0; i < size_t(width) * height; ++i) {
        const auto& query = snapshot.queries[i];
        std::array<uint8_t, 4> color{0, 0, 0, 255};
        if (query.status == Status::Valid) {
            color[0] = color[1] = color[2] = uint8_t(std::clamp(query.visibility, 0.f, 1.f) * 255);
        } else if (query.status == Status::Unmapped) { color[0] = 255; }
        else if (query.status == Status::OutsideProjection) { color[0] = 255; color[1] = 128; }
        else { color[2] = 255; }
        std::copy(color.begin(), color.end(), pixels.begin() + i * 4);
    }
    require(saveRgba8Png(context.outputDirectory / (name + "-receivers.png"), pixels.data(), width, height, log), log);
}

nlohmann::json verify(const Snapshot& snapshot, const VirtualShadowMapPrototypeDesc& desc,
    std::span<const Triangle> triangles, std::span<const scene::RenderMaterial> materials,
    std::span<const Coverage> coverage, std::span<const Page> pages, std::span<const Receiver> inputs,
    const Projection& projection)
{
    size_t checked = 0, skippedEdges = 0, lit = 0, shadowed = 0, unmapped = 0, outside = 0;
    const uint32_t width = desc.virtualPagesX * desc.pageSize, height = desc.virtualPagesY * desc.pageSize;
    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& input = inputs[i].positionBias;
        const auto p = project(projection, {input[0], input[1], input[2], 1});
        const auto& actual = snapshot.queries[i];
        if (p[0] < -1 || p[0] >= 1 || p[1] < -1 || p[1] >= 1 || p[2] < 0 || p[2] > 1) {
            require(actual.status == Status::OutsideProjection, "Receiver outside projection not reported at " + std::to_string(i));
            ++outside; continue;
        }
        const uint32_t texelX = uint32_t((p[0] * .5f + .5f) * width);
        const uint32_t texelY = uint32_t((p[1] * .5f + .5f) * height);
        const uint32_t page = texelY / desc.pageSize * desc.virtualPagesX + texelX / desc.pageSize;
        if (pages[page].physicalPage == UINT32_MAX) {
            require(actual.status == Status::Unmapped, "Missing page not reported at " + std::to_string(i));
            ++unmapped; continue;
        }
        require(actual.status == Status::Valid && actual.virtualPage == page && actual.physicalPage == pages[page].physicalPage,
            "Receiver page identity mismatch at " + std::to_string(i));
        const auto expected = referenceDepth(triangles, materials, coverage, projection,
            (float(texelX) + .5f) / width * 2 - 1, (float(texelY) + .5f) / height * 2 - 1);
        if (!expected.stable) { ++skippedEdges; continue; }
        require(std::isfinite(actual.depth) && std::abs(actual.depth - expected.depth) < 3e-5f,
            "Independent depth mismatch at " + std::to_string(i) + ": " + std::to_string(actual.depth) + " vs " + std::to_string(expected.depth));
        const float visibility = p[2] + input[3] >= expected.depth ? 1.f : 0.f;
        require(actual.visibility == visibility, "Independent visibility mismatch at " + std::to_string(i));
        ++checked; lit += visibility == 1; shadowed += visibility == 0;
    }
    require(checked > size_t(width) * height / 2, "Insufficient independent pixel comparisons");
    require(snapshot.statistics[7] == 0, "Unexpected work overflow");
    return {{"checkedPixels", checked}, {"edgePixelsExcluded", skippedEdges}, {"lit", lit}, {"shadowed", shadowed},
        {"unmapped", unmapped}, {"outsideProjection", outside}, {"statistics", snapshot.statistics}};
}

class VirtualShadowMapPrototypeTest final : public RHITest
{
public:
    VirtualShadowMapPrototypeTest()
    {
        name = "virtual_shadow_map_prototype_pages";
        type = RHITestType::Rendering;
    }

    RHITestResult run(RHITestContext& context) override
    {
        try {
            struct ValidationState
            {
                std::atomic_uint errors{0};
                std::atomic_uint* harness = nullptr;
            } validation{.harness = context.validationMessageCount};
            std::unique_ptr<Device> device;
            const auto created = createDevice({.applicationName = "Virtual shadow map P0",
                .enableValidation = context.enableValidation, .enableSynchronizationValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true, .enableMeshShader = true,
                .enableRayTracingAccelerationStructure = false, .enableRayQuery = false,
                .enableRayTracingPipeline = false, .enableClusterAccelerationStructure = false,
                .validationSink = {[](void* target, const ValidationMessage& message) noexcept {
                    if (message.severity == ValidationSeverity::Error && hasFlag(message.type, ValidationCategory::Validation)) {
                        auto& state = *static_cast<ValidationState*>(target);
                        ++state.errors;
                        if (state.harness) { ++*state.harness; }
                    }
                }, &validation}}).transform([&](auto value) { device = std::move(value); });
            if (hasError(created, Error::Unsupported)) { return RHITestResult::skip("Requires mesh shaders and bindless descriptors"); }
            require(created, "VSM test device");
            require(!device->capabilities().rayQuery && !device->capabilities().rayTracingAccelerationStructure &&
                !device->capabilities().rayTracingPipeline && !device->capabilities().clusterAccelerationStructure, "RT unexpectedly enabled");
            std::filesystem::create_directories(context.outputDirectory);
            std::string log;
            VirtualShadowMapPrototypeDesc desc{.pageSize = 32, .virtualPagesX = 4, .virtualPagesY = 2,
                .physicalPagesX = 4, .physicalPagesY = 2, .maxWorkItems = 256};
            const auto inputs = receivers(desc.virtualPagesX * desc.pageSize, desc.virtualPagesY * desc.pageSize);
            std::vector<Page> pages{{5, 1}, {0, 1}, {6, 1}, {2, 1}, {4, 1}, {1, 1}, {UINT32_MAX, 1}, {3, 1}};
            std::vector<scene::RenderMaterial> materials(1);
            materials[0].doubleSided = true;
            const std::vector coverage{Coverage::Material};
            std::vector<Triangle> triangles{
                triangle({-.95f, -.9f, .55f}, {.95f, -.9f, .55f}, {0, .9f, .55f}),
                triangle({-.6f, -.6f, .8f}, {.6f, -.6f, .8f}, {0, .6f, .8f}),
                triangle({-.9f, .1f, .85f}, {-.7f, .7f, 1.15f}, {-.3f, .1f, .85f}),
                triangle({-.8f, -.8f, 1.2f}, {.8f, -.8f, 1.2f}, {0, .8f, 1.2f}),
                triangle({-.8f, -.8f, -.2f}, {.8f, -.8f, -.2f}, {0, .8f, -.2f})};
            Projection projection;
            VirtualShadowMapPrototype prototype;
            auto initialized = prototype.initialize(*device, desc, {.triangles = triangles, .materials = materials,
                .pages = pages, .receivers = inputs, .projection = projection}, log);
            require(initialized, "Prototype initialize: " + log);
            auto baseline = renderAndRead(*device, prototype, inputs.size(), TextureLayout::Undefined, true);
            saveImages(context, "VSMGeometry", baseline, desc);
            nlohmann::json report{{"rayQuery", false}, {"rayTracingAccelerationStructure", false},
                {"reference", "Independent projected triangle intersection at shadow texel centers"},
                {"pageSize", desc.pageSize}, {"physicalMapping", {5, 0, 6, 2, 4, 1, -1, 3}}};
            report["geometry"] = verify(baseline, desc, triangles, materials, coverage, pages, inputs, projection);
            require(report["geometry"]["shadowed"].get<size_t>() > 500 && report["geometry"]["lit"].get<size_t>() > 500,
                "Geometry fixture lacks lit/shadow coverage");
            require(baseline.statistics[0] > 0 && baseline.statistics[3] == 7 && baseline.statistics[6] > triangles.size(),
                "Cross-page GPU indirect work was not produced");
            const uint32_t atlasWidth = desc.physicalPagesX * desc.pageSize;
            for (uint32_t y = desc.pageSize; y < 2 * desc.pageSize; ++y) {
                for (uint32_t x = 3 * desc.pageSize; x < 4 * desc.pageSize; ++x) {
                    require(baseline.atlas[y * atlasWidth + x] == 0, "Unused physical page polluted by cross-page raster");
                }
            }

            VirtualShadowMapPrototype rejected;
            auto duplicatePages = pages;
            duplicatePages[1].physicalPage = duplicatePages[0].physicalPage;
            require(hasError(rejected.initialize(*device, desc, {.triangles = triangles, .materials = materials,
                .pages = duplicatePages, .receivers = inputs}, log), Error::InvalidArgument), "Duplicate physical ownership accepted");
            auto invalidTriangles = triangles;
            invalidTriangles[0].materialIndex = uint32_t(materials.size());
            require(hasError(rejected.initialize(*device, desc, {.triangles = invalidTriangles, .materials = materials,
                .pages = pages, .receivers = inputs}, log), Error::InvalidArgument), "Out-of-range material accepted");
            auto invalidProjection = projection;
            invalidProjection.rows[15] = 0;
            require(hasError(rejected.initialize(*device, desc, {.triangles = triangles, .materials = materials,
                .pages = pages, .receivers = inputs, .projection = invalidProjection}, log), Error::InvalidArgument),
                "Non-orthographic projection accepted");
            report["rejectedInputs"] = {"duplicate physical mapping", "out-of-range material", "non-orthographic projection"};

            // Exercise the proposed production page dimension, in addition to
            // the small-page regression fixtures used for the other scenarios.
            auto fullPageDesc = desc;
            fullPageDesc.pageSize = 128;
            const auto fullPageInputs = receivers(fullPageDesc.virtualPagesX * fullPageDesc.pageSize,
                fullPageDesc.virtualPagesY * fullPageDesc.pageSize);
            VirtualShadowMapPrototype fullPagePrototype;
            initialized = fullPagePrototype.initialize(*device, fullPageDesc, {.triangles = triangles, .materials = materials,
                .pages = pages, .receivers = fullPageInputs}, log);
            require(initialized, "128-pixel page initialize: " + log);
            const auto fullPageResult = renderAndRead(*device, fullPagePrototype, fullPageInputs.size(), TextureLayout::Undefined, true);
            saveImages(context, "VSMGeometry128", fullPageResult, fullPageDesc);
            report["geometry128"] = verify(fullPageResult, fullPageDesc, triangles, materials, coverage, pages, fullPageInputs, projection);
            report["geometry128"]["pageSize"] = 128;

            auto emptyPages = pages;
            emptyPages[6].physicalPage = 7;
            const std::vector<Triangle> emptyTriangles;
            VirtualShadowMapPrototype emptyPrototype;
            initialized = emptyPrototype.initialize(*device, desc, {.triangles = emptyTriangles, .materials = materials,
                .pages = emptyPages, .receivers = inputs}, log);
            require(initialized, "Empty geometry initialize: " + log);
            const auto emptyResult = renderAndRead(*device, emptyPrototype, inputs.size(), TextureLayout::Undefined, true);
            report["emptyGeometry"] = verify(emptyResult, desc, emptyTriangles, materials, coverage, emptyPages, inputs, projection);
            require(emptyResult.statistics[0] == 0 && emptyResult.statistics[6] == 0 &&
                report["emptyGeometry"]["shadowed"].get<size_t>() == 0 &&
                std::all_of(emptyResult.atlas.begin(), emptyResult.atlas.end(), [](float depth) { return depth == 0; }),
                "Empty geometry did not produce clear depth and valid fully lit queries");
            saveImages(context, "VSMEmpty", emptyResult, desc);

            // All updates occur after renderAndRead has completed its fence.
            for (auto& page : pages) { page.dirty = 0; }
            auto removed = triangles;
            for (auto& primitive : removed) { for (auto& vertex : primitive.vertices) { vertex.position[0] += 4; } }
            require(prototype.updatePages(pages), "Set clean pages");
            require(prototype.updateTriangles(removed), "Remove casters");
            const auto clean = renderAndRead(*device, prototype, inputs.size(), TextureLayout::ShaderRead, false);
            require(clean.atlas == baseline.atlas && clean.statistics[0] == 0 && clean.statistics[3] == 0 &&
                clean.statistics[6] == 0 && clean.statistics[7] == 0, "Zero-dirty frame changed the atlas or issued work");
            require(std::memcmp(clean.queries.data(), baseline.queries.data(), clean.queries.size() * sizeof(Query)) == 0,
                "Zero-dirty frame changed receiver results");
            report["zeroDirty"] = {{"statistics", clean.statistics}, {"atlasBitExact", true}};
            pages[1].dirty = 1;
            require(prototype.updatePages(pages), "Set one dirty page");
            const auto dirty = renderAndRead(*device, prototype, inputs.size(), TextureLayout::ShaderRead, false);
            size_t erased = 0;
            for (uint32_t y = 0; y < desc.physicalPagesY * desc.pageSize; ++y) {
                for (uint32_t x = 0; x < atlasWidth; ++x) {
                    const size_t index = size_t(y) * atlasWidth + x;
                    const bool changedPage = x < desc.pageSize && y < desc.pageSize;
                    require(dirty.atlas[index] == (changedPage ? 0.f : baseline.atlas[index]), "Dirty clear changed a clean physical page");
                    erased += changedPage && baseline.atlas[index] != 0;
                }
            }
            require(erased > 100 && dirty.statistics[3] == 1 && dirty.statistics[0] == 0, "Dirty page clear not exercised");
            report["dirtyClear"] = {{"erasedPixels", erased}, {"statistics", dirty.statistics}, {"cleanPagesBitExact", true}};
            saveImages(context, "VSMDirtyClear", dirty, desc);

            for (auto& page : pages) { page.dirty = 1; }
            projection.rows[3] = .2f;
            require(prototype.updatePages(pages), "Restore dirty pages");
            require(prototype.updateTriangles(triangles), "Restore casters");
            require(prototype.updateProjection(projection), "Change light projection");
            const auto shifted = renderAndRead(*device, prototype, inputs.size(), TextureLayout::ShaderRead, false);
            report["projectionUpdate"] = verify(shifted, desc, triangles, materials, coverage, pages, inputs, projection);
            require(shifted.atlas != baseline.atlas, "Light projection change did not affect world-space casters");
            saveImages(context, "VSMProjection", shifted, desc);

            projection = {};
            std::vector<scene::RenderMaterial> sidedMaterials(2);
            sidedMaterials[1].doubleSided = true;
            std::vector<Triangle> sided;
            rectangle(sided, -.95f, -.8f, -.35f, .8f, .7f, 0);
            rectangle(sided, -.3f, -.8f, .3f, .8f, .7f, 0, true);
            rectangle(sided, .35f, -.8f, .95f, .8f, .7f, 1, true);
            const std::vector sidedCoverage(2, Coverage::Material);
            VirtualShadowMapPrototype sidedPrototype;
            initialized = sidedPrototype.initialize(*device, desc, {.triangles = sided, .materials = sidedMaterials,
                .pages = pages, .receivers = inputs}, log);
            require(initialized, "Sidedness initialize: " + log);
            const auto sidedResult = renderAndRead(*device, sidedPrototype, inputs.size(), TextureLayout::Undefined, true);
            saveImages(context, "VSMSidedness", sidedResult, desc);
            report["sidedness"] = verify(sidedResult, desc, sided, sidedMaterials, sidedCoverage, pages, inputs, projection);

            std::unique_ptr<Texture> alphaTexture;
            std::unique_ptr<TextureView> alphaView;
            require(device->createTexture({.usage = TextureUsageBits::Sampled | TextureUsageBits::TransferDestination,
                .format = Format::RGBA32Sfloat, .width = 1, .height = 1})
                .transform([&](auto value) { alphaTexture = std::move(value); }), "Alpha texture");
            require(device->createTextureView(*alphaTexture, {.format = Format::RGBA32Sfloat})
                .transform([&](auto value) { alphaView = std::move(value); }), "Alpha view");
            auto upload = device->createBuffer({.size = 16, .usage = BufferUsageBits::TransferSource,
                .memoryLocation = MemoryLocation::HostUpload});
            require(upload, "Alpha upload allocation");
            const std::array<float, 4> texel{1, 1, 1, .25f};
            auto* mapped = (*upload)->map(); require(mapped != nullptr, "Alpha upload map");
            std::memcpy(mapped, texel.data(), sizeof(texel)); (*upload)->flush(); (*upload)->unmap();
            {
                bench::GPUCommands gpu(*device->getQueue(QueueType::Graphics));
                require(gpu.initialize(*device), "Alpha upload commands");
                const TextureBarrierDesc before{.texture = alphaTexture.get(), .oldLayout = TextureLayout::Undefined,
                    .newLayout = TextureLayout::TransferDestination, .before = {},
                    .after = {PipelineStageBits::Transfer, AccessBits::TransferWrite}};
                require(gpu.commands->synchronize({.textures = {&before, 1}}), "Alpha upload barrier");
                require((*upload)->slice().and_then([&](const auto& slice) {
                    return gpu.commands->copyBufferToTexture({.texture = alphaTexture.get(), .buffer = slice, .width = 1, .height = 1});
                }), "Alpha upload copy");
                const TextureBarrierDesc after{.texture = alphaTexture.get(), .oldLayout = TextureLayout::TransferDestination,
                    .newLayout = TextureLayout::ShaderRead, .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                    .after = {PipelineStageBits::FragmentShader, AccessBits::ShaderRead}};
                require(gpu.commands->synchronize({.textures = {&after, 1}}), "Alpha sampling barrier");
                require(gpu.submitAndWait(), "Alpha upload submit");
            }
            std::vector<scene::RenderMaterial> masks(4);
            for (auto& material : masks) { material.alphaMode = "MASK"; material.doubleSided = true; }
            masks[0].baseColorFactor.w = .25f;
            masks[1].baseColorFactor.w = .8f;
            masks[1].valueProgram = R"({"version":1,"coverage":{"op":"mul","args":[{"op":"alpha"},{"op":"add","args":[{"op":"uv"},{"op":"parameter","index":0}]}]}})";
            masks[1].valueParameters[0] = .1f;
            masks[3].alphaCutoff = .1f;
            std::vector<Triangle> masked;
            rectangle(masked, -.95f, -.9f, -.55f, .9f, .75f, 0);
            rectangle(masked, -.5f, -.9f, .5f, .9f, .75f, 1);
            rectangle(masked, .55f, -.9f, .95f, -.05f, .75f, 2);
            rectangle(masked, .55f, .05f, .95f, .9f, .75f, 3);
            const std::vector maskCoverage{Coverage::Material, Coverage::CustomUvAlpha, Coverage::ConstantTexture, Coverage::ConstantTexture};
            const std::array<TextureView*, 4> textures{nullptr, nullptr, alphaView.get(), alphaView.get()};
            VirtualShadowMapPrototype maskPrototype;
            initialized = maskPrototype.initialize(*device, desc, {.triangles = masked, .materials = masks, .alphaTextures = textures,
                .pages = pages, .receivers = inputs}, log);
            require(initialized, "Coverage initialize: " + log);
            const auto maskResult = renderAndRead(*device, maskPrototype, inputs.size(), TextureLayout::Undefined, true);
            saveImages(context, "VSMCoverage", maskResult, desc);
            report["coverage"] = verify(maskResult, desc, masked, masks, maskCoverage, pages, inputs, projection);
            require(report["coverage"]["shadowed"].get<size_t>() > 100 && report["coverage"]["lit"].get<size_t>() > 100,
                "Coverage did not produce both holes and blockers");

            desc.maxWorkItems = 1;
            VirtualShadowMapPrototype overflowPrototype;
            initialized = overflowPrototype.initialize(*device, desc, {.triangles = triangles, .materials = materials,
                .pages = pages, .receivers = inputs}, log);
            require(initialized, "Overflow initialize: " + log);
            const auto overflow = renderAndRead(*device, overflowPrototype, inputs.size(), TextureLayout::Undefined, true);
            require(overflow.statistics[0] == 1 && overflow.statistics[6] > 1 && overflow.statistics[7] > 0,
                "Work overflow did not report bounded dispatch and dropped work");
            size_t overflowQueries = 0;
            for (size_t i = 0; i < overflow.queries.size(); ++i) {
                if (baseline.queries[i].status == Status::Valid) {
                    require(overflow.queries[i].status == Status::WorkOverflow, "Incomplete shadow result was presented as valid");
                    ++overflowQueries;
                }
            }
            require(overflowQueries > 100, "Overflow receiver status not exercised");
            report["overflow"] = {{"statistics", overflow.statistics}, {"invalidatedQueries", overflowQueries}};
            saveImages(context, "VSMOverflow", overflow, desc);
            require(validation.errors.load() == 0, "VSM device emitted validation errors: " + std::to_string(validation.errors.load()));
            report["validationEnabled"] = context.enableValidation;
            report["synchronizationValidationEnabled"] = context.enableValidation;
            report["validationErrors"] = validation.errors.load();
            report["scope"] = "P0 fixed pages, world-space test triangles, GPU indirect depth raster and actual GPU receiver queries; not production scene streaming";
            std::ofstream output(context.outputDirectory / "VirtualShadowMapPrototype.json");
            output << report.dump(2) << '\n';
            require(bool(output), "Write VSM evidence JSON");
            return RHITestResult::pass("No-RT GPU page raster/query, independent triangle-depth reference, remapping, clipping, MASK/custom Coverage, dirty isolation, zero work and explicit overflow; PNG/JSON exported");
        } catch (const std::exception& error) {
            return RHITestResult::fail(error.what());
        }
    }
};

METALLIC_REGISTER_RHI_TEST(VirtualShadowMapPrototypeTest);

} // namespace
} // namespace metallic::tests
