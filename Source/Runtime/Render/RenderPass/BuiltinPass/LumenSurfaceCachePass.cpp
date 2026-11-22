#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Render/Core/SceneColorConversion.h"
#include "Runtime/Render/Environment/CelestialLighting.h"
#include "Runtime/Render/SceneLightResources.h"
#include "Runtime/Render/Streamer/ScenePathTraceResources.h"
#include "Runtime/Scene/MeshCards.h"
#include "Runtime/Scene/MeshDistanceField.h"

#include <cmath>
#include <cstring>
#include <optional>
#include <unordered_map>
#include <spdlog/spdlog.h>

namespace metallic::render::builtin_pass {
namespace {

constexpr char kShaderModule[] = "Features/DistanceField/SurfaceCacheCapture";
constexpr uint64_t kCaptureABI = 0x4c554d454e010001ull;
constexpr uint64_t kResolveABI = 0x4c554d454e010002ull;
constexpr uint64_t kEncodeDepthABI = 0x4c554d454e010004ull;
constexpr uint64_t kShadowedLightingABI = 0x4c554d454e010008ull;
constexpr uint32_t kGDFDomainStride = 32;
constexpr std::array kCaptureFormats{Format::RGBA8Unorm, Format::RGBA8Unorm, Format::RGBA16Sfloat};
constexpr std::array kCaptureOutputs{"albedo", "normal", "emissive"};
constexpr SamplerDesc kMaterialSampler{
    .mipFilter = SamplerFilter::Linear,
    .addressU = SamplerAddressMode::Repeat,
    .addressV = SamplerAddressMode::Repeat,
    .addressW = SamplerAddressMode::Repeat,
};
using Float4 = std::array<float, 4>;

struct alignas(16) CaptureVertex {
    Float4 position, normal, tangent, uv;
};

struct alignas(16) SurfaceCard {
    Float4 origin, axisX, axisY, axisZ;
    std::array<uint32_t, 4> metadata;
};

struct alignas(16) CaptureParams {
    std::array<uint32_t, 4> geometry;
    Float4 clipRow0, clipRow1, clipRow2, normalAxisX, normalAxisY;
    Float4 baseColor, emissive, material;
    std::array<uint32_t, 4> textures;
    std::array<Float4, 8> textureTransforms;
};

struct alignas(16) ResolveParams {
    ShaderSampledImage albedo, normal, emissive, hardwareDepth;
    ShaderStorageImage depth, lighting;
    ShaderBuffer cards;
    uint32_t tileResolution;
    Float4 lightDirection, illuminance;
};

struct EncodeDepthParams {
    ShaderSampledImage normal, hardwareDepth;
    ShaderStorageImage depth;
    uint32_t atlasSize;
};

struct alignas(16) ShadowedLightingParams {
    ShaderSampledImage albedo, normal, emissive, depth;
    ShaderStorageImage lighting;
    ShaderBuffer cards;
    uint32_t tileResolution, sunCastsShadow;
    Float4 lightDirection, illuminance;
    ShaderSampledImage pageTable, pageAtlas, coarseMip;
    ShaderBuffer domains, localLights;
    ShaderSampler distanceSampler;
    uint32_t clipmapCount, atlasSize;
    ShaderStorageImage directIrradiance;
    uint32_t exportDirectIrradiance;
    uint32_t padding[2];
};

static_assert(sizeof(CaptureVertex) == 64 && sizeof(SurfaceCard) == 80);
static_assert(sizeof(CaptureParams) == 288 && offsetof(CaptureParams, textureTransforms) == 160);
static_assert(sizeof(ResolveParams) == 64 && offsetof(ResolveParams, lightDirection) == 32);
static_assert(sizeof(EncodeDepthParams) == 16);
static_assert(sizeof(ShadowedLightingParams) == 112 && offsetof(ShadowedLightingParams, pageTable) == 64 &&
    offsetof(ShadowedLightingParams, directIrradiance) == 96);

struct AtlasLayout {
    uint32_t edge = 128;
    uint32_t tile = 16;
    uint32_t tilesPerAxis() const { return edge / tile; }
    uint32_t capacity() const { return tilesPerAxis() * tilesPerAxis(); }
    Rect rect(uint32_t card) const
    {
        return {int32_t((card % tilesPerAxis()) * tile), int32_t((card / tilesPerAxis()) * tile), tile, tile};
    }
};

std::optional<AtlasLayout> readLayout(const RenderGraphProperties& props)
{
    try {
        AtlasLayout layout;
        layout.edge = props.value("atlasSize", layout.edge);
        layout.tile = props.value("tileResolution", layout.tile);
        if (layout.tile < 4 || layout.edge < layout.tile || layout.edge > 2048 || layout.edge % layout.tile != 0) {
            return {};
        }
        return layout;
    } catch (const nlohmann::json::exception&) {
        return {};
    }
}

Float4 pack(float3 v, float w = 0.0f) { return {v.x, v.y, v.z, w}; }
float3 xyz(const auto& v) { return float3(v[0], v[1], v[2]); }

struct InstanceTransform {
    float3 x, y, z, origin;
    float3 vector(float3 v) const { return x * v.x + y * v.y + z * v.z; }
    float3 position(float3 v) const { return vector(v) + origin; }
    float3 normal(float3 v) const
    {
        return normalize(x * (v.x / dot(x, x)) + y * (v.y / dot(y, y)) + z * (v.z / dot(z, z)));
    }
};

SurfaceCard makeCard(const scene::MeshCardBuildRecord& card, const InstanceTransform& transform,
    uint32_t object, uint32_t mesh)
{
    const float3 scaledX = transform.vector(xyz(card.axisX));
    const float3 scaledY = transform.vector(xyz(card.axisY));
    const float3 scaledZ = transform.vector(xyz(card.axisZ));
    const float yLength = length(scaledY);
    const float zLength = length(scaledZ);
    const float3 y = scaledY / yLength;
    const float3 z = scaledZ / zLength;
    // UE TLumenCardOBB::Transform uses Z cross Y to retain card-space handedness across mirrors.
    const float3 x = normalize(cross(z, y));
    return {pack(transform.position(xyz(card.origin))), pack(x, card.extent[0] * length(scaledX)),
        pack(y, card.extent[1] * yLength), pack(z, std::max(card.extent[2] * zLength, 0.01f)),
        {object, mesh, card.axisAlignedDirectionIndex, 0}};
}

void projectCard(CaptureParams& params, const SurfaceCard& card)
{
    const float3 origin = xyz(card.origin);
    const float3 x = xyz(card.axisX) / card.axisX[3];
    const float3 y = -xyz(card.axisY) / card.axisY[3];
    const float3 z = xyz(card.axisZ) / (2.0f * card.axisZ[3]);
    params.clipRow0 = pack(x, -dot(x, origin));
    params.clipRow1 = pack(y, -dot(y, origin));
    params.clipRow2 = pack(z, 0.5f - dot(z, origin));
    params.normalAxisX = card.axisX;
    params.normalAxisY = card.axisY;
}

struct CaptureSection {
    uint32_t firstVertex, vertexCount;
    int32_t materialIndex;
};

struct CaptureCard {
    uint32_t firstSection, sectionCount;
};

struct CaptureGeometryRevision {
    uint64_t sceneIdentity, transforms, visibility;

    bool operator==(const CaptureGeometryRevision&) const = default;
};

struct CaptureGeometry {
    CaptureGeometryRevision revision;
    std::vector<SurfaceCard> cards;
    std::vector<CaptureSection> sections;
    std::vector<CaptureCard> cardsToDraw;
    std::unique_ptr<Buffer> vertexBuffer;
};

struct CaptureDraw {
    Rect rect;
    uint32_t vertexCount;
    EncodedParameters params;
};

class LumenSurfaceCachePass final : public RasterPass {
public:
    bool supportsFrameOverlap() const override { return true; }
    bool supportsExternalFrameOverlap() const override { return false; }
    RenderGraphSceneDependency sceneDependency() const override { return {RenderGraphSceneSource::World}; }
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override;
    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override;
    Result<> execute(RenderGraphExecutionContext& context) override;

private:
    bool shadowedLighting() const;
    bool exportDirectIrradiance() const;
    Result<> prepareCaptureGeometry(const scene::Scene& scene);
    Result<> gather(const scene::Scene& scene, std::vector<SurfaceCard>& cards, std::vector<CaptureCard>& draws,
        std::vector<CaptureSection>& sections, std::vector<CaptureVertex>& vertices) const;
    void bindMaterial(ParameterWriter& writer, const scene::Scene& scene, int32_t index, CaptureParams& params) const;
    Result<> rasterize(RenderGraphExecutionContext& context, CommandBuffer& cmd, std::span<const CaptureDraw> draws);

    Device* device_ = nullptr;
    AtlasLayout layout_;
    std::unordered_map<int32_t, std::vector<scene::MeshCardBuildRecord>> fields_;
    std::optional<CaptureGeometry> captureGeometry_;
    ScenePathTraceResources materialResources_;
    std::unique_ptr<GraphicsPipeline> pipeline_;
    ComputeKernel resolve_;
    ComputeKernel encodeDepth_;
    ComputeKernel shadowedLighting_;
};

bool LumenSurfaceCachePass::shadowedLighting() const
{
    const auto property = properties().find("shadowedLighting");
    return property != properties().end() && property->is_boolean() && property->get<bool>();
}

bool LumenSurfaceCachePass::exportDirectIrradiance() const
{
    const auto property = properties().find("exportDirectIrradiance");
    return property != properties().end() && property->is_boolean() && property->get<bool>();
}

RenderPassReflection LumenSurfaceCachePass::reflect(const RenderGraphCompileContext&) const
{
    RenderPassReflection reflection;
    const auto layout = readLayout(properties()).value_or(AtlasLayout{});
    for (size_t i = 0; i < kCaptureFormats.size(); ++i) {
        reflection.addTextureOutput(kCaptureOutputs[i], "Surface Cache material atlas")
            .texture2D(layout.edge, layout.edge).colorWrite().stageAccess(RenderGraphResourceAccess::TextureSampleRead)
            .format = kCaptureFormats[i];
    }
    reflection.addTextureOutput("hardwareDepth", "Reversed-Z card capture depth")
        .texture2D(layout.edge, layout.edge).depthStencilWrite().stageAccess(RenderGraphResourceAccess::TextureSampleRead)
        .format = Format::D32Sfloat;
    auto& depth = reflection.addTextureOutput("depth", "Card depth, with 1 reserved for uncovered texels")
        .texture2D(layout.edge, layout.edge).storageWrite().stageAccess(RenderGraphResourceAccess::TextureStorageWrite);
    depth.format = Format::R16Unorm;
    if (shadowedLighting()) {
        depth.stageAccess(RenderGraphResourceAccess::TextureSampleRead);
        reflection.addTextureInput("pageTable", "GDF page table").texture3D(0, 0, 0).sampledRead()
            .stageAccess(RenderGraphResourceAccess::TextureSampleRead).format = Format::R32Uint;
        reflection.addTextureInput("pageAtlas", "GDF fine page distances").texture3D(0, 0, 0).sampledRead()
            .stageAccess(RenderGraphResourceAccess::TextureSampleRead).format = Format::R8Unorm;
        reflection.addTextureInput("coarseMip", "GDF coarse distances").texture3D(0, 0, 0).sampledRead()
            .stageAccess(RenderGraphResourceAccess::TextureSampleRead).format = Format::R8Unorm;
        reflection.addBufferInput("domain", "GDF clipmap domains").buffer(0, kGDFDomainStride).storageRead()
            .stageAccess(RenderGraphResourceAccess::BufferStorageRead);
    }
    reflection.addTextureOutput("lighting", "Scene-linear outgoing radiance")
        .texture2D(layout.edge, layout.edge).storageWrite().stageAccess(RenderGraphResourceAccess::TextureStorageWrite)
        .format = Format::RGBA16Sfloat;
    if (exportDirectIrradiance()) {
        reflection.addTextureOutput("directIrradiance", "Direct light irradiance before the card material response")
            .texture2D(layout.edge, layout.edge).storageWrite().stageAccess(RenderGraphResourceAccess::TextureStorageWrite)
            .format = Format::RGBA32Sfloat;
    }
    reflection.addBufferOutput("cards", "World card boxes and scene object identity in atlas tile order")
        .buffer(layout.capacity() * sizeof(SurfaceCard), sizeof(SurfaceCard)).transferWrite()
        .stageAccess(RenderGraphResourceAccess::BufferStorageRead);
    reflection.addBufferOutput("scene", "Card count, tile resolution and scene resource identity (low/high)")
        .buffer(16, sizeof(uint32_t)).bufferLayout("u32").transferWrite();
    return reflection;
}

Result<std::unique_ptr<ShaderModule>> captureShader(Device& device, const char* entry, std::string& log)
{
    auto shader = ShaderRegistry::instance().getShader({.moduleName = kShaderModule,
        .entryPointName = entry, .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, log);
    if (!shader) { return makeError(shader.error()); }
    return ShaderRegistry::instance().getShaderModule(device, {.spirv = shader->spirv, .debugName = entry});
}

Result<> LumenSurfaceCachePass::compile(const RenderGraphCompileContext& context, std::string& log)
{
    const auto exportIrradiance = properties().find("exportDirectIrradiance");
    if (exportIrradiance != properties().end() &&
        (!exportIrradiance->is_boolean() || (exportDirectIrradiance() && !shadowedLighting()))) {
        log = "Surface Cache exportDirectIrradiance must be a boolean and requires shadowedLighting when enabled";
        return makeError(Error::InvalidArgument);
    }
    const auto layout = readLayout(properties());
    if (!layout || !context.device || !context.graphicsQueue || !context.runtimeScene) {
        log = "Surface Cache requires a world scene, graphics queue and an atlas divisible by its tile resolution";
        return makeError(Error::InvalidArgument);
    }
    if (!context.device->capabilities().shaderStorageImageExtendedFormats) {
        log = "Surface Cache requires R16 storage image support";
        return makeError(Error::Unsupported);
    }
    const auto shadowed = properties().find("shadowedLighting");
    if (shadowed != properties().end() && !shadowed->is_boolean()) {
        log = "Surface Cache shadowedLighting must be a boolean";
        return makeError(Error::InvalidArgument);
    }
    captureGeometry_.reset();
    layout_ = *layout;
    device_ = context.device;
    fields_.clear();
    const auto& scene = *context.runtimeScene;
    try {
        for (const auto& value : properties().value("meshCardPaths", std::vector<std::string>{})) {
            const auto path = std::filesystem::path(PROJECT_SOURCE_DIR) / value;
            scene::MeshCards field;
            scene::MeshDistanceFieldGeometry geometry;
            if (!scene::loadMeshCards(path, field, log) ||
                !scene::getMeshDistanceFieldGeometry(scene, field.sourceMeshIndex, geometry, log)) {
                return makeError(Error::InvalidArgument);
            }
            if (field.sourceFingerprint != geometry.fingerprint ||
                !fields_.try_emplace(field.sourceMeshIndex, std::move(field.cards)).second) {
                log = "Mesh card source changed or is bound more than once: " + path.string();
                return makeError(Error::InvalidArgument);
            }
        }
    } catch (const nlohmann::json::exception& e) {
        log = e.what();
        return makeError(Error::InvalidArgument);
    }

    if (!scene.textures().empty()) {
        if (auto r = materialResources_.prepare(*device_, *context.graphicsQueue, properties(), &scene, log); !r) { return r; }
    }
    auto vertex = captureShader(*device_, "surfaceCacheCaptureVertexMain", log);
    if (!vertex) { return makeError(vertex.error()); }
    auto fragment = captureShader(*device_, "surfaceCacheCaptureFragmentMain", log);
    if (!fragment) { return makeError(fragment.error()); }
    auto pipeline = ShaderRegistry::instance().getGraphicsPipeline(*device_, {
        .vertexShader = {vertex->get()}, .fragmentShader = {fragment->get()},
        .colorFormats = {kCaptureFormats[0], kCaptureFormats[1], kCaptureFormats[2]},
        .colorAttachmentCount = 3, .depthStencilFormat = Format::D32Sfloat,
        .rasterization = {.cullMode = CullMode::Back, .frontFace = FrontFace::Clockwise},
        .depthStencil = {.depthTestEnable = true, .depthWriteEnable = true, .depthCompareOp = CompareOp::GreaterEqual},
        .usesBindlessHeap = true});
    if (!pipeline) { return makeError(pipeline.error()); }
    pipeline_ = std::move(*pipeline);
    if (shadowedLighting()) {
        if (auto r = ShaderRegistry::instance().getComputeKernel(*device_,
                {.moduleName = kShaderModule, .entryPointName = "surfaceCacheEncodeDepthMain",
                    .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
                {.parameters = parameterAbi<EncodeDepthParams>(kEncodeDepthABI),
                    .debugName = "LumenSurfaceCacheEncodeDepth"}, encodeDepth_, log); !r) {
            return r;
        }
        return ShaderRegistry::instance().getComputeKernel(*device_,
            {.moduleName = kShaderModule, .entryPointName = "surfaceCacheShadowedLightingMain",
                .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
            {.parameters = parameterAbi<ShadowedLightingParams>(kShadowedLightingABI),
                .debugName = "LumenSurfaceCacheShadowedLighting"}, shadowedLighting_, log);
    }
    return ShaderRegistry::instance().getComputeKernel(*device_,
        {.moduleName = kShaderModule, .entryPointName = "surfaceCacheResolveMain", .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
        {.parameters = parameterAbi<ResolveParams>(kResolveABI), .debugName = "LumenSurfaceCacheResolve"}, resolve_, log);
}

Result<std::unique_ptr<Buffer>> upload(Device& device, std::span<const std::byte> bytes, BufferUsageBits usage)
{
    auto buffer = device.createBuffer({.size = bytes.size(), .usage = usage, .memoryLocation = MemoryLocation::HostUpload});
    if (!buffer) { return makeError(buffer.error()); }
    void* data = (*buffer)->map();
    if (!data) { return makeError(Error::Failure); }
    std::memcpy(data, bytes.data(), bytes.size());
    (*buffer)->flush();
    (*buffer)->unmap();
    return buffer;
}

void appendGeometry(const scene::RenderPrimitive& primitive, const InstanceTransform& transform,
    bool mirrored, std::vector<CaptureVertex>& vertices)
{
    const size_t indexCount = primitive.indices.empty() ? primitive.positions.size() : primitive.indices.size();
    for (size_t triangle = 0; triangle < indexCount; triangle += 3) {
        const auto index = [&](size_t corner) {
            return primitive.indices.empty() ? uint32_t(triangle + corner) : primitive.indices[triangle + corner];
        };
        const std::array indices{index(0), index(mirrored ? 2 : 1), index(mirrored ? 1 : 2)};
        const auto a = transform.position(primitive.positions[indices[0]]);
        const auto b = transform.position(primitive.positions[indices[1]]);
        const auto c = transform.position(primitive.positions[indices[2]]);
        const auto crossProduct = cross(b - a, c - a);
        if (dot(crossProduct, crossProduct) == 0.0f) { continue; }
        const auto faceNormal = normalize(crossProduct);
        Float4 triangleTangent{};
        if (!primitive.hasAuthoredTangents && !primitive.texcoords0.empty()) {
            const auto uv1 = primitive.texcoords0[indices[1]] - primitive.texcoords0[indices[0]];
            const auto uv2 = primitive.texcoords0[indices[2]] - primitive.texcoords0[indices[0]];
            const float determinant = uv1.x * uv2.y - uv1.y * uv2.x;
            if (determinant != 0.0f) {
                const auto tangent = ((b - a) * uv2.y - (c - a) * uv1.y) / determinant;
                const auto bitangent = ((c - a) * uv1.x - (b - a) * uv2.x) / determinant;
                triangleTangent = pack(normalize(tangent), dot(cross(faceNormal, tangent), bitangent) < 0.0f ? -1.0f : 1.0f);
            }
        }
        for (const uint32_t vertex : indices) {
            const auto normal = primitive.hasAuthoredNormals ? transform.normal(primitive.normals[vertex]) : faceNormal;
            auto tangent = triangleTangent;
            if (primitive.hasAuthoredTangents) {
                const auto t = primitive.tangents[vertex];
                tangent = pack(normalize(transform.vector(float3(t.x, t.y, t.z))), mirrored ? -t.w : t.w);
            }
            const auto uv = primitive.texcoords0.empty() ? float2(0.0f) : primitive.texcoords0[vertex];
            vertices.push_back({pack(transform.position(primitive.positions[vertex]), 1.0f),
                pack(normal), tangent, {uv.x, uv.y, 0.0f, 0.0f}});
        }
    }
}

Result<> LumenSurfaceCachePass::gather(const scene::Scene& scene, std::vector<SurfaceCard>& cards,
    std::vector<CaptureCard>& draws, std::vector<CaptureSection>& sections, std::vector<CaptureVertex>& vertices) const
{
    const auto& nodes = scene.renderNodes();
    for (size_t first = 0; first < nodes.size();) {
        const auto& node = nodes[first];
        size_t last = first + 1;
        while (last < nodes.size() && nodes[last].object == node.object) { ++last; }
        const auto& primitive = scene.renderPrimitives()[node.renderPrimitiveIndex];
        const auto field = fields_.find(primitive.meshIndex);
        const size_t firstNode = first;
        first = last;
        if (!node.visible || field == fields_.end() || field->second.empty()) { continue; }

        const auto& m = node.worldMatrix.a;
        const InstanceTransform transform{{m[0], m[1], m[2]}, {m[4], m[5], m[6]}, {m[8], m[9], m[10]}, {m[12], m[13], m[14]}};
        const float determinant = dot(transform.x, cross(transform.y, transform.z));
        if (!std::isfinite(determinant) || determinant == 0.0f) { continue; }
        const float3 x = normalize(transform.x), y = normalize(transform.y), z = normalize(transform.z);
        if (std::abs(dot(x, y)) >= 1.0e-4f || std::abs(dot(x, z)) >= 1.0e-4f || std::abs(dot(y, z)) >= 1.0e-4f) {
            continue;
        }
        if (draws.size() + field->second.size() > cards.size()) {
            spdlog::error("Surface Cache card count exceeds its atlas capacity of {} tiles", cards.size());
            return makeError(Error::InvalidArgument);
        }

        const uint32_t beginSection = uint32_t(sections.size());
        for (const auto& sectionNode : std::span(nodes).subspan(firstNode, last - firstNode)) {
            const auto& part = scene.renderPrimitives()[sectionNode.renderPrimitiveIndex];
            const size_t vertexCount = part.indices.empty() ? part.positions.size() : part.indices.size();
            if (vertexCount > UINT32_MAX - vertices.size()) { return makeError(Error::Unsupported); }
            const auto offset = uint32_t(vertices.size());
            appendGeometry(part, transform, determinant < 0.0f, vertices);
            if (vertices.size() != offset) {
                const auto material = sectionNode.materialIndex >= 0 ? sectionNode.materialIndex : part.materialIndex;
                sections.push_back({offset, uint32_t(vertices.size()) - offset, material});
            }
        }
        const uint32_t sectionCount = uint32_t(sections.size()) - beginSection;
        if (sectionCount == 0) { continue; }
        for (const auto& card : field->second) {
            cards[draws.size()] = makeCard(card, transform, uint32_t(node.object), uint32_t(primitive.meshIndex));
            draws.push_back({beginSection, sectionCount});
        }
    }
    return {};
}

Result<> LumenSurfaceCachePass::prepareCaptureGeometry(const scene::Scene& scene)
{
    // The scene identity covers imported geometry and section material indices.
    const CaptureGeometryRevision revision{scene.resourceIdentity(), scene.geometryTransformRevision(),
        scene.visibilityRevision()};
    if (captureGeometry_ && captureGeometry_->revision == revision) { return {}; }

    CaptureGeometry geometry{.revision = revision};
    geometry.cards.resize(layout_.capacity());
    std::vector<CaptureVertex> vertices;
    if (auto result = gather(scene, geometry.cards, geometry.cardsToDraw, geometry.sections, vertices); !result) {
        return result;
    }
    if (!vertices.empty()) {
        auto buffer = upload(*device_, std::as_bytes(std::span(vertices)), BufferUsageBits::Storage);
        if (!buffer) { return makeError(buffer.error()); }
        geometry.vertexBuffer = std::move(*buffer);
    }
    captureGeometry_ = std::move(geometry);
    return {};
}

void LumenSurfaceCachePass::bindMaterial(ParameterWriter& writer, const scene::Scene& scene,
    int32_t index, CaptureParams& params) const
{
    const auto material = resolveWorkingMaterial(index >= 0 ? scene.materials()[index] : scene::RenderMaterial{});
    params.baseColor = {material.baseColorFactor.x, material.baseColorFactor.y, material.baseColorFactor.z, 1.0f};
    params.emissive = pack(material.emissiveFactor);
    params.material = {material.metallicFactor, material.specularFactor, 1.0f, material.normalTextureScale};
    const std::array textures{material.baseColorTexture, material.metallicRoughnessTexture, material.emissiveTexture,
        material.normalTexture};
    const auto indices = materialResources_.logicalTextureIndices();
    const auto& views = materialResources_.materialTextureViews();
    params.textures.fill(UINT32_MAX);
    for (size_t i = 0; i < textures.size(); ++i) {
        const auto& info = textures[i];
        if (info.textureIndex >= 0 && size_t(info.textureIndex) < indices.size()) {
            const auto physical = indices[info.textureIndex];
            if (physical < views.size() && views[physical]) {
                params.textures[i] = writer.sampledImageHandle(views[physical]).index;
            }
        }
        const auto& uv = info.uvTransform;
        params.textureTransforms[i * 2] = {uv[0], uv[1], uv[2], float(textureColorFlags(info.colorMetadata))};
        params.textureTransforms[i * 2 + 1] = {uv[3], uv[4], uv[5], 0.0f};
    }
}

Result<> LumenSurfaceCachePass::rasterize(RenderGraphExecutionContext& context, CommandBuffer& cmd,
    std::span<const CaptureDraw> draws)
{
    std::array<RenderingAttachmentDesc, 3> colors;
    for (size_t i = 0; i < colors.size(); ++i) {
        colors[i] = {.view = context.outputTexture(kCaptureOutputs[i]).view(),
            .layout = TextureLayout::ColorAttachment, .loadOp = LoadOp::Clear, .storeOp = StoreOp::Store};
    }
    colors[1].clearColor = ColorValue{0.5f, 0.5f, 0.0f, 0.0f};
    const RenderingAttachmentDesc hardwareDepth{.view = context.outputTexture("hardwareDepth").view(),
        .layout = TextureLayout::DepthStencilAttachment, .loadOp = LoadOp::Clear, .storeOp = StoreOp::Store,
        .clearDepth = 0.0f};
    if (auto r = cmd.beginRendering({.renderArea = {0, 0, layout_.edge, layout_.edge},
            .colorAttachments = colors, .depthStencilAttachment = &hardwareDepth}); !r) { return r; }
    const auto execution = pipeline_->execution();
    Result<> result;
    for (const auto& draw : draws) {
        const auto& rect = draw.rect;
        if (result = cmd.setViewport({float(rect.x), float(rect.y), float(rect.width), float(rect.height), 0.0f, 1.0f}); !result) { break; }
        cmd.setScissor(rect);
        if (result = draw.params.bindResources(cmd); !result) { break; }
        const auto root = draw.params.root();
        if (result = cmd.bindExecution(execution, &root, sizeof(root)); !result) { break; }
        if (result = cmd.draw(draw.vertexCount); !result) { break; }
    }
    cmd.endRendering();
    return result;
}

Result<> copyUpload(CommandBuffer& cmd, Buffer& source, Buffer& destination)
{
    const auto src = source.slice({});
    const auto dst = destination.slice({});
    if (!src || !dst) { return makeError(Error::InvalidArgument); }
    return cmd.copyBuffer(*src, *dst);
}

Result<> LumenSurfaceCachePass::execute(RenderGraphExecutionContext& context)
{
    const auto& scene = *context.runtimeScene();
    auto registry = ResourceRegistry::forDevice(*device_);
    if (!registry) { return makeError(registry.error()); }
    auto* frame = RenderFrameContext::from(context.commandBuffer());
    if (auto result = prepareCaptureGeometry(scene); !result) { return result; }

    const auto& geometry = *captureGeometry_;
    const auto identity = geometry.revision.sceneIdentity;
    const std::array metadata{uint32_t(geometry.cardsToDraw.size()), layout_.tile, uint32_t(identity), uint32_t(identity >> 32)};
    auto cardUpload = upload(*device_, std::as_bytes(std::span(geometry.cards)), BufferUsageBits::TransferSource);
    if (!cardUpload) { return makeError(cardUpload.error()); }
    auto sceneUpload = upload(*device_, std::as_bytes(std::span(metadata)), BufferUsageBits::TransferSource);
    if (!sceneUpload) { return makeError(sceneUpload.error()); }

    std::vector<CaptureDraw> draws;
    for (uint32_t card = 0; card < geometry.cardsToDraw.size(); ++card) {
        const auto& range = geometry.cardsToDraw[card];
        for (const auto& section : std::span(geometry.sections).subspan(range.firstSection, range.sectionCount)) {
            ParameterWriter writer(*device_, **registry, frame);
            CaptureParams params{};
            params.geometry = {writer.buffer(geometry.vertexBuffer.get()).index, section.firstVertex,
                writer.samplerHandle(kMaterialSampler).index, 0};
            projectCard(params, geometry.cards[card]);
            bindMaterial(writer, scene, section.materialIndex, params);
            auto encoded = writer.encode(params, kCaptureABI);
            if (!encoded) { return makeError(encoded.error()); }
            draws.push_back({layout_.rect(card), section.vertexCount, std::move(*encoded)});
        }
    }

    const auto sun = buildCelestialLightRecords(resolveWorldEnvironment(&scene, context.world()))[0];
    const auto bindLighting = [&](ParameterWriter& writer, auto& params) {
        params.albedo = writer.sampledImageHandle(context.outputTexture("albedo").view());
        params.normal = writer.sampledImageHandle(context.outputTexture("normal").view());
        params.emissive = writer.sampledImageHandle(context.outputTexture("emissive").view());
        params.lighting = writer.storageImageHandle(context.outputTexture("lighting").view());
        params.cards = writer.buffer(context.outputBuffer("cards").buffer());
        params.tileResolution = layout_.tile;
        params.lightDirection = {-sun.direction[0], -sun.direction[1], -sun.direction[2], 0.0f};
        params.illuminance = {sun.irradiance[0], sun.irradiance[1], sun.irradiance[2], 0.0f};
    };

    using Access = RenderGraphResourceAccess;
    const RenderGraphStageUse uploadUses[]{{"cards", Access::BufferTransferWrite}, {"scene", Access::BufferTransferWrite}};
    const RenderGraphStageUse captureUses[]{{"albedo", Access::TextureColorWrite}, {"normal", Access::TextureColorWrite},
        {"emissive", Access::TextureColorWrite}, {"hardwareDepth", Access::TextureDepthStencilWrite}};
    const uint32_t groups = (layout_.edge + 7) / 8;
    std::array<RenderGraphStage, 4> stages{{
        {"Upload card transforms", uploadUses, [&](CommandBuffer& cmd) -> Result<> {
            if (auto r = copyUpload(cmd, **cardUpload, *context.outputBuffer("cards").buffer()); !r) { return r; }
            return copyUpload(cmd, **sceneUpload, *context.outputBuffer("scene").buffer());
        }},
        {"Capture card materials", captureUses, [&](CommandBuffer& cmd) { return rasterize(context, cmd, draws); },
            RenderGraphPassKind::Raster},
    }};
    if (!shadowedLighting()) {
        ParameterWriter writer(*device_, **registry, frame);
        ResolveParams params{};
        bindLighting(writer, params);
        params.hardwareDepth = writer.sampledImageHandle(context.outputTexture("hardwareDepth").view());
        params.depth = writer.storageImageHandle(context.outputTexture("depth").view());
        auto resolveParams = writer.encode(params, kResolveABI);
        if (!resolveParams) { return makeError(resolveParams.error()); }

        const RenderGraphStageUse resolveUses[]{{"albedo", Access::TextureSampleRead}, {"normal", Access::TextureSampleRead},
            {"emissive", Access::TextureSampleRead}, {"hardwareDepth", Access::TextureSampleRead},
            {"cards", Access::BufferStorageRead}, {"depth", Access::TextureStorageWrite}, {"lighting", Access::TextureStorageWrite}};
        stages[2] = {"Encode depth and shade cards", resolveUses, [&](CommandBuffer& cmd) {
            return resolve_.dispatch(cmd, *resolveParams, groups, groups, 1);
        }};
        return context.executeStages(std::span(stages).first(3));
    }

    const auto pageTable = context.inputTexture("pageTable");
    const auto coarseMip = context.inputTexture("coarseMip");
    const auto domains = context.inputBuffer("domain");
    const auto& table = pageTable.desc();
    const auto& coarse = coarseMip.desc();
    const uint32_t clipmapCount = coarse.depth / coarse.width;
    if (clipmapCount < 1 || clipmapCount > 6 || coarse.width != coarse.height ||
        coarse.depth != coarse.width * clipmapCount || table.width != table.height ||
        table.depth != table.width * clipmapCount || domains.desc().size != clipmapCount * kGDFDomainStride) {
        spdlog::error("Surface Cache shadowed lighting requires matching quantized GDF clipmap resources");
        return makeError(Error::InvalidArgument);
    }

    const auto lightRecords = buildPunctualLightRecords(&scene, resolveSceneLighting(&scene, context.world()));
    auto localLights = upload(*device_, std::as_bytes(std::span(lightRecords)), BufferUsageBits::Storage);
    if (!localLights) { return makeError(localLights.error()); }

    ParameterWriter depthWriter(*device_, **registry, frame);
    const EncodeDepthParams depthParams{
        depthWriter.sampledImageHandle(context.outputTexture("normal").view()),
        depthWriter.sampledImageHandle(context.outputTexture("hardwareDepth").view()),
        depthWriter.storageImageHandle(context.outputTexture("depth").view()), layout_.edge};
    auto encodedDepth = depthWriter.encode(depthParams, kEncodeDepthABI);
    if (!encodedDepth) { return makeError(encodedDepth.error()); }

    ParameterWriter lightingWriter(*device_, **registry, frame);
    ShadowedLightingParams lightingParams{};
    bindLighting(lightingWriter, lightingParams);
    lightingParams.depth = lightingWriter.sampledImageHandle(context.outputTexture("depth").view());
    lightingParams.sunCastsShadow = (sun.flags & kGPUCelestialLightCastsShadow) != 0u;
    lightingParams.pageTable = lightingWriter.sampledImageHandle(pageTable.view());
    lightingParams.pageAtlas = lightingWriter.sampledImageHandle(context.inputTexture("pageAtlas").view());
    lightingParams.coarseMip = lightingWriter.sampledImageHandle(coarseMip.view());
    lightingParams.domains = lightingWriter.buffer(domains.buffer());
    lightingParams.localLights = lightingWriter.buffer(localLights->get());
    lightingParams.distanceSampler = lightingWriter.samplerHandle({});
    lightingParams.clipmapCount = clipmapCount;
    lightingParams.atlasSize = layout_.edge;
    if (exportDirectIrradiance()) {
        lightingParams.directIrradiance = lightingWriter.storageImageHandle(context.outputTexture("directIrradiance").view());
        lightingParams.exportDirectIrradiance = true;
    }
    auto encodedLighting = lightingWriter.encode(lightingParams, kShadowedLightingABI);
    if (!encodedLighting) { return makeError(encodedLighting.error()); }

    const RenderGraphStageUse encodeDepthUses[]{{"normal", Access::TextureSampleRead},
        {"hardwareDepth", Access::TextureSampleRead}, {"depth", Access::TextureStorageWrite}};
    const RenderGraphStageUse lightingUses[]{{"albedo", Access::TextureSampleRead}, {"normal", Access::TextureSampleRead},
        {"emissive", Access::TextureSampleRead}, {"depth", Access::TextureSampleRead}, {"cards", Access::BufferStorageRead},
        {"pageTable", Access::TextureSampleRead}, {"pageAtlas", Access::TextureSampleRead}, {"coarseMip", Access::TextureSampleRead},
        {"domain", Access::BufferStorageRead}, {"lighting", Access::TextureStorageWrite},
        {"directIrradiance", Access::TextureStorageWrite}};
    stages[2] = {"Encode card depth", encodeDepthUses, [&](CommandBuffer& cmd) {
        return encodeDepth_.dispatch(cmd, *encodedDepth, groups, groups, 1);
    }};
    stages[3] = {"Shade cards with GDF shadows", std::span{lightingUses}.first(
        std::size(lightingUses) - (lightingParams.exportDirectIrradiance ? 0u : 1u)), [&](CommandBuffer& cmd) {
        cmd.hostWriteBarrier();
        return shadowedLighting_.dispatch(cmd, *encodedLighting, groups, groups, 1);
    }};
    return context.executeStages(stages);
}

} // namespace

std::unique_ptr<RenderGraphPass> createLumenSurfaceCachePass()
{
    return std::make_unique<LumenSurfaceCachePass>();
}

} // namespace metallic::render::builtin_pass
