#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Render/Core/SceneColorConversion.h"
#include "Runtime/Render/Environment/CelestialLighting.h"
#include "Runtime/Render/GAPI/CommandSubmission.h"
#include "Runtime/Render/ImportanceSampling.h"
#include "Runtime/Render/SceneLightResources.h"
#include "Runtime/Render/Streamer/ScenePathTraceResources.h"
#include "Runtime/Render/Subsystem/EnvironmentLightingSubsystem.h"
#include "Runtime/Render/Subsystem/RenderWorld.h"
#include "Runtime/Scene/MeshCards.h"
#include "Runtime/Scene/MeshDistanceField.h"
#include "../../../../../Shaders/Interop/LumenCardUpdate.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <optional>
#include <unordered_map>
#include <spdlog/spdlog.h>

namespace metallic::render::builtin_pass {
namespace {

constexpr char kShaderModule[] = "Features/DistanceField/SurfaceCacheCapture";
constexpr uint64_t kCaptureABI = 0x4c554d454e010001ull;
constexpr uint64_t kResolveABI = 0x4c554d454e010113ull;
constexpr uint64_t kEncodeDepthABI = 0x4c554d454e010104ull;
constexpr uint64_t kShadowedLightingABI = 0x4c554d454e010114ull;
constexpr uint64_t kComposeLightingABI = 0x4c554d454e01010full;
constexpr uint64_t kHardwarePunctualLightingABI = 0x4c554d454e010115ull;
constexpr uint64_t kHardwareRectLightingABI = 0x4c554d454e010116ull;
constexpr uint64_t kHardwareRectComposeABI = 0x4c554d454e010112ull;
constexpr SlangMacroDefine kHardwareLightingDefines[]{{"METALLIC_SURFACE_CACHE_HARDWARE_LOCAL_SHADOWS", "1"},
    {"METALLIC_SURFACE_CACHE_HARDWARE_RECT_SHADOWS", "1"}};
constexpr uint32_t kGDFDomainStride = 32;
constexpr uint32_t kCardsPerDispatchRow = 128u;
constexpr uint32_t kDefaultRectLightSamples = 8u;
constexpr uint32_t kRectReGIRGridSize = 8u;
constexpr uint32_t kRectReGIRLightsPerCell = 64u;
constexpr float kRectReGIRJitter = 1.0f;
constexpr uint32_t kMinStochasticGridRefreshFrames = 4u;
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

struct CaptureMaterial {
    Float4 baseColor, emissive, material;
    std::array<int32_t, 4> textures;
    std::array<Float4, 8> textureTransforms;

    bool operator==(const CaptureMaterial&) const = default;
};

struct alignas(16) CaptureParams {
    std::array<uint32_t, 4> geometry;
    Float4 clipRow0, clipRow1, clipRow2, normalAxisX, normalAxisY;
    Float4 baseColor, emissive, material;
    std::array<uint32_t, 4> textures;
    std::array<Float4, 8> textureTransforms;
};

struct RectLightingParams {
    ShaderBuffer lights;
    uint32_t count;
    ShaderBuffer grid;
    ShaderSampledImage pdf;
    uint32_t sampleCount;
};

struct SunAtmosphereParams {
    ShaderBuffer parameters;
    ShaderSampledImage transmittance, cloudShadow;
};

struct alignas(16) ResolveParams {
    ShaderSampledImage albedo, normal, emissive, hardwareDepth;
    ShaderStorageImage depth, lighting;
    ShaderBuffer cards;
    uint32_t tileResolution;
    Float4 lightDirection, illuminance;
    ShaderBuffer updateTiles;
    uint32_t updateCount;
    RectLightingParams rectLighting;
    ShaderStorageImage deterministicIrradiance;
    uint32_t separateLighting;
    SunAtmosphereParams atmosphere;
};

struct EncodeDepthParams {
    ShaderSampledImage normal, hardwareDepth;
    ShaderStorageImage depth;
    uint32_t atlasSize, tileResolution;
    ShaderBuffer updateTiles;
    uint32_t updateCount;
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
    ShaderBuffer updateTiles;
    uint32_t updateCount;
    RectLightingParams rectLighting;
    ShaderStorageImage deterministicIrradiance;
    uint32_t separateLighting;
    SunAtmosphereParams atmosphere;
    uint32_t reserved;
};

struct HardwarePunctualLightingParams {
    ShadowedLightingParams base;
    uint64_t scene;
};

struct HardwareRectLightingParams {
    ShadowedLightingParams base;
    uint64_t scene;
    uint32_t samplePhase;
};

struct ComposeLightingParams {
    ShaderSampledImage albedo, normal, emissive, depth, deterministicIrradiance;
    ShaderStorageImage stochasticDirectHistory, lighting, directIrradiance;
    ShaderBuffer cards, cardUpdates, updateTiles;
    uint32_t atlasSize, tileResolution, updateCount, samplePhase, exportDirectIrradiance;
    RectLightingParams rectLighting;
};

struct HardwareRectComposeParams {
    ComposeLightingParams base;
    uint64_t scene;
};

static_assert(sizeof(CaptureVertex) == 64 && sizeof(SurfaceCard) == 80);
static_assert(sizeof(CaptureParams) == 288 && offsetof(CaptureParams, textureTransforms) == 160);
static_assert(sizeof(RectLightingParams) == 20 && offsetof(RectLightingParams, sampleCount) == 16);
static_assert(sizeof(ResolveParams) == 112 && offsetof(ResolveParams, updateTiles) == 64 &&
    offsetof(ResolveParams, rectLighting) == 72 && offsetof(ResolveParams, deterministicIrradiance) == 92 &&
    offsetof(ResolveParams, atmosphere) == 100);
static_assert(sizeof(EncodeDepthParams) == 28 && offsetof(EncodeDepthParams, updateTiles) == 20);
static_assert(sizeof(ShadowedLightingParams) == 160 && offsetof(ShadowedLightingParams, pageTable) == 64 &&
    offsetof(ShadowedLightingParams, directIrradiance) == 96 && offsetof(ShadowedLightingParams, updateTiles) == 104 &&
    offsetof(ShadowedLightingParams, rectLighting) == 112 && offsetof(ShadowedLightingParams, deterministicIrradiance) == 132 &&
    offsetof(ShadowedLightingParams, atmosphere) == 140);
static_assert(sizeof(HardwarePunctualLightingParams) == 176 && offsetof(HardwarePunctualLightingParams, scene) == 160);
static_assert(sizeof(HardwareRectLightingParams) == 176 && offsetof(HardwareRectLightingParams, scene) == 160 &&
    offsetof(HardwareRectLightingParams, samplePhase) == 168);
static_assert(sizeof(ComposeLightingParams) == 84 && offsetof(ComposeLightingParams, cardUpdates) == 36 &&
    offsetof(ComposeLightingParams, rectLighting) == 64);
static_assert(sizeof(HardwareRectComposeParams) == 96 && offsetof(HardwareRectComposeParams, scene) == 88);

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
    uint32_t sourceCard;
    uint64_t transformRevision;
};

struct CaptureGeometryRevision {
    uint64_t sceneIdentity, transforms, visibility, structure;

    bool operator==(const CaptureGeometryRevision&) const = default;
};

struct CaptureGeometry {
    CaptureGeometryRevision revision;
    std::vector<SurfaceCard> cards;
    std::vector<CaptureSection> sections;
    std::vector<CaptureCard> cardsToDraw;
    std::unique_ptr<Buffer> vertexBuffer;
    Float4 rectGridCenterRadius{};
};

struct CaptureDraw {
    Rect rect;
    uint32_t vertexCount;
    EncodedParameters params;
};

struct CaptureMaterials {
    uint64_t revision;
    std::vector<CaptureMaterial> records;
};

bool sameRectLights(std::span<const GPURectLight> a, std::span<const GPURectLight> b)
{
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size_bytes()) == 0);
}

struct RectLightSampling {
    Float4 centerRadius;
    std::vector<GPURectLight> lights;
    std::unique_ptr<Buffer> lightBuffer;
    ImportancePdfTexture pdf;
    ReGIRLightSelector grid;
    uint32_t gridPhase = 0;
};

struct CaptureLighting {
    Float4 direction, illuminance;
    bool sunCastsShadow;
    bool hardwarePunctualShadows;
    bool hardwareRectShadows;
    std::vector<GPUPunctualLight> punctual;
    std::shared_ptr<IESProfilePublication> iesProfiles;
    std::vector<GPURectLight> rectLights;
    std::array<float, 3> clipmapEye{};
    uint32_t rectLightSamples = 0;
    Float4 rectGridCenterRadius{};
    std::optional<std::array<uint64_t, 2>> atmosphereRevisions;

    bool sameRectLighting(const CaptureLighting& other) const
    {
        return hardwareRectShadows == other.hardwareRectShadows && rectLightSamples == other.rectLightSamples &&
            rectGridCenterRadius == other.rectGridCenterRadius && sameRectLights(rectLights, other.rectLights);
    }

    bool sameLocalLights(const CaptureLighting& other) const
    {
        // The first punctual row only contains the count and exposure.
        return hardwarePunctualShadows == other.hardwarePunctualShadows && punctual.size() == other.punctual.size() &&
            (punctual.size() <= 1 || std::memcmp(punctual.data() + 1, other.punctual.data() + 1,
                (punctual.size() - 1) * sizeof(GPUPunctualLight)) == 0) &&
            sameRectLighting(other);
    }

    bool sameLights(const CaptureLighting& other) const
    {
        return direction == other.direction && illuminance == other.illuminance &&
            sunCastsShadow == other.sunCastsShadow && atmosphereRevisions == other.atmosphereRevisions && sameLocalLights(other);
    }

    bool tracesShadows() const
    {
        return (sunCastsShadow && illuminance != Float4{}) || punctual.size() > 1;
    }
};

struct PublishedSurfaceCache {
    std::array<uint64_t, 3> atlasAllocations{};
    std::shared_ptr<const CaptureGeometry> geometry;
    std::shared_ptr<const CaptureMaterials> materials;
    CaptureLighting lighting;
    std::optional<uint64_t> nonSunLightingRevision;
    std::vector<uint32_t> pendingLighting;
    uint32_t nextLightingSlot = 0;
    uint32_t nextStochasticSlot = 0;
};

struct SurfaceCacheUpdate {
    bool fullCapture = true;
    bool uploadCards = true;
    bool publish = false;
    bool sampledStochastic = false;
    std::vector<uint32_t> captureTiles;
    std::vector<uint32_t> lightingTiles;
    std::vector<uint32_t> compositionTiles;
    std::vector<uint32_t> flags;
};

class LumenSurfaceCachePass final : public RasterPass {
public:
    bool supportsFrameOverlap() const override { return true; }
    bool supportsExternalFrameOverlap() const override { return false; }
    std::span<const RenderSubsystemId> requiredSubsystems() const override
    {
        static constexpr std::array required{EnvironmentLightingSubsystem::kSubsystemId};
        return required;
    }
    RenderGraphSceneDependency sceneDependency() const override
    {
        if (hardwarePunctualShadows()) {
            return {.source = RenderGraphSceneSource::Input, .inputs = {"accelerationStructure"},
                .recompileOnTransform = false};
        }
        return {.source = RenderGraphSceneSource::World, .recompileOnTransform = false};
    }
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override;
    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override;
    Result<> declareResourceRequests(const RenderGraphResourceRequestContext& context,
        RenderGraphResourceRequestBuilder& builder, std::string& log) override;
    Result<> execute(RenderGraphExecutionContext& context) override;

private:
    bool shadowedLighting() const;
    bool hardwarePunctualShadows() const;
    bool hardwareRectShadows() const;
    bool exportDirectIrradiance() const;
    bool persistentCards() const;
    bool objectCardRanges() const;
    bool separateLighting() const;
    Result<> prepareCaptureGeometry(const scene::Scene& scene);
    Result<> gather(const scene::Scene& scene, std::vector<SurfaceCard>& cards, std::vector<CaptureCard>& draws,
        std::vector<CaptureSection>& sections, std::vector<CaptureVertex>& vertices) const;
    void bindMaterial(ParameterWriter& writer, const CaptureMaterial& material, CaptureParams& params) const;
    Result<> rasterize(RenderGraphExecutionContext& context, CommandBuffer& cmd, std::span<const CaptureDraw> draws,
        std::span<const uint32_t> captureTiles, bool fullCapture);

    Device* device_ = nullptr;
    AtlasLayout layout_;
    uint32_t maxDirectLightingCardsPerFrame_ = 0;
    uint32_t maxStochasticDirectLightingCardsPerFrame_ = 0;
    uint32_t rectLightSamples_ = kDefaultRectLightSamples;
    std::unordered_map<int32_t, std::vector<scene::MeshCardBuildRecord>> fields_;
    std::shared_ptr<const CaptureGeometry> captureGeometry_;
    std::shared_ptr<std::optional<PublishedSurfaceCache>> published_;
    std::shared_ptr<std::shared_ptr<RectLightSampling>> rectSampling_ =
        std::make_shared<std::shared_ptr<RectLightSampling>>();
    ImportancePdfCompute rectPdfCompute_;
    ScenePathTraceResources materialResources_;
    std::unique_ptr<GraphicsPipeline> pipeline_;
    ComputeKernel resolve_;
    ComputeKernel encodeDepth_;
    ComputeKernel shadowedLighting_;
    ComputeKernel composeLighting_;
};

bool LumenSurfaceCachePass::shadowedLighting() const
{
    const auto property = properties().find("shadowedLighting");
    return property != properties().end() && property->is_boolean() && property->get<bool>();
}

bool LumenSurfaceCachePass::hardwarePunctualShadows() const
{
    const auto property = properties().find("hardwarePunctualShadows");
    return property != properties().end() && property->is_boolean() && property->get<bool>();
}

bool LumenSurfaceCachePass::hardwareRectShadows() const
{
    const auto property = properties().find("hardwareRectShadows");
    return property != properties().end() && property->is_boolean() && property->get<bool>();
}

bool LumenSurfaceCachePass::exportDirectIrradiance() const
{
    const auto property = properties().find("exportDirectIrradiance");
    return property != properties().end() && property->is_boolean() && property->get<bool>();
}

bool LumenSurfaceCachePass::persistentCards() const
{
    const auto property = properties().find("persistentCards");
    return property != properties().end() && property->is_boolean() && property->get<bool>();
}

bool LumenSurfaceCachePass::objectCardRanges() const
{
    const auto property = properties().find("objectCardRanges");
    return property != properties().end() && property->is_boolean() && property->get<bool>();
}

bool LumenSurfaceCachePass::separateLighting() const
{
    const auto property = properties().find("maxStochasticDirectLightingCardsPerFrame");
    return property != properties().end() && property->is_number_integer() && property->get<double>() > 0.0;
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
    const bool separateDirectLighting = separateLighting();
    const bool useGdfShadows = shadowedLighting();
    if (separateDirectLighting || useGdfShadows) {
        depth.stageAccess(RenderGraphResourceAccess::TextureSampleRead);
    }
    if (separateDirectLighting) {
        reflection.addTextureOutput("deterministicIrradiance", "Direct irradiance before stochastic Rect lighting")
            .texture2D(layout.edge, layout.edge).storageWrite().stageAccess(RenderGraphResourceAccess::TextureSampleRead)
            .format = Format::RGBA32Sfloat;
        reflection.addTextureOutput("stochasticDirectHistory", "Accumulated Rect irradiance and batch count")
            .texture2D(layout.edge, layout.edge).storageReadWrite()
            .stageAccess(RenderGraphResourceAccess::TextureStorageReadWrite).format = Format::RGBA32Sfloat;
    }
    if (useGdfShadows) {
        reflection.addTextureInput("pageTable", "GDF page table").texture3D(0, 0, 0).sampledRead()
            .stageAccess(RenderGraphResourceAccess::TextureSampleRead).format = Format::R32Uint;
        reflection.addTextureInput("pageAtlas", "GDF fine page distances").texture3D(0, 0, 0).sampledRead()
            .stageAccess(RenderGraphResourceAccess::TextureSampleRead).format = Format::R8Unorm;
        reflection.addTextureInput("coarseMip", "GDF coarse distances").texture3D(0, 0, 0).sampledRead()
            .stageAccess(RenderGraphResourceAccess::TextureSampleRead).format = Format::R8Unorm;
        reflection.addBufferInput("domain", "GDF clipmap domains").buffer(0, kGDFDomainStride).storageRead()
            .stageAccess(RenderGraphResourceAccess::BufferStorageRead);
    }
    if (hardwarePunctualShadows()) {
        reflection.addAccelerationStructureInput("accelerationStructure", "Current scene TLAS for card point and spot shadows")
            .accelerationStructureRead().stageAccess(RenderGraphResourceAccess::AccelerationStructureShaderRead);
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
    const auto metadataWords = objectCardRanges() ? kLumenCardSceneRangeBase + layout.capacity() * kLumenCardSceneRangeStride :
        kLumenCardSceneHeaderWords;
    reflection.addBufferOutput("scene", "Card count, tile resolution, scene identity and optional object ranges")
        .buffer(metadataWords * sizeof(uint32_t), sizeof(uint32_t)).bufferLayout("u32").transferWrite();
    if (persistentCards()) {
        auto& cardUpdates = reflection.addBufferOutput("cardUpdates", "Capture, direct lighting, and indirect history invalidation flags per tile")
            .buffer(layout.capacity() * sizeof(uint32_t), sizeof(uint32_t)).bufferLayout("u32").transferWrite();
        if (separateLighting()) { cardUpdates.stageAccess(RenderGraphResourceAccess::BufferStorageRead); }
        reflection.addBufferOutput("updateStatistics", "Full capture flag, captured cards, and relit cards")
            .buffer(3 * sizeof(uint32_t), sizeof(uint32_t)).bufferLayout("u32").transferWrite();
    }
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
    const auto hardware = properties().find("hardwarePunctualShadows");
    const bool hardwarePunctual = hardwarePunctualShadows();
    if (hardware != properties().end() && (!hardware->is_boolean() || (hardwarePunctual && !shadowedLighting()))) {
        log = "Surface Cache hardwarePunctualShadows must be a boolean and requires shadowedLighting when enabled";
        return makeError(Error::InvalidArgument);
    }
    const auto rectHardware = properties().find("hardwareRectShadows");
    const bool hardwareRect = hardwareRectShadows();
    if (rectHardware != properties().end() && (!rectHardware->is_boolean() || (hardwareRect && !hardwarePunctual))) {
        log = "Surface Cache hardwareRectShadows must be a boolean and requires hardwarePunctualShadows when enabled";
        return makeError(Error::InvalidArgument);
    }
    const auto caps = context.device->capabilities();
    if (hardwarePunctual && (!caps.rayQuery || !caps.rayTracingAccelerationStructure)) {
        log = "Surface Cache hardwarePunctualShadows requires ray queries and scene acceleration structures";
        return makeError(Error::Unsupported);
    }
    const auto persistent = properties().find("persistentCards");
    if (persistent != properties().end() && !persistent->is_boolean()) {
        log = "Surface Cache persistentCards must be a boolean";
        return makeError(Error::InvalidArgument);
    }
    const auto cardRanges = properties().find("objectCardRanges");
    if (cardRanges != properties().end() && !cardRanges->is_boolean()) {
        log = "Surface Cache objectCardRanges must be a boolean";
        return makeError(Error::InvalidArgument);
    }
    const auto directLightingBudget = properties().find("maxDirectLightingCardsPerFrame");
    if (directLightingBudget != properties().end() && (!directLightingBudget->is_number_integer() ||
            directLightingBudget->get<double>() < 0.0 || directLightingBudget->get<double>() > UINT32_MAX ||
            (*directLightingBudget != 0 && !persistentCards()))) {
        log = "Surface Cache maxDirectLightingCardsPerFrame must be an unsigned integer and requires persistentCards when positive";
        return makeError(Error::InvalidArgument);
    }
    maxDirectLightingCardsPerFrame_ = properties().value("maxDirectLightingCardsPerFrame", 0u);
    const auto stochasticBudget = properties().find("maxStochasticDirectLightingCardsPerFrame");
    if (stochasticBudget != properties().end() && (!stochasticBudget->is_number_integer() ||
        stochasticBudget->get<double>() < 0.0 || stochasticBudget->get<double>() > UINT32_MAX ||
        (separateLighting() && !persistentCards()))) {
        log = "Surface Cache maxStochasticDirectLightingCardsPerFrame must be an unsigned integer and requires persistentCards when positive";
        return makeError(Error::InvalidArgument);
    }
    maxStochasticDirectLightingCardsPerFrame_ = properties().value("maxStochasticDirectLightingCardsPerFrame", 0u);
    const auto rectLightSamples = properties().find("rectLightSamples");
    if (rectLightSamples != properties().end() && (!rectLightSamples->is_number_integer() ||
        rectLightSamples->get<double>() < 0.0 || rectLightSamples->get<double>() > kRectReGIRLightsPerCell)) {
        log = "Surface Cache rectLightSamples must be an integer from 0 to 64";
        return makeError(Error::InvalidArgument);
    }
    rectLightSamples_ = properties().value("rectLightSamples", kDefaultRectLightSamples);
    captureGeometry_.reset();
    if (!persistentCards()) { published_.reset(); }
    else if (!published_) { published_ = std::make_shared<std::optional<PublishedSurfaceCache>>(); }
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
    const std::span<const SlangMacroDefine> hardwareDefines{kHardwareLightingDefines};
    const std::span<const SlangMacroDefine> rectDefines = hardwareRect ? hardwareDefines.last(1) : hardwareDefines.first(0);
    if (separateLighting()) {
        if (auto r = ShaderRegistry::instance().getComputeKernel(*device_,
                {.moduleName = kShaderModule, .entryPointName = "surfaceCacheComposeLightingMain",
                    .searchPath = PROJECT_SOURCE_DIR "/Shaders", .macroDefines = rectDefines},
                {.parameters = hardwareRect ? parameterAbi<HardwareRectComposeParams>(kHardwareRectComposeABI) :
                        parameterAbi<ComposeLightingParams>(kComposeLightingABI),
                    .debugName = "LumenSurfaceCacheComposeLighting"}, composeLighting_, log); !r) {
            return r;
        }
    }
    if (shadowedLighting()) {
        if (auto r = ShaderRegistry::instance().getComputeKernel(*device_,
                {.moduleName = kShaderModule, .entryPointName = "surfaceCacheEncodeDepthMain",
                    .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
                {.parameters = parameterAbi<EncodeDepthParams>(kEncodeDepthABI),
                    .debugName = "LumenSurfaceCacheEncodeDepth"}, encodeDepth_, log); !r) {
            return r;
        }
        const auto lightingDefines = hardwareDefines.first(hardwareRect ? 2u : hardwarePunctual ? 1u : 0u);
        const auto lightingAbi = hardwareRect ? parameterAbi<HardwareRectLightingParams>(kHardwareRectLightingABI) :
            hardwarePunctual ? parameterAbi<HardwarePunctualLightingParams>(kHardwarePunctualLightingABI) :
            parameterAbi<ShadowedLightingParams>(kShadowedLightingABI);
        return ShaderRegistry::instance().getComputeKernel(*device_,
            {.moduleName = kShaderModule, .entryPointName = "surfaceCacheShadowedLightingMain",
                .searchPath = PROJECT_SOURCE_DIR "/Shaders", .macroDefines = lightingDefines},
            {.parameters = lightingAbi,
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

        const auto objectSections = std::span(nodes).subspan(firstNode, last - firstNode);
        // The card layout was baked with every section sharing the same placement.
        if (std::ranges::any_of(objectSections.subspan(1), [&](const auto& section) {
                return !std::ranges::equal(node.worldMatrix.a, section.worldMatrix.a);
            })) { continue; }

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
        for (const auto& sectionNode : objectSections) {
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
        uint32_t sourceCard = 0;
        for (const auto& card : field->second) {
            cards[draws.size()] = makeCard(card, transform, uint32_t(node.object), uint32_t(primitive.meshIndex));
            draws.push_back({beginSection, sectionCount, sourceCard++, node.transformRevision});
        }
    }
    return {};
}

Float4 rectGridCenterRadius(std::span<const SurfaceCard> cards)
{
    Float4 centerRadius{};
    for (size_t axis = 0; axis < 3; ++axis) {
        float lower = std::numeric_limits<float>::infinity();
        float upper = -lower;
        for (const auto& card : cards) {
            const float extent = std::abs(card.axisX[axis] * card.axisX[3]) +
                std::abs(card.axisY[axis] * card.axisY[3]) + std::abs(card.axisZ[axis] * card.axisZ[3]);
            lower = std::min(lower, card.origin[axis] - extent);
            upper = std::max(upper, card.origin[axis] + extent);
        }
        centerRadius[axis] = std::midpoint(lower, upper);
        centerRadius[3] = std::max(centerRadius[3], upper - centerRadius[axis]);
    }
    // Reserve room for the sampling jitter on either side of the card bounds.
    centerRadius[3] = std::max(centerRadius[3], 0.01f) *
        (float(kRectReGIRGridSize) / (float(kRectReGIRGridSize) - 2.0f * kRectReGIRJitter));
    return centerRadius;
}

Result<> LumenSurfaceCachePass::prepareCaptureGeometry(const scene::Scene& scene)
{
    // The scene identity covers imported geometry and section material indices.
    const CaptureGeometryRevision revision{scene.resourceIdentity(), scene.geometryTransformRevision(),
        scene.visibilityRevision(), scene.sceneGraph().structuralRevision()};
    if (captureGeometry_ && captureGeometry_->revision == revision) { return {}; }

    CaptureGeometry geometry{.revision = revision};
    geometry.cards.resize(layout_.capacity());
    std::vector<CaptureVertex> vertices;
    if (auto result = gather(scene, geometry.cards, geometry.cardsToDraw, geometry.sections, vertices); !result) {
        return result;
    }
    if (!geometry.cardsToDraw.empty()) {
        geometry.rectGridCenterRadius = rectGridCenterRadius(std::span(geometry.cards).first(geometry.cardsToDraw.size()));
    }
    if (!vertices.empty()) {
        auto buffer = upload(*device_, std::as_bytes(std::span(vertices)), BufferUsageBits::Storage);
        if (!buffer) { return makeError(buffer.error()); }
        geometry.vertexBuffer = std::move(*buffer);
    }
    captureGeometry_ = std::make_shared<CaptureGeometry>(std::move(geometry));
    return {};
}

CaptureMaterial captureMaterial(const scene::RenderMaterial& source)
{
    const auto material = resolveWorkingMaterial(source);
    CaptureMaterial params{};
    params.baseColor = {material.baseColorFactor.x, material.baseColorFactor.y, material.baseColorFactor.z, 1.0f};
    params.emissive = pack(material.emissiveFactor);
    params.material = {material.metallicFactor, material.specularFactor, 1.0f, material.normalTextureScale};
    const std::array textures{material.baseColorTexture, material.metallicRoughnessTexture, material.emissiveTexture,
        material.normalTexture};
    for (size_t i = 0; i < textures.size(); ++i) {
        const auto& info = textures[i];
        params.textures[i] = info.textureIndex;
        const auto& uv = info.uvTransform;
        params.textureTransforms[i * 2] = {uv[0], uv[1], uv[2], float(textureColorFlags(info.colorMetadata))};
        params.textureTransforms[i * 2 + 1] = {uv[3], uv[4], uv[5], 0.0f};
    }
    return params;
}

const CaptureMaterial& defaultCaptureMaterial()
{
    static const CaptureMaterial material = captureMaterial(scene::RenderMaterial{});
    return material;
}

const CaptureMaterial& captureMaterialAt(const CaptureMaterials& materials, int32_t index)
{
    return index >= 0 ? materials.records[size_t(index)] : defaultCaptureMaterial();
}

std::shared_ptr<const CaptureMaterials> captureMaterials(const scene::Scene& scene)
{
    auto materials = std::make_shared<CaptureMaterials>();
    materials->revision = scene.materialRevision();
    materials->records.reserve(scene.materials().size());
    for (const auto& material : scene.materials()) { materials->records.push_back(captureMaterial(material)); }
    return materials;
}

bool sameCardRoster(const CaptureGeometry& previous, const CaptureGeometry& next)
{
    if (&previous == &next) { return true; }
    if (previous.revision.sceneIdentity != next.revision.sceneIdentity ||
        previous.cardsToDraw.size() != next.cardsToDraw.size()) {
        return false;
    }
    for (size_t i = 0; i < next.cardsToDraw.size(); ++i) {
        if (previous.cards[i].metadata != next.cards[i].metadata ||
            previous.cardsToDraw[i].sourceCard != next.cardsToDraw[i].sourceCard) {
            return false;
        }
    }
    return true;
}

bool sameCardMaterials(size_t slot, const PublishedSurfaceCache& previous, const PublishedSurfaceCache& next)
{
    if (previous.geometry == next.geometry && previous.materials == next.materials) { return true; }
    const auto& oldCard = previous.geometry->cardsToDraw[slot];
    const auto& newCard = next.geometry->cardsToDraw[slot];
    if (oldCard.sectionCount != newCard.sectionCount) { return false; }
    for (size_t i = 0; i < newCard.sectionCount; ++i) {
        const auto& oldSection = previous.geometry->sections[oldCard.firstSection + i];
        const auto& newSection = next.geometry->sections[newCard.firstSection + i];
        if (captureMaterialAt(*previous.materials, oldSection.materialIndex) !=
            captureMaterialAt(*next.materials, newSection.materialIndex)) { return false; }
    }
    return true;
}

SurfaceCacheUpdate planSurfaceCacheUpdate(const PublishedSurfaceCache* previous, PublishedSurfaceCache& next,
    uint32_t capacity, uint32_t maxDirectLightingCardsPerFrame, uint32_t maxStochasticCardsPerFrame)
{
    SurfaceCacheUpdate update;
    update.fullCapture = !previous || !sameCardRoster(*previous->geometry, *next.geometry);
    update.uploadCards = update.fullCapture;
    update.flags.resize(capacity);
    const bool changedLights = previous && !previous->lighting.sameLights(next.lighting);
    const bool changedNonSunLighting = previous && previous->nonSunLightingRevision != next.nonSunLightingRevision;
    const bool resetIndirect = changedLights && (!next.nonSunLightingRevision || changedNonSunLighting ||
        !previous->lighting.sameLocalLights(next.lighting));
    const bool hardwareRectLighting = next.lighting.hardwareRectShadows && !next.lighting.rectLights.empty();
    const bool stochastic = maxStochasticCardsPerFrame != 0u && (hardwareRectLighting || next.lighting.rectLightSamples != 0u);
    const bool changedRectVisibility = previous && hardwareRectLighting &&
        (previous->geometry->revision != next.geometry->revision || previous->materials != next.materials);
    const bool resetStochastic = stochastic &&
        (update.fullCapture || changedRectVisibility || !previous->lighting.sameRectLighting(next.lighting));
    const bool continueLighting = !update.fullCapture && !previous->pendingLighting.empty();
    // An opacity-mode edit can change hardware occlusion on other cards.
    const bool changedShadowMaterials = previous && next.lighting.hardwarePunctualShadows && next.lighting.punctual.size() > 1u &&
        previous->materials != next.materials;
    update.publish = !previous || previous->geometry != next.geometry || previous->materials != next.materials ||
        changedLights || changedNonSunLighting || previous->lighting.clipmapEye != next.lighting.clipmapEye || continueLighting;
    if (!update.publish && !stochastic) { return update; }
    const bool allLighting = update.fullCapture || changedLights || changedShadowMaterials || changedRectVisibility || (next.lighting.tracesShadows() &&
        (previous->geometry->revision != next.geometry->revision || previous->lighting.clipmapEye != next.lighting.clipmapEye));
    const bool budgetLighting = !update.fullCapture && maxDirectLightingCardsPerFrame != 0;
    const uint32_t cardCount = uint32_t(next.geometry->cardsToDraw.size());
    if (!update.fullCapture) {
        next.nextLightingSlot = previous->nextLightingSlot;
        next.nextStochasticSlot = previous->nextStochasticSlot;
    }
    if (continueLighting) { next.pendingLighting = previous->pendingLighting; }
    if (budgetLighting && allLighting && next.pendingLighting.empty()) { next.pendingLighting.resize(cardCount); }
    const uint32_t relightFlags = kLumenCardRelit | (resetIndirect ? kLumenCardResetLighting : 0u);

    for (uint32_t slot = 0; slot < cardCount; ++slot) {
        bool capture = update.fullCapture;
        if (!capture) {
            const bool transformed = previous->geometry->cardsToDraw[slot].transformRevision !=
                next.geometry->cardsToDraw[slot].transformRevision;
            update.uploadCards = update.uploadCards || transformed;
            capture = transformed || !sameCardMaterials(slot, *previous, next);
        }
        if (capture) {
            update.captureTiles.push_back(slot);
            update.flags[slot] |= kLumenCardCaptured;
        }
        const uint32_t pending = next.pendingLighting.empty() ? 0u : next.pendingLighting[slot];
        uint32_t wanted = pending | (allLighting || capture ? relightFlags : 0u);
        if (wanted == 0) { continue; }
        if (!stochastic) { wanted &= ~kLumenCardResetStochasticLighting; }
        else if (resetStochastic || capture) { wanted |= kLumenCardResetStochasticLighting; }
        if (budgetLighting && !capture) { next.pendingLighting[slot] = wanted; }
        else {
            update.lightingTiles.push_back(slot);
            update.flags[slot] |= wanted;
            if (pending != 0) { next.pendingLighting[slot] = 0; }
        }
    }
    if (budgetLighting && !next.pendingLighting.empty()) {
        const auto first = next.nextLightingSlot;
        // Keep the cursor across light edits so an animated light cannot starve later atlas slots.
        for (uint32_t offset = 0; offset < cardCount && update.lightingTiles.size() < maxDirectLightingCardsPerFrame; ++offset) {
            const auto slot = (first + offset) % cardCount;
            const uint32_t wanted = next.pendingLighting[slot];
            if (wanted == 0) { continue; }
            update.lightingTiles.push_back(slot);
            update.flags[slot] |= wanted;
            next.pendingLighting[slot] = 0;
            next.nextLightingSlot = (slot + 1) % cardCount;
        }
    }
    if (std::ranges::none_of(next.pendingLighting, [](uint32_t flags) { return flags != 0; })) { next.pendingLighting.clear(); }

    if (maxStochasticCardsPerFrame == 0u) { return update; }
    update.compositionTiles = update.lightingTiles;
    if (!stochastic) { return update; }

    for (const auto slot : update.lightingTiles) {
        if (update.flags[slot] & kLumenCardResetStochasticLighting) {
            update.flags[slot] |= kLumenCardStochasticRelit;
            update.sampledStochastic = true;
        }
    }
    // Opportunistic refresh has its own round robin. Cards waiting for a
    // mandatory relight still hold the previous deterministic base.
    const auto first = next.nextStochasticSlot;
    uint32_t refreshed = 0u;
    for (uint32_t offset = 0; offset < cardCount && refreshed < maxStochasticCardsPerFrame; ++offset) {
        const auto slot = (first + offset) % cardCount;
        if ((update.flags[slot] & kLumenCardStochasticRelit) ||
            (!next.pendingLighting.empty() && next.pendingLighting[slot] != 0u)) { continue; }
        if (!(update.flags[slot] & kLumenCardRelit)) { update.compositionTiles.push_back(slot); }
        update.flags[slot] |= kLumenCardRelit | kLumenCardStochasticRelit;
        next.nextStochasticSlot = (slot + 1u) % cardCount;
        ++refreshed;
    }
    update.sampledStochastic = update.sampledStochastic || refreshed != 0u;
    update.publish = update.publish || refreshed != 0u;
    return update;
}

void LumenSurfaceCachePass::bindMaterial(ParameterWriter& writer, const CaptureMaterial& material, CaptureParams& params) const
{
    params.baseColor = material.baseColor;
    params.emissive = material.emissive;
    params.material = material.material;
    params.textureTransforms = material.textureTransforms;
    const auto indices = materialResources_.logicalTextureIndices();
    const auto& views = materialResources_.materialTextureViews();
    params.textures.fill(UINT32_MAX);
    for (size_t i = 0; i < material.textures.size(); ++i) {
        const int32_t index = material.textures[i];
        if (index >= 0 && size_t(index) < indices.size()) {
            const auto physical = indices[index];
            if (physical < views.size() && views[physical]) {
                params.textures[i] = writer.sampledImageHandle(views[physical]).index;
            }
        }
    }
}

Result<> LumenSurfaceCachePass::rasterize(RenderGraphExecutionContext& context, CommandBuffer& cmd,
    std::span<const CaptureDraw> draws, std::span<const uint32_t> captureTiles, bool fullCapture)
{
    const LoadOp load = fullCapture ? LoadOp::Clear : LoadOp::Load;
    std::array<RenderingAttachmentDesc, 3> colors;
    for (size_t i = 0; i < colors.size(); ++i) {
        colors[i] = {.view = context.outputTexture(kCaptureOutputs[i]).view(),
            .layout = TextureLayout::ColorAttachment, .loadOp = load, .storeOp = StoreOp::Store};
    }
    colors[1].clearColor = ColorValue{0.5f, 0.5f, 0.0f, 0.0f};
    const RenderingAttachmentDesc hardwareDepth{.view = context.outputTexture("hardwareDepth").view(),
        .layout = TextureLayout::DepthStencilAttachment, .loadOp = load, .storeOp = StoreOp::Store,
        .clearDepth = 0.0f};
    if (auto r = cmd.beginRendering({.renderArea = {0, 0, layout_.edge, layout_.edge},
            .colorAttachments = colors, .depthStencilAttachment = &hardwareDepth}); !r) { return r; }
    if (!fullCapture) {
        for (uint32_t slot : captureTiles) {
            const Rect rect = layout_.rect(slot);
            for (uint32_t attachment = 0; attachment < colors.size(); ++attachment) {
                cmd.clearColorAttachment(attachment, colors[attachment].clearColor, rect);
            }
            cmd.clearDepthAttachment(hardwareDepth.clearDepth, rect);
        }
    }
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

struct CardDispatch {
    std::unique_ptr<Buffer> tiles;
    uint32_t count = 0;
    std::array<uint32_t, 3> groups;

    Result<> record(ComputeKernel& kernel, CommandBuffer& cmd, const EncodedParameters& params,
        bool hostLightBuffer = false) const
    {
        if (tiles || hostLightBuffer) { cmd.hostWriteBarrier(); }
        return kernel.dispatch(cmd, params, groups[0], groups[1], groups[2]);
    }
};

Result<CardDispatch> makeCardDispatch(Device& device, const AtlasLayout& layout, std::span<const uint32_t> tiles,
    bool full)
{
    if (full) {
        const uint32_t groups = (layout.edge + 7u) / 8u;
        return CardDispatch{nullptr, 0u, {groups, groups, 1u}};
    }

    auto buffer = upload(device, std::as_bytes(tiles), BufferUsageBits::Storage);
    if (!buffer) { return makeError(buffer.error()); }
    const uint32_t count = uint32_t(tiles.size());
    const uint32_t groupsPerTile = (layout.tile + 7u) / 8u;
    return CardDispatch{std::move(*buffer), count, {groupsPerTile * std::min(kCardsPerDispatchRow, count),
        groupsPerTile, (count + kCardsPerDispatchRow - 1u) / kCardsPerDispatchRow}};
}

Result<> LumenSurfaceCachePass::declareResourceRequests(const RenderGraphResourceRequestContext& context,
    RenderGraphResourceRequestBuilder& builder, std::string& log)
{
    const auto worldEnvironment = resolveWorldEnvironment(context.runtimeScene(), context.world());
    if (worldEnvironment.source != environment::EnvironmentSource::PhysicalAtmosphere || !worldEnvironment.celestial[0].enabled) {
        return {};
    }
    std::array<double, 3> observer{};
    if (const auto* view = context.viewConstants()) {
        const auto& eye = view->current.eye;
        observer = {eye[0], eye[1], eye[2]};
    }
    const auto declared = context.subsystem<EnvironmentLightingSubsystem>()->declareEnvironmentRequest(context, builder,
        {.environment = worldEnvironment, .observerWorldMetres = observer, .viewId = context.viewKey(), .needCelestial = false}, log);
    if (!declared) { return makeError(declared.error()); }
    return {};
}

Result<> LumenSurfaceCachePass::execute(RenderGraphExecutionContext& context)
{
    const auto& scene = *context.runtimeScene();
    auto registry = ResourceRegistry::forDevice(*device_);
    if (!registry) { return makeError(registry.error()); }
    auto* frame = RenderFrameContext::from(context.commandBuffer());
    if (published_ && !frame) {
        spdlog::error("Persistent Surface Cache requires a tracked render frame");
        return makeError(Error::Unsupported);
    }
    if (auto result = prepareCaptureGeometry(scene); !result) { return result; }

    const auto& geometry = *captureGeometry_;
    const bool shadowed = shadowedLighting();
    const bool hardwarePunctual = hardwarePunctualShadows();
    const bool hardwareRect = hardwareRectShadows();
    const bool separate = maxStochasticDirectLightingCardsPerFrame_ != 0u;
    const bool exportIrradiance = exportDirectIrradiance();
    const auto worldEnvironment = resolveWorldEnvironment(&scene, context.world());
    const auto sun = buildCelestialLightRecords(worldEnvironment)[0];
    const auto* atmosphere = context.resources<EnvironmentResourceSet>();
    PublishedSurfaceCache next{.geometry = captureGeometry_, .lighting = {
        .direction = {-sun.direction[0], -sun.direction[1], -sun.direction[2], 0.0f},
        .illuminance = {sun.irradiance[0], sun.irradiance[1], sun.irradiance[2], 0.0f},
        .sunCastsShadow = shadowed && (sun.flags & kGPUCelestialLightCastsShadow) != 0u,
        .hardwarePunctualShadows = hardwarePunctual,
        .hardwareRectShadows = hardwareRect,
    }};
    if (const auto* world = context.world(); world && world->scene() == &scene) {
        next.nonSunLightingRevision = world->nonSunLightingRevision();
    }
    auto& lighting = next.lighting;
    if (atmosphere) {
        lighting.atmosphereRevisions = std::array{worldEnvironment.atmosphereRevision, worldEnvironment.weatherRevision};
    }
    const auto resolvedLighting = resolveSceneLighting(&scene, context.world());
    auto rectLights = buildRectLightRecords(resolvedLighting.rectLights);
    if (!rectLights) {
        spdlog::error("Surface Cache could not pack rectangular lights");
        return makeError(rectLights.error());
    }
    lighting.rectLights = std::move(*rectLights);
    if (!geometry.cardsToDraw.empty() && rectLightSamples_ != 0u && lighting.rectLights.size() > rectLightSamples_) {
        lighting.rectLightSamples = rectLightSamples_;
        lighting.rectGridCenterRadius = geometry.rectGridCenterRadius;
    }
    TextureHandle pageTable, coarseMip;
    BufferHandle domains;
    uint32_t clipmapCount = 0;
    if (shadowed) {
        pageTable = context.inputTexture("pageTable");
        coarseMip = context.inputTexture("coarseMip");
        domains = context.inputBuffer("domain");
        const auto& table = pageTable.desc();
        const auto& coarse = coarseMip.desc();
        clipmapCount = coarse.depth / coarse.width;
        if (clipmapCount < 1 || clipmapCount > 6 || coarse.width != coarse.height ||
            coarse.depth != coarse.width * clipmapCount || table.width != table.height ||
            table.depth != table.width * clipmapCount || domains.desc().size != clipmapCount * kGDFDomainStride) {
            spdlog::error("Surface Cache shadowed lighting requires matching quantized GDF clipmap resources");
            return makeError(Error::InvalidArgument);
        }
        std::vector<IESProfileAssignment> assignments;
        lighting.punctual = buildPunctualLightRecords(&scene, resolvedLighting, &assignments);
        auto profiles = publishIESProfiles(*device_, context.commandBuffer(), lighting.punctual, assignments);
        if (!profiles) { return makeError(profiles.error()); }
        lighting.iesProfiles = std::move(*profiles);
        if (clipmapCount > 1 && context.viewConstants()) {
            const auto& eye = context.viewConstants()->current.eye;
            lighting.clipmapEye = {eye[0], eye[1], eye[2]};
        }
    }

    SurfaceCacheUpdate update;
    if (published_) {
        // Scene dependency refreshes compile this pass again in place. A graph
        // resource rebuild replaces the atlas and requires another full capture.
        next.atlasAllocations[0] = context.outputTexture("albedo").texture()->memoryInfo().allocationId;
        if (separate) {
            next.atlasAllocations[1] = context.outputTexture("deterministicIrradiance").texture()->memoryInfo().allocationId;
            next.atlasAllocations[2] = context.outputTexture("stochasticDirectHistory").texture()->memoryInfo().allocationId;
        }
        const auto* previous = *published_ && (*published_)->atlasAllocations == next.atlasAllocations
            ? &**published_ : nullptr;
        next.materials = previous && previous->geometry->revision.sceneIdentity == geometry.revision.sceneIdentity &&
                previous->materials->revision == scene.materialRevision()
            ? previous->materials : captureMaterials(scene);
        update = planSurfaceCacheUpdate(previous, next, layout_.capacity(), maxDirectLightingCardsPerFrame_,
            maxStochasticDirectLightingCardsPerFrame_);
    }

    using Access = RenderGraphResourceAccess;
    const RenderGraphStageUse uploadUses[]{{"cards", Access::BufferTransferWrite}, {"scene", Access::BufferTransferWrite}};
    const RenderGraphStageUse updateUses[]{{"cardUpdates", Access::BufferTransferWrite},
        {"updateStatistics", Access::BufferTransferWrite}};
    const RenderGraphStageUse captureUses[]{{"albedo", Access::TextureColorWrite}, {"normal", Access::TextureColorWrite},
        {"emissive", Access::TextureColorWrite}, {"hardwareDepth", Access::TextureDepthStencilWrite}};
    std::vector<RenderGraphStage> stages;
    stages.reserve(7);

    std::unique_ptr<Buffer> cardUpload, sceneUpload, flagsUpload, statisticsUpload;
    if (update.uploadCards) {
        const auto identity = geometry.revision.sceneIdentity;
        const auto cardCount = uint32_t(geometry.cardsToDraw.size());
        const auto sceneOutput = context.outputBuffer("scene");
        std::vector<uint32_t> metadata(sceneOutput.desc().size / sizeof(uint32_t));
        metadata[0] = cardCount;
        metadata[1] = layout_.tile;
        metadata[2] = uint32_t(identity);
        metadata[3] = uint32_t(identity >> 32);
        if (objectCardRanges()) {
            std::vector<std::array<uint32_t, kLumenCardSceneRangeStride>> ranges;
            for (uint32_t first = 0; first < cardCount;) {
                const auto object = geometry.cards[first].metadata[0];
                uint32_t end = first + 1;
                while (end < cardCount && geometry.cards[end].metadata[0] == object) { ++end; }
                ranges.push_back({object, first, end});
                first = end;
            }
            // Preserve ascending atlas-slot order within an object, including ties.
            std::ranges::sort(ranges);
            metadata[kLumenCardSceneHeaderWords] = kLumenCardSceneRangeTag;
            metadata[kLumenCardSceneRangeCount] = uint32_t(ranges.size());
            for (size_t index = 0; index < ranges.size(); ++index) {
                std::ranges::copy(ranges[index], metadata.begin() + kLumenCardSceneRangeBase + index * kLumenCardSceneRangeStride);
            }
        }
        auto cardsBuffer = upload(*device_, std::as_bytes(std::span(geometry.cards)), BufferUsageBits::TransferSource);
        if (!cardsBuffer) { return makeError(cardsBuffer.error()); }
        cardUpload = std::move(*cardsBuffer);
        auto sceneBuffer = upload(*device_, std::as_bytes(std::span(metadata)), BufferUsageBits::TransferSource);
        if (!sceneBuffer) { return makeError(sceneBuffer.error()); }
        sceneUpload = std::move(*sceneBuffer);
        stages.push_back({"Upload card transforms", uploadUses, [&](CommandBuffer& cmd) -> Result<> {
            if (auto r = copyUpload(cmd, *cardUpload, *context.outputBuffer("cards").buffer()); !r) { return r; }
            return copyUpload(cmd, *sceneUpload, *context.outputBuffer("scene").buffer());
        }});
    }
    if (published_) {
        auto flags = upload(*device_, std::as_bytes(std::span(update.flags)), BufferUsageBits::TransferSource);
        if (!flags) { return makeError(flags.error()); }
        flagsUpload = std::move(*flags);
        const std::array statistics{uint32_t(update.fullCapture), uint32_t(update.captureTiles.size()),
            uint32_t(separate ? update.compositionTiles.size() : update.lightingTiles.size())};
        auto stats = upload(*device_, std::as_bytes(std::span(statistics)), BufferUsageBits::TransferSource);
        if (!stats) { return makeError(stats.error()); }
        statisticsUpload = std::move(*stats);
        stages.push_back({"Publish card updates", updateUses, [&](CommandBuffer& cmd) -> Result<> {
            if (auto r = copyUpload(cmd, *flagsUpload, *context.outputBuffer("cardUpdates").buffer()); !r) { return r; }
            return copyUpload(cmd, *statisticsUpload, *context.outputBuffer("updateStatistics").buffer());
        }});
    }

    const bool captures = update.fullCapture || !update.captureTiles.empty();
    const bool relights = update.fullCapture || !update.lightingTiles.empty();
    const bool composes = separate && (update.fullCapture || !update.compositionTiles.empty());
    auto* sceneTLAS = hardwarePunctual && (relights || (hardwareRect && composes)) ?
        context.inputAccelerationStructure("accelerationStructure") : nullptr;
    if (hardwarePunctual && (relights || (hardwareRect && composes)) && !sceneTLAS) {
        spdlog::error("Surface Cache hardware shadows require a published scene TLAS");
        return makeError(Error::InvalidArgument);
    }
    std::vector<CaptureDraw> draws;
    if (captures) {
        for (uint32_t card = 0; card < geometry.cardsToDraw.size(); ++card) {
            if (!update.fullCapture && !(update.flags[card] & kLumenCardCaptured)) { continue; }
            const auto& range = geometry.cardsToDraw[card];
            for (const auto& section : std::span(geometry.sections).subspan(range.firstSection, range.sectionCount)) {
                ParameterWriter writer(*device_, **registry, frame);
                CaptureParams params{};
                params.geometry = {writer.buffer(geometry.vertexBuffer.get()).index, section.firstVertex,
                    writer.samplerHandle(kMaterialSampler).index, 0};
                projectCard(params, geometry.cards[card]);
                if (next.materials) {
                    bindMaterial(writer, captureMaterialAt(*next.materials, section.materialIndex), params);
                } else {
                    const auto material = section.materialIndex >= 0 ?
                        captureMaterial(scene.materials()[section.materialIndex]) : defaultCaptureMaterial();
                    bindMaterial(writer, material, params);
                }
                auto encoded = writer.encode(params, kCaptureABI);
                if (!encoded) { return makeError(encoded.error()); }
                draws.push_back({layout_.rect(card), section.vertexCount, std::move(*encoded)});
            }
        }
        stages.push_back({"Capture card materials", captureUses, [&](CommandBuffer& cmd) {
            return rasterize(context, cmd, draws, update.captureTiles, update.fullCapture);
        }, RenderGraphPassKind::Raster});
    }

    auto rectSampling = *rectSampling_;
    bool publishRectSampling = false;
    uint32_t rectGridPhase = 0u;
    const auto executeAndPublish = [&]() -> Result<> {
        if (auto result = context.executeStages(stages); !result) { return result; }
        if (publishRectSampling) {
            auto transaction = std::make_shared<SubmissionTransaction>(
                [state = rectSampling_, sampling = rectSampling, rectGridPhase] {
                    if (sampling) { sampling->gridPhase = rectGridPhase; }
                    *state = sampling;
                }, [] {});
            if (auto result = context.commandBuffer().addSubmissionTransaction(std::move(transaction)); !result) {
                return result;
            }
        }
        if (!update.publish) { return {}; }
        return context.commandBuffer().addSubmissionTransaction(std::make_shared<SubmissionTransaction>(
            [state = published_, next = std::move(next)]() mutable { *state = std::move(next); }, [] {}));
    };
    if (!relights && !composes) { return executeAndPublish(); }

    std::unique_ptr<Buffer> exactRectLightBuffer;
    if (lighting.rectLightSamples != 0u) {
        if (separate) {
            rectGridPhase = rectSampling ? rectSampling->gridPhase : 0u;
            if (update.sampledStochastic) {
                const uint64_t sweepFrames = (geometry.cardsToDraw.size() - 1u) /
                    maxStochasticDirectLightingCardsPerFrame_ + 1u;
                rectGridPhase = uint32_t(context.frameIndex() / std::max<uint64_t>(sweepFrames, kMinStochasticGridRefreshFrames));
            }
        }
        const bool recreateSampling = !rectSampling || rectSampling->centerRadius != lighting.rectGridCenterRadius ||
            !sameRectLights(rectSampling->lights, lighting.rectLights);
        if (recreateSampling) {
            rectSampling = std::make_shared<RectLightSampling>();
            rectSampling->centerRadius = lighting.rectGridCenterRadius;
            rectSampling->lights = lighting.rectLights;
            auto records = upload(*device_, std::as_bytes(std::span(rectSampling->lights)), BufferUsageBits::Storage);
            if (!records) { return makeError(records.error()); }
            rectSampling->lightBuffer = std::move(*records);

            const uint32_t lightCount = uint32_t(lighting.rectLights.size());
            const auto pdfSize = computeImportancePdfTextureSize(lightCount);
            std::string log;
            const auto initializeSampling = [&]() -> Result<> {
                if (auto r = rectPdfCompute_.initialize(*device_, log); !r) { return r; }
                if (auto r = rectSampling->pdf.initialize(*device_, pdfSize.width, pdfSize.height,
                        "Surface Cache Rect Light PDF", log); !r) { return r; }
                if (auto r = rectSampling->grid.initialize(*device_, log); !r) { return r; }
                return rectSampling->grid.ensureGrid(*device_, kRectReGIRGridSize, kRectReGIRLightsPerCell, log);
            };
            if (auto result = initializeSampling(); !result) {
                spdlog::error("Surface Cache Rect ReGIR initialization failed: {}", log);
                return result;
            }
        }
        if (recreateSampling || rectSampling->gridPhase != rectGridPhase) {
            stages.push_back({"Build Rect ReGIR", {}, [&, recreateSampling](CommandBuffer& cmd) -> Result<> {
                auto& sampling = *rectSampling;
                const auto lightCount = uint32_t(sampling.lights.size());
                if (recreateSampling) {
                    if (auto r = rectPdfCompute_.buildRectLights(cmd, sampling.pdf, *sampling.lightBuffer, lightCount); !r) {
                        return r;
                    }
                }
                const auto& center = sampling.centerRadius;
                return sampling.grid.buildRectLights(cmd, *sampling.pdf.view(), *sampling.lightBuffer, {
                    .lightCount = lightCount, .frameIndex = rectGridPhase, .sceneCenter = {center[0], center[1], center[2]},
                    .sceneRadius = center[3], .samplingJitter = kRectReGIRJitter,
                });
            }});
            publishRectSampling = true;
        }
        if (auto result = context.commandBuffer().retainResource(rectSampling); !result) { return result; }
    } else {
        publishRectSampling = rectSampling != nullptr;
        rectSampling.reset();
    }
    if ((relights || (hardwareRect && composes)) && !rectSampling && !lighting.rectLights.empty()) {
        auto records = upload(*device_, std::as_bytes(std::span(lighting.rectLights)), BufferUsageBits::Storage);
        if (!records) { return makeError(records.error()); }
        exactRectLightBuffer = std::move(*records);
    }
    const auto bindRectLighting = [&](ParameterWriter& writer) -> RectLightingParams {
        if (rectSampling) {
            return {
                .lights = writer.buffer(rectSampling->lightBuffer.get()),
                .count = uint32_t(rectSampling->lights.size()),
                .grid = writer.buffer(rectSampling->grid.buffer()),
                .pdf = writer.sampledImageHandle(rectSampling->pdf.view()),
                .sampleCount = lighting.rectLightSamples,
            };
        }
        if (exactRectLightBuffer) {
            return {.lights = writer.buffer(exactRectLightBuffer.get()), .count = uint32_t(lighting.rectLights.size())};
        }
        return {};
    };
    std::optional<CardDispatch> lightingDispatch, depthDispatch, compositionDispatch;
    std::optional<EncodedParameters> resolveEncoded, depthEncoded, shadowedEncoded, compositionEncoded;
    if (relights) {
        auto dispatch = makeCardDispatch(*device_, layout_, update.lightingTiles, update.fullCapture);
        if (!dispatch) { return makeError(dispatch.error()); }
        lightingDispatch = std::move(*dispatch);
    }
    const auto bindLighting = [&](ParameterWriter& writer, auto& params) {
        params.albedo = writer.sampledImageHandle(context.outputTexture("albedo").view());
        params.normal = writer.sampledImageHandle(context.outputTexture("normal").view());
        params.emissive = writer.sampledImageHandle(context.outputTexture("emissive").view());
        params.lighting = writer.storageImageHandle(context.outputTexture("lighting").view());
        params.cards = writer.buffer(context.outputBuffer("cards").buffer());
        params.tileResolution = layout_.tile;
        params.lightDirection = lighting.direction;
        params.illuminance = lighting.illuminance;
        if (atmosphere) {
            const auto& resources = atmosphere->lighting;
            params.atmosphere = {writer.buffer(resources.atmosphereParametersBuffer),
                writer.sampledImageHandle(resources.transmittanceView), writer.sampledImageHandle(resources.cloudShadowView)};
        }
        params.updateCount = lightingDispatch->count;
        if (lightingDispatch->tiles) { params.updateTiles = writer.buffer(lightingDispatch->tiles.get()); }
        if (!separate || (!hardwareRect && !rectSampling)) { params.rectLighting = bindRectLighting(writer); }
        if (separate) {
            params.deterministicIrradiance = writer.storageImageHandle(context.outputTexture("deterministicIrradiance").view());
            params.separateLighting = 1u;
        }
    };

    const char* lightingOutput = separate ? "deterministicIrradiance" : "lighting";
    const RenderGraphStageUse resolveUses[]{{"albedo", Access::TextureSampleRead}, {"normal", Access::TextureSampleRead},
        {"emissive", Access::TextureSampleRead}, {"hardwareDepth", Access::TextureSampleRead},
        {"cards", Access::BufferStorageRead}, {"depth", Access::TextureStorageWrite}, {lightingOutput, Access::TextureStorageWrite}};
    if (relights && !shadowed) {
        ParameterWriter writer(*device_, **registry, frame);
        ResolveParams params{};
        bindLighting(writer, params);
        params.hardwareDepth = writer.sampledImageHandle(context.outputTexture("hardwareDepth").view());
        params.depth = writer.storageImageHandle(context.outputTexture("depth").view());
        auto encoded = writer.encode(params, kResolveABI);
        if (!encoded) { return makeError(encoded.error()); }
        resolveEncoded = std::move(*encoded);
        stages.push_back({"Encode depth and shade cards", resolveUses, [&](CommandBuffer& cmd) {
            return lightingDispatch->record(resolve_, cmd, *resolveEncoded, exactRectLightBuffer != nullptr);
        }});
    }

    const RenderGraphStageUse encodeDepthUses[]{{"normal", Access::TextureSampleRead},
        {"hardwareDepth", Access::TextureSampleRead}, {"depth", Access::TextureStorageWrite}};
    if (captures && shadowed) {
        auto dispatch = makeCardDispatch(*device_, layout_, update.captureTiles, update.fullCapture);
        if (!dispatch) { return makeError(dispatch.error()); }
        depthDispatch = std::move(*dispatch);
        ParameterWriter writer(*device_, **registry, frame);
        EncodeDepthParams params{writer.sampledImageHandle(context.outputTexture("normal").view()),
            writer.sampledImageHandle(context.outputTexture("hardwareDepth").view()),
            writer.storageImageHandle(context.outputTexture("depth").view()), layout_.edge, layout_.tile,
            {}, depthDispatch->count};
        if (depthDispatch->tiles) { params.updateTiles = writer.buffer(depthDispatch->tiles.get()); }
        auto encoded = writer.encode(params, kEncodeDepthABI);
        if (!encoded) { return makeError(encoded.error()); }
        depthEncoded = std::move(*encoded);
        stages.push_back({"Encode card depth", encodeDepthUses, [&](CommandBuffer& cmd) {
            return depthDispatch->record(encodeDepth_, cmd, *depthEncoded);
        }});
    }

    std::vector<RenderGraphStageUse> lightingUses{{"albedo", Access::TextureSampleRead}, {"normal", Access::TextureSampleRead},
        {"emissive", Access::TextureSampleRead}, {"depth", Access::TextureSampleRead}, {"cards", Access::BufferStorageRead},
        {"pageTable", Access::TextureSampleRead}, {"pageAtlas", Access::TextureSampleRead}, {"coarseMip", Access::TextureSampleRead},
        {"domain", Access::BufferStorageRead}, {lightingOutput, Access::TextureStorageWrite}};
    std::unique_ptr<Buffer> localLights;
    if (relights && shadowed) {
        auto records = upload(*device_, std::as_bytes(std::span(lighting.punctual)), BufferUsageBits::Storage);
        if (!records) { return makeError(records.error()); }
        localLights = std::move(*records);
        ParameterWriter writer(*device_, **registry, frame);
        ShadowedLightingParams params{};
        writer.retain(lighting.iesProfiles);
        bindLighting(writer, params);
        params.depth = writer.sampledImageHandle(context.outputTexture("depth").view());
        params.sunCastsShadow = lighting.sunCastsShadow;
        params.pageTable = writer.sampledImageHandle(pageTable.view());
        params.pageAtlas = writer.sampledImageHandle(context.inputTexture("pageAtlas").view());
        params.coarseMip = writer.sampledImageHandle(coarseMip.view());
        params.domains = writer.buffer(domains.buffer());
        params.localLights = writer.buffer(localLights.get());
        params.distanceSampler = writer.samplerHandle({});
        params.clipmapCount = clipmapCount;
        params.atlasSize = layout_.edge;
        if (!separate && exportIrradiance) {
            params.directIrradiance = writer.storageImageHandle(context.outputTexture("directIrradiance").view());
            params.exportDirectIrradiance = true;
            lightingUses.push_back({"directIrradiance", Access::TextureStorageWrite});
        }
        auto encodeLighting = [&]() -> Result<EncodedParameters> {
            if (!hardwarePunctual) { return writer.encode(params, kShadowedLightingABI); }
            lightingUses.push_back({"accelerationStructure", Access::AccelerationStructureShaderRead});
            const uint64_t accelerationStructure = writer.accelerationStructure(sceneTLAS).value;
            if (hardwareRect) {
                return writer.encode(HardwareRectLightingParams{params, accelerationStructure, uint32_t(context.frameIndex())},
                    kHardwareRectLightingABI);
            }
            return writer.encode(HardwarePunctualLightingParams{params, accelerationStructure}, kHardwarePunctualLightingABI);
        };
        auto encoded = encodeLighting();
        if (!encoded) { return makeError(encoded.error()); }
        shadowedEncoded = std::move(*encoded);
        stages.push_back({hardwarePunctual ? "Shade cards with hardware punctual shadows" : "Shade cards with GDF shadows",
            lightingUses, [&](CommandBuffer& cmd) {
            return lightingDispatch->record(shadowedLighting_, cmd, *shadowedEncoded, true);
        }});
    }

    std::vector<RenderGraphStageUse> composeUses{{"albedo", Access::TextureSampleRead}, {"normal", Access::TextureSampleRead},
        {"emissive", Access::TextureSampleRead}, {"depth", Access::TextureSampleRead},
        {"deterministicIrradiance", Access::TextureSampleRead}, {"stochasticDirectHistory", Access::TextureStorageReadWrite},
        {"cards", Access::BufferStorageRead}, {"cardUpdates", Access::BufferStorageRead},
        {"lighting", Access::TextureStorageWrite}};
    if (composes) {
        auto dispatch = makeCardDispatch(*device_, layout_, update.compositionTiles, update.fullCapture);
        if (!dispatch) { return makeError(dispatch.error()); }
        compositionDispatch = std::move(*dispatch);
        ParameterWriter writer(*device_, **registry, frame);
        ComposeLightingParams params{
            .albedo = writer.sampledImageHandle(context.outputTexture("albedo").view()),
            .normal = writer.sampledImageHandle(context.outputTexture("normal").view()),
            .emissive = writer.sampledImageHandle(context.outputTexture("emissive").view()),
            .depth = writer.sampledImageHandle(context.outputTexture("depth").view()),
            .deterministicIrradiance = writer.sampledImageHandle(context.outputTexture("deterministicIrradiance").view()),
            .stochasticDirectHistory = writer.storageImageHandle(context.outputTexture("stochasticDirectHistory").view()),
            .lighting = writer.storageImageHandle(context.outputTexture("lighting").view()),
            .cards = writer.buffer(context.outputBuffer("cards").buffer()),
            .cardUpdates = writer.buffer(context.outputBuffer("cardUpdates").buffer()),
            .atlasSize = layout_.edge,
            .tileResolution = layout_.tile,
            .updateCount = compositionDispatch->count,
            .samplePhase = uint32_t(context.frameIndex()),
            .exportDirectIrradiance = exportIrradiance,
        };
        if (rectSampling || hardwareRect) { params.rectLighting = bindRectLighting(writer); }
        if (compositionDispatch->tiles) { params.updateTiles = writer.buffer(compositionDispatch->tiles.get()); }
        if (exportIrradiance) {
            params.directIrradiance = writer.storageImageHandle(context.outputTexture("directIrradiance").view());
            composeUses.push_back({"directIrradiance", Access::TextureStorageWrite});
        }
        if (hardwareRect) { composeUses.push_back({"accelerationStructure", Access::AccelerationStructureShaderRead}); }
        auto encoded = hardwareRect ?
            writer.encode(HardwareRectComposeParams{params, writer.accelerationStructure(sceneTLAS).value}, kHardwareRectComposeABI) :
            writer.encode(params, kComposeLightingABI);
        if (!encoded) { return makeError(encoded.error()); }
        compositionEncoded = std::move(*encoded);
        stages.push_back({"Accumulate stochastic direct light and compose cards", composeUses, [&](CommandBuffer& cmd) {
            return compositionDispatch->record(composeLighting_, cmd, *compositionEncoded);
        }});
    }
    return executeAndPublish();
}

} // namespace

std::unique_ptr<RenderGraphPass> createLumenSurfaceCachePass()
{
    return std::make_unique<LumenSurfaceCachePass>();
}

} // namespace metallic::render::builtin_pass
