#include "Runtime/Material/MaterialAsset.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/StrandParameters.h"
#include "Runtime/Render/Material/MaterialRuntime.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Scene/StrandAsset.h"

namespace metallic::render::builtin_pass {
namespace {
constexpr uint32_t kMaxSegments = 4096, kMaxMaterials = 64;

float strandNumber(const RenderGraphProperties& p, const char* key, float fallback, float lo, float hi)
{
    const float v = p.value(key, fallback);
    return std::isfinite(v) ? std::clamp(v, lo, hi) : fallback;
}
uint32_t capacity(const RenderGraphProperties& p)
{
    return uint32_t(std::clamp(p.value("capacity", 4), 1, 8));
}
void store3(float* dest, float3 value)
{
    dest[0] = value.x;
    dest[1] = value.y;
    dest[2] = value.z;
}
StrandCamera strandCamera(const RenderGraphProperties& p, uint32_t width, uint32_t height)
{
    auto camera = p.value("camera", nlohmann::json::object());
    auto eye = camera.value("eye", std::array<float, 3>{0, 0, 2.4f});
    auto center = camera.value("center", std::array<float, 3>{0, 0, 0});
    auto up = camera.value("up", std::array<float, 3>{0, 1, 0});
    const auto unit = [](float3 v, float3 fallback) { return dot(v, v) > 1e-12f ? normalize(v) : fallback; };
    float3 e{eye[0], eye[1], eye[2]}, target{center[0], center[1], center[2]};
    float3 forward = unit(target - e, {0, 0, -1});
    float3 right = unit(cross(forward, float3{up[0], up[1], up[2]}), {1, 0, 0});
    StrandCamera result{};
    store3(result.eyeNear, e);
    store3(result.rightTan, right);
    store3(result.upAspect, unit(cross(right, forward), {0, 1, 0}));
    store3(result.forwardFar, forward);
    result.eyeNear[3] = strandNumber(camera, "znear", 0.01f, 0.0001f, 1000);
    result.forwardFar[3] = std::max(result.eyeNear[3] + 0.01f, strandNumber(camera, "zfar", 100, 0.02f, 100000));
    result.rightTan[3] = std::tan(strandNumber(camera, "fovDegrees", 45, 1, 170) * 0.00872664626f);
    if (camera.value("projection", std::string("perspective")) == "orthographic") {
        result.rightTan[3] = -0.5f * strandNumber(camera, "orthoHeight", 2, 0.0001f, 100000);
    }
    result.upAspect[3] = float(width) / float(height);
    return result;
}
bool upload(Buffer* buffer, const void* data, size_t size)
{
    void* mapped = buffer->map();
    if (!mapped) {
        return false;
    }
    std::memcpy(mapped, data, size);
    buffer->flush({0, size});
    buffer->unmap();
    return true;
}

class StrandPass final : public ComputePass
{
public:
    explicit StrandPass(bool lighting) : lighting_(lighting) {}
    RenderPassReflection reflect(const RenderGraphCompileContext& context) const override
    {
        RenderPassReflection r;
        if (!lighting_) {
            auto& segments = r.addBufferOutput("segments")
                                 .buffer(kMaxSegments * sizeof(StrandSegment), sizeof(StrandSegment))
                                 .shaderRead();
            segments.memoryLocation = MemoryLocation::HostUpload;
            auto& materials = r.addBufferOutput("materials")
                                  .buffer(kMaxMaterials * sizeof(LegacyMaterialPayload), sizeof(LegacyMaterialPayload))
                                  .shaderRead();
            materials.memoryLocation = MemoryLocation::HostUpload;
            auto& frame = r.addBufferOutput("frame").buffer(sizeof(StrandFrame), sizeof(StrandFrame)).shaderRead();
            frame.memoryLocation = MemoryLocation::HostUpload;
            r.addBufferOutput("records")
                .buffer(uint64_t(context.width) * context.height * capacity(properties()) *
                            sizeof(StrandVisibilityRecord),
                        sizeof(StrandVisibilityRecord))
                .storageWrite()
                .transient(RenderGraphInitialization::FullOverwrite);
            auto& counts =
                r.addTextureOutput("counts", "retained, overflow, candidate count, history valid").storageWrite();
            counts.format = Format::RGBA32Uint;
            r.addTextureInput("opaqueDepth", "Optional positive linear view-space Z, same camera/extent")
                .sampledRead()
                .setOptional();
        }
        else {
            for (const char* name : {"segments", "materials", "frame", "records"}) {
                r.addBufferInput(name).storageRead();
            }
            r.addTextureInput("counts").storageRead().format = Format::RGBA32Uint;
            r.addTextureInput("opaqueColor", "Scene-linear opaque background").sampledRead().setOptional();
            for (const char* name : {"color", "motion"}) {
                auto& output =
                    r.addTextureOutput(name).storageWrite().transient(RenderGraphInitialization::FullOverwrite);
                output.format = Format::RGBA32Sfloat;
                output.colorEncoding = DisplayColorEncoding::SceneLinear;
            }
            r.addTextureOutput("identity", "uint strand ID, segment ID, float bits: root-to-tip, coverage")
                .storageWrite()
                .format = Format::RGBA32Uint;
        }
        return r;
    }
    std::vector<RenderGraphRuntimeSetting> runtimeSettings() const override
    {
        if (lighting_) {
            return {runtimeEnumSetting("overflowPolicy", "Overflow Policy", "diagnostic",
                                       {{"Magenta + counts", "diagnostic"}, {"Nearest layers + counts", "nearest"}}),
                    runtimeBoolSetting("shadows", "Strand Shadows", true, true),
                    runtimeFloatSetting("lightIntensity", "Light Intensity", 3, 0, 30)};
        }
        auto cap = runtimeIntSetting("capacity", "Layers per Pixel", 4, 1, 8, true);
        cap.rebuildGraph = true;
        std::vector<RenderGraphRuntimeSetting> settings{
            cap, runtimeFloatSetting("phase", "Groom Phase", 0, -6.2832f, 6.2832f),
            runtimeFloatSetting("density", "Strand LOD Density", 1, 0, 1),
            runtimeFloatSetting("amplitude", "Motion Amplitude", 0.15f, 0, 1)};
        appendCameraRuntimeSettings(settings, {0, 0, 2.4f}, {0, 0, 0}, 45, true);
        return settings;
    }
    Result<> prepare(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (!lighting_) {
            if (uint64_t(context.width) * context.height * capacity(properties()) * sizeof(StrandVisibilityRecord) >
                512ull * 1024 * 1024) {
                log = "Strand record allocation exceeds 512 MiB; reduce capacity or render extent";
                return makeError(Error::InvalidArgument);
            }
            scene::StrandAsset asset;
            auto path = std::filesystem::path(
                properties().value("path", std::string(PROJECT_SOURCE_DIR "/Asset/Strands/NativeGroom.strands.json")));
            if (path.is_relative()) {
                path = std::filesystem::path(PROJECT_SOURCE_DIR) / path;
            }
            if (!scene::loadStrandAsset(path, asset, log)) {
                return makeError(Error::InvalidArgument);
            }
            std::vector<LegacyMaterialPayload> payloads;
            std::vector<scene::RenderMaterial> authored;
            material::MaterialAssetLibrary library(asset.materialRoot);
            for (const auto& uri : asset.materials) {
                material::ResolvedMaterialInstance instance;
                scene::RenderMaterial semantic;
                if (!library.resolve(uri, instance, log) ||
                    !material::lowerMaterialInstance(instance, {}, semantic, log) || !semantic.rtxcrHair) {
                    log += " Strand geometry requires a Fiber material asset";
                    return makeError(Error::InvalidArgument);
                }
                LegacyMaterialPayload p;
                p.textureParams[2] = float(MaterialProgramId::RTXCRChiang);
                p.rtxcrHairBaseColor[3] = 1;
                p.rtxcrHairParams0[0] = semantic.rtxcrHairLongitudinalRoughness;
                p.rtxcrHairParams0[1] = semantic.rtxcrHairAzimuthalRoughness;
                p.rtxcrHairParams0[2] = semantic.rtxcrHairIor;
                p.rtxcrHairParams0[3] = semantic.rtxcrHairCuticleAngleDegrees;
                p.rtxcrHairParams1[0] = semantic.rtxcrHairMelanin;
                p.rtxcrHairParams1[1] = semantic.rtxcrHairMelaninRedness;
                payloads.push_back(p);
                authored.push_back(semantic);
            }
            auto generation = MaterialGeneration::create(payloads, 0, log, authored);
            if (!generation || !generation->supports(MaterialEvaluationTarget::StrandVisibility, log)) {
                return makeError(Error::InvalidArgument);
            }
            std::vector<StrandSegment> segments;
            for (const auto& strand : asset.strands) {
                std::vector<float> lengths(strand.points.size(), 0);
                for (size_t i = 1; i < strand.points.size(); ++i) {
                    float distance = 0;
                    for (size_t c = 0; c < 3; ++c) {
                        const float d = strand.points[i].position[c] - strand.points[i - 1].position[c];
                        distance += d * d;
                    }
                    lengths[i] = lengths[i - 1] + std::sqrt(distance);
                }
                for (uint32_t i = 0; i + 1 < strand.points.size(); ++i) {
                    StrandSegment s;
                    for (size_t c = 0; c < 3; ++c) {
                        s.a[c] = strand.points[i].position[c];
                        s.b[c] = strand.points[i + 1].position[c];
                        s.previousA[c] = strand.points[i].previousPosition[c];
                        s.previousB[c] = strand.points[i + 1].previousPosition[c];
                        s.normalU[c] = strand.normal[c];
                    }
                    s.a[3] = strand.points[i].radius;
                    s.b[3] = strand.points[i + 1].radius;
                    s.normalU[3] = lengths[i] / lengths.back();
                    s.shape[0] = lengths[i + 1] / lengths.back();
                    s.shape[1] = strand.opacity;
                    uint32_t hash = strand.id * 747796405u + 2891336453u;
                    hash = ((hash >> ((hash >> 28) + 4)) ^ hash) * 277803737u;
                    hash = (hash >> 22) ^ hash;
                    s.shape[2] = float(hash & 0xffffffu) / float(0x1000000u) * 0.96875f;
                    s.identity[0] = strand.id;
                    s.identity[1] = i;
                    s.identity[2] = strand.material;
                    segments.push_back(s);
                }
            }
            if (segments.size() != segments_.size() ||
                std::memcmp(segments.data(), segments_.data(), segments.size() * sizeof(StrandSegment)) != 0) {
                history_->valid = false;
            }
            segments_ = std::move(segments);
            generation_ = std::move(generation);
        }
        return {};
    }
    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        device_ = context.device;
        if (!device_) {
            return makeError(Error::InvalidArgument);
        }
        return ShaderRegistry::instance().getComputeKernel(
            *device_,
            {.moduleName = "Features/Strands/NativeStrands",
             .entryPointName = lighting_ ? "strandLightingMain" : "strandVisibilityMain",
             .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
            {.parameters = parameterAbi<StrandParameters>(kStrandABI),
             .debugName = lighting_ ? "StrandLighting" : "StrandVisibility"},
            program_, log);
    }
    Result<> execute(RenderGraphExecutionContext& context) override
    {
        const auto buffer = [&](const char* name) {
            return lighting_ ? context.inputBuffer(name) : context.outputBuffer(name);
        };
        auto segments = buffer("segments"), materials = buffer("materials"), frame = buffer("frame"),
             records = buffer("records");
        if (!segments.valid() || !materials.valid() || !frame.valid() || !records.valid()) {
            return makeError(Error::InvalidArgument);
        }
        if (!lighting_) {
            if (!generation_ || segments_.empty()) {
                return makeError(Error::InvalidArgument);
            }
            StrandFrame f{};
            f.camera = strandCamera(context.properties(), context.width(), context.height());
            f.phase = strandNumber(context.properties(), "phase", 0, -10000, 10000);
            const bool valid = history_->valid && history_->width == context.width() &&
                               history_->height == context.height() &&
                               (context.viewConstants() == nullptr || context.viewConstants()->frame[1] != 0);
            f.previousPhase = valid ? history_->phase : f.phase;
            f.previousCamera = valid ? history_->camera : f.camera;
            f.width = context.width();
            f.height = context.height();
            f.capacity = capacity(context.properties());
            f.segmentCount = uint32_t(segments_.size());
            f.density = strandNumber(context.properties(), "density", 1, 0, 1);
            f.amplitude = strandNumber(context.properties(), "amplitude", 0.15f, 0, 1);
            f.previousAmplitude = valid ? history_->amplitude : f.amplitude;
            f.historyValid = valid ? 1u : 0u;
            if (records.desc().size != uint64_t(f.width) * f.height * f.capacity * sizeof(StrandVisibilityRecord) ||
                !upload(segments.buffer(), segments_.data(), segments_.size() * sizeof(StrandSegment)) ||
                !upload(materials.buffer(), generation_->parameters().data(), generation_->parameters().size_bytes()) ||
                !upload(frame.buffer(), &f, sizeof(f))) {
                return makeError(Error::InvalidArgument);
            }
            auto transaction = context.commandBuffer().addSubmissionTransaction(
                std::make_shared<SubmissionTransaction>([] {}, [history = history_] { history->valid = false; }));
            if (!transaction) {
                return transaction;
            }
            history_->camera = f.camera;
            history_->phase = f.phase;
            history_->amplitude = f.amplitude;
            history_->width = f.width;
            history_->height = f.height;
            history_->valid = true;
        }
        auto registry = ResourceRegistry::forDevice(*device_);
        if (!registry) {
            return makeError(registry.error());
        }
        auto& commands = context.commandBuffer();
        ParameterWriter writer(*device_, **registry, RenderFrameContext::from(commands));
        StrandParameters p{};
        p.segments = writer.bufferSpan<StrandSegment>(segments.buffer());
        p.materials = writer.bufferSpan<LegacyMaterialPayload>(materials.buffer());
        p.frame = writer.bufferSpan<StrandFrame>(frame.buffer());
        p.records = writer.bufferSpan<StrandVisibilityRecord>(records.buffer());
        p.counts = writer.storageImageHandle(
            (lighting_ ? context.inputTexture("counts") : context.outputTexture("counts")).view());
        if (lighting_) {
            p.color = writer.storageImageHandle(context.outputTexture("color").view());
            p.motion = writer.storageImageHandle(context.outputTexture("motion").view());
            p.identity = writer.storageImageHandle(context.outputTexture("identity").view());
            auto opaque = context.inputTexture("opaqueColor");
            if (opaque.valid()) {
                p.opaqueColor = writer.sampledImageHandle(opaque.view());
                p.hasOpaqueColor = 1;
            }
            p.overflowPolicy =
                context.properties().value("overflowPolicy", std::string("diagnostic")) == "nearest" ? 1u : 0u;
            p.shadows = context.properties().value("shadows", true) ? 1u : 0u;
            p.light[0] = 0.4f;
            p.light[1] = 0.5f;
            p.light[2] = 1;
            p.light[3] = strandNumber(context.properties(), "lightIntensity", 3, 0, 30);
            p.background[0] = p.background[1] = p.background[2] = 0.025f;
        }
        else {
            auto depth = context.inputTexture("opaqueDepth");
            if (depth.valid()) {
                p.opaqueDepth = writer.sampledImageHandle(depth.view());
                p.hasOpaqueDepth = 1;
            }
        }
        auto encoded = writer.encode(p, kStrandABI);
        if (!encoded) {
            return makeError(encoded.error());
        }
        return program_.dispatch(commands, *encoded, (context.width() + 7) / 8, (context.height() + 7) / 8);
    }

private:
    struct History
    {
        StrandCamera camera{};
        float phase = 0, amplitude = 0;
        uint32_t width = 0, height = 0;
        bool valid = false;
    };
    bool lighting_;
    std::shared_ptr<History> history_ = std::make_shared<History>();
    Device* device_ = nullptr;
    ComputeKernel program_;
    std::vector<StrandSegment> segments_;
    std::shared_ptr<const MaterialGeneration> generation_;
};
} // namespace
std::unique_ptr<RenderGraphPass> createStrandVisibilityPass()
{
    return std::make_unique<StrandPass>(false);
}
std::unique_ptr<RenderGraphPass> createStrandLightingPass()
{
    return std::make_unique<StrandPass>(true);
}
} // namespace metallic::render::builtin_pass
