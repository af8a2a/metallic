#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>

#ifndef METALLIC_HAS_RTXCR
#define METALLIC_HAS_RTXCR 0
#endif

#ifndef METALLIC_RTXCR_SHADER_INCLUDE_DIR
#define METALLIC_RTXCR_SHADER_INCLUDE_DIR ""
#endif

namespace metallic::render::builtin_pass {
namespace {

constexpr const char* kRtxcrMaterialSampleShaderModuleName = "RtxcrMaterialSample";
constexpr const char* kRtxcrMaterialSampleEntryPoint = "rtxcrMaterialSampleMain";

struct RtxcrMaterialSamplePush {
    uint32_t width = 1;
    uint32_t height = 1;
    uint32_t viewMode = 0;
    uint32_t padding0 = 0;

    float exposure = 1.0f;
    float lightAzimuthDegrees = -35.0f;
    float hairMelanin = 0.55f;
    float hairMelaninRedness = 0.35f;

    float hairLongitudinalRoughness = 0.28f;
    float hairAzimuthalRoughness = 0.35f;
    float hairCuticleAngleDegrees = 3.0f;
    float hairIor = 1.55f;

    float sssScale = 1.0f;
    float sssAnisotropy = 0.0f;
    float sssMaxSampleRadius = 0.12f;
    float padding1 = 0.0f;
};

static_assert(sizeof(RtxcrMaterialSamplePush) == 64);

float floatProperty(
    const RenderGraphProperties& properties,
    const char* key,
    float fallback,
    float minimum,
    float maximum)
{
    const auto iter = properties.find(key);
    if (iter == properties.end() || !iter->is_number()) {
        return fallback;
    }
    return std::clamp(iter->get<float>(), minimum, maximum);
}

uint32_t viewModeProperty(const RenderGraphProperties& properties)
{
    const std::string view = properties.value("view", "overview");
    if (view == "chiang") {
        return 1;
    }
    if (view == "far-field") {
        return 2;
    }
    if (view == "subsurface") {
        return 3;
    }
    return 0;
}

class RtxcrMaterialSamplePass final : public ComputePass {
public:
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addTextureOutput(
                "color",
                "RTXCR Chiang hair, far-field hair, and Burley subsurface showcase")
            .storageReadWrite()
            .format = Format::Rgba8Unorm;
        return reflection;
    }

    std::vector<RenderGraphRuntimeSetting> runtimeSettings() const override
    {
        return {
            runtimeEnumSetting(
                "view",
                "View",
                "overview",
                {
                    {"Overview", "overview"},
                    {"Chiang Hair", "chiang"},
                    {"Far-Field Hair", "far-field"},
                    {"Subsurface", "subsurface"},
                }),
            runtimeFloatSetting("exposure", "Exposure", 1.0f, 0.1f, 8.0f),
            runtimeFloatSetting("lightAzimuthDegrees", "Light Azimuth", -35.0f, -180.0f, 180.0f),
            runtimeFloatSetting("hairMelanin", "Hair Melanin", 0.55f, 0.0f, 1.0f),
            runtimeFloatSetting("hairMelaninRedness", "Hair Redness", 0.35f, 0.0f, 1.0f),
            runtimeFloatSetting(
                "hairLongitudinalRoughness",
                "Longitudinal Roughness",
                0.28f,
                0.02f,
                1.0f),
            runtimeFloatSetting(
                "hairAzimuthalRoughness",
                "Azimuthal Roughness",
                0.35f,
                0.02f,
                1.0f),
            runtimeFloatSetting("hairCuticleAngleDegrees", "Cuticle Angle", 3.0f, -10.0f, 10.0f),
            runtimeFloatSetting("hairIor", "Hair IOR", 1.55f, 1.01f, 2.5f),
            runtimeFloatSetting("sssScale", "SSS Scale", 1.0f, 0.05f, 5.0f),
            runtimeFloatSetting("sssAnisotropy", "SSS Anisotropy", 0.0f, -0.9f, 0.9f),
            runtimeFloatSetting("sssMaxSampleRadius", "SSS Sample Radius", 0.12f, 0.01f, 0.5f),
        };
    }

    Result compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (context.device == nullptr) {
            log = "RtxcrMaterialSamplePass requires a device";
            return makeError(Error::InvalidArgument);
        }
#if !METALLIC_HAS_RTXCR
        log =
            "RtxcrMaterialSamplePass requires the RTXCR Material Library. "
            "Configure CMake with -DRTXCR_ROOT=<RTXCR checkout or material library>.";
        return makeError(Error::Unsupported);
#else
        if (program_.valid()) {
            return {};
        }

        const char* additionalSearchPaths[] = {METALLIC_RTXCR_SHADER_INCLUDE_DIR};
        ShaderCompileResult compileResult;
        Result result = compileSlangShaderToSpirv(
            SlangShaderDesc{
                .moduleName = kRtxcrMaterialSampleShaderModuleName,
                .entryPointName = kRtxcrMaterialSampleEntryPoint,
                .searchPath = kTriangleShaderSearchPath,
                .additionalSearchPaths = additionalSearchPaths,
                .additionalSearchPathCount = static_cast<uint32_t>(std::size(additionalSearchPaths)),
            },
            compileResult);
        if (!result) {
            log = "compileSlangShaderToSpirv(RtxcrMaterialSample.rtxcrMaterialSampleMain) returned ";
            log += resultToString(result);
            if (!compileResult.diagnostics.empty()) {
                log += ": ";
                log += compileResult.diagnostics;
            }
            return result;
        }

        const SceneRayQueryBindingDesc bindings[] = {
            SceneRayQueryBindingDesc{
                .binding = 0,
                .kind = SceneRayQueryBindingKind::StorageImage,
            },
        };
        std::string programLog;
        result = program_.initialize(
            *context.device,
            SceneRayQueryProgramDesc{
                .spirv = compileResult.spirv.data(),
                .byteSize = static_cast<uint64_t>(compileResult.spirv.size() * sizeof(uint32_t)),
                .pushConstantSize = sizeof(RtxcrMaterialSamplePush),
                .bindings = bindings,
                .bindingCount = static_cast<uint32_t>(std::size(bindings)),
                .debugName = "RtxcrMaterialSamplePass",
            },
            programLog);
        if (!programLog.empty()) {
            log += programLog;
        }
        return result;
#endif
    }

    Result execute(RenderGraphExecutionContext& context) override
    {
        TextureHandle color = context.outputTexture("color");
        if (!color.valid() || color.view() == nullptr || !program_.valid()) {
            return makeError(Error::InvalidArgument);
        }

        const RenderGraphProperties& properties = context.properties();
        RtxcrMaterialSamplePush push;
        push.width = context.width();
        push.height = context.height();
        push.viewMode = viewModeProperty(properties);
        push.exposure = floatProperty(properties, "exposure", 1.0f, 0.1f, 8.0f);
        push.lightAzimuthDegrees = floatProperty(
            properties,
            "lightAzimuthDegrees",
            -35.0f,
            -180.0f,
            180.0f);
        push.hairMelanin = floatProperty(properties, "hairMelanin", 0.55f, 0.0f, 1.0f);
        push.hairMelaninRedness = floatProperty(
            properties,
            "hairMelaninRedness",
            0.35f,
            0.0f,
            1.0f);
        push.hairLongitudinalRoughness = floatProperty(
            properties,
            "hairLongitudinalRoughness",
            0.28f,
            0.02f,
            1.0f);
        push.hairAzimuthalRoughness = floatProperty(
            properties,
            "hairAzimuthalRoughness",
            0.35f,
            0.02f,
            1.0f);
        push.hairCuticleAngleDegrees = floatProperty(
            properties,
            "hairCuticleAngleDegrees",
            3.0f,
            -10.0f,
            10.0f);
        push.hairIor = floatProperty(properties, "hairIor", 1.55f, 1.01f, 2.5f);
        push.sssScale = floatProperty(properties, "sssScale", 1.0f, 0.05f, 5.0f);
        push.sssAnisotropy = floatProperty(
            properties,
            "sssAnisotropy",
            0.0f,
            -0.9f,
            0.9f);
        push.sssMaxSampleRadius = floatProperty(
            properties,
            "sssMaxSampleRadius",
            0.12f,
            0.01f,
            0.5f);

        const SceneRayQueryDispatchBinding bindings[] = {
            SceneRayQueryDispatchBinding{
                .binding = 0,
                .textureView = color.view(),
            },
        };
        return program_.dispatch(SceneRayQueryDispatchDesc{
            .commandBuffer = &context.commandBuffer(),
            .bindings = bindings,
            .bindingCount = static_cast<uint32_t>(std::size(bindings)),
            .pushData = &push,
            .pushDataSize = sizeof(push),
            .groupCountX = (context.width() + 7u) / 8u,
            .groupCountY = (context.height() + 7u) / 8u,
            .groupCountZ = 1,
        });
    }

private:
    SceneRayQueryProgram program_;
};

} // namespace

std::unique_ptr<RenderGraphPass> createRtxcrMaterialSamplePass()
{
    return std::make_unique<RtxcrMaterialSamplePass>();
}

} // namespace metallic::render::builtin_pass
