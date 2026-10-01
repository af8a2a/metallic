#include "RHITest.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"

#include <algorithm>
#include <initializer_list>
#include <string_view>

namespace metallic::tests {
namespace {

using render::RenderGraphFieldVisibility;
using render::RenderGraphInitialization;
using render::RenderGraphProperties;
using render::RenderGraphResourceLifetime;
using render::RenderGraphResourceType;

struct ExpectedOutput {
    std::string_view name;
    RenderGraphInitialization initialization;
    bool presentationOutput = false;
    RenderGraphResourceType resourceType = RenderGraphResourceType::Texture2D;
};

RHITestResult checkReflection(std::string_view passName, RenderGraphProperties properties,
    std::initializer_list<ExpectedOutput> expected, const render::RenderGraphCompileContext& context)
{
    auto pass = render::createRenderGraphPass(passName);
    if (!pass) { return RHITestResult::fail("Could not create " + std::string(passName)); }
    pass->setProperties(std::move(properties));
    const auto reflection = pass->reflect(context);
    for (const auto& output : expected) {
        const auto* field = reflection.findField(output.name, RenderGraphFieldVisibility::Output);
        const auto lifetime = output.initialization == RenderGraphInitialization::Unknown
            ? RenderGraphResourceLifetime::Persistent : RenderGraphResourceLifetime::Transient;
        if (!field || field->resourceType != output.resourceType || field->lifetime != lifetime ||
            field->initialization != output.initialization || field->presentationOutput != output.presentationOutput) {
            return RHITestResult::fail(std::string(passName) + "." + std::string(output.name) +
                " has an unexpected lifetime, initialization, resource type or presentation contract");
        }
    }
    for (const auto& field : reflection.fields()) {
        if (field.visibility == RenderGraphFieldVisibility::Output &&
            field.resourceType == RenderGraphResourceType::Texture2D &&
            std::none_of(expected.begin(), expected.end(), [&](const auto& output) { return output.name == field.name; })) {
            return RHITestResult::fail(std::string(passName) + "." + field.name + " is an unchecked texture output");
        }
        if (field.visibility == RenderGraphFieldVisibility::Input &&
            (field.lifetime != RenderGraphResourceLifetime::Persistent ||
                field.initialization != RenderGraphInitialization::Unknown)) {
            return RHITestResult::fail(std::string(passName) + "." + field.name +
                " input must preserve its producer's initialization contract");
        }
    }
    return RHITestResult::pass();
}

class RenderPassTransientInitializationContractsTest final : public RHITest {
public:
    RenderPassTransientInitializationContractsTest()
    {
        type = RHITestType::Resource;
        name = "render_pass_transient_initialization_contracts";
    }

    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.suite = "contract", .layer = bench::Layer::RenderGraph,
            .requirements = {.requiresDevice = false, .validation = bench::Validation::Off, .queues = {}},
            .coverage = {"graph.pass.transient_initialization"}};
    }

    RHITestResult run(RHITestContext&) override { return check(); }
    RHITestResult runCpu(bench::Evidence&) override { return check(); }

private:
    RHITestResult check()
    {
        constexpr auto clear = RenderGraphInitialization::Clear;
        constexpr auto overwrite = RenderGraphInitialization::FullOverwrite;
        constexpr auto unknown = RenderGraphInitialization::Unknown;
        const render::RenderGraphCompileContext context{.width = 512, .height = 320};
        const auto properties = RenderGraphProperties::object();
        const auto verify = [&](std::string_view passName, const RenderGraphProperties& passProperties,
            std::initializer_list<ExpectedOutput> outputs) {
            return checkReflection(passName, passProperties, outputs, context);
        };

        for (const char* passName : {"ClearColorPass", "TriangleRasterPass", "ImageSamplePass"}) {
            auto result = verify(passName, properties, {{"color", clear}});
            if (!result.passed) { return result; }
        }
        for (const char* passName : {"CopyColorPass", "AutoExposurePass", "SliderDebugPass", "LightGridDebugPass",
            "VisibilityBufferMaterialPass", "SceneMaterialVisualizationPass", "RTXDICompositePass", "RTXCRMaterialSamplePass"}) {
            auto result = verify(passName, properties, {{"color", overwrite}});
            if (!result.passed) { return result; }
        }
        for (const char* passName : {"BunnyWireframePass", "SceneMaterialShaderObjectPass"}) {
            auto result = verify(passName, properties, {{"color", clear}, {"depth", clear}});
            if (!result.passed) { return result; }
        }
        for (const char* passName : {"RayTracedShadowPass", "ScreenSpaceShadowPass"}) {
            auto result = verify(passName, properties, {{"shadow", overwrite}});
            if (!result.passed) { return result; }
        }
        auto result = verify("RTXDIConfidencePass", properties,
            {{"diffuseConfidence", overwrite}, {"specularConfidence", overwrite}});
        if (!result.passed) { return result; }

        // The disabled domain is an unwritten dummy. RTAS visualization similarly
        // leaves the raster attachments untouched, despite writing its color.
        for (const bool tessellation : {false, true}) {
            result = verify("VisibilityBufferPass", {{"tessellation", tessellation}},
                {{"color", clear}, {"visibility", clear}, {"depth", clear}, {"domain", tessellation ? clear : unknown}});
            if (!result.passed) { return result; }
        }
        for (const bool rtasVisualization : {false, true}) {
            result = verify("GPUDrivenStreamAssetPass", {{"rtasVisualization", rtasVisualization}},
                {{"color", rtasVisualization ? overwrite : clear}, {"visibility", rtasVisualization ? unknown : clear},
                    {"depth", rtasVisualization ? unknown : clear}});
            if (!result.passed) { return result; }
        }

        // Environment/scene readiness and successful no-write paths retain the
        // previous output. A shader's full dispatch alone cannot make them transient.
        for (const char* passName : {"SceneRayQueryVisualizationPass", "SceneRealtimeLightingPass"}) {
            result = verify(passName, properties, {{"color", unknown}});
            if (!result.passed) { return result; }
        }
        for (const bool exportGuides : {false, true}) {
            if (exportGuides) {
                result = verify("ScenePathTracePass", {{"exportDenoiserGuides", true}},
                    {{"color", unknown}, {"albedo", unknown}, {"specularAlbedo", unknown}, {"normalRoughness", unknown},
                        {"motionVectors", unknown}, {"linearDepth", unknown}, {"specularHitDistance", unknown}, {"depth", unknown}});
            } else {
                result = verify("ScenePathTracePass", properties, {{"color", unknown}});
            }
            if (!result.passed) { return result; }
            if (exportGuides) {
                result = verify("VisibilityBufferDeferredPass", {{"exportUpscalerGuides", true}},
                    {{"color", unknown}, {"motionVectors", unknown}, {"deviceDepth", unknown}});
            } else {
                result = verify("VisibilityBufferDeferredPass", properties, {{"color", unknown}});
            }
            if (!result.passed) { return result; }
        }
        result = verify("SceneRTXDIPass", properties,
            {{"color", unknown}, {"noisyDiffuse", unknown}, {"noisySpecular", unknown}, {"normalRoughness", unknown},
                {"motionVectors", unknown}, {"viewZ", unknown}, {"baseColorMetalness", unknown}, {"emissive", unknown}});
        if (!result.passed) { return result; }
        result = verify("NRDDenoisePass", properties,
            {{"denoisedDiffuse", unknown}, {"denoisedSpecular", unknown}, {"validation", unknown}});
        if (!result.passed) { return result; }
        result = verify("DLSSNRPass", properties, {{"color", unknown}});
        if (!result.passed) { return result; }

        for (const char* passName : {"StreamlineDLSSSRPass", "StreamlineDLSSRRPass"}) {
            for (const bool exportGuides : {false, true}) {
                if (exportGuides) {
                    result = verify(passName, {{"exportOutputGuides", true}},
                        {{"color", unknown}, {"motionVectors", overwrite}, {"depth", overwrite}});
                } else {
                    result = verify(passName, properties, {{"color", unknown}});
                }
                if (!result.passed) { return result; }
            }
        }
        for (const char* passName : {"RenderGraphBufferWritePass", "RenderGraphBufferCopyPass"}) {
            result = verify(passName, properties, {{"data", overwrite, false, RenderGraphResourceType::Buffer}});
            if (!result.passed) { return result; }
        }

        // Presentation remains a pinning/root contract even when its own shader
        // writes the entire image; HDR uses the same lifetime semantics.
        for (const auto mode : {render::DisplayOutputMode::SDR_sRGB, render::DisplayOutputMode::HDR_scRGB,
                render::DisplayOutputMode::HDR10_PQ}) {
            auto displayContext = context;
            displayContext.displayOutput.mode = mode;
            result = checkReflection("FinalBlitPass", properties, {{"color", overwrite, true}}, displayContext);
            if (!result.passed) { return result; }
        }
        return RHITestResult::pass();
    }
};

METALLIC_REGISTER_RHI_TEST(RenderPassTransientInitializationContractsTest);

} // namespace
} // namespace metallic::tests
