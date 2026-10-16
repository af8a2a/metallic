#include "Runtime/Render/Core/NativeShaderWarmupRequests.h"
#include "Runtime/Render/Core/BuiltinShaderRequests.h"
#include "Runtime/Render/HZBSPD.h"

#include <gtest/gtest.h>

namespace metallic::render {
namespace {

bool containsPolicy(const std::vector<NativeComputeWarmupVariant>& variants,
    std::span<const ShaderSpecializationConstant> constants)
{
    return std::any_of(variants.begin(), variants.end(), [&](const auto& variant) {
        return variant.userPushDataSize == sizeof(GPUBufferSpan) && !variant.indirectBindable &&
            std::ranges::equal(variant.specializationConstants, constants);
    });
}

bool graphicsEntry(const ShaderRequest& request)
{
    // Explicitly account for source-only entries; do not make new unowned
    // compute requests silently pass catalog coverage by naming convention.
    const std::vector<std::pair<std::string_view, std::string_view>> entries{
        {"Features/PostProcess/EditorDisplay", "editorDisplayVertex"},
        {"Features/PostProcess/EditorDisplay", "editorDisplayFragment"},
        {"Features/PostProcess/EditorOutput", "editorOutputVertex"},
        {"Features/PostProcess/EditorOutput", "editorOutputPQFragment"},
        {"Features/PostProcess/StreamlineDLSSSupport", "streamlineDlssDepthVertexMain"},
        {"Features/PostProcess/StreamlineDLSSSupport", "streamlineDlssDepthFragmentMain"},
        {"Features/Samples/BunnyWireframe", "bunnyWireframeVertexMain"},
        {"Features/Samples/BunnyWireframe", "bunnyWireframeFragmentMain"},
        {"Features/Samples/ImageSample", "imageSampleVertexMain"},
        {"Features/Samples/ImageSample", "imageSampleFragmentMain"},
        {"Features/Samples/MaterialShaderObject", "materialShaderObjectVertexMain"},
        {"Features/Samples/MaterialShaderObject", "materialShaderObjectFragmentMain"},
        {"Features/Samples/MaterialShaderObject", "materialShaderObjectAlternateFragmentMain"},
        {"Features/VisibilityBuffer/VisibilityBufferComposite", "visibilityBufferCompositeVertexMain"},
        {"Features/VisibilityBuffer/VisibilityBufferComposite", "visibilityBufferCompositeFragmentMain"},
        {"Features/VisibilityBuffer/VisibilityHybridRaster", "hybridResolveVertexMain"},
        {"Features/VisibilityBuffer/VisibilityHybridRaster", "hybridResolveFragmentMain"},
        {"Features/Samples/Triangle", "triangleVertexMain"},
        {"Features/Samples/Triangle", "triangleFragmentMain"},
        {"Features/GPUDriven/GPUDrivenStreamAsset", "gpuDrivenStreamAssetCompositeVertexMain"},
        {"Features/GPUDriven/GPUDrivenStreamAsset", "gpuDrivenStreamAssetCompositeFragmentMain"},
        {"Features/GPUDriven/GPUDrivenStreamAsset", "gpuDrivenStreamAssetFragmentMain"},
        {"Features/GPUDriven/GPUDrivenStreamAsset", "gpuDrivenStreamAssetMeshMain"},
        {"Features/GPUDriven/GPUDrivenStreamAsset", "streamTessellationMesh"},
        {"Features/GPUDriven/GPUDrivenStreamAsset", "streamTessellationFragment"},
        {"Features/GPUDriven/GPUDrivenStreamAsset", "streamTessellationTask"},
    };
    return std::find(entries.begin(), entries.end(),
        std::pair<std::string_view, std::string_view>{request.module, request.entry}) != entries.end();
}

TEST(NativeShaderWarmupRequests, EveryBuiltinComputeHasAnOwnedABIWithoutChangingSourceCatalog)
{
    for (const auto include : {"", "test-rtxcr-include"}) {
        const auto requests = builtinShaderWarmupRequests(include);
        const auto original = requests;
        size_t computeCount = 0, graphicsCount = 0;
        for (const auto& request : requests) {
            SCOPED_TRACE(request.module + "." + request.entry);
            const auto variants = nativeComputeWarmupVariants(request);
            if (graphicsEntry(request)) {
                EXPECT_TRUE(variants.empty());
                ++graphicsCount;
            } else {
                ASSERT_FALSE(variants.empty());
                ++computeCount;
                for (const auto& variant : variants) {
                    EXPECT_GT(variant.userPushDataSize, 0u);
                    EXPECT_EQ(variant.userPushDataSize % 4, 0u);
                    EXPECT_FALSE(variant.indirectBindable);
                    EXPECT_EQ(std::count(variants.begin(), variants.end(), variant), 1);
                }
            }
        }
        EXPECT_GT(computeCount, 80u);
        EXPECT_GT(graphicsCount, 20u);
        EXPECT_EQ(requests, original);
    }
}

TEST(NativeShaderWarmupRequests, InlineLayoutsMatchRuntimeTypesIncludingSPDPipelineCapacity)
{
    struct Case { const char* module; const char* entry; uint32_t bytes; };
    const Case cases[]{
        {"Features/Debug/LightGridDebug", "lightGridDebugMain", sizeof(LightGridDebugParams)},
        {"Features/Debug/SliderDebug", "sliderDebugOverlayMain", sizeof(SliderDebugParams)},
        {"Features/Environment/EnvironmentLightingPrecompute", "environmentLightingPrecomputeMain", sizeof(EnvironmentLightingPrecomputeParams)},
        {"Features/Environment/CloudShadow", "cloudShadowMain", sizeof(AtmospherePrecomputeParams)},
        {"Features/Lighting/BuildReGIR", "buildReGIRMain", sizeof(BuildReGIRParams)},
        {"Features/Lighting/ClusterLightGrid", "clusterLightGridMain", sizeof(ClusterLightGridBuildParams)},
        {"Features/Lighting/PrepareLightsPdf", "prepareLightsPdfMain", sizeof(PrepareLightsPdfParams)},
        {"Features/PostProcess/AutoExposure", "autoExposureReduceMain", sizeof(AutoExposureParams)},
        {"Features/PostProcess/FinalBlit", "finalBlitMain", sizeof(FinalBlitParams)},
        {"Features/PostProcess/ColorResize", "colorResizeMain", sizeof(ColorResizeParams)},
        {"Features/ReSTIR/RTXDIComposite", "rtxdiCompositeMain", sizeof(RTXDICompositeParams)},
        {"Features/ReSTIR/RTXDIConfidence", "rtxdiConfidenceMain", sizeof(RTXDIConfidenceParams)},
        {"Features/PathTracing/SceneSharcMaintenance", "sharcResolveMain", sizeof(SharcMaintenanceParams)},
        {"Features/SmokeTests/RenderGraphBuffer", "renderGraphBufferWriteMain", sizeof(builtin_pass::RenderGraphBufferUserPush)},
        {"Features/PostProcess/StreamlineDLSSSupport", "streamlineDlssAlphaMain", sizeof(StreamlineDLSSAlphaUserPush)},
        {"Features/GPUDriven/HZBSPD", "hzbSpdMain", sizeof(builtin_pass::GPUDrivenPreviewUserPush)},
        {"Features/GPUDriven/GPUDrivenStreamAsset", "gpuDrivenStreamAssetInitializePageTableMain", sizeof(MeshletStreamUserPush)},
        {"Features/GPUDriven/GPUDrivenStreamWorkRaster", "streamClusterRasterWorkControlMain", sizeof(MeshletStreamUserPush)},
        {"Features/VisibilityBuffer/VisibilityHybridRaster", "hybridRasterMain", VisibilityHybridRasterizer::pipelineUserPushDataSize()},
    };
    for (const auto& test : cases) {
        SCOPED_TRACE(test.entry);
        const auto variants = nativeComputeWarmupVariants({test.module, test.entry});
        ASSERT_EQ(variants.size(), 1u);
        EXPECT_EQ(variants[0].userPushDataSize, test.bytes);
        EXPECT_TRUE(variants[0].specializationConstants.empty());
    }
    EXPECT_NE(sizeof(HZBSPDUserPush), sizeof(builtin_pass::GPUDrivenPreviewUserPush));
}

TEST(NativeShaderWarmupRequests, AllRuntimeScenePoliciesSurviveSourceDeduplication)
{
    for (const bool rtxcr : {false, true}) {
        SceneShaderOptions options{.hasRTXCR = rtxcr, .rtxcrInclude = rtxcr ? "test-rtxcr-include" : ""};
        const SlangMacroDefine query{"SHARC_QUERY", "1"}, update{"SHARC_UPDATE", "1"}, single{"METALLIC_PT_SINGLE_SAMPLE", "1"};
        const std::vector<std::vector<SlangMacroDefine>> pathPolicies{{}, {query}, {update}};
        const auto check = [&](SceneShaderProgram program, const SceneShaderOptions& policy, std::span<const SlangMacroDefine> defines) {
            const auto request = makeSceneShaderRequest(program, policy, defines);
            SCOPED_TRACE(request.module + "." + request.entry);
            EXPECT_TRUE(containsPolicy(nativeComputeWarmupVariants(request), sceneShaderSpecializationConstants(program, policy, defines)));
        };
        for (const auto program : {SceneShaderProgram::PathTrace, SceneShaderProgram::PathTraceGuides,
                SceneShaderProgram::OpenPBRPathTrace, SceneShaderProgram::OpenPBRPathTraceGuides}) {
            for (const auto& defines : pathPolicies) { check(program, options, defines); }
        }
        const SlangMacroDefine querySingle[]{query, single};
        check(SceneShaderProgram::OpenPBRPathTraceGuides, options, {&single, 1});
        check(SceneShaderProgram::OpenPBRPathTraceGuides, options, querySingle);
        for (const bool half : {false, true}) {
            options.deferredFloat16 = half;
            check(SceneShaderProgram::Deferred, options, {});
            for (uint32_t materialClass = 0; materialClass < 5; ++materialClass) {
                const auto value = std::to_string(materialClass);
                const SlangMacroDefine material{"MATERIAL_CLASS", value.c_str()};
                check(SceneShaderProgram::DeferredBinned, options, {&material, 1});
            }
        }
    }
}

TEST(NativeShaderWarmupRequests, FoldedFamiliesHaveOnlyTheirClosedRuntimeTuples)
{
    if (sceneWorkingColorSpace() != SceneWorkingColorSpace::ACEScg) { GTEST_SKIP() << "Closed source families use ACEScg"; }
    const SceneShaderOptions options{.hasRTXCR = true, .rtxcrInclude = "test-rtxcr-include"};
    EXPECT_EQ(nativeComputeWarmupVariants(makeSceneShaderRequest(SceneShaderProgram::OpenPBRPathTrace, options)).size(), 4u);
    const SlangMacroDefine single{"METALLIC_PT_SINGLE_SAMPLE", "1"};
    EXPECT_EQ(nativeComputeWarmupVariants(makeSceneShaderRequest(SceneShaderProgram::OpenPBRPathTraceGuides, options, {&single, 1})).size(), 2u);
    // Std3's source contains three theoretical policies, but the current factory
    // keeps Plain OFF in its original static source. Only Plain QUERY and Guide
    // OFF share the specialized source; warming (false,false) there would build
    // an unused PSO and would not warm the actual Plain OFF request.
    const SlangMacroDefine query{"SHARC_QUERY", "1"};
    const auto plainOff = makeSceneShaderRequest(SceneShaderProgram::PathTrace, options);
    const auto plainQuery = makeSceneShaderRequest(SceneShaderProgram::PathTrace, options, {&query, 1});
    const auto guideOff = makeSceneShaderRequest(SceneShaderProgram::PathTraceGuides, options);
    EXPECT_NE(plainOff, plainQuery);
    EXPECT_EQ(plainQuery, guideOff);
    const auto plainVariants = nativeComputeWarmupVariants(plainOff);
    ASSERT_EQ(plainVariants.size(), 1u);
    EXPECT_TRUE(plainVariants[0].specializationConstants.empty());
    const auto sharedVariants = nativeComputeWarmupVariants(guideOff);
    ASSERT_EQ(sharedVariants.size(), 2u);
    EXPECT_TRUE(containsPolicy(sharedVariants, sceneShaderSpecializationConstants(SceneShaderProgram::PathTrace, options, {&query, 1})));
    EXPECT_TRUE(containsPolicy(sharedVariants, sceneShaderSpecializationConstants(SceneShaderProgram::PathTraceGuides, options)));
    EXPECT_FALSE(containsPolicy(sharedVariants, {}));
    EXPECT_EQ(nativeComputeWarmupVariants(makeSceneShaderRequest(SceneShaderProgram::Deferred, {})).size(), 2u);
    const SlangMacroDefine material{"MATERIAL_CLASS", "4"};
    EXPECT_EQ(nativeComputeWarmupVariants(makeSceneShaderRequest(SceneShaderProgram::DeferredBinned, {}, {&material, 1})).size(), 8u);
}

TEST(NativeShaderWarmupRequests, CustomMaterialSourcesUseTheirRuntimeParameterRoot)
{
    const SceneShaderOptions options{.customMaterials = true, .hasRTXCR = true,
        .materialInclude = "generated/material-content-key", .rtxcrInclude = "rtxcr"};
    for (const auto program : {SceneShaderProgram::OpenPBRPathTrace, SceneShaderProgram::DeferredBinned}) {
        const SlangMacroDefine custom[]{{"MATERIAL_CLASS", "4"}, {"MATERIAL_PROGRAM_ID", "17"}};
        const auto request = makeSceneShaderRequest(program, options, custom);
        const auto variants = nativeComputeWarmupVariants(request);
        ASSERT_EQ(variants.size(), 1u);
        EXPECT_TRUE(containsPolicy(variants, {}));
    }
}

TEST(NativeShaderWarmupRequests, UnknownEntriesAndMalformedSpecializedPoliciesAreNotGuessed)
{
    EXPECT_TRUE(nativeComputeWarmupVariants({"UserShader", "main"}).empty());
    EXPECT_TRUE(nativeComputeWarmupVariants({"Features/PostProcess/FinalBlit", "futureKernel"}).empty());
    const SceneShaderOptions options{.hasRTXCR = true, .rtxcrInclude = "rtxcr"};
    auto request = makeSceneShaderRequest(SceneShaderProgram::OpenPBRPathTrace, options);
    request.defines.emplace_back("METALLIC_PT_CACHE_QUERY_SPECIALIZATION", "1");
    EXPECT_TRUE(nativeComputeWarmupVariants(request).empty());
}

TEST(NativeShaderWarmupRequests, ExplicitSplitRequestsDoNotExpandOrdinaryCatalog)
{
    const SceneShaderOptions options{.hasRTXCR = true, .rtxcrInclude = "rtxcr"};
    const SlangMacroDefine single{"METALLIC_PT_SINGLE_SAMPLE", "1"};
    const auto catalog = builtinShaderWarmupRequests("rtxcr");
    for (const bool primary : {false, true}) {
        const auto request = makeOpenPBRPrimarySplitShaderRequest(options, {&single, 1}, primary);
        EXPECT_EQ(std::count(catalog.begin(), catalog.end(), request), 0);
        const auto variants = nativeComputeWarmupVariants(request);
        ASSERT_EQ(variants.size(), 1u);
        if (primary) { EXPECT_TRUE(variants[0].specializationConstants.empty()); }
        else { EXPECT_TRUE(containsPolicy(variants, sceneShaderSpecializationConstants(SceneShaderProgram::OpenPBRPathTraceGuides, options, {&single, 1}))); }
    }
}

TEST(NativeShaderWarmupRequests, NativeAcquisitionControlsPreserveTheRuntimeTupleSet)
{
    const SceneShaderOptions options{.hasRTXCR = true, .rtxcrInclude = "rtxcr"};
    auto request = makeSceneShaderRequest(SceneShaderProgram::OpenPBRPathTrace, options);
    const auto expected = nativeComputeWarmupVariants(request);
    request.optimizationLevel = SlangShaderOptimizationLevel::O2;
    request.descriptorHeapMode = SlangDescriptorHeapMode::Native;
    request.descriptorHeapStrides = {48, 16};
    EXPECT_EQ(nativeComputeWarmupVariants(request), expected);
}

} // namespace
} // namespace metallic::render
