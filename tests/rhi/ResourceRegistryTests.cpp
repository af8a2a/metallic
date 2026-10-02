#include "Runtime/Render/Core/DeferredShadingParameters.h"
#include "Runtime/Render/Core/RealtimeLightingParameters.h"
#include "Runtime/Render/Core/PathTraceGuidesInlineParameters.h"
#include "Runtime/Render/Core/NRCTraceParameters.h"
#include "Runtime/Render/Core/SharcTraceParameters.h"
#include "Runtime/Render/Core/PathTraceInlineParameters.h"
#include "Runtime/Render/Core/RTXDITraceParameters.h"
#include "Runtime/Render/Core/ShadowTraceParameters.h"
#include "Runtime/Render/Core/StreamWorkloadParameters.h"
#include "Runtime/Render/Core/StreamRasterParameters.h"
#include "Runtime/Render/Core/HybridResolveParameters.h"
#include "Runtime/Render/Core/HybridRasterParameters.h"
#include "Runtime/Render/Core/HybridBinParameters.h"
#include "Runtime/Render/Core/StreamClusterCullParameters.h"
#include "Runtime/Render/Core/StreamClassifyParameters.h"
#include "Runtime/Render/Core/StreamCandidateParameters.h"
#include "Runtime/Render/Core/StreamBLASParameters.h"
#include "Runtime/Render/Core/StreamTLASParameters.h"
#include "Runtime/Render/Streamer/MeshletStreamRuntime.h"
#include "Runtime/Render/Core/StreamActiveBuildParameters.h"
#include "Runtime/Render/Core/StreamTraversalParameters.h"
#include "Runtime/Render/Core/StreamPageTableParameters.h"
#include "Runtime/Render/Core/ResidentLODParameters.h"
#include "Runtime/Render/Core/StreamInstanceCullParameters.h"
#include "Runtime/Render/Core/InstanceCullParameters.h"
#include "Runtime/Render/Core/MaterialVisualizationParameters.h"
#include "Runtime/Render/Core/StreamDeferredParameters.h"
#include "Runtime/Render/Core/StreamCompositeParameters.h"
#include "Runtime/Render/Core/MaterialRasterParameters.h"
#include "Runtime/Render/Core/BunnyWireframeParameters.h"
#include "Runtime/Render/Core/ImageSampleParameters.h"
#include "Runtime/Render/Core/RenderGraphBufferParameters.h"
#include "Runtime/Render/Core/StreamSceneParameters.h"
#include "Runtime/Render/Core/DebugProbeParameters.h"
#include "Runtime/Render/Core/MaterialErrorParameters.h"
#include "Runtime/Render/Core/VisibilityMaterialParameters.h"
#include "Runtime/Render/Core/MaterialSampleParameters.h"
#include "Runtime/Render/Core/DebugVisualizationParameters.h"
#include "RHITest.h"
#include "harness/Fixtures.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/PostProcessParameters.h"
#include "Runtime/Render/Core/LightingKernelParameters.h"
#include "Runtime/Render/Core/RTXDIPostProcessParameters.h"
#include "Runtime/Render/Core/PathTraceStageParameters.h"
#include "Runtime/Render/Core/PathTraceParameters.h"
#include "Runtime/Render/Core/ComputeProgram.h"
#include "Runtime/Render/Core/SlangCompiler.h"

#include <array>
#include <cstring>
#include <thread>
#include <map>

namespace metallic::tests {
namespace {

#define REG_REQUIRE(expression) do { \
    const render::Result<> result = (expression); \
    if (!result) { return RHITestResult::fail(std::string(#expression) + ": " + toString(result)); } \
} while (false)
#define REG_CHECK(expression) do { \
    if (!(expression)) { return RHITestResult::fail(#expression); } \
} while (false)

constexpr uint64_t kABI = 0x5245475000000001ull;
struct ProbeParams {
    render::ShaderBuffer source, output;
    uint32_t add, index;
};
static_assert(sizeof(ProbeParams) == 24 && offsetof(ProbeParams, add) == 16);

render::Result<> makeBuffer(render::Device& device, std::unique_ptr<render::Buffer>& buffer, uint32_t value = 0)
{
    auto result = device.createBuffer({.size = 64, .structureStride = 4,
        .usage = render::BufferUsageBits::Storage | render::BufferUsageBits::Indirect,
        .memoryLocation = render::MemoryLocation::HostReadback,
        .queueAccess = render::QueueAccessBits::Graphics | render::QueueAccessBits::Compute}).transform([&](auto rhiValue) { buffer = std::move(rhiValue); });
    if (!result) { return result; }
    void* mapped = buffer->map();
    if (!mapped) { return render::makeError(render::Error::Failure); }
    std::memset(mapped, 0, 64);
    std::memcpy(mapped, &value, 4);
    buffer->flush();
    buffer->unmap();
    return {};
}

struct Commands {
    render::RenderFrameContext frame;
    std::unique_ptr<render::CommandPool> pool;
    std::unique_ptr<render::CommandBuffer> commands;
    ~Commands()
    {
        if (frame.completion().isSubmitted()) { (void)frame.wait(); }
        if (pool) { (void)pool->reset(); }
        (void)frame.reset();
    }
    render::Result<> initialize(render::Device& device, render::Queue& queue)
    {
        auto result = device.createCommandPool(queue).transform([&](auto rhiValue) { pool = std::move(rhiValue); });
        return result ? pool->createCommandBuffer().transform([&](auto rhiValue) { commands = std::move(rhiValue); }) : result;
    }
    render::Result<> begin(uint64_t index)
    {
        auto result = frame.begin(index);
        if (result) { result = pool->reset(); }
        return result ? commands->begin(&frame) : result;
    }
    render::Result<> submit(render::QueueSubmissionTracker& tracker, render::Semaphore& gate)
    {
        auto result = commands->end();
        if (!result) { return result; }
        render::CommandBuffer* buffers[] = {commands.get()};
        render::SemaphoreSubmitDesc wait{.semaphore = &gate, .value = 1};
        return tracker.submit({
            .waitSemaphores = {&wait, 1},
            .commandBuffers = {buffers, 1},
        }, frame);
    }
};

struct Drain {
    render::Queue& queue;
    render::Semaphore& gate;
    ~Drain()
    {
        if (gate.currentValue() < 1) { (void)gate.signal(1); }
        (void)queue.waitIdle();
    }
};

render::Result<> makeKernel(render::Device& device, render::ComputeKernel& kernel, std::string& log,
    render::SlangDescriptorHeapMode mode = render::SlangDescriptorHeapMode::Default,
    render::ParameterTransport transport = render::ParameterTransport::DeviceAddress)
{
    const render::SlangMacroDefine defines[] = {{"INLINE_PARAMETERS", transport == render::ParameterTransport::InlinePush ? "1" : "0"}};
    render::ShaderCompileResult shader;
    auto result = render::compileSlangShaderToSpirv({.moduleName = "RegistryProbe",
        .entryPointName = "registryProbeMain", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders",
        .macroDefines = defines, .descriptorHeapMode = mode}, shader.diagnostics).transform([&](auto value) { shader = std::move(value); });
    if (!result) { log = shader.diagnostics; return result; }
    return kernel.initialize(device, {.spirv = shader.spirv, .parameters = render::parameterAbi<ProbeParams>(kABI, transport)}, log);
}

// Inspect emitted layout, including fields unused by a particular entry point.
// Sharing declarations alone cannot detect a CPU/Slang packing disagreement.
class PostProcessParameterLayoutTest : public RHITest {
public:
    PostProcessParameterLayoutTest() { type = RHITestType::Resource; name = "post_process_parameter_spirv_layout"; }
    uint32_t category = 0;
    RHITestResult run(RHITestContext&) override
    {
        using namespace render;
#define FIELD(Type, Member) uint32_t(offsetof(Type, Member))
        struct Layout {
            const char* name;
            std::vector<uint32_t> offsets;
        };
        const Layout layouts[] = {
            {"Metallic.FinalBlitParams", {FIELD(FinalBlitParams, output), FIELD(FinalBlitParams, source),
                FIELD(FinalBlitParams, lut), FIELD(FinalBlitParams, lutSampler), FIELD(FinalBlitParams, display), FIELD(FinalBlitParams, padding)}},
            {"Metallic.SliderDebugParams", {FIELD(SliderDebugParams, sourceA), FIELD(SliderDebugParams, sourceB),
                FIELD(SliderDebugParams, output), FIELD(SliderDebugParams, display), FIELD(SliderDebugParams, padding)}},
            {"Metallic.AutoExposureParams", {FIELD(AutoExposureParams, source), FIELD(AutoExposureParams, output),
                FIELD(AutoExposureParams, histogram), FIELD(AutoExposureParams, history), FIELD(AutoExposureParams, exposure),
                FIELD(AutoExposureParams, display), FIELD(AutoExposureParams, padding)}},
            {"Metallic.ColorGradingLUTParams", {FIELD(ColorGradingLUTParams, output), FIELD(ColorGradingLUTParams, custom0),
                FIELD(ColorGradingLUTParams, custom1), FIELD(ColorGradingLUTParams, custom2), FIELD(ColorGradingLUTParams, custom3),
                FIELD(ColorGradingLUTParams, reach), FIELD(ColorGradingLUTParams, gamut), FIELD(ColorGradingLUTParams, gammaTable),
                FIELD(ColorGradingLUTParams, sampler), FIELD(ColorGradingLUTParams, padding0), FIELD(ColorGradingLUTParams, padding1),
                FIELD(ColorGradingLUTParams, display)}},
            {"Metallic.ClusterLightGridBuildParams", {FIELD(ClusterLightGridBuildParams, grid), FIELD(ClusterLightGridBuildParams, lights),
                FIELD(ClusterLightGridBuildParams, candidates), FIELD(ClusterLightGridBuildParams, cells), FIELD(ClusterLightGridBuildParams, indices)}},
            {"Metallic.LightGridDebugParams", {FIELD(LightGridDebugParams, grid), FIELD(LightGridDebugParams, cells),
                FIELD(LightGridDebugParams, output), FIELD(LightGridDebugParams, settings)}},
            {"Metallic.PrepareLightsPdfParams", {FIELD(PrepareLightsPdfParams, environment), FIELD(PrepareLightsPdfParams, sourceMip),
                FIELD(PrepareLightsPdfParams, destinationMip), FIELD(PrepareLightsPdfParams, lights), FIELD(PrepareLightsPdfParams, settings)}},
            {"Metallic.BuildReGIRParams", {FIELD(BuildReGIRParams, localLightPdf), FIELD(BuildReGIRParams, output),
                FIELD(BuildReGIRParams, lights), FIELD(BuildReGIRParams, padding0), FIELD(BuildReGIRParams, padding1), FIELD(BuildReGIRParams, settings)}},
            {"Metallic.EnvironmentLightingPrecomputeParams", {FIELD(EnvironmentLightingPrecomputeParams, radiance),
                FIELD(EnvironmentLightingPrecomputeParams, partials), FIELD(EnvironmentLightingPrecomputeParams, coefficients),
                FIELD(EnvironmentLightingPrecomputeParams, specular), FIELD(EnvironmentLightingPrecomputeParams, settings)}},
            {"Metallic.RTXDIConfidenceParams", {FIELD(RTXDIConfidenceParams, noisyDiffuse), FIELD(RTXDIConfidenceParams, noisySpecular),
                FIELD(RTXDIConfidenceParams, baseColorMetalness), FIELD(RTXDIConfidenceParams, motionVectors),
                FIELD(RTXDIConfidenceParams, previousLuminance), FIELD(RTXDIConfidenceParams, currentLuminance),
                FIELD(RTXDIConfidenceParams, gradientA), FIELD(RTXDIConfidenceParams, gradientB),
                FIELD(RTXDIConfidenceParams, previousDiffuseConfidence), FIELD(RTXDIConfidenceParams, previousSpecularConfidence),
                FIELD(RTXDIConfidenceParams, diffuseConfidence), FIELD(RTXDIConfidenceParams, specularConfidence),
                FIELD(RTXDIConfidenceParams, currentDiffuseConfidence), FIELD(RTXDIConfidenceParams, currentSpecularConfidence),
                FIELD(RTXDIConfidenceParams, settings)}},
            {"Metallic.RTXDICompositeParams", {FIELD(RTXDICompositeParams, denoisedDiffuse), FIELD(RTXDICompositeParams, denoisedSpecular),
                FIELD(RTXDICompositeParams, baseColorMetalness), FIELD(RTXDICompositeParams, emissive),
                FIELD(RTXDICompositeParams, output), FIELD(RTXDICompositeParams, settings)}},
            {"Metallic.SharcMaintenanceParams", {FIELD(SharcMaintenanceParams, hashEntries), FIELD(SharcMaintenanceParams, accumulation),
                FIELD(SharcMaintenanceParams, resolved), FIELD(SharcMaintenanceParams, padding0), FIELD(SharcMaintenanceParams, padding1),
                FIELD(SharcMaintenanceParams, settings)}},
            {"Metallic.PathTraceTonemapParams", {FIELD(PathTraceTonemapParams, source), FIELD(PathTraceTonemapParams, output),
                FIELD(PathTraceTonemapParams, historyPrevious), FIELD(PathTraceTonemapParams, settings)}},
            {"Metallic.PathTraceParameters", {FIELD(PathTraceParameters, settings), FIELD(PathTraceParameters, scene), FIELD(PathTraceParameters, output), FIELD(PathTraceParameters, vertices), FIELD(PathTraceParameters, indices), FIELD(PathTraceParameters, primitives), FIELD(PathTraceParameters, instances), FIELD(PathTraceParameters, positions), FIELD(PathTraceParameters, materials), FIELD(PathTraceParameters, historyCurrent), FIELD(PathTraceParameters, historyPrevious), FIELD(PathTraceParameters, materialTextures), FIELD(PathTraceParameters, environment), FIELD(PathTraceParameters, environmentPdf), FIELD(PathTraceParameters, lut2D), FIELD(PathTraceParameters, lut3D), FIELD(PathTraceParameters, lights), FIELD(PathTraceParameters, reGIR), FIELD(PathTraceParameters, punctualPdf), FIELD(PathTraceParameters, albedo), FIELD(PathTraceParameters, specularAlbedo), FIELD(PathTraceParameters, normalRoughness), FIELD(PathTraceParameters, motionVectors), FIELD(PathTraceParameters, linearDepth), FIELD(PathTraceParameters, specularHitDistance), FIELD(PathTraceParameters, depth), FIELD(PathTraceParameters, materialValues), FIELD(PathTraceParameters, ntcLatents), FIELD(PathTraceParameters, ntcConstants), FIELD(PathTraceParameters, ntcWeights), FIELD(PathTraceParameters, ntcInfo), FIELD(PathTraceParameters, ntcSampler)}},
            {"Metallic.UpscalerGuideResolveParams", {FIELD(UpscalerGuideResolveParams, depth), FIELD(UpscalerGuideResolveParams, motion),
                FIELD(UpscalerGuideResolveParams, outputDepth), FIELD(UpscalerGuideResolveParams, outputMotion),
                FIELD(UpscalerGuideResolveParams, jitterX), FIELD(UpscalerGuideResolveParams, jitterY)}},
            {"Metallic.DLSSSupportParams", {FIELD(DLSSSupportParams, depth), FIELD(DLSSSupportParams, color)}},
            {"Metallic.SceneRayQueryVisualizationParams", {FIELD(SceneRayQueryVisualizationParams, scene), FIELD(SceneRayQueryVisualizationParams, output), FIELD(SceneRayQueryVisualizationParams, settings)}},
            {"Metallic.SceneRayQueryVisualizationPush", {FIELD(SceneRayQueryVisualizationPush, eye), FIELD(SceneRayQueryVisualizationPush, center), FIELD(SceneRayQueryVisualizationPush, upProjection), FIELD(SceneRayQueryVisualizationPush, viewport), FIELD(SceneRayQueryVisualizationPush, clipOrtho), FIELD(SceneRayQueryVisualizationPush, mode), FIELD(SceneRayQueryVisualizationPush, width), FIELD(SceneRayQueryVisualizationPush, height), FIELD(SceneRayQueryVisualizationPush, padding)}},
            {"Metallic.RTXCRMaterialSampleParams", {FIELD(RTXCRMaterialSampleParams, output), FIELD(RTXCRMaterialSampleParams, settings)}},
            {"Metallic.VisibilityMaterialParams", {FIELD(VisibilityMaterialParams, output), FIELD(VisibilityMaterialParams, visibility), FIELD(VisibilityMaterialParams, instances), FIELD(VisibilityMaterialParams, materials), FIELD(VisibilityMaterialParams, records), FIELD(VisibilityMaterialParams, meshlets), FIELD(VisibilityMaterialParams, vertices), FIELD(VisibilityMaterialParams, vertexIndices), FIELD(VisibilityMaterialParams, triangles), FIELD(VisibilityMaterialParams, geometries), FIELD(VisibilityMaterialParams, streamRecords), FIELD(VisibilityMaterialParams, groups), FIELD(VisibilityMaterialParams, pages), FIELD(VisibilityMaterialParams, pageTable), FIELD(VisibilityMaterialParams, streamParams), FIELD(VisibilityMaterialParams, settings)}},
            {"Metallic.MaterialErrorParams", {FIELD(MaterialErrorParams, output), FIELD(MaterialErrorParams, color), FIELD(MaterialErrorParams, padding)}},
            {"Metallic.DebugProbeParams", {FIELD(DebugProbeParams, source), FIELD(DebugProbeParams, output), FIELD(DebugProbeParams, settings), FIELD(DebugProbeParams, padding)}},
            {"Metallic.ImageSampleParams", {FIELD(ImageSampleParams, source)}},
            {"Metallic.RenderGraphBufferParams", {FIELD(RenderGraphBufferParams, source), FIELD(RenderGraphBufferParams, output)}},
            {"Metallic.BunnyWireframeParameters", {FIELD(BunnyWireframeParameters, positions), FIELD(BunnyWireframeParameters, transforms), FIELD(BunnyWireframeParameters, settings)}},
            {"Metallic.BunnyWireframeGPUParams", {FIELD(BunnyWireframeGPUParams, eye), FIELD(BunnyWireframeGPUParams, center), FIELD(BunnyWireframeGPUParams, upProjection), FIELD(BunnyWireframeGPUParams, viewport), FIELD(BunnyWireframeGPUParams, clipOrtho), FIELD(BunnyWireframeGPUParams, clearColor), FIELD(BunnyWireframeGPUParams, wireColor), FIELD(BunnyWireframeGPUParams, settings)}},
            {"Metallic.MaterialRasterParameters", {FIELD(MaterialRasterParameters, positions), FIELD(MaterialRasterParameters, materialIndices), FIELD(MaterialRasterParameters, materials), FIELD(MaterialRasterParameters, transforms), FIELD(MaterialRasterParameters, camera), FIELD(MaterialRasterParameters, vertexOffset), FIELD(MaterialRasterParameters, padding0), FIELD(MaterialRasterParameters, padding1), FIELD(MaterialRasterParameters, padding2)}},
            {"Metallic.StreamCompositeParameters", {FIELD(StreamCompositeParameters, colors), FIELD(StreamCompositeParameters, width), FIELD(StreamCompositeParameters, height)}},
            {"Metallic.StreamDeferredParameters", {FIELD(StreamDeferredParameters, settings), FIELD(StreamDeferredParameters, records), FIELD(StreamDeferredParameters, groups), FIELD(StreamDeferredParameters, pageTable), FIELD(StreamDeferredParameters, header), FIELD(StreamDeferredParameters, output), FIELD(StreamDeferredParameters, pages), FIELD(StreamDeferredParameters, visibility), FIELD(StreamDeferredParameters, width), FIELD(StreamDeferredParameters, height), FIELD(StreamDeferredParameters, recordBase), FIELD(StreamDeferredParameters, recordCapacity)}},
            {"Metallic.MaterialVisualizationParameters", {FIELD(MaterialVisualizationParameters, scene), FIELD(MaterialVisualizationParameters, output), FIELD(MaterialVisualizationParameters, vertices), FIELD(MaterialVisualizationParameters, indices), FIELD(MaterialVisualizationParameters, primitives), FIELD(MaterialVisualizationParameters, instances), FIELD(MaterialVisualizationParameters, materials), FIELD(MaterialVisualizationParameters, textures), FIELD(MaterialVisualizationParameters, positions), FIELD(MaterialVisualizationParameters, ntcLatents), FIELD(MaterialVisualizationParameters, ntcConstants), FIELD(MaterialVisualizationParameters, ntcWeights), FIELD(MaterialVisualizationParameters, ntcInfo), FIELD(MaterialVisualizationParameters, ntcSampler), FIELD(MaterialVisualizationParameters, settings)}},
            {"Metallic.SceneMaterialVisualizationPush", {FIELD(SceneMaterialVisualizationPush, eye), FIELD(SceneMaterialVisualizationPush, center), FIELD(SceneMaterialVisualizationPush, upProjection), FIELD(SceneMaterialVisualizationPush, viewport), FIELD(SceneMaterialVisualizationPush, clipOrtho), FIELD(SceneMaterialVisualizationPush, width), FIELD(SceneMaterialVisualizationPush, height), FIELD(SceneMaterialVisualizationPush, mode), FIELD(SceneMaterialVisualizationPush, materialTextureCount), FIELD(SceneMaterialVisualizationPush, bitangentFlip), FIELD(SceneMaterialVisualizationPush, ntcTextureSetCount), FIELD(SceneMaterialVisualizationPush, padding1), FIELD(SceneMaterialVisualizationPush, padding2)}},
            {"Metallic.InstanceCullParameters", {FIELD(InstanceCullParameters, settings), FIELD(InstanceCullParameters, instances), FIELD(InstanceCullParameters, visibility), FIELD(InstanceCullParameters, visibleIds), FIELD(InstanceCullParameters, streamOwners), FIELD(InstanceCullParameters, counter), FIELD(InstanceCullParameters, hzb), FIELD(InstanceCullParameters, phase), FIELD(InstanceCullParameters, padding)}},
            {"Metallic.StreamInstanceCullParameters", {FIELD(StreamInstanceCullParameters, settings), FIELD(StreamInstanceCullParameters, instances), FIELD(StreamInstanceCullParameters, visibility), FIELD(StreamInstanceCullParameters, visibleIds), FIELD(StreamInstanceCullParameters, counter), FIELD(StreamInstanceCullParameters, hzb), FIELD(StreamInstanceCullParameters, phase), FIELD(StreamInstanceCullParameters, width), FIELD(StreamInstanceCullParameters, height), FIELD(StreamInstanceCullParameters, mipCount), FIELD(StreamInstanceCullParameters, hzbValid), FIELD(StreamInstanceCullParameters, cullingFlags), FIELD(StreamInstanceCullParameters, displacementBound), FIELD(StreamInstanceCullParameters, padding)}},
            {"Metallic.ResidentLODParameters", {FIELD(ResidentLODParameters, eye), FIELD(ResidentLODParameters, forward), FIELD(ResidentLODParameters, projection), FIELD(ResidentLODParameters, clusters), FIELD(ResidentLODParameters, records), FIELD(ResidentLODParameters, instances), FIELD(ResidentLODParameters, groups), FIELD(ResidentLODParameters, output), FIELD(ResidentLODParameters, arguments), FIELD(ResidentLODParameters, scratch), FIELD(ResidentLODParameters, offset), FIELD(ResidentLODParameters, count), FIELD(ResidentLODParameters, capacity), FIELD(ResidentLODParameters, instanceCount), FIELD(ResidentLODParameters, groupCount), FIELD(ResidentLODParameters, manualLevel)}},
            {"Metallic.StreamPageTableParameters", {FIELD(StreamPageTableParameters, pages), FIELD(StreamPageTableParameters, patches)}},
            {"Metallic.StreamTraversalParameters", {FIELD(StreamTraversalParameters, settings), FIELD(StreamTraversalParameters, instances), FIELD(StreamTraversalParameters, residentPages), FIELD(StreamTraversalParameters, primitives), FIELD(StreamTraversalParameters, groups), FIELD(StreamTraversalParameters, nodes), FIELD(StreamTraversalParameters, pageTable), FIELD(StreamTraversalParameters, requests), FIELD(StreamTraversalParameters, phase), FIELD(StreamTraversalParameters, threadCount)}},
            {"Metallic.StreamActiveBuildParameters", {FIELD(StreamActiveBuildParameters, settings), FIELD(StreamActiveBuildParameters, activeGroupBuffer), FIELD(StreamActiveBuildParameters, activeHeaderBuffer), FIELD(StreamActiveBuildParameters, demandBuffer), FIELD(StreamActiveBuildParameters, demandStatsBuffer), FIELD(StreamActiveBuildParameters, drawIndirectBuffer), FIELD(StreamActiveBuildParameters, groupBuffer), FIELD(StreamActiveBuildParameters, instanceBuffer), FIELD(StreamActiveBuildParameters, lodLevelBuffer), FIELD(StreamActiveBuildParameters, lodStateBuffer), FIELD(StreamActiveBuildParameters, lodTopologyBuffer), FIELD(StreamActiveBuildParameters, nodeBuffer), FIELD(StreamActiveBuildParameters, pageBuffer), FIELD(StreamActiveBuildParameters, pageTableBuffer), FIELD(StreamActiveBuildParameters, primitiveBuffer), FIELD(StreamActiveBuildParameters, rasterBindingsBuffer), FIELD(StreamActiveBuildParameters, requestBuffer), FIELD(StreamActiveBuildParameters, traversalHeaderBuffer), FIELD(StreamActiveBuildParameters, traversalWorkBuffer), FIELD(StreamActiveBuildParameters, activeBuildPhase), FIELD(StreamActiveBuildParameters, flags)}},
            {"Metallic.StreamTLASParameters", {FIELD(StreamTLASParameters, settings), FIELD(StreamTLASParameters, instances), FIELD(StreamTLASParameters, blasRecords), FIELD(StreamTLASParameters, fallbackAddresses), FIELD(StreamTLASParameters, output)}},
            {"Metallic.StreamBLASParameters", {FIELD(StreamBLASParameters, settings), FIELD(StreamBLASParameters, activeGroupBuffer), FIELD(StreamBLASParameters, activeHeaderBuffer), FIELD(StreamBLASParameters, blasBuildInfoBuffer), FIELD(StreamBLASParameters, blasClusterReferenceBuffer), FIELD(StreamBLASParameters, blasHeaderBuffer), FIELD(StreamBLASParameters, clasAddressBuffer), FIELD(StreamBLASParameters, clasPageTableBuffer), FIELD(StreamBLASParameters, dynamicBlasAddressBuffer), FIELD(StreamBLASParameters, instanceBlasBuffer), FIELD(StreamBLASParameters, scratch), FIELD(StreamBLASParameters, activeBuildPhase), FIELD(StreamBLASParameters, traversalPhase), FIELD(StreamBLASParameters, clasPublicationRevision), FIELD(StreamBLASParameters, padding)}},
            {"Metallic.StreamCandidateParameters", {FIELD(StreamCandidateParameters, headers), FIELD(StreamCandidateParameters, groups), FIELD(StreamCandidateParameters, arguments), FIELD(StreamCandidateParameters, bins), FIELD(StreamCandidateParameters, visibility), FIELD(StreamCandidateParameters, stage), FIELD(StreamCandidateParameters, late)}},
            {"Metallic.StreamClassifyParameters", {FIELD(StreamClassifyParameters, settings), FIELD(StreamClassifyParameters, groups), FIELD(StreamClassifyParameters, pages), FIELD(StreamClassifyParameters, instances), FIELD(StreamClassifyParameters, bins), FIELD(StreamClassifyParameters, tessellationEnabled), FIELD(StreamClassifyParameters, padding)}},
            {"Metallic.StreamClusterCullParameters", {FIELD(StreamClusterCullParameters, settings), FIELD(StreamClusterCullParameters, rasterSettings), FIELD(StreamClusterCullParameters, pages), FIELD(StreamClusterCullParameters, groups), FIELD(StreamClusterCullParameters, header), FIELD(StreamClusterCullParameters, pageTable), FIELD(StreamClusterCullParameters, requests), FIELD(StreamClusterCullParameters, instances), FIELD(StreamClusterCullParameters, visibility), FIELD(StreamClusterCullParameters, records), FIELD(StreamClusterCullParameters, previousHZB), FIELD(StreamClusterCullParameters, currentHZB), FIELD(StreamClusterCullParameters, bins), FIELD(StreamClusterCullParameters, arguments), FIELD(StreamClusterCullParameters, phase), FIELD(StreamClusterCullParameters, stage), FIELD(StreamClusterCullParameters, flags), FIELD(StreamClusterCullParameters, padding)}},
            {"Metallic.HybridBinParameters", {FIELD(HybridBinParameters, bins), FIELD(HybridBinParameters, arguments), FIELD(HybridBinParameters, width), FIELD(HybridBinParameters, height), FIELD(HybridBinParameters, clusterCapacity), FIELD(HybridBinParameters, maxPixels), FIELD(HybridBinParameters, reversedZ), FIELD(HybridBinParameters, subpixelBits), FIELD(HybridBinParameters, producerPixelBuffer), FIELD(HybridBinParameters, inputClusterCount), FIELD(HybridBinParameters, streamMode), FIELD(HybridBinParameters, padding)}},
            {"Metallic.HybridRasterParameters", {FIELD(HybridRasterParameters, queue), FIELD(HybridRasterParameters, pixels), FIELD(HybridRasterParameters, arguments), FIELD(HybridRasterParameters, width), FIELD(HybridRasterParameters, height), FIELD(HybridRasterParameters, capacity), FIELD(HybridRasterParameters, maxPixels), FIELD(HybridRasterParameters, reversedZ), FIELD(HybridRasterParameters, subpixelBits)}},
            {"Metallic.HybridResolveParameters", {FIELD(HybridResolveParameters, pixels), FIELD(HybridResolveParameters, width), FIELD(HybridResolveParameters, reversedZ)}},
            {"Metallic.StreamRasterParameters", {FIELD(StreamRasterParameters, settings), FIELD(StreamRasterParameters, pages), FIELD(StreamRasterParameters, groups), FIELD(StreamRasterParameters, header), FIELD(StreamRasterParameters, pageTable), FIELD(StreamRasterParameters, instances), FIELD(StreamRasterParameters, bins), FIELD(StreamRasterParameters, pixels), FIELD(StreamRasterParameters, visibleRecordBase), FIELD(StreamRasterParameters, visibleRecordCapacity), FIELD(StreamRasterParameters, hasInstances), FIELD(StreamRasterParameters, padding)}},
            {"Metallic.StreamWorkloadParameters", {FIELD(StreamWorkloadParameters, raster), FIELD(StreamWorkloadParameters, counters)}},
            {"Metallic.ShadowTraceParameters", {FIELD(ShadowTraceParameters, settings), FIELD(ShadowTraceParameters, scene), FIELD(ShadowTraceParameters, streamScene), FIELD(ShadowTraceParameters, depth), FIELD(ShadowTraceParameters, penumbra), FIELD(ShadowTraceParameters, normal), FIELD(ShadowTraceParameters, viewZ), FIELD(ShadowTraceParameters, motion), FIELD(ShadowTraceParameters, shadow), FIELD(ShadowTraceParameters, materialTextureCount), FIELD(ShadowTraceParameters, ntcTextureSetCount)}},
            {"Metallic.RTXDITraceParameters", {FIELD(RTXDITraceParameters, settings), FIELD(RTXDITraceParameters, scene), FIELD(RTXDITraceParameters, output), FIELD(RTXDITraceParameters, reservoirCurrent), FIELD(RTXDITraceParameters, reservoirPrevious), FIELD(RTXDITraceParameters, positionCurrent), FIELD(RTXDITraceParameters, positionPrevious), FIELD(RTXDITraceParameters, normalCurrent), FIELD(RTXDITraceParameters, normalPrevious), FIELD(RTXDITraceParameters, noisyDiffuse), FIELD(RTXDITraceParameters, noisySpecular), FIELD(RTXDITraceParameters, normalRoughness), FIELD(RTXDITraceParameters, motionVectors), FIELD(RTXDITraceParameters, viewZ), FIELD(RTXDITraceParameters, baseColorMetalness), FIELD(RTXDITraceParameters, emissive)}},
            {"Metallic.PathTraceInlineParameters", {FIELD(PathTraceInlineParameters, resources), FIELD(PathTraceInlineParameters, settings), FIELD(PathTraceInlineParameters, output), FIELD(PathTraceInlineParameters, historyCurrent), FIELD(PathTraceInlineParameters, historyPrevious)}},
            {"Metallic.SharcTraceParameters", {FIELD(SharcTraceParameters, path), FIELD(SharcTraceParameters, cacheSettings), FIELD(SharcTraceParameters, hashEntries), FIELD(SharcTraceParameters, accumulation), FIELD(SharcTraceParameters, resolved)}},
            {"Metallic.NRCTraceParameters", {FIELD(NRCTraceParameters, path), FIELD(NRCTraceParameters, cacheSettings), FIELD(NRCTraceParameters, queryPath), FIELD(NRCTraceParameters, trainingPath), FIELD(NRCTraceParameters, vertices), FIELD(NRCTraceParameters, radiance), FIELD(NRCTraceParameters, counters)}},
            {"Metallic.PathTraceGuidesInlineParameters", {FIELD(PathTraceGuidesInlineParameters, path), FIELD(PathTraceGuidesInlineParameters, albedo), FIELD(PathTraceGuidesInlineParameters, specularAlbedo), FIELD(PathTraceGuidesInlineParameters, normalRoughness), FIELD(PathTraceGuidesInlineParameters, motionVectors), FIELD(PathTraceGuidesInlineParameters, linearDepth), FIELD(PathTraceGuidesInlineParameters, specularHitDistance), FIELD(PathTraceGuidesInlineParameters, depth)}},
            {"Metallic.RealtimeLightingParameters", {FIELD(RealtimeLightingParameters, path), FIELD(RealtimeLightingParameters, irradiance)}},
            {"Metallic.DeferredShadingParameters", {FIELD(DeferredShadingParameters, path), FIELD(DeferredShadingParameters, resources), FIELD(DeferredShadingParameters, visibility), FIELD(DeferredShadingParameters, depth), FIELD(DeferredShadingParameters, domain), FIELD(DeferredShadingParameters, motion), FIELD(DeferredShadingParameters, deviceDepth), FIELD(DeferredShadingParameters, binIndex), FIELD(DeferredShadingParameters, padding)}},
            {"Metallic.DeferredShadingResources", {FIELD(DeferredShadingResources, vertices), FIELD(DeferredShadingResources, meshlets), FIELD(DeferredShadingResources, records), FIELD(DeferredShadingResources, meshletVertices), FIELD(DeferredShadingResources, triangles), FIELD(DeferredShadingResources, geometries), FIELD(DeferredShadingResources, instances), FIELD(DeferredShadingResources, materials), FIELD(DeferredShadingResources, bins), FIELD(DeferredShadingResources, tiles), FIELD(DeferredShadingResources, irradiance), FIELD(DeferredShadingResources, specular), FIELD(DeferredShadingResources, gridParams), FIELD(DeferredShadingResources, gridLights), FIELD(DeferredShadingResources, gridCandidates), FIELD(DeferredShadingResources, gridCells), FIELD(DeferredShadingResources, gridIndices), FIELD(DeferredShadingResources, shadow), FIELD(DeferredShadingResources, shadowParams), FIELD(DeferredShadingResources, view), FIELD(DeferredShadingResources, streamRecords), FIELD(DeferredShadingResources, streamGroups), FIELD(DeferredShadingResources, streamPages), FIELD(DeferredShadingResources, streamTable), FIELD(DeferredShadingResources, frameInfo), FIELD(DeferredShadingResources, streamParams), FIELD(DeferredShadingResources, feedback), FIELD(DeferredShadingResources, sampler)}},
        };
#undef FIELD
        struct Program { const char* module; const char* entry; uint32_t layout; const char* define = nullptr; };
        const Program programs[] = {
            {"Features/PostProcess/FinalBlit", "finalBlitMain", 0},
            {"Features/PostProcess/FinalBlit", "finalBlitUvMain", 0},
            {"Features/Debug/SliderDebug", "sliderDebugMain", 1},
            {"Features/Debug/SliderDebug", "sliderDebugOverlayMain", 1},
            {"Features/PostProcess/AutoExposure", "autoExposureHistogramMain", 2},
            {"Features/PostProcess/AutoExposure", "autoExposureReduceMain", 2},
            {"Features/PostProcess/AutoExposure", "autoExposureApplyMain", 2},
            {"Features/PostProcess/ColorGradingLUT", "composeColorGradingLUT", 3},
            {"Features/Lighting/ClusterLightGrid", "clusterLightGridMain", 4},
            {"Features/Debug/LightGridDebug", "lightGridDebugMain", 5},
            {"Features/Lighting/PrepareLightsPdf", "prepareLightsPdfMain", 6},
            {"Features/Lighting/BuildReGIR", "buildReGIRMain", 7},
            {"Features/Environment/EnvironmentLightingPrecompute", "environmentLightingPrecomputeMain", 8},
            {"Features/ReSTIR/RTXDIConfidence", "rtxdiConfidenceMain", 9},
            {"Features/ReSTIR/RTXDIComposite", "rtxdiCompositeMain", 10},
            {"Features/PathTracing/SceneSharcMaintenance", "sharcClearMain", 11},
            {"Features/PathTracing/SceneSharcMaintenance", "sharcResolveMain", 11},
            {"Features/PostProcess/ScenePathTraceTonemap", "scenePathTraceTonemapMain", 12},
            {"Features/PathTracing/OpenPBRRayQueryPathTrace", "openPbrRayQueryPathTraceMain", 13},
            {"Features/PathTracing/OpenPBRRayQueryPathTraceGuides", "openPbrRayQueryPathTraceGuidesMain", 13},
            {"Features/PathTracing/ScenePathTraceGuides", "scenePathTraceGuidesMain", 13},
            {"Features/PostProcess/UpscalerGuideResolve", "upscalerGuideResolveMain", 14},
            {"Features/PostProcess/StreamlineDLSSSupport", "streamlineDlssAlphaMain", 15},
            {"Features/PostProcess/StreamlineDLSSSupport", "streamlineDlssDepthFragmentMain", 15},
            {"Features/Debug/SceneRayQueryVisualize", "sceneRayQueryVisualizeMain", 16},
            {"Features/Debug/SceneRayQueryVisualize", "sceneRayQueryVisualizeMain", 17},
            {"Features/Samples/RTXCRMaterialSample", "rtxcrMaterialSampleMain", 18},
            {"Features/VisibilityBuffer/VisibilityBufferMaterial", "visibilityBufferMaterialMain", 19},
            {"Features/Material/MaterialError", "materialErrorMain", 20},
            {"Features/Debug/GPUProbe", "probe", 21},
            {"Features/Samples/ImageSample", "imageSampleFragmentMain", 22},
            {"Features/GPUDriven/StreamComposite", "gpuDrivenStreamAssetCompositeFragmentMain", 27},
            {"Features/GPUDriven/GPUDrivenStreamAsset", "gpuDrivenStreamAssetDeferredMain", 28},
            {"Features/Debug/SceneMaterialVisualize", "sceneMaterialVisualizeMain", 29},
            {"Features/Debug/SceneMaterialVisualize", "sceneMaterialVisualizeMain", 30},
            {"Features/Samples/MaterialShaderObject", "materialShaderObjectVertexMain", 26},
            {"Features/Samples/MaterialShaderObject", "materialShaderObjectFragmentMain", 26},
            {"Features/Samples/MaterialShaderObject", "materialShaderObjectAlternateFragmentMain", 26},
            {"Features/Samples/BunnyWireframe", "bunnyWireframeVertexMain", 24},
            {"Features/Samples/BunnyWireframe", "bunnyWireframeFragmentMain", 24},
            {"Features/Samples/BunnyWireframe", "bunnyWireframeVertexMain", 25},
            {"Features/Samples/BunnyWireframe", "bunnyWireframeFragmentMain", 25},
            {"Features/SmokeTests/RenderGraphBuffer", "renderGraphBufferWriteMain", 23},
            {"Features/SmokeTests/RenderGraphBuffer", "renderGraphBufferCopyMain", 23},
            {"Features/GPUDriven/GPUDrivenCulling", "gpuDrivenPreviewResetMain", 31},
            {"Features/GPUDriven/GPUDrivenCulling", "gpuDrivenPreviewInstanceCullMain", 31},
            {"Features/GPUDriven/GPUDrivenStreamAsset", "gpuDrivenStreamAssetCullResetMain", 32},
            {"Features/GPUDriven/GPUDrivenStreamAsset", "gpuDrivenStreamAssetInstanceCullMain", 32},
            {"Features/GPUDriven/ResidentMeshletLOD", "residentLodResetMain", 33},
            {"Features/GPUDriven/ResidentMeshletLOD", "residentLodSelectMain", 33},
            {"Features/GPUDriven/ResidentMeshletLOD", "residentLodArgumentsMain", 33},
            {"Features/GPUDriven/ResidentMeshletLOD", "residentLodScatterMain", 33},
            {"Features/GPUDriven/GPUDrivenStreamAsset", "gpuDrivenStreamAssetInitializePageTableMain", 34},
            {"Features/GPUDriven/GPUDrivenStreamAsset", "gpuDrivenStreamAssetApplyUpdatesMain", 34},
            {"Features/GPUDriven/GPUDrivenStreamAsset", "gpuDrivenStreamAssetTraversalMain", 35},
            {"Features/GPUDriven/GPUDrivenStreamAsset", "gpuDrivenStreamAssetBuildActiveMain", 36},
            {"Features/GPUDriven/GPUDrivenStreamAsset", "streamCooperativeLodMain", 36},
            {"Features/GPUDriven/GPUDrivenStreamAsset", "streamDistributedDemandMain", 36},
            {"Features/GPUDriven/GPUDrivenStreamAsset", "gpuDrivenStreamAssetBuildTlasInputMain", 37},
            {"Features/GPUDriven/GPUDrivenStreamAsset", "gpuDrivenStreamAssetBuildBlasInputMain", 38},
            {"Features/GPUDriven/GPUDrivenStreamAsset", "streamClusterPrepareMain", 39},
            {"Features/GPUDriven/GPUDrivenStreamAsset", "streamClusterBinMain", 40},
            {"Features/GPUDriven/GPUDrivenStreamAsset", "streamClusterBinP0Main", 40},
            {"Features/GPUDriven/GPUDrivenStreamAsset", "streamClusterCullMain", 41},
            {"Features/GPUDriven/GPUDrivenStreamAsset", "streamClusterCullP0Main", 41},
            {"Features/VisibilityBuffer/VisibilityHybridRaster", "hybridClusterResetMain", 42},
            {"Features/VisibilityBuffer/VisibilityHybridRaster", "hybridClusterHistogramMain", 42},
            {"Features/VisibilityBuffer/VisibilityHybridRaster", "hybridClusterArgumentsMain", 42},
            {"Features/VisibilityBuffer/VisibilityHybridRaster", "hybridClusterScatterMain", 42},
            {"Features/VisibilityBuffer/VisibilityHybridRaster", "hybridResetMain", 43},
            {"Features/VisibilityBuffer/VisibilityHybridRaster", "hybridArgumentsMain", 43},
            {"Features/VisibilityBuffer/VisibilityHybridRaster", "hybridRasterMain", 43},
            {"Features/VisibilityBuffer/VisibilityHybridRaster", "hybridResolveFragmentMain", 44},
            {"Features/GPUDriven/GPUDrivenStreamAsset", "streamClusterRasterMain", 45},
            {"Features/GPUDriven/GPUDrivenStreamAsset", "streamClusterRasterLegacyMain", 45},
            {"Features/GPUDriven/GPUDrivenStreamAsset", "streamClusterRasterPlaneMain", 45},
            {"Features/GPUDriven/GPUDrivenStreamAsset", "streamClusterRasterCooperativeMain", 45},
            {"Features/GPUDriven/GPUDrivenStreamWorkRaster", "streamClusterRasterWorkBinsMain", 45},
            {"Features/GPUDriven/GPUDrivenStreamWorkRaster", "streamClusterRasterWorkControlMain", 45},
            {"Features/GPUDriven/GPUDrivenStreamWorkload", "streamWorkloadResetMain", 46},
            {"Features/GPUDriven/GPUDrivenStreamWorkload", "streamWorkloadMain", 46},
            {"Features/Lighting/ScreenSpaceShadows", "rayTracedShadowsMain", 47},
            {"Features/ReSTIR/SceneRTXDI", "sceneRtxdiMain", 48},
            {"Features/PathTracing/ScenePathTraceInline", "scenePathTraceMain", 49},
            {"Features/PathTracing/OpenPBRRayQueryPathTrace", "openPbrRayQueryPathTraceMain", 49},
            {"Features/PathTracing/ScenePathTraceGuides", "scenePathTraceGuidesMain", 52},
            {"Features/Lighting/SceneRealtimeLighting", "sceneRealtimeLightingMain", 53},
            {"Features/VisibilityBuffer/VisibilityBufferDeferred", "visibilityBufferDeferredMain", 54, "METALLIC_DEFERRED_LIGHT_GRID"},
            {"Features/VisibilityBuffer/VisibilityBufferDeferred", "visibilityBufferDeferredBinnedMain", 54, "METALLIC_DEFERRED_LIGHT_GRID"},
            {"Features/VisibilityBuffer/VisibilityBufferDeferred", "visibilityBufferDeferredMain", 55, "METALLIC_DEFERRED_LIGHT_GRID"},
            {"Features/VisibilityBuffer/VisibilityBufferDeferred", "visibilityBufferDeferredBinnedMain", 55, "METALLIC_DEFERRED_LIGHT_GRID"},
            {"Features/PathTracing/OpenPBRRayQueryPathTraceGuides", "openPbrRayQueryPathTraceGuidesMain", 52},
            {"Features/PathTracing/ScenePathTraceSharc", "scenePathTraceMain", 50, "SHARC_UPDATE"},
            {"Features/PathTracing/ScenePathTraceSharc", "scenePathTraceMain", 50, "SHARC_QUERY"},
            {"Features/PathTracing/ScenePathTraceNRC", "scenePathTraceMain", 51, "NRC_UPDATE"},
            {"Features/PathTracing/ScenePathTraceNRC", "scenePathTraceMain", 51, "NRC_QUERY"},
            {"Features/GPUDriven/GPUDrivenStreamGroupRaster", "streamClusterRasterGroup32Main", 45},
            {"Features/GPUDriven/GPUDrivenStreamGroupRaster", "streamClusterRasterGroup64Main", 45},
            {"Features/GPUDriven/GPUDrivenStreamGroupRaster", "streamClusterRasterGroup128Main", 45},




        };
        for (auto mode : {SlangDescriptorHeapMode::Mapped, SlangDescriptorHeapMode::Native}) {
            for (const auto& program : programs) {
                if ((program.layout >= 54 ? 40u : program.layout == 53 ? 39u : program.layout == 52 ? 38u : program.layout == 51 ? 37u : program.layout == 50 ? 36u : program.layout == 49 ? 35u : program.layout == 48 ? 34u : program.layout == 47 ? 33u : program.layout == 46 ? 32u : program.layout == 45 ? 31u : program.layout == 44 ? 30u : program.layout == 43 ? 29u : program.layout == 42 ? 28u : program.layout == 41 ? 27u : program.layout == 40 ? 26u : program.layout == 39 ? 25u : program.layout == 38 ? 24u : program.layout == 37 ? 23u : program.layout == 36 ? 22u : program.layout == 35 ? 21u : program.layout == 34 ? 20u : program.layout == 33 ? 19u : program.layout == 32 ? 18u : program.layout == 31 ? 17u : program.layout >= 29 ? 16u : program.layout == 28 ? 15u : program.layout == 27 ? 14u : program.layout == 26 ? 13u : program.layout >= 24 ? 12u : program.layout == 23 ? 11u : program.layout == 22 ? 10u : program.layout == 21 ? 9u : program.layout == 20 ? 8u : program.layout == 19 ? 7u : program.layout == 18 ? 6u : program.layout >= 16 ? 5u : program.layout >= 14 ? 0u : program.layout >= 13 ? 4u : program.layout >= 11 ? 3u : program.layout >= 9 ? 2u : program.layout >= 4 ? 1u : 0u) != category) { continue; }
                const SlangMacroDefine defines[] = {{"FINAL_USE_LUT", "1"},
                    {program.define ? program.define : "METALLIC_TEST_PARAMETER_LAYOUT", "1"}};
                const char* capabilities[] = {"spvRayQueryKHR"};
                std::span<const char* const> extraPaths;
#if METALLIC_HAS_RTXCR
                const char* rtxcrPaths[] = {METALLIC_RTXCR_SHADER_INCLUDE_DIR};
                if (category == 6) { extraPaths = rtxcrPaths; }
#endif
                std::string log;
                auto shader = compileSlangShaderToSpirv({.moduleName = program.module, .entryPointName = program.entry,
                    .searchPath = PROJECT_SOURCE_DIR "/Shaders", .additionalSearchPaths = extraPaths, .capabilities = category >= 4 ? std::span<const char* const>(capabilities) : std::span<const char* const>{}, .macroDefines = defines, .descriptorHeapMode = mode}, log);
                if (!shader) { return RHITestResult::fail(log); }
                const auto& layout = layouts[program.layout];
                std::vector<uint32_t> ids;
                std::map<uint32_t, std::map<uint32_t, uint32_t>> offsets;
                const auto& words = shader->spirv;
                for (size_t i = 5; i < words.size();) {
                    const uint32_t count = words[i] >> 16, opcode = words[i] & 0xffff;
                    REG_CHECK(count && count <= words.size() - i);
                    if (opcode == 5 && count >= 3) { // OpName
                        const char* bytes = reinterpret_cast<const char*>(&words[i + 2]);
                        const size_t capacity = (count - 2) * sizeof(uint32_t);
                        const void* terminator = std::memchr(bytes, 0, capacity);
                        REG_CHECK(terminator);
                        const std::string_view name(bytes, static_cast<const char*>(terminator) - bytes);
                        if (name == layout.name || name.starts_with(std::string(layout.name) + "_")) {
                            ids.push_back(words[i + 1]);
                        }
                    } else if (opcode == 72 && count == 5 && words[i + 3] == 35) { // OpMemberDecorate Offset
                        offsets[words[i + 1]][words[i + 2]] = words[i + 4];
                    }
                    i += count;
                }
                bool matched = false;
                for (const auto id : ids) {
                    const auto& members = offsets[id];
                    if (members.size() != layout.offsets.size()) { continue; }
                    bool correct = true;
                    for (uint32_t i = 0; i < layout.offsets.size(); ++i) {
                        const auto member = members.find(i);
                        correct &= member != members.end() && member->second == layout.offsets[i];
                    }
                    matched |= correct;
                }
                if (!matched) { return RHITestResult::fail(std::string(program.entry) + ": C++/SPIR-V parameter offsets disagree"); }
                bool sharedHeader = false;
                for (const auto& dependency : shader->dependencies) {
                    sharedHeader |= dependency.ends_with(category == 40 ? "DeferredShadingParameters.h" : category == 39 ? "RealtimeLightingParameters.h" : category == 38 ? "PathTraceGuidesInlineParameters.h" : category == 37 ? "NRCTraceParameters.h" : category == 36 ? "SharcTraceParameters.h" : category == 35 ? "PathTraceInlineParameters.h" : category == 34 ? "RTXDITraceParameters.h" : category == 33 ? "ShadowTraceParameters.h" : category == 32 ? "StreamWorkloadParameters.h" : category == 31 ? "StreamRasterParameters.h" : category == 30 ? "HybridResolveParameters.h" : category == 29 ? "HybridRasterParameters.h" : category == 28 ? "HybridBinParameters.h" : category == 27 ? "StreamClusterCullParameters.h" : category == 26 ? "StreamClassifyParameters.h" : category == 25 ? "StreamCandidateParameters.h" : category == 24 ? "StreamBLASParameters.h" : category == 23 ? "StreamTLASParameters.h" : category == 22 ? "StreamActiveBuildParameters.h" : category == 21 ? "StreamTraversalParameters.h" : category == 20 ? "StreamPageTableParameters.h" : category == 19 ? "ResidentLODParameters.h" : category == 18 ? "StreamInstanceCullParameters.h" : category == 17 ? "InstanceCullParameters.h" : category == 16 ? "MaterialVisualizationParameters.h" : category == 15 ? "StreamDeferredParameters.h" : category == 14 ? "StreamCompositeParameters.h" : category == 13 ? "MaterialRasterParameters.h" : category == 12 ? "BunnyWireframeParameters.h" : category == 11 ? "RenderGraphBufferParameters.h" : category == 10 ? "ImageSampleParameters.h" : category == 9 ? "DebugProbeParameters.h" : category == 8 ? "MaterialErrorParameters.h" : category == 7 ? "VisibilityMaterialParameters.h" : category == 6 ? "MaterialSampleParameters.h" : category == 5 ? "DebugVisualizationParameters.h" : category == 4 ? "PathTraceParameters.h" : category == 3 ? "PathTraceStageParameters.h" : category == 2 ? "RTXDIPostProcessParameters.h" : category == 1 ? "LightingKernelParameters.h" : "PostProcessParameters.h");
                }
                REG_CHECK(sharedHeader); // Layout edits must invalidate the shader cache.
            }
        }
        return RHITestResult::pass("Mapped/native offsets and shared header cache dependency");
    }
};
METALLIC_REGISTER_RHI_TEST(PostProcessParameterLayoutTest);

class LightingParameterLayoutTest final : public PostProcessParameterLayoutTest {
public:
    LightingParameterLayoutTest() { category = 1; name = "lighting_parameter_spirv_layout"; }
};
METALLIC_REGISTER_RHI_TEST(LightingParameterLayoutTest);

class RTXDIParameterLayoutTest final : public PostProcessParameterLayoutTest {
public:
    RTXDIParameterLayoutTest() { category = 2; name = "rtxdi_parameter_spirv_layout"; }
};
METALLIC_REGISTER_RHI_TEST(RTXDIParameterLayoutTest);

class PathTraceStageParameterLayoutTest final : public PostProcessParameterLayoutTest {
public:
    PathTraceStageParameterLayoutTest() { category = 3; name = "path_trace_stage_parameter_spirv_layout"; }
};
METALLIC_REGISTER_RHI_TEST(PathTraceStageParameterLayoutTest);

class PathTraceParameterLayoutTest final : public PostProcessParameterLayoutTest {
public:
    PathTraceParameterLayoutTest() { category = 4; name = "path_trace_parameter_spirv_layout"; }
};
METALLIC_REGISTER_RHI_TEST(PathTraceParameterLayoutTest);

class DebugVisualizationParameterLayoutTest final : public PostProcessParameterLayoutTest {
public:
    DebugVisualizationParameterLayoutTest() { category = 5; name = "debug_visualization_parameter_spirv_layout"; }
};
METALLIC_REGISTER_RHI_TEST(DebugVisualizationParameterLayoutTest);

#if METALLIC_HAS_RTXCR
class MaterialSampleParameterLayoutTest final : public PostProcessParameterLayoutTest {
public:
    MaterialSampleParameterLayoutTest() { category = 6; name = "material_sample_parameter_spirv_layout"; }
};
METALLIC_REGISTER_RHI_TEST(MaterialSampleParameterLayoutTest);
#endif

class VisibilityMaterialParameterLayoutTest final : public PostProcessParameterLayoutTest {
public:
    VisibilityMaterialParameterLayoutTest() { category = 7; name = "visibility_material_parameter_spirv_layout"; }
};
METALLIC_REGISTER_RHI_TEST(VisibilityMaterialParameterLayoutTest);

class MaterialErrorParameterLayoutTest final : public PostProcessParameterLayoutTest {
public:
    MaterialErrorParameterLayoutTest() { category = 8; name = "material_error_parameter_spirv_layout"; }
};
METALLIC_REGISTER_RHI_TEST(MaterialErrorParameterLayoutTest);

class DebugProbeParameterLayoutTest final : public PostProcessParameterLayoutTest {
public:
    DebugProbeParameterLayoutTest() { category = 9; name = "debug_probe_parameter_spirv_layout"; }
};
METALLIC_REGISTER_RHI_TEST(DebugProbeParameterLayoutTest);

class ImageSampleParameterLayoutTest final : public PostProcessParameterLayoutTest {
public:
    ImageSampleParameterLayoutTest() { category = 10; name = "image_sample_parameter_spirv_layout"; }
};
METALLIC_REGISTER_RHI_TEST(ImageSampleParameterLayoutTest);

class GraphBufferParameterLayoutTest final : public PostProcessParameterLayoutTest {
public:
    GraphBufferParameterLayoutTest() { category = 11; name = "graph_buffer_parameter_spirv_layout"; }
};
METALLIC_REGISTER_RHI_TEST(GraphBufferParameterLayoutTest);

class BunnyWireframeParameterLayoutTest final : public PostProcessParameterLayoutTest {
public:
    BunnyWireframeParameterLayoutTest() { category = 12; name = "bunny_wireframe_parameter_spirv_layout"; }
};
METALLIC_REGISTER_RHI_TEST(BunnyWireframeParameterLayoutTest);

class MaterialRasterParameterLayoutTest final : public PostProcessParameterLayoutTest {
public:
    MaterialRasterParameterLayoutTest() { category = 13; name = "material_raster_parameter_spirv_layout"; }
};
METALLIC_REGISTER_RHI_TEST(MaterialRasterParameterLayoutTest);

class StreamCompositeParameterLayoutTest final : public PostProcessParameterLayoutTest {
public:
    StreamCompositeParameterLayoutTest() { category = 14; name = "stream_composite_parameter_spirv_layout"; }
};
METALLIC_REGISTER_RHI_TEST(StreamCompositeParameterLayoutTest);

class StreamDeferredParameterLayoutTest final : public PostProcessParameterLayoutTest {
public:
    StreamDeferredParameterLayoutTest() { category = 15; name = "stream_deferred_parameter_spirv_layout"; }
};
METALLIC_REGISTER_RHI_TEST(StreamDeferredParameterLayoutTest);

class MaterialVisualizationParameterLayoutTest final : public PostProcessParameterLayoutTest {
public:
    MaterialVisualizationParameterLayoutTest() { category = 16; name = "material_visualization_parameter_spirv_layout"; }
};
METALLIC_REGISTER_RHI_TEST(MaterialVisualizationParameterLayoutTest);

class InstanceCullParameterLayoutTest final : public PostProcessParameterLayoutTest {
public:
    InstanceCullParameterLayoutTest() { category = 17; name = "instance_cull_parameter_spirv_layout"; }
};
METALLIC_REGISTER_RHI_TEST(InstanceCullParameterLayoutTest);

class StreamInstanceCullParameterLayoutTest final : public PostProcessParameterLayoutTest {
public:
    StreamInstanceCullParameterLayoutTest() { category = 18; name = "stream_instance_cull_parameter_spirv_layout"; }
};
METALLIC_REGISTER_RHI_TEST(StreamInstanceCullParameterLayoutTest);

class ResidentLODParameterLayoutTest final : public PostProcessParameterLayoutTest {
public:
    ResidentLODParameterLayoutTest() { category = 19; name = "resident_lod_parameter_spirv_layout"; }
};
METALLIC_REGISTER_RHI_TEST(ResidentLODParameterLayoutTest);










class SharcTypedMaintenanceTest final : public RHITest {
public:
    SharcTypedMaintenanceTest() { type = RHITestType::Resource; name = "sharc_typed_maintenance_bounds_and_eviction"; }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        bench::TestDevice device;
        REG_REQUIRE(bench::createTestDevice(context, {.applicationName = "SHaRC typed stages",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true})
            .transform([&](auto value) { device = std::move(value); }));
        auto& queue = *device->getQueue(QueueType::Graphics);
        auto registry = device->resourceRegistry();
        REG_CHECK(registry);
        QueueSubmissionTracker tracker;
        REG_REQUIRE(tracker.initialize(*device, queue));
        // No legacy cacheParams buffer is provided. The inline entry count and
        // stale threshold must control both kernels, including the tail lanes.
        for (uint32_t testCase = 0; testCase < 3; ++testCase) {
            const bool clear = testCase == 0;
            ComputeKernel kernel;
            std::string log;
            auto shader = compileSlangShaderToSpirv({.moduleName = "Features/PathTracing/SceneSharcMaintenance",
                .entryPointName = clear ? "sharcClearMain" : "sharcResolveMain",
                .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, log);
            if (!shader) { return RHITestResult::fail(log); }
            REG_REQUIRE(kernel.initialize(*device, {.spirv = shader->spirv,
                .parameters = parameterAbi<SharcMaintenanceParams>(kSharcMaintenanceABI, ParameterTransport::InlinePush)}, log));
            std::array<std::unique_ptr<Buffer>, 3> buffers;
            std::array<std::vector<uint32_t>, 3> initial;
            for (uint32_t b = 0; b < 3; ++b) {
                const uint32_t words = b == 0 ? 2 : 4;
                initial[b].resize(32 * words, clear ? 0xffffffffu : 0u);
                if (!clear) {
                    for (uint32_t entry = 0; entry < 32; ++entry) {
                        if (b == 0) { initial[b][entry * words] = 1; }
                        // SharcPackedData.sampleData: seven stale frames.
                        if (b == 2) { initial[b][entry * words + 2] = 7u << 16; }
                    }
                }
                REG_REQUIRE(device->createBuffer({.size = initial[b].size() * sizeof(uint32_t),
                    .structureStride = words * sizeof(uint32_t), .usage = BufferUsageBits::Storage,
                    .memoryLocation = MemoryLocation::HostReadback})
                    .transform([&](auto value) { buffers[b] = std::move(value); }));
                void* mapped = buffers[b]->map();
                REG_CHECK(mapped);
                std::memcpy(mapped, initial[b].data(), initial[b].size() * sizeof(uint32_t));
                buffers[b]->flush();
                buffers[b]->unmap();
            }
            Commands recording;
            REG_REQUIRE(recording.initialize(*device, queue));
            REG_REQUIRE(recording.begin(testCase));
            {
                ParameterWriter writer(*device, **registry, &recording.frame);
                SharcMaintenanceParams params{};
                params.hashEntries = writer.buffer(buffers[0].get());
                params.accumulation = writer.buffer(buffers[1].get());
                params.resolved = writer.buffer(buffers[2].get());
                params.settings.entriesNum = 17;
                params.settings.sceneScale = 1.0f;
                params.settings.accumulationFrameNum = 20;
                params.settings.staleFrameNumMax = testCase == 1 ? 8 : 9;
                auto encoded = writer.encode(params, kSharcMaintenanceABI, ParameterTransport::InlinePush);
                REG_CHECK(encoded);
                REG_REQUIRE(kernel.dispatch(*recording.commands, *encoded, 1));
            }
            const MemoryBarrierDesc hostRead{{PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
                {PipelineStageBits::Host, AccessBits::HostRead}};
            REG_REQUIRE(recording.commands->synchronize({.memory = {&hostRead, 1}}));
            std::unique_ptr<Semaphore> gate;
            REG_REQUIRE(device->createSemaphore().transform([&](auto value) { gate = std::move(value); }));
            Drain drain{queue, *gate};
            REG_REQUIRE(recording.submit(tracker, *gate));
            kernel.clear(); // Submission owns the pipeline and immutable parameters.
            REG_REQUIRE(gate->signal(1));
            REG_REQUIRE(recording.frame.wait(5'000'000'000ull));
            for (uint32_t b = 0; b < 3; ++b) {
                const uint32_t words = b == 0 ? 2 : 4;
                auto expected = initial[b];
                for (uint32_t entry = 0; entry < 17; ++entry) {
                    if (clear || testCase == 1) {
                        for (uint32_t word = 0; word < words; ++word) { expected[entry * words + word] = 0; }
                    } else if (b == 2) { expected[entry * words + 2] = (8u << 16) | 1u; }
                }
                buffers[b]->invalidate();
                void* mapped = buffers[b]->map();
                REG_CHECK(mapped);
                const bool equal = std::memcmp(mapped, expected.data(), expected.size() * sizeof(uint32_t)) == 0;
                buffers[b]->unmap();
                REG_CHECK(equal);
            }
        }
        REG_CHECK((*registry)->stats().parameterBytes == 0);
        return RHITestResult::pass("Clear/resolve respect inline bounds; thresholds 8/9 evict/retain; no parameter buffer upload");
    }
};
METALLIC_REGISTER_RHI_TEST(SharcTypedMaintenanceTest);

class RegistryIdentityTest final : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"registry.identity.capacity.contract", "registry.descriptor.recycle.contract"}, bench::Layer::Core, "binding", "binding");
    }

    RegistryIdentityTest() { type = RHITestType::Resource; name = "registry_identity_capacity_and_views"; }
    RHITestResult run(RHITestContext& context) override
    {
        bench::TestDevice device;
        REG_REQUIRE(bench::createTestDevice(context, {.applicationName = "Registry identity", .enableValidation = context.enableValidation,
            .enableBindlessDescriptorHeap = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); }));
        render::ResourceRegistry registry;
        REG_REQUIRE(registry.initialize(*device, {.maxSamplers = 2, .maxSampledImages = 2,
            .maxStorageImages = 1, .maxBuffers = 2}));
        std::unique_ptr<render::Buffer> a, b, c;
        REG_REQUIRE(makeBuffer(*device, a));
        REG_REQUIRE(makeBuffer(*device, b));
        REG_REQUIRE(makeBuffer(*device, c));
        render::ResourceLease aLease, bLease, duplicate, cLease;
        REG_REQUIRE(registry.storageBuffer(*a).transform([&](auto value) { aLease = std::move(value); }));
        REG_REQUIRE(registry.storageBuffer(*b).transform([&](auto value) { bLease = std::move(value); }));
        REG_REQUIRE(registry.storageBuffer(*b).transform([&](auto value) { duplicate = std::move(value); }));
        REG_CHECK(bLease.shaderValue() == duplicate.shaderValue());
        REG_CHECK(registry.stats().descriptorWrites == 2 && registry.stats().cacheHits == 1);
        std::weak_ptr<void> oldAllocation = a->retainAllocation();
        a.reset();
        REG_CHECK(!oldAllocation.expired());
        const auto exhausted = registry.storageBuffer(*c);
        REG_CHECK(render::hasError(exhausted, render::Error::OutOfMemory));
        REG_CHECK(!exhausted.has_value());
        const auto releasedIndex = aLease.shaderValue();
        aLease = {};
        REG_CHECK(oldAllocation.expired());
        REG_REQUIRE(registry.storageBuffer(*c).transform([&](auto value) { cLease = std::move(value); }));
        REG_CHECK(cLease.shaderValue() == releasedIndex);
        REG_CHECK(registry.stats().liveDescriptors == 2);

        std::unique_ptr<render::Texture> texture;
        std::unique_ptr<render::TextureView> first, second;
        REG_REQUIRE(device->createTexture({.usage = render::TextureUsageBits::Sampled | render::TextureUsageBits::Storage,
            .format = render::Format::RGBA8Unorm}).transform([&](auto rhiValue) { texture = std::move(rhiValue); }));
        REG_REQUIRE(device->createTextureView(*texture, {}).transform([&](auto rhiValue) { first = std::move(rhiValue); }));
        REG_REQUIRE(device->createTextureView(*texture, {.format = render::Format::RGBA8Unorm}).transform([&](auto rhiValue) { second = std::move(rhiValue); }));
        render::ResourceLease imageA, imageB, generalImage, storageImage;
        bool written = false;
        REG_REQUIRE(registry.sampledImage(*first, render::ResourceState::ShaderRead, &written).transform([&](auto value) { imageA = std::move(value); }));
        REG_CHECK(written);
        REG_REQUIRE(registry.sampledImage(*second, render::ResourceState::ShaderRead, &written).transform([&](auto value) { imageB = std::move(value); }));
        REG_CHECK(!written);
        REG_CHECK(imageA.shaderValue() == imageB.shaderValue());
        REG_REQUIRE(registry.sampledImage(*second, render::ResourceState::General).transform([&](auto value) { generalImage = std::move(value); }));
        REG_CHECK(generalImage.shaderValue() != imageA.shaderValue());
        REG_REQUIRE(registry.storageImage(*second).transform([&](auto value) { storageImage = std::move(value); }));
        REG_CHECK(storageImage.kind() == render::ShaderResourceKind::StorageImage);
        REG_CHECK(!first->hasNativeView() && !second->hasNativeView());
        std::weak_ptr<void> textureAllocation = first->retainTexture();
        texture.reset(); first.reset(); second.reset();
        REG_CHECK(!textureAllocation.expired());
        imageA = {}; imageB = {}; generalImage = {}; storageImage = {};
        REG_CHECK(textureAllocation.expired());
        registry.collect();
        REG_CHECK(registry.stats().liveDescriptors == 2);
        render::ResourceLease samplerA, samplerB;
        REG_REQUIRE(registry.sampler({}).transform([&](auto value) { samplerA = std::move(value); }));
        REG_REQUIRE(registry.sampler({}).transform([&](auto value) { samplerB = std::move(value); }));
        REG_CHECK(samplerA.shaderValue() == samplerB.shaderValue());

        render::ResourceRegistry other;
        REG_REQUIRE(other.initialize(*device, {.maxSamplers = 1, .maxSampledImages = 1,
            .maxStorageImages = 1, .maxBuffers = 1}));
        REG_CHECK(registry.owns(bLease) && !other.owns(bLease) && !registry.owns({}));
        render::RenderFrameContext frame;
        REG_REQUIRE(frame.begin(0));
        render::ParameterWriter writer(*device, frame, other);
        REG_CHECK(render::hasError(writer.use(bLease), render::Error::InvalidArgument));
        const auto invalid = writer.encode(ProbeParams{}, kABI);
        REG_CHECK(render::hasError(invalid, render::Error::InvalidArgument));
        render::ParameterWriter staleWriter(*device, frame, registry);
        const auto encoded = staleWriter.encode(ProbeParams{}, kABI);
        REG_CHECK(encoded && encoded->valid());
        frame.cancel();
        REG_REQUIRE(frame.begin(1));
        const auto stale = staleWriter.encode(ProbeParams{}, kABI);
        REG_CHECK(render::hasError(stale, render::Error::InvalidArgument));
        REG_CHECK(encoded->valid()); // A failed encode cannot overwrite an earlier packet.
        frame.cancel();
        return RHITestResult::pass();
    }
};

class RegistrySubmissionTest : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"parameters.submission.lifetime.readback"}, bench::Layer::Core, "binding", "binding", {"readback.bin"});
    }

    explicit RegistrySubmissionTest(render::ParameterTransport transport = render::ParameterTransport::DeviceAddress)
        : transport_(transport)
    {
        type = RHITestType::Command;
        name = transport == render::ParameterTransport::InlinePush
            ? "registry_inline_submission_lifetime" : "registry_typed_submission_lifetime";
    }
    RHITestResult run(RHITestContext& context) override
    {
        bench::TestDevice device;
        REG_REQUIRE(bench::createTestDevice(context, {.applicationName = "Registry lifetime", .enableValidation = context.enableValidation,
            .enableBindlessDescriptorHeap = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); }));
        auto& queue = *device->getQueue(render::QueueType::Graphics);
        std::shared_ptr<render::ResourceRegistry> registry, sameRegistry;
        REG_REQUIRE(device->resourceRegistry().transform([&](auto rhiValue) { registry = std::move(rhiValue); }));
        REG_REQUIRE(device->resourceRegistry().transform([&](auto rhiValue) { sameRegistry = std::move(rhiValue); }));
        REG_CHECK(registry == sameRegistry);
        render::ComputeKernel firstKernel, secondKernel;
        std::string log;
        REG_REQUIRE(makeKernel(*device, firstKernel, log, render::SlangDescriptorHeapMode::Default, transport_));
        REG_REQUIRE(makeKernel(*device, secondKernel, log, render::SlangDescriptorHeapMode::Default, transport_));
        std::unique_ptr<render::Buffer> source, output;
        REG_REQUIRE(makeBuffer(*device, source, 11));
        REG_REQUIRE(makeBuffer(*device, output));
        std::weak_ptr<void> oldAllocation = source->retainAllocation();
        render::QueueSubmissionTracker tracker;
        REG_REQUIRE(tracker.initialize(*device, queue));
        Commands first, second;
        REG_REQUIRE(first.initialize(*device, queue));
        REG_REQUIRE(second.initialize(*device, queue));
        std::unique_ptr<render::Semaphore> gate;
        REG_REQUIRE(device->createSemaphore().transform([&](auto rhiValue) { gate = std::move(rhiValue); }));
        Drain drain{queue, *gate};
        REG_REQUIRE(first.begin(0));
        render::EncodedParameters stale;
        {
            render::ParameterWriter writer(*device, first.frame, *registry);
            ProbeParams params{writer.buffer(source.get()), writer.buffer(output.get()), 100, 0};
            render::EncodedParameters encoded;
            REG_REQUIRE(writer.encode(params, kABI, transport_).transform([&](auto value) { encoded = std::move(value); }));
            if (transport_ == render::ParameterTransport::InlinePush) {
                REG_CHECK(encoded.address() == 0 && encoded.inlineData().size() == sizeof(params));
                REG_CHECK(registry->stats().parameterBytes == 0 && registry->stats().parameterCapacity == 0);
            }
            stale = encoded;
            REG_REQUIRE(firstKernel.dispatch(*first.commands, encoded, 1));
            params.add = 200; params.index = 1;
            REG_REQUIRE(writer.encode(params, kABI, transport_).transform([&](auto value) { encoded = std::move(value); }));
            render::BufferBarrierDesc barrier{
                .buffer = output.get(),
                .before = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
                .after = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
            };
            if (auto commandResult = first.commands->synchronize({.buffers = {&barrier, 1}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
            REG_REQUIRE(secondKernel.dispatch(*first.commands, encoded, 1));
            // Force the parameter arena to grow without moving already encoded roots.
            std::array<uint32_t, 17000> burst{};
            render::EncodedParameters oversized;
            REG_REQUIRE(writer.encode(burst, kABI + 2).transform([&](auto value) { oversized = std::move(value); }));
            REG_CHECK(oversized.address() != stale.address());
            params.add = 999; // Encoded packets must not reference this mutable CPU struct.
            render::EncodedParameters wrong;
            REG_REQUIRE(writer.encode(params, kABI + 1, transport_).transform([&](auto value) { wrong = std::move(value); }));
            REG_CHECK(render::hasError(firstKernel.dispatch(*first.commands, wrong, 1), render::Error::InvalidArgument));
        }
        source.reset();
        REG_REQUIRE(makeBuffer(*device, source, 22));
        REG_REQUIRE(first.submit(tracker, *gate));
        REG_CHECK(!first.frame.completion().isComplete());
        REG_CHECK(!oldAllocation.expired());
        REG_CHECK(render::hasError(firstKernel.dispatch(*first.commands, stale, 1), render::Error::InvalidArgument));
        REG_REQUIRE(second.begin(1));
        REG_CHECK(render::hasError(secondKernel.dispatch(*second.commands, stale, 1), render::Error::InvalidArgument));
        {
            render::ParameterWriter writer(*device, second.frame, *registry);
            ProbeParams params{writer.buffer(source.get()), writer.buffer(output.get()), 300, 2};
            render::EncodedParameters encoded;
            REG_REQUIRE(writer.encode(params, kABI, transport_).transform([&](auto value) { encoded = std::move(value); }));
            REG_CHECK(transport_ == render::ParameterTransport::InlinePush || encoded.address() != stale.address());
            REG_REQUIRE(secondKernel.dispatch(*second.commands, encoded, 1));
        }
        REG_REQUIRE(second.submit(tracker, *gate));
        firstKernel.clear(); secondKernel.clear(); stale = {};
        REG_CHECK(!oldAllocation.expired());
        REG_REQUIRE(gate->signal(1));
        REG_REQUIRE(second.frame.wait(5'000'000'000ull));
        output->invalidate();
        std::array<uint32_t, 3> values{};
        void* mapped = output->map();
        REG_CHECK(mapped != nullptr);
        std::memcpy(values.data(), mapped, sizeof(values));
        bench::readbackEvidence(context, "readback.bin", std::span<const uint32_t>(values));
        output->unmap();
        REG_CHECK((values == std::array<uint32_t, 3>{111, 211, 322}));
        REG_REQUIRE(first.pool->reset());
        REG_REQUIRE(first.frame.reset());
        REG_CHECK(oldAllocation.expired());
        REG_REQUIRE(second.pool->reset());
        REG_REQUIRE(second.frame.reset());
        registry->collect();
        REG_CHECK(registry->stats().liveDescriptors == 2);
        const uint64_t capacity = registry->stats().parameterCapacity;

        REG_REQUIRE(makeKernel(*device, firstKernel, log, render::SlangDescriptorHeapMode::Default, transport_));
        REG_REQUIRE(first.begin(2));
        std::weak_ptr<void> cancelledAllocation = source->retainAllocation();
        render::EncodedParameters cancelled;
        {
            render::ParameterWriter writer(*device, first.frame, *registry);
            ProbeParams params{writer.buffer(source.get()), writer.buffer(output.get()), 1, 0};
            render::EncodedParameters encoded;
            REG_REQUIRE(writer.encode(params, kABI, transport_).transform([&](auto value) { encoded = std::move(value); }));
            REG_REQUIRE(firstKernel.dispatch(*first.commands, encoded, 1));
            cancelled = encoded;
        }
        source.reset();
        REG_CHECK(!cancelledAllocation.expired());
        REG_REQUIRE(first.commands->end());
        // A frame can still be recording while this command buffer has ended.
        REG_CHECK(render::hasError(firstKernel.dispatch(*first.commands, cancelled, 1), render::Error::InvalidArgument));
        cancelled = {};
        REG_REQUIRE(first.pool->reset());
        first.frame.cancel();
        REG_CHECK(cancelledAllocation.expired());
        REG_CHECK(registry->stats().parameterCapacity == capacity);
        registry->collect();
        REG_CHECK(registry->stats().liveDescriptors == 1);
        return RHITestResult::pass();
    }
private:
    render::ParameterTransport transport_;
};

class RegistryInlineSubmissionTest final : public RegistrySubmissionTest {
public:
    RegistryInlineSubmissionTest() : RegistrySubmissionTest(render::ParameterTransport::InlinePush) {}
};

METALLIC_REGISTER_RHI_TEST(RegistryIdentityTest);
METALLIC_REGISTER_RHI_TEST(RegistrySubmissionTest);
METALLIC_REGISTER_RHI_TEST(RegistryInlineSubmissionTest);

class RegistryPipelinedParametersTest final : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"parameters.pipelined.append.readback"}, bench::Layer::Core, "binding", "binding", {"readback.bin"});
    }

    RegistryPipelinedParametersTest() { type = RHITestType::Command; name = "registry_pipelined_parameter_append"; }
    RHITestResult run(RHITestContext& context) override
    {
        for (auto mode : {render::SlangDescriptorHeapMode::Mapped, render::SlangDescriptorHeapMode::Native}) {
            bench::TestDevice device;
            REG_REQUIRE(bench::createTestDevice(context, {.applicationName = "Pipelined parameters", .enableValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true}).transform([&](auto value) { device = std::move(value); }));
            auto& queue = *device->getQueue(render::QueueType::Graphics);
            std::shared_ptr<render::ResourceRegistry> registry;
            REG_REQUIRE(device->resourceRegistry().transform([&](auto value) { registry = std::move(value); }));
            render::ComputeKernel kernel;
            std::string log;
            REG_REQUIRE(makeKernel(*device, kernel, log, mode));
            std::unique_ptr<render::Buffer> source, output;
            REG_REQUIRE(makeBuffer(*device, source, 11));
            REG_REQUIRE(makeBuffer(*device, output));
            render::RenderFrameContext frame;
            std::array<render::CommandRecordingContext, 3> recordings;
            render::QueueSubmissionTracker tracker;
            std::unique_ptr<render::Semaphore> gate;
            REG_REQUIRE(device->createSemaphore().transform([&](auto value) { gate = std::move(value); }));
            Drain drain{queue, *gate};
            REG_REQUIRE(tracker.initialize(*device, queue));
            REG_REQUIRE(frame.begin(0, UINT64_MAX, render::FrameSubmissionMode::Pipelined));
            render::ParameterWriter writer(*device, frame, *registry);
            ProbeParams params{writer.buffer(source.get()), writer.buffer(output.get()), 0, 0};
            std::array<render::EncodedParameters, 3> packets;
            for (uint32_t i = 0; i < 3; ++i) {
                // Iteration 1 appends while batch 0 is pending behind a gate;
                // iteration 2 appends after prior batches complete, frame open.
                params.add = (i + 1) * 100;
                params.index = i;
                REG_REQUIRE(writer.encode(params, kABI).transform([&](auto value) { packets[i] = std::move(value); }));
                if (i) { REG_CHECK(packets[i].address() > packets[i - 1].address()); }
                REG_REQUIRE(recordings[i].initialize(*device, queue));
                render::CommandBuffer* commands = nullptr;
                REG_REQUIRE(recordings[i].prepare(frame).transform([&](auto value) { commands = value; }));
                REG_REQUIRE(recordings[i].record([&]() -> render::Result<> {
                    render::BufferBarrierDesc barrier{
                        .buffer = output.get(),
                        .before = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
                        .after = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
                    };
                    if (auto commandResult = commands->synchronize({.buffers = {&barrier, 1}}); !commandResult) { return commandResult; }
                    auto result = kernel.dispatch(*commands, packets[i], 1);
                    // Re-read the very first packet after additional uploads.
                    if (result && i == 2) {
                        if (auto commandResult = commands->synchronize({.buffers = {&barrier, 1}}); !commandResult) { return commandResult; }
                        result = kernel.dispatch(*commands, packets[0], 1);
                    }
                    return result ? commands->end() : result;
                }));
                render::RecordedBatch batch;
                REG_REQUIRE(batch.seal(frame, {&commands, 1}));
                render::SemaphoreSubmitDesc wait{.semaphore = gate.get(), .value = 1};
                render::SubmissionReceipt receipt;
                REG_REQUIRE(tracker.submitBatch(batch, {
                    .waitSemaphores = {i == 0 ? &wait : nullptr, i == 0 ? 1u : 0u},
                }, frame).transform([&](auto value) { receipt = std::move(value); }));
                REG_CHECK(receipt.accepted() && frame.recording() && !frame.completion().isSubmitted());
                if (i == 0) { REG_CHECK(!receipt.completion().isComplete()); }
                if (i == 1) {
                    REG_REQUIRE(gate->signal(1));
                    REG_REQUIRE(receipt.completion().wait(5'000'000'000ull));
                    REG_CHECK(!frame.completion().isComplete());
                }
            }
            REG_REQUIRE(frame.sealRecording());
            render::EncodedParameters rejected;
            REG_CHECK(!writer.encode(params, kABI).transform([&](auto value) { rejected = std::move(value); }));
            REG_REQUIRE(frame.finishSubmission());
            REG_REQUIRE(frame.wait(5'000'000'000ull));
            output->invalidate();
            auto* mapped = output->map();
            REG_CHECK(mapped);
            std::array<uint32_t, 3> values;
            std::memcpy(values.data(), mapped, sizeof(values));
            bench::readbackEvidence(context, "readback.bin", std::span<const uint32_t>(values));
            output->unmap();
            REG_CHECK((values == std::array<uint32_t, 3>{111, 211, 311}));
        }
        return RHITestResult::pass();
    }
};
METALLIC_REGISTER_RHI_TEST(RegistryPipelinedParametersTest);

class RegistryPartialSubmissionTest final : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        auto metadata = bench::gpuMetadata({"parameters.partialSubmission.retention.contract"}, bench::Layer::Core, "async", "sync");
        metadata.requirements.queues.push_back(render::QueueType::Copy);
        return metadata;
    }

    RegistryPartialSubmissionTest() { type = RHITestType::Command; name = "registry_partial_multi_queue_retention"; }
    RHITestResult run(RHITestContext& context) override
    {
        bench::TestDevice device;
        REG_REQUIRE(bench::createTestDevice(context, {.applicationName = "Registry partial submission",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); }));
        auto& graphics = *device->getQueue(render::QueueType::Graphics);
        auto* copy = device->getQueue(render::QueueType::Copy);
        if (!copy) { return RHITestResult::skip("Requires a copy queue"); }
        std::shared_ptr<render::ResourceRegistry> registry;
        REG_REQUIRE(device->resourceRegistry().transform([&](auto rhiValue) { registry = std::move(rhiValue); }));
        render::ComputeKernel kernel;
        std::string log;
        REG_REQUIRE(makeKernel(*device, kernel, log));
        std::unique_ptr<render::Buffer> source, output;
        REG_REQUIRE(makeBuffer(*device, source, 17));
        REG_REQUIRE(makeBuffer(*device, output));
        std::weak_ptr<void> allocation = source->retainAllocation();
        auto owner = std::make_shared<uint32_t>(19);
        std::weak_ptr<void> transitiveOwner = owner;
        render::QueueSubmissionTracker graphicsTracker, copyTracker;
        REG_REQUIRE(graphicsTracker.initialize(*device, graphics));
        REG_REQUIRE(copyTracker.initialize(*device, *copy));
        Commands recording;
        REG_REQUIRE(recording.initialize(*device, graphics));
        std::unique_ptr<render::Semaphore> gate;
        REG_REQUIRE(device->createSemaphore().transform([&](auto rhiValue) { gate = std::move(rhiValue); }));
        Drain drain{*copy, *gate};
        REG_REQUIRE(recording.begin(0));
        {
            render::ParameterWriter writer(*device, recording.frame, *registry);
            writer.retain(owner);
            ProbeParams params{writer.buffer(source.get()), writer.buffer(output.get()), 1, 0};
            render::EncodedParameters encoded;
            REG_REQUIRE(writer.encode(params, kABI).transform([&](auto value) { encoded = std::move(value); }));
            REG_REQUIRE(kernel.dispatch(*recording.commands, encoded, 1));
        }
        REG_REQUIRE(recording.commands->end());
        source.reset(); owner.reset(); kernel.clear();
        render::CommandBuffer* buffers[] = {recording.commands.get()};
        render::GPUCompletionPoint graphicsDone, copyDone, rejected;
        REG_REQUIRE(graphicsTracker.submitSegment({.commandBuffers = {buffers, 1}}, recording.frame).transform([&](auto value) { graphicsDone = std::move(value); }));
        render::SemaphoreSubmitDesc wait{.semaphore = gate.get(), .value = 1};
        REG_REQUIRE(copyTracker.submitSegment({.waitSemaphores = {&wait, 1}}, recording.frame).transform([&](auto value) { copyDone = std::move(value); }));
        REG_CHECK(!copyTracker.submitSegment({.commandBuffers = std::array<render::CommandBuffer*, 1>{nullptr}}, recording.frame).transform([&](auto value) { rejected = std::move(value); }));
        recording.frame.cancel(); // Must seal accepted segments, not release their packets.
        REG_REQUIRE(graphicsDone.wait(5'000'000'000ull));
        registry->collect();
        REG_CHECK(!recording.frame.completion().isComplete() && !copyDone.isComplete());
        REG_CHECK(!allocation.expired() && !transitiveOwner.expired());
        REG_REQUIRE(gate->signal(1));
        REG_REQUIRE(recording.frame.wait(5'000'000'000ull));
        REG_REQUIRE(recording.pool->reset());
        REG_REQUIRE(recording.frame.reset());
        REG_CHECK(allocation.expired() && transitiveOwner.expired());
        return RHITestResult::pass();
    }
};
METALLIC_REGISTER_RHI_TEST(RegistryPartialSubmissionTest);

class RegistryTextureSubmissionTest final : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"binding.texture.array.lifetime.readback"}, bench::Layer::Core, "binding", "binding", {"readback.bin"});
    }

    RegistryTextureSubmissionTest() { type = RHITestType::Rendering; name = "registry_texture_array_submission_lifetime"; }
    RHITestResult run(RHITestContext& context) override
    {
        bench::TestDevice device;
        REG_REQUIRE(bench::createTestDevice(context, {.applicationName = "Registry texture array",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); }));
        auto& queue = *device->getQueue(render::QueueType::Graphics);
        std::shared_ptr<render::ResourceRegistry> registry;
        REG_REQUIRE(device->resourceRegistry().transform([&](auto rhiValue) { registry = std::move(rhiValue); }));
        struct Params { render::ShaderStorageImage image; uint64_t samples; render::ShaderBuffer output; };
        static_assert(sizeof(Params) == 24);
        const char* entries[] = {"registryTextureWriteMain", "registryTextureReadMain"};
        std::array<render::ComputeKernel, 2> kernels;
        std::string log;
        for (size_t i = 0; i < kernels.size(); ++i) {
            render::ShaderCompileResult shader;
            REG_REQUIRE(render::compileSlangShaderToSpirv({.moduleName = "RegistryTextureProbe",
                .entryPointName = entries[i], .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, shader.diagnostics).transform([&](auto value) { shader = std::move(value); }));
            REG_REQUIRE(kernels[i].initialize(*device, {.spirv = shader.spirv,
                .parameters = render::parameterAbi<Params>(kABI + 3)}, log));
        }
        std::unique_ptr<render::Texture> image;
        std::unique_ptr<render::TextureView> view;
        std::unique_ptr<render::Buffer> output;
        REG_REQUIRE(makeBuffer(*device, output));
        REG_REQUIRE(device->createTexture({.usage = render::TextureUsageBits::Sampled | render::TextureUsageBits::Storage,
            .format = render::Format::R32Uint}).transform([&](auto rhiValue) { image = std::move(rhiValue); }));
        REG_REQUIRE(device->createTextureView(*image, {}).transform([&](auto rhiValue) { view = std::move(rhiValue); }));
        std::weak_ptr<void> allocation = view->retainTexture();
        render::QueueSubmissionTracker tracker;
        REG_REQUIRE(tracker.initialize(*device, queue));
        Commands recording;
        REG_REQUIRE(recording.initialize(*device, queue));
        std::unique_ptr<render::Semaphore> gate;
        REG_REQUIRE(device->createSemaphore().transform([&](auto rhiValue) { gate = std::move(rhiValue); }));
        Drain drain{queue, *gate};
        REG_REQUIRE(recording.begin(0));
        {
            render::ParameterWriter writer(*device, recording.frame, *registry);
            const std::array<render::TextureView*, 3> views{view.get(), view.get(), view.get()};
            Params params{writer.storageImage(view.get()), writer.sampledImages(views), writer.buffer(output.get())};
            render::EncodedParameters encoded;
            REG_REQUIRE(writer.encode(params, kABI + 3).transform([&](auto value) { encoded = std::move(value); }));
            REG_CHECK(registry->stats().descriptorWrites == 3); // storage image, sampled image, output
            render::TextureBarrierDesc barrier{
                .texture = image.get(),
                .oldLayout = render::TextureLayout::Undefined,
                .newLayout = render::TextureLayout::General,
                .before = {},
                .after = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
            };
            if (auto commandResult = recording.commands->synchronize({.textures = {&barrier, 1}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
            REG_REQUIRE(kernels[0].dispatch(*recording.commands, encoded, 1));
            barrier.oldLayout = render::TextureLayout::General; barrier.before = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite}; barrier.newLayout = render::TextureLayout::ShaderRead; barrier.after = {render::PipelineStageBits::AllCommands, render::AccessBits::ShaderRead};
            if (auto commandResult = recording.commands->synchronize({.textures = {&barrier, 1}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
            REG_REQUIRE(kernels[1].dispatch(*recording.commands, encoded, 1));
        }
        image.reset(); view.reset();
        REG_CHECK(!allocation.expired());
        REG_REQUIRE(recording.submit(tracker, *gate));
        kernels = {};
        REG_REQUIRE(gate->signal(1));
        REG_REQUIRE(recording.frame.wait(5'000'000'000ull));
        output->invalidate();
        void* mapped = output->map();
        REG_CHECK(mapped != nullptr);
        std::array<uint32_t, 3> values;
        std::memcpy(values.data(), mapped, sizeof(values));
        bench::readbackEvidence(context, "readback.bin", std::span<const uint32_t>(values));
        output->unmap();
        REG_CHECK((values == std::array<uint32_t, 3>{1001, 1001, 1001}));
        REG_REQUIRE(recording.pool->reset());
        REG_REQUIRE(recording.frame.reset());
        REG_CHECK(allocation.expired());
        return RHITestResult::pass();
    }
};
METALLIC_REGISTER_RHI_TEST(RegistryTextureSubmissionTest);
// Native provenance and narrowing are checked without creating descriptors.
class BufferSliceValidationTest final : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"bufferSlice.range.provenance.contract"}, bench::Layer::Core, "binding", "binding");
    }

    BufferSliceValidationTest() { type = RHITestType::Resource; name = "buffer_slice_range_and_provenance"; }
    RHITestResult run(RHITestContext& context) override
    {
        bench::TestDevice device, other;
        REG_REQUIRE(bench::createTestDevice(context, {.applicationName = "Buffer slice ranges",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); }));
        REG_REQUIRE(bench::createTestDevice(context, {.applicationName = "Buffer slice foreign source",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}, true).transform([&](auto rhiValue) { other = std::move(rhiValue); }));
        std::unique_ptr<render::Buffer> buffer;
        REG_REQUIRE(makeBuffer(*device, buffer));
        render::BufferSlice parent, child, invalid, empty;
        REG_REQUIRE(buffer->slice({16, 32}).transform([&](auto rhiValue) { parent = std::move(rhiValue); }));
        REG_REQUIRE(parent.subslice({8, 8}).transform([&](auto rhiValue) { child = std::move(rhiValue); }));
        REG_CHECK(child.offset() == 24 && child.size() == 8);
        REG_CHECK(child.deviceAddress() == buffer->deviceAddress() + 24);
        REG_CHECK(child.deviceIdentity() == device->identity());
        REG_CHECK(child.allocationIdentity() == buffer->retainAllocation().get());
        REG_REQUIRE(child.validateData(device->identity(), 4, 4));
        REG_CHECK(render::hasError(child.validateData(other->identity(), 4, 4), render::Error::InvalidArgument));
        REG_REQUIRE(parent.subslice({32}).transform([&](auto value) { empty = std::move(value); }));
        REG_CHECK(empty.valid() && empty.size() == 0);
        REG_CHECK(!empty.validateData(device->identity(), 4, 4));
        for (uint64_t offset : {uint64_t(33), UINT64_MAX}) {
            const auto rejected = parent.subslice({offset});
            REG_CHECK(render::hasError(rejected, render::Error::InvalidArgument));
            REG_CHECK(parent.offset() == 16 && parent.size() == 32);
        }
        REG_CHECK(render::hasError(parent.subslice({0, 33}), render::Error::InvalidArgument));
        REG_CHECK(render::hasError(parent.subslice({31, UINT64_MAX - 1}), render::Error::InvalidArgument));
        REG_REQUIRE(parent.subslice({1, 8}).transform([&](auto rhiValue) { invalid = std::move(rhiValue); }));
        REG_CHECK(!invalid.validateData(device->identity(), 4, 4));
        REG_REQUIRE(parent.subslice({0, 12}).transform([&](auto rhiValue) { invalid = std::move(rhiValue); }));
        REG_CHECK(!invalid.validateData(device->identity(), 8, 4));
        REG_CHECK(!child.validateData(device->identity(), 0, 4));
        REG_CHECK(!child.validateData(device->identity(), 4, 0));
        REG_CHECK(!child.validateData(device->identity(), 4, 3));
        REG_CHECK(!child.validateData(device->identity(), 3, 4));
        REG_CHECK(!child.validate(device->identity(), render::BufferUsageBits::Storage | render::BufferUsageBits::TransferSource));
        REG_REQUIRE(parent.subslice({8, 8}).transform([&](auto rhiValue) { parent = std::move(rhiValue); }));
        REG_CHECK(parent.offset() == child.offset() && parent.size() == child.size());

        std::weak_ptr<void> allocation = buffer->retainAllocation();
        const auto address = child.deviceAddress();
        render::Buffer moved = std::move(*buffer);
        REG_CHECK(!buffer->retainAllocation());
        REG_REQUIRE(makeBuffer(*device, buffer, 99));
        moved = {};
        REG_CHECK(!allocation.expired() && child.deviceAddress() == address);
        parent = {}; invalid = {}; empty = {};
        std::shared_ptr<render::ResourceRegistry> registry;
        REG_REQUIRE(device->resourceRegistry().transform([&](auto rhiValue) { registry = std::move(rhiValue); }));
        render::RenderFrameContext frame;
        REG_REQUIRE(frame.begin(0));
        render::EncodedParameters packet;
        {
            render::ParameterWriter invalidWriter(*other, frame, *registry);
            invalidWriter.dataBuffer<uint32_t>(child);
            REG_CHECK(!invalidWriter.status());
            REG_CHECK(!invalidWriter.encode(render::ShaderDataSpan{}, kABI + 4).transform([&](auto value) { packet = std::move(value); }) && !packet.valid());
            render::ParameterWriter writer(*device, frame, *registry);
            const auto data = writer.dataBuffer<uint32_t>(child);
            REG_CHECK(data.address == address && data.count == 2 && data.stride == 4);
            REG_REQUIRE(writer.encode(data, kABI + 4).transform([&](auto value) { packet = std::move(value); }));
        }
        child = {};
        REG_CHECK(!allocation.expired());
        packet = {};
        frame.cancel();
        REG_REQUIRE(frame.reset());
        REG_CHECK(allocation.expired());
        REG_CHECK(registry->stats().descriptorWrites == 0);
        return RHITestResult::pass();
    }
};
METALLIC_REGISTER_RHI_TEST(BufferSliceValidationTest);

class BufferSliceSubmissionTest final : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"bufferSlice.bda.copy.indirect.readback"}, bench::Layer::Core, "binding", "binding", {"readback.bin"});
    }

    BufferSliceSubmissionTest() { type = RHITestType::Rendering; name = "buffer_slice_bda_copy_compute_indirect_lifetime"; }
    RHITestResult run(RHITestContext& context) override
    {
        bench::TestDevice device;
        REG_REQUIRE(bench::createTestDevice(context, {.applicationName = "Buffer slice data chain",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); }));
        auto& queue = *device->getQueue(render::QueueType::Graphics);
        std::shared_ptr<render::ResourceRegistry> registry;
        REG_REQUIRE(device->resourceRegistry().transform([&](auto rhiValue) { registry = std::move(rhiValue); }));
        struct Params { render::ShaderDataSpan source, output, arguments; uint32_t add; };
        static_assert(sizeof(Params) == 56 && offsetof(Params, add) == 48);
        std::array<render::ComputeKernel, 2> kernels;
        const char* entries[] = {"dataProduceMain", "dataIndirectMain"};
        std::string log;
        for (size_t i = 0; i < kernels.size(); ++i) {
            render::ShaderCompileResult shader;
            auto result = render::compileSlangShaderToSpirv({.moduleName = "DataSliceProbe",
                .entryPointName = entries[i], .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, shader.diagnostics).transform([&](auto value) { shader = std::move(value); });
            if (!result) { return RHITestResult::fail(shader.diagnostics); }
            REG_REQUIRE(kernels[i].initialize(*device, {.spirv = shader.spirv,
                .parameters = render::parameterAbi<Params>(kABI + 5)}, log));
        }
        render::ShaderCompileResult shader;
        REG_REQUIRE(render::compileSlangShaderToSpirv({.moduleName = "DataSliceProbe",
            .entryPointName = "dataAdapterMain", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, shader.diagnostics).transform([&](auto value) { shader = std::move(value); }));
        render::ComputeProgram adapter;
        const render::ComputeProgramBindingDesc layout{.binding = 0,
            .kind = render::ComputeResourceBindingKind::DataBuffer, .dataStride = 4, .dataAlignment = 4};
        REG_REQUIRE(adapter.initialize(*device, {
            .spirv = shader.spirv,
            .bindings = {&layout, 1},
            .requiresRayQuery = false,
        }, log));
        std::unique_ptr<render::Buffer> source, work, output;
        REG_REQUIRE(device->createBuffer({.size = 64, .usage = render::BufferUsageBits::TransferSource,
            .memoryLocation = render::MemoryLocation::HostUpload}).transform([&](auto rhiValue) { source = std::move(rhiValue); }));
        REG_REQUIRE(device->createBuffer({.size = 64, .usage = render::BufferUsageBits::ShaderDeviceAddress |
            render::BufferUsageBits::TransferDestination | render::BufferUsageBits::Indirect}).transform([&](auto rhiValue) { work = std::move(rhiValue); }));
        REG_REQUIRE(device->createBuffer({.size = 64, .usage = render::BufferUsageBits::ShaderDeviceAddress |
            render::BufferUsageBits::TransferSource | render::BufferUsageBits::TransferDestination,
            .memoryLocation = render::MemoryLocation::HostReadback}).transform([&](auto rhiValue) { output = std::move(rhiValue); }));
        auto* sourceWords = static_cast<uint32_t*>(source->map());
        REG_CHECK(sourceWords);
        for (uint32_t i = 0; i < 16; ++i) { sourceWords[i] = 100 + i; }
        source->flush(); source->unmap();
        auto* outputWords = static_cast<uint32_t*>(output->map());
        REG_CHECK(outputWords);
        for (uint32_t i = 0; i < 16; ++i) { outputWords[i] = 0xdeadbeef; }
        output->flush(); output->unmap();
        std::weak_ptr<void> sourceAllocation = source->retainAllocation(), workAllocation = work->retainAllocation();
        render::QueueSubmissionTracker tracker;
        REG_REQUIRE(tracker.initialize(*device, queue));
        Commands recording;
        REG_REQUIRE(recording.initialize(*device, queue));
        std::unique_ptr<render::Semaphore> gate;
        REG_REQUIRE(device->createSemaphore().transform([&](auto rhiValue) { gate = std::move(rhiValue); }));
        Drain drain{queue, *gate};
        REG_REQUIRE(recording.begin(0));
        {
            render::BufferSlice from, data, to, arguments, invalid;
            REG_REQUIRE(source->slice({8, 16}).transform([&](auto rhiValue) { from = std::move(rhiValue); }));
            REG_REQUIRE(work->slice({16, 16}).transform([&](auto rhiValue) { data = std::move(rhiValue); }));
            REG_REQUIRE(work->slice({48, 12}).transform([&](auto rhiValue) { arguments = std::move(rhiValue); }));
            REG_REQUIRE(output->slice({20, 16}).transform([&](auto rhiValue) { to = std::move(rhiValue); }));
            // Transfer-only memory is not a shader data buffer.
            REG_CHECK(!from.validateData(device->identity(), 4, 4));
            REG_REQUIRE(to.subslice({0, 12}).transform([&](auto rhiValue) { invalid = std::move(rhiValue); }));
            REG_CHECK(!recording.commands->copyBuffer(from, invalid));
            REG_CHECK(!recording.commands->copyBuffer(to, to));
            REG_CHECK(!recording.commands->dispatchIndirect(to));
            REG_REQUIRE(arguments.subslice({1, 8}).transform([&](auto rhiValue) { invalid = std::move(rhiValue); }));
            REG_CHECK(!recording.commands->dispatchIndirect(invalid));
            render::BufferBarrierDesc workBarrier{
                .buffer = work.get(),
                .before = {},
                .after = {render::PipelineStageBits::Transfer, render::AccessBits::TransferWrite},
            };
            if (auto commandResult = recording.commands->synchronize({.buffers = {&workBarrier, 1}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
            REG_REQUIRE(recording.commands->copyBuffer(from, data));
            workBarrier.before = {render::PipelineStageBits::Transfer, render::AccessBits::TransferWrite}; workBarrier.after = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite};
            if (auto commandResult = recording.commands->synchronize({.buffers = {&workBarrier, 1}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
            render::BufferBarrierDesc outputBarrier{
                .buffer = output.get(),
                .before = {},
                .after = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
            };
            if (auto commandResult = recording.commands->synchronize({.buffers = {&outputBarrier, 1}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
            render::ParameterWriter writer(*device, recording.frame, *registry);
            const Params params{writer.dataBuffer<uint32_t>(data), writer.dataBuffer<uint32_t>(to),
                writer.dataBuffer<uint32_t>(arguments), 7};
            render::EncodedParameters encoded;
            REG_REQUIRE(writer.encode(params, kABI + 5).transform([&](auto value) { encoded = std::move(value); }));
            REG_REQUIRE(kernels[0].dispatch(*recording.commands, encoded, 1));
            outputBarrier.before = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite};
            workBarrier.before = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite}; workBarrier.after = {render::PipelineStageBits::DrawIndirect, render::AccessBits::IndirectRead};
            const render::BufferBarrierDesc barriers[] = {outputBarrier, workBarrier};
            if (auto commandResult = recording.commands->synchronize({.buffers = {barriers, 2}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
            REG_REQUIRE(kernels[1].dispatchIndirect(*recording.commands, encoded, arguments));
            if (auto commandResult = recording.commands->synchronize({.buffers = {&outputBarrier, 1}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
            render::ComputeDispatchBinding binding{.binding = 0, .data = to};
            render::ComputeDispatchDesc dispatch{.commandBuffer = recording.commands.get(), .bindings = {&binding, 1}};
            binding.range.offset = 4;
            REG_CHECK(render::hasError(adapter.dispatch(dispatch), render::Error::InvalidArgument));
            binding.range.offset = 0;
            REG_REQUIRE(adapter.dispatch(dispatch));
        }
        source.reset(); work.reset();
        REG_CHECK(!sourceAllocation.expired() && !workAllocation.expired());
        REG_CHECK(registry->stats().descriptorWrites == 0 && registry->stats().liveDescriptors == 0);
        REG_REQUIRE(recording.submit(tracker, *gate));
        kernels = {}; adapter.clear();
        REG_CHECK(!recording.frame.completion().isComplete());
        REG_CHECK(!sourceAllocation.expired() && !workAllocation.expired());
        REG_REQUIRE(gate->signal(1));
        REG_REQUIRE(recording.frame.wait(5'000'000'000ull));
        output->invalidate();
        outputWords = static_cast<uint32_t*>(output->map());
        REG_CHECK(outputWords);
        std::array<uint32_t, 16> values;
        std::memcpy(values.data(), outputWords, sizeof(values));
        bench::readbackEvidence(context, "readback.bin", std::span<const uint32_t>(values));
        output->unmap();
        for (uint32_t i = 0; i < values.size(); ++i) {
            const auto expected = i >= 5 && i < 9 ? 221 + (i - 5) * 2 : 0xdeadbeef;
            REG_CHECK(values[i] == expected);
        }
        REG_REQUIRE(recording.pool->reset());
        REG_REQUIRE(recording.frame.reset());
        REG_CHECK(sourceAllocation.expired() && workAllocation.expired());
        return RHITestResult::pass();
    }
};
METALLIC_REGISTER_RHI_TEST(BufferSliceSubmissionTest);

class UpscalerGuideInlineTest final : public RHITest {
public:
    UpscalerGuideInlineTest() { type = RHITestType::Resource; name = "upscaler_guide_inline_output"; }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        bench::TestDevice device;
        REG_REQUIRE(bench::createTestDevice(context, {.applicationName = "Inline guide resolve",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true})
            .transform([&](auto value) { device = std::move(value); }));
        auto& queue = *device->getQueue(QueueType::Graphics);
        auto registry = device->resourceRegistry(); REG_CHECK(registry);
        ComputeKernel fill, resolve;
        std::string log;
        auto shader = compileSlangShaderToSpirv({.moduleName = "UpscalerGuideProbe", .entryPointName = "guideFillMain",
            .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, log);
        if (!shader) { return RHITestResult::fail(log); }
        const auto abi = parameterAbi<UpscalerGuideResolveParams>(kUpscalerGuideResolveABI, ParameterTransport::InlinePush);
        REG_REQUIRE(fill.initialize(*device, {.spirv = shader->spirv, .parameters = abi}, log));
        shader = compileSlangShaderToSpirv({.moduleName = "Features/PostProcess/UpscalerGuideResolve",
            .entryPointName = "upscalerGuideResolveMain", .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, log);
        if (!shader) { return RHITestResult::fail(log); }
        REG_REQUIRE(resolve.initialize(*device, {.spirv = shader->spirv, .parameters = abi}, log));
        std::array<std::unique_ptr<Texture>, 4> textures;
        std::array<std::unique_ptr<TextureView>, 4> views;
        for (uint32_t i = 0; i < 4; ++i) {
            REG_REQUIRE(device->createTexture({.usage = TextureUsageBits::Sampled | TextureUsageBits::Storage | TextureUsageBits::TransferSource,
                .format = i % 2 == 0 ? Format::R32Sfloat : Format::RG32Sfloat, .width = i < 2 ? 2u : 3u, .height = i < 2 ? 2u : 3u})
                .transform([&](auto value) { textures[i] = std::move(value); }));
            REG_REQUIRE(device->createTextureView(*textures[i], {}).transform([&](auto value) { views[i] = std::move(value); }));
        }
        std::array<std::unique_ptr<Buffer>, 2> readbacks;
        for (uint32_t i = 0; i < 2; ++i) {
            REG_REQUIRE(device->createBuffer({.size = 9u * (i + 1) * sizeof(float), .usage = BufferUsageBits::TransferDestination,
                .memoryLocation = MemoryLocation::HostReadback}).transform([&](auto value) { readbacks[i] = std::move(value); }));
        }
        QueueSubmissionTracker tracker; REG_REQUIRE(tracker.initialize(*device, queue));
        Commands recording; REG_REQUIRE(recording.initialize(*device, queue)); REG_REQUIRE(recording.begin(0));
        for (const auto& texture : textures) {
            TextureBarrierDesc barrier{.texture = texture.get(), .oldLayout = TextureLayout::Undefined, .newLayout = TextureLayout::General,
                .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite}};
            REG_REQUIRE(recording.commands->synchronize({.textures = {&barrier, 1}}));
        }
        {
            ParameterWriter writer(*device, **registry, &recording.frame);
            UpscalerGuideResolveParams params{};
            params.outputDepth = writer.storageImage(views[0].get()); params.outputMotion = writer.storageImage(views[1].get());
            auto encoded = writer.encode(params, kUpscalerGuideResolveABI, ParameterTransport::InlinePush); REG_CHECK(encoded);
            REG_REQUIRE(fill.dispatch(*recording.commands, *encoded, 1));
            for (uint32_t i = 0; i < 2; ++i) {
                TextureBarrierDesc barrier{.texture = textures[i].get(), .oldLayout = TextureLayout::General, .newLayout = TextureLayout::ShaderRead,
                    .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite}, .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead}};
                REG_REQUIRE(recording.commands->synchronize({.textures = {&barrier, 1}}));
            }
            params.depth = writer.sampledImage(views[0].get()); params.motion = writer.sampledImage(views[1].get());
            params.outputDepth = writer.storageImage(views[2].get()); params.outputMotion = writer.storageImage(views[3].get());
            params.jitterX = 0.25f; params.jitterY = -0.25f;
            encoded = writer.encode(params, kUpscalerGuideResolveABI, ParameterTransport::InlinePush); REG_CHECK(encoded);
            REG_REQUIRE(resolve.dispatch(*recording.commands, *encoded, 1));
        }
        for (uint32_t i = 0; i < 2; ++i) {
            TextureBarrierDesc barrier{.texture = textures[i + 2].get(), .oldLayout = TextureLayout::General, .newLayout = TextureLayout::TransferSource,
                .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite}, .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}};
            REG_REQUIRE(recording.commands->synchronize({.textures = {&barrier, 1}}));
            recording.commands->copyTextureToBuffer({.texture = textures[i + 2].get(), .buffer = readbacks[i].get(), .width = 3, .height = 3});
        }
        const MemoryBarrierDesc host{{PipelineStageBits::Transfer, AccessBits::TransferWrite}, {PipelineStageBits::Host, AccessBits::HostRead}};
        REG_REQUIRE(recording.commands->synchronize({.memory = {&host, 1}}));
        std::unique_ptr<Semaphore> gate; REG_REQUIRE(device->createSemaphore().transform([&](auto value) { gate = std::move(value); }));
        Drain drain{queue, *gate}; REG_REQUIRE(recording.submit(tracker, *gate)); REG_REQUIRE(gate->signal(1));
        REG_REQUIRE(recording.frame.wait(5'000'000'000ull));
        for (uint32_t i = 0; i < 2; ++i) {
            readbacks[i]->invalidate(); const auto* data = static_cast<const float*>(readbacks[i]->map()); REG_CHECK(data);
            std::array<float, 18> values{}; std::memcpy(values.data(), data, 9 * (i + 1) * sizeof(float)); readbacks[i]->unmap();
            for (uint32_t y = 0; y < 3; ++y) { for (uint32_t x = 0; x < 3; ++x) {
                const bool foreground = x > 0;
                const uint32_t pixel = y * 3 + x;
                if (i == 0) { REG_CHECK(values[pixel] == (foreground ? 0.25f : 0.75f)); }
                else {
                    REG_CHECK(values[pixel * 2] == (foreground ? 0.125f : 0.0f));
                    REG_CHECK(values[pixel * 2 + 1] == ((foreground || y == 2) ? 0.125f : 0.0f));
                }
            }}
        }
        return RHITestResult::pass("Inline handles/jitter, nearest foreground depth, UV motion and partial workgroup");
    }
};
METALLIC_REGISTER_RHI_TEST(UpscalerGuideInlineTest);

class StreamCandidateLayoutTest final : public PostProcessParameterLayoutTest {
public:
    StreamCandidateLayoutTest() { category = 25; name = "stream_candidate_parameter_spirv_layout"; }
};
METALLIC_REGISTER_RHI_TEST(StreamCandidateLayoutTest);

class StreamClassifyLayoutTest final : public PostProcessParameterLayoutTest {
public:
    StreamClassifyLayoutTest() { category = 26; name = "stream_classify_parameter_spirv_layout"; }
};
METALLIC_REGISTER_RHI_TEST(StreamClassifyLayoutTest);
class StreamClusterCullLayoutTest final : public PostProcessParameterLayoutTest {
public:
    StreamClusterCullLayoutTest() { category = 27; name = "stream_cluster_cull_parameter_spirv_layout"; }
};
METALLIC_REGISTER_RHI_TEST(StreamClusterCullLayoutTest);
class HybridBinLayoutTest final : public PostProcessParameterLayoutTest {
public:
    HybridBinLayoutTest() { category = 28; name = "hybrid_bin_parameter_spirv_layout"; }
};
METALLIC_REGISTER_RHI_TEST(HybridBinLayoutTest);
class HybridRasterLayoutTest final : public PostProcessParameterLayoutTest {
public:
    HybridRasterLayoutTest() { category = 29; name = "hybrid_raster_parameter_spirv_layout"; }
};
METALLIC_REGISTER_RHI_TEST(HybridRasterLayoutTest);
class HybridResolveLayoutTest final : public PostProcessParameterLayoutTest {
public:
    HybridResolveLayoutTest() { category = 30; name = "hybrid_resolve_parameter_spirv_layout"; }
};
METALLIC_REGISTER_RHI_TEST(HybridResolveLayoutTest);
class StreamWorkloadLayoutTest final : public PostProcessParameterLayoutTest {
public:
    StreamWorkloadLayoutTest() { category = 32; name = "stream_workload_parameter_spirv_layout"; }
};
METALLIC_REGISTER_RHI_TEST(StreamWorkloadLayoutTest);
class ShadowTraceLayoutTest final : public PostProcessParameterLayoutTest {
public:
    ShadowTraceLayoutTest() { category = 33; name = "shadow_trace_parameter_spirv_layout"; }
};
METALLIC_REGISTER_RHI_TEST(ShadowTraceLayoutTest);
class RTXDITraceLayoutTest final : public PostProcessParameterLayoutTest {
public:
    RTXDITraceLayoutTest() { category = 34; name = "rtxdi_trace_parameter_spirv_layout"; }
};
METALLIC_REGISTER_RHI_TEST(RTXDITraceLayoutTest);
class PathTraceInlineLayoutTest final : public PostProcessParameterLayoutTest {
public:
    PathTraceInlineLayoutTest() { category = 35; name = "path_trace_inline_parameter_spirv_layout"; }
};
METALLIC_REGISTER_RHI_TEST(PathTraceInlineLayoutTest);
class SharcTraceLayoutTest final : public PostProcessParameterLayoutTest {
public:
    SharcTraceLayoutTest() { category = 36; name = "sharc_trace_parameter_spirv_layout"; }
};
METALLIC_REGISTER_RHI_TEST(SharcTraceLayoutTest);
class NRCTraceLayoutTest final : public PostProcessParameterLayoutTest {
public:
    NRCTraceLayoutTest() { category = 37; name = "nrc_trace_parameter_spirv_layout"; }
};
METALLIC_REGISTER_RHI_TEST(NRCTraceLayoutTest);
class PathTraceGuidesInlineLayoutTest final : public PostProcessParameterLayoutTest {
public:
    PathTraceGuidesInlineLayoutTest() { category = 38; name = "path_trace_guides_inline_parameter_spirv_layout"; }
};
METALLIC_REGISTER_RHI_TEST(PathTraceGuidesInlineLayoutTest);
class RealtimeLightingLayoutTest final : public PostProcessParameterLayoutTest {
public:
    RealtimeLightingLayoutTest() { category = 39; name = "realtime_lighting_parameter_spirv_layout"; }
};
METALLIC_REGISTER_RHI_TEST(RealtimeLightingLayoutTest);
class DeferredShadingLayoutTest final : public PostProcessParameterLayoutTest {
public:
    DeferredShadingLayoutTest() { category = 40; name = "deferred_shading_parameter_spirv_layout"; }
};
METALLIC_REGISTER_RHI_TEST(DeferredShadingLayoutTest);








class StreamRasterLayoutTest final : public PostProcessParameterLayoutTest {
public:
    StreamRasterLayoutTest() { category = 31; name = "stream_raster_parameter_spirv_layout"; }
};
METALLIC_REGISTER_RHI_TEST(StreamRasterLayoutTest);







class StreamBLASLayoutTest final : public PostProcessParameterLayoutTest {
public:
    StreamBLASLayoutTest() { category = 24; name = "stream_blas_parameter_spirv_layout"; }
};
METALLIC_REGISTER_RHI_TEST(StreamBLASLayoutTest);

class StreamTLASLayoutTest final : public PostProcessParameterLayoutTest {
public:
    StreamTLASLayoutTest() { category = 23; name = "stream_tlas_parameter_spirv_layout"; }
};
METALLIC_REGISTER_RHI_TEST(StreamTLASLayoutTest);

class StreamTLASInputTest final : public RHITest {
public:
    StreamTLASInputTest() { type = RHITestType::Resource; name = "stream_tlas_inline_inputs"; }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        bench::TestDevice device;
        REG_REQUIRE(bench::createTestDevice(context, {.applicationName = "Stream TLAS inputs",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true})
            .transform([&](auto value) { device = std::move(value); }));
        auto& queue = *device->getQueue(QueueType::Graphics);
        auto registry = device->resourceRegistry();
        REG_CHECK(registry);
        ComputeKernel kernel;
        std::string log;
        auto shader = compileSlangShaderToSpirv({.moduleName = kMeshletStreamShaderModuleName,
            .entryPointName = kMeshletStreamTLASInputEntryPoint, .searchPath = kMeshletStreamShaderSearchPath}, log);
        if (!shader) { return RHITestResult::fail(log); }
        REG_REQUIRE(kernel.initialize(*device, {.spirv = shader->spirv,
            .parameters = parameterAbi<StreamTLASParameters>(kStreamTLASABI, ParameterTransport::InlinePush)}, log));
        std::unique_ptr<Buffer> output;
        REG_REQUIRE(device->createBuffer({.size = 8 * 64, .structureStride = 64,
            .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostReadback})
            .transform([&](auto value) { output = std::move(value); }));
        auto* initial = output->map();
        REG_CHECK(initial);
        std::memset(initial, 0xcd, 8 * 64);
        output->flush(); output->unmap();
        QueueSubmissionTracker tracker;
        REG_REQUIRE(tracker.initialize(*device, queue));
        Commands recording;
        REG_REQUIRE(recording.initialize(*device, queue));
        REG_REQUIRE(recording.begin(0));
        {
            MeshletStreamGPUParams settings{};
            settings.sceneInstanceCount = 7; settings.scenePrimitiveCount = 1;
            settings.blasStorageAddressLow = 0xfffffff0u; settings.blasStorageAddressHigh = 4;
            std::array<MeshletStreamGPUInstance, 7> instances{};
            for (auto& instance : instances) {
                instance.visible = 1;
                for (uint32_t component = 0; component < 4; ++component) {
                    instance.world0[component] = float(component + 1);
                    instance.world1[component] = float(component + 5);
                    instance.world2[component] = float(component + 9);
                    instance.world3[component] = float(component + 13);
                }
            }
            instances[2].visible = 0;
            instances[4].primitiveIndex = 99;
            std::array<MeshletStreamGPUInstanceBLAS, 6> records{};
            for (auto& record : records) {
                record.flags = 2; record.selectedClusterCount = record.insertedClusterCount = 1;
                record.cachedValid = 1; record.storageCapacity = 256; record.storageOffset = 32;
            }
            records[1].flags = 1; // Explicit fallback.
            records[3].flags = 6; // Overflow invalidates an otherwise ready dynamic BLAS.
            records[4].flags = 0; // Invalid primitive must not read fallback memory.
            records[5].insertedClusterCount = 0; // Incomplete dynamic build uses fallback.
            const uint64_t fallback = 0x1122334455667788ull;
            ParameterWriter writer(*device, **registry, &recording.frame);
            StreamTLASParameters params{
                .settings = {writer.data(&settings, sizeof(settings), 16), 1, sizeof(settings)},
                .instances = {writer.data(instances.data(), sizeof(instances), 16), 7, sizeof(instances[0])},
                .blasRecords = {writer.data(records.data(), sizeof(records), 16), 6, sizeof(records[0])},
                .fallbackAddresses = {writer.data(&fallback, sizeof(fallback), 8), 1, 8},
                .output = writer.dataBuffer(output.get(), 64, 16),
            };
            auto encoded = writer.encode(params, kStreamTLASABI, ParameterTransport::InlinePush);
            REG_CHECK(encoded);
            REG_REQUIRE(kernel.dispatch(*recording.commands, *encoded, 1));
        }
        const MemoryBarrierDesc hostRead{{PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
            {PipelineStageBits::Host, AccessBits::HostRead}};
        REG_REQUIRE(recording.commands->synchronize({.memory = {&hostRead, 1}}));
        std::unique_ptr<Semaphore> gate;
        REG_REQUIRE(device->createSemaphore().transform([&](auto value) { gate = std::move(value); }));
        Drain drain{queue, *gate};
        REG_REQUIRE(recording.submit(tracker, *gate));
        REG_REQUIRE(gate->signal(1));
        REG_REQUIRE(recording.frame.wait(5'000'000'000ull));
        output->invalidate();
        const auto* mapped = static_cast<const uint32_t*>(output->map());
        REG_CHECK(mapped);
        std::array<uint32_t, 128> actual;
        std::memcpy(actual.data(), mapped, sizeof(actual)); output->unmap();
        for (uint32_t i = 0; i < 6; ++i) {
            for (uint32_t row = 0; row < 3; ++row) {
                for (uint32_t column = 0; column < 4; ++column) {
                    const float expected = float(row + column * 4 + 1);
                    uint32_t bits; std::memcpy(&bits, &expected, sizeof(bits));
                    REG_CHECK(actual[i * 16 + row * 4 + column] == bits);
                }
            }
            REG_CHECK(actual[i * 16 + 12] == (i | ((i == 2 || i == 4) ? 0u : 0xff000000u)));
            REG_CHECK(actual[i * 16 + 13] == 0);
            REG_CHECK(actual[i * 16 + 14] == (i == 0 || i == 2 ? 16u : i == 4 ? 0u : 0x55667788u));
            REG_CHECK(actual[i * 16 + 15] == (i == 0 || i == 2 ? 5u : i == 4 ? 0u : 0x11223344u));
        }
        for (uint32_t i = 6 * 16; i < actual.size(); ++i) { REG_CHECK(actual[i] == 0xcdcdcdcdu); }
        REG_CHECK((*registry)->stats().descriptorWrites == 0);
        return RHITestResult::pass("TLAS transforms, address carry, dynamic/fallback/hidden/overflow/incomplete cases, BDA bounds and submission lifetime; zero descriptors");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamTLASInputTest);

class StreamActiveBuildLayoutTest final : public PostProcessParameterLayoutTest {
public:
    StreamActiveBuildLayoutTest() { category = 22; name = "stream_active_build_parameter_spirv_layout"; }
};
METALLIC_REGISTER_RHI_TEST(StreamActiveBuildLayoutTest);

class StreamTraversalLayoutTest final : public PostProcessParameterLayoutTest {
public:
    StreamTraversalLayoutTest() { category = 21; name = "stream_traversal_parameter_spirv_layout"; }
};
METALLIC_REGISTER_RHI_TEST(StreamTraversalLayoutTest);

class StreamPageTableLayoutTest final : public PostProcessParameterLayoutTest {
public:
    StreamPageTableLayoutTest() { category = 20; name = "stream_page_table_parameter_spirv_layout"; }
};
METALLIC_REGISTER_RHI_TEST(StreamPageTableLayoutTest);

class StreamPageTableInlineTest final : public RHITest {
public:
    StreamPageTableInlineTest() { type = RHITestType::Resource; name = "stream_page_table_inline_snapshots"; }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        bench::TestDevice device;
        REG_REQUIRE(bench::createTestDevice(context, {.applicationName = "Stream page snapshots",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true})
            .transform([&](auto value) { device = std::move(value); }));
        auto& queue = *device->getQueue(QueueType::Graphics);
        auto registry = device->resourceRegistry();
        REG_CHECK(registry);
        std::array<ComputeKernel, 2> kernels;
        const char* entries[] = {"gpuDrivenStreamAssetInitializePageTableMain", "gpuDrivenStreamAssetApplyUpdatesMain"};
        std::string log;
        for (size_t i = 0; i < kernels.size(); ++i) {
            auto shader = compileSlangShaderToSpirv({.moduleName = "Features/GPUDriven/GPUDrivenStreamAsset",
                .entryPointName = entries[i], .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, log);
            if (!shader) { return RHITestResult::fail(log); }
            REG_REQUIRE(kernels[i].initialize(*device, {.spirv = shader->spirv,
                .parameters = parameterAbi<StreamPageTableParameters>(kStreamPageTableABI, ParameterTransport::InlinePush)}, log));
        }
        std::array<std::unique_ptr<Buffer>, 2> outputs;
        for (size_t i = 0; i < outputs.size(); ++i) {
            REG_REQUIRE(makeBuffer(*device, outputs[i]));
            auto* words = static_cast<uint32_t*>(outputs[i]->map());
            REG_CHECK(words);
            for (uint32_t page = 0; page < 8; ++page) {
                words[page * 2] = 0xffffffffu;
                words[page * 2 + 1] = 73u + page;
            }
            outputs[i]->flush();
            outputs[i]->unmap();
        }
        QueueSubmissionTracker tracker;
        REG_REQUIRE(tracker.initialize(*device, queue));
        Commands recording;
        REG_REQUIRE(recording.initialize(*device, queue));
        REG_REQUIRE(recording.begin(0));
        const MemoryBarrierDesc compute{{PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
            {PipelineStageBits::ComputeShader, AccessBits::ShaderRead | AccessBits::ShaderWrite}};
        for (uint32_t stage = 0; stage < 4; ++stage) {
            ParameterWriter writer(*device, **registry, &recording.frame);
            // Reuse and overwrite host storage after encoding; each packet must own its bytes.
            std::array<std::array<uint32_t, 2>, 3> patches{{{2u, stage == 1 ? 17u : 0u}, {5u, 29u}, {99u, 91u}}};
            StreamPageTableParameters params{.pages = writer.dataBuffer(outputs[stage == 0 ? 0 : 1].get(), 8, 8)};
            if (stage != 0) {
                params.patches = {writer.data(patches.data(), sizeof(patches), 8), stage == 2 ? 1u : 3u, 8};
                if (stage == 3) { params.patches.count = 0; }
            }
            auto encoded = writer.encode(params, kStreamPageTableABI, ParameterTransport::InlinePush);
            REG_CHECK(encoded);
            patches = {};
            REG_REQUIRE(kernels[stage == 0 ? 0 : 1].dispatch(*recording.commands, *encoded, 1));
            REG_REQUIRE(recording.commands->synchronize({.memory = {&compute, 1}}));
        }
        const MemoryBarrierDesc hostRead{{PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
            {PipelineStageBits::Host, AccessBits::HostRead}};
        REG_REQUIRE(recording.commands->synchronize({.memory = {&hostRead, 1}}));
        std::unique_ptr<Semaphore> gate;
        REG_REQUIRE(device->createSemaphore().transform([&](auto value) { gate = std::move(value); }));
        Drain drain{queue, *gate};
        REG_REQUIRE(recording.submit(tracker, *gate));
        REG_REQUIRE(gate->signal(1));
        REG_REQUIRE(recording.frame.wait(5'000'000'000ull));
        for (size_t i = 0; i < outputs.size(); ++i) {
            outputs[i]->invalidate();
            const auto* mapped = static_cast<const uint32_t*>(outputs[i]->map());
            REG_CHECK(mapped);
            std::array<uint32_t, 16> actual;
            std::memcpy(actual.data(), mapped, sizeof(actual));
            outputs[i]->unmap();
            for (uint32_t page = 0; page < 8; ++page) {
                REG_CHECK(actual[page * 2] == (i == 0 || page == 2 ? 0u : page == 5 ? 29u : 0xffffffffu));
                REG_CHECK(actual[page * 2 + 1] == (i == 0 ? 0u : 73u + page));
            }
        }
        REG_CHECK((*registry)->stats().descriptorWrites == 0);
        return RHITestResult::pass("Inline page initialization, independent patch snapshots, unload, empty and invalid patches, preserved request frames; zero descriptor writes");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamPageTableInlineTest);

class StreamDataDecodeTest final : public RHITest {
public:
    StreamDataDecodeTest() { type = RHITestType::Resource; name = "stream_data_decode_bounds"; }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        bench::TestDevice device;
        REG_REQUIRE(bench::createTestDevice(context, {.applicationName = "Stream BDA bounds",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true})
            .transform([&](auto value) { device = std::move(value); }));
        auto& queue = *device->getQueue(QueueType::Graphics);
        auto registry = device->resourceRegistry();
        REG_CHECK(registry);
        struct Params { StreamSceneParameters stream; ShaderDataSpan output; };
        ComputeKernel kernel, surfaceKernel;
        std::string log;
        auto shader = compileSlangShaderToSpirv({.moduleName = "StreamDataDecodeProbe",
            .entryPointName = "streamDataDecodeMain", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, log);
        if (!shader) { return RHITestResult::fail(log); }
        REG_REQUIRE(kernel.initialize(*device, {.spirv = shader->spirv, .parameters = parameterAbi<Params>(kABI + 6)}, log));
        auto surfaceShader = compileSlangShaderToSpirv({.moduleName = "StreamRaySurfaceProbe",
            .entryPointName = "streamRaySurfaceMain", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, log);
        if (!surfaceShader) { return RHITestResult::fail(log); }
        REG_REQUIRE(surfaceKernel.initialize(*device, {.spirv = surfaceShader->spirv, .parameters = parameterAbi<Params>(kABI + 6)}, log));
        std::unique_ptr<Buffer> output;
        REG_REQUIRE(device->createBuffer({.size = 40 * sizeof(uint32_t), .structureStride = 4,
            .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostReadback})
            .transform([&](auto value) { output = std::move(value); }));
        // Poison every result so a skipped invocation cannot look like a rejected input.
        void* initial = output->map();
        REG_CHECK(initial);
        std::memset(initial, 0xff, 40 * sizeof(uint32_t));
        output->flush();
        output->unmap();
        QueueSubmissionTracker tracker;
        REG_REQUIRE(tracker.initialize(*device, queue));
        Commands recording;
        REG_REQUIRE(recording.initialize(*device, queue));
        REG_REQUIRE(recording.begin(0));
        {
            // One resident page, one cluster, one triangle with three float3 positions.
            std::array<uint32_t, 63> words{};
            words[2] = 1; words[9] = 112; words[10] = 208; words[11] = 244;
            words[12] = 249; words[14] = 1; words[20] = 4;
            words[29] = 3; words[31] = 1;
            words[55] = 0x3f800000; words[59] = 0x3f800000; words[61] = 0x00020100;
            const std::array<uint32_t, 2> table{2, 0};
            ParameterWriter writer(*device, **registry, &recording.frame);
            Params params{};
            params.stream.pages = {writer.data(words.data(), sizeof(words)), uint32_t(words.size()), 4};
            params.stream.pageTable = {writer.data(table.data(), sizeof(table)), 1, 8};
            std::array<uint32_t, 24> instance{};
            instance[1] = 17;
            instance[4] = instance[9] = instance[14] = instance[19] = 0x3f800000; // Identity world transform.
            const std::array<uint32_t, 4> header{0, 0, 23, 0};
            params.stream.instances = {writer.data(instance.data(), sizeof(instance)), 1, 96};
            params.stream.header = {writer.data(header.data(), sizeof(header)), 3, 4};
            params.output = writer.dataBuffer(output.get(), 4, 4);
            auto encoded = writer.encode(params, kABI + 6);
            REG_CHECK(encoded);
            REG_REQUIRE(kernel.dispatch(*recording.commands, *encoded, 2));
            REG_REQUIRE(surfaceKernel.dispatch(*recording.commands, *encoded, 1));
        }
        const MemoryBarrierDesc hostRead{{PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
            {PipelineStageBits::Host, AccessBits::HostRead}};
        REG_REQUIRE(recording.commands->synchronize({.memory = {&hostRead, 1}}));
        std::unique_ptr<Semaphore> gate;
        REG_REQUIRE(device->createSemaphore().transform([&](auto value) { gate = std::move(value); }));
        Drain drain{queue, *gate};
        REG_REQUIRE(recording.submit(tracker, *gate));
        REG_REQUIRE(gate->signal(1));
        REG_REQUIRE(recording.frame.wait(5'000'000'000ull));
        output->invalidate();
        const auto* values = static_cast<const uint32_t*>(output->map());
        REG_CHECK(values);
        std::array<uint32_t, 40> actual{};
        std::memcpy(actual.data(), values, sizeof(actual));
        output->unmap();
        for (uint32_t i = 0; i < 24; ++i) { REG_CHECK(actual[i] == (i == 9 ? 17u : i == 10 ? 23u : (i == 0 || i == 11) ? 1u : 0u)); }
        for (uint32_t i = 0; i < 16; ++i) {
            const bool expected = i <= 2 || i == 7 || i == 10 || i == 11;
            REG_CHECK(actual[24 + i] == uint32_t(expected));
        }
        REG_CHECK((*registry)->stats().descriptorWrites == 0);
        return RHITestResult::pass("BDA triangle values; invalid page/cluster/triangle, truncated buffers, empty table, null resources, instance/header bounds and wrong strides rejected; explicit surface/alpha provider, ray-independent normals/TBN, bitangent flip and mask/blend thresholds");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamDataDecodeTest);

class ResourceRangeContractTest final : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"resource.range.shaderInput.contract"}, bench::Layer::RHI, "binding", "binding");
    }

    ResourceRangeContractTest() { type = RHITestType::Resource; name = "resource_range_and_shader_input_contract"; }
    RHITestResult run(RHITestContext& context) override
    {
        using render::BufferRange;
        using render::Error;
        const auto tail = BufferRange{16}.resolve(64);
        REG_CHECK(tail && tail->offset == 16 && tail->size == 48);
        const auto end = BufferRange{64}.resolve(64);
        REG_CHECK(end && end->size == 0);
        REG_CHECK(render::hasError(BufferRange{65, 0}.resolve(64), Error::InvalidArgument));
        REG_CHECK(render::hasError(BufferRange{16, UINT64_MAX - 1}.resolve(64), Error::InvalidArgument));
        REG_CHECK(render::hasError(BufferRange{UINT64_MAX - 2, 4}.resolve(UINT64_MAX), Error::InvalidArgument));

        auto& device = context.device;
        std::unique_ptr<render::Buffer> buffer;
        REG_REQUIRE(makeBuffer(device, buffer));
        auto slice = buffer->slice({16, 32});
        REG_CHECK(slice && slice->offset() == 16 && slice->size() == 32);
        auto sub = slice->subslice({8});
        REG_CHECK(sub && sub->offset() == 24 && sub->size() == 24);
        const auto empty = slice->subslice({32});
        REG_CHECK(empty && empty->size() == 0);
        REG_CHECK(render::hasError(slice->subslice({31, 2}), Error::InvalidArgument));
        REG_CHECK(render::hasError(buffer->slice({UINT64_MAX, 1}), Error::InvalidArgument));
        if (device.capabilities().bindlessDescriptorHeap) {
            auto view = device.createBufferView(*buffer, {.range = {16}});
            REG_CHECK(view && (*view)->desc().range == *tail);
            REG_CHECK(render::hasError(device.createBufferView(*buffer, {.range = {64}}), Error::InvalidArgument));
            REG_CHECK(render::hasError(device.createBufferView(*buffer, {.range = {16, UINT64_MAX - 1}}), Error::InvalidArgument));
        }

        auto texture = device.createTexture({.usage = render::TextureUsageBits::Sampled,
            .format = render::Format::RGBA8Unorm, .width = 8, .height = 8, .mipCount = 3, .layerCount = 2});
        REG_CHECK(texture);
        const render::TextureSubresourceRange range{1, 2, 1, 1};
        REG_CHECK(range.valid(3, 2));
        auto view = device.createTextureView(**texture, {.range = range});
        REG_CHECK(view && (*view)->desc().range.baseMip == 1 && (*view)->desc().range.layerCount == 1);
        for (const auto invalid : std::array<render::TextureSubresourceRange, 5>{{
                 {3, 1, 0, 1}, {0, 0, 0, 1}, {1, UINT32_MAX, 0, 1}, {0, 1, 2, 1}, {0, 1, 1, UINT32_MAX}}}) {
            REG_CHECK(!invalid.valid(3, 2));
            REG_CHECK(render::hasError(device.createTextureView(**texture, {.range = invalid}), Error::InvalidArgument));
        }

        // Word spans cannot represent misaligned byte lengths. Malformed word streams
        // must still be rejected before reaching Vulkan or the OMM transformer.
        const std::array<uint32_t, 4> truncated{0x07230203u};
        const std::array<uint32_t, 5> badMagic{};
        const std::array<uint32_t, 6> badInstruction{0x07230203u, 0x00010600u, 0, 1, 0, 0};
        REG_CHECK(render::hasError(device.createShaderModule({}), Error::InvalidArgument));
        REG_CHECK(render::hasError(device.createShaderModule({.spirv = truncated}), Error::InvalidArgument));
        REG_CHECK(render::hasError(device.createShaderModule({.spirv = badMagic}), Error::InvalidArgument));
        REG_CHECK(render::hasError(device.createShaderModule({.spirv = badInstruction}), Error::InvalidArgument));
        return RHITestResult::pass();
    }
};
METALLIC_REGISTER_RHI_TEST(ResourceRangeContractTest);


class SynchronizationScopesTest final : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"synchronization.atomic.validation.contract"}, bench::Layer::RHI, "core", "sync");
    }

    SynchronizationScopesTest() { type = RHITestType::Command; name = "synchronization_scopes_batch_and_validation"; }
    RHITestResult run(RHITestContext& context) override
    {
        using S = render::PipelineStageBits;
        using A = render::AccessBits;
        auto& device = context.device;
        Commands recording;
        REG_REQUIRE(recording.initialize(device, context.graphicsQueue));
        REG_REQUIRE(recording.begin(0));
        auto& command = *recording.commands;
        std::array<std::unique_ptr<render::Buffer>, 3> buffers;
        std::array<render::BufferBarrierDesc, 3> barriers;
        for (uint32_t i = 0; i < buffers.size(); ++i) {
            REG_REQUIRE(makeBuffer(device, buffers[i]));
            barriers[i] = {
                .buffer = buffers[i].get(),
                .before = {S::ComputeShader, A::ShaderWrite},
                .after = {S::ComputeShader, A::ShaderRead},
            };
        }
        REG_REQUIRE(command.synchronize({.buffers = {barriers.data(), 3}}));
        auto stats = command.synchronizationStats();
        REG_CHECK(stats.calls == 1 && stats.memoryBarriers == 1 && stats.coalescedResources == 3 && stats.imageTransitions == 0);
        for (auto& barrier : barriers) { barrier.before.access = A::ShaderRead; }
        REG_REQUIRE(command.synchronize({.buffers = {barriers.data(), 3}}));
        REG_CHECK(command.synchronizationStats().calls == 2); // Explicit read/read scopes still order execution.
        std::array<render::MemoryBarrierDesc, 2> memory{{
            {{S::ComputeShader, A::ShaderWrite}, {S::DrawIndirect, A::IndirectRead}},
            {{S::Transfer, A::TransferWrite}, {S::ComputeShader, A::ShaderRead}},
        }};
        REG_REQUIRE(command.synchronize({.memory = {memory.data(), 2}}));
        REG_CHECK(command.synchronizationStats().memoryBarriers == 4); // Keep distinct stage pairs.
        const std::array<render::SyncScope, 5> invalid{{
            {S::Transfer, A::ShaderWrite}, {S::ComputeShader, A::IndirectRead},
            {S::None, A::MemoryRead}, {static_cast<S>(1ull << 63), A::None}, {S::Transfer, static_cast<A>(1ull << 63)},
        }};
        for (const auto scope : invalid) {
            memory[1].after = scope;
            REG_CHECK(render::hasError(command.synchronize({.memory = {memory.data(), 2}}), render::Error::InvalidArgument));
            REG_CHECK(command.synchronizationStats().calls == 3); // Validation is atomic.
        }
        barriers[0].range.offset = 64;
        REG_CHECK(render::hasError(command.synchronize({.buffers = {barriers.data(), 3}}), render::Error::InvalidArgument));
        REG_REQUIRE(command.end());
        REG_CHECK(render::hasError(command.synchronize({}), render::Error::InvalidArgument));
        recording.frame.cancel();
        REG_REQUIRE(recording.begin(1));
        REG_CHECK(command.synchronizationStats().calls == 0);
        recording.frame.cancel();
        return RHITestResult::pass();
    }
};
METALLIC_REGISTER_RHI_TEST(SynchronizationScopesTest);

// Capture encoding without submitting work. Restore Volk's entry point even on
// an assertion failure; these tests execute serially in the RHI test process.
struct BarrierEncodingCapture {
    inline static BarrierEncodingCapture* active = nullptr;
    PFN_vkCmdPipelineBarrier2 original = vkCmdPipelineBarrier2;
    uint32_t calls = 0;
    std::vector<VkMemoryBarrier2> memory;
    std::vector<VkImageMemoryBarrier2> images;
    BarrierEncodingCapture()
    {
        active = this;
        vkCmdPipelineBarrier2 = capture;
    }
    ~BarrierEncodingCapture()
    {
        vkCmdPipelineBarrier2 = original;
        active = nullptr;
    }
    static VKAPI_ATTR void VKAPI_CALL capture(VkCommandBuffer, const VkDependencyInfo* dependency)
    {
        ++active->calls;
        active->memory.clear();
        active->images.clear();
        for (uint32_t i = 0; i < dependency->memoryBarrierCount; ++i) {
            active->memory.push_back(dependency->pMemoryBarriers[i]);
        }
        for (uint32_t i = 0; i < dependency->imageMemoryBarrierCount; ++i) {
            active->images.push_back(dependency->pImageMemoryBarriers[i]);
        }
    }
};

class SynchronizationEncodingTest final : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"synchronization.explicitScopes.encoding"}, bench::Layer::RHI, "core", "sync");
    }

    SynchronizationEncodingTest() { type = RHITestType::Command; name = "synchronization_explicit_scopes_and_layouts"; }
    RHITestResult run(RHITestContext& context) override
    {
        using S = render::PipelineStageBits;
        using A = render::AccessBits;
        using L = render::TextureLayout;
        Commands recording;
        REG_REQUIRE(recording.initialize(context.device, context.graphicsQueue));
        REG_REQUIRE(recording.begin(0));
        auto& command = *recording.commands;
        auto texture = context.device.createTexture({.usage = render::TextureUsageBits::Storage | render::TextureUsageBits::Sampled,
            .format = render::Format::RGBA8Unorm, .width = 4, .height = 4});
        REG_CHECK(texture);
        std::unique_ptr<render::Buffer> buffer;
        REG_REQUIRE(makeBuffer(context.device, buffer));
        BarrierEncodingCapture capture;

        render::TextureBarrierDesc image{.texture = texture->get(), .oldLayout = L::General, .newLayout = L::General};
        render::BufferBarrierDesc bytes{.buffer = buffer.get()};
        REG_REQUIRE(command.synchronize({.textures = {&image, 1}, .buffers = {&bytes, 1}}));
        REG_CHECK(capture.calls == 0); // General layout must not invent accesses.

        image.oldLayout = L::Undefined;
        image.after = {S::ComputeShader, A::ShaderWrite};
        REG_REQUIRE(command.synchronize({.textures = {&image, 1}}));
        REG_CHECK(capture.calls == 1 && capture.images.size() == 1);
        REG_CHECK(capture.images[0].srcStageMask == VK_PIPELINE_STAGE_2_NONE && capture.images[0].srcAccessMask == 0);
        REG_CHECK(capture.images[0].dstStageMask == VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT &&
            capture.images[0].dstAccessMask == VK_ACCESS_2_SHADER_WRITE_BIT);

        // A semaphore-covered producer needs an empty source scope even when
        // its old layout is defined. Unified layouts must preserve that scope.
        image.oldLayout = L::General;
        image.newLayout = L::ShaderRead;
        image.after = {S::ComputeShader, A::ShaderRead};
        REG_REQUIRE(command.synchronize({.textures = {&image, 1}}));
        if (context.device.capabilities().unifiedImageLayouts) {
            REG_CHECK(capture.memory.size() == 1 && capture.images.empty());
            REG_CHECK(capture.memory[0].srcStageMask == 0 && capture.memory[0].srcAccessMask == 0);
        } else {
            REG_CHECK(capture.images.size() == 1 && capture.memory.empty());
            REG_CHECK(capture.images[0].srcStageMask == 0 && capture.images[0].srcAccessMask == 0);
            REG_CHECK(capture.images[0].newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }

        image.oldLayout = image.newLayout;
        image.before = bytes.before = {S::ComputeShader, A::None};
        image.after = bytes.after = {S::FragmentShader, A::None};
        REG_REQUIRE(command.synchronize({.textures = {&image, 1}, .buffers = {&bytes, 1}}));
        REG_CHECK(capture.memory.size() == 1 && capture.images.empty());
        REG_CHECK(capture.memory[0].srcStageMask == VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT &&
            capture.memory[0].dstStageMask == VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);
        REG_CHECK(capture.memory[0].srcAccessMask == 0 && capture.memory[0].dstAccessMask == 0);
        const auto beforeInvalid = capture.calls;
        bytes.after = {S::None, A::ShaderRead};
        REG_CHECK(render::hasError(command.synchronize({
            .textures = {&image, 1},
            .buffers = {&bytes, 1},
        }), render::Error::InvalidArgument));
        image.newLayout = static_cast<L>(255);
        REG_CHECK(render::hasError(command.synchronize({.textures = {&image, 1}}), render::Error::InvalidArgument));
        image.newLayout = L::Undefined;
        REG_CHECK(render::hasError(command.synchronize({.textures = {&image, 1}}), render::Error::InvalidArgument));
        REG_CHECK(capture.calls == beforeInvalid);
        REG_REQUIRE(command.end());
        recording.frame.cancel();
        return RHITestResult::pass();
    }
};
METALLIC_REGISTER_RHI_TEST(SynchronizationEncodingTest);

class PreparedExecutionViewsTest final : public RHITest {
public:
    PreparedExecutionViewsTest() { type = RHITestType::Rendering; name = "prepared_execution_lazy_views_layout_policy"; }
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::comparisonMetadata({"layouts.optimal.unified.preparedViews.draw.copy"}, bench::Layer::RHI,
            "core", {"core-unified", "unifiedLayouts", bench::Capability::UnifiedLayouts});
    }
    RHITestResult run(RHITestContext& context) override
    {
        constexpr uint32_t extent = 32, bytes = extent * extent * 4;
        std::array<uint8_t, bytes> reference{};
        bool unifiedTested = false;
        std::vector<uint8_t> observations;
        const auto variants = context.deviceDesc ? std::vector<bool>{context.deviceDesc->preferUnifiedImageLayouts} : std::vector<bool>{false, true};
        for (bool preferUnified : variants) {
            std::atomic_uint errors{0};
            bench::TestDevice device;
            REG_REQUIRE(bench::createTestDevice(context, {.applicationName = "Prepared execution lifetime", .enableValidation = context.enableValidation,
                .validationSink = {[](void* target, const render::ValidationMessage& message) noexcept {
                    if (message.severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
                        ++*static_cast<std::atomic_uint*>(target);
                    }
                }, &errors}, .preferUnifiedImageLayouts = preferUnified}).transform([&](auto value) { device = std::move(value); }));
            const bool unified = device->capabilities().unifiedImageLayouts;
            REG_CHECK(preferUnified || !unified);
            unifiedTested |= unified;
            auto& queue = *device->getQueue(render::QueueType::Graphics);
            std::array<render::ShaderCompileResult, 2> compiled;
            std::array<std::unique_ptr<render::ShaderModule>, 2> modules;
            const char* entries[] = {"triangleVertexMain", "triangleFragmentMain"};
            for (uint32_t i = 0; i < modules.size(); ++i) {
                REG_REQUIRE(render::compileSlangShaderToSpirv({.moduleName = "Features/Samples/Triangle",
                    .entryPointName = entries[i], .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, compiled[i].diagnostics).transform([&](auto value) { compiled[i] = std::move(value); }));
                REG_REQUIRE(device->createShaderModule({
                    .spirv = compiled[i].spirv,
                }).transform([&](auto value) { modules[i] = std::move(value); }));
            }
            auto foreignDevice = bench::createTestDevice(context, {.enableValidation = context.enableValidation}, true);
            REG_CHECK(foreignDevice);
            auto foreign = foreignDevice->get()->createShaderModule({.spirv = compiled[0].spirv});
            REG_CHECK(foreign);
            const render::ShaderStageDesc fragment{modules[1].get()};
            REG_CHECK(render::hasError(device->createGraphicsPipeline({.vertexShader = {foreign->get()},
                .fragmentShader = fragment}), render::Error::InvalidArgument));
            REG_CHECK(render::hasError(device->createGraphicsShaderObjectProgram({.vertexShader = {foreign->get()},
                .fragmentShader = fragment}), render::Error::InvalidArgument));
            REG_CHECK(render::hasError(device->createComputePipeline({.computeShader = {foreign->get()}}), render::Error::InvalidArgument));
            for (const char* entry : std::array<const char*, 2>{nullptr, ""}) {
                REG_CHECK(render::hasError(device->createGraphicsPipeline({.vertexShader = {modules[0].get(), entry},
                    .fragmentShader = fragment}), render::Error::InvalidArgument));
                REG_CHECK(render::hasError(device->createGraphicsShaderObjectProgram({.vertexShader = {modules[0].get(), entry},
                    .fragmentShader = fragment}), render::Error::InvalidArgument));
                REG_CHECK(render::hasError(device->createComputePipeline({.computeShader = {modules[0].get(), entry}}), render::Error::InvalidArgument));
            }
            compiled = {}; // Both executable forms must use the module's owned words.
            std::unique_ptr<render::GraphicsPipeline> pipeline;
            REG_REQUIRE(device->createGraphicsPipeline({
                .vertexShader = {modules[0].get()},
                .fragmentShader = {modules[1].get()},
                .colorFormat = render::Format::RGBA8Unorm,
            }).transform([&](auto value) { pipeline = std::move(value); }));
            std::unique_ptr<render::GraphicsShaderObjectProgram> program;
            REG_REQUIRE(device->createGraphicsShaderObjectProgram({.vertexShader = {modules[0].get()}, .fragmentShader = {modules[1].get()}}).transform([&](auto value) { program = std::move(value); }));
            std::array<render::PreparedExecution, 3> executions{pipeline->execution(), program->execution(), pipeline->execution()};
            auto invalidState = program->execution({.colorAttachmentCount = 9});
            // Snapshots survive hot replacement of all source objects before recording.
            pipeline.reset(); program.reset(); modules = {};
            render::QueueSubmissionTracker tracker;
            REG_REQUIRE(tracker.initialize(*device, queue));
            Commands recording;
            REG_REQUIRE(recording.initialize(*device, queue));
            std::unique_ptr<render::Semaphore> gate;
            REG_REQUIRE(device->createSemaphore().transform([&](auto value) { gate = std::move(value); }));
            Drain drain{queue, *gate};
            REG_REQUIRE(recording.begin(0));
            auto& command = *recording.commands;
            REG_CHECK(render::hasError(command.bindExecution({}), render::Error::InvalidArgument));
            REG_CHECK(render::hasError(command.bindExecution(invalidState), render::Error::InvalidArgument));
            invalidState = {};
            std::array<std::unique_ptr<render::Buffer>, 3> readbacks;
            std::array<std::weak_ptr<void>, 3> allocations;
            for (uint32_t i = 0; i < executions.size(); ++i) {
                std::unique_ptr<render::Texture> texture;
                REG_REQUIRE(device->createTexture({.usage = render::TextureUsageBits::ColorAttachment | render::TextureUsageBits::TransferSource,
                    .format = render::Format::RGBA8Unorm, .width = extent, .height = extent}).transform([&](auto value) { texture = std::move(value); }));
                std::unique_ptr<render::TextureView> view;
                REG_REQUIRE(device->createTextureView(*texture, {}).transform([&](auto value) { view = std::move(value); }));
                REG_CHECK(!view->hasNativeView());
                REG_CHECK(render::hasError(device->createTextureView(*texture, {.range = {.baseMip = 1}}).transform([](auto) {}), render::Error::InvalidArgument));
                REG_CHECK(render::vulkan::nativeImageLayout(*view, render::ResourceState::ColorAttachment) ==
                    (unified ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL));
                REG_CHECK(!view->hasNativeView());
                allocations[i] = view->retainTexture();
                REG_REQUIRE(device->createBuffer({.size = bytes, .usage = render::BufferUsageBits::TransferDestination,
                    .memoryLocation = render::MemoryLocation::HostReadback}).transform([&](auto value) { readbacks[i] = std::move(value); }));
                render::TextureBarrierDesc barrier{
                    .texture = texture.get(),
                    .oldLayout = render::TextureLayout::Undefined,
                    .newLayout = render::TextureLayout::ColorAttachment,
                    .before = {},
                    .after = {render::PipelineStageBits::ColorAttachment, render::AccessBits::ColorRead | render::AccessBits::ColorWrite},
                };
                REG_REQUIRE(command.synchronize({.textures = {&barrier, 1}}));
                render::RenderingAttachmentDesc attachment{.view = view.get(), .state = render::ResourceState::ColorAttachment,
                    .loadOp = render::LoadOp::Clear, .clearColor = {0, 0, 0, 1}};
                REG_REQUIRE(command.beginRendering({.renderArea = {0, 0, extent, extent}, .colorAttachments = {&attachment, 1}}));
                REG_CHECK(view->hasNativeView());
                const auto native = render::vulkan::nativeImageView(*view);
                REG_CHECK(native != VK_NULL_HANDLE && native == render::vulkan::nativeImageView(*view));
                // Simulate the SDK/DGC boundary, then explicitly establish the new state.
                if (i == 1) { render::vulkan::notifyExternalDescriptorSetBinding(command); }
                REG_REQUIRE(command.bindExecution(executions[i]));
                command.setViewport({0, 0, float(extent), float(extent), 0, 1});
                command.setScissor({0, 0, extent, extent});
                command.draw(3);
                command.endRendering();
                barrier.oldLayout = render::TextureLayout::ColorAttachment; barrier.before = {render::PipelineStageBits::ColorAttachment, render::AccessBits::ColorRead | render::AccessBits::ColorWrite};
                barrier.newLayout = render::TextureLayout::TransferSource; barrier.after = {render::PipelineStageBits::Transfer, render::AccessBits::TransferRead};
                REG_REQUIRE(command.synchronize({.textures = {&barrier, 1}}));
                command.copyTextureToBuffer({.texture = texture.get(), .buffer = readbacks[i].get(), .width = extent, .height = extent});
                view.reset(); texture.reset();
                REG_CHECK(!allocations[i].expired());
            }
            const auto stats = command.synchronizationStats();
            REG_CHECK(stats.imageTransitions == (unified ? 3 : 6));
            REG_CHECK(stats.memoryBarriers == (unified ? 3 : 0));
            executions = {}; // Only recorded commands now own the native programs.
            REG_REQUIRE(recording.submit(tracker, *gate));
            REG_CHECK(!recording.frame.completion().isComplete());
            REG_REQUIRE(gate->signal(1));
            REG_REQUIRE(recording.frame.wait(5'000'000'000ull));
            for (uint32_t i = 0; i < readbacks.size(); ++i) {
                readbacks[i]->invalidate();
                const auto* pixels = static_cast<const uint8_t*>(readbacks[i]->map());
                REG_CHECK(pixels && pixels[(extent / 2 * extent + extent / 2) * 4] > 0);
                if ((!preferUnified || context.evidence) && i == 0) { std::memcpy(reference.data(), pixels, bytes); }
                bench::readbackEvidence(context, "readback.bin", std::span<const uint8_t>(pixels, bytes));
                observations.insert(observations.end(), pixels, pixels + bytes);
                const bool same = std::memcmp(reference.data(), pixels, bytes) == 0;
                readbacks[i]->unmap();
                REG_CHECK(same);
            }
            REG_REQUIRE(recording.pool->reset());
            REG_REQUIRE(recording.frame.reset());
            for (const auto& allocation : allocations) { REG_CHECK(allocation.expired()); }
            // Cancellation also releases an exported native view without submitting it.
            REG_REQUIRE(recording.begin(1));
            std::weak_ptr<void> cancelled;
            {
                auto texture = device->createTexture({.usage = render::TextureUsageBits::ColorAttachment, .format = render::Format::RGBA8Unorm});
                REG_CHECK(texture);
                auto view = device->createTextureView(**texture, {});
                REG_CHECK(view);
                cancelled = (*view)->retainTexture();
                REG_REQUIRE(command.useNativeTextureView(**view));
            }
            REG_CHECK(!cancelled.expired());
            recording.frame.cancel();
            REG_REQUIRE(recording.pool->reset());
            REG_REQUIRE(recording.frame.reset());
            REG_CHECK(cancelled.expired());
            REG_CHECK(errors.load() == 0);
        }
        bench::comparisonEvidence(context, {{"extent", extent}, {"draws", 3}}, observations,
            context.deviceDesc && context.deviceDesc->preferUnifiedImageLayouts);
        if (context.evidence) { return RHITestResult::pass("prepared views and three readbacks passed; parent compares layout policies"); }
        return RHITestResult::pass(unifiedTested ? "GENERAL and optimal layouts produced identical PSO/shader-object readback" :
            "Optimal-layout fallback passed; unified image layouts unavailable on this device");
    }
};
METALLIC_REGISTER_RHI_TEST(PreparedExecutionViewsTest);

class ParallelRegistryTest final : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"parameters.parallel.readback"}, bench::Layer::Core, "binding", "binding", {"readback.bin"});
    }

    ParallelRegistryTest() { type = RHITestType::Command; name = "parallel_registry_packets"; }
    RHITestResult run(RHITestContext& context) override
    {
        for (auto mode : {render::SlangDescriptorHeapMode::Mapped, render::SlangDescriptorHeapMode::Native}) {
            auto result = runMode(context, mode);
            if (!result.passed) { return result; }
        }
        return RHITestResult::pass("Four concurrent writers sharing a registry and kernel in mapped/native modes");
    }
private:
    static RHITestResult runMode(RHITestContext& context, render::SlangDescriptorHeapMode mode)
    {
        bench::TestDevice device;
        REG_REQUIRE(bench::createTestDevice(context, {.applicationName = "Parallel registry", .enableValidation = context.enableValidation,
            .enableBindlessDescriptorHeap = true}).transform([&](auto value) { device = std::move(value); }));
        auto& queue = *device->getQueue(render::QueueType::Graphics);
        std::shared_ptr<render::ResourceRegistry> registry;
        REG_REQUIRE(device->resourceRegistry().transform([&](auto value) { registry = std::move(value); }));
        render::ComputeKernel kernel;
        std::string log;
        REG_REQUIRE(makeKernel(*device, kernel, log, mode));
        std::unique_ptr<render::Buffer> source, output;
        REG_REQUIRE(makeBuffer(*device, source, 73));
        REG_REQUIRE(makeBuffer(*device, output));
        std::weak_ptr<void> allocation = source->retainAllocation();
        render::RenderFrameContext frame;
        std::array<render::CommandRecordingContext, 4> contexts;
        std::array<render::CommandBuffer*, 4> commands{};
        std::array<render::Result<>, 4> results;
        render::QueueSubmissionTracker tracker;
        REG_REQUIRE(tracker.initialize(*device, queue));
        REG_REQUIRE(frame.begin(0));
        for (uint32_t i = 0; i < contexts.size(); ++i) {
            REG_REQUIRE(contexts[i].initialize(*device, queue));
            REG_REQUIRE(contexts[i].prepare(frame).transform([&](auto value) { commands[i] = value; }));
        }
        std::vector<std::jthread> workers;
        for (uint32_t i = 0; i < contexts.size(); ++i) {
            workers.emplace_back([&, i] {
                results[i] = contexts[i].record([&]() -> render::Result<> {
                    render::ParameterWriter writer(*device, frame, *registry);
                    ProbeParams params{writer.buffer(source.get()), writer.buffer(output.get()), i, i};
                    render::EncodedParameters encoded;
                    auto result = writer.encode(params, kABI).transform([&](auto value) { encoded = std::move(value); });
                    if (result) { result = kernel.dispatch(*commands[i], encoded, 1); }
                    return result ? commands[i]->end() : result;
                });
            });
        }
        workers.clear(); // jthread joins every local resource/parameter writer.
        for (const auto& recorded : results) { REG_REQUIRE(recorded); }
        source.reset();
        kernel.clear();
        REG_CHECK(!allocation.expired());
        REG_REQUIRE(tracker.submit({.commandBuffers = commands}, frame));
        REG_REQUIRE(frame.wait(5'000'000'000ull));
        output->invalidate();
        auto* mapped = output->map();
        REG_CHECK(mapped != nullptr);
        std::array<uint32_t, 4> actual{};
        std::memcpy(actual.data(), mapped, sizeof(actual));
        bench::readbackEvidence(context, "readback.bin", std::span<const uint32_t>(actual));
        output->unmap();
        REG_CHECK((actual == std::array<uint32_t, 4>{73, 74, 75, 76}));
        for (auto& recording : contexts) { REG_REQUIRE(recording.reset()); }
        REG_REQUIRE(frame.reset());
        registry->collect();
        REG_CHECK(allocation.expired());
        return RHITestResult::pass();
    }
};
METALLIC_REGISTER_RHI_TEST(ParallelRegistryTest);

class PreparedDispatchParallelTest final : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"compute.prepared.parallel.lifetime.readback"}, bench::Layer::Core, "binding", "binding", {"readback.bin"}, true);
    }

    PreparedDispatchParallelTest() { type = RHITestType::Rendering; name = "prepared_dispatch_parallel_snapshot_lifetime"; }
    RHITestResult run(RHITestContext& context) override
    {
        for (const auto mode : {render::SlangDescriptorHeapMode::Mapped, render::SlangDescriptorHeapMode::Native}) {
            bench::TestDevice device;
            REG_REQUIRE(bench::createTestDevice(context, {.applicationName = "Prepared dispatch snapshot",
                .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}).transform([&](auto value) { device = std::move(value); }));
            auto& queue = *device->getQueue(render::QueueType::Graphics);
            render::ShaderCompileResult shader;
            REG_REQUIRE(render::compileSlangShaderToSpirv({.moduleName = "FrameResourceProbe", .entryPointName = "copyValue",
                .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders", .descriptorHeapMode = mode}, shader.diagnostics).transform([&](auto value) { shader = std::move(value); }));
            render::ComputeProgram programs[2];
            render::ComputeProgramBindingDesc layout[] = {
                {.binding = 0, .kind = render::ComputeResourceBindingKind::StorageBuffer},
                {.binding = 1, .kind = render::ComputeResourceBindingKind::StorageBuffer}};
            std::string log;
            for (auto& program : programs) {
                REG_REQUIRE(program.initialize(*device, {
                    .spirv = shader.spirv,
                    .pushConstantSize = 4,
                    .bindings = {layout, 2},
                    .requiresRayQuery = false,
                }, log));
                std::swap(layout[0], layout[1]);
            }
            std::unique_ptr<render::Buffer> input, output, arguments;
            REG_REQUIRE(makeBuffer(*device, input, 137));
            REG_REQUIRE(makeBuffer(*device, output));
            REG_REQUIRE(makeBuffer(*device, arguments, 1));
            auto* counts = static_cast<uint32_t*>(arguments->map());
            REG_CHECK(counts);
            for (uint32_t i = 0; i < 6; ++i) { counts[i] = 1; }
            arguments->flush(); arguments->unmap();
            std::weak_ptr<void> inputLife = input->retainAllocation(), argumentLife = arguments->retainAllocation();
            render::RenderFrameContext frame;
            render::CommandRecordingContext contexts[2];
            render::QueueSubmissionTracker tracker;
            REG_REQUIRE(tracker.initialize(*device, queue));
            std::unique_ptr<render::Semaphore> gate;
            REG_REQUIRE(device->createSemaphore().transform([&](auto value) { gate = std::move(value); }));
            Drain drain{queue, *gate};
            REG_REQUIRE(frame.begin(0));
            render::PreparedComputeDispatch packets[2];
            render::Result<> outcomes[2];
            uint32_t indices[] = {0, 1, 2};
            const render::ComputeDispatchBinding bindings[] = {{.binding = 0, .buffer = input.get()}, {.binding = 1, .buffer = output.get()}};
            std::jthread first([&] {
                outcomes[0] = programs[0].prepareDispatch(frame, {
                    .bindings = {bindings, 2},
                    .pushData = &indices[0],
                    .pushDataSize = 4,
                }).transform([&](auto value) { packets[0] = std::move(value); });
            });
            std::jthread second([&] {
                const render::ComputeIndirectDispatch items[] = {
                    {.pushData = &indices[1]}, {.pushData = &indices[2], .argumentOffset = 12, .program = &programs[1]}};
                outcomes[1] = programs[0].prepareIndirectBatch(frame, {
                    .bindings = {bindings, 2},
                    .pushDataSize = 4,
                    .indirectArguments = arguments.get(),
                }, items).transform([&](auto value) { packets[1] = std::move(value); });
            });
            first.join(); second.join();
            for (const auto& outcome : outcomes) { REG_REQUIRE(outcome); }
            const auto failed = programs[0].prepareDispatch(frame, {
                .bindings = {bindings, 2},
                .pushData = &indices[0],
                .pushDataSize = 3,
            });
            REG_CHECK(render::hasError(failed, render::Error::InvalidArgument) && packets[0].valid());
            // Preparation owns constant bytes, permutations, descriptors and argument ranges.
            indices[0] = indices[1] = indices[2] = 15;
            input.reset(); arguments.reset(); programs[0].clear(); programs[1].clear();
            REG_CHECK(!inputLife.expired() && !argumentLife.expired());
            render::CommandBuffer* commands[2]{};
            for (uint32_t i = 0; i < 2; ++i) {
                REG_REQUIRE(contexts[i].initialize(*device, queue));
                REG_REQUIRE(contexts[i].prepare(frame).transform([&](auto value) { commands[i] = value; }));
            }
            const render::BufferBarrierDesc barrier{
                .buffer = output.get(),
                .before = {},
                .after = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
            };
            if (auto commandResult = commands[0]->synchronize({.buffers = {&barrier, 1}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
            std::jthread recordA([&] { outcomes[0] = contexts[0].record([&]() -> render::Result<> {
                auto recorded = packets[0].record(*commands[0]); return recorded ? commands[0]->end() : recorded; }); });
            std::jthread recordB([&] { outcomes[1] = contexts[1].record([&]() -> render::Result<> {
                auto recorded = packets[1].record(*commands[1]); return recorded ? commands[1]->end() : recorded; }); });
            recordA.join(); recordB.join();
            for (const auto& outcome : outcomes) { REG_REQUIRE(outcome); }
            auto stale = packets[0];
            packets[0] = {}; packets[1] = {};
            const render::SemaphoreSubmitDesc wait{.semaphore = gate.get(), .value = 1};
            REG_REQUIRE(tracker.submit({
                .waitSemaphores = {&wait, 1},
                .commandBuffers = {commands, 2},
            }, frame));
            REG_CHECK(!inputLife.expired() && !argumentLife.expired());
            REG_REQUIRE(gate->signal(1)); REG_REQUIRE(frame.wait());
            output->invalidate();
            auto* values = static_cast<uint32_t*>(output->map());
            REG_CHECK(values);
            bench::readbackEvidence(context, "readback.bin", std::span<const uint32_t>(values, 16));
            const bool correct = values[0] == 137 && values[1] == 137 && values[2] == 137 && values[15] == 0;
            output->unmap(); REG_CHECK(correct);
            for (auto& recording : contexts) { REG_REQUIRE(recording.reset()); }
            REG_REQUIRE(frame.reset()); REG_REQUIRE(frame.begin(1));
            REG_REQUIRE(contexts[0].prepare(frame).transform([&](auto value) { commands[0] = value; }));
            REG_CHECK(!stale.record(*commands[0]));
            stale = {};
            REG_CHECK(inputLife.expired() && argumentLife.expired());
            frame.cancel();
            REG_REQUIRE(contexts[0].reset()); REG_REQUIRE(frame.reset());
        }
        return RHITestResult::pass("Mapped/native: concurrent preparation and recording, frozen constants, indirect permutations, lifetime and stale generation");
    }
};
METALLIC_REGISTER_RHI_TEST(PreparedDispatchParallelTest);

// Standalone packets own their parameter storage; a batch can be prepared and
// recorded after the writer, source wrappers and kernel wrappers are destroyed.
class KernelPreparedDispatchTest final : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"compute.prepared.direct.indirectBatch.readback", "compute.abi.staleTail.contract"}, bench::Layer::Core, "binding", "binding", {"readback.bin"}, true);
    }

    KernelPreparedDispatchTest() { type = RHITestType::Rendering; name = "compute_kernel_prepared_standalone_batch"; }
    RHITestResult run(RHITestContext& context) override
    {
        for (const auto transport : {render::ParameterTransport::DeviceAddress, render::ParameterTransport::InlinePush}) {
            for (const auto mode : {render::SlangDescriptorHeapMode::Mapped, render::SlangDescriptorHeapMode::Native}) {
                bench::TestDevice device;
                REG_REQUIRE(bench::createTestDevice(context, {.applicationName = "Kernel prepared dispatch",
                    .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true})
                    .transform([&](auto value) { device = std::move(value); }));
                auto registry = device->resourceRegistry();
                REG_CHECK(registry);
                auto& queue = *device->getQueue(render::QueueType::Graphics);
                std::unique_ptr<render::Buffer> input, output, arguments;
                REG_REQUIRE(makeBuffer(*device, input, 9));
                REG_REQUIRE(makeBuffer(*device, output));
                REG_REQUIRE(makeBuffer(*device, arguments));
                auto* counts = static_cast<uint32_t*>(arguments->map());
                REG_CHECK(counts);
                for (uint32_t i = 0; i < 6; ++i) { counts[i] = 1; }
                arguments->flush(); arguments->unmap();
                std::weak_ptr<void> inputLife = input->retainAllocation(), argumentLife = arguments->retainAllocation();
                Commands recording;
                REG_REQUIRE(recording.initialize(*device, queue));
                std::unique_ptr<render::Semaphore> gate;
                REG_REQUIRE(device->createSemaphore().transform([&](auto value) { gate = std::move(value); }));
                Drain drain{queue, *gate};
                REG_REQUIRE(recording.commands->begin()); // Deliberately no frame.
                render::PreparedComputeDispatch direct, batch, rejected;
                {
                    render::ComputeKernel kernels[2];
                    std::string log;
                    for (auto& kernel : kernels) { REG_REQUIRE(makeKernel(*device, kernel, log, mode, transport)); }
                    render::ParameterWriter writer(*device, **registry);
                    ProbeParams params{writer.buffer(input.get()), writer.buffer(output.get()), 1, 0};
                    auto first = writer.encode(params, kABI, transport);
                    REG_CHECK(first);
                    REG_CHECK(first->inlineData().empty() == (transport == render::ParameterTransport::DeviceAddress));
                    const auto before = (*registry)->stats().parameterBytes;
                    const auto wrongTransport = transport == render::ParameterTransport::InlinePush
                        ? render::ParameterTransport::DeviceAddress : render::ParameterTransport::InlinePush;
                    auto mismatch = writer.encode(params, kABI, wrongTransport);
                    REG_CHECK(mismatch && !kernels[0].prepareDispatch(*mismatch, 1));
                    if (transport == render::ParameterTransport::DeviceAddress) {
                        REG_CHECK((*registry)->stats().parameterBytes == before);
                    }
                    REG_REQUIRE(kernels[0].prepareDispatch(*first, 1).transform([&](auto value) { direct = std::move(value); }));
                    REG_CHECK(!kernels[0].prepareDispatch(*first, 0));
                    auto wrongAbi = writer.encode(params, kABI + 1, transport);
                    REG_CHECK(wrongAbi && !kernels[0].prepareDispatch(*wrongAbi, 1));
                    REG_CHECK(!kernels[0].prepareIndirectBatch({}));
                    render::ComputeIndirectParameters items[2];
                    for (uint32_t i = 0; i < 2; ++i) {
                        params.add = i + 2; params.index = i + 1;
                        REG_REQUIRE(writer.encode(params, kABI, transport).transform([&](auto value) { items[i].parameters = std::move(value); }));
                        REG_REQUIRE(arguments->slice({12 * i, 12}).transform([&](auto value) { items[i].arguments = std::move(value); }));
                        items[i].kernel = &kernels[i];
                    }
                    REG_REQUIRE(kernels[0].prepareIndirectBatch(items).transform([&](auto value) { batch = std::move(value); }));
                    auto saved = items[1].arguments;
                    items[1].arguments = {};
                    REG_CHECK(!kernels[0].prepareIndirectBatch(items));
                    items[1].arguments = saved;
                    params.add = 999; params.index = 15;
                    REG_REQUIRE(writer.encode(params, kABI, transport).transform([&](auto value) { items[0].parameters = std::move(value); }));
                    render::RenderFrameContext frame;
                    REG_REQUIRE(frame.begin(0));
                    render::ParameterWriter scopedWriter(*device, frame, **registry);
                    const ProbeParams scoped{scopedWriter.buffer(input.get()), scopedWriter.buffer(output.get()), 999, 15};
                    REG_REQUIRE(scopedWriter.encode(scoped, kABI, transport).transform([&](auto value) { items[1].parameters = std::move(value); }));
                    REG_REQUIRE(kernels[0].prepareIndirectBatch(items).transform([&](auto value) { rejected = std::move(value); }));
                    frame.cancel();
                }
                input.reset(); arguments.reset();
                REG_CHECK(!inputLife.expired() && !argumentLife.expired());
                // No prefix of a batch may execute when a later parameter packet is stale.
                REG_CHECK(render::hasError(rejected.record(*recording.commands), render::Error::InvalidArgument));
                rejected = {};
                REG_REQUIRE(direct.record(*recording.commands));
                REG_REQUIRE(batch.record(*recording.commands));
                direct = {}; batch = {};
                REG_REQUIRE(recording.commands->end());
                render::CommandBuffer* submitted[] = {recording.commands.get()};
                REG_REQUIRE(queue.submit({.commandBuffers = submitted}));
                REG_REQUIRE(queue.waitIdle());
                output->invalidate();
                const auto* actual = static_cast<const uint32_t*>(output->map());
                REG_CHECK(actual);
                bench::readbackEvidence(context, "readback.bin", std::span<const uint32_t>(actual, 16));
                const bool correct = actual[0] == 10 && actual[1] == 11 && actual[2] == 12 && actual[15] == 0;
                output->unmap();
                REG_CHECK(correct);
                REG_REQUIRE(recording.pool->reset());
                REG_REQUIRE(recording.commands->begin()); // Reuse releases the submitted command snapshot.
                REG_REQUIRE(recording.commands->end());
                REG_CHECK(inputLife.expired() && argumentLife.expired());
            }
        }
        return RHITestResult::pass("BDA/inline, mapped/native: standalone storage, direct/batch execution, ABI checks, stale-tail rejection and retained allocations");
    }
};
METALLIC_REGISTER_RHI_TEST(KernelPreparedDispatchTest);

// Exercise the common prepared resource-table path in mapped and native modes.
// Two writes to the same word require a memory-only dependency between dispatches.
class BatchMemoryBarrierTest final : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"compute.batch.memoryBarrier.readback", "compute.batch.error.contract"}, bench::Layer::Core, "binding", "sync", {"readback.bin"}, true);
    }

    BatchMemoryBarrierTest() { type = RHITestType::Rendering; name = "compute_batch_memory_barrier_and_error_propagation"; }
    RHITestResult run(RHITestContext& context) override
    {
        for (uint32_t path = 0; path < 2; ++path) {
            bench::TestDevice device;
            REG_REQUIRE(bench::createTestDevice(context, {.applicationName = "Batch barriers",
                .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true})
                .transform([&](auto value) { device = std::move(value); }));
            auto& queue = *device->getQueue(render::QueueType::Graphics);
            render::ShaderCompileResult shader;
            REG_REQUIRE(render::compileSlangShaderToSpirv({
                .moduleName = "BatchBarrierProbe",
                .entryPointName = "batchBarrierMain",
                .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders",
                .descriptorHeapMode = path == 1 ? render::SlangDescriptorHeapMode::Native : render::SlangDescriptorHeapMode::Mapped,
            }, shader.diagnostics).transform([&](auto value) { shader = std::move(value); }));
            const render::ComputeProgramBindingDesc layout{.binding = 0, .kind = render::ComputeResourceBindingKind::StorageBuffer};
            render::ComputeProgram program;
            std::string log;
            REG_REQUIRE(program.initialize(*device, {
                .spirv = shader.spirv,
                .bindings = {&layout, 1},
                .requiresRayQuery = false,
            }, log));
            std::unique_ptr<render::Buffer> output, arguments;
            REG_REQUIRE(makeBuffer(*device, output));
            REG_REQUIRE(makeBuffer(*device, arguments));
            auto* counts = static_cast<uint32_t*>(arguments->map());
            REG_CHECK(counts);
            for (uint32_t i = 0; i < 6; ++i) { counts[i] = 1; }
            arguments->flush(); arguments->unmap();
            Commands recording;
            REG_REQUIRE(recording.initialize(*device, queue));
            render::QueueSubmissionTracker tracker;
            REG_REQUIRE(tracker.initialize(*device, queue));
            std::unique_ptr<render::Semaphore> gate;
            REG_REQUIRE(device->createSemaphore({.initialValue = 1}).transform([&](auto value) { gate = std::move(value); }));
            Drain drain{queue, *gate};
            const render::ComputeDispatchBinding binding{.binding = 0, .buffer = output.get()};
            const render::ComputeIndirectDispatch items[] = {{.argumentOffset = 0}, {.argumentOffset = 12}};
            render::MemoryBarrierDesc memory{
                .before = {render::PipelineStageBits::ComputeShader, render::AccessBits::ShaderWrite},
                .after = {render::PipelineStageBits::ComputeShader, render::AccessBits::ShaderRead | render::AccessBits::ShaderWrite}};
            const render::BarrierDesc barrier{.memory = {&memory, 1}};
            REG_REQUIRE(recording.begin(0));
            render::ComputeDispatchDesc dispatch{
                .commandBuffer = recording.commands.get(),
                .bindings = {&binding, 1},
                .indirectArguments = arguments.get(),
            };
            REG_REQUIRE(program.dispatchIndirectBatch(dispatch, items, barrier));
            REG_CHECK(recording.commands->synchronizationStats().memoryBarriers == 1);
            REG_REQUIRE(recording.submit(tracker, *gate));
            REG_REQUIRE(recording.frame.wait());
            output->invalidate();
            const auto* value = static_cast<const uint32_t*>(output->map());
            REG_CHECK(value);
            const auto actual = *value;
            bench::readbackEvidence(context, "readback.bin", std::span<const uint32_t>(value, 16));
            output->unmap();
            REG_CHECK(actual == 2);

            REG_REQUIRE(recording.begin(1));
            memory.after = {render::PipelineStageBits::Transfer, render::AccessBits::ShaderRead};
            REG_CHECK(render::hasError(program.dispatchIndirectBatch(dispatch, items, barrier), render::Error::InvalidArgument));
            REG_CHECK(recording.commands->synchronizationStats().calls == 0);
            recording.frame.cancel(); // Discard the first dispatch of the rejected batch.
        }
        return RHITestResult::pass("Prepared mapped/native: memory-only ordering and barrier errors");
    }
};
METALLIC_REGISTER_RHI_TEST(BatchMemoryBarrierTest);

#undef REG_REQUIRE
#undef REG_CHECK
} // namespace
} // namespace metallic::tests
