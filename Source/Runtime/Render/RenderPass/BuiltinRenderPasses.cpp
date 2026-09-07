#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"

namespace metallic::render {

void registerBuiltInRenderGraphPasses()
{
    static bool registered = false;
    if (registered) {
        return;
    }
    registered = true;

    registerRenderGraphPassType(
        "LightGridDebugPass",
        "Visualize clustered light coverage, depth slices and list overflow with an asset-free test bench",
        []() { return builtin_pass::createLightGridDebugPass(); });
    registerRenderGraphPassType(
        "ClearColorPass",
        "Clear a color texture",
        []() { return builtin_pass::createClearColorPass(); });
    registerRenderGraphPassType(
        "CopyColorPass",
        "Copy a color texture",
        []() { return builtin_pass::createCopyColorPass(); });
    registerRenderGraphPassType(
        "AutoExposurePass",
        "Histogram EV100 exposure, temporal eye adaptation and HDR tone mapping",
        []() { return builtin_pass::createAutoExposurePass(); });
    registerRenderGraphPassType(
        "SliderDebugPass",
        "Reveal two pixel-aligned color paths with an interactive divider, preserving HDR",
        []() { return builtin_pass::createSliderDebugPass(); });
    registerRenderGraphPassType(
        "FinalBlitPass",
        "Present a color texture; show a UV gradient when no usable source is connected",
        []() { return builtin_pass::createFinalBlitPass(); });
    registerRenderGraphPassType(
        "TriangleRasterPass",
        "Rasterize the built-in triangle shader",
        []() { return builtin_pass::createTriangleRasterPass(); });
    registerRenderGraphPassType(
        "ImageSamplePass",
        "Draw a fullscreen sampled image",
        []() { return builtin_pass::createImageSamplePass(); });
    registerRenderGraphPassType(
        "BunnyWireframePass",
        "Draw the Stanford Bunny glTF as a barycentric wireframe",
        []() { return builtin_pass::createBunnyWireframePass(); });
    registerRenderGraphPassType(
        "SceneMaterialShaderObjectPass",
        "Draw glTF material colors with VK_EXT_shader_object",
        []() { return builtin_pass::createSceneMaterialShaderObjectPass(); });
    registerRenderGraphPassType(
        "SceneMaterialVisualizationPass",
        "Visualize glTF material parameters with RayQuery",
        []() { return builtin_pass::createSceneMaterialVisualizationPass(); });
    registerRenderGraphPassType(
        "SceneRayQueryVisualizationPass",
        "Visualize a glTF acceleration structure with RayQuery",
        []() { return builtin_pass::createSceneRayQueryVisualizationPass(); });
    registerRenderGraphPassType(
        "VisibilityBufferPass",
        "Rasterize raw visibility/depth with Wave32 AS/MS culling and optional ID/depth visualization",
        []() { return builtin_pass::createVisibilityBufferPass(); });
    registerRenderGraphPassType(
        "GPUDrivenStreamAssetPass",
        "Stream and visualize meshlet streamasset pages with mesh shaders",
        []() { return builtin_pass::createGPUDrivenStreamAssetPass(); });
    registerRenderGraphPassType(
        "SceneRealtimeLightingPass",
        "Real-time photometric punctual lights, ray-query shadows and SH environment GI",
        []() { return builtin_pass::createSceneRealtimeLightingPass(); });
    registerRenderGraphPassType(
        "ScenePathTracePass",
        "Path trace a glTF scene with RayQuery",
        []() { return builtin_pass::createScenePathTracePass(); });
    registerRenderGraphPassType(
        "SceneRtxdiPass",
        "Render many-light direct illumination with ReSTIR DI reservoir resampling",
        []() { return builtin_pass::createSceneRtxdiPass(); });
    registerRenderGraphPassType(
        "RtxdiConfidencePass",
        "Convert RTXDI temporal lighting gradients into NRD confidence inputs",
        []() { return builtin_pass::createRtxdiConfidencePass(); });
    registerRenderGraphPassType(
        "RtxdiCompositePass",
        "Composite NRD-denoised RTXDI diffuse and specular lighting",
        []() { return builtin_pass::createRtxdiCompositePass(); });
    registerRenderGraphPassType(
        "RtxcrMaterialSamplePass",
        "Visualize RTXCR Chiang hair, far-field hair, and subsurface material models",
        []() { return builtin_pass::createRtxcrMaterialSamplePass(); });
    registerRenderGraphPassType(
        "NrdDenoisePass",
        "Denoise connected NRD radiance resources",
        []() { return builtin_pass::createNrdDenoisePass(); });
    registerRenderGraphPassType(
        "StreamlineDlssSrPass",
        "Upscale an HDR color target with NVIDIA DLSS Super Resolution",
        []() { return builtin_pass::createStreamlineDlssSrPass(); });
    registerRenderGraphPassType(
        "StreamlineDlssRrPass",
        "Denoise a path traced HDR color target with NVIDIA DLSS Ray Reconstruction",
        []() { return builtin_pass::createStreamlineDlssRrPass(); });
    registerRenderGraphPassType(
        "RenderGraphBufferWritePass",
        "Write a known byte pattern into a graph buffer",
        []() { return builtin_pass::createRenderGraphBufferWritePass(); });
    registerRenderGraphPassType(
        "RenderGraphBufferCopyPass",
        "Copy a graph buffer through bindless compute",
        []() { return builtin_pass::createRenderGraphBufferCopyPass(); });
}

} // namespace metallic::render
