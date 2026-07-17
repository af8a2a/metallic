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
        "ClearColorPass",
        "Clear a color texture",
        []() { return builtin_pass::createClearColorPass(); });
    registerRenderGraphPassType(
        "CopyColorPass",
        "Copy a color texture",
        []() { return builtin_pass::createCopyColorPass(); });
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
        "GPUDrivenPreviewPass",
        "Visualize glTF meshlet clusters with mesh shaders",
        []() { return builtin_pass::createGPUDrivenPreviewPass(); });
    registerRenderGraphPassType(
        "GPUDrivenStreamAssetPass",
        "Stream and visualize meshlet streamasset pages with mesh shaders",
        []() { return builtin_pass::createGPUDrivenStreamAssetPass(); });
    registerRenderGraphPassType(
        "ScenePathTracePass",
        "Path trace a glTF scene with RayQuery",
        []() { return builtin_pass::createScenePathTracePass(); });
    registerRenderGraphPassType(
        "SceneRtxdiPass",
        "Render many-light direct illumination with ReSTIR DI reservoir resampling",
        []() { return builtin_pass::createSceneRtxdiPass(); });
    registerRenderGraphPassType(
        "RtxdiCompositePass",
        "Composite NRD-denoised RTXDI diffuse and specular lighting",
        []() { return builtin_pass::createRtxdiCompositePass(); });
    registerRenderGraphPassType(
        "NrdDenoisePass",
        "Denoise connected NRD radiance resources",
        []() { return builtin_pass::createNrdDenoisePass(); });
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
