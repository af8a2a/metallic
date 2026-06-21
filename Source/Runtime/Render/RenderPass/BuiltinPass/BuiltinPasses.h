#pragma once

#include <memory>

namespace metallic::render {

class RenderGraphPass;

namespace builtin_pass {

std::unique_ptr<RenderGraphPass> createClearColorPass();
std::unique_ptr<RenderGraphPass> createCopyColorPass();
std::unique_ptr<RenderGraphPass> createTriangleRasterPass();
std::unique_ptr<RenderGraphPass> createImageSamplePass();
std::unique_ptr<RenderGraphPass> createBunnyWireframePass();
std::unique_ptr<RenderGraphPass> createSceneMaterialShaderObjectPass();
std::unique_ptr<RenderGraphPass> createSceneMaterialVisualizationPass();
std::unique_ptr<RenderGraphPass> createSceneRayQueryVisualizationPass();
std::unique_ptr<RenderGraphPass> createGPUDrivenPreviewPass();
std::unique_ptr<RenderGraphPass> createScenePathTracePass();
std::unique_ptr<RenderGraphPass> createNrdDenoisePass();
std::unique_ptr<RenderGraphPass> createStreamlineDlssRrPass();
std::unique_ptr<RenderGraphPass> createRenderGraphBufferWritePass();
std::unique_ptr<RenderGraphPass> createRenderGraphBufferCopyPass();

} // namespace builtin_pass
} // namespace metallic::render
