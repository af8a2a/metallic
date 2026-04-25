#include "imgui_metal_bridge.h"

#include "metal_resource_utils.h"
#include "metal_runtime.h"

#include "imgui.h"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

struct ImGuiMetal4State {
    MTL::Device* device = nullptr;
    MTL::Texture* fontTexture = nullptr;
    MTL::SamplerState* sampler = nullptr;
    MTL::DepthStencilState* depthStencilState = nullptr;
    MTL::RenderPipelineState* pipelineState = nullptr;
    MTL::PixelFormat colorFormat = MTL::PixelFormatInvalid;
    MTL::PixelFormat depthFormat = MTL::PixelFormatInvalid;
    uint32_t framebufferWidth = 0;
    uint32_t framebufferHeight = 0;
};

ImGuiMetal4State g_imguiMetal4;

MTL::Device* metalDevice(void* handle) {
    return static_cast<MTL::Device*>(handle);
}

MTL::Texture* metalTexture(void* handle) {
    return static_cast<MTL::Texture*>(handle);
}

MTL4::CommandBuffer* metalCommandBuffer(void* handle) {
    return static_cast<MTL4::CommandBuffer*>(handle);
}

MTL4::RenderCommandEncoder* metalRenderEncoder(void* handle) {
    return static_cast<MTL4::RenderCommandEncoder*>(handle);
}

MTL::ResourceID nullResourceID() {
    return MTL::ResourceID{0};
}

bool createFontTexture() {
    if (!g_imguiMetal4.device || g_imguiMetal4.fontTexture) {
        return g_imguiMetal4.fontTexture != nullptr;
    }

    ImGuiIO& io = ImGui::GetIO();
    unsigned char* pixels = nullptr;
    int width = 0;
    int height = 0;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
    if (!pixels || width <= 0 || height <= 0) {
        return false;
    }

    auto* desc = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatRGBA8Unorm,
                                                             static_cast<NS::UInteger>(width),
                                                             static_cast<NS::UInteger>(height),
                                                             false);
    desc->setUsage(MTL::TextureUsageShaderRead);
    desc->setStorageMode(MTL::StorageModeShared);

    auto* texture = g_imguiMetal4.device->newTexture(desc);
    if (!texture) {
        return false;
    }
    texture->setLabel(NS::String::string("ImGui Font Texture", NS::UTF8StringEncoding));
    texture->replaceRegion(MTL::Region(0, 0, 0,
                                        static_cast<NS::UInteger>(width),
                                        static_cast<NS::UInteger>(height),
                                        1),
                           0,
                           pixels,
                           static_cast<NS::UInteger>(width) * 4);
    metalTrackAllocation(g_imguiMetal4.device, texture);
    g_imguiMetal4.fontTexture = texture;
    io.Fonts->SetTexID(static_cast<ImTextureID>(reinterpret_cast<uintptr_t>(texture)));
    return true;
}

bool createSampler() {
    if (!g_imguiMetal4.device || g_imguiMetal4.sampler) {
        return g_imguiMetal4.sampler != nullptr;
    }

    auto* desc = MTL::SamplerDescriptor::alloc()->init();
    desc->setMinFilter(MTL::SamplerMinMagFilterLinear);
    desc->setMagFilter(MTL::SamplerMinMagFilterLinear);
    desc->setMipFilter(MTL::SamplerMipFilterLinear);
    desc->setSAddressMode(MTL::SamplerAddressModeClampToEdge);
    desc->setTAddressMode(MTL::SamplerAddressModeClampToEdge);
    desc->setRAddressMode(MTL::SamplerAddressModeClampToEdge);
    g_imguiMetal4.sampler = g_imguiMetal4.device->newSamplerState(desc);
    desc->release();
    return g_imguiMetal4.sampler != nullptr;
}

bool createDepthStencilState() {
    if (!g_imguiMetal4.device || g_imguiMetal4.depthStencilState) {
        return g_imguiMetal4.depthStencilState != nullptr;
    }

    auto* desc = MTL::DepthStencilDescriptor::alloc()->init();
    desc->setDepthCompareFunction(MTL::CompareFunctionAlways);
    desc->setDepthWriteEnabled(false);
    g_imguiMetal4.depthStencilState = g_imguiMetal4.device->newDepthStencilState(desc);
    desc->release();
    return g_imguiMetal4.depthStencilState != nullptr;
}

MTL::Library* createShaderLibrary() {
    static constexpr const char* kShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

struct Uniforms {
    float4x4 projectionMatrix;
};

struct VertexIn {
    float2 position  [[attribute(0)]];
    float2 texCoords [[attribute(1)]];
    uchar4 color     [[attribute(2)]];
};

struct VertexOut {
    float4 position [[position]];
    float2 texCoords;
    float4 color;
};

vertex VertexOut vertex_main(VertexIn in [[stage_in]],
                             constant Uniforms& uniforms [[buffer(1)]]) {
    VertexOut out;
    out.position = uniforms.projectionMatrix * float4(in.position, 0.0, 1.0);
    out.texCoords = in.texCoords;
    out.color = float4(in.color) / float4(255.0);
    return out;
}

fragment half4 fragment_main(VertexOut in [[stage_in]],
                             texture2d<half, access::sample> texture [[texture(0)]],
                             sampler textureSampler [[sampler(0)]]) {
    return half4(in.color) * texture.sample(textureSampler, in.texCoords);
}
)";

    NS::Error* error = nullptr;
    auto* source = NS::String::string(kShaderSource, NS::UTF8StringEncoding);
    return g_imguiMetal4.device->newLibrary(source, nullptr, &error);
}

bool ensurePipeline() {
    if (!g_imguiMetal4.device ||
        g_imguiMetal4.colorFormat == MTL::PixelFormatInvalid) {
        return false;
    }
    if (g_imguiMetal4.pipelineState) {
        return true;
    }

    auto* library = createShaderLibrary();
    if (!library) {
        return false;
    }

    auto* vertexFunction = library->newFunction(NS::String::string("vertex_main", NS::UTF8StringEncoding));
    auto* fragmentFunction = library->newFunction(NS::String::string("fragment_main", NS::UTF8StringEncoding));
    if (!vertexFunction || !fragmentFunction) {
        if (vertexFunction) vertexFunction->release();
        if (fragmentFunction) fragmentFunction->release();
        library->release();
        return false;
    }

    auto* vertexDescriptor = MTL::VertexDescriptor::alloc()->init();
    vertexDescriptor->attributes()->object(0)->setOffset(offsetof(ImDrawVert, pos));
    vertexDescriptor->attributes()->object(0)->setFormat(MTL::VertexFormatFloat2);
    vertexDescriptor->attributes()->object(0)->setBufferIndex(0);
    vertexDescriptor->attributes()->object(1)->setOffset(offsetof(ImDrawVert, uv));
    vertexDescriptor->attributes()->object(1)->setFormat(MTL::VertexFormatFloat2);
    vertexDescriptor->attributes()->object(1)->setBufferIndex(0);
    vertexDescriptor->attributes()->object(2)->setOffset(offsetof(ImDrawVert, col));
    vertexDescriptor->attributes()->object(2)->setFormat(MTL::VertexFormatUChar4);
    vertexDescriptor->attributes()->object(2)->setBufferIndex(0);
    vertexDescriptor->layouts()->object(0)->setStride(sizeof(ImDrawVert));
    vertexDescriptor->layouts()->object(0)->setStepRate(1);
    vertexDescriptor->layouts()->object(0)->setStepFunction(MTL::VertexStepFunctionPerVertex);

    auto* pipelineDesc = MTL::RenderPipelineDescriptor::alloc()->init();
    pipelineDesc->setVertexFunction(vertexFunction);
    pipelineDesc->setFragmentFunction(fragmentFunction);
    pipelineDesc->setVertexDescriptor(vertexDescriptor);
    pipelineDesc->colorAttachments()->object(0)->setPixelFormat(g_imguiMetal4.colorFormat);
    pipelineDesc->colorAttachments()->object(0)->setBlendingEnabled(true);
    pipelineDesc->colorAttachments()->object(0)->setRgbBlendOperation(MTL::BlendOperationAdd);
    pipelineDesc->colorAttachments()->object(0)->setAlphaBlendOperation(MTL::BlendOperationAdd);
    pipelineDesc->colorAttachments()->object(0)->setSourceRGBBlendFactor(MTL::BlendFactorSourceAlpha);
    pipelineDesc->colorAttachments()->object(0)->setSourceAlphaBlendFactor(MTL::BlendFactorOne);
    pipelineDesc->colorAttachments()->object(0)->setDestinationRGBBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
    pipelineDesc->colorAttachments()->object(0)->setDestinationAlphaBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
    if (g_imguiMetal4.depthFormat != MTL::PixelFormatInvalid) {
        pipelineDesc->setDepthAttachmentPixelFormat(g_imguiMetal4.depthFormat);
    }

    NS::Error* error = nullptr;
    g_imguiMetal4.pipelineState = g_imguiMetal4.device->newRenderPipelineState(pipelineDesc, &error);

    pipelineDesc->release();
    vertexDescriptor->release();
    vertexFunction->release();
    fragmentFunction->release();
    library->release();
    return g_imguiMetal4.pipelineState != nullptr;
}

void invalidatePipelineIfFormatChanged(MTL::PixelFormat colorFormat, MTL::PixelFormat depthFormat) {
    if (g_imguiMetal4.colorFormat == colorFormat &&
        g_imguiMetal4.depthFormat == depthFormat) {
        return;
    }

    if (g_imguiMetal4.pipelineState) {
        g_imguiMetal4.pipelineState->release();
        g_imguiMetal4.pipelineState = nullptr;
    }
    g_imguiMetal4.colorFormat = colorFormat;
    g_imguiMetal4.depthFormat = depthFormat;
}

MTL::Texture* textureFromImGuiId(ImTextureID textureId) {
    if (textureId == ImTextureID_Invalid || textureId == 0) {
        return g_imguiMetal4.fontTexture;
    }
    return reinterpret_cast<MTL::Texture*>(static_cast<uintptr_t>(textureId));
}

void setupRenderState(ImDrawData* drawData,
                      MTL4::RenderCommandEncoder* encoder,
                      MTL4::ArgumentTable* vertexTable,
                      MTL4::ArgumentTable* fragmentTable,
                      MTL::GPUAddress vertexAddress,
                      MTL::GPUAddress projectionAddress) {
    encoder->setCullMode(MTL::CullModeNone);
    encoder->setDepthStencilState(g_imguiMetal4.depthStencilState);
    encoder->setRenderPipelineState(g_imguiMetal4.pipelineState);

    MTL::Viewport viewport{};
    viewport.originX = 0.0;
    viewport.originY = 0.0;
    viewport.width = drawData->DisplaySize.x * drawData->FramebufferScale.x;
    viewport.height = drawData->DisplaySize.y * drawData->FramebufferScale.y;
    viewport.znear = 0.0;
    viewport.zfar = 1.0;
    encoder->setViewport(viewport);

    if (vertexTable) {
        vertexTable->setAddress(vertexAddress, sizeof(ImDrawVert), 0);
        vertexTable->setAddress(projectionAddress, 1);
        encoder->setArgumentTable(vertexTable, MTL::RenderStageVertex);
    }
    if (fragmentTable) {
        fragmentTable->setSamplerState(g_imguiMetal4.sampler ? g_imguiMetal4.sampler->gpuResourceID() : nullResourceID(), 0);
        encoder->setArgumentTable(fragmentTable, MTL::RenderStageFragment);
    }
}

} // namespace

extern "C" void imguiInit(void* mtlDevice) {
    if (g_imguiMetal4.device) {
        return;
    }

    g_imguiMetal4 = {};
    g_imguiMetal4.device = metalDevice(mtlDevice);
    if (g_imguiMetal4.device) {
        g_imguiMetal4.device->retain();
    }

    ImGuiIO& io = ImGui::GetIO();
    io.BackendRendererName = "metallic_metal4_imgui";
    io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;

    createSampler();
    createDepthStencilState();
    createFontTexture();
}

extern "C" void imguiNewFrame(void* /*mtlRenderPassDescriptor*/) {
    createFontTexture();
}

extern "C" void imguiNewFrameForTargets(void* colorTextureHandle, void* depthTextureHandle) {
    auto* colorTexture = metalTexture(colorTextureHandle);
    auto* depthTexture = metalTexture(depthTextureHandle);
    const MTL::PixelFormat colorFormat = colorTexture ? colorTexture->pixelFormat() : MTL::PixelFormatInvalid;
    const MTL::PixelFormat depthFormat = depthTexture ? depthTexture->pixelFormat() : MTL::PixelFormatInvalid;
    invalidatePipelineIfFormatChanged(colorFormat, depthFormat);

    g_imguiMetal4.framebufferWidth = colorTexture ? static_cast<uint32_t>(colorTexture->width()) : 0;
    g_imguiMetal4.framebufferHeight = colorTexture ? static_cast<uint32_t>(colorTexture->height()) : 0;
    createFontTexture();
    ensurePipeline();
}

extern "C" void imguiRenderDrawData(void* mtlCommandBuffer, void* mtlRenderCommandEncoder) {
    ImDrawData* drawData = ImGui::GetDrawData();
    auto* commandBuffer = metalCommandBuffer(mtlCommandBuffer);
    auto* encoder = metalRenderEncoder(mtlRenderCommandEncoder);
    if (!drawData || !commandBuffer || !encoder) {
        return;
    }

    const int framebufferWidth = static_cast<int>(drawData->DisplaySize.x * drawData->FramebufferScale.x);
    const int framebufferHeight = static_cast<int>(drawData->DisplaySize.y * drawData->FramebufferScale.y);
    if (framebufferWidth <= 0 || framebufferHeight <= 0 || drawData->CmdListsCount == 0) {
        return;
    }

    if (!createFontTexture() || !createSampler() || !createDepthStencilState() || !ensurePipeline()) {
        return;
    }

    const size_t vertexBufferSize = static_cast<size_t>(drawData->TotalVtxCount) * sizeof(ImDrawVert);
    const size_t indexBufferSize = static_cast<size_t>(drawData->TotalIdxCount) * sizeof(ImDrawIdx);
    if (vertexBufferSize == 0 || indexBufferSize == 0) {
        return;
    }

    MetalUploadAllocation vertexAllocation;
    MetalUploadAllocation indexAllocation;
    MetalUploadAllocation projectionAllocation;
    std::vector<uint8_t> vertexScratch(vertexBufferSize);
    std::vector<uint8_t> indexScratch(indexBufferSize);
    uint8_t* vertexWrite = vertexScratch.data();
    uint8_t* indexWrite = indexScratch.data();

    for (int listIndex = 0; listIndex < drawData->CmdListsCount; ++listIndex) {
        const ImDrawList* drawList = drawData->CmdLists[listIndex];
        const size_t vertexBytes = static_cast<size_t>(drawList->VtxBuffer.Size) * sizeof(ImDrawVert);
        const size_t indexBytes = static_cast<size_t>(drawList->IdxBuffer.Size) * sizeof(ImDrawIdx);
        std::memcpy(vertexWrite, drawList->VtxBuffer.Data, vertexBytes);
        std::memcpy(indexWrite, drawList->IdxBuffer.Data, indexBytes);
        vertexWrite += vertexBytes;
        indexWrite += indexBytes;
    }

    if (!metalRuntimeUploadBytes(commandBuffer, vertexScratch.data(), vertexScratch.size(), 256, vertexAllocation) ||
        !metalRuntimeUploadBytes(commandBuffer, indexScratch.data(), indexScratch.size(), 256, indexAllocation)) {
        return;
    }

    const float left = drawData->DisplayPos.x;
    const float right = drawData->DisplayPos.x + drawData->DisplaySize.x;
    const float top = drawData->DisplayPos.y;
    const float bottom = drawData->DisplayPos.y + drawData->DisplaySize.y;
    const float projection[4][4] = {
        { 2.0f / (right - left), 0.0f, 0.0f, 0.0f },
        { 0.0f, 2.0f / (top - bottom), 0.0f, 0.0f },
        { 0.0f, 0.0f, 1.0f, 0.0f },
        { (right + left) / (left - right), (top + bottom) / (bottom - top), 0.0f, 1.0f },
    };
    if (!metalRuntimeUploadBytes(commandBuffer, projection, sizeof(projection), 256, projectionAllocation)) {
        return;
    }

    auto* vertexTable = static_cast<MTL4::ArgumentTable*>(
        metalRuntimeArgumentTable(commandBuffer, MetalArgumentTableSlot::Vertex));
    auto* fragmentTable = static_cast<MTL4::ArgumentTable*>(
        metalRuntimeArgumentTable(commandBuffer, MetalArgumentTableSlot::Fragment));

    setupRenderState(drawData,
                     encoder,
                     vertexTable,
                     fragmentTable,
                     vertexAllocation.gpuAddress,
                     projectionAllocation.gpuAddress);

    const ImVec2 clipOffset = drawData->DisplayPos;
    const ImVec2 clipScale = drawData->FramebufferScale;
    uint64_t globalVertexOffset = 0;
    uint64_t globalIndexOffset = 0;

    for (int listIndex = 0; listIndex < drawData->CmdListsCount; ++listIndex) {
        const ImDrawList* drawList = drawData->CmdLists[listIndex];
        for (int commandIndex = 0; commandIndex < drawList->CmdBuffer.Size; ++commandIndex) {
            const ImDrawCmd* drawCommand = &drawList->CmdBuffer[commandIndex];
            if (drawCommand->ElemCount == 0) {
                continue;
            }

            if (drawCommand->UserCallback) {
                if (drawCommand->UserCallback == ImDrawCallback_ResetRenderState) {
                    setupRenderState(drawData,
                                     encoder,
                                     vertexTable,
                                     fragmentTable,
                                     vertexAllocation.gpuAddress,
                                     projectionAllocation.gpuAddress);
                } else {
                    drawCommand->UserCallback(drawList, drawCommand);
                }
                continue;
            }

            ImVec2 clipMin((drawCommand->ClipRect.x - clipOffset.x) * clipScale.x,
                           (drawCommand->ClipRect.y - clipOffset.y) * clipScale.y);
            ImVec2 clipMax((drawCommand->ClipRect.z - clipOffset.x) * clipScale.x,
                           (drawCommand->ClipRect.w - clipOffset.y) * clipScale.y);
            clipMin.x = std::max(clipMin.x, 0.0f);
            clipMin.y = std::max(clipMin.y, 0.0f);
            clipMax.x = std::min(clipMax.x, static_cast<float>(framebufferWidth));
            clipMax.y = std::min(clipMax.y, static_cast<float>(framebufferHeight));
            if (clipMax.x <= clipMin.x || clipMax.y <= clipMin.y) {
                continue;
            }

            MTL::ScissorRect scissor{};
            scissor.x = static_cast<NS::UInteger>(clipMin.x);
            scissor.y = static_cast<NS::UInteger>(clipMin.y);
            scissor.width = static_cast<NS::UInteger>(clipMax.x - clipMin.x);
            scissor.height = static_cast<NS::UInteger>(clipMax.y - clipMin.y);
            encoder->setScissorRect(scissor);

            auto* drawTexture = textureFromImGuiId(drawCommand->GetTexID());
            if (fragmentTable) {
                fragmentTable->setTexture(drawTexture ? drawTexture->gpuResourceID() : nullResourceID(), 0);
                encoder->setArgumentTable(fragmentTable, MTL::RenderStageFragment);
            }

            if (vertexTable) {
                const uint64_t vertexOffset =
                    (globalVertexOffset + drawCommand->VtxOffset) * sizeof(ImDrawVert);
                vertexTable->setAddress(vertexAllocation.gpuAddress + vertexOffset,
                                        sizeof(ImDrawVert),
                                        0);
                encoder->setArgumentTable(vertexTable, MTL::RenderStageVertex);
            }

            const MTL::IndexType indexType = sizeof(ImDrawIdx) == 2 ? MTL::IndexTypeUInt16 : MTL::IndexTypeUInt32;
            const uint64_t indexOffset =
                indexAllocation.gpuAddress +
                (globalIndexOffset + drawCommand->IdxOffset) * sizeof(ImDrawIdx);
            encoder->drawIndexedPrimitives(MTL::PrimitiveTypeTriangle,
                                           drawCommand->ElemCount,
                                           indexType,
                                           indexOffset,
                                           drawCommand->ElemCount * sizeof(ImDrawIdx),
                                           1,
                                           0,
                                           0);
        }
        globalVertexOffset += static_cast<uint64_t>(drawList->VtxBuffer.Size);
        globalIndexOffset += static_cast<uint64_t>(drawList->IdxBuffer.Size);
    }
}

extern "C" void imguiShutdown(void) {
    if (g_imguiMetal4.pipelineState) {
        g_imguiMetal4.pipelineState->release();
    }
    if (g_imguiMetal4.depthStencilState) {
        g_imguiMetal4.depthStencilState->release();
    }
    if (g_imguiMetal4.sampler) {
        g_imguiMetal4.sampler->release();
    }
    if (g_imguiMetal4.fontTexture) {
        metalReleaseHandle(g_imguiMetal4.fontTexture);
    }
    if (g_imguiMetal4.device) {
        g_imguiMetal4.device->release();
    }

    ImGuiIO& io = ImGui::GetIO();
    io.BackendRendererName = nullptr;
    io.BackendFlags &= ~ImGuiBackendFlags_RendererHasVtxOffset;

    g_imguiMetal4 = {};
}
