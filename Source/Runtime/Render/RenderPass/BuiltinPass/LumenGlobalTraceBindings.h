#pragma once

#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"

#include <optional>
#include <spdlog/spdlog.h>

namespace metallic::render::builtin_pass {

constexpr uint32_t kLumenGlobalTraceDomainStride = 32;
constexpr uint32_t kLumenGlobalTraceCardStride = 80;

struct ShaderLumenGlobalTraceBindings {
    ShaderSampledImage pageTable, pageAtlas, coarseMip;
    ShaderBuffer domains, objectGrid, objectGridScene, cards, cardScene;
    ShaderSampledImage cardDepth, cardLighting;
    ShaderSampler sampler;
};
static_assert(sizeof(ShaderLumenGlobalTraceBindings) == 44);

inline void reflectLumenGlobalTraceInputs(RenderPassReflection& reflection)
{
    reflection.addTextureInput("pageTable", "GDF page table").texture3D(0, 0, 0).sampledRead().format = Format::R32Uint;
    reflection.addTextureInput("pageAtlas", "GDF fine page distances").texture3D(0, 0, 0).sampledRead().format = Format::R8Unorm;
    reflection.addTextureInput("coarseMip", "GDF coarse distances").texture3D(0, 0, 0).sampledRead().format = Format::R8Unorm;
    reflection.addBufferInput("domain", "GDF clipmap domains").buffer(0, kLumenGlobalTraceDomainStride).storageRead();
    reflection.addBufferInput("objectGrid", "Candidate scene objects per GDF page cell").buffer(0, 16).storageRead();
    reflection.addBufferInput("objectGridScene", "GDF scene resource identity").buffer(8, 4).storageRead();
    reflection.addBufferInput("cards", "Surface Cache card transforms").buffer(0, kLumenGlobalTraceCardStride).storageRead();
    reflection.addBufferInput("cardScene", "Surface Cache atlas metadata and scene identity").buffer(16, 4).storageRead();
    auto& cardDepth = reflection.addTextureInput("cardDepth", "Surface Cache depth").sampledRead();
    cardDepth.format = Format::R16Unorm;
    cardDepth.matchOutputExtent = false;
    auto& cardLighting = reflection.addTextureInput("cardLighting", "Surface Cache outgoing radiance").sampledRead();
    cardLighting.format = Format::RGBA16Sfloat;
    cardLighting.matchOutputExtent = false;
}

inline std::optional<uint32_t> lumenGlobalTraceClipmapCount(const RenderGraphExecutionContext& context)
{
    const auto pageTable = context.inputTexture("pageTable");
    const auto coarseMip = context.inputTexture("coarseMip");
    const auto domains = context.inputBuffer("domain");
    const auto& table = pageTable.desc();
    const auto& coarse = coarseMip.desc();
    const uint32_t clipmapCount = coarse.depth / coarse.width;
    if (clipmapCount < 1 || clipmapCount > 6 || coarse.width != coarse.height ||
        coarse.depth != coarse.width * clipmapCount || table.width != table.height ||
        table.depth != table.width * clipmapCount || domains.desc().size != clipmapCount * kLumenGlobalTraceDomainStride) {
        spdlog::error("Lumen Global Tracing requires matching quantized GDF clipmap resources");
        return std::nullopt;
    }

    return clipmapCount;
}

inline ShaderLumenGlobalTraceBindings bindLumenGlobalTrace(const RenderGraphExecutionContext& context,
    ParameterWriter& writer)
{
    ShaderLumenGlobalTraceBindings bindings{};
    bindings.pageTable = writer.sampledImageHandle(context.inputTexture("pageTable").view());
    bindings.pageAtlas = writer.sampledImageHandle(context.inputTexture("pageAtlas").view());
    bindings.coarseMip = writer.sampledImageHandle(context.inputTexture("coarseMip").view());
    bindings.domains = writer.buffer(context.inputBuffer("domain").buffer());
    bindings.objectGrid = writer.buffer(context.inputBuffer("objectGrid").buffer());
    bindings.objectGridScene = writer.buffer(context.inputBuffer("objectGridScene").buffer());
    bindings.cards = writer.buffer(context.inputBuffer("cards").buffer());
    bindings.cardScene = writer.buffer(context.inputBuffer("cardScene").buffer());
    bindings.cardDepth = writer.sampledImageHandle(context.inputTexture("cardDepth").view());
    bindings.cardLighting = writer.sampledImageHandle(context.inputTexture("cardLighting").view());
    bindings.sampler = writer.samplerHandle({});
    return bindings;
}

} // namespace metallic::render::builtin_pass
