#include "pipeline_builder.h"

#include "frame_context.h"
#include "render_pass.h"

#include <string_view>

#include <spdlog/spdlog.h>

namespace {

struct ProducerBindingInfo {
    const PassDecl* pass = nullptr;
    const EdgeDecl* edge = nullptr;
};

ProducerBindingInfo findProducerBinding(const PipelineAsset& asset, const std::string& resourceId) {
    ProducerBindingInfo result;
    for (const auto& pass : asset.passes) {
        if (!pass.enabled) {
            continue;
        }

        for (const auto& edge : asset.edges) {
            if (edge.passId != pass.id ||
                edge.direction != "output" ||
                edge.resourceId != resourceId) {
                continue;
            }

            result.pass = &pass;
            result.edge = &edge;
        }
    }
    return result;
}

void appendBindings(const PipelineAsset& asset,
                    const PassDecl& passDecl,
                    const std::vector<PassSlotInfo>& slots,
                    const char* direction,
                    std::vector<PassResourceBinding>& outBindings) {
    outBindings.clear();
    outBindings.reserve(slots.size());
    const bool isInputBinding = std::string_view(direction) == "input";
    for (const auto& slot : slots) {
        const EdgeDecl* edge = asset.findEdge(passDecl.id, direction, slot.key);
        if (!edge) {
            continue;
        }
        const ResourceDecl* resource = asset.findResourceById(edge->resourceId);
        if (!resource) {
            continue;
        }

        PassResourceBinding binding;
        binding.slotKey = slot.key;
        binding.resourceId = resource->id;
        binding.resourceName = resource->name;

        if (isInputBinding) {
            const ProducerBindingInfo producer = findProducerBinding(asset, resource->id);
            if (producer.pass) {
                binding.producerPassId = producer.pass->id;
                binding.producerPassType = producer.pass->type;
            }
            if (producer.edge) {
                binding.producerSlotKey = producer.edge->slotKey;
            }
        }

        outBindings.push_back(std::move(binding));
    }
}

} // namespace

PipelineBuilder::PipelineBuilder(const RenderContext& ctx)
    : m_ctx(ctx) {}

bool PipelineBuilder::needsRebuild(int width, int height) const {
    return !m_built || m_builtWidth != width || m_builtHeight != height;
}

bool PipelineBuilder::build(const PipelineAsset& asset,
                            const PipelineRuntimeContext& rtCtx,
                            int width,
                            int height) {
    m_lastError.clear();

    if (!asset.validate(m_lastError)) {
        return false;
    }

    m_fg = FrameGraph{};
    m_resourceMap.clear();
    m_passes.clear();
    m_backbufferRes = FGResource{};

    std::unordered_map<std::string, FGResource> importedByKey;
    for (const auto& resource : asset.resources) {
        if (resource.kind == "backbuffer") {
            if (!rtCtx.backbufferRhi) {
                continue;
            }
            if (!m_backbufferRes.isValid()) {
                m_backbufferRes = m_fg.import(resource.name.c_str(), rtCtx.backbufferRhi);
            }
            m_resourceMap[resource.id] = m_backbufferRes;
            continue;
        }

        if (resource.kind != "imported") {
            continue;
        }

        auto cachedIt = importedByKey.find(resource.importKey);
        if (cachedIt != importedByKey.end()) {
            m_resourceMap[resource.id] = cachedIt->second;
            continue;
        }

        auto runtimeIt = rtCtx.importedTexturesRhi.find(resource.importKey);
        if (runtimeIt == rtCtx.importedTexturesRhi.end() || !runtimeIt->second.nativeHandle()) {
            continue;
        }

        FGResource imported = m_fg.import(resource.name.c_str(),
                                          const_cast<RhiTextureHandle*>(&runtimeIt->second));
        importedByKey[resource.importKey] = imported;
        m_resourceMap[resource.id] = imported;
    }

    const auto sortedOrder = asset.topologicalSort(false);
    for (const size_t idx : sortedOrder) {
        const PassDecl& passDecl = asset.passes[idx];
        if (!passDecl.enabled) {
            spdlog::debug("PipelineBuilder: skipping disabled pass '{}'", passDecl.name);
            continue;
        }

        const PassTypeInfo* typeInfo = PassRegistry::instance().getTypeInfo(passDecl.type);
        if (!typeInfo) {
            m_lastError = "Unknown pass type '" + passDecl.type + "'";
            m_built = false;
            return false;
        }

        PassConfig config;
        config.name = passDecl.name;
        config.type = passDecl.type;
        config.enabled = passDecl.enabled;
        config.sideEffect = passDecl.sideEffect;
        config.config = passDecl.config;
        appendBindings(asset, passDecl, typeInfo->inputSlots, "input", config.inputBindings);
        appendBindings(asset, passDecl, typeInfo->outputSlots, "output", config.outputBindings);

        auto pass = PassRegistry::instance().create(passDecl.type, config, m_ctx, width, height);
        if (!pass) {
            m_lastError = "Failed to create pass of type '" + passDecl.type + "'";
            m_built = false;
            return false;
        }

        pass->configure(config);
        pass->setSideEffectEnabled(config.sideEffect);
        pass->setRuntimeContext(&rtCtx);

        for (const auto& binding : config.inputBindings) {
            auto resIt = m_resourceMap.find(binding.resourceId);
            if (resIt == m_resourceMap.end() || !resIt->second.isValid()) {
                m_lastError = "Pass '" + passDecl.name + "' could not resolve input slot '" +
                              binding.slotKey + "' to resource '" + binding.resourceName + "'";
                m_built = false;
                return false;
            }
            pass->setInput(binding.slotKey, resIt->second);
        }

        for (const auto& binding : config.outputBindings) {
            const ResourceDecl* resource = asset.findResourceById(binding.resourceId);
            if (!resource || resource->kind == "transient") {
                continue;
            }

            auto resIt = m_resourceMap.find(binding.resourceId);
            if (resIt == m_resourceMap.end() || !resIt->second.isValid()) {
                m_lastError = "Pass '" + passDecl.name + "' could not resolve output target slot '" +
                              binding.slotKey + "' to resource '" + binding.resourceName + "'";
                m_built = false;
                return false;
            }
            pass->setOutputTarget(binding.slotKey, resIt->second);
        }

        RenderPass* passPtr = pass.get();
        m_passes.push_back(passPtr);
        m_fg.addPass(std::move(pass));

        for (const auto& binding : config.outputBindings) {
            const ResourceDecl* resource = asset.findResourceById(binding.resourceId);
            if (!resource || resource->kind != "transient") {
                continue;
            }

            FGResource output = passPtr->getOutput(binding.slotKey);
            if (!output.isValid()) {
                m_lastError = "Pass '" + passDecl.name + "' did not publish output slot '" +
                              binding.slotKey + "'";
                m_built = false;
                return false;
            }
            m_resourceMap[binding.resourceId] = output;
        }
    }

    for (const auto& resource : asset.resources) {
        if (resource.kind != "backbuffer") {
            continue;
        }
        auto it = m_resourceMap.find(resource.id);
        if (it != m_resourceMap.end() && it->second.isValid()) {
            m_fg.exportResource(it->second);
        }
    }

    m_builtWidth = width;
    m_builtHeight = height;
    m_built = true;

    spdlog::info("PipelineBuilder: built pipeline '{}' with {} passes",
                 asset.name,
                 m_passes.size());
    return true;
}

void PipelineBuilder::updateFrame(RhiTexture* backbuffer, const FrameContext* frameCtx) {
    if (m_backbufferRes.isValid()) {
        m_fg.updateImport(m_backbufferRes, backbuffer);
    }
    setFrameContext(frameCtx);
}

void PipelineBuilder::compile() {
    m_fg.compile();
}

void PipelineBuilder::execute(RhiCommandBuffer& commandBuffer, RhiFrameGraphBackend& backend) {
    m_fg.execute(commandBuffer, backend);
}

void PipelineBuilder::setFrameContext(const FrameContext* ctx) {
    for (auto* pass : m_passes) {
        pass->setFrameContext(ctx);
    }
}

void PipelineBuilder::setRuntimeContext(const PipelineRuntimeContext* ctx) {
    for (auto* pass : m_passes) {
        pass->setRuntimeContext(ctx);
    }
}

FGResource PipelineBuilder::getResource(const std::string& name) const {
    auto it = m_resourceMap.find(name);
    if (it != m_resourceMap.end()) {
        return it->second;
    }
    return FGResource{};
}

RhiFormat PipelineBuilder::parsePixelFormat(const std::string& format) const {
    static const std::unordered_map<std::string, RhiFormat> formatMap = {
        {"R8Unorm", RhiFormat::R8Unorm},
        {"R16Float", RhiFormat::R16Float},
        {"R32Float", RhiFormat::R32Float},
        {"R32Uint", RhiFormat::R32Uint},
        {"RG8Unorm", RhiFormat::RG8Unorm},
        {"RG16Float", RhiFormat::RG16Float},
        {"RG32Float", RhiFormat::RG32Float},
        {"RGBA8Unorm", RhiFormat::RGBA8Unorm},
        {"RGBA8Srgb", RhiFormat::RGBA8Srgb},
        {"RGBA8_SRGB", RhiFormat::RGBA8Srgb},
        {"BGRA8Unorm", RhiFormat::BGRA8Unorm},
        {"RGBA16Float", RhiFormat::RGBA16Float},
        {"RGBA32Float", RhiFormat::RGBA32Float},
        {"Depth32Float", RhiFormat::D32Float},
        {"Depth16Unorm", RhiFormat::D16Unorm},
    };

    auto it = formatMap.find(format);
    if (it != formatMap.end()) {
        return it->second;
    }
    spdlog::warn("PipelineBuilder: unknown pixel format '{}', defaulting to BGRA8Unorm", format);
    return RhiFormat::BGRA8Unorm;
}

FGTextureDesc PipelineBuilder::parseTextureDesc(const ResourceDecl& decl, int width, int height) const {
    FGTextureDesc desc;

    if (decl.size == "screen" || decl.size.empty()) {
        desc.width = width;
        desc.height = height;
    } else {
        const size_t xPos = decl.size.find('x');
        if (xPos != std::string::npos) {
            desc.width = std::stoi(decl.size.substr(0, xPos));
            desc.height = std::stoi(decl.size.substr(xPos + 1));
        } else {
            desc.width = width;
            desc.height = height;
        }
    }

    desc.format = parsePixelFormat(decl.format);

    if (decl.format.find("Depth") != std::string::npos) {
        desc.usage = RhiTextureUsage::RenderTarget | RhiTextureUsage::ShaderRead;
    } else {
        desc.usage = RhiTextureUsage::RenderTarget | RhiTextureUsage::ShaderRead | RhiTextureUsage::ShaderWrite;
    }

    desc.storageMode = RhiTextureStorageMode::Private;
    return desc;
}
