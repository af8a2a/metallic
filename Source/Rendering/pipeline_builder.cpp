#include "pipeline_builder.h"
#include "render_pass.h"
#include <spdlog/spdlog.h>

PipelineBuilder::PipelineBuilder(const RenderContext& ctx, const PipelineRuntimeContext& rtCtx)
    : m_ctx(ctx), m_rtCtx(rtCtx) {}

bool PipelineBuilder::build(const PipelineAsset& asset, FrameGraph& fg, int width, int height) {
    m_resourceMap.clear();
    m_lastError.clear();

    // Validate the pipeline first
    if (!asset.validate(m_lastError)) {
        return false;
    }

    // Import special resources
    if (m_rtCtx.backbuffer) {
        m_resourceMap["$backbuffer"] = fg.import("backbuffer", m_rtCtx.backbuffer);
    }

    // Import any pre-existing textures
    for (const auto& [name, tex] : m_rtCtx.importedTextures) {
        if (tex) {
            m_resourceMap[name] = fg.import(name.c_str(), tex);
        }
    }

    // Get topologically sorted pass order
    auto sortedOrder = asset.topologicalSort();

    // Create passes in sorted order
    for (size_t idx : sortedOrder) {
        const auto& passDecl = asset.passes[idx];

        if (!passDecl.enabled) {
            spdlog::debug("PipelineBuilder: skipping disabled pass '{}'", passDecl.name);
            continue;
        }

        // Build PassConfig from declaration
        PassConfig config;
        config.name = passDecl.name;
        config.type = passDecl.type;
        config.inputs = passDecl.inputs;
        config.outputs = passDecl.outputs;
        config.enabled = passDecl.enabled;
        config.sideEffect = passDecl.sideEffect;
        config.config = passDecl.config;

        // Resolve input resources
        for (const auto& input : passDecl.inputs) {
            if (m_resourceMap.find(input) == m_resourceMap.end()) {
                m_lastError = "Pass '" + passDecl.name + "' references unknown resource: " + input;
                return false;
            }
        }

        // Create the pass via registry
        auto pass = PassRegistry::instance().create(passDecl.type, config, m_ctx, width, height);
        if (!pass) {
            m_lastError = "Failed to create pass of type '" + passDecl.type + "'";
            return false;
        }

        // Store pass pointer before moving
        RenderPass* passPtr = pass.get();

        // Add pass to frame graph
        fg.addPass(std::move(pass));

        // After setup, the pass may have created output resources
        // We need a way to retrieve them - this requires passes to expose their outputs
        // For now, we'll handle this in the pass configure() method
    }

    spdlog::info("PipelineBuilder: built pipeline '{}' with {} passes",
                 asset.name, sortedOrder.size());
    return true;
}

FGResource PipelineBuilder::getResource(const std::string& name) const {
    auto it = m_resourceMap.find(name);
    if (it != m_resourceMap.end()) {
        return it->second;
    }
    return FGResource{};
}

MTL::PixelFormat PipelineBuilder::parsePixelFormat(const std::string& format) const {
    static const std::unordered_map<std::string, MTL::PixelFormat> formatMap = {
        {"R8Unorm", MTL::PixelFormatR8Unorm},
        {"R16Float", MTL::PixelFormatR16Float},
        {"R32Float", MTL::PixelFormatR32Float},
        {"R32Uint", MTL::PixelFormatR32Uint},
        {"RG8Unorm", MTL::PixelFormatRG8Unorm},
        {"RG16Float", MTL::PixelFormatRG16Float},
        {"RG32Float", MTL::PixelFormatRG32Float},
        {"RGBA8Unorm", MTL::PixelFormatRGBA8Unorm},
        {"BGRA8Unorm", MTL::PixelFormatBGRA8Unorm},
        {"RGBA16Float", MTL::PixelFormatRGBA16Float},
        {"RGBA32Float", MTL::PixelFormatRGBA32Float},
        {"Depth32Float", MTL::PixelFormatDepth32Float},
        {"Depth16Unorm", MTL::PixelFormatDepth16Unorm},
    };

    auto it = formatMap.find(format);
    if (it != formatMap.end()) {
        return it->second;
    }
    spdlog::warn("PipelineBuilder: unknown pixel format '{}', defaulting to BGRA8Unorm", format);
    return MTL::PixelFormatBGRA8Unorm;
}

FGTextureDesc PipelineBuilder::parseTextureDesc(const ResourceDecl& decl, int width, int height) const {
    FGTextureDesc desc;

    // Parse size
    if (decl.size == "screen" || decl.size.empty()) {
        desc.width = width;
        desc.height = height;
    } else {
        // Parse "WxH" format
        size_t xPos = decl.size.find('x');
        if (xPos != std::string::npos) {
            desc.width = std::stoi(decl.size.substr(0, xPos));
            desc.height = std::stoi(decl.size.substr(xPos + 1));
        } else {
            desc.width = width;
            desc.height = height;
        }
    }

    // Parse format
    desc.format = parsePixelFormat(decl.format);

    // Set usage based on format
    if (decl.format.find("Depth") != std::string::npos) {
        desc.usage = MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead;
    } else {
        desc.usage = MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite;
    }

    desc.storageMode = MTL::StorageModePrivate;

    return desc;
}
