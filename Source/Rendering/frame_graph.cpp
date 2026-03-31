#include "frame_graph.h"
#include "render_pass.h"
#include <algorithm>
#include <cassert>

#include "imgui.h"
#include <spdlog/spdlog.h>

namespace {

void appendUniqueResource(std::vector<FGResource>& resources, FGResource resource) {
    const auto it = std::find_if(resources.begin(), resources.end(), [resource](FGResource existing) {
        return existing.id == resource.id;
    });
    if (it == resources.end()) {
        resources.push_back(resource);
    }
}

uint32_t findLatestVersion(const std::vector<FGResourceNode>& resources, uint32_t resourceIndex) {
    uint32_t latest = resourceIndex;
    bool advanced = false;
    do {
        advanced = false;
        for (uint32_t ri = 0; ri < resources.size(); ++ri) {
            if (resources[ri].previousVersion == latest) {
                latest = ri;
                advanced = true;
            }
        }
    } while (advanced);
    return latest;
}

bool textureDescMatches(const FGTextureDesc& lhs, const FGTextureDesc& rhs) {
    return lhs.width == rhs.width &&
           lhs.height == rhs.height &&
           lhs.format == rhs.format &&
           lhs.usage == rhs.usage &&
           lhs.storageMode == rhs.storageMode;
}

} // namespace

// --- FGBuilder ---

FGBuilder::FGBuilder(FrameGraph& fg, uint32_t passIndex)
    : m_fg(fg), m_passIndex(passIndex) {}

FGResource FGBuilder::create(const char* name, const FGTextureDesc& desc) {
    FGResource res;
    res.id = static_cast<uint32_t>(m_fg.m_resources.size());
    FGResourceNode node;
    node.name = name;
    node.kind = FGResourceKind::Texture;
    node.desc = desc;
    node.imported = false;
    node.producer = m_passIndex;
    node.physicalResource = res.id;
    m_fg.m_resources.push_back(std::move(node));
    // Creating a resource implicitly writes it
    appendUniqueResource(m_fg.m_passes[m_passIndex].writes, res);
    return res;
}

FGResource FGBuilder::createToken(const char* name) {
    FGResource res;
    res.id = static_cast<uint32_t>(m_fg.m_resources.size());
    FGResourceNode node;
    node.name = name;
    node.kind = FGResourceKind::Token;
    node.imported = false;
    node.producer = m_passIndex;
    node.physicalResource = res.id;
    m_fg.m_resources.push_back(std::move(node));
    appendUniqueResource(m_fg.m_passes[m_passIndex].writes, res);
    return res;
}

FGResource FGBuilder::read(FGResource resource) {
    assert(resource.isValid());
    FGResource latestResource = resource;
    latestResource.id = findLatestVersion(m_fg.m_resources, resource.id);
    appendUniqueResource(m_fg.m_passes[m_passIndex].reads, latestResource);
    return latestResource;
}

FGResource FGBuilder::write(FGResource resource) {
    assert(resource.isValid());
    assert(resource.id < m_fg.m_resources.size());

    resource.id = findLatestVersion(m_fg.m_resources, resource.id);

    auto& pass = m_fg.m_passes[m_passIndex];
    auto& node = m_fg.m_resources[resource.id];
    if (node.producer == m_passIndex) {
        appendUniqueResource(pass.writes, resource);
        return resource;
    }

    for (FGResource existingWrite : pass.writes) {
        const auto& existingNode = m_fg.m_resources[existingWrite.id];
        if (existingWrite.id == resource.id || existingNode.previousVersion == resource.id) {
            return existingWrite;
        }
    }

    FGResource versionedResource;
    versionedResource.id = static_cast<uint32_t>(m_fg.m_resources.size());

    FGResourceNode versionedNode;
    versionedNode.name = node.name;
    versionedNode.kind = node.kind;
    versionedNode.desc = node.desc;
    versionedNode.imported = false;
    versionedNode.producer = m_passIndex;
    versionedNode.physicalResource =
        node.physicalResource != UINT32_MAX ? node.physicalResource : resource.id;
    versionedNode.previousVersion = resource.id;
    if (node.exported) {
        versionedNode.exported = true;
        node.exported = false;
    }
    m_fg.m_resources.push_back(std::move(versionedNode));
    appendUniqueResource(pass.writes, versionedResource);
    return versionedResource;
}

FGResource FGBuilder::readHistory(const char* name, const FGTextureDesc& desc) {
    const uint32_t slotIndex = m_fg.findOrCreateHistorySlot(name, desc);
    auto& slot = m_fg.m_historySlots[slotIndex];

    if (slot.readResource == UINT32_MAX) {
        FGResource resource;
        resource.id = static_cast<uint32_t>(m_fg.m_resources.size());

        FGResourceNode node;
        node.name = name;
        node.kind = FGResourceKind::Texture;
        node.desc = desc;
        node.imported = true;
        node.historySlot = slotIndex;
        node.historyRead = true;
        node.physicalResource = resource.id;
        m_fg.m_resources.push_back(std::move(node));
        slot.readResource = resource.id;
    }

    FGResource resource;
    resource.id = slot.readResource;
    appendUniqueResource(m_fg.m_passes[m_passIndex].reads, resource);
    return resource;
}

FGResource FGBuilder::writeHistory(const char* name, const FGTextureDesc& desc) {
    const uint32_t slotIndex = m_fg.findOrCreateHistorySlot(name, desc);
    auto& slot = m_fg.m_historySlots[slotIndex];

    if (slot.writeResource != UINT32_MAX) {
        const auto& existingNode = m_fg.m_resources[slot.writeResource];
        if (existingNode.producer != m_passIndex) {
            const auto& existingPass = m_fg.m_passes[existingNode.producer];
            const auto& currentPass = m_fg.m_passes[m_passIndex];
            spdlog::warn(
                "FrameGraph: history slot '{}' already uses writer '{}'; pass '{}' will alias the existing history output",
                name,
                existingPass.name,
                currentPass.name);
        }
        FGResource resource;
        resource.id = slot.writeResource;
        appendUniqueResource(m_fg.m_passes[m_passIndex].writes, resource);
        return resource;
    }

    FGResource resource;
    resource.id = static_cast<uint32_t>(m_fg.m_resources.size());

    FGResourceNode node;
    node.name = name;
    node.kind = FGResourceKind::Texture;
    node.desc = desc;
    node.producer = m_passIndex;
    node.historySlot = slotIndex;
    node.historyWrite = true;
    node.physicalResource = resource.id;
    m_fg.m_resources.push_back(std::move(node));
    slot.writeResource = resource.id;

    appendUniqueResource(m_fg.m_passes[m_passIndex].writes, resource);
    return resource;
}

FGResource FGBuilder::setColorAttachment(uint32_t index, FGResource resource,
                                         RhiLoadAction load, RhiStoreAction store,
                                         RhiClearColor clear) {
    auto& pass = m_fg.m_passes[m_passIndex];
    assert(index < 8);
    assert(m_fg.m_resources[resource.id].kind == FGResourceKind::Texture);
    if (load == RhiLoadAction::Load) {
        read(resource);
    }
    const FGResource writeResource = write(resource);
    pass.colorAttachments[index] = {writeResource, load, store, clear};
    if (index >= pass.colorAttachmentCount)
        pass.colorAttachmentCount = index + 1;
    return writeResource;
}

FGResource FGBuilder::setDepthAttachment(FGResource resource,
                                         RhiLoadAction load, RhiStoreAction store,
                                         double clearDepth) {
    auto& pass = m_fg.m_passes[m_passIndex];
    assert(m_fg.m_resources[resource.id].kind == FGResourceKind::Texture);
    if (load == RhiLoadAction::Load) {
        read(resource);
    }
    const FGResource writeResource = write(resource);
    pass.depthAttachment = {writeResource, load, store, clearDepth, true};
    return writeResource;
}

// --- FrameGraph ---

FGResource FrameGraph::import(const char* name, RhiTexture* texture) {
    FGResource res;
    res.id = static_cast<uint32_t>(m_resources.size());
    FGResourceNode node;
    node.name = name;
    node.kind = FGResourceKind::Texture;
    node.texture = texture;
    node.imported = true;
    node.physicalResource = res.id;
    m_resources.push_back(std::move(node));
    return res;
}

void FrameGraph::exportResource(FGResource resource) {
    assert(resource.isValid() && resource.id < m_resources.size());
    const uint32_t latestVersion = findLatestVersion(m_resources, resource.id);
    m_resources[resource.id].exported = false;
    m_resources[latestVersion].exported = true;
}

void FrameGraph::updateImport(FGResource res, RhiTexture* texture) {
    assert(res.isValid() && res.id < m_resources.size());
    assert(m_resources[res.id].imported);
    m_resources[res.id].texture = texture;
}

void FrameGraph::resetTransients() {
    for (uint32_t ri = 0; ri < m_resources.size(); ++ri) {
        auto& res = m_resources[ri];
        if (res.kind == FGResourceKind::Texture &&
            res.historySlot == UINT32_MAX &&
            !res.imported &&
            res.physicalResource == ri) {
            res.ownedTexture.reset();
            res.texture = nullptr;
        }
    }
}

uint32_t FrameGraph::findOrCreateHistorySlot(const char* name, const FGTextureDesc& desc) {
    assert(name != nullptr);

    const auto existingIt = m_historySlotLookup.find(name);
    if (existingIt != m_historySlotLookup.end()) {
        auto& slot = m_historySlots[existingIt->second];
        assert(textureDescMatches(slot.desc, desc) && "History resource description mismatch");
        return existingIt->second;
    }

    const uint32_t slotIndex = static_cast<uint32_t>(m_historySlots.size());
    FGHistorySlot slot;
    slot.name = name;
    slot.desc = desc;
    m_historySlots.push_back(std::move(slot));
    m_historySlotLookup.emplace(name, slotIndex);
    return slotIndex;
}

void FrameGraph::ensureHistoryResources(RhiFrameGraphBackend& backend) {
    for (auto& slot : m_historySlots) {
        bool reallocated = false;
        for (auto& texture : slot.textures) {
            const bool needsCreate =
                !texture ||
                texture->width() != slot.desc.width ||
                texture->height() != slot.desc.height;
            if (needsCreate) {
                texture = backend.createTexture(slot.desc);
                reallocated = true;
            }
        }

        if (reallocated) {
            slot.readIndex = 0;
            slot.valid = false;
        }

        slot.writtenThisFrame = false;
    }
}

void FrameGraph::addPass(std::unique_ptr<RenderPass> pass) {
    RenderPass* passPtr = pass.get();
    m_ownedPasses.push_back(std::move(pass));
    passPtr->m_frameGraph = this;
    uint32_t passIndex = static_cast<uint32_t>(m_passes.size());
    m_passes.push_back({});
    auto& node = m_passes.back();
    node.name = passPtr->name();
    node.type = passPtr->passType();
    node.hasSideEffect = passPtr->hasSideEffectEnabled();
    m_passData.push_back({});

    FGBuilder builder(*this, passIndex);
    passPtr->setup(builder);

    switch (passPtr->passType()) {
        case FGPassType::Render:
            node.executeRender = [passPtr](RhiRenderCommandEncoder& encoder) { passPtr->executeRender(encoder); };
            break;
        case FGPassType::Compute:
            node.executeCompute = [passPtr](RhiComputeCommandEncoder& encoder) { passPtr->executeCompute(encoder); };
            break;
        case FGPassType::Blit:
            node.executeBlit = [passPtr](RhiBlitCommandEncoder& encoder) { passPtr->executeBlit(encoder); };
            break;
    }
}

void FrameGraph::compile() {
    for (auto& pass : m_passes) {
        pass.refCount = 0;
    }
    for (auto& res : m_resources) {
        res.refCount = 0;
        res.lastUser = 0;
    }

    std::vector<uint32_t> livePasses;
    livePasses.reserve(m_passes.size());

    auto addPassRef = [&](uint32_t passIndex) {
        assert(passIndex < m_passes.size());
        auto& pass = m_passes[passIndex];
        const bool wasDead = pass.refCount == 0;
        ++pass.refCount;
        if (wasDead) {
            livePasses.push_back(passIndex);
        }
    };

    auto addResourceRef = [&](uint32_t resourceIndex) {
        assert(resourceIndex < m_resources.size());
        auto& resource = m_resources[resourceIndex];
        ++resource.refCount;
        if (resource.producer != UINT32_MAX) {
            addPassRef(resource.producer);
        }
    };

    for (uint32_t pi = 0; pi < m_passes.size(); ++pi) {
        if (m_passes[pi].hasSideEffect) {
            addPassRef(pi);
        }
    }

    for (uint32_t ri = 0; ri < m_resources.size(); ++ri) {
        if (m_resources[ri].exported) {
            addResourceRef(ri);
        }
    }

    for (const auto& slot : m_historySlots) {
        if (slot.writeResource != UINT32_MAX) {
            addResourceRef(slot.writeResource);
        }
    }

    for (size_t cursor = 0; cursor < livePasses.size(); ++cursor) {
        const uint32_t pi = livePasses[cursor];
        const auto& pass = m_passes[pi];
        for (const auto& read : pass.reads) {
            addResourceRef(read.id);
        }
    }

    for (uint32_t pi = 0; pi < m_passes.size(); ++pi) {
        if (m_passes[pi].refCount == 0) continue;
        auto& pass = m_passes[pi];
        for (auto& r : pass.reads) {
            m_resources[r.id].lastUser = std::max(m_resources[r.id].lastUser, pi);
        }
        for (auto& r : pass.writes) {
            m_resources[r.id].lastUser = std::max(m_resources[r.id].lastUser, pi);
        }
    }
}

RhiTexture* FrameGraph::getTexture(FGResource res) const {
    assert(res.isValid() && res.id < m_resources.size());
    return resolveTexture(res.id);
}

bool FrameGraph::isHistoryValid(FGResource res) const {
    if (!res.isValid() || res.id >= m_resources.size()) {
        return false;
    }

    const auto& resource = m_resources[res.id];
    if (resource.historySlot == UINT32_MAX) {
        return false;
    }

    return m_historySlots[resource.historySlot].valid;
}

void FrameGraph::commitHistory(FGResource res) {
    if (!res.isValid() || res.id >= m_resources.size()) {
        return;
    }

    const auto& resource = m_resources[res.id];
    if (!resource.historyWrite || resource.historySlot == UINT32_MAX) {
        return;
    }

    m_historySlots[resource.historySlot].writtenThisFrame = true;
}

RhiTexture* FrameGraph::resolveTexture(uint32_t resourceId) const {
    assert(resourceId < m_resources.size());
    const auto& resource = m_resources[resourceId];
    assert(resource.kind == FGResourceKind::Texture);

    if (resource.historySlot != UINT32_MAX) {
        const auto& slot = m_historySlots[resource.historySlot];
        const uint32_t textureIndex = resource.historyWrite ? (1u - slot.readIndex) : slot.readIndex;
        return slot.textures[textureIndex].get();
    }

    const uint32_t physicalResource =
        resource.physicalResource != UINT32_MAX ? resource.physicalResource : resourceId;
    assert(physicalResource < m_resources.size());
    assert(m_resources[physicalResource].kind == FGResourceKind::Texture);
    return m_resources[physicalResource].texture;
}

void FrameGraph::execute(RhiCommandBuffer& commandBuffer, RhiFrameGraphBackend& backend) {
    MICROPROFILE_SCOPEI("FrameGraph", "Execute", 0xff00ff00);

    ensureHistoryResources(backend);

    for (uint32_t pi = 0; pi < m_passes.size(); pi++) {
        auto& pass = m_passes[pi];
        if (pass.refCount == 0) continue;

        MICROPROFILE_SCOPEI("FrameGraph", pass.name.c_str(), 0xff0088ff);

        // Create transient textures at their producer pass
        for (uint32_t ri = 0; ri < m_resources.size(); ri++) {
            auto& res = m_resources[ri];
            if (res.kind == FGResourceKind::Texture &&
                res.historySlot == UINT32_MAX &&
                !res.imported &&
                res.producer == pi &&
                res.physicalResource == ri &&
                res.texture == nullptr) {
                res.ownedTexture = backend.createTexture(res.desc);
                res.texture = res.ownedTexture.get();
            }
        }

        if (pass.type == FGPassType::Render) {
            {
                auto isSamePhysicalResource = [&](FGResource lhs, FGResource rhs) {
                    if (!lhs.isValid() || !rhs.isValid()) {
                        return false;
                    }
                    const uint32_t lhsPhysical =
                        m_resources[lhs.id].physicalResource != UINT32_MAX ? m_resources[lhs.id].physicalResource : lhs.id;
                    const uint32_t rhsPhysical =
                        m_resources[rhs.id].physicalResource != UINT32_MAX ? m_resources[rhs.id].physicalResource : rhs.id;
                    return lhsPhysical == rhsPhysical;
                };

                auto isAttachmentRead = [&](FGResource resource) {
                    for (uint32_t ci = 0; ci < pass.colorAttachmentCount; ++ci) {
                        if (isSamePhysicalResource(pass.colorAttachments[ci].resource, resource)) {
                            return true;
                        }
                    }
                    return pass.depthAttachment.bound &&
                           isSamePhysicalResource(pass.depthAttachment.resource, resource);
                };

                for (auto& read : pass.reads) {
                    if (m_resources[read.id].kind != FGResourceKind::Texture) {
                        continue;
                    }
                    if (!isAttachmentRead(read)) {
                        commandBuffer.prepareTextureForSampling(resolveTexture(read.id));
                    }
                }
                // Attachment transitions happen inside beginRenderPass; flush the
                // sampler-read transitions here so they land in the same barrier batch.
                commandBuffer.flushBarriers();
            }

            RhiRenderPassDesc renderPassDesc;
            renderPassDesc.label = pass.name.c_str();
            for (uint32_t ci = 0; ci < pass.colorAttachmentCount; ci++) {
                auto& ca = pass.colorAttachments[ci];
                renderPassDesc.colorAttachments[ci] = {
                    resolveTexture(ca.resource.id),
                    ca.loadAction,
                    ca.storeAction,
                    ca.clearColor,
                };
            }
            renderPassDesc.colorAttachmentCount = pass.colorAttachmentCount;
            if (pass.depthAttachment.bound) {
                renderPassDesc.depthAttachment = {
                    resolveTexture(pass.depthAttachment.resource.id),
                    pass.depthAttachment.loadAction,
                    pass.depthAttachment.storeAction,
                    pass.depthAttachment.clearDepth,
                    true,
                };
            }

            auto encoder = commandBuffer.beginRenderPass(renderPassDesc);
            pass.executeRender(*encoder);
        } else if (pass.type == FGPassType::Compute) {
            // Transition declared read textures to SHADER_READ_ONLY_OPTIMAL before the pass
            for (auto& read : pass.reads) {
                if (m_resources[read.id].kind == FGResourceKind::Texture) {
                    commandBuffer.prepareTextureForSampling(resolveTexture(read.id));
                }
            }
            commandBuffer.flushBarriers();

            RhiComputePassDesc computePassDesc;
            computePassDesc.label = pass.name.c_str();
            auto encoder = commandBuffer.beginComputePass(computePassDesc);
            pass.executeCompute(*encoder);
        } else if (pass.type == FGPassType::Blit) {
            RhiBlitPassDesc blitPassDesc;
            blitPassDesc.label = pass.name.c_str();
            auto encoder = commandBuffer.beginBlitPass(blitPassDesc);
            pass.executeBlit(*encoder);
        }
    }

    for (auto& slot : m_historySlots) {
        if (!slot.writtenThisFrame) {
            continue;
        }
        slot.readIndex = 1u - slot.readIndex;
        slot.valid = true;
    }
}

void FrameGraph::reset() {
    m_resources.clear();
    m_passes.clear();
    m_passData.clear();
    m_ownedPasses.clear();
    m_historySlots.clear();
    m_historySlotLookup.clear();
}

// --- Visualization helpers ---

namespace {

std::string dotEscapeLabel(std::string_view value) {
    std::string out;
    out.reserve(value.size());

    for (char c : value) {
        switch (c) {
            case '\\':
                out += "\\\\";
                break;
            case '"':
                out += "\\\"";
                break;
            case '{':
            case '}':
            case '|':
                out += '\\';
                out += c;
                break;
            case '\n':
                out += "\\n";
                break;
            default:
                out += c;
                break;
        }
    }

    return out;
}

const char* pixelFormatName(RhiFormat fmt) {
    switch (fmt) {
        case RhiFormat::R8Unorm:            return "R8Unorm";
        case RhiFormat::R16Float:           return "R16Float";
        case RhiFormat::R32Float:           return "R32Float";
        case RhiFormat::R32Uint:            return "R32Uint";
        case RhiFormat::RG8Unorm:           return "RG8Unorm";
        case RhiFormat::RG16Float:          return "RG16Float";
        case RhiFormat::RG32Float:          return "RG32Float";
        case RhiFormat::RGBA8Unorm:         return "RGBA8";
        case RhiFormat::RGBA8Srgb:          return "RGBA8Srgb";
        case RhiFormat::BGRA8Unorm:         return "BGRA8";
        case RhiFormat::RGBA16Float:        return "RGBA16Float";
        case RhiFormat::RGBA32Float:        return "RGBA32Float";
        case RhiFormat::D32Float:           return "Depth32F";
        case RhiFormat::D16Unorm:           return "Depth16";
        default:                            return "Unknown";
    }
}

const char* passTypeName(FGPassType type) {
    switch (type) {
        case FGPassType::Render:  return "Render";
        case FGPassType::Compute: return "Compute";
        case FGPassType::Blit:    return "Blit";
    }
    return "Unknown";
}

const char* resourceKindName(FGResourceKind kind) {
    switch (kind) {
        case FGResourceKind::Texture: return "Texture";
        case FGResourceKind::Token:   return "Token";
    }
    return "Unknown";
}

const char* storageModeName(RhiTextureStorageMode storageMode) {
    switch (storageMode) {
        case RhiTextureStorageMode::Private: return "Private";
        case RhiTextureStorageMode::Shared:  return "Shared";
    }
    return "Unknown";
}

bool hasTextureUsage(RhiTextureUsage usage, RhiTextureUsage flag) {
    return (static_cast<uint32_t>(usage) & static_cast<uint32_t>(flag)) != 0;
}

std::string textureUsageSummary(RhiTextureUsage usage) {
    if (usage == RhiTextureUsage::None) {
        return "None";
    }

    std::string summary;
    auto appendFlag = [&summary](const char* label) {
        if (!summary.empty()) {
            summary += " | ";
        }
        summary += label;
    };

    if (hasTextureUsage(usage, RhiTextureUsage::RenderTarget)) {
        appendFlag("RT");
    }
    if (hasTextureUsage(usage, RhiTextureUsage::ShaderRead)) {
        appendFlag("Read");
    }
    if (hasTextureUsage(usage, RhiTextureUsage::ShaderWrite)) {
        appendFlag("Write");
    }

    return summary.empty() ? "None" : summary;
}

uint32_t resourceWidth(const FGResourceNode& resource) {
    if (resource.desc.width != 0) {
        return resource.desc.width;
    }
    if (resource.texture != nullptr) {
        return resource.texture->width();
    }
    return 0;
}

uint32_t resourceHeight(const FGResourceNode& resource) {
    if (resource.desc.height != 0) {
        return resource.desc.height;
    }
    if (resource.texture != nullptr) {
        return resource.texture->height();
    }
    return 0;
}

bool resourceHasKnownFormat(const FGResourceNode& resource) {
    return resource.kind == FGResourceKind::Texture &&
           (resource.desc.width != 0 || resource.desc.height != 0 ||
            resource.historyRead || resource.historyWrite || !resource.imported);
}

std::string resourceResidencyLabel(const FGResourceNode& resource) {
    if (resource.historyRead) {
        return "History Read";
    }
    if (resource.historyWrite) {
        return "History Write";
    }
    if (resource.imported) {
        return "Imported";
    }
    return "Transient";
}

std::string abbreviateLabel(std::string_view value, size_t maxLength) {
    if (value.size() <= maxLength) {
        return std::string(value);
    }
    if (maxLength <= 3) {
        return std::string(value.substr(0, maxLength));
    }

    std::string out(value.substr(0, maxLength - 3));
    out += "...";
    return out;
}

bool pointInRect(const ImVec2& point, const ImVec2& min, const ImVec2& max) {
    return point.x >= min.x && point.y >= min.y && point.x < max.x && point.y < max.y;
}

ImU32 resourceTimelineColor(const FGResourceNode& resource, bool live) {
    const uint8_t alpha = live ? 225 : 110;
    if (resource.kind == FGResourceKind::Token) {
        return IM_COL32(140, 146, 156, alpha);
    }
    if (resource.historyRead || resource.historyWrite) {
        return IM_COL32(224, 184, 82, alpha);
    }
    if (resource.imported) {
        return IM_COL32(82, 146, 255, alpha);
    }
    return IM_COL32(72, 196, 168, alpha);
}

ImU32 resourceTimelineBorderColor(const FGResourceNode& resource, bool live) {
    if (resource.exported) {
        return live ? IM_COL32(255, 228, 136, 255) : IM_COL32(170, 150, 98, 180);
    }
    return live ? IM_COL32(235, 239, 244, 140) : IM_COL32(120, 124, 132, 90);
}

ImU32 passTimelineHeaderColor(const FGPassNode& pass) {
    if (pass.refCount == 0) {
        return IM_COL32(84, 84, 88, 180);
    }

    switch (pass.type) {
        case FGPassType::Render:  return IM_COL32(84, 128, 196, 220);
        case FGPassType::Compute: return IM_COL32(80, 160, 120, 220);
        case FGPassType::Blit:    return IM_COL32(192, 132, 78, 220);
    }
    return IM_COL32(92, 92, 92, 220);
}

std::string resourceMetaLabel(const FGResourceNode& resource, uint32_t aliasGroup) {
    std::string label;

    if (resource.kind == FGResourceKind::Texture) {
        const uint32_t width = resourceWidth(resource);
        const uint32_t height = resourceHeight(resource);
        if (width > 0 || height > 0) {
            label += std::to_string(width);
            label += "x";
            label += std::to_string(height);
            if (resourceHasKnownFormat(resource)) {
                label += " ";
                label += pixelFormatName(resource.desc.format);
            }
        } else {
            label += "Texture";
        }
    } else {
        label += "Token";
    }

    label += " | ";
    label += resourceResidencyLabel(resource);
    label += " | Alias ";
    label += std::to_string(aliasGroup + 1);
    if (resource.exported) {
        label += " | External";
    }
    return label;
}

void drawResourceTimelineImGui(const std::vector<FGResourceNode>& resources,
                               const std::vector<FGPassNode>& passes) {
    if (resources.empty()) {
        ImGui::TextDisabled("No frame graph resources recorded.");
        return;
    }

    static ImGuiTextFilter resourceFilter;
    static bool showCulledResources = false;
    static bool sortByFirstUse = true;

    resourceFilter.Draw("Filter", 220.0f);
    ImGui::SameLine();
    ImGui::Checkbox("Show Culled", &showCulledResources);
    ImGui::SameLine();
    ImGui::Checkbox("Sort By First Use", &sortByFirstUse);
    ImGui::TextDisabled("Blue = imported, gold = history, teal = transient, gray = token, green dots = reads, orange squares = writes.");

    const size_t passCount = passes.size();
    std::vector<std::vector<uint8_t>> readMasks(resources.size(), std::vector<uint8_t>(passCount, 0));
    std::vector<std::vector<uint8_t>> writeMasks(resources.size(), std::vector<uint8_t>(passCount, 0));

    for (size_t passIndex = 0; passIndex < passCount; ++passIndex) {
        const auto& pass = passes[passIndex];
        for (FGResource read : pass.reads) {
            if (read.id < resources.size()) {
                readMasks[read.id][passIndex] = 1;
            }
        }
        for (FGResource write : pass.writes) {
            if (write.id < resources.size()) {
                writeMasks[write.id][passIndex] = 1;
            }
        }
    }

    std::unordered_map<uint32_t, uint32_t> aliasGroupLookup;
    aliasGroupLookup.reserve(resources.size());

    struct ResourceTimelineRow {
        uint32_t resourceIndex = 0;
        uint32_t firstPass = 0;
        uint32_t lastPass = 0;
        uint32_t aliasGroup = 0;
        bool hasAccess = false;
    };

    std::vector<ResourceTimelineRow> rows;
    rows.reserve(resources.size());

    for (uint32_t resourceIndex = 0; resourceIndex < resources.size(); ++resourceIndex) {
        const auto& resource = resources[resourceIndex];
        if (!resourceFilter.PassFilter(resource.name.c_str())) {
            continue;
        }
        if (!showCulledResources && resource.refCount == 0) {
            continue;
        }

        const uint32_t physicalId =
            resource.physicalResource != UINT32_MAX ? resource.physicalResource : resourceIndex;
        auto [it, inserted] =
            aliasGroupLookup.emplace(physicalId, static_cast<uint32_t>(aliasGroupLookup.size()));
        (void)inserted;

        ResourceTimelineRow row;
        row.resourceIndex = resourceIndex;
        row.aliasGroup = it->second;
        row.firstPass = resource.producer != UINT32_MAX ? resource.producer : 0;
        row.lastPass = row.firstPass;

        for (uint32_t passIndex = 0; passIndex < passCount; ++passIndex) {
            if (readMasks[resourceIndex][passIndex] == 0 &&
                writeMasks[resourceIndex][passIndex] == 0) {
                continue;
            }

            if (!row.hasAccess) {
                row.firstPass = passIndex;
                row.lastPass = passIndex;
                row.hasAccess = true;
            } else {
                row.firstPass = std::min(row.firstPass, passIndex);
                row.lastPass = std::max(row.lastPass, passIndex);
            }
        }

        if (!row.hasAccess && resource.producer != UINT32_MAX) {
            row.hasAccess = true;
        }

        rows.push_back(row);
    }

    if (rows.empty()) {
        ImGui::TextDisabled("No resources match the current filter.");
        return;
    }

    if (sortByFirstUse) {
        std::stable_sort(rows.begin(), rows.end(), [](const ResourceTimelineRow& lhs,
                                                      const ResourceTimelineRow& rhs) {
            if (lhs.firstPass != rhs.firstPass) {
                return lhs.firstPass < rhs.firstPass;
            }
            return lhs.resourceIndex < rhs.resourceIndex;
        });
    }

    const float headerHeight = 42.0f;
    const float rowHeight = 36.0f;
    const float labelWidth = 320.0f;
    const float passWidth = passCount > 10 ? 96.0f : 116.0f;
    const float desiredHeight =
        std::min(520.0f, headerHeight + rowHeight * static_cast<float>(rows.size()) + 12.0f);

    if (!ImGui::BeginChild("ResourceTimelineCanvas",
                           ImVec2(0.0f, desiredHeight),
                           ImGuiChildFlags_Borders,
                           ImGuiWindowFlags_HorizontalScrollbar)) {
        ImGui::EndChild();
        return;
    }

    const float totalWidth = labelWidth + passWidth * static_cast<float>(std::max<size_t>(passCount, 1)) + 12.0f;
    const float totalHeight = headerHeight + rowHeight * static_cast<float>(rows.size()) + 4.0f;
    ImGui::InvisibleButton("##frame_graph_resource_timeline_canvas", ImVec2(totalWidth, totalHeight));

    ImDrawList* drawList = ImGui::GetWindowDrawList();
    const ImVec2 canvasMin = ImGui::GetItemRectMin();
    const ImVec2 canvasMax = ImGui::GetItemRectMax();
    const ImVec2 mousePos = ImGui::GetIO().MousePos;
    const float lanesStartX = canvasMin.x + labelWidth;
    const float headerBottomY = canvasMin.y + headerHeight;

    drawList->AddRectFilled(canvasMin, canvasMax, IM_COL32(16, 18, 22, 255), 6.0f);
    drawList->AddRect(canvasMin, canvasMax, IM_COL32(50, 56, 68, 255), 6.0f);

    drawList->AddRectFilled(canvasMin,
                            ImVec2(canvasMin.x + labelWidth, headerBottomY),
                            IM_COL32(26, 29, 36, 255),
                            6.0f,
                            ImDrawFlags_RoundCornersTopLeft);
    drawList->AddText(ImVec2(canvasMin.x + 10.0f, canvasMin.y + 8.0f),
                      IM_COL32(245, 245, 245, 255),
                      "Resources");
    drawList->AddText(ImVec2(canvasMin.x + 10.0f, canvasMin.y + 22.0f),
                      IM_COL32(156, 164, 176, 255),
                      "Logical lifetime by pass");

    for (uint32_t passIndex = 0; passIndex < passCount; ++passIndex) {
        const auto& pass = passes[passIndex];
        const float x0 = lanesStartX + passWidth * static_cast<float>(passIndex);
        const float x1 = x0 + passWidth;

        drawList->AddRectFilled(ImVec2(x0, headerBottomY),
                                ImVec2(x1, canvasMax.y),
                                pass.refCount > 0 ? IM_COL32(32, 42, 56, 72) : IM_COL32(20, 22, 26, 72));
        drawList->AddRectFilled(ImVec2(x0, canvasMin.y),
                                ImVec2(x1, headerBottomY),
                                passTimelineHeaderColor(pass),
                                passIndex + 1 == passCount ? 6.0f : 0.0f,
                                passIndex + 1 == passCount ? ImDrawFlags_RoundCornersTopRight : ImDrawFlags_None);
        drawList->AddLine(ImVec2(x0, canvasMin.y),
                          ImVec2(x0, canvasMax.y),
                          IM_COL32(58, 66, 82, 180));

        const std::string passId = "P" + std::to_string(passIndex);
        const std::string passLabel = abbreviateLabel(pass.name, 14);
        drawList->AddText(ImVec2(x0 + 8.0f, canvasMin.y + 6.0f),
                          IM_COL32(250, 250, 250, 255),
                          passId.c_str());
        drawList->AddText(ImVec2(x0 + 8.0f, canvasMin.y + 20.0f),
                          IM_COL32(232, 236, 242, 255),
                          passLabel.c_str());

        if (pointInRect(mousePos, ImVec2(x0, canvasMin.y), ImVec2(x1, headerBottomY))) {
            ImGui::BeginTooltip();
            ImGui::Text("P%u %s", passIndex, pass.name.c_str());
            ImGui::Separator();
            ImGui::Text("Type: %s", passTypeName(pass.type));
            ImGui::Text("Status: %s", pass.refCount > 0 ? "Live" : "Culled");
            ImGui::Text("Refs: %u", pass.refCount);
            ImGui::Text("Reads: %zu", pass.reads.size());
            ImGui::Text("Writes: %zu", pass.writes.size());
            ImGui::Text("Side Effect: %s", pass.hasSideEffect ? "Yes" : "No");
            ImGui::EndTooltip();
        }
    }
    drawList->AddLine(ImVec2(lanesStartX, canvasMin.y),
                      ImVec2(lanesStartX, canvasMax.y),
                      IM_COL32(72, 80, 98, 255),
                      1.5f);

    for (size_t rowIndex = 0; rowIndex < rows.size(); ++rowIndex) {
        const ResourceTimelineRow& row = rows[rowIndex];
        const FGResourceNode& resource = resources[row.resourceIndex];
        const bool live = resource.refCount > 0;

        const float y0 = headerBottomY + rowHeight * static_cast<float>(rowIndex);
        const float y1 = y0 + rowHeight;
        const ImVec2 rowMin(canvasMin.x, y0);
        const ImVec2 rowMax(canvasMax.x, y1);

        const ImU32 rowBg = (rowIndex % 2 == 0) ? IM_COL32(24, 27, 34, 180) : IM_COL32(20, 23, 29, 180);
        drawList->AddRectFilled(rowMin, rowMax, rowBg);
        drawList->AddLine(ImVec2(canvasMin.x, y1),
                          ImVec2(canvasMax.x, y1),
                          IM_COL32(48, 54, 66, 168));

        const std::string title =
            "#" + std::to_string(row.resourceIndex) + " " + abbreviateLabel(resource.name, 34);
        const std::string meta =
            abbreviateLabel(resourceMetaLabel(resource, row.aliasGroup), 52);
        drawList->AddText(ImVec2(canvasMin.x + 10.0f, y0 + 6.0f),
                          live ? IM_COL32(245, 245, 245, 255) : IM_COL32(150, 155, 162, 220),
                          title.c_str());
        drawList->AddText(ImVec2(canvasMin.x + 10.0f, y0 + 20.0f),
                          IM_COL32(150, 158, 170, 230),
                          meta.c_str());

        if (passCount > 0 && row.hasAccess) {
            const float barCenterY = y0 + rowHeight * 0.5f;
            const float barHalfHeight = 6.0f;
            const float barMinX = lanesStartX + passWidth * static_cast<float>(row.firstPass) + 8.0f;
            const float barMaxX = lanesStartX + passWidth * static_cast<float>(row.lastPass + 1) - 8.0f;
            const ImVec2 barMin(barMinX, barCenterY - barHalfHeight);
            const ImVec2 barMax(barMaxX, barCenterY + barHalfHeight);

            drawList->AddRectFilled(barMin,
                                    barMax,
                                    resourceTimelineColor(resource, live),
                                    6.0f);
            drawList->AddRect(barMin,
                              barMax,
                              resourceTimelineBorderColor(resource, live),
                              6.0f,
                              ImDrawFlags_None,
                              resource.exported ? 2.0f : 1.0f);

            if (resource.imported) {
                drawList->AddTriangleFilled(ImVec2(barMin.x - 6.0f, barCenterY),
                                            ImVec2(barMin.x, barCenterY - 5.0f),
                                            ImVec2(barMin.x, barCenterY + 5.0f),
                                            resourceTimelineColor(resource, live));
            }
            if (resource.exported) {
                drawList->AddTriangleFilled(ImVec2(barMax.x + 6.0f, barCenterY),
                                            ImVec2(barMax.x, barCenterY - 5.0f),
                                            ImVec2(barMax.x, barCenterY + 5.0f),
                                            resourceTimelineBorderColor(resource, live));
            }

            for (uint32_t passIndex = 0; passIndex < passCount; ++passIndex) {
                const bool hasRead = readMasks[row.resourceIndex][passIndex] != 0;
                const bool hasWrite = writeMasks[row.resourceIndex][passIndex] != 0;
                if (!hasRead && !hasWrite) {
                    continue;
                }

                const float centerX = lanesStartX + passWidth * (static_cast<float>(passIndex) + 0.5f);
                const float readY = hasRead && hasWrite ? barCenterY - 5.0f : barCenterY;
                const float writeY = hasRead && hasWrite ? barCenterY + 5.0f : barCenterY;

                if (hasRead) {
                    drawList->AddCircleFilled(ImVec2(centerX, readY),
                                              3.5f,
                                              IM_COL32(106, 222, 138, live ? 255 : 180));
                }
                if (hasWrite) {
                    drawList->AddRectFilled(ImVec2(centerX - 4.0f, writeY - 4.0f),
                                            ImVec2(centerX + 4.0f, writeY + 4.0f),
                                            IM_COL32(255, 170, 75, live ? 255 : 180),
                                            2.0f);
                }
            }
        }

        if (pointInRect(mousePos, rowMin, rowMax)) {
            ImGui::BeginTooltip();
            ImGui::Text("#%u %s", row.resourceIndex, resource.name.c_str());
            ImGui::Separator();
            ImGui::Text("Kind: %s", resourceKindName(resource.kind));
            ImGui::Text("Residency: %s", resourceResidencyLabel(resource).c_str());
            ImGui::Text("Status: %s", live ? "Live" : "Culled");
            ImGui::Text("Refs: %u", resource.refCount);
            ImGui::Text("Alias Group: %u", row.aliasGroup + 1);
            if (resource.previousVersion != UINT32_MAX) {
                ImGui::Text("Previous Version: #%u", resource.previousVersion);
            }
            if (resource.producer != UINT32_MAX && resource.producer < passes.size()) {
                ImGui::Text("Producer: P%u %s", resource.producer, passes[resource.producer].name.c_str());
            } else {
                ImGui::Text("Producer: External");
            }
            if (resource.refCount > 0 && resource.lastUser < passes.size()) {
                ImGui::Text("Last User: P%u %s", resource.lastUser, passes[resource.lastUser].name.c_str());
            } else {
                ImGui::Text("Last User: -");
            }

            if (resource.kind == FGResourceKind::Texture) {
                const uint32_t width = resourceWidth(resource);
                const uint32_t height = resourceHeight(resource);
                const bool hasDescriptorInfo = resourceHasKnownFormat(resource);
                if (width > 0 || height > 0) {
                    ImGui::Text("Size: %ux%u", width, height);
                } else {
                    ImGui::Text("Size: -");
                }
                ImGui::Text("Format: %s",
                            hasDescriptorInfo ? pixelFormatName(resource.desc.format) : "Unknown");
                if (hasDescriptorInfo) {
                    ImGui::Text("Usage: %s", textureUsageSummary(resource.desc.usage).c_str());
                    ImGui::Text("Storage: %s", storageModeName(resource.desc.storageMode));
                } else {
                    ImGui::Text("Usage: -");
                    ImGui::Text("Storage: -");
                }
            }

            if (passCount > 0) {
                ImGui::Separator();
                ImGui::TextUnformatted("Pass Access");
                bool anyAccess = false;
                for (uint32_t passIndex = 0; passIndex < passCount; ++passIndex) {
                    const bool hasRead = readMasks[row.resourceIndex][passIndex] != 0;
                    const bool hasWrite = writeMasks[row.resourceIndex][passIndex] != 0;
                    if (!hasRead && !hasWrite) {
                        continue;
                    }

                    const char* accessLabel = hasRead && hasWrite ? "read + write" : (hasRead ? "read" : "write");
                    ImGui::BulletText("P%u %s (%s)",
                                      passIndex,
                                      passes[passIndex].name.c_str(),
                                      accessLabel);
                    anyAccess = true;
                }
                if (!anyAccess) {
                    ImGui::TextDisabled("No pass access");
                }
            }
            ImGui::EndTooltip();
        }
    }

    ImGui::EndChild();
}

} // namespace

// --- Graphviz DOT export ---

void FrameGraph::exportGraphviz(std::ostream& os) const {
    os << "digraph FrameGraph {\n";
    os << "  rankdir=LR;\n";
    os << "  node [fontname=\"Helvetica\", fontsize=10];\n";
    os << "  edge [fontname=\"Helvetica\", fontsize=9];\n\n";

    // Imported resources cluster
    bool hasImported = false;
    for (auto& res : m_resources) {
        if (res.imported) { hasImported = true; break; }
    }
    if (hasImported) {
        os << "  subgraph cluster_imported {\n";
        os << "    label=\"Imported\";\n";
        os << "    style=dashed;\n";
        for (uint32_t ri = 0; ri < m_resources.size(); ri++) {
            auto& res = m_resources[ri];
            if (!res.imported) continue;
            os << "    R" << ri << " [shape=record, style=\"rounded,filled\", "
               << "fillcolor=lightsteelblue, label=\"{" << dotEscapeLabel(res.name)
               << " | Imported " << resourceKindName(res.kind)
               << (res.exported ? " | External" : "")
               << " | Refs: " << res.refCount << "}\"];\n";
        }
        os << "  }\n\n";
    }

    // Pass nodes and their created resources (clustered)
    for (uint32_t pi = 0; pi < m_passes.size(); pi++) {
        auto& pass = m_passes[pi];
        bool live = pass.refCount > 0;

        // Cluster: resources created by this pass
        bool hasCreated = false;
        for (auto& res : m_resources) {
            if (!res.imported && res.producer == pi) { hasCreated = true; break; }
        }
        if (hasCreated) {
            os << "  subgraph cluster_P" << pi << " {\n";
            os << "    label=\"\";\n";
            os << "    style=dashed;\n";
            // Pass node inside cluster
            os << "    P" << pi << " [shape=record, style=\"rounded,filled\", "
               << "fillcolor=" << (live ? "orange" : "lightgray")
               << ", label=\"{" << dotEscapeLabel(pass.name) << " | "
               << (pass.hasSideEffect ? "* " : "")
               << "Refs: " << pass.refCount << " | " << passTypeName(pass.type)
               << "}\"];\n";
            // Resource nodes inside cluster
            for (uint32_t ri = 0; ri < m_resources.size(); ri++) {
                auto& res = m_resources[ri];
                if (res.imported || res.producer != pi) continue;
                os << "    R" << ri << " [shape=record, style=\"rounded,filled\", "
                   << "fillcolor=skyblue, label=\"{" << dotEscapeLabel(res.name);
                if (res.kind == FGResourceKind::Texture) {
                    os << " | " << res.desc.width << "x" << res.desc.height
                       << " " << pixelFormatName(res.desc.format);
                } else {
                    os << " | " << resourceKindName(res.kind);
                }
                os
                   << (res.exported ? " | External" : "")
                   << " | Refs: " << res.refCount << "}\"];\n";
            }
            os << "  }\n\n";
        } else {
            // Pass node without cluster
            os << "  P" << pi << " [shape=record, style=\"rounded,filled\", "
               << "fillcolor=" << (live ? "orange" : "lightgray")
               << ", label=\"{" << dotEscapeLabel(pass.name) << " | "
               << (pass.hasSideEffect ? "* " : "")
               << "Refs: " << pass.refCount << " | " << passTypeName(pass.type)
               << "}\"];\n";
        }
    }

    os << "\n";

    // Edges
    for (uint32_t pi = 0; pi < m_passes.size(); pi++) {
        auto& pass = m_passes[pi];
        // Write edges: pass -> resource (orangered)
        for (auto& w : pass.writes) {
            os << "  P" << pi << " -> R" << w.id
               << " [color=orangered];\n";
        }
        // Read edges: resource -> pass (yellowgreen)
        for (auto& r : pass.reads) {
            os << "  R" << r.id << " -> P" << pi
               << " [color=yellowgreen];\n";
        }
    }

    os << "}\n";
}

// --- ImGui debug window ---

void FrameGraph::debugImGui() const {
    if (!ImGui::Begin("FrameGraph Debug")) {
        ImGui::End();
        return;
    }

    const auto livePassCount = static_cast<uint32_t>(std::count_if(
        m_passes.begin(),
        m_passes.end(),
        [](const FGPassNode& pass) { return pass.refCount > 0; }));
    const auto liveResourceCount = static_cast<uint32_t>(std::count_if(
        m_resources.begin(),
        m_resources.end(),
        [](const FGResourceNode& resource) { return resource.refCount > 0; }));

    ImGui::Text("Live Passes: %u / %zu", livePassCount, m_passes.size());
    ImGui::SameLine();
    ImGui::Text("Live Resources: %u / %zu", liveResourceCount, m_resources.size());
    if (!m_historySlots.empty()) {
        ImGui::SameLine();
        ImGui::Text("History Slots: %zu", m_historySlots.size());
    }

    if (ImGui::CollapsingHeader("Resource Timeline", ImGuiTreeNodeFlags_DefaultOpen)) {
        drawResourceTimelineImGui(m_resources, m_passes);
    }

    // Passes table
    if (ImGui::CollapsingHeader("Passes", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::BeginTable("passes", 7,
                ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable)) {
            ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, 24.0f);
            ImGui::TableSetupColumn("Name");
            ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthFixed, 60.0f);
            ImGui::TableSetupColumn("Refs", ImGuiTableColumnFlags_WidthFixed, 36.0f);
            ImGui::TableSetupColumn("Side Effect", ImGuiTableColumnFlags_WidthFixed, 72.0f);
            ImGui::TableSetupColumn("Reads");
            ImGui::TableSetupColumn("Writes");
            ImGui::TableHeadersRow();

            for (uint32_t pi = 0; pi < m_passes.size(); pi++) {
                auto& pass = m_passes[pi];
                bool culled = pass.refCount == 0;

                ImGui::TableNextRow();
                if (culled) ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));

                ImGui::TableSetColumnIndex(0);
                ImGui::Text("%u", pi);
                ImGui::TableSetColumnIndex(1);
                ImGui::TextUnformatted(pass.name.c_str());
                ImGui::TableSetColumnIndex(2);
                ImGui::TextUnformatted(passTypeName(pass.type));
                ImGui::TableSetColumnIndex(3);
                ImGui::Text("%u", pass.refCount);
                ImGui::TableSetColumnIndex(4);
                ImGui::TextUnformatted(pass.hasSideEffect ? "Yes" : "-");

                // Reads column
                ImGui::TableSetColumnIndex(5);
                for (size_t i = 0; i < pass.reads.size(); i++) {
                    if (i > 0) ImGui::SameLine(0, 0); ImGui::Text("%s%s",
                        m_resources[pass.reads[i].id].name.c_str(),
                        i + 1 < pass.reads.size() ? ", " : "");
                }

                // Writes column
                ImGui::TableSetColumnIndex(6);
                for (size_t i = 0; i < pass.writes.size(); i++) {
                    if (i > 0) ImGui::SameLine(0, 0); ImGui::Text("%s%s",
                        m_resources[pass.writes[i].id].name.c_str(),
                        i + 1 < pass.writes.size() ? ", " : "");
                }

                if (culled) ImGui::PopStyleColor();
            }
            ImGui::EndTable();
        }
    }

    // Resources table
    if (ImGui::CollapsingHeader("Resources", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::BeginTable("resources", 8,
                ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable)) {
            ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, 24.0f);
            ImGui::TableSetupColumn("Name");
            ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthFixed, 64.0f);
            ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_WidthFixed, 80.0f);
            ImGui::TableSetupColumn("Format", ImGuiTableColumnFlags_WidthFixed, 72.0f);
            ImGui::TableSetupColumn("Refs", ImGuiTableColumnFlags_WidthFixed, 36.0f);
            ImGui::TableSetupColumn("Producer");
            ImGui::TableSetupColumn("Last User");
            ImGui::TableHeadersRow();

            for (uint32_t ri = 0; ri < m_resources.size(); ri++) {
                auto& res = m_resources[ri];
                ImGui::TableNextRow();

                ImGui::TableSetColumnIndex(0);
                ImGui::Text("%u", ri);
                ImGui::TableSetColumnIndex(1);
                ImGui::TextUnformatted(res.name.c_str());
                ImGui::TableSetColumnIndex(2);
                const char* residency = res.imported ? "Imported" : "Transient";
                if (res.historyRead) {
                    residency = "History Read";
                } else if (res.historyWrite) {
                    residency = "History Write";
                }
                if (res.exported)
                    ImGui::Text("%s %s, External", residency, resourceKindName(res.kind));
                else
                    ImGui::Text("%s %s", residency, resourceKindName(res.kind));
                ImGui::TableSetColumnIndex(3);
                if (res.kind == FGResourceKind::Texture && (resourceWidth(res) > 0 || resourceHeight(res) > 0))
                    ImGui::Text("%ux%u", resourceWidth(res), resourceHeight(res));
                else
                    ImGui::TextUnformatted("-");
                ImGui::TableSetColumnIndex(4);
                if (res.kind == FGResourceKind::Texture && resourceHasKnownFormat(res))
                    ImGui::TextUnformatted(pixelFormatName(res.desc.format));
                else
                    ImGui::TextUnformatted("-");
                ImGui::TableSetColumnIndex(5);
                ImGui::Text("%u", res.refCount);
                ImGui::TableSetColumnIndex(6);
                if (res.producer != UINT32_MAX)
                    ImGui::TextUnformatted(m_passes[res.producer].name.c_str());
                else
                    ImGui::TextUnformatted("-");
                ImGui::TableSetColumnIndex(7);
                if (res.refCount > 0 && res.lastUser < m_passes.size())
                    ImGui::TextUnformatted(m_passes[res.lastUser].name.c_str());
                else
                    ImGui::TextUnformatted("-");
            }
            ImGui::EndTable();
        }
    }

    ImGui::End();
}

void FrameGraph::renderPassUI() {
    bool hasAnyUI = false;
    for (auto& pass : m_ownedPasses) {
        // Probe: only show window if at least one pass overrides renderUI
        // We always iterate — renderUI() is a no-op by default
        hasAnyUI = true;
        break;
    }
    if (!hasAnyUI)
        return;

    if (!ImGui::Begin("Render Passes")) {
        ImGui::End();
        return;
    }

    for (auto& pass : m_ownedPasses) {
        if (ImGui::CollapsingHeader(pass->name(), ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::PushID(pass.get());
            pass->renderUI();
            ImGui::PopID();
        }
    }

    ImGui::End();
}
