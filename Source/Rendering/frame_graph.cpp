#include "frame_graph.h"
#include "render_pass.h"
#include <algorithm>
#include <cassert>

#ifdef _WIN32
#include "vulkan_frame_graph.h"
#endif

#include "imgui.h"

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
    appendUniqueResource(m_fg.m_passes[m_passIndex].reads, resource);
    return resource;
}

FGResource FGBuilder::write(FGResource resource) {
    assert(resource.isValid());
    assert(resource.id < m_fg.m_resources.size());

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
        if (res.kind == FGResourceKind::Texture && !res.imported && res.physicalResource == ri) {
            res.ownedTexture.reset();
            res.texture = nullptr;
        }
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

RhiTexture* FrameGraph::resolveTexture(uint32_t resourceId) const {
    assert(resourceId < m_resources.size());
    const auto& resource = m_resources[resourceId];
    assert(resource.kind == FGResourceKind::Texture);
    const uint32_t physicalResource =
        resource.physicalResource != UINT32_MAX ? resource.physicalResource : resourceId;
    assert(physicalResource < m_resources.size());
    assert(m_resources[physicalResource].kind == FGResourceKind::Texture);
    return m_resources[physicalResource].texture;
}

void FrameGraph::execute(RhiCommandBuffer& commandBuffer, RhiFrameGraphBackend& backend) {
    MICROPROFILE_SCOPEI("FrameGraph", "Execute", 0xff00ff00);

    for (uint32_t pi = 0; pi < m_passes.size(); pi++) {
        auto& pass = m_passes[pi];
        if (pass.refCount == 0) continue;

        MICROPROFILE_SCOPEI("FrameGraph", pass.name.c_str(), 0xff0088ff);

        // Create transient textures at their producer pass
        for (uint32_t ri = 0; ri < m_resources.size(); ri++) {
            auto& res = m_resources[ri];
            if (res.kind == FGResourceKind::Texture &&
                !res.imported &&
                res.producer == pi &&
                res.physicalResource == ri &&
                res.texture == nullptr) {
                res.ownedTexture = backend.createTexture(res.desc);
                res.texture = res.ownedTexture.get();
            }
        }

        if (pass.type == FGPassType::Render) {
#ifdef _WIN32
            if (auto* vkCommandBuffer = dynamic_cast<VulkanCommandBuffer*>(&commandBuffer)) {
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
                        vkCommandBuffer->transitionTexture(resolveTexture(read.id),
                                                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                    }
                }
            }
#endif

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
}

void FrameGraph::reset() {
    m_resources.clear();
    m_passes.clear();
    m_passData.clear();
    m_ownedPasses.clear();
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
                if (res.exported)
                    ImGui::Text("%s %s, External", residency, resourceKindName(res.kind));
                else
                    ImGui::Text("%s %s", residency, resourceKindName(res.kind));
                ImGui::TableSetColumnIndex(3);
                if (res.kind == FGResourceKind::Texture && !res.imported)
                    ImGui::Text("%ux%u", res.desc.width, res.desc.height);
                else
                    ImGui::TextUnformatted("-");
                ImGui::TableSetColumnIndex(4);
                if (res.kind == FGResourceKind::Texture && !res.imported)
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
                if (res.lastUser < m_passes.size())
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
