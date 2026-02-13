#include "frame_graph.h"
#include "render_pass.h"
#include <algorithm>
#include <cassert>

#include "imgui.h"

// --- FGBuilder ---

FGBuilder::FGBuilder(FrameGraph& fg, uint32_t passIndex)
    : m_fg(fg), m_passIndex(passIndex) {}

FGResource FGBuilder::create(const char* name, const FGTextureDesc& desc) {
    FGResource res;
    res.id = static_cast<uint32_t>(m_fg.m_resources.size());
    FGResourceNode node;
    node.name = name;
    node.desc = desc;
    node.imported = false;
    node.producer = m_passIndex;
    m_fg.m_resources.push_back(std::move(node));
    // Creating a resource implicitly writes it
    m_fg.m_passes[m_passIndex].writes.push_back(res);
    return res;
}

FGResource FGBuilder::read(FGResource resource) {
    assert(resource.isValid());
    m_fg.m_passes[m_passIndex].reads.push_back(resource);
    return resource;
}

FGResource FGBuilder::write(FGResource resource) {
    assert(resource.isValid());
    m_fg.m_passes[m_passIndex].writes.push_back(resource);
    return resource;
}

void FGBuilder::setColorAttachment(uint32_t index, FGResource resource,
                                   MTL::LoadAction load, MTL::StoreAction store,
                                   MTL::ClearColor clear) {
    auto& pass = m_fg.m_passes[m_passIndex];
    assert(index < 8);
    pass.colorAttachments[index] = {resource, load, store, clear};
    if (index >= pass.colorAttachmentCount)
        pass.colorAttachmentCount = index + 1;
    // Implicitly writes the attachment
    write(resource);
}

void FGBuilder::setDepthAttachment(FGResource resource,
                                   MTL::LoadAction load, MTL::StoreAction store,
                                   double clearDepth) {
    auto& pass = m_fg.m_passes[m_passIndex];
    pass.depthAttachment = {resource, load, store, clearDepth, true};
    write(resource);
}

void FGBuilder::setSideEffect() {
    m_fg.m_passes[m_passIndex].hasSideEffect = true;
}

// --- FrameGraph ---

FGResource FrameGraph::import(const char* name, MTL::Texture* texture) {
    FGResource res;
    res.id = static_cast<uint32_t>(m_resources.size());
    FGResourceNode node;
    node.name = name;
    node.texture = texture;
    node.imported = true;
    m_resources.push_back(std::move(node));
    return res;
}

void FrameGraph::updateImport(FGResource res, MTL::Texture* texture) {
    assert(res.isValid() && res.id < m_resources.size());
    assert(m_resources[res.id].imported);
    m_resources[res.id].texture = texture;
}

void FrameGraph::resetTransients() {
    // Release and null transient textures so execute() reallocates them
    for (auto* tex : m_transientTextures) {
        tex->release();
    }
    m_transientTextures.clear();
    for (auto& res : m_resources) {
        if (!res.imported) {
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
    m_passData.push_back({});

    FGBuilder builder(*this, passIndex);
    passPtr->setup(builder);

    switch (passPtr->passType()) {
        case FGPassType::Render:
            node.executeRender = [passPtr](MTL::RenderCommandEncoder* enc) { passPtr->executeRender(enc); };
            break;
        case FGPassType::Compute:
            node.executeCompute = [passPtr](MTL::ComputeCommandEncoder* enc) { passPtr->executeCompute(enc); };
            break;
        case FGPassType::Blit:
            node.executeBlit = [passPtr](MTL::BlitCommandEncoder* enc) { passPtr->executeBlit(enc); };
            break;
    }
}

void FrameGraph::compile() {
    // Step 1: Initialize refCounts — side-effect passes start at 1
    for (auto& pass : m_passes) {
        pass.refCount = pass.hasSideEffect ? 1 : 0;
    }

    // Step 2: Count readers per resource → resource.refCount
    for (auto& res : m_resources) {
        res.refCount = 0;
    }
    for (uint32_t pi = 0; pi < m_passes.size(); pi++) {
        auto& pass = m_passes[pi];
        for (auto& r : pass.reads) {
            m_resources[r.id].refCount++;
        }
    }

    // Step 3: Propagate resource refCounts to producer passes
    for (uint32_t ri = 0; ri < m_resources.size(); ri++) {
        auto& res = m_resources[ri];
        if (res.refCount > 0 && res.producer != UINT32_MAX) {
            m_passes[res.producer].refCount += res.refCount;
        }
    }

    // Step 4: Cull passes with refCount == 0
    // (We don't remove them, just skip during execute)

    // Step 5: Calculate lastUser for each transient resource
    for (uint32_t pi = 0; pi < m_passes.size(); pi++) {
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

MTL::Texture* FrameGraph::getTexture(FGResource res) const {
    assert(res.isValid() && res.id < m_resources.size());
    return m_resources[res.id].texture;
}

void FrameGraph::execute(MTL::CommandBuffer* cmdBuf, MTL::Device* device, TracyMetalCtxHandle tracyCtx) {
    for (uint32_t pi = 0; pi < m_passes.size(); pi++) {
        auto& pass = m_passes[pi];
        if (pass.refCount == 0) continue;

        // Create transient textures at their producer pass
        for (uint32_t ri = 0; ri < m_resources.size(); ri++) {
            auto& res = m_resources[ri];
            if (!res.imported && res.producer == pi && res.texture == nullptr) {
                auto* texDesc = MTL::TextureDescriptor::texture2DDescriptor(
                    res.desc.format, res.desc.width, res.desc.height, false);
                texDesc->setStorageMode(res.desc.storageMode);
                texDesc->setUsage(res.desc.usage);
                res.texture = device->newTexture(texDesc);
                m_transientTextures.push_back(res.texture);
            }
        }

        if (pass.type == FGPassType::Render) {
            auto* rpDesc = MTL::RenderPassDescriptor::alloc()->init();
            for (uint32_t ci = 0; ci < pass.colorAttachmentCount; ci++) {
                auto& ca = pass.colorAttachments[ci];
                auto* att = rpDesc->colorAttachments()->object(ci);
                att->setTexture(m_resources[ca.resource.id].texture);
                att->setLoadAction(ca.loadAction);
                att->setStoreAction(ca.storeAction);
                att->setClearColor(ca.clearColor);
            }
            if (pass.depthAttachment.bound) {
                auto* da = rpDesc->depthAttachment();
                da->setTexture(m_resources[pass.depthAttachment.resource.id].texture);
                da->setLoadAction(pass.depthAttachment.loadAction);
                da->setStoreAction(pass.depthAttachment.storeAction);
                da->setClearDepth(pass.depthAttachment.clearDepth);
            }

            static TracyMetalSrcLoc srcLocs[64];
            srcLocs[pi] = {pass.name.c_str(), "FrameGraph::execute", __FILE__, __LINE__, 0};
            auto gpuZone = tracyMetalZoneBeginRender(tracyCtx, rpDesc, &srcLocs[pi]);

            auto* encoder = cmdBuf->renderCommandEncoder(rpDesc);
            pass.executeRender(encoder);
            encoder->endEncoding();
            tracyMetalZoneEnd(gpuZone);
            rpDesc->release();
        } else if (pass.type == FGPassType::Compute) {
            auto* cpDesc = MTL::ComputePassDescriptor::alloc()->init();

            static TracyMetalSrcLoc srcLocs[64];
            srcLocs[pi] = {pass.name.c_str(), "FrameGraph::execute", __FILE__, __LINE__, 0};
            auto gpuZone = tracyMetalZoneBeginCompute(tracyCtx, cpDesc, &srcLocs[pi]);

            auto* encoder = cmdBuf->computeCommandEncoder(cpDesc);
            cpDesc->release();
            pass.executeCompute(encoder);
            encoder->endEncoding();
            tracyMetalZoneEnd(gpuZone);
        } else if (pass.type == FGPassType::Blit) {
            auto* bpDesc = MTL::BlitPassDescriptor::alloc()->init();

            static TracyMetalSrcLoc srcLocs[64];
            srcLocs[pi] = {pass.name.c_str(), "FrameGraph::execute", __FILE__, __LINE__, 0};
            auto gpuZone = tracyMetalZoneBeginBlit(tracyCtx, bpDesc, &srcLocs[pi]);

            auto* encoder = cmdBuf->blitCommandEncoder(bpDesc);
            bpDesc->release();
            pass.executeBlit(encoder);
            encoder->endEncoding();
            tracyMetalZoneEnd(gpuZone);
        }

        // Release transient textures after their last user
        for (uint32_t ri = 0; ri < m_resources.size(); ri++) {
            auto& res = m_resources[ri];
            if (!res.imported && res.lastUser == pi && res.texture != nullptr) {
                res.texture->release();
                res.texture = nullptr;
            }
        }
    }
    m_transientTextures.clear();
}

void FrameGraph::reset() {
    m_resources.clear();
    m_passes.clear();
    m_passData.clear();
    m_transientTextures.clear();
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

const char* pixelFormatName(MTL::PixelFormat fmt) {
    switch (fmt) {
        case MTL::PixelFormatBGRA8Unorm:    return "BGRA8";
        case MTL::PixelFormatRGBA8Unorm:    return "RGBA8";
        case MTL::PixelFormatR32Uint:       return "R32Uint";
        case MTL::PixelFormatR32Float:      return "R32Float";
        case MTL::PixelFormatRG32Float:     return "RG32Float";
        case MTL::PixelFormatRGBA32Float:   return "RGBA32Float";
        case MTL::PixelFormatRGBA16Float:   return "RGBA16Float";
        case MTL::PixelFormatDepth32Float:  return "Depth32F";
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
               << " | Imported | Refs: " << res.refCount << "}\"];\n";
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
                   << "fillcolor=skyblue, label=\"{" << dotEscapeLabel(res.name)
                   << " | " << res.desc.width << "x" << res.desc.height
                   << " " << pixelFormatName(res.desc.format)
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
                ImGui::TextUnformatted(res.imported ? "Imported" : "Transient");
                ImGui::TableSetColumnIndex(3);
                if (!res.imported)
                    ImGui::Text("%ux%u", res.desc.width, res.desc.height);
                else
                    ImGui::TextUnformatted("-");
                ImGui::TableSetColumnIndex(4);
                ImGui::TextUnformatted(res.imported ? "-" : pixelFormatName(res.desc.format));
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
