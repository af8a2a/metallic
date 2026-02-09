#include "frame_graph.h"
#include <algorithm>
#include <cassert>

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
}
