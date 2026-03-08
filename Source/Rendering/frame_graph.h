#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "rhi_backend.h"

struct FGResource {
    uint32_t id = UINT32_MAX;
    bool isValid() const { return id != UINT32_MAX; }
};

using FGTextureDesc = RhiTextureDesc;

struct FGResourceNode {
    std::string name;
    FGTextureDesc desc;
    RhiTexture* texture = nullptr;
    std::unique_ptr<RhiTexture> ownedTexture;
    bool imported = false;
    uint32_t refCount = 0;
    uint32_t producer = UINT32_MAX;
    uint32_t lastUser = 0;
};

enum class FGPassType { Render, Compute, Blit };

struct FGColorAttachment {
    FGResource resource;
    RhiLoadAction loadAction = RhiLoadAction::Clear;
    RhiStoreAction storeAction = RhiStoreAction::Store;
    RhiClearColor clearColor{};
};

struct FGDepthAttachment {
    FGResource resource;
    RhiLoadAction loadAction = RhiLoadAction::Clear;
    RhiStoreAction storeAction = RhiStoreAction::DontCare;
    double clearDepth = 1.0;
    bool bound = false;
};

struct FGPassNode {
    std::string name;
    FGPassType type = FGPassType::Render;
    uint32_t refCount = 0;
    bool hasSideEffect = false;

    std::vector<FGResource> reads;
    std::vector<FGResource> writes;

    FGColorAttachment colorAttachments[8];
    uint32_t colorAttachmentCount = 0;
    FGDepthAttachment depthAttachment;

    std::function<void(RhiRenderCommandEncoder&)> executeRender;
    std::function<void(RhiComputeCommandEncoder&)> executeCompute;
    std::function<void(RhiBlitCommandEncoder&)> executeBlit;
};

class FrameGraph;
class RenderPass;

class FGBuilder {
public:
    FGBuilder(FrameGraph& fg, uint32_t passIndex);

    FGResource create(const char* name, const FGTextureDesc& desc);
    FGResource read(FGResource resource);
    FGResource write(FGResource resource);

    void setColorAttachment(uint32_t index, FGResource resource,
                            RhiLoadAction load, RhiStoreAction store,
                            RhiClearColor clear = RhiClearColor());
    void setDepthAttachment(FGResource resource,
                            RhiLoadAction load, RhiStoreAction store,
                            double clearDepth = 1.0);
    void setSideEffect();

private:
    FrameGraph& m_fg;
    uint32_t m_passIndex;
};

class FrameGraph {
    friend class FGBuilder;
public:
    FGResource import(const char* name, RhiTexture* texture);
    void updateImport(FGResource res, RhiTexture* texture);
    void resetTransients();

    void addPass(std::unique_ptr<RenderPass> pass);

    template<typename Data, typename Setup, typename Exec>
    Data& addRenderPass(const char* name, Setup&& setup, Exec&& exec);

    template<typename Data, typename Setup, typename Exec>
    Data& addComputePass(const char* name, Setup&& setup, Exec&& exec);

    template<typename Data, typename Setup, typename Exec>
    Data& addBlitPass(const char* name, Setup&& setup, Exec&& exec);

    void compile();
    void execute(RhiCommandBuffer& commandBuffer, RhiFrameGraphBackend& backend);
    void reset();

    void exportGraphviz(std::ostream& os) const;
    void debugImGui() const;
    void renderPassUI();

    RhiTexture* getTexture(FGResource res) const;

private:
    std::vector<FGResourceNode> m_resources;
    std::vector<FGPassNode> m_passes;
    std::vector<std::unique_ptr<RenderPass>> m_ownedPasses;

    struct PassDataHolder {
        void* data = nullptr;
        void (*deleter)(void*) = nullptr;
        ~PassDataHolder() { if (data && deleter) deleter(data); }
        PassDataHolder() = default;
        PassDataHolder(PassDataHolder&& o) noexcept : data(o.data), deleter(o.deleter) { o.data = nullptr; }
        PassDataHolder& operator=(PassDataHolder&& o) noexcept {
            if (this != &o) {
                if (data && deleter) {
                    deleter(data);
                }
                data = o.data;
                deleter = o.deleter;
                o.data = nullptr;
            }
            return *this;
        }
        PassDataHolder(const PassDataHolder&) = delete;
        PassDataHolder& operator=(const PassDataHolder&) = delete;
    };
    std::vector<PassDataHolder> m_passData;
};

template<typename Data, typename Setup, typename Exec>
Data& FrameGraph::addRenderPass(const char* name, Setup&& setup, Exec&& exec) {
    uint32_t passIndex = static_cast<uint32_t>(m_passes.size());
    m_passes.push_back({});
    auto& pass = m_passes.back();
    pass.name = name;
    pass.type = FGPassType::Render;

    PassDataHolder holder;
    holder.data = new Data{};
    holder.deleter = [](void* p) { delete static_cast<Data*>(p); };
    m_passData.push_back(std::move(holder));

    Data& data = *static_cast<Data*>(m_passData.back().data);
    FGBuilder builder(*this, passIndex);
    setup(builder, data);

    pass.executeRender = [&data, fn = std::forward<Exec>(exec)](RhiRenderCommandEncoder& encoder) {
        fn(const_cast<const Data&>(data), encoder);
    };
    return data;
}

template<typename Data, typename Setup, typename Exec>
Data& FrameGraph::addComputePass(const char* name, Setup&& setup, Exec&& exec) {
    uint32_t passIndex = static_cast<uint32_t>(m_passes.size());
    m_passes.push_back({});
    auto& pass = m_passes.back();
    pass.name = name;
    pass.type = FGPassType::Compute;

    PassDataHolder holder;
    holder.data = new Data{};
    holder.deleter = [](void* p) { delete static_cast<Data*>(p); };
    m_passData.push_back(std::move(holder));

    Data& data = *static_cast<Data*>(m_passData.back().data);
    FGBuilder builder(*this, passIndex);
    setup(builder, data);

    pass.executeCompute = [&data, fn = std::forward<Exec>(exec)](RhiComputeCommandEncoder& encoder) {
        fn(const_cast<const Data&>(data), encoder);
    };
    return data;
}

template<typename Data, typename Setup, typename Exec>
Data& FrameGraph::addBlitPass(const char* name, Setup&& setup, Exec&& exec) {
    uint32_t passIndex = static_cast<uint32_t>(m_passes.size());
    m_passes.push_back({});
    auto& pass = m_passes.back();
    pass.name = name;
    pass.type = FGPassType::Blit;

    PassDataHolder holder;
    holder.data = new Data{};
    holder.deleter = [](void* p) { delete static_cast<Data*>(p); };
    m_passData.push_back(std::move(holder));

    Data& data = *static_cast<Data*>(m_passData.back().data);
    FGBuilder builder(*this, passIndex);
    setup(builder, data);

    pass.executeBlit = [&data, fn = std::forward<Exec>(exec)](RhiBlitCommandEncoder& encoder) {
        fn(const_cast<const Data&>(data), encoder);
    };
    return data;
}

