#pragma once

#include <Metal/Metal.hpp>
#include <cstdint>
#include <functional>
#include <ostream>
#include <memory>
#include <string>
#include <vector>

#include "tracy_metal.h"

// --- Resource Handle ---

struct FGResource {
    uint32_t id = UINT32_MAX;
    bool isValid() const { return id != UINT32_MAX; }
};

// --- Texture Descriptor (transient resources) ---

struct FGTextureDesc {
    uint32_t width = 0;
    uint32_t height = 0;
    MTL::PixelFormat format = MTL::PixelFormatBGRA8Unorm;
    MTL::TextureUsage usage = MTL::TextureUsageRenderTarget;
    MTL::StorageMode storageMode = MTL::StorageModePrivate;

    static FGTextureDesc renderTarget(uint32_t w, uint32_t h, MTL::PixelFormat fmt) {
        return {w, h, fmt, MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead, MTL::StorageModePrivate};
    }
    static FGTextureDesc depthTarget(uint32_t w, uint32_t h) {
        return {w, h, MTL::PixelFormatDepth32Float, MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead, MTL::StorageModePrivate};
    }
    static FGTextureDesc storageTexture(uint32_t w, uint32_t h, MTL::PixelFormat fmt) {
        return {w, h, fmt, MTL::TextureUsageShaderWrite | MTL::TextureUsageShaderRead, MTL::StorageModePrivate};
    }
};

// --- Internal Node Types ---

struct FGResourceNode {
    std::string name;
    FGTextureDesc desc;
    MTL::Texture* texture = nullptr; // set during execute (transient) or import
    bool imported = false;
    uint32_t refCount = 0;
    uint32_t producer = UINT32_MAX;  // pass index that creates this resource
    uint32_t lastUser = UINT32_MAX;  // pass index of last live reader/writer
};

enum class FGPassType { Render, Compute, Blit };

struct FGColorAttachment {
    FGResource resource;
    MTL::LoadAction loadAction = MTL::LoadActionClear;
    MTL::StoreAction storeAction = MTL::StoreActionStore;
    MTL::ClearColor clearColor = MTL::ClearColor(0, 0, 0, 1);
    bool bound = false;
};

struct FGDepthAttachment {
    FGResource resource;
    MTL::LoadAction loadAction = MTL::LoadActionClear;
    MTL::StoreAction storeAction = MTL::StoreActionDontCare;
    double clearDepth = 1.0;
    bool bound = false;
};

struct FGPassNode {
    std::string name;
    FGPassType type;
    uint32_t refCount = 0;
    bool hasSideEffect = false;

    std::vector<FGResource> reads;
    std::vector<FGResource> writes;

    // Render pass config
    FGColorAttachment colorAttachments[8];
    uint32_t colorAttachmentCount = 0;
    FGDepthAttachment depthAttachment;

    // Execute callbacks (only one is set based on type)
    std::function<void(MTL::RenderCommandEncoder*)> executeRender;
    std::function<void(MTL::ComputeCommandEncoder*)> executeCompute;
    std::function<void(MTL::BlitCommandEncoder*)> executeBlit;
};

// --- Builder ---

class FrameGraph; // forward decl
class RenderPass; // forward decl

class FGBuilder {
public:
    FGBuilder(FrameGraph& fg, uint32_t passIndex);

    FGResource create(const char* name, const FGTextureDesc& desc);
    FGResource read(FGResource resource);
    FGResource write(FGResource resource);

    void setColorAttachment(uint32_t index, FGResource resource,
                            MTL::LoadAction load, MTL::StoreAction store,
                            MTL::ClearColor clear = MTL::ClearColor(0, 0, 0, 1));
    void setDepthAttachment(FGResource resource,
                            MTL::LoadAction load, MTL::StoreAction store,
                            double clearDepth = 1.0);
    void setSideEffect();

private:
    FrameGraph& m_fg;
    uint32_t m_passIndex;
};

// --- FrameGraph ---

class FrameGraph {
    friend class FGBuilder;
public:
    FGResource import(const char* name, MTL::Texture* texture);

    void addPass(std::unique_ptr<RenderPass> pass);

    template<typename Data, typename Setup, typename Exec>
    Data& addRenderPass(const char* name, Setup&& setup, Exec&& exec);

    template<typename Data, typename Setup, typename Exec>
    Data& addComputePass(const char* name, Setup&& setup, Exec&& exec);

    template<typename Data, typename Setup, typename Exec>
    Data& addBlitPass(const char* name, Setup&& setup, Exec&& exec);

    void compile();
    void execute(MTL::CommandBuffer* cmdBuf, MTL::Device* device, TracyMetalCtxHandle tracyCtx);
    void reset();

    void exportGraphviz(std::ostream& os) const;
    void debugImGui() const;
    void renderPassUI();

    MTL::Texture* getTexture(FGResource res) const;

private:
    std::vector<FGResourceNode> m_resources;
    std::vector<FGPassNode> m_passes;
    std::vector<std::unique_ptr<RenderPass>> m_ownedPasses;

    // Type-erased pass data storage
    struct PassDataHolder {
        void* data = nullptr;
        void (*deleter)(void*) = nullptr;
        ~PassDataHolder() { if (data && deleter) deleter(data); }
        PassDataHolder() = default;
        PassDataHolder(PassDataHolder&& o) noexcept : data(o.data), deleter(o.deleter) { o.data = nullptr; }
        PassDataHolder& operator=(PassDataHolder&& o) noexcept {
            if (this != &o) { if (data && deleter) deleter(data); data = o.data; deleter = o.deleter; o.data = nullptr; }
            return *this;
        }
        PassDataHolder(const PassDataHolder&) = delete;
        PassDataHolder& operator=(const PassDataHolder&) = delete;
    };
    std::vector<PassDataHolder> m_passData;

    // Transient textures created during execute
    std::vector<MTL::Texture*> m_transientTextures;
};

// --- Template implementations ---

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

    pass.executeRender = [&data, fn = std::forward<Exec>(exec)](MTL::RenderCommandEncoder* enc) {
        fn(const_cast<const Data&>(data), enc);
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

    pass.executeCompute = [&data, fn = std::forward<Exec>(exec)](MTL::ComputeCommandEncoder* enc) {
        fn(const_cast<const Data&>(data), enc);
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

    pass.executeBlit = [&data, fn = std::forward<Exec>(exec)](MTL::BlitCommandEncoder* enc) {
        fn(const_cast<const Data&>(data), enc);
    };
    return data;
}
