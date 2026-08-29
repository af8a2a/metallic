#pragma once

#include "Runtime/Render/GAPI/Rhi.h"
#include "Runtime/Render/Subsystem/RenderWorld.h"

#include <any>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace metallic::render {

class HistoryResourceManager;
class RenderSubsystemHost;

using RenderSubsystemId = std::string_view;

struct RenderSubsystemInitContext {
    Device& device;
    RenderSubsystemHost& host;
};

struct RenderSubsystemFrameContext {
    Device& device;
    RenderSubsystemHost& host;
    RenderWorld* world = nullptr;
    HistoryResourceManager* historyResources = nullptr;
    Streamer* streamer = nullptr;
    CommandBuffer* commandBuffer = nullptr;
    uint64_t frameIndex = 0;
    uint32_t frameSlot = 0;
};

class RenderSubsystemShaderReload {
public:
    virtual ~RenderSubsystemShaderReload() = default;
    virtual void commit() noexcept = 0;
};

class IRenderSubsystem {
public:
    virtual ~IRenderSubsystem() = default;

    virtual Result initialize(const RenderSubsystemInitContext&, std::string&) { return {}; }
    virtual void onWorldChanged(RenderWorld*) {}
    virtual Result beginFrame(
        const RenderSubsystemFrameContext&,
        RenderChangeBits&,
        std::string&)
    {
        return {};
    }
    virtual Result recordPreGraph(const RenderSubsystemFrameContext&, std::string&) { return {}; }
    virtual Result recordPostGraph(const RenderSubsystemFrameContext&, std::string&) { return {}; }
    virtual Result prepareShaderReload(
        const RenderSubsystemInitContext&,
        std::unique_ptr<RenderSubsystemShaderReload>& outReload,
        std::string&)
    {
        outReload.reset();
        return {};
    }
    virtual void endFrame(const RenderSubsystemFrameContext&) {}
    virtual void shutdown() {}
};

using RenderSubsystemFactory = std::function<std::unique_ptr<IRenderSubsystem>()>;

struct RenderSubsystemRegistration {
    std::string id;
    std::vector<std::string> dependencies;
    RenderSubsystemFactory factory;
};

class RenderSubsystemHost {
public:
    RenderSubsystemHost();
    ~RenderSubsystemHost();

    RenderSubsystemHost(const RenderSubsystemHost&) = delete;
    RenderSubsystemHost& operator=(const RenderSubsystemHost&) = delete;
    RenderSubsystemHost(RenderSubsystemHost&&) noexcept = delete;
    RenderSubsystemHost& operator=(RenderSubsystemHost&&) noexcept = delete;

    bool registerSubsystem(RenderSubsystemRegistration registration, std::string& log);

    template <typename T>
    bool registerSubsystem(std::span<const RenderSubsystemId> dependencies, std::string& log)
    {
        RenderSubsystemRegistration registration;
        registration.id = std::string(T::kSubsystemId);
        registration.dependencies.reserve(dependencies.size());
        for (RenderSubsystemId dependency : dependencies) {
            registration.dependencies.emplace_back(dependency);
        }
        registration.factory = []() { return std::make_unique<T>(); };
        return registerSubsystem(std::move(registration), log);
    }

    template <typename T>
    bool registerSubsystem(std::string& log)
    {
        return registerSubsystem<T>({}, log);
    }

    template <typename T>
    bool configure(typename T::Desc desc, std::string& log)
    {
        const std::string id(T::kSubsystemId);
        if (isActive(id)) {
            log = "Render subsystem '" + id + "' is already active";
            return false;
        }
        configurations_[id] = std::move(desc);
        return true;
    }

    template <typename T>
    const typename T::Desc* configuration() const
    {
        const auto iter = configurations_.find(std::string(T::kSubsystemId));
        if (iter == configurations_.end()) {
            return nullptr;
        }
        return std::any_cast<typename T::Desc>(&iter->second);
    }

    Result initialize(Device& device, uint32_t frameSlotCount, std::string& log);
    Result activate(std::span<const RenderSubsystemId> ids, std::string& log);
    Result activate(RenderSubsystemId id, std::string& log);
    void setWorld(RenderWorld* world);

    Result beginFrame(
        uint64_t frameIndex,
        uint32_t frameSlot,
        HistoryResourceManager* historyResources,
        std::string& log);
    Result recordPreGraph(
        CommandBuffer& commandBuffer,
        Streamer* streamer,
        std::span<const RenderSubsystemId> requiredSubsystems,
        std::string& log);
    Result recordPostGraph(
        CommandBuffer& commandBuffer,
        Streamer* streamer,
        std::span<const RenderSubsystemId> requiredSubsystems,
        std::string& log);
    Result reloadShaders(std::string& log);
    void endFrame();
    void shutdown();

    IRenderSubsystem* get(RenderSubsystemId id);
    const IRenderSubsystem* get(RenderSubsystemId id) const;

    template <typename T>
    T* get()
    {
        return dynamic_cast<T*>(get(T::kSubsystemId));
    }

    template <typename T>
    const T* get() const
    {
        return dynamic_cast<const T*>(get(T::kSubsystemId));
    }

    bool isRegistered(RenderSubsystemId id) const;
    bool isActive(RenderSubsystemId id) const;
    RenderChangeBits lastChanges() const { return lastChanges_; }
    Device* device() const { return device_; }
    RenderWorld* world() const { return world_; }
    uint32_t frameSlotCount() const { return frameSlotCount_; }

    void retire(std::shared_ptr<void> resource);

private:
    struct Record;

    Result activateRecursive(const std::string& id, std::vector<std::string>& stack, std::string& log);
    bool dependencyClosure(
        std::span<const RenderSubsystemId> ids,
        std::vector<std::string>& outIds,
        std::string& log) const;
    RenderSubsystemFrameContext frameContext(CommandBuffer* commandBuffer, Streamer* streamer) const;

    std::unordered_map<std::string, std::unique_ptr<Record>> records_;
    std::unordered_map<std::string, std::any> configurations_;
    std::vector<std::string> activeOrder_;
    std::vector<std::string> preGraphOrder_;
    std::vector<std::vector<std::shared_ptr<void>>> retiredByFrameSlot_;
    Device* device_ = nullptr;
    RenderWorld* world_ = nullptr;
    HistoryResourceManager* historyResources_ = nullptr;
    uint64_t frameIndex_ = 0;
    uint32_t frameSlot_ = 0;
    uint32_t frameSlotCount_ = 0;
    RenderChangeBits lastChanges_ = RenderChangeBits::None;
    size_t begunSubsystemCount_ = 0;
    bool frameActive_ = false;
};

} // namespace metallic::render
