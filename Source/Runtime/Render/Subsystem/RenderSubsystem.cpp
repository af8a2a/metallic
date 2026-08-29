#include "Runtime/Render/Subsystem/RenderSubsystem.h"

#include "Runtime/Render/HistoryResources.h"

#include <algorithm>
#include <utility>

namespace metallic::render {

struct RenderSubsystemHost::Record {
    RenderSubsystemRegistration registration;
    std::unique_ptr<IRenderSubsystem> instance;
    bool activating = false;
};

RenderSubsystemHost::RenderSubsystemHost() = default;

RenderSubsystemHost::~RenderSubsystemHost()
{
    shutdown();
}

bool RenderSubsystemHost::registerSubsystem(
    RenderSubsystemRegistration registration,
    std::string& log)
{
    if (registration.id.empty() || !registration.factory) {
        log = "Render subsystem registration requires a non-empty id and factory";
        return false;
    }
    if (records_.contains(registration.id)) {
        log = "Render subsystem '" + registration.id + "' is already registered";
        return false;
    }
    for (const std::string& dependency : registration.dependencies) {
        if (dependency.empty() || dependency == registration.id) {
            log = "Render subsystem '" + registration.id + "' has an invalid dependency";
            return false;
        }
    }

    const std::string id = registration.id;
    auto record = std::make_unique<Record>();
    record->registration = std::move(registration);
    records_.emplace(id, std::move(record));
    return true;
}

Result RenderSubsystemHost::initialize(Device& device, uint32_t frameSlotCount, std::string& log)
{
    if (frameSlotCount == 0) {
        log = "RenderSubsystemHost frameSlotCount must be non-zero";
        return makeError(Error::InvalidArgument);
    }
    if (device_ != nullptr && device_ != &device) {
        log = "RenderSubsystemHost cannot be rebound to another Device without shutdown";
        return makeError(Error::InvalidArgument);
    }
    device_ = &device;
    frameSlotCount_ = frameSlotCount;
    retiredByFrameSlot_.resize(frameSlotCount_);
    return {};
}

Result RenderSubsystemHost::activate(RenderSubsystemId id, std::string& log)
{
    const RenderSubsystemId ids[] = {id};
    return activate(ids, log);
}

Result RenderSubsystemHost::activate(std::span<const RenderSubsystemId> ids, std::string& log)
{
    if (device_ == nullptr) {
        log = "RenderSubsystemHost must be initialized before activation";
        return makeError(Error::InvalidArgument);
    }
    for (RenderSubsystemId id : ids) {
        std::vector<std::string> stack;
        Result result = activateRecursive(std::string(id), stack, log);
        if (!result) {
            return result;
        }
    }
    return {};
}

Result RenderSubsystemHost::activateRecursive(
    const std::string& id,
    std::vector<std::string>& stack,
    std::string& log)
{
    auto iter = records_.find(id);
    if (iter == records_.end()) {
        log = "Render subsystem dependency is not registered: '" + id + "'";
        return makeError(Error::InvalidArgument);
    }
    Record& record = *iter->second;
    if (record.instance != nullptr) {
        return {};
    }
    if (record.activating) {
        auto cycleBegin = std::find(stack.begin(), stack.end(), id);
        log = "Render subsystem dependency cycle: ";
        for (auto cycleIter = cycleBegin; cycleIter != stack.end(); ++cycleIter) {
            if (cycleIter != cycleBegin) {
                log += " -> ";
            }
            log += *cycleIter;
        }
        log += " -> " + id;
        return makeError(Error::InvalidArgument);
    }

    record.activating = true;
    stack.push_back(id);
    for (const std::string& dependency : record.registration.dependencies) {
        Result result = activateRecursive(dependency, stack, log);
        if (!result) {
            stack.pop_back();
            record.activating = false;
            return result;
        }
    }
    stack.pop_back();
    record.activating = false;

    std::unique_ptr<IRenderSubsystem> instance = record.registration.factory();
    if (instance == nullptr) {
        log = "Render subsystem factory returned null for '" + id + "'";
        return makeError(Error::Failure);
    }
    Result result = instance->initialize(RenderSubsystemInitContext{*device_, *this}, log);
    if (!result) {
        log = "Render subsystem '" + id + "' initialization failed: " + log;
        return result;
    }
    instance->onWorldChanged(world_);
    record.instance = std::move(instance);
    activeOrder_.push_back(id);
    return {};
}

void RenderSubsystemHost::setWorld(RenderWorld* world)
{
    if (world_ == world) {
        return;
    }
    world_ = world;
    for (const std::string& id : activeOrder_) {
        records_.at(id)->instance->onWorldChanged(world_);
    }
}

Result RenderSubsystemHost::beginFrame(
    uint64_t frameIndex,
    uint32_t frameSlot,
    HistoryResourceManager* historyResources,
    std::string& log)
{
    if (device_ == nullptr || frameActive_ || frameSlot >= frameSlotCount_) {
        log = "RenderSubsystemHost beginFrame received invalid frame state";
        return makeError(Error::InvalidArgument);
    }
    retiredByFrameSlot_[frameSlot].clear();
    frameIndex_ = frameIndex;
    frameSlot_ = frameSlot;
    historyResources_ = historyResources;
    frameActive_ = true;
    begunSubsystemCount_ = 0;
    preGraphOrder_.clear();
    lastChanges_ = world_ != nullptr ? world_->consumeChanges() : RenderChangeBits::None;

    const RenderSubsystemFrameContext context = frameContext(nullptr, nullptr);
    for (const std::string& id : activeOrder_) {
        RenderChangeBits subsystemChanges = RenderChangeBits::None;
        Result result = records_.at(id)->instance->beginFrame(context, subsystemChanges, log);
        ++begunSubsystemCount_;
        lastChanges_ |= subsystemChanges;
        if (!result) {
            log = "Render subsystem '" + id + "' beginFrame failed: " + log;
            endFrame();
            return result;
        }
    }
    if (historyResources_ != nullptr) {
        if (hasRenderChange(lastChanges_, RenderChangeBits::InvalidateTemporalHistory)) {
            historyResources_->invalidateAll();
        }
    }
    return {};
}

bool RenderSubsystemHost::dependencyClosure(
    std::span<const RenderSubsystemId> ids,
    std::vector<std::string>& outIds,
    std::string& log) const
{
    std::unordered_map<std::string, bool> included;
    std::function<bool(const std::string&)> include = [&](const std::string& id) {
        const auto iter = records_.find(id);
        if (iter == records_.end() || iter->second->instance == nullptr) {
            log = "Render subsystem is not active: '" + id + "'";
            return false;
        }
        if (included[id]) {
            return true;
        }
        for (const std::string& dependency : iter->second->registration.dependencies) {
            if (!include(dependency)) {
                return false;
            }
        }
        included[id] = true;
        return true;
    };
    for (RenderSubsystemId id : ids) {
        if (!include(std::string(id))) {
            return false;
        }
    }
    for (const std::string& id : activeOrder_) {
        if (included[id]) {
            outIds.push_back(id);
        }
    }
    return true;
}

Result RenderSubsystemHost::recordPreGraph(
    CommandBuffer& commandBuffer,
    Streamer* streamer,
    std::span<const RenderSubsystemId> requiredSubsystems,
    std::string& log)
{
    if (!frameActive_) {
        log = "RenderSubsystemHost recordPreGraph requires an active frame";
        return makeError(Error::InvalidArgument);
    }
    std::vector<std::string> closure;
    if (!dependencyClosure(requiredSubsystems, closure, log)) {
        return makeError(Error::InvalidArgument);
    }
    preGraphOrder_.clear();
    const RenderSubsystemFrameContext context = frameContext(&commandBuffer, streamer);
    for (const std::string& id : closure) {
        Result result = records_.at(id)->instance->recordPreGraph(context, log);
        preGraphOrder_.push_back(id);
        if (!result) {
            log = "Render subsystem '" + id + "' recordPreGraph failed: " + log;
            return result;
        }
    }
    return {};
}

Result RenderSubsystemHost::recordPostGraph(
    CommandBuffer& commandBuffer,
    Streamer* streamer,
    std::span<const RenderSubsystemId> requiredSubsystems,
    std::string& log)
{
    if (!frameActive_) {
        log = "RenderSubsystemHost recordPostGraph requires an active frame";
        return makeError(Error::InvalidArgument);
    }
    (void)requiredSubsystems;
    const RenderSubsystemFrameContext context = frameContext(&commandBuffer, streamer);
    Result firstFailure;
    std::string failures;
    for (auto iter = preGraphOrder_.rbegin(); iter != preGraphOrder_.rend(); ++iter) {
        std::string hookLog;
        Result result = records_.at(*iter)->instance->recordPostGraph(context, hookLog);
        if (!result) {
            if (firstFailure) {
                firstFailure = result;
            }
            if (!failures.empty()) {
                failures += '\n';
            }
            failures += "Render subsystem '" + *iter + "' recordPostGraph failed: " + hookLog;
        }
    }
    preGraphOrder_.clear();
    if (!firstFailure) {
        log = std::move(failures);
    }
    return firstFailure;
}

Result RenderSubsystemHost::reloadShaders(std::string& log)
{
    log.clear();
    if (device_ == nullptr || frameActive_) {
        log = "RenderSubsystemHost shader reload requires an initialized host outside a frame";
        return makeError(Error::InvalidArgument);
    }

    std::vector<std::unique_ptr<RenderSubsystemShaderReload>> preparedReloads;
    preparedReloads.reserve(activeOrder_.size());
    for (const std::string& id : activeOrder_) {
        std::string subsystemLog;
        std::unique_ptr<RenderSubsystemShaderReload> preparedReload;
        Result result = records_.at(id)->instance->prepareShaderReload(
            RenderSubsystemInitContext{*device_, *this},
            preparedReload,
            subsystemLog);
        if (!result) {
            log = "Render subsystem '" + id + "' shader reload failed";
            if (!subsystemLog.empty()) {
                log += ": " + subsystemLog;
            }
            return result;
        }
        if (!subsystemLog.empty()) {
            if (!log.empty()) {
                log += '\n';
            }
            log += "Render subsystem '" + id + "': " + subsystemLog;
        }
        if (preparedReload != nullptr) {
            preparedReloads.push_back(std::move(preparedReload));
        }
    }
    for (const std::unique_ptr<RenderSubsystemShaderReload>& preparedReload : preparedReloads) {
        preparedReload->commit();
    }
    return {};
}

void RenderSubsystemHost::endFrame()
{
    if (!frameActive_) {
        return;
    }
    const RenderSubsystemFrameContext context = frameContext(nullptr, nullptr);
    for (size_t index = begunSubsystemCount_; index > 0; --index) {
        records_.at(activeOrder_[index - 1])->instance->endFrame(context);
    }
    preGraphOrder_.clear();
    begunSubsystemCount_ = 0;
    historyResources_ = nullptr;
    frameActive_ = false;
}

void RenderSubsystemHost::shutdown()
{
    endFrame();
    for (auto iter = activeOrder_.rbegin(); iter != activeOrder_.rend(); ++iter) {
        Record& record = *records_.at(*iter);
        record.instance->shutdown();
        record.instance.reset();
    }
    activeOrder_.clear();
    preGraphOrder_.clear();
    retiredByFrameSlot_.clear();
    device_ = nullptr;
    world_ = nullptr;
    frameSlotCount_ = 0;
    lastChanges_ = RenderChangeBits::None;
}

IRenderSubsystem* RenderSubsystemHost::get(RenderSubsystemId id)
{
    const auto iter = records_.find(std::string(id));
    return iter == records_.end() ? nullptr : iter->second->instance.get();
}

const IRenderSubsystem* RenderSubsystemHost::get(RenderSubsystemId id) const
{
    const auto iter = records_.find(std::string(id));
    return iter == records_.end() ? nullptr : iter->second->instance.get();
}

bool RenderSubsystemHost::isRegistered(RenderSubsystemId id) const
{
    return records_.contains(std::string(id));
}

bool RenderSubsystemHost::isActive(RenderSubsystemId id) const
{
    return get(id) != nullptr;
}

void RenderSubsystemHost::retire(std::shared_ptr<void> resource)
{
    if (resource == nullptr) {
        return;
    }
    if (!frameActive_ || frameSlot_ >= retiredByFrameSlot_.size()) {
        return;
    }
    retiredByFrameSlot_[frameSlot_].push_back(std::move(resource));
}

RenderSubsystemFrameContext RenderSubsystemHost::frameContext(
    CommandBuffer* commandBuffer,
    Streamer* streamer) const
{
    return RenderSubsystemFrameContext{
        .device = *device_,
        .host = const_cast<RenderSubsystemHost&>(*this),
        .world = world_,
        .historyResources = historyResources_,
        .streamer = streamer,
        .commandBuffer = commandBuffer,
        .frameIndex = frameIndex_,
        .frameSlot = frameSlot_,
    };
}

} // namespace metallic::render
