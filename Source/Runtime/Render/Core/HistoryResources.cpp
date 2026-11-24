#include "Runtime/Render/Core/ResourceSynchronization.h"
#include "Runtime/Render/Core/HistoryResources.h"
#include "Runtime/Render/Core/RenderFrameContext.h"

#include <algorithm>
#include <array>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace metallic::render {
namespace {

constexpr uint32_t kDefaultHistorySlotCount = 2;
constexpr uint32_t kMaxHistorySlotCount = 3;

bool textureDescSatisfies(const TextureDesc& existing, const TextureDesc& requested)
{
    return existing.type == requested.type &&
        hasFlag(existing.usage, requested.usage) &&
        existing.format == requested.format &&
        existing.width == requested.width &&
        existing.height == requested.height &&
        existing.depth == requested.depth &&
        existing.mipCount == requested.mipCount &&
        existing.layerCount == requested.layerCount &&
        existing.memoryLocation == requested.memoryLocation;
}

bool textureViewDescEquals(const TextureViewDesc& lhs, const TextureViewDesc& rhs)
{
    return lhs.format == rhs.format &&
        lhs.range.baseMip == rhs.range.baseMip &&
        lhs.range.mipCount == rhs.range.mipCount &&
        lhs.range.baseLayer == rhs.range.baseLayer &&
        lhs.range.layerCount == rhs.range.layerCount &&
        lhs.forceArray == rhs.forceArray;
}

bool bufferDescSatisfies(const BufferDesc& existing, const BufferDesc& requested)
{
    return existing.size == requested.size &&
        existing.structureStride == requested.structureStride &&
        hasFlag(existing.usage, requested.usage) &&
        existing.memoryLocation == requested.memoryLocation;
}

bool bufferViewDescEquals(const BufferViewDesc& lhs, const BufferViewDesc& rhs)
{
    return lhs.type == rhs.type &&
        lhs.range.offset == rhs.range.offset &&
        lhs.range.size == rhs.range.size &&
        lhs.structureStride == rhs.structureStride;
}

bool optionalBufferViewDescEquals(
    const std::optional<BufferViewDesc>& lhs,
    const std::optional<BufferViewDesc>& rhs)
{
    if (lhs.has_value() != rhs.has_value()) {
        return false;
    }
    if (!lhs.has_value()) {
        return true;
    }
    return bufferViewDescEquals(*lhs, *rhs);
}

TextureViewDesc normalizeTextureViewDesc(const TextureDesc& textureDesc, TextureViewDesc viewDesc)
{
    if (viewDesc.format == Format::Unknown) {
        viewDesc.format = textureDesc.format;
    }
    return viewDesc;
}

BufferViewDesc normalizeBufferViewDesc(const BufferDesc& bufferDesc, BufferViewDesc viewDesc)
{
    if (viewDesc.range.offset < bufferDesc.size && viewDesc.range.size == UINT64_MAX) {
        viewDesc.range.size = bufferDesc.size - viewDesc.range.offset;
    }
    if (viewDesc.structureStride == 0) {
        viewDesc.structureStride = bufferDesc.structureStride;
    }
    return viewDesc;
}

bool nameIsEmpty(std::string_view name)
{
    return name.empty();
}

} // namespace

struct HistoryResourceManager::Impl {
    enum class ResourceKind : uint8_t {
        None,
        Texture,
        Buffer,
    };

    struct TextureSlot {
        std::unique_ptr<Texture> texture;
        std::unique_ptr<TextureView> view;
        ResourceState state = ResourceState::Undefined;
        uint64_t generation = 0;
        uint64_t publishedVersion = 0;
        uint64_t writtenFrameIndex = 0;
        bool valid = false;
    };

    struct BufferSlot {
        std::unique_ptr<Buffer> buffer;
        std::unique_ptr<BufferView> view;
        ResourceState state = ResourceState::Undefined;
        uint64_t generation = 0;
        uint64_t publishedVersion = 0;
        uint64_t writtenFrameIndex = 0;
        bool valid = false;
    };

    struct Record {
        ResourceKind kind = ResourceKind::None;
        bool preserveGeometryMotion = false;
        uint64_t generation = 0;
        uint64_t nextVersion = 1;
        TextureDesc textureDesc;
        TextureViewDesc textureViewDesc;
        BufferDesc bufferDesc;
        std::optional<BufferViewDesc> bufferViewDesc;
        std::vector<TextureSlot> textureSlots;
        std::vector<BufferSlot> bufferSlots;
    };

    Device* device = nullptr;
    uint32_t slotCount = kDefaultHistorySlotCount;
    uint32_t currentSlot = 0;
    uint32_t previousSlot = 1;
    uint64_t frameIndex = 0;
    uint64_t invalidationRevision = 1;
    uint64_t reprojectionInvalidationRevision = 1;
    uint64_t sunLightingReprojectionInvalidationRevision = 1;
    uint64_t motionReprojectionInvalidationRevision = 1;
    uint64_t geometryReprojectionInvalidationRevision = 1;
    std::unordered_map<std::string, std::shared_ptr<Record>> records;

    void advanceInvalidationRevision()
    {
        ++invalidationRevision;
        if (invalidationRevision == 0) {
            invalidationRevision = 1;
        }
    }

    bool textureRecordComplete(const Record& record) const
    {
        if (record.kind != ResourceKind::Texture || record.textureSlots.size() != slotCount) {
            return false;
        }
        for (const TextureSlot& slot : record.textureSlots) {
            if (slot.texture == nullptr || slot.view == nullptr) {
                return false;
            }
        }
        return true;
    }

    bool bufferRecordComplete(const Record& record) const
    {
        if (record.kind != ResourceKind::Buffer || record.bufferSlots.size() != slotCount) {
            return false;
        }
        for (const BufferSlot& slot : record.bufferSlots) {
            if (slot.buffer == nullptr) {
                return false;
            }
            if (record.bufferViewDesc.has_value() && slot.view == nullptr) {
                return false;
            }
        }
        return true;
    }

    bool slotDataValid(const TextureSlot& slot, const Record& record) const
    {
        return slot.valid && slot.generation == record.generation;
    }

    bool slotDataValid(const BufferSlot& slot, const Record& record) const
    {
        return slot.valid && slot.generation == record.generation;
    }

    Record* findRecord(std::string_view name)
    {
        const auto iter = records.find(std::string(name));
        return iter == records.end() ? nullptr : iter->second.get();
    }

    const Record* findRecord(std::string_view name) const
    {
        const auto iter = records.find(std::string(name));
        return iter == records.end() ? nullptr : iter->second.get();
    }

    static void invalidateRecord(Record& record)
    {
        for (TextureSlot& slot : record.textureSlots) {
            slot.valid = false;
        }
        for (BufferSlot& slot : record.bufferSlots) {
            slot.valid = false;
        }
    }

    Result<> createTextureRecord(
        const TextureDesc& desc,
        const TextureViewDesc& viewDesc,
        uint64_t generation,
        Record& outRecord) const
    {
        if (device == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        Record record;
        record.kind = ResourceKind::Texture;
        record.generation = generation;
        record.textureDesc = desc;
        record.textureViewDesc = viewDesc;
        record.textureSlots.resize(slotCount);

        for (TextureSlot& slot : record.textureSlots) {
            Result<> result = device->createTexture(desc).transform([&](auto rhiValue) { slot.texture = std::move(rhiValue); });
            if (!result || slot.texture == nullptr) {
                return result ? makeError(Error::Failure) : result;
            }

            result = device->createTextureView(*slot.texture, viewDesc).transform([&](auto rhiValue) { slot.view = std::move(rhiValue); });
            if (!result || slot.view == nullptr) {
                return result ? makeError(Error::Failure) : result;
            }

            slot.state = ResourceState::Undefined;
            slot.generation = generation;
            slot.valid = false;
        }

        outRecord = std::move(record);
        return {};
    }

    Result<> createBufferRecord(
        const BufferDesc& desc,
        const std::optional<BufferViewDesc>& viewDesc,
        uint64_t generation,
        Record& outRecord) const
    {
        if (device == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        Record record;
        record.kind = ResourceKind::Buffer;
        record.generation = generation;
        record.bufferDesc = desc;
        record.bufferViewDesc = viewDesc;
        record.bufferSlots.resize(slotCount);

        for (BufferSlot& slot : record.bufferSlots) {
            Result<> result = device->createBuffer(desc).transform([&](auto rhiValue) { slot.buffer = std::move(rhiValue); });
            if (!result || slot.buffer == nullptr) {
                return result ? makeError(Error::Failure) : result;
            }

            if (viewDesc.has_value()) {
                result = device->createBufferView(*slot.buffer, *viewDesc).transform([&](auto rhiValue) { slot.view = std::move(rhiValue); });
                if (!result || slot.view == nullptr) {
                    return result ? makeError(Error::Failure) : result;
                }
                record.bufferViewDesc = slot.view->desc();
            }

            slot.state = ResourceState::Undefined;
            slot.generation = generation;
            slot.valid = false;
        }

        outRecord = std::move(record);
        return {};
    }

    uint32_t slotIndex(HistorySlot slot) const
    {
        switch (slot) {
        case HistorySlot::Current:
            return currentSlot;
        case HistorySlot::Previous:
            return previousSlot;
        }
        return currentSlot;
    }

    bool isValidFrameOffset(int32_t frameOffset) const
    {
        if (frameOffset == 0) {
            return true;
        }
        if (frameOffset > 0 || slotCount < 2) {
            return false;
        }
        const int64_t minOffset = -static_cast<int64_t>(slotCount - 1u);
        return static_cast<int64_t>(frameOffset) >= minOffset;
    }

    std::optional<uint32_t> trySlotIndexForOffset(int32_t frameOffset) const
    {
        if (!isValidFrameOffset(frameOffset)) {
            return std::nullopt;
        }
        const uint32_t count = std::max(slotCount, 1u);
        if (frameOffset == 0) {
            return currentSlot;
        }
        const uint64_t back = static_cast<uint64_t>(-static_cast<int64_t>(frameOffset));
        return static_cast<uint32_t>((currentSlot + count - back) % count);
    }
};

HistoryResourceManager::HistoryResourceManager()
    : impl_(std::make_unique<Impl>())
{
}

HistoryResourceManager::~HistoryResourceManager() = default;
HistoryResourceManager::HistoryResourceManager(HistoryResourceManager&&) noexcept = default;
HistoryResourceManager& HistoryResourceManager::operator=(HistoryResourceManager&&) noexcept = default;

Result<> HistoryResourceManager::initialize(Device& device, uint32_t slotCount)
{
    ++impl_->reprojectionInvalidationRevision;
    ++impl_->sunLightingReprojectionInvalidationRevision;
    ++impl_->motionReprojectionInvalidationRevision;
    ++impl_->geometryReprojectionInvalidationRevision;
    impl_->records.clear();
    impl_->device = &device;
    impl_->slotCount = std::clamp(slotCount, kDefaultHistorySlotCount, kMaxHistorySlotCount);
    impl_->currentSlot = 0;
    impl_->previousSlot = impl_->slotCount - 1;
    impl_->frameIndex = 0;
    impl_->advanceInvalidationRevision();
    return {};
}

void HistoryResourceManager::reset()
{
    ++impl_->reprojectionInvalidationRevision;
    ++impl_->sunLightingReprojectionInvalidationRevision;
    ++impl_->motionReprojectionInvalidationRevision;
    ++impl_->geometryReprojectionInvalidationRevision;
    impl_->records.clear();
    impl_->device = nullptr;
    impl_->slotCount = kDefaultHistorySlotCount;
    impl_->currentSlot = 0;
    impl_->previousSlot = 1;
    impl_->frameIndex = 0;
    impl_->advanceInvalidationRevision();
}

uint32_t HistoryResourceManager::slotCount() const
{
    return impl_->slotCount;
}

bool HistoryResourceManager::isValidFrameOffset(int32_t frameOffset) const
{
    return impl_->isValidFrameOffset(frameOffset);
}

void HistoryResourceManager::beginFrame(uint64_t frameIndex)
{
    impl_->frameIndex = frameIndex;
    const uint32_t count = std::max(impl_->slotCount, 1u);
    impl_->currentSlot = static_cast<uint32_t>(frameIndex % count);
    impl_->previousSlot = (impl_->currentSlot + count - 1u) % count;

    for (auto& [name, record] : impl_->records) {
        (void)name;
        if (record->kind == Impl::ResourceKind::Texture && impl_->currentSlot < record->textureSlots.size()) {
            record->textureSlots[impl_->currentSlot].valid = false;
        } else if (record->kind == Impl::ResourceKind::Buffer && impl_->currentSlot < record->bufferSlots.size()) {
            record->bufferSlots[impl_->currentSlot].valid = false;
        }
    }
}

void HistoryResourceManager::invalidate(std::string_view name)
{
    Impl::Record* record = impl_->findRecord(name);
    if (record == nullptr) {
        return;
    }
    Impl::invalidateRecord(*record);
}

void HistoryResourceManager::invalidateAll(HistoryInvalidationReason reason)
{
    for (auto& [name, record] : impl_->records) {
        (void)name;
        if (reason == HistoryInvalidationReason::GeometryMotion && record->preserveGeometryMotion) {
            continue;
        }
        Impl::invalidateRecord(*record);
    }
    impl_->advanceInvalidationRevision();
    if (reason != HistoryInvalidationReason::CameraMotion) {
        ++impl_->reprojectionInvalidationRevision;
        if (reason != HistoryInvalidationReason::SunLightingChange) {
            ++impl_->sunLightingReprojectionInvalidationRevision;
            if (reason != HistoryInvalidationReason::GeometryMotion) { ++impl_->motionReprojectionInvalidationRevision; }
            if (reason != HistoryInvalidationReason::LightingChange) { ++impl_->geometryReprojectionInvalidationRevision; }
        }
    }
}

uint64_t HistoryResourceManager::frameIndex() const
{
    return impl_->frameIndex;
}

uint64_t HistoryResourceManager::invalidationRevision() const
{
    return impl_->invalidationRevision;
}

uint64_t HistoryResourceManager::reprojectionInvalidationRevision() const
{
    return impl_->reprojectionInvalidationRevision;
}

uint64_t HistoryResourceManager::sunLightingReprojectionInvalidationRevision() const
{
    return impl_->sunLightingReprojectionInvalidationRevision;
}

uint64_t HistoryResourceManager::motionReprojectionInvalidationRevision() const
{
    return impl_->motionReprojectionInvalidationRevision;
}

uint64_t HistoryResourceManager::geometryReprojectionInvalidationRevision() const
{
    return impl_->geometryReprojectionInvalidationRevision;
}

Result<> HistoryResourceManager::ensureTexture(
    std::string_view name,
    const TextureDesc& desc,
    TextureViewDesc viewDesc,
    bool preserveGeometryMotion)
{
    if (impl_->device == nullptr || nameIsEmpty(name) || desc.type != TextureType::Texture2D) {
        return makeError(Error::InvalidArgument);
    }

    const TextureViewDesc normalizedViewDesc = normalizeTextureViewDesc(desc, viewDesc);
    const Impl::Record* existing = impl_->findRecord(name);
    if (existing != nullptr &&
        impl_->textureRecordComplete(*existing) &&
        existing->preserveGeometryMotion == preserveGeometryMotion &&
        textureDescSatisfies(existing->textureDesc, desc) &&
        textureViewDescEquals(existing->textureViewDesc, normalizedViewDesc)) {
        return {};
    }

    const uint64_t generation = existing != nullptr ? existing->generation + 1 : 1;
    Impl::Record newRecord;
    Result<> result = impl_->createTextureRecord(desc, normalizedViewDesc, generation, newRecord);
    if (!result) {
        return result;
    }
    newRecord.preserveGeometryMotion = preserveGeometryMotion;

    impl_->records[std::string(name)] = std::make_shared<Impl::Record>(std::move(newRecord));
    return {};
}

Result<> HistoryResourceManager::ensureBuffer(
    std::string_view name,
    const BufferDesc& desc,
    const BufferViewDesc* viewDesc)
{
    if (impl_->device == nullptr || nameIsEmpty(name)) {
        return makeError(Error::InvalidArgument);
    }

    const std::optional<BufferViewDesc> normalizedViewDesc =
        viewDesc != nullptr
            ? std::optional<BufferViewDesc>(normalizeBufferViewDesc(desc, *viewDesc))
            : std::nullopt;
    const Impl::Record* existing = impl_->findRecord(name);
    if (existing != nullptr &&
        impl_->bufferRecordComplete(*existing) &&
        bufferDescSatisfies(existing->bufferDesc, desc) &&
        optionalBufferViewDescEquals(existing->bufferViewDesc, normalizedViewDesc)) {
        return {};
    }

    const uint64_t generation = existing != nullptr ? existing->generation + 1 : 1;
    Impl::Record newRecord;
    Result<> result = impl_->createBufferRecord(desc, normalizedViewDesc, generation, newRecord);
    if (!result) {
        return result;
    }

    impl_->records[std::string(name)] = std::make_shared<Impl::Record>(std::move(newRecord));
    return {};
}

HistoryTextureRef HistoryResourceManager::texture(std::string_view name, HistorySlot slot) const
{
    return textureAtOffset(name, slot == HistorySlot::Previous ? -1 : 0);
}

HistoryTextureRef HistoryResourceManager::textureAtOffset(std::string_view name, int32_t frameOffset) const
{
    const auto found = impl_->records.find(std::string(name));
    if (found == impl_->records.end() || found->second->kind != Impl::ResourceKind::Texture) {
        return {};
    }
    const auto indexOpt = impl_->trySlotIndexForOffset(frameOffset);
    if (!indexOpt) {
        return {};
    }
    const auto& record = *found->second;
    const uint32_t index = *indexOpt;
    if (index >= record.textureSlots.size()) {
        return {};
    }
    const Impl::TextureSlot& textureSlot = record.textureSlots[index];
    uint64_t backingId = 0;
    if (textureSlot.texture != nullptr) {
        const auto mem = textureSlot.texture->memoryInfo();
        backingId = mem.backingAllocationId != 0 ? mem.backingAllocationId : mem.allocationId;
    }
    const bool valid = impl_->slotDataValid(textureSlot, record);
    return HistoryTextureRef{
        .texture = textureSlot.texture.get(),
        .view = textureSlot.view.get(),
        .desc = &record.textureDesc,
        .valid = valid,
        .state = textureSlot.state,
        .allocationGeneration = record.generation,
        .backingAllocationId = backingId,
        .version = valid ? textureSlot.publishedVersion : 0,
        .writtenFrameIndex = valid ? textureSlot.writtenFrameIndex : 0,
        .owner = found->second,
    };
}

HistoryBufferRef HistoryResourceManager::buffer(std::string_view name, HistorySlot slot) const
{
    return bufferAtOffset(name, slot == HistorySlot::Previous ? -1 : 0);
}

HistoryBufferRef HistoryResourceManager::bufferAtOffset(std::string_view name, int32_t frameOffset) const
{
    const auto found = impl_->records.find(std::string(name));
    if (found == impl_->records.end() || found->second->kind != Impl::ResourceKind::Buffer) {
        return {};
    }
    const auto indexOpt = impl_->trySlotIndexForOffset(frameOffset);
    if (!indexOpt) {
        return {};
    }
    const auto& record = *found->second;
    const uint32_t index = *indexOpt;
    if (index >= record.bufferSlots.size()) {
        return {};
    }
    const Impl::BufferSlot& bufferSlot = record.bufferSlots[index];
    uint64_t backingId = 0;
    if (bufferSlot.buffer != nullptr) {
        const auto mem = bufferSlot.buffer->memoryInfo();
        backingId = mem.backingAllocationId != 0 ? mem.backingAllocationId : mem.allocationId;
    }
    const bool valid = impl_->slotDataValid(bufferSlot, record);
    return HistoryBufferRef{
        .buffer = bufferSlot.buffer.get(),
        .view = bufferSlot.view.get(),
        .desc = &record.bufferDesc,
        .viewDesc = record.bufferViewDesc.has_value() ? &(*record.bufferViewDesc) : nullptr,
        .valid = valid,
        .state = bufferSlot.state,
        .allocationGeneration = record.generation,
        .backingAllocationId = backingId,
        .version = valid ? bufferSlot.publishedVersion : 0,
        .writtenFrameIndex = valid ? bufferSlot.writtenFrameIndex : 0,
        .owner = found->second,
    };
}

bool HistoryResourceManager::hasPrevious(std::string_view name) const
{
    const Impl::Record* record = impl_->findRecord(name);
    if (record == nullptr) {
        return false;
    }

    if (record->kind == Impl::ResourceKind::Texture) {
        return impl_->slotDataValid(record->textureSlots[impl_->previousSlot], *record);
    }
    if (record->kind == Impl::ResourceKind::Buffer) {
        return impl_->slotDataValid(record->bufferSlots[impl_->previousSlot], *record);
    }
    return false;
}

void HistoryResourceManager::markWritten(std::string_view name)
{
    Impl::Record* record = impl_->findRecord(name);
    if (record == nullptr) {
        return;
    }

    if (record->kind == Impl::ResourceKind::Texture) {
        Impl::TextureSlot& slot = record->textureSlots[impl_->currentSlot];
        slot.valid = true;
        slot.generation = record->generation;
    } else if (record->kind == Impl::ResourceKind::Buffer) {
        Impl::BufferSlot& slot = record->bufferSlots[impl_->currentSlot];
        slot.valid = true;
        slot.generation = record->generation;
    }
}

Result<> HistoryResourceManager::publishTextureState(
    CommandBuffer& commandBuffer,
    std::string_view name,
    HistorySlot slot,
    ResourceState state,
    bool written)
{
    return publishTextureStateAtOffset(commandBuffer, name, slot == HistorySlot::Previous ? -1 : 0, state, written);
}

Result<> HistoryResourceManager::publishTextureStateAtOffset(
    CommandBuffer& commandBuffer,
    std::string_view name,
    int32_t frameOffset,
    ResourceState state,
    bool written)
{
    const auto found = impl_->records.find(std::string(name));
    if (found == impl_->records.end() || found->second->kind != Impl::ResourceKind::Texture ||
        state == ResourceState::Undefined) {
        return makeError(Error::InvalidArgument);
    }
    const auto indexOpt = impl_->trySlotIndexForOffset(frameOffset);
    if (!indexOpt) { return makeError(Error::InvalidArgument); }
    const auto record = found->second;
    const uint32_t index = *indexOpt;
    if (index >= record->textureSlots.size()) { return makeError(Error::InvalidArgument); }
    auto& textureSlot = record->textureSlots[index];
    if (textureSlot.texture == nullptr) { return makeError(Error::InvalidArgument); }
    auto result = commandBuffer.retainResource(record);
    if (!result) { return result; }
    const ResourceState before = textureSlot.state;
    result = commandBuffer.addSubmissionTransaction(std::make_shared<SubmissionTransaction>(
        [] {}, [record, index, before] {
            auto& cancelledSlot = record->textureSlots[index];
            cancelledSlot.state = before;
            cancelledSlot.valid = false;
            cancelledSlot.publishedVersion = 0;
            cancelledSlot.writtenFrameIndex = 0;
        }));
    if (!result) { return result; }
    textureSlot.state = state;
    if (written) {
        textureSlot.valid = true;
        textureSlot.generation = record->generation;
        textureSlot.publishedVersion = record->nextVersion++;
        textureSlot.writtenFrameIndex = impl_->frameIndex;
    }
    return {};
}

Result<> HistoryResourceManager::publishBufferState(
    CommandBuffer& commandBuffer,
    std::string_view name,
    HistorySlot slot,
    ResourceState state,
    bool written)
{
    return publishBufferStateAtOffset(commandBuffer, name, slot == HistorySlot::Previous ? -1 : 0, state, written);
}

Result<> HistoryResourceManager::publishBufferStateAtOffset(
    CommandBuffer& commandBuffer,
    std::string_view name,
    int32_t frameOffset,
    ResourceState state,
    bool written)
{
    const auto found = impl_->records.find(std::string(name));
    if (found == impl_->records.end() || found->second->kind != Impl::ResourceKind::Buffer ||
        state == ResourceState::Undefined) {
        return makeError(Error::InvalidArgument);
    }
    const auto indexOpt = impl_->trySlotIndexForOffset(frameOffset);
    if (!indexOpt) { return makeError(Error::InvalidArgument); }
    const auto record = found->second;
    const uint32_t index = *indexOpt;
    if (index >= record->bufferSlots.size()) { return makeError(Error::InvalidArgument); }
    auto& bufferSlot = record->bufferSlots[index];
    if (bufferSlot.buffer == nullptr) { return makeError(Error::InvalidArgument); }
    auto result = commandBuffer.retainResource(record);
    if (!result) { return result; }
    const ResourceState before = bufferSlot.state;
    result = commandBuffer.addSubmissionTransaction(std::make_shared<SubmissionTransaction>(
        [] {}, [record, index, before] {
            auto& cancelledSlot = record->bufferSlots[index];
            cancelledSlot.state = before;
            cancelledSlot.valid = false;
            cancelledSlot.publishedVersion = 0;
            cancelledSlot.writtenFrameIndex = 0;
        }));
    if (!result) { return result; }
    bufferSlot.state = state;
    if (written) {
        bufferSlot.valid = true;
        bufferSlot.generation = record->generation;
        bufferSlot.publishedVersion = record->nextVersion++;
        bufferSlot.writtenFrameIndex = impl_->frameIndex;
    }
    return {};
}

Result<> HistoryResourceManager::commitAcceptedTextureStateAtOffset(
    std::string_view name,
    int32_t frameOffset,
    ResourceState state,
    bool written,
    uint64_t expectedGeneration,
    std::optional<uint32_t> explicitSlotIndex)
{
    const auto found = impl_->records.find(std::string(name));
    if (found == impl_->records.end() || found->second->kind != Impl::ResourceKind::Texture ||
        state == ResourceState::Undefined) {
        return makeError(Error::InvalidArgument);
    }
    const auto indexOpt = explicitSlotIndex.has_value() ? explicitSlotIndex : impl_->trySlotIndexForOffset(frameOffset);
    if (!indexOpt) { return makeError(Error::InvalidArgument); }
    auto& record = *found->second;
    if (expectedGeneration != 0 && record.generation != expectedGeneration) {
        return {};
    }
    const uint32_t index = *indexOpt;
    if (index >= record.textureSlots.size()) { return makeError(Error::InvalidArgument); }
    auto& textureSlot = record.textureSlots[index];
    if (textureSlot.texture == nullptr) { return makeError(Error::InvalidArgument); }
    textureSlot.state = state;
    if (written) {
        textureSlot.valid = true;
        textureSlot.generation = record.generation;
        textureSlot.publishedVersion = record.nextVersion++;
        textureSlot.writtenFrameIndex = impl_->frameIndex;
    }
    return {};
}

Result<> HistoryResourceManager::commitAcceptedBufferStateAtOffset(
    std::string_view name,
    int32_t frameOffset,
    ResourceState state,
    bool written,
    uint64_t expectedGeneration,
    std::optional<uint32_t> explicitSlotIndex)
{
    const auto found = impl_->records.find(std::string(name));
    if (found == impl_->records.end() || found->second->kind != Impl::ResourceKind::Buffer ||
        state == ResourceState::Undefined) {
        return makeError(Error::InvalidArgument);
    }
    const auto indexOpt = explicitSlotIndex.has_value() ? explicitSlotIndex : impl_->trySlotIndexForOffset(frameOffset);
    if (!indexOpt) { return makeError(Error::InvalidArgument); }
    auto& record = *found->second;
    if (expectedGeneration != 0 && record.generation != expectedGeneration) {
        return {};
    }
    const uint32_t index = *indexOpt;
    if (index >= record.bufferSlots.size()) { return makeError(Error::InvalidArgument); }
    auto& bufferSlot = record.bufferSlots[index];
    if (bufferSlot.buffer == nullptr) { return makeError(Error::InvalidArgument); }
    bufferSlot.state = state;
    if (written) {
        bufferSlot.valid = true;
        bufferSlot.generation = record.generation;
        bufferSlot.publishedVersion = record.nextVersion++;
        bufferSlot.writtenFrameIndex = impl_->frameIndex;
    }
    return {};
}

void HistoryResourceManager::invalidateSlotAtOffset(
    std::string_view name,
    int32_t frameOffset,
    uint64_t expectedGeneration,
    std::optional<uint32_t> explicitSlotIndex)
{
    const auto found = impl_->records.find(std::string(name));
    if (found == impl_->records.end()) { return; }
    const auto indexOpt = explicitSlotIndex.has_value() ? explicitSlotIndex : impl_->trySlotIndexForOffset(frameOffset);
    if (!indexOpt) { return; }
    auto& record = *found->second;
    if (expectedGeneration != 0 && record.generation != expectedGeneration) { return; }
    const uint32_t index = *indexOpt;
    if (record.kind == Impl::ResourceKind::Texture && index < record.textureSlots.size()) {
        auto& slot = record.textureSlots[index];
        slot.valid = false;
        slot.publishedVersion = 0;
        slot.writtenFrameIndex = 0;
    } else if (record.kind == Impl::ResourceKind::Buffer && index < record.bufferSlots.size()) {
        auto& slot = record.bufferSlots[index];
        slot.valid = false;
        slot.publishedVersion = 0;
        slot.writtenFrameIndex = 0;
    }
}

std::optional<uint32_t> HistoryResourceManager::slotIndexForOffset(int32_t frameOffset) const
{
    return impl_->trySlotIndexForOffset(frameOffset);
}

bool HistoryResourceManager::hasRecord(std::string_view name) const
{
    return impl_->findRecord(name) != nullptr;
}

uint64_t HistoryResourceManager::peekNextVersion(std::string_view name) const
{
    const Impl::Record* record = impl_->findRecord(name);
    return record != nullptr ? record->nextVersion : 0;
}

Result<> HistoryResourceManager::transitionTexture(
    CommandBuffer& commandBuffer,
    std::string_view name,
    HistorySlot slot,
    ResourceState after,
    bool forceBarrier)
{
    Impl::Record* record = impl_->findRecord(name);
    if (record == nullptr || record->kind != Impl::ResourceKind::Texture) {
        return makeError(Error::InvalidArgument);
    }

    Impl::TextureSlot& textureSlot = record->textureSlots[impl_->slotIndex(slot)];
    if (textureSlot.texture == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    if (metallic::render::RenderFrameContext::from(commandBuffer) != nullptr) {
        auto retained = commandBuffer.retainResource(impl_->records.at(std::string(name)));
        if (!retained) { return retained; }
    }
    // General history contains cross-frame read/write dependencies even without
    // a layout change. Do not rely on a host wait to order these GPU accesses.
    if (textureSlot.state == after && after != ResourceState::General && !forceBarrier) {
        return {};
    }

    TextureBarrierDesc barrier{
        .texture = textureSlot.texture.get(),
        .oldLayout = textureLayoutForResourceState(textureSlot.state),
        .newLayout = textureLayoutForResourceState(after),
        .before = resourceSyncScope(textureSlot.state, PipelineStageBits::AllCommands),
        .after = resourceSyncScope(after, PipelineStageBits::AllCommands),
        .range = {.baseMip = 0, .mipCount = record->textureDesc.mipCount, .baseLayer = 0, .layerCount = record->textureDesc.layerCount},
    };
    if (auto commandResult = commandBuffer.synchronize(BarrierDesc{
        .textures = {&barrier, 1},
    }); !commandResult) { return commandResult; }
    textureSlot.state = after;
    return {};
}

Result<> HistoryResourceManager::transitionBuffer(
    CommandBuffer& commandBuffer,
    std::string_view name,
    HistorySlot slot,
    ResourceState after,
    bool forceBarrier)
{
    Impl::Record* record = impl_->findRecord(name);
    if (record == nullptr || record->kind != Impl::ResourceKind::Buffer) {
        return makeError(Error::InvalidArgument);
    }

    Impl::BufferSlot& bufferSlot = record->bufferSlots[impl_->slotIndex(slot)];
    if (bufferSlot.buffer == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    if (metallic::render::RenderFrameContext::from(commandBuffer) != nullptr) {
        auto retained = commandBuffer.retainResource(impl_->records.at(std::string(name)));
        if (!retained) { return retained; }
    }
    if (bufferSlot.state == after && after != ResourceState::General && !forceBarrier) {
        return {};
    }

    BufferBarrierDesc barrier{
        .buffer = bufferSlot.buffer.get(),
        .before = resourceSyncScope(bufferSlot.state, PipelineStageBits::AllCommands),
        .after = resourceSyncScope(after, PipelineStageBits::AllCommands),
        .range = {.offset = 0, .size = record->bufferDesc.size},
    };
    if (auto commandResult = commandBuffer.synchronize(BarrierDesc{
        .buffers = {&barrier, 1},
    }); !commandResult) { return commandResult; }
    bufferSlot.state = after;
    return {};
}

} // namespace metallic::render
