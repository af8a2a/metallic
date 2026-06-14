#include "Runtime/Render/history_resources.h"

#include <array>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

namespace metallic::render {
namespace {

constexpr uint32_t kHistorySlotCount = 2;

bool textureDescEquals(const TextureDesc& lhs, const TextureDesc& rhs)
{
    return lhs.type == rhs.type &&
        lhs.usage == rhs.usage &&
        lhs.format == rhs.format &&
        lhs.width == rhs.width &&
        lhs.height == rhs.height &&
        lhs.depth == rhs.depth &&
        lhs.mipCount == rhs.mipCount &&
        lhs.layerCount == rhs.layerCount &&
        lhs.memoryLocation == rhs.memoryLocation;
}

bool textureViewDescEquals(const TextureViewDesc& lhs, const TextureViewDesc& rhs)
{
    return lhs.format == rhs.format &&
        lhs.baseMip == rhs.baseMip &&
        lhs.mipCount == rhs.mipCount &&
        lhs.baseLayer == rhs.baseLayer &&
        lhs.layerCount == rhs.layerCount;
}

bool bufferDescEquals(const BufferDesc& lhs, const BufferDesc& rhs)
{
    return lhs.size == rhs.size &&
        lhs.structureStride == rhs.structureStride &&
        lhs.usage == rhs.usage &&
        lhs.memoryLocation == rhs.memoryLocation;
}

bool bufferViewDescEquals(const BufferViewDesc& lhs, const BufferViewDesc& rhs)
{
    return lhs.type == rhs.type &&
        lhs.offset == rhs.offset &&
        lhs.size == rhs.size &&
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
    if (viewDesc.mipCount == 0) {
        viewDesc.mipCount = 1;
    }
    if (viewDesc.layerCount == 0) {
        viewDesc.layerCount = 1;
    }
    return viewDesc;
}

BufferViewDesc normalizeBufferViewDesc(const BufferDesc& bufferDesc, BufferViewDesc viewDesc)
{
    if (viewDesc.offset < bufferDesc.size && viewDesc.size == UINT64_MAX) {
        viewDesc.size = bufferDesc.size - viewDesc.offset;
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
        bool valid = false;
    };

    struct BufferSlot {
        std::unique_ptr<Buffer> buffer;
        std::unique_ptr<BufferView> view;
        ResourceState state = ResourceState::Undefined;
        uint64_t generation = 0;
        bool valid = false;
    };

    struct Record {
        ResourceKind kind = ResourceKind::None;
        uint64_t generation = 0;
        TextureDesc textureDesc;
        TextureViewDesc textureViewDesc;
        BufferDesc bufferDesc;
        std::optional<BufferViewDesc> bufferViewDesc;
        std::array<TextureSlot, kHistorySlotCount> textureSlots;
        std::array<BufferSlot, kHistorySlotCount> bufferSlots;
    };

    Device* device = nullptr;
    uint32_t currentSlot = 0;
    uint32_t previousSlot = 1;
    uint64_t frameIndex = 0;
    std::unordered_map<std::string, Record> records;

    static bool textureRecordComplete(const Record& record)
    {
        if (record.kind != ResourceKind::Texture) {
            return false;
        }
        for (const TextureSlot& slot : record.textureSlots) {
            if (slot.texture == nullptr || slot.view == nullptr) {
                return false;
            }
        }
        return true;
    }

    static bool bufferRecordComplete(const Record& record)
    {
        if (record.kind != ResourceKind::Buffer) {
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
        return iter == records.end() ? nullptr : &iter->second;
    }

    const Record* findRecord(std::string_view name) const
    {
        const auto iter = records.find(std::string(name));
        return iter == records.end() ? nullptr : &iter->second;
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

    Result createTextureRecord(
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

        for (TextureSlot& slot : record.textureSlots) {
            Result result = device->createTexture(desc, slot.texture);
            if (!result || slot.texture == nullptr) {
                return result ? makeError(Error::Failure) : result;
            }

            result = device->createTextureView(*slot.texture, viewDesc, slot.view);
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

    Result createBufferRecord(
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

        for (BufferSlot& slot : record.bufferSlots) {
            Result result = device->createBuffer(desc, slot.buffer);
            if (!result || slot.buffer == nullptr) {
                return result ? makeError(Error::Failure) : result;
            }

            if (viewDesc.has_value()) {
                result = device->createBufferView(*slot.buffer, *viewDesc, slot.view);
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
};

HistoryResourceManager::HistoryResourceManager()
    : impl_(std::make_unique<Impl>())
{
}

HistoryResourceManager::~HistoryResourceManager() = default;
HistoryResourceManager::HistoryResourceManager(HistoryResourceManager&&) noexcept = default;
HistoryResourceManager& HistoryResourceManager::operator=(HistoryResourceManager&&) noexcept = default;

Result HistoryResourceManager::initialize(Device& device)
{
    impl_->records.clear();
    impl_->device = &device;
    impl_->currentSlot = 0;
    impl_->previousSlot = 1;
    impl_->frameIndex = 0;
    return {};
}

void HistoryResourceManager::reset()
{
    impl_->records.clear();
    impl_->device = nullptr;
    impl_->currentSlot = 0;
    impl_->previousSlot = 1;
    impl_->frameIndex = 0;
}

void HistoryResourceManager::beginFrame(uint64_t frameIndex)
{
    impl_->frameIndex = frameIndex;
    impl_->currentSlot = static_cast<uint32_t>(frameIndex & 1ull);
    impl_->previousSlot = impl_->currentSlot ^ 1u;

    for (auto& [name, record] : impl_->records) {
        (void)name;
        if (record.kind == Impl::ResourceKind::Texture) {
            record.textureSlots[impl_->currentSlot].valid = false;
        } else if (record.kind == Impl::ResourceKind::Buffer) {
            record.bufferSlots[impl_->currentSlot].valid = false;
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

void HistoryResourceManager::invalidateAll()
{
    for (auto& [name, record] : impl_->records) {
        (void)name;
        Impl::invalidateRecord(record);
    }
}

Result HistoryResourceManager::ensureTexture(
    std::string_view name,
    const TextureDesc& desc,
    TextureViewDesc viewDesc)
{
    if (impl_->device == nullptr || nameIsEmpty(name) || desc.type != TextureType::Texture2D) {
        return makeError(Error::InvalidArgument);
    }

    const TextureViewDesc normalizedViewDesc = normalizeTextureViewDesc(desc, viewDesc);
    const Impl::Record* existing = impl_->findRecord(name);
    if (existing != nullptr &&
        Impl::textureRecordComplete(*existing) &&
        textureDescEquals(existing->textureDesc, desc) &&
        textureViewDescEquals(existing->textureViewDesc, normalizedViewDesc)) {
        return {};
    }

    const uint64_t generation = existing != nullptr ? existing->generation + 1 : 1;
    Impl::Record newRecord;
    Result result = impl_->createTextureRecord(desc, normalizedViewDesc, generation, newRecord);
    if (!result) {
        return result;
    }

    impl_->records[std::string(name)] = std::move(newRecord);
    return {};
}

Result HistoryResourceManager::ensureBuffer(
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
        Impl::bufferRecordComplete(*existing) &&
        bufferDescEquals(existing->bufferDesc, desc) &&
        optionalBufferViewDescEquals(existing->bufferViewDesc, normalizedViewDesc)) {
        return {};
    }

    const uint64_t generation = existing != nullptr ? existing->generation + 1 : 1;
    Impl::Record newRecord;
    Result result = impl_->createBufferRecord(desc, normalizedViewDesc, generation, newRecord);
    if (!result) {
        return result;
    }

    impl_->records[std::string(name)] = std::move(newRecord);
    return {};
}

HistoryTextureRef HistoryResourceManager::texture(std::string_view name, HistorySlot slot) const
{
    const Impl::Record* record = impl_->findRecord(name);
    if (record == nullptr || record->kind != Impl::ResourceKind::Texture) {
        return {};
    }

    const Impl::TextureSlot& textureSlot = record->textureSlots[impl_->slotIndex(slot)];
    return HistoryTextureRef{
        .texture = textureSlot.texture.get(),
        .view = textureSlot.view.get(),
        .desc = &record->textureDesc,
        .valid = impl_->slotDataValid(textureSlot, *record),
    };
}

HistoryBufferRef HistoryResourceManager::buffer(std::string_view name, HistorySlot slot) const
{
    const Impl::Record* record = impl_->findRecord(name);
    if (record == nullptr || record->kind != Impl::ResourceKind::Buffer) {
        return {};
    }

    const Impl::BufferSlot& bufferSlot = record->bufferSlots[impl_->slotIndex(slot)];
    return HistoryBufferRef{
        .buffer = bufferSlot.buffer.get(),
        .view = bufferSlot.view.get(),
        .desc = &record->bufferDesc,
        .viewDesc = record->bufferViewDesc.has_value() ? &(*record->bufferViewDesc) : nullptr,
        .valid = impl_->slotDataValid(bufferSlot, *record),
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

Result HistoryResourceManager::transitionTexture(
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
    if (textureSlot.state == after && !forceBarrier) {
        return {};
    }

    TextureBarrierDesc barrier{
        .texture = textureSlot.texture.get(),
        .before = textureSlot.state,
        .after = after,
        .baseMip = 0,
        .mipCount = record->textureDesc.mipCount,
        .baseLayer = 0,
        .layerCount = record->textureDesc.layerCount,
    };
    commandBuffer.barrier(BarrierDesc{
        .textures = &barrier,
        .textureCount = 1,
    });
    textureSlot.state = after;
    return {};
}

Result HistoryResourceManager::transitionBuffer(
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
    if (bufferSlot.state == after && !forceBarrier) {
        return {};
    }

    BufferBarrierDesc barrier{
        .buffer = bufferSlot.buffer.get(),
        .before = bufferSlot.state,
        .after = after,
        .offset = 0,
        .size = record->bufferDesc.size,
    };
    commandBuffer.barrier(BarrierDesc{
        .buffers = &barrier,
        .bufferCount = 1,
    });
    bufferSlot.state = after;
    return {};
}

} // namespace metallic::render
