#pragma once

#include "Runtime/Render/Core/ResourceState.h"
#include "Runtime/Render/Debug/RenderDebug.h"
#include "Runtime/Render/Streamer/UploadStreamer.h"
#include "Runtime/Render/Profiling/RenderGraphProfile.h"
#include <chrono>

#include "Runtime/Render/GAPI/RHI.h"
#include "Runtime/Render/Core/DisplayOutput.h"
#include "Runtime/Render/Core/RenderView.h"
#include "Runtime/Render/RenderGraph/RenderGraphResourceLifecycle.h"
#include "Runtime/Render/Subsystem/RenderSubsystem.h"
#include "Runtime/Render/Streamer/SceneStreamingTypes.h"

#include "json.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <typeindex>
#include <utility>
#include <vector>

namespace metallic::scene {
class Scene;
}

namespace metallic::render {

class IRenderDebugObserver;
struct RenderGraphExecutionPassSnapshot;
struct RenderGraphExecutionResourceSnapshot;

class HistoryResourceManager;
struct MeshletStreamFrameDesc;

using RenderGraphProperties = nlohmann::json;

// Asset scenes retain their legacy local camera unless they explicitly opt in
// to the shared viewport. Scene ownership and camera ownership are independent.
inline bool renderGraphUsesLocalView(const RenderGraphProperties& properties)
{
    return properties.value("viewBinding",
        properties.value("sceneBinding", "world") == "asset" ? "local" : "global") == "local";
}

enum class RenderGraphFieldVisibility : uint8_t {
    Input,
    Output,
};

enum class RenderGraphResourceType : uint8_t {
    Texture2D,
    Buffer,
    AccelerationStructure,
};

enum class RenderGraphResourceAccess : uint8_t {
    None,
    TextureSampleRead,
    TextureColorWrite,
    TextureDepthStencilWrite,
    TextureTransferRead,
    TextureTransferWrite,
    TextureStorageReadWrite,
    BufferShaderRead,
    BufferStorageReadWrite,
    BufferTransferRead,
    BufferTransferWrite,
    BufferConstantRead,
    TextureStorageRead,
    TextureStorageWrite,
    BufferStorageRead,
    BufferStorageWrite,
    BufferIndirectRead,
    TextureSampleReadGeneral,
    AccelerationStructureBuildRead,
    AccelerationStructureBuildWrite,
    AccelerationStructureBuildReadWrite,
    AccelerationStructureShaderRead,
    BufferAccelerationStructureBuildRead,
    BufferAccelerationStructureScratchReadWrite,
};

enum class RenderGraphBindlessAccess : uint8_t {
    None,
    SampledImage,
    Buffer,
};

enum class RenderGraphPassKind : uint8_t {
    Raster,
    Compute,
    Unsafe,
    RayTracing,
};

struct RenderGraphInternalAccess {
    RenderGraphResourceAccess access = RenderGraphResourceAccess::None;
    RenderGraphPassKind kind = RenderGraphPassKind::Compute;
    bool operator==(const RenderGraphInternalAccess&) const = default;
};

enum class RenderGraphRuntimeSettingType : uint8_t {
    Bool,
    Int,
    Float,
    Float3,
    Color4,
    Enum,
    ActionCounter,
    Float4,
    String,
};

struct RenderGraphRuntimeSettingOption {
    std::string label;
    RenderGraphProperties value;
};

struct RenderGraphRuntimeSetting {
    std::string key;
    std::string label;
    RenderGraphRuntimeSettingType type = RenderGraphRuntimeSettingType::Bool;
    RenderGraphProperties defaultValue;
    RenderGraphProperties minValue;
    RenderGraphProperties maxValue;
    std::vector<RenderGraphRuntimeSettingOption> options;
    bool invalidateHistory = false;
    bool rebuildGraph = false;
};

enum class RenderGraphResourceLifetime : uint8_t { Persistent, Transient };
enum class RenderGraphInitialization : uint8_t { Unknown, Clear, FullOverwrite };

enum class RenderGraphTemporalAccess : uint8_t {
    None,
    ReadPrevious,
    WriteCurrent,
};

struct RenderGraphTemporalDesc {
    std::string key;
    int32_t frameOffset = 0;
    RenderGraphTemporalAccess access = RenderGraphTemporalAccess::None;
    bool graphOwnedHistory = false;
    bool perPassScope = false;
    bool requiresExternalHistory = false;
    bool operator==(const RenderGraphTemporalDesc&) const = default;
};

struct RenderGraphField {
    std::string name;
    std::string description;
    RenderGraphFieldVisibility visibility = RenderGraphFieldVisibility::Output;
    RenderGraphResourceType resourceType = RenderGraphResourceType::Texture2D;
    RenderGraphResourceAccess access = RenderGraphResourceAccess::TextureColorWrite;
    RenderGraphBindlessAccess bindlessAccess = RenderGraphBindlessAccess::None;
    Format format = Format::RGBA8Unorm;
    TextureUsageBits usage = TextureUsageBits::ColorAttachment;
    BufferUsageBits bufferUsage = BufferUsageBits::None;
    BufferViewType bufferViewType = BufferViewType::Raw;
    ResourceState state = ResourceState::ColorAttachment;
    bool optional = false;
    // Presentation outputs are execution roots without a manual markOutput().
    bool presentationOutput = false;
    RenderGraphResourceLifetime lifetime = RenderGraphResourceLifetime::Persistent;
    // Clear/FullOverwrite covers every successful pass invocation's entire
    // initial resource use (all texels/bytes), including internal stages. A write alone is insufficient;
    // the executor separately excludes passes it can skip for scene readiness.
    RenderGraphInitialization initialization = RenderGraphInitialization::Unknown;
    DisplayColorEncoding colorEncoding = DisplayColorEncoding::sRGB;
    // Disable for resampled inputs or auxiliary 2D outputs sized independently of the pass.
    bool matchOutputExtent = true;
    uint32_t width = 0;
    uint32_t height = 0;
    // ResourceType::Texture2D is the legacy graph texture category. The native
    // dimension is explicit so volume resources receive real Texture3D views.
    TextureType textureType = TextureType::Texture2D;
    uint32_t depth = 1;
    // Texture2D arrays use explicit layers; zero is input-only and accepts the
    // producer's layer count. Ordinary Texture2D inputs remain single-layer.
    uint32_t layerCount = 1;
    uint64_t size = 0;
    uint32_t structureStride = 0;
    // Output element schema registered by renderDebugLayouts(). Empty means raw;
    // stride alone never identifies a type. Input aliases inherit the producer schema.
    std::string debugLayout;
    MemoryLocation memoryLocation = MemoryLocation::Device;
    // Aggregate internal operations; access/state remain the stable pass boundary.
    std::vector<RenderGraphInternalAccess> internalAccesses;
    RenderGraphTemporalDesc temporal;

    bool isTemporal() const { return temporal.access != RenderGraphTemporalAccess::None; }

    bool operator==(const RenderGraphField&) const = default;

    RenderGraphField& texture2D(uint32_t newWidth = 0, uint32_t newHeight = 0);
    // Depth zero is allowed on inputs to accept a producer's declared depth.
    RenderGraphField& texture3D(uint32_t newWidth, uint32_t newHeight, uint32_t newDepth);
    RenderGraphField& buffer(uint64_t newSize, uint32_t newStructureStride = 0);
    RenderGraphField& bufferLayout(std::string layout);
    RenderGraphField& accelerationStructure();
    RenderGraphField& accelerationStructureRead();
    RenderGraphField& buildRead();
    RenderGraphField& buildWrite();
    RenderGraphField& buildReadWrite();
    RenderGraphField& setOptional(bool value = true);
    RenderGraphField& transient(RenderGraphInitialization initialization);
    RenderGraphField& temporalRead(std::string key, int32_t frameOffset = -1, bool graphOwned = false, bool perPassScope = true);
    RenderGraphField& temporalWrite(std::string key, bool graphOwned = false, bool perPassScope = true);
    RenderGraphField& sampledRead();
    RenderGraphField& colorWrite();
    RenderGraphField& depthStencilWrite();
    RenderGraphField& storageRead();
    RenderGraphField& storageWrite();
    RenderGraphField& storageReadWrite();
    RenderGraphField& transferRead();
    RenderGraphField& transferWrite();
    RenderGraphField& shaderRead();
    RenderGraphField& constantRead();
    RenderGraphField& bindlessSampledImage();
    RenderGraphField& bindlessBuffer();
    RenderGraphField& hostReadback();
    RenderGraphField& stageAccess(RenderGraphResourceAccess access,
        RenderGraphPassKind kind = RenderGraphPassKind::Compute);
};

class RenderPassReflection {
public:
    RenderGraphField& addInput(std::string name, std::string description = {});
    RenderGraphField& addBindlessSampledInput(std::string name, std::string description = {});
    RenderGraphField& addOutput(std::string name, std::string description = {});
    RenderGraphField& addTextureInput(std::string name, std::string description = {});
    RenderGraphField& addTextureOutput(std::string name, std::string description = {});
    RenderGraphField& addBufferInput(std::string name, std::string description = {});
    RenderGraphField& addBufferOutput(std::string name, std::string description = {});
    RenderGraphField& addAccelerationStructureInput(std::string name, std::string description = {});
    RenderGraphField& addAccelerationStructureOutput(std::string name, std::string description = {});

    const RenderGraphField* findField(
        std::string_view name,
        RenderGraphFieldVisibility visibility) const;
    const std::vector<RenderGraphField>& fields() const { return fields_; }

    bool operator==(const RenderPassReflection&) const = default;

private:
    std::vector<RenderGraphField> fields_;
};

struct RenderGraphCompileContext {
    Device* device = nullptr;
    Queue* graphicsQueue = nullptr;
    const scene::Scene* runtimeScene = nullptr;
    std::shared_ptr<PreparedSceneResources> preparedScene;
    RenderWorld* renderWorld = nullptr;
    RenderSubsystemHost* subsystemHost = nullptr;
    uint32_t width = 1;
    uint32_t height = 1;
    Format defaultFormat = Format::RGBA8Unorm;
    bool debugReadback = false;
    RenderView* renderView = nullptr;
    DisplayOutputParameters displayOutput;

    // Actual viewport extent. width/height remain the fixed render/reconstruction
    // target; only presentation passes resample to these dimensions.
    uint32_t presentationWidth = 0;
    uint32_t presentationHeight = 0;

    // Reload must keep the last successful graph instead of an initial error material.
    bool shaderReload = false;

    RenderWorld* world() const { return renderWorld; }
    RenderSubsystemHost* subsystems() const { return subsystemHost; }

    template <typename T>
    T* subsystem() const
    {
        return subsystemHost != nullptr ? subsystemHost->get<T>() : nullptr;
    }
};

struct RenderGraphResource {
    RenderGraphResourceType type = RenderGraphResourceType::Texture2D;
    Texture* texture = nullptr;
    TextureView* view = nullptr;
    TextureDesc desc;
    DisplayColorEncoding colorEncoding = DisplayColorEncoding::sRGB;
    Buffer* buffer = nullptr;
    BufferView* bufferView = nullptr;
    BufferDesc bufferDesc;
    BufferViewDesc bufferViewDesc;
    // Published by the owning pass after its builder selects TLAS or PTLAS.
    RayTracingAccelerationStructure* accelerationStructure = nullptr;
    uint64_t logicalResourceId = 0;
    // Scheduled boundary state; GPU completion is tracked by the frame context.
    ResourceState state = ResourceState::Undefined;
    BindlessHandle bindlessHandle;
    BindlessHandle sampledImageBindlessHandle;
    std::string debugLayout;
};

class TextureHandle {
public:
    TextureHandle() = default;

    bool valid() const;
    Texture* texture() const;
    TextureView* view() const;
    const TextureDesc& desc() const;
    const BindlessHandle& bindlessHandle() const;

private:
    explicit TextureHandle(RenderGraphResource* resource);

    RenderGraphResource* resource_ = nullptr;

    friend class RenderGraphExecutionContext;
};

class BufferHandle {
public:
    BufferHandle() = default;

    bool valid() const;
    Buffer* buffer() const;
    BufferView* view() const;
    const BufferDesc& desc() const;
    const BufferViewDesc& viewDesc() const;
    const BindlessHandle& bindlessHandle() const;

private:
    explicit BufferHandle(RenderGraphResource* resource);

    RenderGraphResource* resource_ = nullptr;

    friend class RenderGraphExecutionContext;
};

struct TemporalResourceBinding {
    LogicalResourceId logicalId;
    ResourceVersionId version;
    TextureHandle texture;
    BufferHandle buffer;
    bool valid = false;
    uint64_t invalidationRevision = 0;
};

class RenderGraphResourceRequestContext {
public:
    RenderGraphResourceRequestContext(
        Device* device,
        uint64_t frameIndex,
        uint64_t viewKey,
        uint32_t width,
        uint32_t height,
        std::string passName,
        const RenderGraphProperties& properties,
        const scene::Scene* runtimeScene,
        RenderWorld* world,
        RenderSubsystemHost* subsystems,
        const ViewConstants* viewConstants = nullptr,
        std::function<std::optional<std::array<double, 3>>(std::string_view)> observerResolver = {})
        : device_(device)
        , frameIndex_(frameIndex)
        , viewKey_(viewKey)
        , width_(width)
        , height_(height)
        , passName_(std::move(passName))
        , properties_(properties)
        , runtimeScene_(runtimeScene)
        , world_(world)
        , subsystems_(subsystems)
        , viewConstants_(viewConstants)
        , observerResolver_(std::move(observerResolver))
    {
    }

    Device* device() const { return device_; }
    uint64_t frameIndex() const { return frameIndex_; }
    uint64_t viewKey() const { return viewKey_; }
    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }
    const std::string& passName() const { return passName_; }
    const RenderGraphProperties& properties() const { return properties_; }
    const scene::Scene* runtimeScene() const { return runtimeScene_; }
    RenderWorld* world() const { return world_; }
    RenderSubsystemHost* subsystems() const { return subsystems_; }
    const ViewConstants* viewConstants() const { return viewConstants_; }

    template <typename T>
    T* subsystem() const
    {
        return subsystems_ != nullptr ? subsystems_->get<T>() : nullptr;
    }

    std::optional<std::array<double, 3>> inputObserver(std::string_view inputFieldName) const
    {
        return observerResolver_ ? observerResolver_(inputFieldName) : std::nullopt;
    }

private:
    Device* device_ = nullptr;
    uint64_t frameIndex_ = 0;
    uint64_t viewKey_ = 0;
    uint32_t width_ = 1;
    uint32_t height_ = 1;
    std::string passName_;
    const RenderGraphProperties& properties_;
    const scene::Scene* runtimeScene_ = nullptr;
    RenderWorld* world_ = nullptr;
    RenderSubsystemHost* subsystems_ = nullptr;
    const ViewConstants* viewConstants_ = nullptr;
    std::function<std::optional<std::array<double, 3>>(std::string_view)> observerResolver_;
};

class RenderGraphResourceRequestBuilder {
public:
    virtual ~RenderGraphResourceRequestBuilder() = default;

    virtual void publishOutputObserver(
        std::string_view outputFieldName,
        const std::array<double, 3>& observerWorldMetres) = 0;
    virtual Result<> bindResourceSetRaw(
        std::type_index type,
        std::shared_ptr<const void> data,
        std::shared_ptr<void> owner,
        LogicalResourceId setId,
        ResourceVersionId version) = 0;
    virtual Result<> registerProducerWork(RenderGraphProviderProducerWork work) = 0;
    virtual Result<> requireProducer(uint64_t producerDedupKey) = 0;
    virtual Result<> addExternalDependency(const GPUCompletionPoint& completion, std::string reason = {}) = 0;
    virtual Result<> declareExternalDomain(RenderGraphExternalDomainContract contract) = 0;
    virtual Result<> trackPublishedResource(const ExternalResourcePublication& publication, bool writes = false) = 0;

    template <typename T>
    Result<> bindResourceSet(
        std::shared_ptr<const T> data,
        std::shared_ptr<void> owner,
        LogicalResourceId setId,
        ResourceVersionId version)
    {
        return bindResourceSetRaw(
            std::type_index(typeid(T)),
            std::static_pointer_cast<const void>(std::move(data)),
            std::move(owner),
            std::move(setId),
            std::move(version));
    }
};

struct RenderPreparationTask {
    std::string name;
    uint32_t workload = 1;
    std::function<Result<>()> prepare;
};

struct RenderGraphStageUse {
    // Reflected field or named private import. Use input.field / output.field
    // when a short field name is ambiguous. Access is for this stage.
    std::string_view resource;
    RenderGraphResourceAccess access = RenderGraphResourceAccess::None;
};

struct RenderGraphStage {
    std::string_view name;
    std::span<const RenderGraphStageUse> uses;
    std::function<Result<>(CommandBuffer&)> record;
    RenderGraphPassKind kind = RenderGraphPassKind::Compute;
    // An opaque operation may retain an existing disjoint GPU fork/join. Its
    // callback must join completely and fulfill the declared boundary accesses.
    bool allowParallelCompute = false;
};

using RenderGraphComputeStage = RenderGraphStage;

struct RenderGraphBufferImport {
    std::string_view name;
    BufferSlice buffer;
    // Incoming access contract on this queue. executeComputeStages also limits
    // stage permissions to this contract; executeStages uses allocation usage.
    // Content invalidation/reset does not discard prior GPU hazards. The caller orders
    // prior external work before this sequence; imports do not introduce waits.
    RenderGraphResourceAccess access = RenderGraphResourceAccess::BufferStorageReadWrite;
};

struct RenderGraphTextureImport {
    std::string_view name;
    Texture* texture = nullptr;
    TextureView* view = nullptr;
    ResourceState initialState = ResourceState::Undefined;
    // Undefined means leave the last declared layout. The owner must track it
    // transactionally or retire the allocation if recording is cancelled.
    ResourceState finalState = ResourceState::Undefined;
};

struct RenderGraphAccelerationStructureImport {
    std::string_view name;
    RayTracingAccelerationStructure* accelerationStructure = nullptr;
    RenderGraphResourceAccess access = RenderGraphResourceAccess::AccelerationStructureBuildReadWrite;
};

class RenderGraphExecutionContext {
public:
    CommandBuffer& commandBuffer() const { return *commandBuffer_; }
    uint64_t frameIndex() const { return frameIndex_; }
    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }
    // Final display viewport, independent of this pass's internal render extent.
    uint32_t displayWidth() const { return displayWidth_; }
    uint32_t displayHeight() const { return displayHeight_; }
    const std::string& passName() const { return passName_; }
    const RenderGraphProperties& properties() const { return properties_; }
    HistoryResourceManager* historyResources() const { return historyResources_; }
    // Shared between passes of this graph and view. Ordinary camera motion preserves these textures.
    HistoryResourceManager& graphHistoryResources() const { return graphHistoryResources_; }
    Streamer* streamer() const { return streamer_; }
    const scene::Scene* runtimeScene() const { return runtimeScene_; }
    const PreparedSceneResources* preparedScene() const { return preparedScene_.get(); }
    RenderWorld* world() const { return world_; }
    RenderSubsystemHost* subsystems() const { return subsystems_; }
    const ViewConstants* viewConstants() const { return viewConstants_; }
    Buffer* viewConstantsBuffer() const { return viewConstantsBuffer_; }

    template <typename T>
    T* subsystem() const
    {
        return subsystems_ != nullptr ? subsystems_->get<T>() : nullptr;
    }

    const FrameExecutionResourceSnapshot* resourceSnapshot() const { return resourceSnapshot_.get(); }

    template <typename T>
    const T* resources() const
    {
        return resourceSnapshot_ != nullptr ? resourceSnapshot_->resourceSet<T>() : nullptr;
    }

    template <typename T>
    const T* resourceSet() const
    {
        return resources<T>();
    }

    TemporalResourceBinding temporalTexture(std::string_view fieldName) const;
    TemporalResourceBinding temporalBuffer(std::string_view fieldName) const;
    bool temporalValid(std::string_view fieldName) const;
    void setTemporalWritten(std::string_view fieldName, bool written);

    RenderGraphResource* resource(std::string_view fieldName) const;
    RenderGraphResource* input(std::string_view fieldName) const;
    RenderGraphResource* output(std::string_view fieldName) const;
    TextureHandle texture(std::string_view fieldName) const;
    TextureHandle inputTexture(std::string_view fieldName) const;
    TextureHandle outputTexture(std::string_view fieldName) const;
    BufferHandle buffer(std::string_view fieldName) const;
    BufferHandle inputBuffer(std::string_view fieldName) const;
    BufferHandle outputBuffer(std::string_view fieldName) const;
    RayTracingAccelerationStructure* inputAccelerationStructure(std::string_view fieldName) const;
    RayTracingAccelerationStructure* outputAccelerationStructure(std::string_view fieldName) const;
    // An empty scene may publish null. The owner keeps the public wrapper alive;
    // the graph retains its native allocation through recording/submission.
    Result<> publishAccelerationStructure(std::string_view fieldName, RayTracingAccelerationStructure* accelerationStructure);
    const BindlessHandle* bindlessResource(std::string_view fieldName) const;
    const BindlessHandle* bindlessInput(std::string_view fieldName) const;
    using CommandRecorder = std::function<Result<>(CommandBuffer&)>;
    bool supportsParallelCompute() const { return bool(parallelRecorder_); }
    // Fork after current commands; run disjoint compute/graphics branches, then
    // join before subsequent commands. Reacquire commandBuffer() after this call.
    // Only declared shared resources may cross queues. No submission occurs here.
    Result<> parallelCompute(const CommandRecorder& compute, const CommandRecorder& graphics);
    // One synchronous, single-queue sequence per pass execution. All declarations
    // are checked before recording. Callbacks must stay within their declared
    // accesses; graph fields cannot exceed reflection or change image layouts.
    // Buffer slices retain allocation identity; planning currently covers whole
    // allocations. Keep all pass GPU resource accesses inside these stages.
    Result<> executeComputeStages(std::span<const RenderGraphComputeStage> stages,
        std::span<const RenderGraphBufferImport> imports = {});
    // General single-queue stages. Reflected stageAccess declarations authorize
    // internal uses; graph image layouts are restored to the pass boundary.
    Result<> executeStages(std::span<const RenderGraphStage> stages,
        std::span<const RenderGraphBufferImport> buffers = {},
        std::span<const RenderGraphTextureImport> textures = {},
        std::span<const RenderGraphAccelerationStructureImport> accelerationStructures = {});
    // Join independent CPU jobs before returning, including on failure. Capture
    // frozen inputs and distinct output slots; never capture this context or
    // mutate frame/history/subsystems, issue commands, or publish from a job.
    // Thread-safe registry encoding is allowed. Apply outputs only on success.
    Result<> prepareJoined(std::span<const RenderPreparationTask> tasks);
    // GPU intervals use the command buffer's actual queue. Outer scopes follow
    // commandBuffer() across a fork/join; branch scopes bind their explicit buffer.
    // GPU scopes also emit nested debug labels, balanced per recording across a
    // fork/join. publishCpuProfile() contributes CPU metadata only, never labels.
    class ProfileScope {
    public:
        ProfileScope() = default;
        ProfileScope(RenderGraphExecutionContext& context, std::string_view name, CommandBuffer* commands);
        ~ProfileScope();
        ProfileScope(ProfileScope&& other) noexcept;
        ProfileScope& operator=(ProfileScope&& other) noexcept;
        ProfileScope(const ProfileScope&) = delete;
        ProfileScope& operator=(const ProfileScope&) = delete;
        void end();
        void next(std::string_view name);
    private:
        RenderGraphExecutionContext* context_ = nullptr;
        CommandBuffer* commands_ = nullptr;
        uint32_t index_ = UINT32_MAX;
        uint32_t parent_ = UINT32_MAX;
        std::chrono::steady_clock::time_point begin_;
    };
    ProfileScope profileScope(std::string_view name) { return {*this, name, nullptr}; }
    ProfileScope profileScope(CommandBuffer& commands, std::string_view name) { return {*this, name, &commands}; }
    void publishCpuProfile(std::span<const RenderGraphProfileSection> sections)
    {
        if (cpuProfile_) { cpuProfile_(sections, profileParent_); }
    }
    void publishStreamingProfile(SceneStreamingProfile sample)
    {
        sample.passName = passName_;
        if (streamingProfile_) { streamingProfile_(std::move(sample)); }
    }
    bool debugEnabled() const { return debugObserver_ != nullptr; }
    bool detailedDebugSnapshots() const;
    void debugCheckpoint(std::string_view name, std::span<const DebugResourceBinding> resources = {},
        const RenderGraphProperties& values = RenderGraphProperties::object());

private:
    std::shared_ptr<PreparedSceneResources> preparedScene_;
    struct Binding {
        std::string fieldName;
        RenderGraphResource* resource = nullptr;
        RenderGraphFieldVisibility visibility = RenderGraphFieldVisibility::Output;
        RenderGraphBindlessAccess bindlessAccess = RenderGraphBindlessAccess::None;
        SyncScope scope;
        bool internalLayouts = false;
        BindlessHandle bindlessHandle;
        BindlessHandle sampledImageBindlessHandle;
        LogicalResourceId logicalId;
        ResourceVersionId version;
        bool temporal = false;
        bool temporalValid = false;
        bool temporalWritten = true;
        uint64_t temporalInvalidationRevision = 0;
    };

    RenderGraphExecutionContext(
        CommandBuffer& commandBuffer,
        uint64_t frameIndex,
        uint32_t width,
        uint32_t height,
        std::string passName,
        const RenderGraphProperties& properties,
        std::vector<Binding> bindings,
        HistoryResourceManager* historyResources,
        HistoryResourceManager& graphHistoryResources,
        Streamer* streamer,
        const scene::Scene* runtimeScene,
        RenderWorld* world,
        RenderSubsystemHost* subsystems);

    using ParallelRecorder = std::function<Result<>(RenderGraphExecutionContext&, const CommandRecorder&, const CommandRecorder&)>;
    ParallelRecorder parallelRecorder_;
    uint32_t preparationWorkerLimit_ = 1;
    uint32_t preparationBatchWorkload_ = 1;
    uint32_t preparationTaskCount_ = 0;
    std::vector<RenderGraphWorkerInterval>* workerIntervals_ = nullptr;
    std::function<uint32_t(CommandBuffer&, std::string_view, uint32_t)> beginProfile_;
    std::function<void(CommandBuffer&, uint32_t, RenderGraphCPUInterval)> endProfile_;
    std::function<void(std::span<const RenderGraphProfileSection>, uint32_t)> cpuProfile_;
    std::function<void(SceneStreamingProfile)> streamingProfile_;
    uint32_t profileParent_ = UINT32_MAX;
    CommandBuffer* commandBuffer_ = nullptr;
    uint64_t frameIndex_ = 0;
    uint32_t width_ = 1;
    uint32_t height_ = 1;
    uint32_t displayWidth_ = 1;
    uint32_t displayHeight_ = 1;
    std::string passName_;
    const RenderGraphProperties& properties_;
    std::vector<Binding> bindings_;
    HistoryResourceManager* historyResources_ = nullptr;
    HistoryResourceManager& graphHistoryResources_;
    Streamer* streamer_ = nullptr;
    const scene::Scene* runtimeScene_ = nullptr;
    RenderWorld* world_ = nullptr;
    RenderSubsystemHost* subsystems_ = nullptr;
    std::shared_ptr<const FrameExecutionResourceSnapshot> resourceSnapshot_;
    const ViewConstants* viewConstants_ = nullptr;
    Buffer* viewConstantsBuffer_ = nullptr;
    IRenderDebugObserver* debugObserver_ = nullptr;
    RenderGraphExecutionPassSnapshot* executionCapture_ = nullptr;
    std::vector<RenderGraphExecutionResourceSnapshot>* capturedImports_ = nullptr;
    uint32_t debugPassId_ = 0;
    bool debugAfterPassPublished_ = false;
    bool computeStagesExecuted_ = false;
    bool computeStagesActive_ = false;
    bool stagesAllowParallel_ = false;
    Result<> executeStagesImpl(std::span<const RenderGraphStage> stages,
        std::span<const RenderGraphBufferImport> buffers,
        std::span<const RenderGraphTextureImport> textures,
        std::span<const RenderGraphAccelerationStructureImport> accelerationStructures, bool computeOnly);

    friend class RenderGraphExecutor;
};

// Scene dependencies are part of the pass contract, like texture/buffer inputs.
// World follows the bound document, falling back to path only without a valid world.
// An authored sceneBinding="asset" explicitly opts a World pass into a separate asset.
// Input inherits one producer's scene; every connected listed input must come from it.
// Unconnected inputs are allowed only when their reflection declares them optional.
enum class RenderGraphSceneSource { None, World, Input };
struct RenderGraphSceneDependency {
    RenderGraphSceneSource source = RenderGraphSceneSource::None;
    std::vector<std::string> inputs;
    bool operator==(const RenderGraphSceneDependency&) const = default;
};

enum class CPURecordingPolicy { Serial, ParallelJoined };

class RenderGraphPass {
public:
    virtual ~RenderGraphPass() = default;
    virtual RenderGraphSceneDependency sceneDependency() const { return {}; }
    virtual SceneStreamingRequirements sceneResourcesRequired(const RenderGraphCompileContext&) const { return {}; }
    // Pure view description; scene scheduling and IO belong to StreamerSubsystem.
    virtual void describeSceneView(const RenderGraphExecutionContext&, MeshletStreamFrameDesc&) const {}
    // Publish output encodings on the coordinator in graph order, before
    // resources are frozen for parallel recording. Only output metadata may
    // change here; do not issue commands or mutate input/pass/global state.
    virtual void prepareResourceMetadata(RenderGraphExecutionContext&) const {}
    // Pure CPU/snapshot resource request declaration before pass recording.
    // Must never record GPU commands directly; conditional producer work is
    // registered with the builder and scheduled by RenderGraphExecutor.
    virtual Result<> declareResourceRequests(
        const RenderGraphResourceRequestContext&,
        RenderGraphResourceRequestBuilder&,
        std::string&)
    {
        return {};
    }
    // Optional explicit external/SDK domain contract for passes owning native
    // resources or opaque SDK accesses outside reflected graph fields.
    virtual std::optional<RenderGraphExternalDomainContract> externalDomainContract() const
    {
        return std::nullopt;
    }
    // Render-only camera, HZB and descriptor setup before the subsystem's traversal.
    virtual Result<> prepareExecution(RenderGraphExecutionContext&) { return {}; }
    virtual void sceneTraversalCheckpoint(RenderGraphExecutionContext&, std::string_view) const {}

    virtual RenderPassReflection reflect(const RenderGraphCompileContext& context) const = 0;
    virtual RenderGraphPassKind kind() const;
    virtual QueueType queueType() const;
    virtual std::span<const RenderSubsystemId> requiredSubsystems() const;
    virtual std::vector<RenderGraphRuntimeSetting> runtimeSettings() const;
    virtual std::vector<std::string> debugCheckpoints() const { return {"AfterPass"}; }
    // Opt in only when execute() preserves resources/descriptors used by earlier
    // submissions. Legacy passes with singleton host uploads/readbacks wait.
    virtual bool supportsFrameOverlap() const { return false; }
    // External recording does not always add the graph's inter-frame GPU waits.
    virtual bool supportsExternalFrameOverlap() const { return supportsFrameOverlap(); }
    // Opt in only when GPU dependencies are reflected graph resources and any
    // private resources support the selected queue family. Other passes execute
    // on graphics and form an ordering boundary for independent graph branches.
    virtual bool supportsAsyncQueue() const { return false; }
    // Includes prepareExecution, execute, and acceptance callbacks: none may
    // modify host data/descriptors used by an earlier batch, or wait for later
    // recordings. Private resources must survive the aggregate frame completion.
    // Scene/SDK subsystems require a separate audit before graph-level opt-in.
    virtual bool supportsPipelinedSubmission() const { return false; }
    // Opt in ONLY when every cross-frame GPU resource hazard (read/write, private
    // allocations, history, bindless/BDA) is declared through graph fields,
    // temporal fields, or RenderGraph resource providers. Independent of
    // supportsFrameOverlap(), supportsAsyncQueue(), and supportsPipelinedSubmission().
    virtual bool supportsCrossFrameResourceScheduling() const { return false; }
    // Independent of GPU queue selection and frame overlap. ParallelJoined may
    // touch only this pass's private state, immutable prepared inputs and its
    // command buffer. No frame/global mutation, SDK hooks or nested task waits.
    // prepareExecution remains on the coordinator. Each context is handed back
    // before its sealed batch submits; other contexts may still be recording.
    virtual CPURecordingPolicy cpuRecordingPolicy() const { return CPURecordingPolicy::Serial; }
    virtual uint32_t recordingWorkload() const { return 1; }
    virtual Result<> prepare(const RenderGraphCompileContext& context, std::string& log);
    virtual Result<> compile(const RenderGraphCompileContext& context, std::string& log);
    virtual Result<> execute(RenderGraphExecutionContext& context) = 0;

    void setProperties(RenderGraphProperties properties) { properties_ = std::move(properties); }
    const RenderGraphProperties& properties() const { return properties_; }

private:
    RenderGraphProperties properties_ = RenderGraphProperties::object();
};

class RasterPass : public RenderGraphPass {
public:
    RenderGraphPassKind kind() const override;
    QueueType queueType() const override;
};

class ComputePass : public RenderGraphPass {
public:
    RenderGraphPassKind kind() const override;
    QueueType queueType() const override;
};

// Unsafe passes may record mixed graphics, compute, or transfer commands and are
// kept on the graphics queue until the graph can prove finer-grained hazards.
class UnsafePass : public RenderGraphPass {
public:
    RenderGraphPassKind kind() const override;
    QueueType queueType() const override;
};

using RenderGraphPassFactory = std::function<std::unique_ptr<RenderGraphPass>()>;

struct RenderGraphPassInfo {
    std::string type;
    std::string description;
    RenderGraphPassKind kind = RenderGraphPassKind::Unsafe;
    QueueType queueType = QueueType::Graphics;
};

const char* renderGraphPassKindName(RenderGraphPassKind kind);
bool registerRenderGraphPassType(
    std::string type,
    std::string description,
    RenderGraphPassFactory factory);
void registerBuiltInRenderGraphPasses();
RenderGraphSceneDependency renderGraphPassSceneDependency(std::string_view type);
std::unique_ptr<RenderGraphPass> createRenderGraphPass(std::string_view type);
std::vector<RenderGraphPassInfo> listRenderGraphPassTypes();

} // namespace metallic::render
