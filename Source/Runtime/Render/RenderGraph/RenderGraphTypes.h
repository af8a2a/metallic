#pragma once

#include "Runtime/Render/Profiling/RenderGraphProfile.h"
#include <chrono>

#include "Runtime/Render/GAPI/Rhi.h"
#include "Runtime/Render/DisplayOutput.h"
#include "Runtime/Render/RenderView.h"
#include "Runtime/Render/Subsystem/RenderSubsystem.h"
#include "Runtime/Render/Streamer/SceneStreamingTypes.h"

#include "json.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace metallic::scene {
class Scene;
}

namespace metallic::render {

class IRenderDebugObserver;
struct DebugResourceBinding;

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

struct RenderGraphField {
    std::string name;
    std::string description;
    RenderGraphFieldVisibility visibility = RenderGraphFieldVisibility::Output;
    RenderGraphResourceType resourceType = RenderGraphResourceType::Texture2D;
    RenderGraphResourceAccess access = RenderGraphResourceAccess::TextureColorWrite;
    RenderGraphBindlessAccess bindlessAccess = RenderGraphBindlessAccess::None;
    Format format = Format::Rgba8Unorm;
    TextureUsageBits usage = TextureUsageBits::ColorAttachment;
    BufferUsageBits bufferUsage = BufferUsageBits::None;
    BufferViewType bufferViewType = BufferViewType::Raw;
    ResourceState state = ResourceState::ColorAttachment;
    bool optional = false;
    // Presentation outputs are execution roots without a manual markOutput().
    bool presentationOutput = false;
    DisplayColorEncoding colorEncoding = DisplayColorEncoding::Srgb;
    // Disable for inputs that can be resampled to a different output extent.
    bool matchOutputExtent = true;
    uint32_t width = 0;
    uint32_t height = 0;
    uint64_t size = 0;
    uint32_t structureStride = 0;
    MemoryLocation memoryLocation = MemoryLocation::Device;
    // Aggregate internal operations; access/state remain the stable pass boundary.
    std::vector<RenderGraphInternalAccess> internalAccesses;

    bool operator==(const RenderGraphField&) const = default;

    RenderGraphField& texture2D(uint32_t newWidth = 0, uint32_t newHeight = 0);
    RenderGraphField& buffer(uint64_t newSize, uint32_t newStructureStride = 0);
    RenderGraphField& setOptional(bool value = true);
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
    Format defaultFormat = Format::Rgba8Unorm;
    bool debugReadback = false;
    RenderView* renderView = nullptr;
    DisplayOutputParameters displayOutput;

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
    DisplayColorEncoding colorEncoding = DisplayColorEncoding::Srgb;
    Buffer* buffer = nullptr;
    BufferView* bufferView = nullptr;
    BufferDesc bufferDesc;
    BufferViewDesc bufferViewDesc;
    // Scheduled boundary state; GPU completion is tracked by the frame context.
    ResourceState state = ResourceState::Undefined;
    BindlessHandle bindlessHandle;
    BindlessHandle sampledImageBindlessHandle;
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

class RenderGraphExecutionContext {
public:
    CommandBuffer& commandBuffer() const { return *commandBuffer_; }
    uint64_t frameIndex() const { return frameIndex_; }
    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }
    const std::string& passName() const { return passName_; }
    const RenderGraphProperties& properties() const { return properties_; }
    HistoryResourceManager* historyResources() const { return historyResources_; }
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

    RenderGraphResource* resource(std::string_view fieldName) const;
    RenderGraphResource* input(std::string_view fieldName) const;
    RenderGraphResource* output(std::string_view fieldName) const;
    TextureHandle texture(std::string_view fieldName) const;
    TextureHandle inputTexture(std::string_view fieldName) const;
    TextureHandle outputTexture(std::string_view fieldName) const;
    BufferHandle buffer(std::string_view fieldName) const;
    BufferHandle inputBuffer(std::string_view fieldName) const;
    BufferHandle outputBuffer(std::string_view fieldName) const;
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
        std::span<const RenderGraphTextureImport> textures = {});
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
        Streamer* streamer,
        const scene::Scene* runtimeScene,
        RenderWorld* world,
        RenderSubsystemHost* subsystems);

    using ParallelRecorder = std::function<Result<>(RenderGraphExecutionContext&, const CommandRecorder&, const CommandRecorder&)>;
    ParallelRecorder parallelRecorder_;
    uint32_t preparationWorkerLimit_ = 1;
    uint32_t preparationBatchWorkload_ = 1;
    uint32_t preparationTaskCount_ = 0;
    std::function<uint32_t(CommandBuffer&, std::string_view, uint32_t)> beginProfile_;
    std::function<void(CommandBuffer&, uint32_t, double)> endProfile_;
    std::function<void(std::span<const RenderGraphProfileSection>, uint32_t)> cpuProfile_;
    std::function<void(SceneStreamingProfile)> streamingProfile_;
    uint32_t profileParent_ = UINT32_MAX;
    CommandBuffer* commandBuffer_ = nullptr;
    uint64_t frameIndex_ = 0;
    uint32_t width_ = 1;
    uint32_t height_ = 1;
    std::string passName_;
    const RenderGraphProperties& properties_;
    std::vector<Binding> bindings_;
    HistoryResourceManager* historyResources_ = nullptr;
    Streamer* streamer_ = nullptr;
    const scene::Scene* runtimeScene_ = nullptr;
    RenderWorld* world_ = nullptr;
    RenderSubsystemHost* subsystems_ = nullptr;
    const ViewConstants* viewConstants_ = nullptr;
    Buffer* viewConstantsBuffer_ = nullptr;
    IRenderDebugObserver* debugObserver_ = nullptr;
    uint32_t debugPassId_ = 0;
    bool debugAfterPassPublished_ = false;
    bool computeStagesExecuted_ = false;
    bool computeStagesActive_ = false;
    bool stagesAllowParallel_ = false;
    Result<> executeStagesImpl(std::span<const RenderGraphStage> stages,
        std::span<const RenderGraphBufferImport> buffers,
        std::span<const RenderGraphTextureImport> textures, bool computeOnly);

    friend class RenderGraphExecutor;
};

// Scene dependencies are part of the pass contract, like texture/buffer inputs.
// World follows the bound document, falling back to path only without a valid world.
// An authored sceneBinding="asset" explicitly opts a World pass into a separate asset.
// Input inherits one producer's scene; every listed input must come from that producer.
enum class RenderGraphSceneSource { None, World, Input };
struct RenderGraphSceneDependency {
    RenderGraphSceneSource source = RenderGraphSceneSource::None;
    std::vector<std::string> inputs;
    bool operator==(const RenderGraphSceneDependency&) const = default;
};

enum class CpuRecordingPolicy { Serial, ParallelJoined };

class RenderGraphPass {
public:
    virtual ~RenderGraphPass() = default;
    virtual RenderGraphSceneDependency sceneDependency() const { return {}; }
    virtual SceneStreamingRequirements sceneResourcesRequired(const RenderGraphCompileContext&) const { return {}; }
    // Pure view description; scene scheduling and IO belong to StreamerSubsystem.
    virtual void describeSceneView(const RenderGraphExecutionContext&, MeshletStreamFrameDesc&) const {}
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
    // Opt in only when GPU dependencies are reflected graph resources and any
    // private resources support the selected queue family. Other passes execute
    // on graphics and form an ordering boundary for independent graph branches.
    virtual bool supportsAsyncQueue() const { return false; }
    // Includes prepareExecution, execute, and acceptance callbacks: none may
    // modify host data/descriptors used by an earlier batch, or wait for later
    // recordings. Private resources must survive the aggregate frame completion.
    // Scene/SDK subsystems require a separate audit before graph-level opt-in.
    virtual bool supportsPipelinedSubmission() const { return false; }
    // Independent of GPU queue selection and frame overlap. ParallelJoined may
    // touch only this pass's private state, immutable prepared inputs and its
    // command buffer. No frame/global mutation, SDK hooks or nested task waits.
    // prepareExecution remains on the coordinator. Each context is handed back
    // before its sealed batch submits; other contexts may still be recording.
    virtual CpuRecordingPolicy cpuRecordingPolicy() const { return CpuRecordingPolicy::Serial; }
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
