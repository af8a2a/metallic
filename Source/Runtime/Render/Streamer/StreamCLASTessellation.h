#pragma once

#include "Runtime/Render/GAPI/RHI.h"

#include <memory>
#include <string>
#include <functional>
#include <string_view>

namespace metallic::scene { class Scene; class MeshletStreamAsset; }
namespace metallic::render {

class ResourceRegistry;
class ScenePathTraceResources;
struct MeshletStreamFrameDesc;

enum class StreamCLASTessellationBuildMode : uint32_t { Triangles, Templates, Invalid };

inline StreamCLASTessellationBuildMode parseStreamCLASTessellationBuildMode(std::string_view value)
{
    if (value == "triangles") { return StreamCLASTessellationBuildMode::Triangles; }
    if (value == "templates") { return StreamCLASTessellationBuildMode::Templates; }
    return StreamCLASTessellationBuildMode::Invalid;
}

struct StreamCLASTessellationSettings {
    bool enabled = false;
    float edgePixels = 8.0f;
    uint32_t maxFactor = 4;
    uint32_t maxSplitDepth = 2;
    uint32_t maxPatches = 16384;
    StreamCLASTessellationBuildMode buildMode = StreamCLASTessellationBuildMode::Triangles;
};

struct StreamCLASTessellationInputs {
    Buffer* pages = nullptr;
    Buffer* pageTable = nullptr;
    Buffer* instances = nullptr;
    Buffer* primitives = nullptr;
    Buffer* groups = nullptr;
    Buffer* params = nullptr;
    Buffer* representations = nullptr;
    Buffer* representationGroups = nullptr;
    Buffer* representationState = nullptr;
    Buffer* blasHeader = nullptr;
    Buffer* tlasInstances = nullptr;
    uint32_t pageDescriptor = 0;
    uint32_t pageTableDescriptor = 0;
};

// Per-frame displaced geometry shares the runtime's ordered AS lifetime. A
// generated BLAS is published only when every patch of its instance fits.
class StreamCLASTessellation {
public:
    StreamCLASTessellation();
    ~StreamCLASTessellation();
    Result<> initialize(Device& device, std::shared_ptr<ResourceRegistry> registry,
        const scene::MeshletStreamAsset& asset, uint32_t maxPatches,
        StreamCLASTessellationBuildMode buildMode, std::string& log);
    // Capacity replacements in the same runtime retain semantic history and
    // material revisions independently of the generated geometry allocations.
    void reusePersistentState(const StreamCLASTessellation& previous);
    Result<> updateMaterials(const scene::Scene& scene, const ScenePathTraceResources& resources,
        std::string& log);
    Result<> record(CommandBuffer& commands, const StreamCLASTessellationInputs& inputs,
        const MeshletStreamFrameDesc& frame, std::string& log,
        const std::function<void(std::string_view)>& checkpoint = {});
    Result<> publish(CommandBuffer& commands);
    uint32_t patchCapacity() const;
    StreamCLASTessellationBuildMode buildMode() const;
    std::string diagnosticIdentity() const;
    Buffer* statisticsBuffer() const;
    Buffer* metadataBuffer() const;
    Buffer* vertexBuffer() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metallic::render
