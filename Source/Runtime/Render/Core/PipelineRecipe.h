#pragma once

#include "Runtime/Render/GAPI/RHI.h"

#include <filesystem>
#include <string>

namespace metallic::render {

enum class PipelineRecipeType : uint8_t { Compute, Graphics };

struct PipelineRecipeShader {
    std::vector<uint32_t> spirv;
    std::string moduleName;
    std::string sourceEntryPoint;
    std::string debugMode;
    std::string descriptorHeapMode;
    std::string entryPoint = "main";
    uint64_t inputSpirvHash = 0;
};

// Exported device configuration allows replay to preserve feature-dependent
// SPIR-V rewriting. These are enabled source-device features, not deductions
// from the shader's filename or fabricated render-pass state.
struct PipelineRecipeDevice {
    DeviceCapabilities capabilities;
    std::string deviceName;
    std::string driverName;
    uint32_t vendorId = 0;
    uint32_t deviceId = 0;
    uint32_t driverVersion = 0;
    uint32_t apiVersion = 0;
    std::string pipelineCacheUUID;
    bool validationEnabled = false;
    bool synchronizationValidationEnabled = false;
};

// Owns input bytecode and every currently exposed RHI PSO state. The cache
// handle is intentionally supplied by the experiment when reconstructing.
// Backend implicit state is identified by expectedPsoHash; changes to that
// state's contract require a schema version bump.
struct PipelineRecipe {
    PipelineRecipeType type = PipelineRecipeType::Compute;
    uint64_t expectedPsoHash = 0;
    std::string cacheGroup;
    PipelineRecipeDevice sourceDevice;
    PipelineRecipeShader computeShader;
    PipelineRecipeShader vertexShader;
    PipelineRecipeShader taskShader;
    PipelineRecipeShader meshShader;
    PipelineRecipeShader fragmentShader;
    uint32_t taskRequiredSubgroupSize = 0;
    bool taskRequireFullSubgroups = false;
    std::array<Format, GraphicsPipelineDesc::kMaxColorAttachments> colorFormats{};
    uint32_t colorAttachmentCount = 0;
    Format depthStencilFormat = Format::Unknown;
    PrimitiveTopology topology = PrimitiveTopology::TriangleList;
    RasterizationState rasterization;
    DepthStencilState depthStencil;
    bool usesBindlessHeap = false;
    bool indirectBindable = false;
    uint32_t bindlessUserPushDataSize = 0;

    ComputePipelineDesc computeDesc(ShaderModule* module, PipelineCache* cache = nullptr) const;
    GraphicsPipelineDesc graphicsDesc(ShaderModule* vertex, ShaderModule* task,
        ShaderModule* mesh, ShaderModule* fragment, PipelineCache* cache = nullptr) const;
};

// A manifest and content-addressed sibling .spv files form one portable recipe.
// Import validates schema, required state, SPIR-V size and input-content hashes.
[[nodiscard]] Result<PipelineRecipe> readPipelineRecipe(const std::filesystem::path& path, std::string& log);
[[nodiscard]] Result<> writePipelineRecipe(const PipelineRecipe& recipe, const std::filesystem::path& path, std::string& log);

} // namespace metallic::render
