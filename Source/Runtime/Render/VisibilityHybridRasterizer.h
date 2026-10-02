#pragma once
#include "Runtime/Render/Core/StreamClusterCullParameters.h"
#include "Runtime/Render/Core/StreamCandidateParameters.h"
#include "Runtime/Render/Core/ComputeKernel.h"

#include "Runtime/Render/GAPI/RHI.h"
#include <array>
#include <string>

namespace metallic::render {

// GPU cluster classification/compaction, micropolygon rasterization and depth
// resolve, with the triangle-queue path retained for runtime comparisons.
class VisibilityHybridRasterizer {
public:
    Result<> initialize(Device& device, uint32_t width, uint32_t height, std::string& log,
        uint32_t capacity = 262144, uint32_t clusterCapacity = 1);
    // Extent changes within the pixel allocation preserve buffers, bindings and GPU states.
    // Previously recorded commands already contain their own push-constant extent.
    bool supportsRenderExtent(uint32_t width, uint32_t height) const;
    Result<> setRenderExtent(uint32_t width, uint32_t height);
    [[nodiscard]] Result<> begin(CommandBuffer& commands, float maxPixels, bool reversedZ);
    Result<> resolve(CommandBuffer& commands, Texture& visibilityTexture, TextureView& visibility,
        Texture& depthTexture, TextureView& depth, bool softwareRasterized = false);
    Result<> beginClusters(CommandBuffer& commands, float maxPixels, bool reversedZ,
        uint32_t producerPixelBuffer, uint32_t inputCount, bool stream, bool compact = false, bool tessellation = false);
    // beginClusters must publish the bin header before candidate preparation.
    Result<> prepareStreamClusterCandidates(CommandBuffer& commands, const ComputeKernel& kernel,
        ParameterWriter& writer, StreamCandidateParameters params);
    // Batch metadata culling, then dispatch geometry classification only for
    // survivors. The parameter packets retain resources for both stages.
    Result<> cullStreamClusters(CommandBuffer& commands, const ComputeKernel& kernel,
        ParameterWriter& writer, StreamClusterCullParameters params);
    Buffer& candidateArguments() const { return *candidateArguments_; }
    Result<> finishClusterBins(CommandBuffer& commands);
    Buffer& workloadBuffer() const { return *workloadBuffer_; }
    Buffer& clusterBuffer() const { return *clusterBuffer_; }
    Buffer& clusterArguments() const { return *clusterArguments_; }
    uint32_t clusterCapacity() const { return settings_.clusterCapacity; }
    static constexpr uint32_t kDispatchWidth = 65535;
    static constexpr uint32_t kSoftwareBin = 4;
    static constexpr uint32_t kCandidateBuildArgumentsOffset = 24;
    Buffer& queueBuffer() const { return *buffers_[0]; }
    Buffer& pixelBuffer() const { return *buffers_[1]; }
    uint32_t width() const { return settings_.width; }
    uint32_t height() const { return settings_.height; }

private:
    [[nodiscard]] Result<> prepareClusterCandidates(CommandBuffer& commands);
    Result<EncodedParameters> encodeRasterParameters(CommandBuffer& commands);
    Result<EncodedParameters> encodeBinParameters(CommandBuffer& commands);
    Device* device_ = nullptr;
    // Host settings only; GPU entrypoints use their shared parameter ABI.
    struct Settings {
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t capacity = 0;
        float maxPixels = 8.0f;
        uint32_t reversedZ = 1;
        uint32_t subpixelBits = 8;
        uint32_t clusterCapacity = 1;
        uint32_t producerPixelBuffer = 0;
        uint32_t inputClusterCount = 0;
        uint32_t streamMode = 0;
    } settings_;
    std::array<std::unique_ptr<Buffer>, 3> buffers_;
    std::array<std::unique_ptr<ShaderModule>, 5> shaders_;
    std::array<ComputeKernel, 3> rasterKernels_;
    std::array<std::unique_ptr<GraphicsPipeline>, 2> resolve_;
    std::unique_ptr<Buffer> workloadBuffer_;
    std::unique_ptr<Buffer> clusterBuffer_;
    std::unique_ptr<Buffer> clusterArguments_;
    std::unique_ptr<Buffer> candidateArguments_;
    bool compactCandidates_ = false;
    std::array<ComputeKernel, 4> clusterKernels_;
    bool clusterInitialized_ = false;
    bool initialized_ = false;
};

} // namespace metallic::render
