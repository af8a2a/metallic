#pragma once

#include "Runtime/Render/GAPI/Rhi.h"
#include <array>
#include <string>

namespace metallic::render {

// GPU cluster classification/compaction, micropolygon rasterization and depth
// resolve, with the triangle-queue path retained for runtime comparisons.
class VisibilityHybridRasterizer {
public:
    Result initialize(Device& device, uint32_t width, uint32_t height, std::string& log,
        uint32_t capacity = 262144, uint32_t clusterCapacity = 1);
    void begin(CommandBuffer& commands, float maxPixels, bool reversedZ);
    Result resolve(CommandBuffer& commands, Texture& visibilityTexture, TextureView& visibility,
        Texture& depthTexture, TextureView& depth, bool softwareRasterized = false);
    Result beginClusters(CommandBuffer& commands, float maxPixels, bool reversedZ,
        uint32_t producerPixelBuffer, uint32_t inputCount, bool stream);
    void finishClusterBins(CommandBuffer& commands);
    Buffer& clusterBuffer() const { return *clusterBuffer_; }
    Buffer& clusterArguments() const { return *clusterArguments_; }
    uint32_t clusterCapacity() const { return push_.clusterCapacity; }
    static constexpr uint32_t kDispatchWidth = 65535;
    static constexpr uint32_t kSoftwareBin = 4;
    Buffer& queueBuffer() const { return *buffers_[0]; }
    Buffer& pixelBuffer() const { return *buffers_[1]; }
    uint32_t width() const { return push_.width; }
    uint32_t height() const { return push_.height; }

private:
    struct Push {
        uint32_t queueBuffer = 0;
        uint32_t pixelBuffer = 0;
        uint32_t argumentsBuffer = 0;
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t capacity = 0;
        float maxPixels = 8.0f;
        uint32_t reversedZ = 1;
        uint32_t subpixelBits = 8;
        uint32_t clusterBuffer = 0;
        uint32_t clusterArgumentsBuffer = 0;
        uint32_t clusterCapacity = 1;
        uint32_t producerPixelBuffer = 0;
        uint32_t inputClusterCount = 0;
        uint32_t streamMode = 0;
    } push_;
    std::array<std::unique_ptr<Buffer>, 3> buffers_;
    std::unique_ptr<BindlessHeap> heap_;
    std::array<std::unique_ptr<ShaderModule>, 5> shaders_;
    std::array<std::unique_ptr<ComputePipeline>, 3> compute_;
    std::array<std::unique_ptr<GraphicsPipeline>, 2> resolve_;
    std::unique_ptr<Buffer> clusterBuffer_;
    std::unique_ptr<Buffer> clusterArguments_;
    std::array<std::unique_ptr<ShaderModule>, 4> clusterShaders_;
    std::array<std::unique_ptr<ComputePipeline>, 4> clusterPipelines_;
    bool clusterInitialized_ = false;
    bool initialized_ = false;
};

} // namespace metallic::render
