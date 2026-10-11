#pragma once

#include "Runtime/Render/GAPI/RHI.h"
#include "Runtime/Scene/Scene.h"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace metallic::render {

inline constexpr uint32_t kMaxNeuralTextureSets = 64;
inline constexpr uint32_t kInvalidNeuralTextureSetIndex = UINT32_MAX;

// Explicit input for reserved NTC consumers. Ordinary scene textures carry no NTC metadata.
// Channel selection belongs to the future consumer's decoding contract.
struct NeuralTextureBinding {
    uint32_t logicalTextureIndex = UINT32_MAX;
    int32_t imageIndex = scene::kInvalidSceneIndex;
};

struct NeuralTextureMemoryStats {
    uint32_t textureSetCount = 0;
    uint32_t cooperativeVectorTextureSetCount = 0;
    uint32_t genericInt8TextureSetCount = 0;
    uint32_t replacedTextureCount = 0;
    uint64_t conventionalTextureBytes = 0;
    uint64_t latentTextureBytes = 0;
    uint64_t weightBytes = 0;
    uint64_t metadataBytes = 0;

    uint64_t neuralBytes() const
    {
        return latentTextureBytes + weightBytes + metadataBytes;
    }
};

// Reserved NTC resource helper for future virtual-texture consumers.
// Ordinary scene loading and material rendering do not instantiate this helper.
class NeuralTextureResources final {
public:
    NeuralTextureResources();
    ~NeuralTextureResources();

    NeuralTextureResources(NeuralTextureResources&&) noexcept;
    NeuralTextureResources& operator=(NeuralTextureResources&&) noexcept;

    NeuralTextureResources(const NeuralTextureResources&) = delete;
    NeuralTextureResources& operator=(const NeuralTextureResources&) = delete;

    Result<> prepare(
        Device& device,
        const scene::Scene& scene,
        std::span<const NeuralTextureBinding> bindings,
        std::string& log);
    Result<> recordUploads(CommandBuffer& commandBuffer);
    void releaseUploadBuffers();
    void clear();

    bool active() const;
    bool uploaded() const;
    bool cooperativeVectorActive() const;
    uint64_t pendingUploadByteSize() const;
    uint32_t pendingUploadRegionCount() const;
    uint32_t textureSetCount() const;
    uint32_t logicalTextureSetIndex(uint32_t logicalTextureIndex) const;
    const std::vector<uint32_t>& logicalTextureSetIndices() const;
    const std::array<TextureView*, kMaxNeuralTextureSets>& latentTextureViews() const;
    Buffer* constantsBuffer() const;
    Buffer* weightsBuffer() const;
    Buffer* setInfoBuffer() const;
    const SamplerDesc& latentSampler() const;
    const NeuralTextureMemoryStats& memoryStats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metallic::render
