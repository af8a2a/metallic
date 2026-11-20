#pragma once

#include "Runtime/Render/Core/ShaderResourceABI.h"
#include "Runtime/Render/GAPI/RHI.h"
#include "Runtime/Scene/MeshDistanceField.h"

#include <array>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace metallic::render {

class ParameterWriter;

struct MeshDistanceFieldGPUBindings {
    GPUResourceHandle<ResourceViewKind::RawBuffer> mips;
    GPUResourceHandle<ResourceViewKind::RawBuffer> indirection;
    GPUResourceHandle<ResourceViewKind::SampledImage> atlas;
    GPUSamplerHandle atlasSampler;
};

class MeshDistanceFieldGPU {
public:
    Result<> initialize(Device& device, Queue& queue, const scene::Scene& scene,
        std::span<const std::filesystem::path> paths, std::string& log);
    Result<std::unique_ptr<Buffer>> uploadInstances(const scene::Scene& scene,
        uint32_t& count, uint32_t& unsupportedTransforms,
        std::unique_ptr<Buffer>* objectGridObjects = nullptr) const;
    MeshDistanceFieldGPUBindings bind(ParameterWriter& writer) const;
    Texture& atlas() const { return *atlas_; }
    uint32_t bricksPerAxis() const { return atlas_->desc().width / scene::kMeshDistanceFieldBrickSize; }
    size_t assetCount() const { return assets_.size(); }

private:
    struct Asset {
        int32_t meshIndex;
        uint32_t mipBase;
        std::array<float, 3> localCenter;
        std::array<float, 3> normalizedExtent;
        std::array<float, 3> geometryCenter;
        std::array<float, 3> geometryExtent;
        float volumeToLocal;
    };

    Device* device_ = nullptr;
    std::vector<Asset> assets_;
    std::unique_ptr<Texture> atlas_;
    std::unique_ptr<TextureView> atlasView_;
    std::unique_ptr<Buffer> mips_, indirection_;
};

} // namespace metallic::render
