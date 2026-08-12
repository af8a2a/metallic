#pragma once

#include "Runtime/Render/GAPI/Rhi.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace metallic::render {

inline constexpr uint32_t kImportancePdfMaxMipCount = 16;

struct ImportancePdfSize {
    uint32_t width = 1;
    uint32_t height = 1;
    uint32_t mipCount = 1;
};

ImportancePdfSize computeImportancePdfTextureSize(uint32_t maxItems);

class ImportancePdfTexture final {
public:
    ImportancePdfTexture();
    ~ImportancePdfTexture();

    ImportancePdfTexture(ImportancePdfTexture&&) noexcept;
    ImportancePdfTexture& operator=(ImportancePdfTexture&&) noexcept;

    ImportancePdfTexture(const ImportancePdfTexture&) = delete;
    ImportancePdfTexture& operator=(const ImportancePdfTexture&) = delete;

    Result initialize(
        Device& device,
        uint32_t sourceWidth,
        uint32_t sourceHeight,
        std::string_view debugName,
        std::string& log);
    void clear();

    void beginGpuBuild(CommandBuffer& commandBuffer);
    void synchronizeGpuBuild(CommandBuffer& commandBuffer);
    void endGpuBuild(CommandBuffer& commandBuffer);

    bool valid() const;
    TextureView* view() const;
    TextureView* const* mipViews() const;
    uint32_t mipViewCount() const;
    uint32_t sourceWidth() const;
    uint32_t sourceHeight() const;
    uint32_t textureWidth() const;
    uint32_t textureHeight() const;
    uint32_t mipCount() const;
    uint64_t byteSize() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class ImportancePdfCompute final {
public:
    ImportancePdfCompute();
    ~ImportancePdfCompute();

    ImportancePdfCompute(ImportancePdfCompute&&) noexcept;
    ImportancePdfCompute& operator=(ImportancePdfCompute&&) noexcept;

    ImportancePdfCompute(const ImportancePdfCompute&) = delete;
    ImportancePdfCompute& operator=(const ImportancePdfCompute&) = delete;

    Result initialize(Device& device, std::string& log);
    Result buildLocalLights(
        CommandBuffer& commandBuffer,
        TextureView& environmentMap,
        ImportancePdfTexture& localLightPdf,
        uint32_t lightCount,
        float localLightIntensity,
        float sceneRadius);
    Result buildEnvironment(
        CommandBuffer& commandBuffer,
        TextureView& environmentMap,
        ImportancePdfTexture& environmentPdf);
    void clear();
    bool valid() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metallic::render
