#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <vulkan/vulkan.h>

#ifdef _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#include <shobjidl.h>
#endif

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "camera.h"
#include "cluster_streaming_service.h"
#include "frame_context.h"
#include "frame_graph.h"
#include "nsight_markers.h"
#include "input.h"
#include "imgui.h"
#include "imgui_internal.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"
#include "pipeline_asset.h"
#include "pipeline_builder.h"
#include "render_pass.h"
#include "render_uniforms.h"
#include "rhi_backend.h"
#include "bindless_scene_constants.h"
#include "raytraced_shadows.h"
#include "rhi_resource_utils.h"
#include "rhi_shader_utils.h"
#include "shader_manager.h"
#include "slang_compiler.h"
#include "visibility_constants.h"
#include "scene_context.h"
#include "scene_graph_ui.h"
#include "cluster_lod_builder.h"
#include "vulkan_backend.h"
#include "vulkan_frame_graph.h"
#include "vulkan_descriptor_buffer.h"
#include "vulkan_transient_allocator.h"
#include "vulkan_upload_service.h"
#include "streamline_context.h"

#if defined(METALLIC_HAS_AFTERMATH) && METALLIC_HAS_AFTERMATH
#include "aftermath_tracker.h"
#endif

#include <spdlog/spdlog.h>
#include <tracy/Tracy.hpp>

namespace {

#ifdef _WIN32
std::string openGltfFileDialog(GLFWwindow* window) {
    std::string result;
    HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
    if (FAILED(hr)) return result;

    IFileOpenDialog* pDialog = nullptr;
    hr = CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_ALL,
                          IID_IFileOpenDialog, reinterpret_cast<void**>(&pDialog));
    if (SUCCEEDED(hr)) {
        COMDLG_FILTERSPEC filters[] = {
            { L"glTF Files", L"*.gltf;*.glb" },
            { L"All Files",  L"*.*" },
        };
        pDialog->SetFileTypes(2, filters);
        pDialog->SetTitle(L"Open glTF Scene");

        HWND hwnd = glfwGetWin32Window(window);
        hr = pDialog->Show(hwnd);
        if (SUCCEEDED(hr)) {
            IShellItem* pItem = nullptr;
            hr = pDialog->GetResult(&pItem);
            if (SUCCEEDED(hr)) {
                PWSTR filePath = nullptr;
                hr = pItem->GetDisplayName(SIGDN_FILESYSPATH, &filePath);
                if (SUCCEEDED(hr) && filePath) {
                    int len = WideCharToMultiByte(CP_UTF8, 0, filePath, -1, nullptr, 0, nullptr, nullptr);
                    if (len > 0) {
                        result.resize(len - 1);
                        WideCharToMultiByte(CP_UTF8, 0, filePath, -1, result.data(), len, nullptr, nullptr);
                    }
                    CoTaskMemFree(filePath);
                }
                pItem->Release();
            }
        }
        pDialog->Release();
    }
    CoUninitialize();
    return result;
}
#endif

struct Vertex {
    float position[3];
    float color[3];
};

struct AppState {
    InputState input;
    bool framebufferResized = false;
};

static_assert(std::is_standard_layout_v<AppState>);
static_assert(offsetof(AppState, input) == 0);

static constexpr int kRenderResolutionBaseWidth = 1920;
static constexpr int kRenderResolutionBaseHeight = 1080;
static constexpr float kMinRenderScale = 0.25f;
static constexpr float kMaxRenderScale = 2.0f;

std::string formatByteCountShort(uint64_t byteCount) {
    static constexpr const char* kUnits[] = {"B", "KB", "MB", "GB"};
    static constexpr uint32_t kUnitCount = sizeof(kUnits) / sizeof(kUnits[0]);
    double value = static_cast<double>(byteCount);
    uint32_t unitIndex = 0u;
    while (value >= 1024.0 && unitIndex + 1u < kUnitCount) {
        value /= 1024.0;
        ++unitIndex;
    }

    char buffer[64] = {};
    if (unitIndex == 0u) {
        std::snprintf(buffer, sizeof(buffer), "%llu %s",
                      static_cast<unsigned long long>(byteCount),
                      kUnits[unitIndex]);
    } else {
        std::snprintf(buffer, sizeof(buffer), "%.1f %s", value, kUnits[unitIndex]);
    }
    return buffer;
}

struct StreamingDashboardHistory {
    static constexpr int kSampleCount = 120;

    std::array<float, kSampleCount> loadRequestHistory = {};
    std::array<float, kSampleCount> unloadRequestHistory = {};
    uint32_t lastFrameIndex = UINT32_MAX;
    int sampleCount = 0;
    int nextSample = 0;

    void push(uint32_t frameIndex, const ClusterStreamingService::StreamingStats& stats) {
        if (lastFrameIndex == frameIndex) {
            return;
        }

        lastFrameIndex = frameIndex;
        loadRequestHistory[nextSample] = static_cast<float>(stats.loadRequestsThisFrame);
        unloadRequestHistory[nextSample] = static_cast<float>(stats.unloadRequestsThisFrame);
        nextSample = (nextSample + 1) % kSampleCount;
        sampleCount = std::min(sampleCount + 1, kSampleCount);
    }

    int plotOffset() const {
        return sampleCount < kSampleCount ? 0 : nextSample;
    }

    float maxRequestValue() const {
        float maxValue = 1.0f;
        for (int sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex) {
            maxValue = std::max(maxValue, loadRequestHistory[sampleIndex]);
            maxValue = std::max(maxValue, unloadRequestHistory[sampleIndex]);
        }
        return maxValue;
    }
};

void checkImGuiVkResult(VkResult result) {
    if (result == VK_SUCCESS) {
        return;
    }
    spdlog::error("ImGui Vulkan error: {}", static_cast<int>(result));
}

template <typename VkHandle>
VkHandle nativeToVkHandle(void* handle) {
    if constexpr (std::is_pointer_v<VkHandle>) {
        return reinterpret_cast<VkHandle>(handle);
    } else {
        return static_cast<VkHandle>(reinterpret_cast<uint64_t>(handle));
    }
}

std::string formatVulkanVersion(uint32_t version) {
    return std::to_string(VK_API_VERSION_MAJOR(version)) + "." +
           std::to_string(VK_API_VERSION_MINOR(version)) + "." +
           std::to_string(VK_API_VERSION_PATCH(version));
}

struct AtmosphereTextures {
    RhiTextureHandle transmittance;
    RhiTextureHandle scattering;
    RhiTextureHandle irradiance;

    bool isValid() const {
        return transmittance.nativeHandle() &&
               scattering.nativeHandle() &&
               irradiance.nativeHandle();
    }

    void release() {
        rhiReleaseHandle(transmittance);
        rhiReleaseHandle(scattering);
        rhiReleaseHandle(irradiance);
    }
};

void eraseResourceById(PipelineAsset& asset, const std::string& resourceId) {
    asset.removeEdgesForResource(resourceId);
    asset.resources.erase(
        std::remove_if(asset.resources.begin(),
                       asset.resources.end(),
                       [&resourceId](const ResourceDecl& resource) {
                           return resource.id == resourceId;
                       }),
        asset.resources.end());
}

void erasePassByType(PipelineAsset& asset, const char* passType) {
    std::vector<std::string> passIds;
    for (const auto& pass : asset.passes) {
        if (pass.type == passType) {
            passIds.push_back(pass.id);
        }
    }

    for (const auto& passId : passIds) {
        asset.removeEdgesForPass(passId);
    }

    asset.passes.erase(
        std::remove_if(asset.passes.begin(),
                       asset.passes.end(),
                       [passType](const PassDecl& pass) {
                           return pass.type == passType;
                       }),
        asset.passes.end());
}

ResourceDecl* findFirstResourceByKind(PipelineAsset& asset, const char* resourceKind) {
    for (auto& resource : asset.resources) {
        if (resource.kind == resourceKind) {
            return &resource;
        }
    }
    return nullptr;
}

const ResourceDecl* findFirstResourceByKind(const PipelineAsset& asset, const char* resourceKind) {
    for (const auto& resource : asset.resources) {
        if (resource.kind == resourceKind) {
            return &resource;
        }
    }
    return nullptr;
}

EdgeDecl* findPassSlotBinding(PipelineAsset& asset,
                              const PassDecl* pass,
                              const char* direction,
                              const char* slotKey) {
    if (!pass) {
        return nullptr;
    }
    return asset.findEdge(pass->id, direction, slotKey);
}

const EdgeDecl* findPassSlotBinding(const PipelineAsset& asset,
                                    const PassDecl* pass,
                                    const char* direction,
                                    const char* slotKey) {
    if (!pass) {
        return nullptr;
    }
    return asset.findEdge(pass->id, direction, slotKey);
}

const ResourceDecl* findPassBoundResource(const PipelineAsset& asset,
                                          const PassDecl* pass,
                                          const char* direction,
                                          const char* slotKey) {
    const EdgeDecl* edge = findPassSlotBinding(asset, pass, direction, slotKey);
    return edge ? asset.findResourceById(edge->resourceId) : nullptr;
}

void ensurePassSlotBinding(PipelineAsset& asset,
                           const PassDecl& pass,
                           const char* direction,
                           const char* slotKey,
                           const std::string& resourceId) {
    if (EdgeDecl* edge = asset.findEdge(pass.id, direction, slotKey)) {
        edge->resourceId = resourceId;
        return;
    }

    asset.edges.push_back({
        generatePipelineAssetGuid(),
        pass.id,
        slotKey,
        direction,
        resourceId,
    });
}

void erasePassSlotBinding(PipelineAsset& asset,
                          const PassDecl& pass,
                          const char* direction,
                          const char* slotKey) {
    asset.edges.erase(
        std::remove_if(asset.edges.begin(),
                       asset.edges.end(),
                       [&](const EdgeDecl& edge) {
                           return edge.passId == pass.id &&
                                  edge.direction == direction &&
                                  edge.slotKey == slotKey;
                       }),
        asset.edges.end());
}

bool passProducesResource(const PipelineAsset& asset,
                          const PassDecl& pass,
                          const std::string& resourceId) {
    if (resourceId.empty()) {
        return false;
    }

    for (const auto& edge : asset.edges) {
        if (edge.passId == pass.id &&
            edge.direction == "output" &&
            edge.resourceId == resourceId) {
            return true;
        }
    }
    return false;
}

PassDecl* findFirstEnabledPassByType(PipelineAsset& asset, const char* passType) {
    for (auto& pass : asset.passes) {
        if (pass.enabled && pass.type == passType) {
            return &pass;
        }
    }
    return nullptr;
}

std::string ensureTonemapOutputResource(PipelineAsset& asset,
                                        const std::string& tonemapPassId,
                                        const std::string& backbufferId) {
    if (PassDecl* outputPass = findFirstEnabledPassByType(asset, "OutputPass")) {
        if (const ResourceDecl* outputSource =
                findPassBoundResource(asset, outputPass, "input", "source")) {
            if (outputSource->id != backbufferId && outputSource->kind != "backbuffer") {
                if (PassDecl* tonemapPass = asset.findPassById(tonemapPassId)) {
                    ensurePassSlotBinding(asset, *tonemapPass, "output", "output", outputSource->id);
                }
                return outputSource->id;
            }
        }
    }

    if (const PassDecl* tonemapPass = asset.findPassById(tonemapPassId)) {
        if (const ResourceDecl* tonemapTarget =
                findPassBoundResource(asset, tonemapPass, "output", "output")) {
            if (tonemapTarget->id != backbufferId && tonemapTarget->kind != "backbuffer") {
                return tonemapTarget->id;
            }
        }
    }

    const PassDecl* tonemapPass = asset.findPassById(tonemapPassId);
    if (!tonemapPass) {
        return {};
    }

    ResourceDecl tonemapOutput;
    tonemapOutput.id = generatePipelineAssetGuid();
    tonemapOutput.name = "Tonemap Output";
    tonemapOutput.kind = "transient";
    tonemapOutput.type = "texture";
    tonemapOutput.format = "RGBA8Srgb";
    tonemapOutput.size = "screen";
    tonemapOutput.editorPos = {
        tonemapPass->editorPos[0] + 160.0f,
        tonemapPass->editorPos[1] + 170.0f,
    };

    const std::string tonemapOutputId = tonemapOutput.id;
    asset.resources.push_back(std::move(tonemapOutput));
    if (PassDecl* tonemapPassMutable = asset.findPassById(tonemapPassId)) {
        ensurePassSlotBinding(asset, *tonemapPassMutable, "output", "output", tonemapOutputId);
    }
    return tonemapOutputId;
}

PassDecl* ensureDisplayOutputPass(PipelineAsset& asset,
                                  const std::string& sourceResourceId,
                                  const std::string& backbufferId) {
    if (PassDecl* outputPass = findFirstEnabledPassByType(asset, "OutputPass")) {
        if (!sourceResourceId.empty()) {
            const ResourceDecl* outputSource =
                findPassBoundResource(asset, outputPass, "input", "source");
            if (!outputSource || outputSource->id == backbufferId || outputSource->kind == "backbuffer") {
                ensurePassSlotBinding(asset, *outputPass, "input", "source", sourceResourceId);
            }
        }
        ensurePassSlotBinding(asset, *outputPass, "output", "target", backbufferId);
        return outputPass;
    }

    PassDecl outputPass;
    outputPass.id = generatePipelineAssetGuid();
    outputPass.name = "Output";
    outputPass.type = "OutputPass";
    outputPass.enabled = true;
    outputPass.sideEffect = false;
    outputPass.config = nlohmann::json::object();

    if (const PassDecl* tonemapPass = findFirstEnabledPassByType(asset, "TonemapPass")) {
        outputPass.editorPos = {
            tonemapPass->editorPos[0] + 320.0f,
            tonemapPass->editorPos[1],
        };
    }

    auto insertIt = std::find_if(asset.passes.begin(),
                                 asset.passes.end(),
                                 [](const PassDecl& pass) {
                                     return pass.enabled && pass.type == "ImGuiOverlayPass";
                                 });
    auto insertedIt = asset.passes.insert(insertIt, std::move(outputPass));
    if (!sourceResourceId.empty()) {
        ensurePassSlotBinding(asset, *insertedIt, "input", "source", sourceResourceId);
    }
    ensurePassSlotBinding(asset, *insertedIt, "output", "target", backbufferId);
    return &*insertedIt;
}

void normalizePipelineFinalDisplayOutput(PipelineAsset& asset) {
    const ResourceDecl* backbuffer = findFirstResourceByKind(asset, "backbuffer");
    if (!backbuffer) {
        return;
    }

    const std::string backbufferId = backbuffer->id;
    bool reroutedTonemapBackbufferWrite = false;
    bool createdOutputPass = false;

    if (PassDecl* tonemapPass = findFirstEnabledPassByType(asset, "TonemapPass")) {
        const bool hadOutputPass = findFirstEnabledPassByType(asset, "OutputPass") != nullptr;
        const ResourceDecl* tonemapTarget =
            findPassBoundResource(asset, tonemapPass, "output", "output");
        const bool tonemapWritesBackbuffer =
            tonemapTarget && tonemapTarget->id == backbufferId;

        std::string tonemapOutputId;
        if (!tonemapTarget || tonemapWritesBackbuffer || tonemapTarget->kind == "backbuffer") {
            tonemapOutputId = ensureTonemapOutputResource(asset, tonemapPass->id, backbufferId);
            reroutedTonemapBackbufferWrite = tonemapWritesBackbuffer;
        } else {
            tonemapOutputId = tonemapTarget->id;
        }

        if (!tonemapOutputId.empty()) {
            ensureDisplayOutputPass(asset, tonemapOutputId, backbufferId);
            createdOutputPass = !hadOutputPass && findFirstEnabledPassByType(asset, "OutputPass") != nullptr;
        }
    } else if (PassDecl* outputPass = findFirstEnabledPassByType(asset, "OutputPass")) {
        ensurePassSlotBinding(asset, *outputPass, "output", "target", backbufferId);
    }

    if (reroutedTonemapBackbufferWrite) {
        spdlog::warn("Pipeline '{}' bound TonemapPass directly to the backbuffer; restoring Tonemap -> OutputPass display chain for Vulkan",
                     asset.name);
    } else if (createdOutputPass) {
        spdlog::warn("Pipeline '{}' had TonemapPass without OutputPass; inserted OutputPass to present to the backbuffer",
                     asset.name);
    }

    if (PassDecl* imguiPass = findFirstEnabledPassByType(asset, "ImGuiOverlayPass")) {
        ensurePassSlotBinding(asset, *imguiPass, "output", "target", backbufferId);
    }
}

void disablePipelineAutoExposure(PipelineAsset& asset) {
    std::vector<std::string> producedResourceIds;
    for (const auto& pass : asset.passes) {
        if (pass.type != "AutoExposurePass") {
            continue;
        }
        if (const EdgeDecl* edge = findPassSlotBinding(asset, &pass, "output", "exposureLut")) {
            producedResourceIds.push_back(edge->resourceId);
        }
    }

    erasePassByType(asset, "AutoExposurePass");

    for (auto& pass : asset.passes) {
        if (pass.type != "TonemapPass") {
            continue;
        }

        erasePassSlotBinding(asset, pass, "input", "exposureLut");
        pass.config["autoExposure"] = false;
    }

    for (const auto& resourceId : producedResourceIds) {
        eraseResourceById(asset, resourceId);
    }
}

void disableVisibilityRayTracing(PipelineAsset& asset) {
    std::vector<std::string> producedResourceIds;
    for (const auto& pass : asset.passes) {
        if (pass.type != "ShadowRayPass") {
            continue;
        }
        if (const EdgeDecl* edge = findPassSlotBinding(asset, &pass, "output", "shadowMap")) {
            producedResourceIds.push_back(edge->resourceId);
        }
    }

    erasePassByType(asset, "ShadowRayPass");

    for (auto& pass : asset.passes) {
        if (pass.type != "DeferredLightingPass") {
            continue;
        }

        erasePassSlotBinding(asset, pass, "input", "shadowMap");
    }

    for (const auto& resourceId : producedResourceIds) {
        eraseResourceById(asset, resourceId);
    }
}

enum class VisibilityUpscalerMode {
    None,
    TAA,
    DLSS,
};

struct VisibilityUpscalerSelection {
    VisibilityUpscalerMode activeMode = VisibilityUpscalerMode::None;
    bool hasTaaPass = false;
    bool hasDlssPass = false;
    std::string activePostSource;
    std::string diagnostic;
};

const PassDecl* findFirstEnabledPassByType(const PipelineAsset& asset, const char* passType) {
    for (const auto& pass : asset.passes) {
        if (pass.enabled && pass.type == passType) {
            return &pass;
        }
    }
    return nullptr;
}

const PassDecl* findEnabledProducerForResource(const PipelineAsset& asset, const std::string& resourceName) {
    if (resourceName.empty()) {
        return nullptr;
    }
    for (const auto& pass : asset.passes) {
        if (!pass.enabled) {
            continue;
        }

        if (passProducesResource(asset, pass, resourceName)) {
            return &pass;
        }
    }
    return nullptr;
}

VisibilityUpscalerSelection analyzeVisibilityUpscalerSelection(const PipelineAsset& asset) {
    VisibilityUpscalerSelection selection;

    const PassDecl* tonemapPass = findFirstEnabledPassByType(asset, "TonemapPass");
    const PassDecl* outputPass = findFirstEnabledPassByType(asset, "OutputPass");
    const PassDecl* autoExposurePass = findFirstEnabledPassByType(asset, "AutoExposurePass");

    selection.hasTaaPass = findFirstEnabledPassByType(asset, "TAAPass") != nullptr;
    selection.hasDlssPass = findFirstEnabledPassByType(asset, "StreamlineDlssPass") != nullptr;

    const ResourceDecl* tonemapSource =
        findPassBoundResource(asset, tonemapPass, "input", "source");
    const ResourceDecl* outputSource =
        findPassBoundResource(asset, outputPass, "input", "source");
    const ResourceDecl* autoExposureSource =
        findPassBoundResource(asset, autoExposurePass, "input", "source");

    const ResourceDecl* activePostSource = tonemapSource ? tonemapSource : outputSource;
    selection.activePostSource = activePostSource ? activePostSource->name : std::string{};

    const PassDecl* activePostProducer =
        activePostSource ? findEnabledProducerForResource(asset, activePostSource->id) : nullptr;
    const PassDecl* autoExposureProducer =
        autoExposureSource ? findEnabledProducerForResource(asset, autoExposureSource->id) : nullptr;

    if (activePostProducer) {
        if (activePostProducer->type == "TAAPass") {
            selection.activeMode = VisibilityUpscalerMode::TAA;
        } else if (activePostProducer->type == "StreamlineDlssPass") {
            selection.activeMode = VisibilityUpscalerMode::DLSS;
        }
    }

    if (selection.activeMode != VisibilityUpscalerMode::None) {
        return selection;
    }

    if (tonemapPass && !tonemapSource) {
        if (autoExposureProducer && autoExposureProducer->type == "StreamlineDlssPass") {
            selection.diagnostic =
                "TonemapPass.source is unbound. AutoExposurePass already reads "
                "StreamlineDlssPass.dlssOutput; connect TonemapPass.source to the same output "
                "and keep TonemapPass.exposureLut as the second input.";
        } else if (autoExposureProducer && autoExposureProducer->type == "TAAPass") {
            selection.diagnostic =
                "TonemapPass.source is unbound. AutoExposurePass already reads "
                "TAAPass.taaOutput; connect TonemapPass.source to the same output and keep "
                "TonemapPass.exposureLut as the second input.";
        } else {
            selection.diagnostic =
                "TonemapPass.source is unbound. Connect it to DeferredLightingPass.lightingOutput, "
                "TAAPass.taaOutput, or StreamlineDlssPass.dlssOutput.";
        }
        return selection;
    }

    if (!tonemapPass && outputPass && !outputSource) {
        selection.diagnostic =
            "OutputPass.source is unbound. Connect it to DeferredLightingPass.lightingOutput, "
            "TAAPass.taaOutput, or StreamlineDlssPass.dlssOutput.";
        return selection;
    }

    if (selection.hasDlssPass) {
        if (autoExposureProducer && autoExposureProducer->type == "StreamlineDlssPass") {
            selection.diagnostic =
                "StreamlineDlssPass exists and AutoExposurePass.source already reads "
                "StreamlineDlssPass.dlssOutput, but the active post chain does not.";
        } else if (!selection.activePostSource.empty()) {
            selection.diagnostic =
                "StreamlineDlssPass exists, but the active post chain still uses '" +
                selection.activePostSource + "'.";
        } else {
            selection.diagnostic =
                "StreamlineDlssPass is present, but TonemapPass.source/OutputPass.source is not "
                "bound to StreamlineDlssPass.dlssOutput.";
        }
        return selection;
    }

    if (selection.hasTaaPass) {
        if (autoExposureProducer && autoExposureProducer->type == "TAAPass") {
            selection.diagnostic =
                "TAAPass exists and AutoExposurePass.source already reads TAAPass.taaOutput, "
                "but the active post chain does not.";
        } else if (!selection.activePostSource.empty()) {
            selection.diagnostic =
                "TAAPass exists, but the active post chain still uses '" +
                selection.activePostSource + "'.";
        } else {
            selection.diagnostic =
                "TAAPass is present, but TonemapPass.source/OutputPass.source is not bound to "
                "TAAPass.taaOutput.";
        }
    }

    return selection;
}

VisibilityUpscalerMode detectVisibilityUpscalerMode(const PipelineAsset& asset) {
    return analyzeVisibilityUpscalerSelection(asset).activeMode;
}

const char* visibilityUpscalerModeName(VisibilityUpscalerMode mode) {
    switch (mode) {
    case VisibilityUpscalerMode::TAA:
        return "TAA";
    case VisibilityUpscalerMode::DLSS:
        return "DLSS";
    default:
        return "None";
    }
}

void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
    auto* state = static_cast<AppState*>(glfwGetWindowUserPointer(window));
    if (state && width > 0 && height > 0) {
        state->framebufferResized = true;
    }
}

std::vector<float> loadFloatData(const std::string& path, size_t expectedCount) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        spdlog::warn("Atmosphere: missing texture data {}", path);
        return {};
    }

    file.seekg(0, std::ios::end);
    const size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    if (size == 0 || size % sizeof(float) != 0) {
        spdlog::warn("Atmosphere: invalid data size {} ({} bytes)", path, size);
        return {};
    }

    std::vector<float> data(size / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size));
    if (!file) {
        spdlog::warn("Atmosphere: failed to read {}", path);
        return {};
    }

    if (expectedCount > 0 && data.size() != expectedCount) {
        spdlog::warn("Atmosphere: unexpected element count in {} ({} vs {})",
                     path,
                     data.size(),
                     expectedCount);
    }

    return data;
}

bool loadPipelineAssetChecked(const std::string& path,
                              const char* label,
                              PipelineAsset& outAsset) {
    PipelineAsset loaded = PipelineAsset::load(path);
    if (loaded.name.empty()) {
        spdlog::error("Failed to load {} pipeline from '{}'", label, path);
        return false;
    }

    std::string validationError;
    if (!loaded.validate(validationError)) {
        spdlog::error("Invalid {} pipeline '{}': {}", label, path, validationError);
        return false;
    }

    outAsset = std::move(loaded);
    spdlog::info("Loaded {} pipeline: {} ({} passes, {} resources)",
                 label,
                 outAsset.name,
                 outAsset.passes.size(),
                 outAsset.resources.size());
    return true;
}

bool loadAtmosphereTextures(const RhiDevice& device,
                            const char* projectRoot,
                            AtmosphereTextures& outTextures) {
    constexpr uint32_t kTransmittanceWidth = 256;
    constexpr uint32_t kTransmittanceHeight = 64;
    constexpr uint32_t kScatteringWidth = 256;
    constexpr uint32_t kScatteringHeight = 128;
    constexpr uint32_t kScatteringDepth = 32;
    constexpr uint32_t kIrradianceWidth = 64;
    constexpr uint32_t kIrradianceHeight = 16;

    const std::string basePath = std::string(projectRoot) + "/Asset/Atmosphere/";
    const auto transmittance = loadFloatData(basePath + "transmittance.dat",
                                             static_cast<size_t>(kTransmittanceWidth) *
                                                 kTransmittanceHeight * 4);
    const auto scattering = loadFloatData(basePath + "scattering.dat",
                                          static_cast<size_t>(kScatteringWidth) *
                                              kScatteringHeight * kScatteringDepth * 4);
    const auto irradiance = loadFloatData(basePath + "irradiance.dat",
                                          static_cast<size_t>(kIrradianceWidth) *
                                              kIrradianceHeight * 4);

    if (transmittance.empty() || scattering.empty() || irradiance.empty()) {
        return false;
    }

    outTextures.release();
    outTextures.transmittance = rhiCreateTexture2D(device,
                                                   kTransmittanceWidth,
                                                   kTransmittanceHeight,
                                                   RhiFormat::RGBA32Float,
                                                   false,
                                                   1,
                                                   RhiTextureStorageMode::Shared,
                                                   RhiTextureUsage::ShaderRead);
    outTextures.scattering = rhiCreateTexture3D(device,
                                                kScatteringWidth,
                                                kScatteringHeight,
                                                kScatteringDepth,
                                                RhiFormat::RGBA32Float,
                                                RhiTextureStorageMode::Shared,
                                                RhiTextureUsage::ShaderRead);
    outTextures.irradiance = rhiCreateTexture2D(device,
                                                kIrradianceWidth,
                                                kIrradianceHeight,
                                                RhiFormat::RGBA32Float,
                                                false,
                                                1,
                                                RhiTextureStorageMode::Shared,
                                                RhiTextureUsage::ShaderRead);

    if (!outTextures.isValid()) {
        outTextures.release();
        return false;
    }

    rhiUploadTexture2D(outTextures.transmittance,
                       kTransmittanceWidth,
                       kTransmittanceHeight,
                       transmittance.data(),
                       static_cast<size_t>(kTransmittanceWidth) * 4 * sizeof(float));
    rhiUploadTexture3D(outTextures.scattering,
                       kScatteringWidth,
                       kScatteringHeight,
                       kScatteringDepth,
                       scattering.data(),
                       static_cast<size_t>(kScatteringWidth) * 4 * sizeof(float),
                       static_cast<size_t>(kScatteringWidth) * kScatteringHeight * 4 * sizeof(float));
    rhiUploadTexture2D(outTextures.irradiance,
                       kIrradianceWidth,
                       kIrradianceHeight,
                       irradiance.data(),
                       static_cast<size_t>(kIrradianceWidth) * 4 * sizeof(float));
    return true;
}

PipelineAsset makeSceneColorPostPipelineAsset() {
    PipelineAsset asset;
    asset.schemaVersion = kPipelineAssetSchemaVersion;
    asset.name = "VulkanPostPipeline";

    ResourceDecl sceneColor;
    sceneColor.id = generatePipelineAssetGuid();
    sceneColor.name = "Scene Color";
    sceneColor.kind = "imported";
    sceneColor.type = "texture";
    sceneColor.format = "RGBA16Float";
    sceneColor.size = "screen";
    sceneColor.importKey = "sceneColor";

    ResourceDecl tonemapOutput;
    tonemapOutput.id = generatePipelineAssetGuid();
    tonemapOutput.name = "Tonemap Output";
    tonemapOutput.kind = "transient";
    tonemapOutput.type = "texture";
    tonemapOutput.format = "RGBA8Srgb";
    tonemapOutput.size = "screen";

    ResourceDecl backbuffer;
    backbuffer.id = generatePipelineAssetGuid();
    backbuffer.name = "Backbuffer";
    backbuffer.kind = "backbuffer";
    backbuffer.type = "texture";
    backbuffer.format = "BGRA8Unorm";
    backbuffer.size = "screen";

    asset.resources.push_back(sceneColor);
    asset.resources.push_back(tonemapOutput);
    asset.resources.push_back(backbuffer);

    PassDecl tonemapPass;
    tonemapPass.id = generatePipelineAssetGuid();
    tonemapPass.name = "Tonemap";
    tonemapPass.type = "TonemapPass";
    tonemapPass.enabled = true;
    tonemapPass.sideEffect = false;
    tonemapPass.config = {
        {"method", "Clip"},
        {"exposure", 1.0},
        {"contrast", 1.0},
        {"brightness", 1.0},
        {"saturation", 1.0},
        {"vignette", 0.0},
        {"dither", false},
        {"autoExposure", false},
    };

    PassDecl outputPass;
    outputPass.id = generatePipelineAssetGuid();
    outputPass.name = "Output";
    outputPass.type = "OutputPass";
    outputPass.enabled = true;
    outputPass.sideEffect = false;
    outputPass.config = nlohmann::json::object();

    PassDecl imguiOverlayPass;
    imguiOverlayPass.id = generatePipelineAssetGuid();
    imguiOverlayPass.name = "ImGui Overlay";
    imguiOverlayPass.type = "ImGuiOverlayPass";
    imguiOverlayPass.enabled = true;
    imguiOverlayPass.sideEffect = false;
    imguiOverlayPass.config = nlohmann::json::object();

    asset.passes.push_back(tonemapPass);
    asset.passes.push_back(outputPass);
    asset.passes.push_back(imguiOverlayPass);

    asset.edges.push_back({generatePipelineAssetGuid(), tonemapPass.id, "source", "input", sceneColor.id});
    asset.edges.push_back({generatePipelineAssetGuid(), tonemapPass.id, "output", "output", tonemapOutput.id});
    asset.edges.push_back({generatePipelineAssetGuid(), outputPass.id, "source", "input", tonemapOutput.id});
    asset.edges.push_back({generatePipelineAssetGuid(), outputPass.id, "target", "output", backbuffer.id});
    asset.edges.push_back({generatePipelineAssetGuid(), imguiOverlayPass.id, "target", "output", backbuffer.id});

    return asset;
}

float4 orbitCameraWorldPosition(const OrbitCamera& camera) {
    const float cosAzimuth = std::cos(camera.azimuth);
    const float sinAzimuth = std::sin(camera.azimuth);
    const float cosElevation = std::cos(camera.elevation);
    const float sinElevation = std::sin(camera.elevation);

    return float4(camera.target.x + camera.distance * cosElevation * sinAzimuth,
                  camera.target.y + camera.distance * sinElevation,
                  camera.target.z + camera.distance * cosElevation * cosAzimuth,
                  1.0f);
}

void setNvproStyle() {
    ImGui::StyleColorsDark();

    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 0.0f;
    style.WindowBorderSize = 0.0f;
    style.ColorButtonPosition = ImGuiDir_Right;
    style.FrameRounding = 2.0f;
    style.FrameBorderSize = 1.0f;
    style.GrabRounding = 4.0f;
    style.IndentSpacing = 12.0f;
    style.ScrollbarSize = 14.0f;
    style.WindowPadding = ImVec2(8, 8);
    style.FramePadding = ImVec2(4, 3);
    style.ItemSpacing = ImVec2(8, 4);
    style.ItemInnerSpacing = ImVec2(4, 4);

    ImVec4* c = style.Colors;
    ImVec4 bg(0.2f, 0.2f, 0.2f, 1.0f);
    ImVec4 bgDark(0.135f, 0.135f, 0.135f, 1.0f);
    ImVec4 frameBg(0.05f, 0.05f, 0.05f, 0.5f);
    ImVec4 border(0.4f, 0.4f, 0.4f, 0.5f);
    ImVec4 normal(0.465f, 0.465f, 0.525f, 1.0f);
    ImVec4 active(0.365f, 0.365f, 0.425f, 1.0f);
    ImVec4 hovered(0.565f, 0.565f, 0.625f, 1.0f);

    c[ImGuiCol_WindowBg] = bg;
    c[ImGuiCol_MenuBarBg] = bg;
    c[ImGuiCol_ScrollbarBg] = bg;
    c[ImGuiCol_PopupBg] = bgDark;
    c[ImGuiCol_ChildBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
    c[ImGuiCol_Border] = border;
    c[ImGuiCol_FrameBg] = frameBg;
    c[ImGuiCol_FrameBgHovered] = hovered;
    c[ImGuiCol_FrameBgActive] = normal;

    c[ImGuiCol_Header] = normal;
    c[ImGuiCol_HeaderHovered] = hovered;
    c[ImGuiCol_HeaderActive] = active;

    c[ImGuiCol_Button] = normal;
    c[ImGuiCol_ButtonHovered] = hovered;
    c[ImGuiCol_ButtonActive] = active;

    c[ImGuiCol_SliderGrab] = normal;
    c[ImGuiCol_SliderGrabActive] = active;

    c[ImGuiCol_CheckMark] = normal;
    c[ImGuiCol_TextSelectedBg] = normal;

    c[ImGuiCol_Separator] = normal;
    c[ImGuiCol_SeparatorHovered] = hovered;
    c[ImGuiCol_SeparatorActive] = active;

    c[ImGuiCol_ResizeGrip] = normal;
    c[ImGuiCol_ResizeGripHovered] = hovered;
    c[ImGuiCol_ResizeGripActive] = active;

    c[ImGuiCol_Tab] = ImVec4(0.05f, 0.05f, 0.05f, 0.5f);
    c[ImGuiCol_TabHovered] = ImVec4(0.465f, 0.495f, 0.525f, 1.0f);
    c[ImGuiCol_TabSelected] = ImVec4(0.282f, 0.290f, 0.302f, 1.0f);
    c[ImGuiCol_TabSelectedOverline] = normal;
    c[ImGuiCol_TabDimmed] = ImVec4(0.05f, 0.05f, 0.05f, 0.35f);
    c[ImGuiCol_TabDimmedSelected] = ImVec4(0.18f, 0.18f, 0.20f, 1.0f);
    c[ImGuiCol_TabDimmedSelectedOverline] = ImVec4(0.5f, 0.5f, 0.5f, 0.0f);

    c[ImGuiCol_TitleBg] = ImVec4(0.125f, 0.125f, 0.125f, 1.0f);
    c[ImGuiCol_TitleBgActive] = ImVec4(0.465f, 0.465f, 0.465f, 1.0f);
    c[ImGuiCol_TitleBgCollapsed] = ImVec4(0.125f, 0.125f, 0.125f, 0.5f);

    c[ImGuiCol_ModalWindowDimBg] = ImVec4(0.465f, 0.465f, 0.465f, 0.350f);

    c[ImGuiCol_DockingPreview] = ImVec4(0.465f, 0.465f, 0.525f, 0.7f);
    c[ImGuiCol_DockingEmptyBg] = ImVec4(0.1f, 0.1f, 0.1f, 1.0f);

    c[ImGuiCol_Text] = ImVec4(0.86f, 0.86f, 0.86f, 1.0f);
    c[ImGuiCol_TextDisabled] = ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
    c[ImGuiCol_NavCursor] = normal;

    ImGui::SetColorEditOptions(ImGuiColorEditFlags_Float | ImGuiColorEditFlags_PickerHueWheel);
}

static bool s_dockLayoutInitialized = false;

ImGuiID beginDockspace(bool& showSceneGraphWindow,
                       bool& showGraphDebug,
                       bool& showRenderPassUI,
                       bool& showImGuiDemo) {
    const ImGuiID dockspaceId =
        ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(), ImGuiDockNodeFlags_None);

    if (!s_dockLayoutInitialized) {
        s_dockLayoutInitialized = true;
        ImGui::DockBuilderRemoveNode(dockspaceId);
        ImGui::DockBuilderAddNode(dockspaceId, ImGuiDockNodeFlags_DockSpace);
        ImGui::DockBuilderSetNodeSize(dockspaceId, ImGui::GetMainViewport()->Size);

        ImGuiID centerId = dockspaceId;
        ImGuiID rightId = ImGui::DockBuilderSplitNode(centerId, ImGuiDir_Right, 0.25f, nullptr, &centerId);
        ImGuiID bottomId = ImGui::DockBuilderSplitNode(centerId, ImGuiDir_Down, 0.06f, nullptr, &centerId);
        ImGuiID inspectorId = ImGui::DockBuilderSplitNode(rightId, ImGuiDir_Down, 0.35f, nullptr, &rightId);

        ImGui::DockBuilderDockWindow("Viewport", centerId);
        ImGui::DockBuilderDockWindow("Render Passes", rightId);
        ImGui::DockBuilderDockWindow("Vulkan Sponza", rightId);
        ImGui::DockBuilderDockWindow("Scene Browser", rightId);
        ImGui::DockBuilderDockWindow("Inspector", inspectorId);
        ImGui::DockBuilderDockWindow("Scene Loader", bottomId);

        ImGui::DockBuilderFinish(dockspaceId);
    }

    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("Scene Browser", nullptr, &showSceneGraphWindow);
            ImGui::MenuItem("FrameGraph", nullptr, &showGraphDebug);
            ImGui::MenuItem("Render Passes", nullptr, &showRenderPassUI);
            ImGui::MenuItem("ImGui Demo", nullptr, &showImGuiDemo);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    return dockspaceId;
}

} // namespace

bool s_viewportHovered = false;

int main() {
    if (!glfwInit()) {
        spdlog::error("Failed to initialize GLFW");
        return 1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "Metallic - Vulkan Sponza", nullptr, nullptr);
    if (!window) {
        spdlog::error("Failed to create GLFW window");
        glfwTerminate();
        return 1;
    }

    OrbitCamera previewCamera;
    AppState appState{};
    appState.input.camera = &previewCamera;
    glfwSetWindowUserPointer(window, &appState);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    setupInputCallbacks(window, &appState.input);

    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window, &width, &height);

    RhiCreateInfo createInfo;
    createInfo.window = window;
    createInfo.width = static_cast<uint32_t>(width);
    createInfo.height = static_cast<uint32_t>(height);
    createInfo.applicationName = "Metallic";
    createInfo.enableValidation = true;
    createInfo.requireVulkan14 = true;
    // Enable timeline semaphores for async compute / transfer queue synchronisation (Vulkan 1.2 core).
    createInfo.enableTimelineSemaphore = true;
    // On-disk caches: PSO binaries and compiled SPIR-V modules.
    createInfo.pipelineCacheDir = "cache/pipelines";
    createInfo.shaderCacheDir   = "cache/shaders";
    setSlangShaderCacheDir(createInfo.shaderCacheDir);

    // --- Streamline / DLSS initialization (before RHI so SL can intercept device creation) ---
    StreamlineContext streamlineCtx;
    bool dlssAvailable = false;
    DlssPreset dlssPreset = DlssPreset::Balanced;
    uint32_t dlssRenderWidth = 0;
    uint32_t dlssRenderHeight = 0;
    float visibilityRenderScale = 1.0f;
    PipelineUiControls pipelineUiControls{};

#ifdef METALLIC_HAS_STREAMLINE
    if (streamlineCtx.init(PROJECT_SOURCE_DIR)) {
        createInfo.vkGetDeviceProcAddrProxy = streamlineCtx.vulkanDeviceProcAddrProxy();
        StreamlineVulkanRequirements slReqs;
        if (streamlineCtx.queryVulkanRequirements(slReqs)) {
            for (auto ext : slReqs.instanceExtensions) {
                createInfo.extraInstanceExtensions.push_back(ext);
            }
            for (auto ext : slReqs.deviceExtensions) {
                createInfo.extraDeviceExtensions.push_back(ext);
            }
            createInfo.enableTimelineSemaphore = slReqs.needsTimelineSemaphore;
        }
    }
#endif

#if defined(METALLIC_HAS_AFTERMATH) && METALLIC_HAS_AFTERMATH
    AftermathTracker::getInstance().initialize();
#endif

    std::string backendError;
    auto rhi = createRhiContext(RhiBackendType::Vulkan, createInfo, backendError);
    if (!rhi) {
        spdlog::error("Failed to create Vulkan backend: {}", backendError);
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

#if defined(METALLIC_HAS_AFTERMATH) && METALLIC_HAS_AFTERMATH
    setSlangCompileCallback([](const char* /*sourcePath*/, const uint32_t* spirvData, size_t spirvSizeBytes) {
        std::span<const uint32_t> data(spirvData, spirvSizeBytes / sizeof(uint32_t));
        AftermathTracker::getInstance().addShaderBinary(data);
    });
#endif

    const auto triangleSpirv = compileSlangGraphicsBinary(RhiBackendType::Vulkan,
                                                          "Shaders/Vertex/triangle",
                                                          PROJECT_SOURCE_DIR);
    if (triangleSpirv.empty()) {
        spdlog::error("Failed to compile SPIR-V for triangle shader");
        rhi->waitIdle();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    const std::array<Vertex, 3> vertices = {{
        {{0.0f, -0.6f, 0.0f}, {1.0f, 0.3f, 0.2f}},
        {{0.6f, 0.6f, 0.0f}, {0.2f, 0.9f, 0.4f}},
        {{-0.6f, 0.6f, 0.0f}, {0.2f, 0.5f, 1.0f}},
    }};

    const RhiNativeHandles& native = rhi->nativeHandles();
    VkDevice vkDevice = nativeToVkHandle<VkDevice>(native.device);
    VkPhysicalDevice vkPhysicalDevice = nativeToVkHandle<VkPhysicalDevice>(native.physicalDevice);
    VmaAllocator vmaAllocator = getVulkanAllocator(*rhi);
    RhiDeviceHandle deviceHandle(native.device, rhi.get());
    RhiCommandQueueHandle queueHandle(native.queue);
    const uint32_t swapchainImageCount = std::max(2u, native.swapchainImageCount);
    const VkFormat swapchainFormat = static_cast<VkFormat>(native.colorFormat);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    setNvproStyle();
    ImGui_ImplGlfw_InitForOther(window, true);

    VkPipelineRenderingCreateInfoKHR imguiRenderingInfo{
        VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
    imguiRenderingInfo.colorAttachmentCount = 1;
    imguiRenderingInfo.pColorAttachmentFormats = &swapchainFormat;

    ImGui_ImplVulkan_InitInfo imguiInitInfo{};
    imguiInitInfo.ApiVersion = native.apiVersion;
    imguiInitInfo.Instance = nativeToVkHandle<VkInstance>(native.instance);
    imguiInitInfo.PhysicalDevice = vkPhysicalDevice;
    imguiInitInfo.Device = vkDevice;
    imguiInitInfo.QueueFamily = native.graphicsQueueFamily;
    imguiInitInfo.Queue = nativeToVkHandle<VkQueue>(native.queue);
    imguiInitInfo.DescriptorPool = nativeToVkHandle<VkDescriptorPool>(native.descriptorPool);
    imguiInitInfo.MinImageCount = swapchainImageCount;
    imguiInitInfo.ImageCount = swapchainImageCount;
    imguiInitInfo.UseDynamicRendering = true;
    imguiInitInfo.CheckVkResultFn = checkImGuiVkResult;
    imguiInitInfo.PipelineInfoMain.Subpass = 0;
    imguiInitInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    imguiInitInfo.PipelineInfoMain.PipelineRenderingCreateInfo = imguiRenderingInfo;

    if (!ImGui_ImplVulkan_Init(&imguiInitInfo)) {
        spdlog::error("Failed to initialize Dear ImGui for Vulkan");
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        rhi->waitIdle();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    const RhiFeatures& features = rhi->features();
    ShaderManagerProfile shaderProfile = ShaderManagerProfile::vulkanVisibility();
    ShaderCompileMode shaderCompileMode = ShaderCompileMode::Release;
    ShaderManager shaderManager(deviceHandle,
                                PROJECT_SOURCE_DIR,
                                features.meshShaders,
                                features.meshShaders,
                                shaderProfile,
                                shaderCompileMode);
    if (!shaderManager.buildAll()) {
        spdlog::error("Failed to build Vulkan visibility shader set");
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        rhi->waitIdle();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    PipelineRuntimeContext& runtimeContext = shaderManager.runtimeContext();
    ClusterStreamingService clusterStreamingService;
    runtimeContext.rhi = rhi.get();
    runtimeContext.clusterStreamingService = &clusterStreamingService;
    auto refreshClusterStreamingMemoryBudget = [&]() {
        const VulkanMemoryBudgetInfo vulkanMemoryBudget = getVulkanMemoryBudgetInfo(*rhi);
        ClusterStreamingService::MemoryBudgetInfo streamingMemoryBudget{};
        streamingMemoryBudget.available = vulkanMemoryBudget.available;
        streamingMemoryBudget.heapCount = vulkanMemoryBudget.heapCount;
        streamingMemoryBudget.totalBudgetBytes = vulkanMemoryBudget.totalBudgetBytes;
        streamingMemoryBudget.totalUsageBytes = vulkanMemoryBudget.totalUsageBytes;
        streamingMemoryBudget.totalHeadroomBytes = vulkanMemoryBudget.totalHeadroomBytes;
        streamingMemoryBudget.deviceLocalBudgetBytes = vulkanMemoryBudget.deviceLocalBudgetBytes;
        streamingMemoryBudget.deviceLocalUsageBytes = vulkanMemoryBudget.deviceLocalUsageBytes;
        streamingMemoryBudget.deviceLocalHeadroomBytes =
            vulkanMemoryBudget.deviceLocalHeadroomBytes;
        clusterStreamingService.applyAutoMemoryBudget(streamingMemoryBudget);

        const ClusterStreamingService::MemoryBudgetInfo& appliedMemoryBudget =
            clusterStreamingService.memoryBudgetInfo();
        if (appliedMemoryBudget.available) {
            spdlog::info(
                "Cluster streaming: device-local VRAM headroom {} (budget {}, usage {}), auto pool target {}",
                formatByteCountShort(appliedMemoryBudget.deviceLocalHeadroomBytes),
                formatByteCountShort(appliedMemoryBudget.deviceLocalBudgetBytes),
                formatByteCountShort(appliedMemoryBudget.deviceLocalUsageBytes),
                formatByteCountShort(appliedMemoryBudget.targetStorageBytes));
        } else {
            spdlog::info(
                "Cluster streaming: VK_EXT_memory_budget unavailable, keeping configured pool {}",
                formatByteCountShort(clusterStreamingService.streamingStorageCapacityBytes()));
        }
    };
    refreshClusterStreamingMemoryBudget();
    auto hasRenderPipeline = [&](const char* name) {
        auto it = runtimeContext.renderPipelinesRhi.find(name);
        return it != runtimeContext.renderPipelinesRhi.end() && it->second.nativeHandle() != nullptr;
    };
    auto hasComputePipeline = [&](const char* name) {
        auto it = runtimeContext.computePipelinesRhi.find(name);
        return it != runtimeContext.computePipelinesRhi.end() && it->second.nativeHandle() != nullptr;
    };
    auto importRuntimeTexture = [&](const char* name, const RhiTexture& texture) {
        shaderManager.importTexture(name, texture);
    };

    RhiBufferHandle vertexBuffer =
        rhiCreateSharedBuffer(deviceHandle, vertices.data(), sizeof(vertices), "TriangleVB");
    if (!vertexBuffer.nativeHandle()) {
        spdlog::error("Failed to create shared vertex buffer for Vulkan RenderGraph test");
        rhi->waitIdle();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    RhiVertexDescriptorHandle vertexDescriptor = rhiCreateVertexDescriptor();
    rhiVertexDescriptorSetAttribute(vertexDescriptor,
                                    0,
                                    RhiVertexFormat::Float3,
                                    static_cast<uint32_t>(offsetof(Vertex, position)),
                                    0);
    rhiVertexDescriptorSetAttribute(vertexDescriptor,
                                    1,
                                    RhiVertexFormat::Float3,
                                    static_cast<uint32_t>(offsetof(Vertex, color)),
                                    0);
    rhiVertexDescriptorSetLayout(vertexDescriptor, 0, sizeof(Vertex));

    std::string triangleShaderSource(reinterpret_cast<const char*>(triangleSpirv.data()),
                                     triangleSpirv.size() * sizeof(uint32_t));
    RhiRenderPipelineSourceDesc trianglePipelineDesc;
    trianglePipelineDesc.vertexEntry = "vertexMain";
    trianglePipelineDesc.fragmentEntry = "fragmentMain";
    trianglePipelineDesc.colorFormat = RhiFormat::RGBA16Float;
    trianglePipelineDesc.vertexDescriptor = &vertexDescriptor;

    std::string trianglePipelineError;
    RhiGraphicsPipelineHandle trianglePipeline =
        rhiCreateRenderPipelineFromSource(deviceHandle,
                                          triangleShaderSource,
                                          trianglePipelineDesc,
                                          trianglePipelineError);
    if (!trianglePipeline.nativeHandle()) {
        spdlog::error("Failed to create triangle pipeline: {}", trianglePipelineError);
        rhi->waitIdle();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    RhiSamplerDesc samplerDesc;
    samplerDesc.minFilter = RhiSamplerFilterMode::Linear;
    samplerDesc.magFilter = RhiSamplerFilterMode::Linear;
    samplerDesc.addressModeS = RhiSamplerAddressMode::ClampToEdge;
    samplerDesc.addressModeT = RhiSamplerAddressMode::ClampToEdge;
    samplerDesc.addressModeR = RhiSamplerAddressMode::ClampToEdge;
    RhiSamplerHandle linearSampler = rhiCreateSampler(deviceHandle, samplerDesc);
    if (!linearSampler.nativeHandle()) {
        spdlog::error("Failed to create Vulkan sampler");
        rhi->waitIdle();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
    shaderManager.importSampler("atmosphere", linearSampler);

    AtmosphereTextures atmosphereTextures;
    if (!loadAtmosphereTextures(deviceHandle, PROJECT_SOURCE_DIR, atmosphereTextures)) {
        spdlog::warn("Failed to load atmosphere textures; SkyPass will clear to black");
    } else {
        importRuntimeTexture("transmittance", atmosphereTextures.transmittance);
        importRuntimeTexture("scattering", atmosphereTextures.scattering);
        importRuntimeTexture("irradiance", atmosphereTextures.irradiance);
    }

    SceneContext sceneCtx(deviceHandle, queueHandle, PROJECT_SOURCE_DIR);
    const std::string defaultGltfPath = std::string(PROJECT_SOURCE_DIR) + "/Asset/Sponza/glTF/Sponza.gltf";
    bool previewSceneReady = sceneCtx.loadScene(defaultGltfPath);
    if (!previewSceneReady) {
        spdlog::warn("Failed to load Vulkan Sponza scene; falling back to triangle path");
    }

    RaytracedShadowResources shadowResources;
    bool rtShadowsAvailable = false;
    bool enableRTShadows = true;
    if (previewSceneReady && rhi->features().rayTracing) {
        if (buildAccelerationStructures(deviceHandle,
                                        queueHandle,
                                        sceneCtx.mesh(),
                                        sceneCtx.sceneGraph(),
                                        shadowResources) &&
            createShadowPipeline(deviceHandle, shadowResources, PROJECT_SOURCE_DIR)) {
            rtShadowsAvailable = true;
            spdlog::info("Vulkan raytraced shadows enabled");
        } else {
            spdlog::warn("Failed to initialize Vulkan raytraced shadows; continuing with fallback lighting");
            shadowResources.release();
        }
    } else if (!rhi->features().rayTracing) {
        spdlog::info("Vulkan ray tracing not supported on this device; shadow pass stays disabled");
    }

    // --- Streamline: set Vulkan device and query DLSS availability ---
#ifdef METALLIC_HAS_STREAMLINE
    if (streamlineCtx.isInitialized()) {
        VkInstance vkInstance = nativeToVkHandle<VkInstance>(native.instance);
        if (streamlineCtx.setVulkanDevice(vkInstance, vkPhysicalDevice, vkDevice,
                                           nativeToVkHandle<VkQueue>(native.queue),
                                            native.graphicsQueueFamily, 0)) {
            vulkanSetStreamlineCommandHooks(streamlineCtx.vulkanBeginCommandBufferHook(),
                                           streamlineCtx.vulkanCmdBindPipelineHook(),
                                           streamlineCtx.vulkanCmdBindDescriptorSetsHook());
            vulkanSetStreamlineHookedCommandsEnabled(true);
            dlssAvailable = streamlineCtx.isDlssAvailable();
        }
    }
#endif

    const std::string visibilityPipelinePath =
        std::string(PROJECT_SOURCE_DIR) + "/Pipelines/visibilitybuffer.json";
    const std::string clusterVisPipelinePath =
        std::string(PROJECT_SOURCE_DIR) + "/Pipelines/cluster_vis.json";
    PipelineAsset visibilityPipelineBaseAsset;
    PipelineAsset visibilityPipelineAsset;
    PipelineAsset clusterVisPipelineAsset;
    bool visibilityPipelineBaseLoaded =
        loadPipelineAssetChecked(visibilityPipelinePath, "Vulkan visibility", visibilityPipelineBaseAsset);
    bool clusterVisPipelineLoaded =
        loadPipelineAssetChecked(clusterVisPipelinePath, "Cluster visualization", clusterVisPipelineAsset);
    if (clusterVisPipelineLoaded) {
        erasePassByType(clusterVisPipelineAsset, "ImGuiOverlayPass");
    }
    bool useClusterVisMode = false;
    bool visibilityPipelineAssetLoaded = false;
    bool visibilityAutoExposureAvailable = false;
    bool visibilityTaaAvailable = false;
    bool visibilityGpuCullingAvailable = false;
    bool useVisibilityRenderGraph = false;
    bool atmosphereSkyAvailable = false;
    VisibilityUpscalerMode visibilityUpscalerMode = VisibilityUpscalerMode::None;
    VisibilityUpscalerSelection visibilityUpscalerSelection;
    bool dlssStateDirty = true;

    auto refreshVisibilityPipelineState = [&]() {
        visibilityPipelineAssetLoaded = visibilityPipelineBaseLoaded;
        visibilityPipelineAsset =
            visibilityPipelineBaseLoaded ? visibilityPipelineBaseAsset : PipelineAsset{};
        visibilityAutoExposureAvailable =
            hasComputePipeline("HistogramPass") && hasComputePipeline("AutoExposurePass");
        visibilityTaaAvailable = hasComputePipeline("TAAPass");
        visibilityGpuCullingAvailable =
            hasComputePipeline("InstanceClassifyPass") &&
            hasComputePipeline("MeshletCullPass") &&
            hasComputePipeline("BuildIndirectPass") &&
            hasRenderPipeline("VisibilityIndirectPass");

        auto validateVisibilityAsset = [&](const char* label) {
            std::string validationError;
            if (!visibilityPipelineAsset.validate(validationError)) {
                spdlog::error("{}: {}", label, validationError);
                visibilityPipelineAssetLoaded = false;
                return false;
            }
            return true;
        };

        if (visibilityPipelineAssetLoaded) {
            normalizePipelineFinalDisplayOutput(visibilityPipelineAsset);
            erasePassByType(visibilityPipelineAsset, "ImGuiOverlayPass");
        }

        if (visibilityPipelineAssetLoaded && !validateVisibilityAsset("Invalid Vulkan visibility pipeline")) {
            visibilityUpscalerMode = VisibilityUpscalerMode::None;
            dlssStateDirty = true;
        }

        if (visibilityPipelineAssetLoaded && !visibilityAutoExposureAvailable) {
            disablePipelineAutoExposure(visibilityPipelineAsset);
            if (validateVisibilityAsset("Invalid Vulkan visibility pipeline without AutoExposure")) {
                spdlog::warn("Vulkan visibility auto-exposure unavailable; falling back to manual exposure");
            }
        }

        visibilityUpscalerSelection = visibilityPipelineAssetLoaded
            ? analyzeVisibilityUpscalerSelection(visibilityPipelineAsset)
            : VisibilityUpscalerSelection{};
        visibilityUpscalerMode = visibilityUpscalerSelection.activeMode;
        if (visibilityUpscalerMode == VisibilityUpscalerMode::TAA && !visibilityTaaAvailable) {
            spdlog::warn("Vulkan visibility pipeline selects TAAPass, but the TAA shader is unavailable; the pass will forward its source input");
        }
        if (visibilityUpscalerMode == VisibilityUpscalerMode::DLSS && !dlssAvailable) {
            spdlog::warn("Vulkan visibility pipeline selects StreamlineDlssPass, but DLSS is unavailable; the pass will forward its source input");
        }
        dlssStateDirty = true;

        useVisibilityRenderGraph =
            visibilityPipelineAssetLoaded &&
            previewSceneReady &&
            hasRenderPipeline("VisibilityPass") &&
            hasComputePipeline("DeferredLightingPass") &&
            (hasRenderPipeline("TonemapPass") || hasRenderPipeline("OutputPass")) &&
            sceneCtx.mesh().positionBuffer.nativeHandle() &&
            sceneCtx.mesh().normalBuffer.nativeHandle() &&
            sceneCtx.mesh().uvBuffer.nativeHandle() &&
            sceneCtx.mesh().indexBuffer.nativeHandle() &&
            sceneCtx.meshlets().meshletBuffer.nativeHandle() &&
            sceneCtx.meshlets().meshletVertices.nativeHandle() &&
            sceneCtx.meshlets().meshletTriangles.nativeHandle() &&
            sceneCtx.meshlets().boundsBuffer.nativeHandle() &&
            sceneCtx.meshlets().materialIDs.nativeHandle() &&
            sceneCtx.materials().materialBuffer.nativeHandle() &&
            sceneCtx.materials().sampler.nativeHandle() &&
            sceneCtx.shadowDummyTex().nativeHandle() &&
            sceneCtx.skyFallbackTex().nativeHandle();
        atmosphereSkyAvailable = hasRenderPipeline("SkyPass") && atmosphereTextures.isValid();
    };

    auto logVisibilityMode = [&]() {
        if (useVisibilityRenderGraph) {
            std::string mode =
                "Vulkan RenderGraph mode: ClusterStreamingUpdatePass -> MeshletCullPass -> VisibilityPass -> SkyPass -> DeferredLightingPass";
            if (visibilityUpscalerMode == VisibilityUpscalerMode::TAA) {
                mode += " -> TAAPass";
            } else if (visibilityUpscalerMode == VisibilityUpscalerMode::DLSS) {
                mode += " -> StreamlineDlssPass";
            }
            if (findFirstEnabledPassByType(visibilityPipelineAsset, "AutoExposurePass")) {
                mode += " -> AutoExposurePass";
            }
            if (findFirstEnabledPassByType(visibilityPipelineAsset, "TonemapPass")) {
                mode += " -> TonemapPass";
            }
            if (findFirstEnabledPassByType(visibilityPipelineAsset, "OutputPass")) {
                mode += " -> OutputPass";
            }
            spdlog::info("{}", mode);
            if (visibilityGpuCullingAvailable) {
                spdlog::info("Vulkan visibility dispatch: GPU-driven indirect meshlet path");
            } else {
                spdlog::info("Vulkan visibility dispatch: CPU meshlet path");
            }
            if (visibilityUpscalerMode == VisibilityUpscalerMode::DLSS &&
                runtimeContext.upscaler && runtimeContext.upscaler->isEnabled()) {
                spdlog::info("Vulkan render resolution: {} x {} -> {} x {} (DLSS)",
                             runtimeContext.renderWidth,
                             runtimeContext.renderHeight,
                             runtimeContext.displayWidth,
                             runtimeContext.displayHeight);
            } else {
                spdlog::info("Vulkan render resolution: {} x {} (base {} x {}, scale {:.2f}) -> {} x {}",
                             runtimeContext.renderWidth,
                             runtimeContext.renderHeight,
                             kRenderResolutionBaseWidth,
                             kRenderResolutionBaseHeight,
                             visibilityRenderScale,
                             runtimeContext.displayWidth,
                             runtimeContext.displayHeight);
            }
        } else {
            spdlog::info("Vulkan RenderGraph mode: Triangle -> TonemapPass -> OutputPass fallback");
        }
    };

    auto syncVisibilityUpscalerState = [&](int displayWidth, int displayHeight) {
        visibilityRenderScale = std::clamp(visibilityRenderScale, kMinRenderScale, kMaxRenderScale);

        int renderWidth = std::max(1, static_cast<int>(std::lround(
            static_cast<float>(kRenderResolutionBaseWidth) * visibilityRenderScale)));
        int renderHeight = std::max(1, static_cast<int>(std::lround(
            static_cast<float>(kRenderResolutionBaseHeight) * visibilityRenderScale)));
        bool enableDlssEvaluation = false;

#ifdef METALLIC_HAS_STREAMLINE
        if (visibilityUpscalerMode == VisibilityUpscalerMode::DLSS &&
            dlssAvailable &&
            dlssPreset != DlssPreset::Off) {
            uint32_t optimalWidth = static_cast<uint32_t>(displayWidth);
            uint32_t optimalHeight = static_cast<uint32_t>(displayHeight);
            if (streamlineCtx.getOptimalRenderSize(dlssPreset,
                                                   static_cast<uint32_t>(displayWidth),
                                                   static_cast<uint32_t>(displayHeight),
                                                   optimalWidth,
                                                   optimalHeight)) {
                renderWidth = static_cast<int>(optimalWidth);
                renderHeight = static_cast<int>(optimalHeight);
                enableDlssEvaluation = true;
                if (dlssStateDirty ||
                    runtimeContext.displayWidth != displayWidth ||
                    runtimeContext.displayHeight != displayHeight) {
                    if (!streamlineCtx.setDlssOptions(dlssPreset,
                                                      static_cast<uint32_t>(displayWidth),
                                                      static_cast<uint32_t>(displayHeight))) {
                        spdlog::warn("Failed to configure DLSS; StreamlineDlssPass will run in pass-through mode");
                        renderWidth = displayWidth;
                        renderHeight = displayHeight;
                        enableDlssEvaluation = false;
                    }
                    dlssStateDirty = false;
                }
            } else if (dlssStateDirty) {
                spdlog::warn("Failed to query optimal DLSS render size; StreamlineDlssPass will run in pass-through mode");
                dlssStateDirty = false;
            }
        } else {
            dlssStateDirty = true;
        }
#endif

        runtimeContext.upscaler = streamlineCtx.isInitialized() ? static_cast<IUpscalerIntegration*>(&streamlineCtx) : nullptr;
        runtimeContext.displayWidth = displayWidth;
        runtimeContext.displayHeight = displayHeight;
        runtimeContext.renderWidth = renderWidth;
        runtimeContext.renderHeight = renderHeight;
        dlssRenderWidth = static_cast<uint32_t>(renderWidth);
        dlssRenderHeight = static_cast<uint32_t>(renderHeight);
    };

    refreshVisibilityPipelineState();
    syncVisibilityUpscalerState(width, height);
    logVisibilityMode();

    VulkanFrameGraphBackend frameGraphBackend(vkDevice, vkPhysicalDevice, vmaAllocator);
    frameGraphBackend.setTransientPool(&getVulkanTransientPool(*rhi));
    VulkanDescriptorManager legacyDescriptorManager;
    VulkanDescriptorBufferManager descriptorBufferManager;
    IVulkanDescriptorBackend* descriptorBackend = nullptr;
    {
        const auto& limits = rhi->limits();
        if (rhi->features().descriptorBuffer) {
            const auto& dbProps = getVulkanDescriptorBufferProperties(*rhi);
            descriptorBufferManager.init(vkDevice, vkPhysicalDevice, vmaAllocator, dbProps,
                                         limits.minUniformBufferOffsetAlignment,
                                         limits.nonCoherentAtomSize,
                                         limits.maxUniformBufferRange);
            descriptorBackend = &descriptorBufferManager;
            spdlog::info("Vulkan: using VK_EXT_descriptor_buffer path");
        } else {
            legacyDescriptorManager.init(vkDevice, vkPhysicalDevice, vmaAllocator,
                                         limits.minUniformBufferOffsetAlignment,
                                         limits.nonCoherentAtomSize,
                                         limits.maxUniformBufferRange);
            descriptorBackend = &legacyDescriptorManager;
        }
    }
    // --- Upload / readback service layer ---
    VulkanUploadService uploadService;
    {
        VkQueue gfxQueue = nativeToVkHandle<VkQueue>(native.queue);
        VkQueue xferQueue = nativeToVkHandle<VkQueue>(native.transferQueue);
        uint32_t xferFamily = native.transferQueueFamily;
        VkSemaphore xferSemaphore = nativeToVkHandle<VkSemaphore>(native.transferTimelineSemaphore);
        uploadService.init(vkDevice, vmaAllocator,
                           gfxQueue, native.graphicsQueueFamily,
                           xferQueue, xferFamily, xferSemaphore,
                           &getVulkanUploadRing(*rhi));
        vulkanSetUploadService(&uploadService);
    }
    VulkanReadbackService readbackService;
    readbackService.init(vkDevice, &getVulkanReadbackHeap(*rhi), 2);
    runtimeContext.readbackService = &readbackService;

    if (previewSceneReady) {
        if (!sceneCtx.materials().textureViews.empty()) {
            descriptorBackend->updateBindlessSampledTextures(sceneCtx.materials().textureViews.data(),
                                                            0,
                                                            static_cast<uint32_t>(sceneCtx.materials().textureViews.size()));
        }
        descriptorBackend->updateBindlessSampler(METALLIC_BINDLESS_SCENE_SAMPLER_INDEX,
                                                &sceneCtx.materials().sampler);
        runtimeContext.useBindlessSceneTextures = true;
    }
    VulkanResourceStateTracker imageTracker;
    streamlineCtx.setImageLayoutTracker(&imageTracker);
    VulkanImportedTexture backbufferTexture;

    RhiTextureHandle sceneColorTexture;
    RhiTextureHandle viewportDisplayTexture;
    VkDescriptorSet viewportImguiDescriptor = VK_NULL_HANDLE;
    VkSampler viewportImguiSampler = VK_NULL_HANDLE;

    auto createViewportSampler = [&]() {
        if (viewportImguiSampler != VK_NULL_HANDLE) return;
        VkSamplerCreateInfo samplerCI{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        samplerCI.magFilter = VK_FILTER_LINEAR;
        samplerCI.minFilter = VK_FILTER_LINEAR;
        samplerCI.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        vkCreateSampler(vkDevice, &samplerCI, nullptr, &viewportImguiSampler);
    };

    auto recreateViewportDisplayTexture = [&](uint32_t w, uint32_t h) {
        if (viewportImguiDescriptor != VK_NULL_HANDLE) {
            ImGui_ImplVulkan_RemoveTexture(viewportImguiDescriptor);
            viewportImguiDescriptor = VK_NULL_HANDLE;
        }
        rhiReleaseHandle(viewportDisplayTexture);
        viewportDisplayTexture = rhiCreateTexture2D(deviceHandle, w, h,
            RhiFormat::BGRA8Unorm, false, 1,
            RhiTextureStorageMode::Private,
            RhiTextureUsage::RenderTarget | RhiTextureUsage::ShaderRead);
        if (!viewportDisplayTexture.nativeHandle()) return;

        createViewportSampler();
        VkImageView iv = getVulkanImageView(&viewportDisplayTexture);
        if (iv != VK_NULL_HANDLE) {
            viewportImguiDescriptor = ImGui_ImplVulkan_AddTexture(
                viewportImguiSampler, iv, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
    };

    auto recreateSceneColorTexture = [&](uint32_t targetWidth, uint32_t targetHeight) {
        rhiReleaseHandle(sceneColorTexture);
        rhiReleaseHandle(viewportDisplayTexture);
        runtimeContext.importedTexturesRhi.erase("sceneColor");
        sceneColorTexture = rhiCreateTexture2D(deviceHandle,
                                               targetWidth,
                                               targetHeight,
                                               RhiFormat::RGBA16Float,
                                               false,
                                               1,
                                               RhiTextureStorageMode::Private,
                                               RhiTextureUsage::RenderTarget | RhiTextureUsage::ShaderRead);
        if (!sceneColorTexture.nativeHandle()) {
            return false;
        }
        importRuntimeTexture("sceneColor", sceneColorTexture);
        return true;
    };

    if (!recreateSceneColorTexture(createInfo.width, createInfo.height)) {
        spdlog::error("Failed to create offscreen scene color texture");
        rhi->waitIdle();
        uploadService.destroy();
        vulkanSetUploadService(nullptr);
        readbackService.destroy();
        descriptorBufferManager.destroy();
        legacyDescriptorManager.destroy();
        atmosphereTextures.release();
        shadowResources.release();
        sceneCtx.unloadScene();
        rhiReleaseHandle(linearSampler);
        rhiReleaseHandle(trianglePipeline);
        rhiReleaseHandle(vertexDescriptor);
        rhiReleaseHandle(vertexBuffer);
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    recreateViewportDisplayTexture(createInfo.width, createInfo.height);

    struct TrianglePassData {
        FGResource colorTarget;
    };

    FrameGraph sceneGraph;
    FGResource sceneColorRes = sceneGraph.import("sceneColor", &sceneColorTexture);
    const RhiGraphicsPipeline* trianglePipelinePtr = &trianglePipeline;
    RhiBuffer* triangleBufferPtr = &vertexBuffer;
    TrianglePassData& trianglePassData = sceneGraph.addRenderPass<TrianglePassData>(
        "Triangle Pass",
        [sceneColorRes](FGBuilder& builder, TrianglePassData& data) {
            data.colorTarget = builder.setColorAttachment(0,
                                                          sceneColorRes,
                                                          RhiLoadAction::Clear,
                                                          RhiStoreAction::Store,
                                                          RhiClearColor(0.08, 0.09, 0.12, 1.0));
        },
        [trianglePipelinePtr, triangleBufferPtr](const TrianglePassData&, RhiRenderCommandEncoder& encoder) {
            encoder.setRenderPipeline(*trianglePipelinePtr);
            encoder.setFrontFacingWinding(RhiWinding::CounterClockwise);
            encoder.setCullMode(RhiCullMode::None);
            encoder.setVertexBuffer(triangleBufferPtr, 0, 0);
            encoder.drawPrimitives(RhiPrimitiveType::Triangle, 0, 3);
        });
    sceneGraph.exportResource(trianglePassData.colorTarget);
    sceneGraph.compile();

    const double depthClearValue = ML_DEPTH_REVERSED ? 0.0 : 1.0;
    RhiDepthStencilStateHandle depthState =
        rhiCreateDepthStencilState(deviceHandle, true, ML_DEPTH_REVERSED);
    auto makeRenderContext = [&]() -> RenderContext {
        return RenderContext{
            sceneCtx.mesh(),
            sceneCtx.meshlets(),
            sceneCtx.materials(),
            sceneCtx.sceneGraph(),
            sceneCtx.gpuScene(),
            sceneCtx.clusterLod(),
            shadowResources,
            depthState,
            sceneCtx.shadowDummyTex(),
            sceneCtx.skyFallbackTex(),
            depthClearValue,
        };
    };
    auto rebuildRenderContext = [&](RenderContext& ctx) {
        ctx.~RenderContext();
        new (&ctx) RenderContext(makeRenderContext());
    };
    RenderContext renderContext = makeRenderContext();

    runtimeContext.backbufferRhi = &backbufferTexture;
    runtimeContext.resourceFactory = &frameGraphBackend;
    runtimeContext.upscaler = streamlineCtx.isInitialized() ? static_cast<IUpscalerIntegration*>(&streamlineCtx) : nullptr;
    runtimeContext.uiControls = &pipelineUiControls;

    const PipelineAsset sceneColorPostAsset = makeSceneColorPostPipelineAsset();
    PipelineBuilder postBuilder(renderContext);
    auto rebuildPostBuilder = [&](int targetWidth, int targetHeight) {
        const PipelineAsset& activePostAsset =
            (useClusterVisMode && clusterVisPipelineLoaded) ? clusterVisPipelineAsset :
            useVisibilityRenderGraph ? visibilityPipelineAsset : sceneColorPostAsset;
        if (!postBuilder.build(activePostAsset, runtimeContext, targetWidth, targetHeight)) {
            spdlog::error("Failed to build Vulkan post pipeline: {}", postBuilder.lastError());
            return false;
        }
        postBuilder.compile();
        return true;
    };

    auto cleanupRuntimeResources = [&]() {
        rhi->waitIdle();
        if (viewportImguiDescriptor != VK_NULL_HANDLE) {
            ImGui_ImplVulkan_RemoveTexture(viewportImguiDescriptor);
            viewportImguiDescriptor = VK_NULL_HANDLE;
        }
        if (viewportImguiSampler != VK_NULL_HANDLE) {
            vkDestroySampler(vkDevice, viewportImguiSampler, nullptr);
            viewportImguiSampler = VK_NULL_HANDLE;
        }
        vulkanSetUploadService(nullptr);
        uploadService.destroy();
        readbackService.destroy();
        streamlineCtx.shutdown();
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        postBuilder.frameGraph().reset();
        sceneGraph.reset();
        descriptorBufferManager.destroy();
        legacyDescriptorManager.destroy();
        atmosphereTextures.release();
        shadowResources.release();
        sceneCtx.unloadScene();
        rhiReleaseHandle(depthState);
        rhiReleaseHandle(linearSampler);
        rhiReleaseHandle(trianglePipeline);
        rhiReleaseHandle(sceneColorTexture);
        rhiReleaseHandle(viewportDisplayTexture);
        rhiReleaseHandle(vertexDescriptor);
        rhiReleaseHandle(vertexBuffer);
    };

    VkImageLayout sceneColorLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    FrameContext frameContext;
    previewCamera.initFromBounds(sceneCtx.mesh().bboxMin, sceneCtx.mesh().bboxMax);
    previewCamera.distance *= 0.8f;
    previewCamera.azimuth = 0.55f;
    previewCamera.elevation = 0.35f;
    if (!previewSceneReady) {
        previewCamera.nearZ = 0.1f;
        previewCamera.farZ = 100.0f;
    }
    DirectionalLight sunLight;
    if (previewSceneReady) {
        sunLight = sceneCtx.sceneGraph().getSunDirectionalLight();
    } else {
        sunLight.direction = normalize(float3(0.35f, 0.85f, 0.25f));
        sunLight.color = float3(1.0f, 0.98f, 0.95f);
        sunLight.intensity = 2.5f;
    }
    std::vector<uint32_t> previewVisibleMeshletNodes;
    std::vector<uint32_t> previewVisibleIndexNodes;
    auto refreshPreviewSceneState = [&]() {
        if (!previewSceneReady) {
            return;
        }

        sceneCtx.updateGpuScene();
        sunLight = sceneCtx.sceneGraph().getSunDirectionalLight();

        previewVisibleMeshletNodes.clear();
        previewVisibleIndexNodes.clear();
        const bool gpuDrivenVisibilityPath =
            useVisibilityRenderGraph && visibilityGpuCullingAvailable;
        if (gpuDrivenVisibilityPath) {
            return;
        }
        for (const auto& node : sceneCtx.sceneGraph().nodes) {
            if (!sceneCtx.sceneGraph().isNodeVisible(node.id)) {
                continue;
            }
            if (node.meshletCount > 0) {
                previewVisibleMeshletNodes.push_back(node.id);
            }
            if (node.indexCount > 0) {
                previewVisibleIndexNodes.push_back(node.id);
            }
        }
    };
    if (previewSceneReady) {
        previewVisibleMeshletNodes.reserve(sceneCtx.sceneGraph().nodes.size());
        previewVisibleIndexNodes.reserve(sceneCtx.sceneGraph().nodes.size());
        refreshPreviewSceneState();
    } else {
        previewVisibleIndexNodes = {0};
    }
    bool showSceneGraphWindow = true;
    bool showGraphDebug = true;
    bool showRenderPassUI = true;
    bool showImGuiDemo = false;
    bool reloadKeyDown = false;
    bool pipelineReloadKeyDown = false;
    bool shaderReloadRequested = false;
    bool pipelineReloadRequested = false;
    bool streamingPipelineResetRequested = false;
    bool postBuilderNeedsRebuild = false;
    bool visibilityHistoryResetRequested = false;
    double lastFrameTime = glfwGetTime();
    float4x4 prevView = float4x4::Identity();
    float4x4 prevProj = float4x4::Identity();
    float4x4 prevCullView = float4x4::Identity();
    float4x4 prevCullProj = float4x4::Identity();
    float4 prevCameraWorldPos = float4(0.0f, 0.0f, 0.0f, 1.0f);
    bool hasPrevMatrices = false;
    uint32_t frameIndex = 0;

    auto applyDlssPresetChange = [&](DlssPreset newPreset) {
        if (newPreset == dlssPreset) {
            return;
        }
        dlssPreset = newPreset;
        dlssStateDirty = true;
        visibilityHistoryResetRequested = true;
        hasPrevMatrices = false;
        streamlineCtx.resetHistory();
        postBuilderNeedsRebuild = true;
    };

    auto refreshPipelineUiControls = [&]() {
        pipelineUiControls.enableRTShadows = &enableRTShadows;
        pipelineUiControls.rtShadowsAvailable = rtShadowsAvailable;
        pipelineUiControls.useVisibilityRenderGraph = useVisibilityRenderGraph;
        pipelineUiControls.hasDlssPass = visibilityUpscalerSelection.hasDlssPass;
        pipelineUiControls.dlssAvailable = dlssAvailable;
        pipelineUiControls.dlssEnabled = runtimeContext.upscaler && runtimeContext.upscaler->isEnabled();
        pipelineUiControls.dlssIsActiveUpscaler =
            visibilityUpscalerMode == VisibilityUpscalerMode::DLSS;
        pipelineUiControls.currentPreset = dlssPreset;
        pipelineUiControls.dlssRenderWidth = dlssRenderWidth;
        pipelineUiControls.dlssRenderHeight = dlssRenderHeight;
        pipelineUiControls.displayWidth = width;
        pipelineUiControls.displayHeight = height;
        pipelineUiControls.dlssDiagnostic = visibilityUpscalerSelection.diagnostic;
        pipelineUiControls.onDlssPresetChanged = applyDlssPresetChange;
        pipelineUiControls.onResetDlssHistory = [&]() {
            streamlineCtx.resetHistory();
        };
        runtimeContext.uiControls = &pipelineUiControls;
    };

    auto rebuildActivePipeline = [&](int targetWidth, int targetHeight) {
        rhi->waitIdle();
        syncVisibilityUpscalerState(targetWidth, targetHeight);
        const int buildWidth = useVisibilityRenderGraph ? runtimeContext.renderWidth : targetWidth;
        const int buildHeight = useVisibilityRenderGraph ? runtimeContext.renderHeight : targetHeight;
        if (!useVisibilityRenderGraph &&
            (!sceneColorTexture.nativeHandle() ||
             sceneColorTexture.width() != static_cast<uint32_t>(targetWidth) ||
             sceneColorTexture.height() != static_cast<uint32_t>(targetHeight))) {
            if (!recreateSceneColorTexture(static_cast<uint32_t>(targetWidth),
                                           static_cast<uint32_t>(targetHeight))) {
                spdlog::error("Failed to recreate offscreen scene color texture");
                return false;
            }
        }
        sceneColorLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (!rebuildPostBuilder(buildWidth, buildHeight)) {
            return false;
        }
        return true;
    };

    if (!rebuildActivePipeline(width, height)) {
        cleanupRuntimeResources();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    uint32_t uploadFrameCounter = 0;
    while (!glfwWindowShouldClose(window)) {
        ZoneScopedN("VulkanRenderGraphFrame");

        glfwPollEvents();
        const bool f5Down = glfwGetKey(window, GLFW_KEY_F5) == GLFW_PRESS;
        if (f5Down && !reloadKeyDown) {
            shaderReloadRequested = true;
        }
        reloadKeyDown = f5Down;
        const bool f6Down = glfwGetKey(window, GLFW_KEY_F6) == GLFW_PRESS;
        if (f6Down && !pipelineReloadKeyDown) {
            pipelineReloadRequested = true;
        }
        pipelineReloadKeyDown = f6Down;
        glfwGetFramebufferSize(window, &width, &height);
        if (width == 0 || height == 0) {
            glfwWaitEvents();
            continue;
        }

        if (appState.framebufferResized) {
            rhi->waitIdle();
            rhi->resize(static_cast<uint32_t>(width), static_cast<uint32_t>(height));
            ImGui_ImplVulkan_SetMinImageCount(std::max(2u, rhi->nativeHandles().swapchainImageCount));
            recreateViewportDisplayTexture(static_cast<uint32_t>(width), static_cast<uint32_t>(height));
            postBuilderNeedsRebuild = true;
            appState.framebufferResized = false;
            dlssStateDirty = true;
            visibilityHistoryResetRequested = true;
            if (visibilityUpscalerMode == VisibilityUpscalerMode::DLSS) {
                streamlineCtx.resetHistory();
            }
        }

        if (shaderReloadRequested) {
            shaderReloadRequested = false;
            rhi->waitIdle();
            spdlog::info("Reloading Vulkan visibility shaders...");
            const bool previousVisibilityRenderGraph = useVisibilityRenderGraph;
            const bool previousAutoExposure = visibilityAutoExposureAvailable;
            const bool previousTaa = visibilityTaaAvailable;
            const bool previousAtmosphereSky = atmosphereSkyAvailable;
            const VisibilityUpscalerMode previousUpscalerMode = visibilityUpscalerMode;
            auto [reloaded, failed] = shaderManager.reloadAll();
            refreshVisibilityPipelineState();

            if (rtShadowsAvailable) {
                if (reloadShadowPipeline(deviceHandle, shadowResources, PROJECT_SOURCE_DIR)) {
                    reloaded++;
                    spdlog::info("Reloaded Vulkan RT shadow shader");
                } else {
                    failed++;
                    spdlog::warn("Failed to reload Vulkan RT shadow shader; keeping previous pipeline");
                }
            }

            if (failed == 0) {
                spdlog::info("All {} Vulkan visibility shaders reloaded successfully", reloaded);
            } else {
                spdlog::warn("{} Vulkan visibility shaders reloaded, {} failed (keeping old pipelines)",
                             reloaded,
                             failed);
            }

            if (previousVisibilityRenderGraph != useVisibilityRenderGraph ||
                previousAutoExposure != visibilityAutoExposureAvailable ||
                previousTaa != visibilityTaaAvailable ||
                previousAtmosphereSky != atmosphereSkyAvailable ||
                previousUpscalerMode != visibilityUpscalerMode) {
                logVisibilityMode();
            }
            postBuilderNeedsRebuild = true;
            dlssStateDirty = true;
            visibilityHistoryResetRequested = true;
            if (visibilityUpscalerMode == VisibilityUpscalerMode::DLSS) streamlineCtx.resetHistory();
        }

        if (pipelineReloadRequested) {
            pipelineReloadRequested = false;
            spdlog::info("Reloading Vulkan visibility pipeline asset...");

            PipelineAsset reloadedVisibilityAsset;
            if (loadPipelineAssetChecked(visibilityPipelinePath,
                                         "Vulkan visibility",
                                         reloadedVisibilityAsset)) {
                const bool previousVisibilityRenderGraph = useVisibilityRenderGraph;
                const bool previousAutoExposure = visibilityAutoExposureAvailable;
                const bool previousTaa = visibilityTaaAvailable;
                const bool previousAtmosphereSky = atmosphereSkyAvailable;
                const VisibilityUpscalerMode previousUpscalerMode = visibilityUpscalerMode;
                visibilityPipelineBaseAsset = std::move(reloadedVisibilityAsset);
                visibilityPipelineBaseLoaded = true;
                refreshVisibilityPipelineState();

                if (previousVisibilityRenderGraph != useVisibilityRenderGraph ||
                    previousAutoExposure != visibilityAutoExposureAvailable ||
                    previousTaa != visibilityTaaAvailable ||
                    previousAtmosphereSky != atmosphereSkyAvailable ||
                    previousUpscalerMode != visibilityUpscalerMode) {
                    logVisibilityMode();
                }
                postBuilderNeedsRebuild = true;
                streamingPipelineResetRequested = true;
            } else if (visibilityPipelineBaseLoaded) {
                spdlog::warn("Keeping previous Vulkan visibility pipeline: {}",
                             visibilityPipelineBaseAsset.name);
            }
            dlssStateDirty = true;
            visibilityHistoryResetRequested = true;
            if (visibilityUpscalerMode == VisibilityUpscalerMode::DLSS) streamlineCtx.resetHistory();
        }

        syncVisibilityUpscalerState(width, height);
        refreshPipelineUiControls();
        const int activeBuildWidth = useVisibilityRenderGraph ? runtimeContext.renderWidth : width;
        const int activeBuildHeight = useVisibilityRenderGraph ? runtimeContext.renderHeight : height;
        if (postBuilderNeedsRebuild || postBuilder.needsRebuild(activeBuildWidth, activeBuildHeight)) {
            if (!rebuildActivePipeline(width, height)) {
                break;
            }
            postBuilderNeedsRebuild = false;
            if (streamingPipelineResetRequested) {
                clusterStreamingService.resetForPipelineReload();
                streamingPipelineResetRequested = false;
            }
        }

        if (!rhi->beginFrame()) {
            if (vulkanIsDeviceLost(*rhi)) {
                spdlog::critical("Ending Vulkan main loop after device loss: {}",
                                 vulkanDeviceLostMessage(*rhi));
                break;
            }
            continue;
        }

        VkImage backbufferImage = getVulkanCurrentBackbufferImage(*rhi);
        VkImageView backbufferImageView = getVulkanCurrentBackbufferImageView(*rhi);
        VkExtent2D backbufferExtent = getVulkanCurrentBackbufferExtent(*rhi);
        backbufferTexture.set(backbufferImage,
                              backbufferImageView,
                              backbufferExtent.width,
                              backbufferExtent.height,
                              static_cast<VkFormat>(native.colorFormat),
                              RhiTextureUsage::RenderTarget);

        descriptorBackend->resetFrame();
        uploadService.beginFrame(uploadFrameCounter);
        readbackService.beginFrame(uploadFrameCounter);
        ++uploadFrameCounter;
        const auto barrierStats = imageTracker.stats();  // capture before clear
        imageTracker.clear();
        if (backbufferImage != VK_NULL_HANDLE) {
            imageTracker.setLayout(backbufferImage, getVulkanCurrentBackbufferLayout(*rhi));
        }
        if (sceneColorTexture.nativeHandle()) {
            imageTracker.setLayout(getVulkanImage(&sceneColorTexture), sceneColorLayout);
        }

        VulkanCommandBuffer commandBuffer(getVulkanCurrentCommandBuffer(*rhi),
                                          vkDevice,
                                          descriptorBackend,
                                          &imageTracker,
                                          getVulkanGpuProfiler(*rhi),
                                          getVulkanCurrentComputeCommandBuffer(*rhi));

        // Record any deferred uploads staged since last frame
        VkCommandBuffer nativeCmd = getVulkanCurrentCommandBuffer(*rhi);
        if (uploadService.hasPendingUploads()) {
            uploadService.recordPendingUploads(nativeCmd);
        }
        readbackService.recordPendingReadbacks(nativeCmd);

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        const ImGuiID dockspaceId =
            beginDockspace(showSceneGraphWindow, showGraphDebug, showRenderPassUI, showImGuiDemo);

        // --- Viewport window: display scene render output ---
        {
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
            ImGui::SetNextWindowDockID(dockspaceId, ImGuiCond_FirstUseEver);
            ImGui::Begin("Viewport", nullptr, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
            s_viewportHovered = ImGui::IsWindowHovered();
            ImVec2 viewportSize = ImGui::GetContentRegionAvail();
            if (viewportSize.x > 0 && viewportSize.y > 0 &&
                viewportImguiDescriptor != VK_NULL_HANDLE) {
                ImGui::Image(reinterpret_cast<ImTextureID>(viewportImguiDescriptor), viewportSize);
            }
            ImGui::End();
            ImGui::PopStyleVar();
        }

        ImGui::SetNextWindowDockID(dockspaceId, ImGuiCond_FirstUseEver);
        ImGui::Begin("Vulkan Sponza");
        const RhiDeviceInfo& deviceInfo = rhi->deviceInfo();
        const VulkanToolingInfo& toolingInfo = getVulkanToolingInfo(*rhi);
        const VulkanGpuFrameDiagnostics& gpuFrameDiagnostics = getVulkanLatestFrameDiagnostics(*rhi);
        const VulkanPipelineCacheTelemetry pipelineTelemetry = getVulkanPipelineCacheTelemetry(*rhi);
        const std::vector<SlangDiagnosticRecord> slangDiagnostics = getRecentSlangDiagnostics();
        const bool autoExposureEnabled =
            useVisibilityRenderGraph &&
            findFirstEnabledPassByType(visibilityPipelineAsset, "AutoExposurePass") != nullptr;
        ImGui::Text("Resolution: %d x %d", width, height);
        ImGui::TextUnformatted(useVisibilityRenderGraph ? "Pipeline: Visibility Buffer" : "Pipeline: Triangle Fallback");
        if (clusterVisPipelineLoaded && hasRenderPipeline("ClusterRenderPass")) {
            ImGui::SameLine();
            if (ImGui::Checkbox("Cluster Vis", &useClusterVisMode)) {
                spdlog::info("Cluster visualization mode: {}", useClusterVisMode ? "ON" : "OFF");
                postBuilderNeedsRebuild = true;
            }
        }
        ImGui::Text("Upscaler: %s", visibilityUpscalerModeName(visibilityUpscalerMode));
        ImGui::TextUnformatted(autoExposureEnabled ? "Exposure: Auto" : "Exposure: Manual");
        ImGui::TextUnformatted(visibilityGpuCullingAvailable ? "Visibility Dispatch: GPU" : "Visibility Dispatch: CPU");
        if (ImGui::CollapsingHeader("Diagnostics", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("Adapter: %s", deviceInfo.adapterName.c_str());
            ImGui::Text("Driver: %s", deviceInfo.driverName.c_str());
            ImGui::Text("API: %s", formatVulkanVersion(deviceInfo.apiVersion).c_str());
            ImGui::Separator();
            ImGui::Text("Debug Utils: %s", toolingInfo.debugUtils ? "Enabled" : "Unavailable");
            ImGui::Text("Nsight NVTX: %s", metallic::nsightMarkersAvailable() ? "Enabled" : "Unavailable");
            ImGui::Text("Validation Messenger: %s",
                        toolingInfo.validationMessenger ? "Active" : "Inactive");
            ImGui::Text("RenderDoc Layer: %s",
                        toolingInfo.renderDocLayerAvailable ? "Available" : "Not detected");
            ImGui::Text("Pipeline Statistics: %s",
                        toolingInfo.pipelineStatistics ? "Enabled" : "Disabled");
            ImGui::Text("Diagnostic Checkpoints: %s",
                        toolingInfo.diagnosticCheckpoints ? "Enabled" : "Disabled");
            ImGui::Text("Device Fault Extension: %s",
                        toolingInfo.deviceFault ? "Enabled" : "Disabled");
            if (vulkanIsDeviceLost(*rhi)) {
                ImGui::Separator();
                ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f),
                                   "Device Lost: %s",
                                   vulkanDeviceLostMessage(*rhi).c_str());
            }
        }
        if (ImGui::CollapsingHeader("Pipeline Telemetry", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("Graphics PSOs compiled: %u", pipelineTelemetry.graphicsPipelinesCompiled);
            ImGui::Text("Compute PSOs compiled:  %u", pipelineTelemetry.computePipelinesCompiled);
            ImGui::Text("Total compile time:     %.2f ms", pipelineTelemetry.totalCompileMs);
        }
        if (ImGui::CollapsingHeader("GPU Timings", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("Last completed frame: #%llu",
                        static_cast<unsigned long long>(gpuFrameDiagnostics.frameIndex));
            ImGui::Text("Total GPU time: %.3f ms", gpuFrameDiagnostics.totalGpuMs);
            if (gpuFrameDiagnostics.scopes.empty()) {
                ImGui::TextDisabled("No pass timings captured yet.");
            } else if (ImGui::BeginTable("GpuTimings", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                ImGui::TableSetupColumn("Pass");
                ImGui::TableSetupColumn("Time (ms)");
                ImGui::TableSetupColumn("Graphics Stats");
                ImGui::TableSetupColumn("Compute Stats");
                ImGui::TableHeadersRow();
                for (const VulkanGpuScopeTiming& scope : gpuFrameDiagnostics.scopes) {
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::TextUnformatted(scope.label.c_str());
                    ImGui::TableSetColumnIndex(1);
                    ImGui::Text("%.3f", scope.durationMs);
                    ImGui::TableSetColumnIndex(2);
                    if (scope.pipelineStats.valid) {
                        ImGui::Text("VS %llu / FS %llu",
                                    static_cast<unsigned long long>(scope.pipelineStats.vertexShaderInvocations),
                                    static_cast<unsigned long long>(scope.pipelineStats.fragmentShaderInvocations));
                    } else {
                        ImGui::TextDisabled("-");
                    }
                    ImGui::TableSetColumnIndex(3);
                    if (scope.pipelineStats.valid &&
                        (scope.pipelineStats.computeShaderInvocations != 0 ||
                         scope.pipelineStats.taskShaderInvocations != 0 ||
                         scope.pipelineStats.meshShaderInvocations != 0)) {
                        ImGui::Text("CS %llu / TS %llu / MS %llu",
                                    static_cast<unsigned long long>(scope.pipelineStats.computeShaderInvocations),
                                    static_cast<unsigned long long>(scope.pipelineStats.taskShaderInvocations),
                                    static_cast<unsigned long long>(scope.pipelineStats.meshShaderInvocations));
                    } else {
                        ImGui::TextDisabled("-");
                    }
                }
                ImGui::EndTable();
            }
        }
        if (ImGui::CollapsingHeader("Shader Diagnostics")) {
            if (slangDiagnostics.empty()) {
                ImGui::TextDisabled("No recent Slang diagnostics.");
            } else {
                for (auto it = slangDiagnostics.rbegin(); it != slangDiagnostics.rend(); ++it) {
                    ImGui::Separator();
                    ImGui::Text("%s", it->stage.c_str());
                    if (!it->shaderPath.empty()) {
                        ImGui::TextDisabled("%s", it->shaderPath.c_str());
                    }
                    ImGui::PushTextWrapPos();
                    ImGui::TextUnformatted(it->message.c_str());
                    ImGui::PopTextWrapPos();
                }
            }
        }

        if (ImGui::CollapsingHeader("Barrier Stats (prev frame)")) {
            ImGui::Text("Image barriers:   %u", barrierStats.imageBarriers);
            ImGui::Text("Buffer barriers:  %u", barrierStats.bufferBarriers);
            ImGui::Text("Memory barriers:  %u", barrierStats.memoryBarriers);
            ImGui::Text("Redundant skips:  %u", barrierStats.redundantSkips);
            ImGui::Text("Flush calls:      %u", barrierStats.flushCalls);
            ImGui::Text("Empty flushes:    %u", barrierStats.emptyFlushCalls);
        }
        if (ImGui::CollapsingHeader("Shader Cache")) {
            SlangCompileStats shaderStats = getSlangCompileStats();
            ImGui::Text("Cache hits:       %u", shaderStats.cacheHits);
            ImGui::Text("Cache misses:     %u", shaderStats.cacheMisses);
            ImGui::Text("Compiles:         %u", shaderStats.compileCount);
            ImGui::Text("Total compile:    %.1f ms", shaderStats.totalCompileTimeMs);
            const char* modeLabel = (shaderCompileMode == ShaderCompileMode::Release)
                                    ? "Release" : "Debug";
            ImGui::Text("Compile mode:     %s", modeLabel);
            if (ImGui::Button("Reset Stats")) {
                resetSlangCompileStats();
            }
        }
        if (useVisibilityRenderGraph) {
            ImGui::Text("Render Base: %d x %d", kRenderResolutionBaseWidth, kRenderResolutionBaseHeight);
            const bool allowManualRenderScale = visibilityUpscalerMode != VisibilityUpscalerMode::DLSS;
            if (!allowManualRenderScale) {
                ImGui::BeginDisabled();
            }
            float requestedRenderScale = visibilityRenderScale;
            if (ImGui::SliderFloat("Render Scale",
                                   &requestedRenderScale,
                                   kMinRenderScale,
                                   kMaxRenderScale,
                                   "%.2f")) {
                requestedRenderScale = std::clamp(requestedRenderScale, kMinRenderScale, kMaxRenderScale);
                if (std::abs(requestedRenderScale - visibilityRenderScale) > 0.0001f) {
                    visibilityRenderScale = requestedRenderScale;
                    postBuilderNeedsRebuild = true;
                    visibilityHistoryResetRequested = true;
                    hasPrevMatrices = false;
                }
            }
            if (!allowManualRenderScale) {
                ImGui::EndDisabled();
                ImGui::TextUnformatted("Render Scale is disabled while DLSS is the active upscaler.");
            }
            ImGui::Text("Render Resolution: %d x %d",
                        runtimeContext.renderWidth,
                        runtimeContext.renderHeight);
        }
        if (ImGui::Button("Reload Shaders (F5)")) {
            shaderReloadRequested = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Reload Pipeline (F6)")) {
            pipelineReloadRequested = true;
        }
        ImGui::Checkbox("FrameGraph Debug", &showGraphDebug);
        ImGui::Checkbox("Render Pass UI", &showRenderPassUI);
        ImGui::Checkbox("Scene Browser", &showSceneGraphWindow);
        ImGui::Checkbox("ImGui Demo", &showImGuiDemo);

        const ClusterStreamingService::DebugStats& streamingStats =
            clusterStreamingService.debugStats();
        const ClusterStreamingService::StreamingStats& streamingTelemetry =
            clusterStreamingService.streamingStats();
        static StreamingDashboardHistory streamingHistory;
        streamingHistory.push(frameIndex, streamingTelemetry);

        if (ImGui::CollapsingHeader("Cluster Streaming")) {
            const ClusterStreamingService::MemoryBudgetInfo& memoryBudgetInfo =
                clusterStreamingService.memoryBudgetInfo();
            static constexpr uint64_t kMiB = 1024ull * 1024ull;
            static constexpr int kMinStreamingStorageBudgetMB = 64;
            static constexpr int kMaxStreamingStorageBudgetMB = 2048;
            static constexpr int kStreamingStorageBudgetStepMB = 16;
            ImGui::Text("Streaming: %s",
                        clusterStreamingService.streamingEnabled() ? "Enabled" : "Disabled");
            ImGui::Text("Resources: %s",
                        streamingStats.resourcesReady ? "Ready" : "Pending");
            if (ImGui::BeginCombo("Budget Preset",
                                  ClusterStreamingService::budgetPresetLabel(
                                      clusterStreamingService.budgetPreset()))) {
                for (uint32_t presetIndex = 0u;
                     presetIndex <
                         static_cast<uint32_t>(ClusterStreamingService::BudgetPreset::Custom);
                     ++presetIndex) {
                    const auto preset =
                        static_cast<ClusterStreamingService::BudgetPreset>(presetIndex);
                    const bool selected = clusterStreamingService.budgetPreset() == preset;
                    if (ImGui::Selectable(ClusterStreamingService::budgetPresetLabel(preset),
                                          selected)) {
                        clusterStreamingService.setBudgetPreset(preset);
                        if (preset == ClusterStreamingService::BudgetPreset::Auto) {
                            refreshClusterStreamingMemoryBudget();
                        }
                        visibilityHistoryResetRequested = true;
                    }
                    if (selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
            ImGui::Text("Budget target: %s, %u dynamic groups",
                        formatByteCountShort(clusterStreamingService.streamingStorageCapacityBytes()).c_str(),
                        clusterStreamingService.streamingBudgetGroups());
            int storageBudgetMb = static_cast<int>(
                std::max<uint64_t>(1ull, clusterStreamingService.streamingStorageCapacityBytes() / kMiB));
            if (ImGui::SliderInt("Storage Pool Target (MB)",
                                 &storageBudgetMb,
                                 kMinStreamingStorageBudgetMB,
                                 kMaxStreamingStorageBudgetMB)) {
                storageBudgetMb = std::max(
                    kMinStreamingStorageBudgetMB,
                    ((storageBudgetMb + (kStreamingStorageBudgetStepMB / 2)) /
                     kStreamingStorageBudgetStepMB) *
                        kStreamingStorageBudgetStepMB);
                clusterStreamingService.setStreamingStorageCapacityBytes(
                    uint64_t(storageBudgetMb) * kMiB);
                visibilityHistoryResetRequested = true;
            }
            const int maxDynamicGroupBudget = std::max(
                1024,
                std::max(static_cast<int>(streamingStats.activeResidencyGroupCount),
                         static_cast<int>(clusterStreamingService.streamingBudgetGroups())));
            int dynamicGroupBudget = static_cast<int>(clusterStreamingService.streamingBudgetGroups());
            if (ImGui::SliderInt("Dynamic Group Budget",
                                 &dynamicGroupBudget,
                                 1,
                                 maxDynamicGroupBudget)) {
                clusterStreamingService.setStreamingBudgetGroups(
                    static_cast<uint32_t>(std::max(1, dynamicGroupBudget)));
                visibilityHistoryResetRequested = true;
            }
            ImGui::TextDisabled(
                "Manual edits switch the current scene to Custom and are restored on scene switch.");
            if (ImGui::Button("Reset Scene Budget to Auto")) {
                clusterStreamingService.setBudgetPreset(ClusterStreamingService::BudgetPreset::Auto);
                refreshClusterStreamingMemoryBudget();
                visibilityHistoryResetRequested = true;
            }
            ImGui::SameLine();
            if (ImGui::Button("Refresh VRAM Budget")) {
                refreshClusterStreamingMemoryBudget();
            }
            if (memoryBudgetInfo.available) {
                const std::string deviceLocalBudgetLabel =
                    formatByteCountShort(memoryBudgetInfo.deviceLocalUsageBytes) + " / " +
                    formatByteCountShort(memoryBudgetInfo.deviceLocalBudgetBytes);
                const float deviceLocalBudgetRatio =
                    memoryBudgetInfo.deviceLocalBudgetBytes != 0u
                        ? float(double(memoryBudgetInfo.deviceLocalUsageBytes) /
                                double(memoryBudgetInfo.deviceLocalBudgetBytes))
                        : 0.0f;
                ImGui::Text("Device-local VRAM");
                ImGui::ProgressBar(
                    deviceLocalBudgetRatio, ImVec2(-1.0f, 0.0f), deviceLocalBudgetLabel.c_str());
                ImGui::Text("VRAM headroom: %s, auto pool target: %s",
                            formatByteCountShort(memoryBudgetInfo.deviceLocalHeadroomBytes).c_str(),
                            formatByteCountShort(memoryBudgetInfo.targetStorageBytes).c_str());
            } else {
                ImGui::TextDisabled("VRAM budget: VK_EXT_memory_budget unavailable");
            }
            bool gpuStatsReadbackEnabled = clusterStreamingService.gpuStatsReadbackEnabled();
            if (ImGui::Checkbox("GPU Stats Readback", &gpuStatsReadbackEnabled)) {
                clusterStreamingService.setGpuStatsReadbackEnabled(gpuStatsReadbackEnabled);
            }
            bool adaptiveBudgetEnabled = clusterStreamingService.adaptiveBudgetEnabled();
            if (ImGui::Checkbox("Adaptive Unload Age", &adaptiveBudgetEnabled)) {
                clusterStreamingService.setAdaptiveBudgetEnabled(adaptiveBudgetEnabled);
            }
            ImGui::Text("Unload age: effective %u, base %u, adjustments %u",
                        streamingTelemetry.effectiveAgeThreshold,
                        streamingTelemetry.configuredAgeThreshold,
                        streamingTelemetry.adaptiveBudgetAdjustmentCount);
            ImGui::Text("Adaptive signals: failed %.2f, pool %.1f%%",
                        streamingTelemetry.smoothedFailedAllocations,
                        streamingTelemetry.smoothedStorageUtilization * 100.0f);

            const float residentGroupRatio =
                streamingStats.activeResidencyGroupCount != 0u
                    ? float(double(streamingTelemetry.residentGroupCount) /
                            double(streamingStats.activeResidencyGroupCount))
                    : 0.0f;
            const std::string residentGroupLabel =
                std::to_string(streamingTelemetry.residentGroupCount) + " / " +
                std::to_string(streamingStats.activeResidencyGroupCount) +
                " groups";
            ImGui::Text("Resident groups");
            ImGui::ProgressBar(residentGroupRatio, ImVec2(-1.0f, 0.0f), residentGroupLabel.c_str());
            ImGui::Text("%u always-resident, %u dynamic",
                        streamingTelemetry.alwaysResidentGroupCount,
                        streamingTelemetry.dynamicResidentGroupCount);

            const float storagePoolRatio =
                streamingTelemetry.storagePoolCapacityBytes != 0u
                    ? float(double(streamingTelemetry.storagePoolUsedBytes) /
                            double(streamingTelemetry.storagePoolCapacityBytes))
                    : 0.0f;
            const std::string storagePoolLabel =
                formatByteCountShort(streamingTelemetry.storagePoolUsedBytes) + " / " +
                formatByteCountShort(streamingTelemetry.storagePoolCapacityBytes);
            ImGui::Text("Storage pool usage");
            ImGui::ProgressBar(storagePoolRatio, ImVec2(-1.0f, 0.0f), storagePoolLabel.c_str());

            const float transferRatio =
                std::clamp(streamingTelemetry.transferUtilization, 0.0f, 1.0f);
            const std::string transferLabel =
                formatByteCountShort(streamingTelemetry.transferBytesThisFrame) + " / " +
                formatByteCountShort(clusterStreamingService.effectiveStreamingTransferCapacityBytes());
            ImGui::Text("Transfer bandwidth");
            ImGui::ProgressBar(transferRatio, ImVec2(-1.0f, 0.0f), transferLabel.c_str());
            ImGui::Text("Transfer utilization: %.1f%%",
                        streamingTelemetry.transferUtilization * 100.0f);
            if (streamingTelemetry.cpuUnloadFallbackActive ||
                streamingTelemetry.graphicsTransferFallbackActive) {
                ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.25f, 1.0f),
                                   "Fallbacks: CPU FIFO unload %s%s",
                                   streamingTelemetry.cpuUnloadFallbackActive ? "active" : "idle",
                                   streamingTelemetry.graphicsTransferFallbackActive
                                       ? ", graphics copy path"
                                       : "");
                if (streamingTelemetry.cpuUnloadFallbackActive) {
                    ImGui::TextDisabled("CPU fallback frame %u, queued %u unloads",
                                        streamingTelemetry.cpuUnloadFallbackFrameIndex,
                                        streamingTelemetry.cpuUnloadFallbackGroupCount);
                }
            } else if (streamingTelemetry.gpuAgeFilterDispatchMissing) {
                ImGui::TextDisabled("Fallbacks: age filter missing, CPU FIFO standby (frame %u)",
                                    streamingTelemetry.gpuAgeFilterDispatchMissingFrameIndex);
            }

            ImGui::Text("Requests: load %u (%u executed, %u deferred), unload %u (%u executed)",
                        streamingTelemetry.loadRequestsThisFrame,
                        streamingTelemetry.loadsExecutedThisFrame,
                        streamingTelemetry.loadsDeferredThisFrame,
                        streamingTelemetry.unloadRequestsThisFrame,
                        streamingTelemetry.unloadsExecutedThisFrame);
            ImGui::Text("Pending load/unload: %u / %u (%u confirmed)",
                        streamingStats.pendingResidencyGroupCount,
                        streamingStats.pendingUnloadGroupCount,
                        streamingStats.confirmedUnloadGroupCount);
            ImGui::Text("Failed allocations this frame: %u",
                        streamingTelemetry.failedAllocations);
            if (!clusterStreamingService.gpuStatsReadbackEnabled()) {
                ImGui::TextDisabled("GPU stats: disabled");
            } else if (streamingTelemetry.gpuStatsValid) {
                const std::string gpuCopiedBytesLabel =
                    formatByteCountShort(streamingTelemetry.gpuCopiedBytes);
                ImGui::Text("GPU stats: frame %u, patches %u, copied %s",
                            streamingTelemetry.gpuStatsFrameIndex,
                            streamingTelemetry.gpuAppliedPatchCount,
                            gpuCopiedBytesLabel.c_str());
                ImGui::Text("GPU unloads: %u, average age %.1f",
                            streamingTelemetry.gpuUnloadRequestCount,
                            streamingTelemetry.gpuAverageUnloadAge);
                if (streamingTelemetry.gpuErrorUpdateCount != 0u ||
                    streamingTelemetry.gpuErrorAgeFilterCount != 0u ||
                    streamingTelemetry.gpuErrorAllocationCount != 0u ||
                    streamingTelemetry.gpuErrorPageTableCount != 0u) {
                    ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.2f, 1.0f),
                                       "GPU errors: update %u, age %u, alloc %u, page %u",
                                       streamingTelemetry.gpuErrorUpdateCount,
                                       streamingTelemetry.gpuErrorAgeFilterCount,
                                       streamingTelemetry.gpuErrorAllocationCount,
                                       streamingTelemetry.gpuErrorPageTableCount);
                }
            } else {
                ImGui::TextDisabled("GPU stats: pending readback");
            }

            const float requestHistoryMax = streamingHistory.maxRequestValue();
            ImGui::PlotLines("Load Requests / Frame",
                             streamingHistory.loadRequestHistory.data(),
                             streamingHistory.sampleCount,
                             streamingHistory.plotOffset(),
                             nullptr,
                             0.0f,
                             requestHistoryMax,
                             ImVec2(0.0f, 60.0f));
            ImGui::PlotLines("Unload Requests / Frame",
                             streamingHistory.unloadRequestHistory.data(),
                             streamingHistory.sampleCount,
                             streamingHistory.plotOffset(),
                             nullptr,
                             0.0f,
                             requestHistoryMax,
                             ImVec2(0.0f, 60.0f));

            std::array<float, ClusterStreamingService::kStreamingAgeHistogramBucketCount>
                ageHistogramValues = {};
            float ageHistogramMax = 1.0f;
            for (size_t bucketIndex = 0; bucketIndex < ageHistogramValues.size(); ++bucketIndex) {
                ageHistogramValues[bucketIndex] =
                    static_cast<float>(streamingTelemetry.ageHistogram[bucketIndex]);
                ageHistogramMax = std::max(ageHistogramMax, ageHistogramValues[bucketIndex]);
            }
            const std::string ageHistogramOverlay =
                "bucket width " + std::to_string(streamingTelemetry.ageHistogramBucketWidth) +
                "f, max age " + std::to_string(streamingTelemetry.ageHistogramMaxAge) + "f";
            ImGui::PlotHistogram("Resident Age Histogram",
                                 ageHistogramValues.data(),
                                 static_cast<int>(ageHistogramValues.size()),
                                 0,
                                 ageHistogramOverlay.c_str(),
                                 0.0f,
                                 ageHistogramMax,
                                 ImVec2(0.0f, 80.0f));

            if (!streamingTelemetry.totalGroupsPerLod.empty()) {
                ImGui::Text("Per-LOD residency");
                for (size_t lodIndex = 0; lodIndex < streamingTelemetry.totalGroupsPerLod.size();
                     ++lodIndex) {
                    const uint32_t totalGroups = streamingTelemetry.totalGroupsPerLod[lodIndex];
                    const uint32_t residentGroups =
                        lodIndex < streamingTelemetry.residentGroupsPerLod.size()
                            ? streamingTelemetry.residentGroupsPerLod[lodIndex]
                            : 0u;
                    const float lodRatio =
                        totalGroups != 0u
                            ? float(double(residentGroups) / double(totalGroups))
                            : 0.0f;
                    const std::string lodLabel =
                        "LOD " + std::to_string(lodIndex) + ": " +
                        std::to_string(residentGroups) + " / " +
                        std::to_string(totalGroups) + " groups";
                    ImGui::ProgressBar(lodRatio, ImVec2(-1.0f, 0.0f), lodLabel.c_str());
                }
            } else {
                ImGui::TextDisabled("Per-LOD residency: unavailable");
            }

            ImGui::Separator();
            ImGui::Text("Resident heap: %u / %u clusters",
                        streamingStats.residentHeapUsed,
                        streamingStats.residentHeapCapacity);
            ImGui::Text("Task ring: %u total, %u free, %u prepared, %u transferred, %u queued",
                        streamingStats.streamingTaskCapacity,
                        streamingStats.freeStreamingTaskCount,
                        streamingStats.preparedStreamingTaskCount,
                        streamingStats.transferSubmittedTaskCount,
                        streamingStats.updateQueuedTaskCount);
            if (streamingStats.selectedTransferTaskIndex != UINT32_MAX) {
                ImGui::Text("Transfer task: slot %u, %s staged",
                            streamingStats.selectedTransferTaskIndex,
                            formatByteCountShort(streamingStats.selectedTransferBytes).c_str());
            } else {
                ImGui::TextDisabled("Transfer task: idle");
            }
            if (streamingStats.selectedUpdateTaskIndex != UINT32_MAX) {
                ImGui::Text("Update task: slot %u, %u patches, wait %llu",
                            streamingStats.selectedUpdateTaskIndex,
                            streamingStats.selectedUpdatePatchCount,
                            static_cast<unsigned long long>(
                                streamingStats.selectedUpdateTransferWaitValue));
            } else {
                ImGui::TextDisabled("Update task: idle");
            }
        }

        // LOD stats panel
        if (previewSceneReady && sceneCtx.clusterLod().lodLevelCount > 0) {
            drawClusterLODStats(sceneCtx.clusterLod());
        }

        ImGui::End();

        if (previewSceneReady && showSceneGraphWindow) {
            ::drawSceneGraphUI(sceneCtx.sceneGraph());
        }

        // --- Runtime scene loading UI ---
        {
            static char scenePathBuf[512] = {};
            if (ImGui::Begin("Scene Loader")) {
                ImGui::Text("Current: %s", sceneCtx.isSceneLoaded() ? sceneCtx.scene().filePath().c_str() : "(none)");
                ImGui::InputText("glTF Path", scenePathBuf, sizeof(scenePathBuf));
                ImGui::SameLine();
                if (ImGui::Button("Browse...")) {
#ifdef _WIN32
                    std::string picked = openGltfFileDialog(window);
                    if (!picked.empty()) {
                        std::snprintf(scenePathBuf, sizeof(scenePathBuf), "%s", picked.c_str());
                    }
#endif
                }
                if (ImGui::Button("Load Scene") && scenePathBuf[0] != '\0') {
                    rhi->waitIdle();
                    shadowResources.release();
                    rtShadowsAvailable = false;

                    if (sceneCtx.loadScene(scenePathBuf)) {
                        previewSceneReady = true;
                        rebuildRenderContext(renderContext);

                        if (!sceneCtx.materials().textureViews.empty()) {
                            descriptorBackend->updateBindlessSampledTextures(
                                sceneCtx.materials().textureViews.data(), 0,
                                static_cast<uint32_t>(sceneCtx.materials().textureViews.size()));
                        }
                        descriptorBackend->updateBindlessSampler(
                            METALLIC_BINDLESS_SCENE_SAMPLER_INDEX, &sceneCtx.materials().sampler);

                        if (rhi->features().rayTracing && enableRTShadows) {
                            if (buildAccelerationStructures(deviceHandle, queueHandle,
                                                            sceneCtx.mesh(), sceneCtx.sceneGraph(),
                                                            shadowResources) &&
                                createShadowPipeline(deviceHandle, shadowResources, PROJECT_SOURCE_DIR)) {
                                rtShadowsAvailable = true;
                            } else {
                                shadowResources.release();
                            }
                        }

                        previewCamera.initFromBounds(sceneCtx.mesh().bboxMin, sceneCtx.mesh().bboxMax);
                        previewCamera.distance *= 0.8f;
                        sunLight = sceneCtx.sceneGraph().getSunDirectionalLight();
                        refreshVisibilityPipelineState();
                        postBuilderNeedsRebuild = true;
                        visibilityHistoryResetRequested = true;
                        hasPrevMatrices = false;
                    } else {
                        previewSceneReady = false;
                    }
                }
            }
            ImGui::End();
        }

        refreshPreviewSceneState();

        const int renderWidth = useVisibilityRenderGraph ? runtimeContext.renderWidth : width;
        const int renderHeight = useVisibilityRenderGraph ? runtimeContext.renderHeight : height;
        const float aspect =
            static_cast<float>(width) / static_cast<float>(std::max(height, 1));
        const float3 sunDirection = normalize(sunLight.direction);
        const float4x4 view = previewCamera.viewMatrix();
        const float4x4 unjitteredProj = previewCamera.projectionMatrix(aspect);
        const float4 cameraWorldPos = orbitCameraWorldPosition(previewCamera);
        const float3 cameraWorldPos3(cameraWorldPos.x, cameraWorldPos.y, cameraWorldPos.z);
        const float3 cameraForward = normalize(previewCamera.target - cameraWorldPos3);
        const float3 worldUp(0.0f, 1.0f, 0.0f);
        const float3 cameraRight = normalize(cross(cameraForward, worldUp));
        const float3 cameraUp = cross(cameraRight, cameraForward);
        float4x4 proj = unjitteredProj;
        float2 jitterOffset = float2(0.0f, 0.0f);
        const bool enableVisibilityTAA =
            useVisibilityRenderGraph &&
            visibilityUpscalerMode == VisibilityUpscalerMode::TAA &&
            visibilityTaaAvailable;
        const bool enableVisibilityDlss =
            useVisibilityRenderGraph &&
            visibilityUpscalerMode == VisibilityUpscalerMode::DLSS &&
            runtimeContext.upscaler && runtimeContext.upscaler->isEnabled();
        const bool needsJitter = enableVisibilityTAA || enableVisibilityDlss;
        if (needsJitter) {
            jitterOffset = OrbitCamera::haltonJitter(frameIndex);
            proj = OrbitCamera::jitteredProjectionMatrix(previewCamera.fovY,
                                                         aspect,
                                                         previewCamera.nearZ,
                                                         previewCamera.farZ,
                                                         jitterOffset.x,
                                                         jitterOffset.y,
                                                         static_cast<uint32_t>(renderWidth),
                                                         static_cast<uint32_t>(renderHeight));
        }

        const bool gpuDrivenVisibilityPath =
            useVisibilityRenderGraph && visibilityGpuCullingAvailable;

        uint32_t visibilityInstanceCount = 0;
        if (useVisibilityRenderGraph) {
            static bool warnedInstanceOverflow = false;
            if (!warnedInstanceOverflow &&
                sceneCtx.gpuScene().instanceCount > (kVisibilityInstanceMask + 1u)) {
                spdlog::warn("GPU scene instance limit exceeded for visibility encoding ({} > {}), overflowing instances will be dropped in GPU visibility mode",
                             sceneCtx.gpuScene().instanceCount,
                             kVisibilityInstanceMask + 1);
                warnedInstanceOverflow = true;
            }

            if (!gpuDrivenVisibilityPath) {
                visibilityInstanceCount = static_cast<uint32_t>(
                    std::min<size_t>(previewVisibleMeshletNodes.size(),
                                     static_cast<size_t>(kVisibilityInstanceMask + 1u)));
            }
        }

        RhiNativeCommandBufferHandle nativeCommandBuffer(getVulkanCurrentCommandBuffer(*rhi));
        frameContext = FrameContext{};
        frameContext.width = renderWidth;
        frameContext.height = renderHeight;
        frameContext.view = view;
        frameContext.proj = proj;
        frameContext.unjitteredProj = unjitteredProj;
        frameContext.cameraWorldPos = cameraWorldPos;
        frameContext.cameraRight = float4(cameraRight.x, cameraRight.y, cameraRight.z, 0.0f);
        frameContext.cameraUp = float4(cameraUp.x, cameraUp.y, cameraUp.z, 0.0f);
        frameContext.cameraForward = float4(cameraForward.x, cameraForward.y, cameraForward.z, 0.0f);
        frameContext.cameraNearZ = previewCamera.nearZ;
        frameContext.cameraFovY = previewCamera.fovY;
        frameContext.prevView = hasPrevMatrices ? prevView : view;
        frameContext.prevProj = hasPrevMatrices ? prevProj : unjitteredProj;
        frameContext.prevCullView = hasPrevMatrices ? prevCullView : view;
        frameContext.prevCullProj = hasPrevMatrices ? prevCullProj : proj;
        frameContext.prevCameraWorldPos =
            hasPrevMatrices ? prevCameraWorldPos : frameContext.cameraWorldPos;
        frameContext.jitterOffset = jitterOffset;
        frameContext.frameIndex = frameIndex;
        frameContext.enableTAA = enableVisibilityTAA;
        frameContext.displayWidth = width;
        frameContext.displayHeight = height;
        frameContext.renderWidth = renderWidth;
        frameContext.renderHeight = renderHeight;
        frameContext.historyReset = visibilityHistoryResetRequested;
        frameContext.worldLightDir = float4(sunDirection.x, sunDirection.y, sunDirection.z, 0.0f);
        frameContext.viewLightDir = view * frameContext.worldLightDir;
        frameContext.lightColorIntensity =
            float4(sunLight.color.x, sunLight.color.y, sunLight.color.z, sunLight.intensity);
        frameContext.meshletCount = sceneCtx.meshlets().meshletCount;
        frameContext.materialCount = sceneCtx.materials().materialCount;
        frameContext.textureCount = static_cast<uint32_t>(sceneCtx.materials().textures.size());
        if (!gpuDrivenVisibilityPath) {
            frameContext.visibleMeshletNodes = previewVisibleMeshletNodes;
            if (useVisibilityRenderGraph &&
                frameContext.visibleMeshletNodes.size() > static_cast<size_t>(visibilityInstanceCount)) {
                frameContext.visibleMeshletNodes.resize(visibilityInstanceCount);
            }
            frameContext.visibleIndexNodes = previewVisibleIndexNodes;
        }
        frameContext.visibilityInstanceCount = visibilityInstanceCount;
        frameContext.depthClearValue = depthClearValue;
        frameContext.cameraFarZ = previewCamera.farZ;
        {
            const double now = glfwGetTime();
            frameContext.deltaTime = static_cast<float>(now - lastFrameTime);
            lastFrameTime = now;
        }
        frameContext.enableRTShadows =
            useVisibilityRenderGraph && rtShadowsAvailable && enableRTShadows;
        frameContext.enableAtmosphereSky = atmosphereSkyAvailable;
        frameContext.gpuDrivenCulling = gpuDrivenVisibilityPath;
        frameContext.renderMode = useVisibilityRenderGraph ? 2 : 0;

        postBuilder.updateFrame(&backbufferTexture, &frameContext);

        FrameGraph& activeFg = postBuilder.frameGraph();
        if (showGraphDebug) {
            ImGui::SetNextWindowDockID(dockspaceId, ImGuiCond_FirstUseEver);
            activeFg.debugImGui();
        }
        if (showRenderPassUI) {
            ImGui::SetNextWindowDockID(dockspaceId, ImGuiCond_FirstUseEver);
            activeFg.renderPassUI();
        }
        if (showImGuiDemo) {
            ImGui::SetNextWindowDockID(dockspaceId, ImGuiCond_FirstUseEver);
            ImGui::ShowDemoWindow(&showImGuiDemo);
        }
        ImGui::Render();

        if (!useVisibilityRenderGraph) {
            sceneGraph.execute(commandBuffer, frameGraphBackend);
        }

        if (useVisibilityRenderGraph && rtShadowsAvailable) {
            updateTLAS(nativeCommandBuffer, sceneCtx.sceneGraph(), shadowResources);
        }

        // Ensure sceneColorTexture is in SHADER_READ_ONLY_OPTIMAL for ImGui::Image sampling
        // The frame graph doesn't know about the ImGui descriptor reference, so we insert
        // a manual barrier. On the first frame (UNDEFINED), this transitions to readable state.
        if (sceneColorTexture.nativeHandle() && !useVisibilityRenderGraph) {
            VkImage sceneColorImage = getVulkanImage(&sceneColorTexture);
            if (sceneColorImage != VK_NULL_HANDLE) {
                VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                barrier.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
                                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                barrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                                        VK_ACCESS_2_SHADER_WRITE_BIT;
                barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
                barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                barrier.oldLayout = sceneColorLayout;
                barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                barrier.image = sceneColorImage;
                barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

                VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                depInfo.imageMemoryBarrierCount = 1;
                depInfo.pImageMemoryBarriers = &barrier;
                vkCmdPipelineBarrier2(nativeCmd, &depInfo);

                sceneColorLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                imageTracker.setLayout(sceneColorImage, sceneColorLayout);
            }
        }

        postBuilder.execute(commandBuffer, frameGraphBackend);

        // Blit backbuffer to viewport display texture for ImGui::Image
        if (viewportDisplayTexture.nativeHandle() && backbufferImage != VK_NULL_HANDLE) {
            VkImage vpImage = getVulkanImage(&viewportDisplayTexture);
            if (vpImage != VK_NULL_HANDLE) {
                uint32_t vpW = viewportDisplayTexture.width();
                uint32_t vpH = viewportDisplayTexture.height();

                VkImageMemoryBarrier2 barriers[2] = {};
                barriers[0] = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                barriers[0].srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
                barriers[0].srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
                barriers[0].dstStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
                barriers[0].dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
                barriers[0].oldLayout = imageTracker.getLayout(backbufferImage);
                barriers[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                barriers[0].image = backbufferImage;
                barriers[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

                barriers[1] = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                barriers[1].srcStageMask = VK_PIPELINE_STAGE_2_NONE;
                barriers[1].srcAccessMask = VK_ACCESS_2_NONE;
                barriers[1].dstStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
                barriers[1].dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                barriers[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                barriers[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                barriers[1].image = vpImage;
                barriers[1].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

                VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                dep.imageMemoryBarrierCount = 2;
                dep.pImageMemoryBarriers = barriers;
                vkCmdPipelineBarrier2(nativeCmd, &dep);

                VkImageBlit2 blitRegion{VK_STRUCTURE_TYPE_IMAGE_BLIT_2};
                blitRegion.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                blitRegion.srcOffsets[1] = {static_cast<int32_t>(width), static_cast<int32_t>(height), 1};
                blitRegion.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                blitRegion.dstOffsets[1] = {static_cast<int32_t>(vpW), static_cast<int32_t>(vpH), 1};

                VkBlitImageInfo2 blitInfo{VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2};
                blitInfo.srcImage = backbufferImage;
                blitInfo.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                blitInfo.dstImage = vpImage;
                blitInfo.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                blitInfo.regionCount = 1;
                blitInfo.pRegions = &blitRegion;
                blitInfo.filter = VK_FILTER_LINEAR;
                vkCmdBlitImage2(nativeCmd, &blitInfo);

                VkImageMemoryBarrier2 postBarriers[2] = {};
                postBarriers[0] = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                postBarriers[0].srcStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
                postBarriers[0].srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
                postBarriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
                postBarriers[0].dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
                postBarriers[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                postBarriers[0].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                postBarriers[0].image = backbufferImage;
                postBarriers[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

                postBarriers[1] = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                postBarriers[1].srcStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
                postBarriers[1].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                postBarriers[1].dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
                postBarriers[1].dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                postBarriers[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                postBarriers[1].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                postBarriers[1].image = vpImage;
                postBarriers[1].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

                VkDependencyInfo postDep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                postDep.imageMemoryBarrierCount = 2;
                postDep.pImageMemoryBarriers = postBarriers;
                vkCmdPipelineBarrier2(nativeCmd, &postDep);

                imageTracker.setLayout(backbufferImage, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            }
        }

        // Render ImGui directly to backbuffer (ImGuiOverlayPass removed from pipeline)
        {
            VkRenderingAttachmentInfo colorAttach{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            colorAttach.imageView = backbufferImageView;
            colorAttach.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            colorAttach.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            colorAttach.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

            VkRenderingInfo renderInfo{VK_STRUCTURE_TYPE_RENDERING_INFO};
            renderInfo.renderArea = {{0, 0}, backbufferExtent};
            renderInfo.layerCount = 1;
            renderInfo.colorAttachmentCount = 1;
            renderInfo.pColorAttachments = &colorAttach;

            vkCmdBeginRendering(nativeCmd, &renderInfo);
            ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), nativeCmd);
            vkCmdEndRendering(nativeCmd);
        }

        // After pipeline execution, sceneColorTexture is ready for ImGui sampling next frame
        if (sceneColorTexture.nativeHandle()) {
            sceneColorLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
        visibilityHistoryResetRequested = false;

        if (native.transferTimelineSemaphore != nullptr) {
            const uint64_t transferWaitValue =
                clusterStreamingService.consumePendingTransferWaitValue();
            if (transferWaitValue != 0u) {
                vulkanEnqueueGraphicsTimelineWait(
                    *rhi,
                    nativeToVkHandle<VkSemaphore>(native.transferTimelineSemaphore),
                    transferWaitValue,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
            }
        }

        prevView = view;
        prevProj = unjitteredProj;
        prevCullView = view;
        prevCullProj = proj;
        prevCameraWorldPos = frameContext.cameraWorldPos;
        hasPrevMatrices = true;
        frameIndex++;

        // If any pass routed work to the dedicated async compute queue, submit it now.
        if (commandBuffer.hadAsyncComputeWork()) {
            vulkanScheduleAsyncComputeSubmit(*rhi);
            if (vulkanIsDeviceLost(*rhi)) {
                spdlog::critical("Async compute submit reported device loss: {}",
                                 vulkanDeviceLostMessage(*rhi));
                break;
            }
        }

        rhi->endFrame();
        if (vulkanIsDeviceLost(*rhi)) {
            spdlog::critical("Graphics submit/present reported device loss: {}",
                             vulkanDeviceLostMessage(*rhi));
            break;
        }
        FrameMark;
    }

    cleanupRuntimeResources();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
