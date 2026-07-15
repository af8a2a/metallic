#include "Editor/EditorApplication.h"

#include "Runtime/Render/GAPI/Rhi.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/RenderSample.h"
#include "imnodes.h"
#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_vulkan.h"
#include "imgui_internal.h"

#include <SDL3/SDL.h>
#include <spdlog/spdlog.h>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include <cderr.h>
#include <commdlg.h>
#endif

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

namespace metallic {
namespace {

constexpr int kBaseWindowWidth = 1600;
constexpr int kBaseWindowHeight = 900;
constexpr uint32_t kMaxViewportPreviewSize = 2048;
constexpr uint32_t kViewportResizeSettleFrames = 3;
constexpr const char* kRenderPassDragPayload = "METALLIC_RENDER_PASS_TYPE";
constexpr uint32_t kSwapchainImageCount = 3;
constexpr uint32_t kMinSwapchainImageCount = 2;
constexpr int kNoViewportCameraDragButton = -1;
constexpr float kKeyboardMoveRate = 2.5f;
constexpr float kFastCameraMoveMultiplier = 5.0f;
constexpr float kSlowCameraMoveMultiplier = 0.1f;
constexpr float kMaxDollyDisplacement = 0.99f;
constexpr const char* kDefaultRenderSampleId = "pathtracing-sample";
constexpr const char* kDefaultImGuiIni = R"ini([Window][Viewport]
Pos=0,28
Size=1821,794
Collapsed=0
DockId=0x00000003,0

[Window][Scene Browser]
Pos=1824,28
Size=576,1322
Collapsed=0
DockId=0x00000002,0

[Window][Inspector]
Pos=1824,28
Size=576,1322
Collapsed=0
DockId=0x00000002,1

[Window][Assets]
Pos=0,825
Size=1821,525
Collapsed=0
DockId=0x00000004,0

[Window][Console]
Pos=0,825
Size=1821,525
Collapsed=0
DockId=0x00000004,1

[Window][MetallicEditorDockspace]
Pos=0,0
Size=2400,1350
Collapsed=0

[Window][Debug##Default]
Pos=60,60
Size=400,400
Collapsed=0

[Window][Render Graph Editor]
Pos=504,153
Size=1830,1140
Collapsed=0

[Window][Profiler]
Pos=0,825
Size=1821,525
Collapsed=0
DockId=0x00000004,3

[Window][NVML Monitor]
Pos=0,825
Size=1821,525
Collapsed=0
DockId=0x00000004,2

[Window][Statistics]
Pos=0,825
Size=1821,525
Collapsed=0
DockId=0x00000004,4

[Table][0x331D395F,5]
RefScale=20
Column 0  Weight=1.0000
Column 1  Width=92
Column 2  Width=72
Column 3  Width=72
Column 4  Width=72

[Docking][Data]
DockSpace     ID=0xB0446515 Window=0x3660BDC2 Pos=0,28 Size=2400,1322 Split=X
  DockNode    ID=0x00000001 Parent=0xB0446515 SizeRef=1821,1350 Split=Y
    DockNode  ID=0x00000003 Parent=0x00000001 SizeRef=1821,794 CentralNode=1 Selected=0xC450F867
    DockNode  ID=0x00000004 Parent=0x00000001 SizeRef=1821,525 Selected=0x9B5D3198
  DockNode    ID=0x00000002 Parent=0xB0446515 SizeRef=576,1350 Selected=0xE601B12F

)ini";

using StartupClock = std::chrono::steady_clock;

double elapsedMilliseconds(StartupClock::time_point begin)
{
    return std::chrono::duration<double, std::milli>(StartupClock::now() - begin).count();
}

class StartupLogScope {
public:
    explicit StartupLogScope(std::string label)
        : label_(std::move(label))
    {
        spdlog::info("[Startup] Begin {}", label_);
    }

    ~StartupLogScope()
    {
        spdlog::info("[Startup] End {} in {:.2f} ms", label_, elapsedMilliseconds(begin_));
    }

private:
    std::string label_;
    StartupClock::time_point begin_ = StartupClock::now();
};

float getMainDisplayScale()
{
    const SDL_DisplayID display = SDL_GetPrimaryDisplay();
    const float scale = display != 0 ? SDL_GetDisplayContentScale(display) : 1.0f;
    return std::max(scale, 1.0f);
}

std::filesystem::path resolveGraphAssetPath(const char* path)
{
    std::filesystem::path assetPath(path == nullptr ? "" : path);
    if (assetPath.empty()) {
        assetPath = "Pipelines/default.metallic_graph.json";
    }
    if (assetPath.is_relative()) {
        assetPath = std::filesystem::path(PROJECT_SOURCE_DIR) / assetPath;
    }
    return assetPath;
}

std::filesystem::path resolveSceneAssetPath(const char* path)
{
    std::filesystem::path assetPath(path == nullptr ? "" : path);
    if (assetPath.empty()) {
        return {};
    }
    if (assetPath.is_relative()) {
        assetPath = std::filesystem::path(PROJECT_SOURCE_DIR) / assetPath;
    }
    return assetPath;
}

std::string lowerPathExtension(const std::filesystem::path& path)
{
    std::string extension = path.extension().string();
    std::transform(
        extension.begin(),
        extension.end(),
        extension.begin(),
        [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    return extension;
}

bool isSceneFilePath(const std::filesystem::path& path)
{
    const std::string extension = lowerPathExtension(path);
    return extension == ".gltf" || extension == ".glb";
}

bool isRenderGraphFilePath(const std::filesystem::path& path)
{
    return lowerPathExtension(path) == ".json" &&
        path.filename().string().find(".metallic_graph") != std::string::npos;
}

bool isSceneAwareRenderPassType(const std::string& type)
{
    return type == "BunnyWireframePass" ||
        type == "GPUDrivenPreviewPass" ||
        type == "GPUDrivenStreamAssetPass" ||
        type == "SceneRayQueryVisualizationPass" ||
        type == "SceneMaterialVisualizationPass" ||
        type == "ScenePathTracePass";
}

std::string firstScenePathFromGraph(const render::RenderGraph& graph)
{
    for (const render::RenderGraphNode& node : graph.nodes()) {
        if (!isSceneAwareRenderPassType(node.type) || !node.properties.is_object()) {
            continue;
        }
        auto pathIter = node.properties.find("path");
        if (pathIter != node.properties.end() && pathIter->is_string()) {
            return pathIter->get<std::string>();
        }
    }
    return {};
}

std::string firstEnvironmentPathFromGraph(const render::RenderGraph& graph)
{
    for (const render::RenderGraphNode& node : graph.nodes()) {
        if (node.type != "ScenePathTracePass" || !node.properties.is_object()) {
            continue;
        }
        auto environmentIter = node.properties.find("environment");
        if (environmentIter == node.properties.end() || !environmentIter->is_object()) {
            continue;
        }
        auto pathIter = environmentIter->find("path");
        if (pathIter != environmentIter->end() && pathIter->is_string()) {
            return pathIter->get<std::string>();
        }
    }
    return {};
}

std::string displayPathForProperty(const std::filesystem::path& path)
{
    std::error_code relativeError;
    const std::filesystem::path project = std::filesystem::path(PROJECT_SOURCE_DIR);
    const std::filesystem::path relative = std::filesystem::relative(path, project, relativeError);
    const std::string genericRelative = relative.generic_string();
    if (!relativeError &&
        !genericRelative.empty() &&
        genericRelative != "." &&
        genericRelative.rfind("../", 0) != 0 &&
        genericRelative.rfind("..\\", 0) != 0 &&
        genericRelative != "..") {
        return genericRelative;
    }
    return path.string();
}

#if defined(_WIN32)
void copyWideToBuffer(const std::wstring& value, wchar_t* buffer, size_t bufferSize)
{
    if (buffer == nullptr || bufferSize == 0) {
        return;
    }
    std::fill(buffer, buffer + bufferSize, L'\0');
    const size_t copySize = std::min(value.size(), bufferSize - 1);
    std::copy_n(value.data(), copySize, buffer);
}

std::filesystem::path normalizeDialogPath(const std::filesystem::path& path)
{
    if (path.empty()) {
        return {};
    }

    std::error_code error;
    std::filesystem::path normalized = path.is_relative()
        ? std::filesystem::absolute(path, error)
        : path;
    if (error) {
        normalized = path;
    }

    std::filesystem::path canonical = std::filesystem::weakly_canonical(normalized, error);
    if (!error && !canonical.empty()) {
        normalized = canonical;
    }

    normalized.make_preferred();
    return normalized;
}

const char* commonDialogErrorName(DWORD error)
{
    switch (error) {
    case CDERR_DIALOGFAILURE:
        return "CDERR_DIALOGFAILURE";
    case CDERR_STRUCTSIZE:
        return "CDERR_STRUCTSIZE";
    case CDERR_INITIALIZATION:
        return "CDERR_INITIALIZATION";
    case CDERR_NOTEMPLATE:
        return "CDERR_NOTEMPLATE";
    case CDERR_NOHINSTANCE:
        return "CDERR_NOHINSTANCE";
    case CDERR_LOADSTRFAILURE:
        return "CDERR_LOADSTRFAILURE";
    case CDERR_FINDRESFAILURE:
        return "CDERR_FINDRESFAILURE";
    case CDERR_LOADRESFAILURE:
        return "CDERR_LOADRESFAILURE";
    case CDERR_LOCKRESFAILURE:
        return "CDERR_LOCKRESFAILURE";
    case CDERR_MEMALLOCFAILURE:
        return "CDERR_MEMALLOCFAILURE";
    case CDERR_MEMLOCKFAILURE:
        return "CDERR_MEMLOCKFAILURE";
    case CDERR_NOHOOK:
        return "CDERR_NOHOOK";
    case CDERR_REGISTERMSGFAIL:
        return "CDERR_REGISTERMSGFAIL";
    case FNERR_SUBCLASSFAILURE:
        return "FNERR_SUBCLASSFAILURE";
    case FNERR_INVALIDFILENAME:
        return "FNERR_INVALIDFILENAME";
    case FNERR_BUFFERTOOSMALL:
        return "FNERR_BUFFERTOOSMALL";
    default:
        return "unknown";
    }
}
#endif

std::filesystem::path openSceneFileDialog(
    SDL_Window* window,
    const std::filesystem::path& initialPath,
    std::string& error)
{
#if defined(_WIN32)
    HWND owner = nullptr;
    if (window != nullptr) {
        const SDL_PropertiesID properties = SDL_GetWindowProperties(window);
        owner = static_cast<HWND>(
            SDL_GetPointerProperty(properties, SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr));
    }

    std::array<wchar_t, 32768> filename{};
    std::array<wchar_t, 32768> initialDirectory{};
    if (!initialPath.empty()) {
        const std::filesystem::path dialogInitialPath = normalizeDialogPath(initialPath);
        if (dialogInitialPath.has_extension()) {
            copyWideToBuffer(dialogInitialPath.filename().wstring(), filename.data(), filename.size());
            copyWideToBuffer(
                dialogInitialPath.parent_path().wstring(),
                initialDirectory.data(),
                initialDirectory.size());
        } else {
            copyWideToBuffer(dialogInitialPath.wstring(), initialDirectory.data(), initialDirectory.size());
        }
    }

    OPENFILENAMEW openFilename{};
    openFilename.lStructSize = sizeof(openFilename);
    openFilename.hwndOwner = owner;
    openFilename.lpstrFilter =
        L"3D Scene Files (*.gltf;*.glb)\0*.gltf;*.glb\0"
        L"glTF Text (*.gltf)\0*.gltf\0"
        L"glTF Binary (*.glb)\0*.glb\0"
        L"All Files (*.*)\0*.*\0";
    openFilename.nFilterIndex = 1;
    openFilename.lpstrFile = filename.data();
    openFilename.nMaxFile = static_cast<DWORD>(filename.size());
    openFilename.lpstrInitialDir = initialDirectory[0] != L'\0' ? initialDirectory.data() : nullptr;
    openFilename.lpstrTitle = L"Load 3D Scene";
    openFilename.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;

    if (GetOpenFileNameW(&openFilename) != FALSE) {
        error.clear();
        return std::filesystem::path(filename.data());
    }

    const DWORD dialogError = CommDlgExtendedError();
    if (dialogError != 0) {
        error = "Open scene dialog failed: " + std::string(commonDialogErrorName(dialogError)) +
            " (" + std::to_string(dialogError) + ")";
    }
    return {};
#else
    (void)window;
    (void)initialPath;
    error = "Native scene file dialog is only implemented on Windows.";
    return {};
#endif
}

std::filesystem::path openEnvironmentFileDialog(
    SDL_Window* window,
    const std::filesystem::path& initialPath,
    std::string& error)
{
#if defined(_WIN32)
    HWND owner = nullptr;
    if (window != nullptr) {
        const SDL_PropertiesID properties = SDL_GetWindowProperties(window);
        owner = static_cast<HWND>(
            SDL_GetPointerProperty(properties, SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr));
    }

    std::array<wchar_t, 32768> filename{};
    std::array<wchar_t, 32768> initialDirectory{};
    if (!initialPath.empty()) {
        const std::filesystem::path dialogInitialPath = normalizeDialogPath(initialPath);
        if (dialogInitialPath.has_extension()) {
            copyWideToBuffer(dialogInitialPath.filename().wstring(), filename.data(), filename.size());
            copyWideToBuffer(
                dialogInitialPath.parent_path().wstring(),
                initialDirectory.data(),
                initialDirectory.size());
        } else {
            copyWideToBuffer(dialogInitialPath.wstring(), initialDirectory.data(), initialDirectory.size());
        }
    }

    OPENFILENAMEW openFilename{};
    openFilename.lStructSize = sizeof(openFilename);
    openFilename.hwndOwner = owner;
    openFilename.lpstrFilter =
        L"HDR Environment (*.hdr)\0*.hdr\0"
        L"All Files (*.*)\0*.*\0";
    openFilename.nFilterIndex = 1;
    openFilename.lpstrFile = filename.data();
    openFilename.nMaxFile = static_cast<DWORD>(filename.size());
    openFilename.lpstrInitialDir = initialDirectory[0] != L'\0' ? initialDirectory.data() : nullptr;
    openFilename.lpstrTitle = L"Load Environment";
    openFilename.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;

    if (GetOpenFileNameW(&openFilename) != FALSE) {
        error.clear();
        return std::filesystem::path(filename.data());
    }

    const DWORD dialogError = CommDlgExtendedError();
    if (dialogError != 0) {
        error = "Open environment dialog failed: " + std::string(commonDialogErrorName(dialogError)) +
            " (" + std::to_string(dialogError) + ")";
    }
    return {};
#else
    (void)window;
    (void)initialPath;
    error = "Native environment file dialog is only implemented on Windows.";
    return {};
#endif
}

std::string makeUniqueNodeName(const render::RenderGraph& graph, const std::string& type)
{
    std::string base = type;
    constexpr const char* suffix = "Pass";
    if (base.size() > std::strlen(suffix) &&
        base.compare(base.size() - std::strlen(suffix), std::strlen(suffix), suffix) == 0) {
        base.resize(base.size() - std::strlen(suffix));
    }
    if (base.empty()) {
        base = "Pass";
    }

    std::string name = base;
    uint32_t index = 1;
    while (graph.findNode(name) != nullptr) {
        name = base + std::to_string(++index);
    }
    return name;
}

render::RenderGraphProperties defaultPropertiesForPass(const std::string& type)
{
    if (type == "ClearColorPass") {
        return render::RenderGraphProperties{
            {"color", {0.04f, 0.06f, 0.09f, 1.0f}},
        };
    }
    if (type == "BunnyWireframePass") {
        return render::RenderGraphProperties{
            {"path", "Asset/StandfordBunny/scene.gltf"},
            {"camera", {
                {"projection", "perspective"},
                {"fovDegrees", 60.0f},
                {"znear", 0.1f},
                {"zfar", 10000.0f},
                {"reversedZ", true},
                {"eye", {-0.0168404f, 0.110154f, 0.22f}},
                {"center", {-0.0168404f, 0.110154f, -0.00153695f}},
                {"up", {0.0f, 1.0f, 0.0f}},
            }},
        };
    }
    if (type == "SceneRayQueryVisualizationPass") {
        return render::RenderGraphProperties{
            {"path", "Asset/StandfordBunny/scene.gltf"},
            {"granularity", "instance"},
            {"camera", {
                {"projection", "perspective"},
                {"fovDegrees", 60.0f},
                {"znear", 0.1f},
                {"zfar", 10000.0f},
                {"reversedZ", true},
                {"eye", {-0.0168404f, 0.110154f, 0.22f}},
                {"center", {-0.0168404f, 0.110154f, -0.00153695f}},
                {"up", {0.0f, 1.0f, 0.0f}},
            }},
        };
    }
    if (type == "SceneMaterialVisualizationPass") {
        return render::RenderGraphProperties{
            {"path", "Asset/ABeautifulGame/glTF/ABeautifulGame.gltf"},
            {"mode", "material"},
            {"camera", {
                {"projection", "perspective"},
                {"fovDegrees", 45.0f},
                {"znear", 0.001f},
                {"zfar", 10000.0f},
                {"reversedZ", true},
                {"eye", {0.0f, 0.42f, 1.15f}},
                {"center", {0.0f, 0.075f, 0.0f}},
                {"up", {0.0f, 1.0f, 0.0f}},
            }},
        };
    }
    if (type == "GPUDrivenPreviewPass") {
        return render::RenderGraphProperties{
            {"path", "Asset/SuperSponza/NewSponza_Main_glTF_003.gltf"},
            {"mode", "meshlet"},
            {"camera", {
                {"projection", "perspective"},
                {"fovDegrees", 60.0f},
                {"znear", 0.1f},
                {"zfar", 10000.0f},
                {"reversedZ", true},
                {"eye", {0.0f, 2.0f, 8.0f}},
                {"center", {0.0f, 1.0f, 0.0f}},
                {"up", {0.0f, 1.0f, 0.0f}},
            }},
        };
    }
    if (type == "GPUDrivenStreamAssetPass") {
        return render::RenderGraphProperties{
            {"path", "Asset/SuperSponza/NewSponza_Main_glTF_003.gltf"},
            {"autoBuildStreamAsset", false},
            {"enableGpuLodSelection", true},
            {"selectedLodLevel", 0},
            {"maxResidentPages", 4096},
            {"maxPageUploadsPerFrame", 64},
            {"pageLoadWorkerCount", 2},
            {"maxPageLoadsInFlight", 128},
            {"maxActiveGroups", 262144},
            {"maxTraversalWorkers", 1024},
            {"maxTraversalWorkItems", 1048576},
            {"debugColorMode", "page"},
            {"camera", {
                {"projection", "perspective"},
                {"fovDegrees", 60.0f},
                {"znear", 0.1f},
                {"zfar", 10000.0f},
                {"reversedZ", true},
                {"eye", {0.0f, 2.0f, 8.0f}},
                {"center", {0.0f, 1.0f, 0.0f}},
                {"up", {0.0f, 1.0f, 0.0f}},
            }},
        };
    }
    if (type == "ScenePathTracePass") {
        return render::RenderGraphProperties{
            {"path", "Asset/ABeautifulGame/glTF/ABeautifulGame.gltf"},
            {"bsdf", "openpbr"},
            {"maxDepth", 12},
            {"samples", 2},
            {"accumulate", true},
            {"environment", {
                {"enabled", true},
                {"visible", true},
                {"path", "Asset/ABeautifulGame/environment.hdr"},
                {"intensity", 1.0f},
                {"rotationDegrees", 0.0f},
            }},
            {"camera", {
                {"projection", "perspective"},
                {"fovDegrees", 45.0f},
                {"znear", 0.001f},
                {"zfar", 10000.0f},
                {"reversedZ", true},
                {"eye", {0.0f, 0.42f, 1.15f}},
                {"center", {0.0f, 0.075f, 0.0f}},
                {"up", {0.0f, 1.0f, 0.0f}},
            }},
        };
    }
    if (type == "NrdDenoisePass") {
        return render::RenderGraphProperties{
            {"denoiser", "REBLUR"},
            {"enableValidation", true},
            {"resetSerial", 0},
        };
    }
    return render::RenderGraphProperties::object();
}

render::RenderGraphNode* findBunnyWireframeNode(render::RenderGraph& graph)
{
    for (const render::RenderGraphNode& node : graph.nodes()) {
        if (isSceneAwareRenderPassType(node.type)) {
            return graph.findNode(node.id);
        }
    }
    return nullptr;
}

bool isVec3Property(const render::RenderGraphProperties& value)
{
    if (!value.is_array() || value.size() < 3) {
        return false;
    }
    return value[0].is_number() && value[1].is_number() && value[2].is_number();
}

void storeVec3Property(render::RenderGraphProperties& object, const char* key, const float values[3])
{
    object[key] = {values[0], values[1], values[2]};
}

void storeVec3Property(render::RenderGraphProperties& object, const char* key, const float3& values)
{
    object[key] = {values.x, values.y, values.z};
}

void readVec3Property(const render::RenderGraphProperties& object, const char* key, float outValues[3])
{
    const render::RenderGraphProperties& value = object.at(key);
    outValues[0] = value[0].get<float>();
    outValues[1] = value[1].get<float>();
    outValues[2] = value[2].get<float>();
}

float3 readVec3Property3(const render::RenderGraphProperties& object, const char* key)
{
    float values[3] = {};
    readVec3Property(object, key, values);
    return float3(values[0], values[1], values[2]);
}

void cameraDefaultsFromBounds(const scene::Bounds& bounds, float eye[3], float center[3])
{
    if (bounds.valid) {
        const float3 boundsCenter = bounds.center();
        const float distance = std::max(bounds.radius() * 2.0f, 0.1f);
        center[0] = boundsCenter.x;
        center[1] = boundsCenter.y;
        center[2] = boundsCenter.z;
        eye[0] = boundsCenter.x;
        eye[1] = boundsCenter.y;
        eye[2] = boundsCenter.z + distance;
        return;
    }

    center[0] = -0.0168404f;
    center[1] = 0.110154f;
    center[2] = -0.00153695f;
    eye[0] = center[0];
    eye[1] = center[1];
    eye[2] = 0.22f;
}

void ensureFloatProperty(
    render::RenderGraphProperties& object,
    const char* key,
    float fallback)
{
    auto iter = object.find(key);
    if (iter == object.end() || !iter->is_number()) {
        object[key] = fallback;
    }
}

void ensureBoolProperty(
    render::RenderGraphProperties& object,
    const char* key,
    bool fallback)
{
    auto iter = object.find(key);
    if (iter == object.end() || !iter->is_boolean()) {
        object[key] = fallback;
    }
}

void ensureVec3Property(
    render::RenderGraphProperties& object,
    const char* key,
    const float fallback[3])
{
    auto iter = object.find(key);
    if (iter == object.end() || !isVec3Property(*iter)) {
        storeVec3Property(object, key, fallback);
    }
}

void ensureCameraProperties(render::RenderGraphProperties& properties, const scene::Bounds& bounds)
{
    if (!properties.is_object()) {
        properties = render::RenderGraphProperties::object();
    }

    render::RenderGraphProperties& camera = properties["camera"];
    if (!camera.is_object()) {
        camera = render::RenderGraphProperties::object();
    }

    float defaultEye[3] = {};
    float defaultCenter[3] = {};
    cameraDefaultsFromBounds(bounds, defaultEye, defaultCenter);
    const float defaultUp[3] = {0.0f, 1.0f, 0.0f};

    auto projectionIter = camera.find("projection");
    if (projectionIter == camera.end() || !projectionIter->is_string() ||
        (*projectionIter != "perspective" && *projectionIter != "orthographic")) {
        camera["projection"] = "perspective";
    }
    ensureFloatProperty(camera, "fovDegrees", 60.0f);
    ensureFloatProperty(camera, "znear", 0.1f);
    ensureFloatProperty(camera, "zfar", 10000.0f);
    ensureBoolProperty(camera, "reversedZ", true);
    ensureVec3Property(camera, "eye", defaultEye);
    ensureVec3Property(camera, "center", defaultCenter);
    ensureVec3Property(camera, "up", defaultUp);
}

float propertyFloatOr(const render::RenderGraphProperties& object, const char* key, float fallback)
{
    auto iter = object.find(key);
    if (iter == object.end() || !iter->is_number()) {
        return fallback;
    }
    const float value = iter->get<float>();
    return std::isfinite(value) ? value : fallback;
}

float3 normalizedOr(const float3& value, const float3& fallback)
{
    const float len = length(value);
    if (len <= 0.000001f || !std::isfinite(len)) {
        return fallback;
    }
    return value / len;
}

float3 perpendicularTo(const float3& axis)
{
    const float3 reference = std::abs(axis.x) < 0.9f
        ? float3(1.0f, 0.0f, 0.0f)
        : float3(0.0f, 0.0f, 1.0f);
    return normalizedOr(reference - axis * dot(reference, axis), float3(1.0f, 0.0f, 0.0f));
}

struct CameraFrame {
    float3 eye{0.0f};
    float3 center{0.0f};
    float3 up{0.0f, 1.0f, 0.0f};
    float3 forward{0.0f, 0.0f, -1.0f};
    float3 right{1.0f, 0.0f, 0.0f};
    float3 viewUp{0.0f, 1.0f, 0.0f};
    float distance = 1.0f;
};

struct CameraViewDimensions {
    float width = 1.0f;
    float height = 1.0f;
};

float cameraDefaultOrthoHeight(const render::RenderGraphProperties& camera, float distance)
{
    constexpr float kPi = 3.14159265358979323846f;
    const float fovRadians = std::clamp(
        propertyFloatOr(camera, "fovDegrees", 60.0f),
        1.0f,
        179.0f) * (kPi / 180.0f);
    return std::max(2.0f * std::max(distance, 0.001f) * std::tan(fovRadians * 0.5f), 0.0001f);
}

float cameraOrthoHeight(const render::RenderGraphProperties& camera, float distance)
{
    return std::max(
        propertyFloatOr(camera, "orthoHeight", cameraDefaultOrthoHeight(camera, distance)),
        0.0001f);
}

CameraFrame cameraFrameFrom(const render::RenderGraphProperties& camera)
{
    CameraFrame frame;
    frame.eye = readVec3Property3(camera, "eye");
    frame.center = readVec3Property3(camera, "center");
    frame.up = normalizedOr(readVec3Property3(camera, "up"), float3(0.0f, 1.0f, 0.0f));
    const float3 view = frame.center - frame.eye;
    frame.distance = length(view);
    frame.forward = normalizedOr(view, float3(0.0f, 0.0f, -1.0f));
    frame.right = normalizedOr(cross(frame.forward, frame.up), perpendicularTo(frame.forward));
    frame.viewUp = normalizedOr(cross(frame.right, frame.forward), frame.up);
    return frame;
}

CameraViewDimensions cameraViewDimensions(
    const render::RenderGraphProperties& camera,
    const CameraFrame& frame,
    float viewportWidth,
    float viewportHeight)
{
    const float aspect = std::max(viewportWidth / std::max(viewportHeight, 1.0f), 0.001f);
    const std::string projection = camera["projection"].get<std::string>();
    const float height = projection == "orthographic"
        ? cameraOrthoHeight(camera, frame.distance)
        : cameraDefaultOrthoHeight(camera, frame.distance);
    return CameraViewDimensions{
        .width = std::max(height * aspect, 0.0001f),
        .height = height,
    };
}

bool translateCamera(render::RenderGraphProperties& camera, const float3& offset)
{
    const float amount = length(offset);
    if (amount <= 0.000001f || !std::isfinite(amount)) {
        return false;
    }

    CameraFrame frame = cameraFrameFrom(camera);
    storeVec3Property(camera, "eye", frame.eye + offset);
    storeVec3Property(camera, "center", frame.center + offset);
    storeVec3Property(camera, "up", frame.up);
    return true;
}

bool orbitCamera(float deltaX, float deltaY, float width, float height, float eye[3], const float center[3], float up[3])
{
    if (deltaX == 0.0f && deltaY == 0.0f) {
        return false;
    }

    constexpr float kPi = 3.14159265358979323846f;
    constexpr float kPolePad = 0.001f;
    const float2 displacement(
        deltaX / std::max(width, 1.0f),
        deltaY / std::max(height, 1.0f));

    const float3 cameraUp = normalizedOr(float3(up[0], up[1], up[2]), float3(0.0f, 1.0f, 0.0f));
    const float3 origin(center[0], center[1], center[2]);
    const float3 position(eye[0], eye[1], eye[2]);
    const float radius = length(position - origin);
    if (radius <= 0.000001f || !std::isfinite(radius)) {
        return false;
    }

    float3 centerToEye = (position - origin) / radius;
    const float cosElev = std::clamp(dot(centerToEye, cameraUp), -1.0f, 1.0f);
    float3 horizontal = centerToEye - cameraUp * cosElev;
    const float sinElev = length(horizontal);
    const float elev = std::atan2(sinElev, cosElev);
    horizontal = sinElev < 0.000001f ? perpendicularTo(cameraUp) : horizontal / sinElev;

    const float yaw = -displacement.x * 2.0f * kPi;
    const float yawC = std::cos(yaw);
    const float yawS = std::sin(yaw);
    horizontal = horizontal * yawC + cross(cameraUp, horizontal) * yawS;

    const float pitch = displacement.y * 2.0f * kPi;
    const float newElev = std::clamp(elev - pitch, kPolePad, kPi - kPolePad);
    centerToEye = (cameraUp * std::cos(newElev) + horizontal * std::sin(newElev)) * radius;

    const float3 newEye = origin + centerToEye;
    eye[0] = newEye.x;
    eye[1] = newEye.y;
    eye[2] = newEye.z;
    up[0] = cameraUp.x;
    up[1] = cameraUp.y;
    up[2] = cameraUp.z;
    return true;
}

bool dollyCamera(float wheel, render::RenderGraphProperties& camera)
{
    if (wheel == 0.0f) {
        return false;
    }

    float eye[3] = {};
    float center[3] = {};
    readVec3Property(camera, "eye", eye);
    readVec3Property(camera, "center", center);

    const std::string projection = camera["projection"].get<std::string>();
    constexpr float kWheelZoomRate = 0.1f;
    const float factor = std::pow(1.0f - kWheelZoomRate, wheel);

    if (projection == "orthographic") {
        const float3 eyeVec(eye[0], eye[1], eye[2]);
        const float3 centerVec(center[0], center[1], center[2]);
        const float distance = std::max(length(eyeVec - centerVec), 0.001f);
        constexpr float kPi = 3.14159265358979323846f;
        const float fovRadians = std::clamp(
            propertyFloatOr(camera, "fovDegrees", 60.0f),
            1.0f,
            179.0f) * (kPi / 180.0f);
        const float defaultHeight = std::max(2.0f * distance * std::tan(fovRadians * 0.5f), 0.0001f);
        const float currentHeight = std::max(propertyFloatOr(camera, "orthoHeight", defaultHeight), 0.0001f);
        camera["orthoHeight"] = std::max(currentHeight * factor, 0.0001f);
        return true;
    }

    const float3 offset(float3(eye[0], eye[1], eye[2]) - float3(center[0], center[1], center[2]));
    const float distance = length(offset);
    if (distance <= 0.000001f || !std::isfinite(distance)) {
        return false;
    }

    const float newDistance = std::max(distance * factor, 0.0001f);
    const float3 newEye = float3(center[0], center[1], center[2]) + (offset / distance) * newDistance;
    eye[0] = newEye.x;
    eye[1] = newEye.y;
    eye[2] = newEye.z;
    storeVec3Property(camera, "eye", eye);
    return true;
}

bool dragDollyCamera(
    float deltaX,
    float deltaY,
    float width,
    float height,
    render::RenderGraphProperties& camera)
{
    if (deltaX == 0.0f && deltaY == 0.0f) {
        return false;
    }

    const float2 displacement(
        deltaX / std::max(width, 1.0f),
        deltaY / std::max(height, 1.0f));
    const float amount = std::abs(displacement.x) > std::abs(displacement.y)
        ? displacement.x
        : -displacement.y;
    if (amount == 0.0f) {
        return false;
    }

    const std::string projection = camera["projection"].get<std::string>();
    if (projection == "orthographic") {
        const CameraFrame frame = cameraFrameFrom(camera);
        const float currentHeight = cameraOrthoHeight(camera, frame.distance);
        camera["orthoHeight"] = std::max(currentHeight * std::max(1.0f - amount, 0.0001f), 0.0001f);
        return true;
    }

    CameraFrame frame = cameraFrameFrom(camera);
    if (frame.distance <= 0.000001f || !std::isfinite(frame.distance) || amount >= kMaxDollyDisplacement) {
        return false;
    }

    const float3 movement = (frame.center - frame.eye) * amount;
    const float movementLength = length(movement);
    if (movementLength <= 0.000001f || !std::isfinite(movementLength)) {
        return false;
    }

    storeVec3Property(camera, "eye", frame.eye + movement);
    storeVec3Property(camera, "up", frame.up);
    return true;
}

bool panCamera(
    float deltaX,
    float deltaY,
    float width,
    float height,
    render::RenderGraphProperties& camera)
{
    if (deltaX == 0.0f && deltaY == 0.0f) {
        return false;
    }

    const CameraFrame frame = cameraFrameFrom(camera);
    const CameraViewDimensions view = cameraViewDimensions(camera, frame, width, height);
    const float2 displacement(
        deltaX / std::max(width, 1.0f),
        deltaY / std::max(height, 1.0f));
    const float3 offset =
        frame.right * (-displacement.x * view.width) +
        frame.viewUp * (displacement.y * view.height);
    return translateCamera(camera, offset);
}

bool keyboardMoveCamera(float rightAmount, float forwardAmount, render::RenderGraphProperties& camera)
{
    if (rightAmount == 0.0f && forwardAmount == 0.0f) {
        return false;
    }

    const CameraFrame frame = cameraFrameFrom(camera);
    const float3 offset = frame.right * rightAmount + frame.forward * forwardAmount;
    return translateCamera(camera, offset);
}

void copyToBuffer(const std::string& value, char* buffer, size_t bufferSize)
{
    if (buffer == nullptr || bufferSize == 0) {
        return;
    }
    std::memset(buffer, 0, bufferSize);
    const size_t copySize = std::min(value.size(), bufferSize - 1);
    std::memcpy(buffer, value.data(), copySize);
}

bool isMarkedRenderGraphOutput(const render::RenderGraph& graph, std::string_view fullName)
{
    for (const render::RenderGraphOutput& output : graph.outputs()) {
        if (render::makeRenderGraphFieldName(output.passName, output.fieldName) == fullName) {
            return true;
        }
    }
    return false;
}

void loadDefaultImGuiLayoutIfMissing()
{
    ImGuiIO& io = ImGui::GetIO();
    if (io.IniFilename == nullptr || io.IniFilename[0] == '\0') {
        return;
    }

    std::error_code error;
    if (std::filesystem::exists(io.IniFilename, error)) {
        return;
    }

    ImGui::LoadIniSettingsFromMemory(kDefaultImGuiIni, std::strlen(kDefaultImGuiIni));
}

render::RenderGraphNode* findRenderGraphNodeForOutput(
    render::RenderGraph& graph,
    std::string_view outputName)
{
    std::string passName;
    std::string fieldName;
    if (!render::splitRenderGraphFieldName(outputName, passName, fieldName)) {
        return nullptr;
    }
    return graph.findNode(passName);
}

bool isCameraRuntimeSetting(const render::RenderGraphRuntimeSetting& setting)
{
    return setting.key == "camera" || setting.key.rfind("camera.", 0) == 0;
}

bool hasVisibleRuntimeSettings(
    const std::vector<render::RenderGraphRuntimeSetting>& settings,
    bool hideCameraSettings)
{
    for (const render::RenderGraphRuntimeSetting& setting : settings) {
        if (hideCameraSettings && isCameraRuntimeSetting(setting)) {
            continue;
        }
        return true;
    }
    return false;
}

const render::RenderGraphProperties* nestedProperty(
    const render::RenderGraphProperties& properties,
    std::string_view key)
{
    const render::RenderGraphProperties* current = &properties;
    size_t begin = 0;
    while (begin < key.size()) {
        const size_t dot = key.find('.', begin);
        const std::string part(key.substr(begin, dot == std::string_view::npos ? key.size() - begin : dot - begin));
        if (part.empty() || !current->is_object()) {
            return nullptr;
        }
        const auto iter = current->find(part);
        if (iter == current->end()) {
            return nullptr;
        }
        if (dot == std::string_view::npos) {
            return &(*iter);
        }
        current = &(*iter);
        begin = dot + 1;
    }
    return nullptr;
}

void setNestedProperty(
    render::RenderGraphProperties& properties,
    std::string_view key,
    render::RenderGraphProperties value)
{
    if (!properties.is_object()) {
        properties = render::RenderGraphProperties::object();
    }

    render::RenderGraphProperties* current = &properties;
    size_t begin = 0;
    while (begin < key.size()) {
        const size_t dot = key.find('.', begin);
        const std::string part(key.substr(begin, dot == std::string_view::npos ? key.size() - begin : dot - begin));
        if (part.empty()) {
            return;
        }
        if (dot == std::string_view::npos) {
            (*current)[part] = std::move(value);
            return;
        }
        render::RenderGraphProperties& child = (*current)[part];
        if (!child.is_object()) {
            child = render::RenderGraphProperties::object();
        }
        current = &child;
        begin = dot + 1;
    }
}

void mergeRuntimeProperties(
    render::RenderGraphProperties& destination,
    const render::RenderGraphProperties& source)
{
    if (!source.is_object()) {
        return;
    }
    if (!destination.is_object()) {
        destination = render::RenderGraphProperties::object();
    }
    for (auto iter = source.begin(); iter != source.end(); ++iter) {
        if (iter.value().is_object() && destination.contains(iter.key()) && destination[iter.key()].is_object()) {
            mergeRuntimeProperties(destination[iter.key()], iter.value());
            continue;
        }
        destination[iter.key()] = iter.value();
    }
}

render::RenderGraphProperties effectiveNodeProperties(const render::RenderGraphNode& node)
{
    render::RenderGraphProperties properties = node.properties.is_object()
        ? node.properties
        : render::RenderGraphProperties::object();
    mergeRuntimeProperties(properties, node.runtimeProperties);
    return properties;
}

render::RenderGraphProperties runtimeSettingValue(
    const render::RenderGraphNode& node,
    const render::RenderGraphRuntimeSetting& setting)
{
    if (const render::RenderGraphProperties* value = nestedProperty(node.runtimeProperties, setting.key)) {
        return *value;
    }
    if (const render::RenderGraphProperties* value = nestedProperty(node.properties, setting.key)) {
        return *value;
    }
    return setting.defaultValue;
}

float floatValueOr(const render::RenderGraphProperties& value, float fallback)
{
    if (!value.is_number()) {
        return fallback;
    }
    const float result = value.get<float>();
    return std::isfinite(result) ? result : fallback;
}

int intValueOr(const render::RenderGraphProperties& value, int fallback)
{
    if (!value.is_number_integer() && !value.is_number_unsigned()) {
        return fallback;
    }
    return value.get<int>();
}

bool boolValueOr(const render::RenderGraphProperties& value, bool fallback)
{
    return value.is_boolean() ? value.get<bool>() : fallback;
}

std::string stringValueOr(const render::RenderGraphProperties& value, std::string fallback)
{
    return value.is_string() ? value.get<std::string>() : std::move(fallback);
}

void floatArrayValueOr(
    const render::RenderGraphProperties& value,
    float* outValues,
    size_t count,
    const render::RenderGraphProperties& fallback)
{
    for (size_t index = 0; index < count; ++index) {
        outValues[index] = 0.0f;
        if (fallback.is_array() && fallback.size() > index && fallback[index].is_number()) {
            outValues[index] = fallback[index].get<float>();
        }
        if (value.is_array() && value.size() > index && value[index].is_number()) {
            outValues[index] = value[index].get<float>();
        }
    }
}

bool drawRuntimeSettingControl(
    const render::RenderGraphRuntimeSetting& setting,
    const render::RenderGraphProperties& currentValue,
    render::RenderGraphProperties& outValue)
{
    const char* label = setting.label.empty() ? setting.key.c_str() : setting.label.c_str();
    switch (setting.type) {
    case render::RenderGraphRuntimeSettingType::Bool: {
        bool value = boolValueOr(currentValue, boolValueOr(setting.defaultValue, false));
        if (ImGui::Checkbox(label, &value)) {
            outValue = value;
            return true;
        }
        return false;
    }
    case render::RenderGraphRuntimeSettingType::Int: {
        int value = intValueOr(currentValue, intValueOr(setting.defaultValue, 0));
        const int minValue = intValueOr(setting.minValue, 0);
        const int maxValue = std::max(intValueOr(setting.maxValue, 100), minValue);
        if (ImGui::SliderInt(label, &value, minValue, maxValue)) {
            outValue = std::clamp(value, minValue, maxValue);
            return true;
        }
        return false;
    }
    case render::RenderGraphRuntimeSettingType::Float: {
        float value = floatValueOr(currentValue, floatValueOr(setting.defaultValue, 0.0f));
        const float minValue = floatValueOr(setting.minValue, 0.0f);
        const float maxValue = std::max(floatValueOr(setting.maxValue, 1.0f), minValue);
        if (ImGui::SliderFloat(label, &value, minValue, maxValue, "%.3f")) {
            outValue = std::clamp(value, minValue, maxValue);
            return true;
        }
        return false;
    }
    case render::RenderGraphRuntimeSettingType::Float3: {
        float values[3] = {};
        floatArrayValueOr(currentValue, values, 3, setting.defaultValue);
        if (ImGui::InputFloat3(label, values, "%.6f")) {
            outValue = {values[0], values[1], values[2]};
            return true;
        }
        return false;
    }
    case render::RenderGraphRuntimeSettingType::Color4: {
        float values[4] = {};
        floatArrayValueOr(currentValue, values, 4, setting.defaultValue);
        if (ImGui::ColorEdit4(label, values)) {
            outValue = {values[0], values[1], values[2], values[3]};
            return true;
        }
        return false;
    }
    case render::RenderGraphRuntimeSettingType::Enum: {
        if (setting.options.empty()) {
            return false;
        }
        int selectedIndex = 0;
        for (int index = 0; index < static_cast<int>(setting.options.size()); ++index) {
            if (setting.options[static_cast<size_t>(index)].value == currentValue) {
                selectedIndex = index;
                break;
            }
        }
        const char* preview = setting.options[static_cast<size_t>(selectedIndex)].label.c_str();
        bool changed = false;
        if (ImGui::BeginCombo(label, preview)) {
            for (int index = 0; index < static_cast<int>(setting.options.size()); ++index) {
                const bool selected = index == selectedIndex;
                if (ImGui::Selectable(setting.options[static_cast<size_t>(index)].label.c_str(), selected)) {
                    selectedIndex = index;
                    changed = true;
                }
                if (selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }
        if (changed) {
            outValue = setting.options[static_cast<size_t>(selectedIndex)].value;
            return true;
        }
        return false;
    }
    case render::RenderGraphRuntimeSettingType::ActionCounter: {
        if (ImGui::Button(label)) {
            const int value = std::max(intValueOr(currentValue, intValueOr(setting.defaultValue, 0)), 0);
            outValue = value + 1;
            return true;
        }
        return false;
    }
    }

    return false;
}
ImU32 colorForPassType(const std::string& type)
{
    uint32_t hash = 2166136261u;
    for (const char c : type) {
        hash ^= static_cast<uint8_t>(c);
        hash *= 16777619u;
    }

    const float hue = static_cast<float>(hash % 360u) / 360.0f;
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    ImGui::ColorConvertHSVtoRGB(hue, 0.72f, 0.78f, r, g, b);
    return ImGui::GetColorU32(ImVec4(r, g, b, 1.0f));
}

const char* renderGraphFieldVisibilityName(render::RenderGraphFieldVisibility visibility)
{
    return visibility == render::RenderGraphFieldVisibility::Input ? "Input" : "Output";
}

const char* renderGraphResourceTypeName(render::RenderGraphResourceType type)
{
    switch (type) {
    case render::RenderGraphResourceType::Texture2D:
        return "Texture2D";
    case render::RenderGraphResourceType::Buffer:
        return "Buffer";
    }

    return "Unknown";
}

const char* renderGraphResourceAccessName(render::RenderGraphResourceAccess access)
{
    switch (access) {
    case render::RenderGraphResourceAccess::None:
        return "None";
    case render::RenderGraphResourceAccess::TextureSampleRead:
        return "SampleRead";
    case render::RenderGraphResourceAccess::TextureColorWrite:
        return "ColorWrite";
    case render::RenderGraphResourceAccess::TextureDepthStencilWrite:
        return "DepthStencilWrite";
    case render::RenderGraphResourceAccess::TextureTransferRead:
        return "TransferRead";
    case render::RenderGraphResourceAccess::TextureTransferWrite:
        return "TransferWrite";
    case render::RenderGraphResourceAccess::TextureStorageReadWrite:
        return "StorageReadWrite";
    case render::RenderGraphResourceAccess::BufferShaderRead:
        return "ShaderRead";
    case render::RenderGraphResourceAccess::BufferStorageReadWrite:
        return "StorageReadWrite";
    case render::RenderGraphResourceAccess::BufferTransferRead:
        return "TransferRead";
    case render::RenderGraphResourceAccess::BufferTransferWrite:
        return "TransferWrite";
    case render::RenderGraphResourceAccess::BufferConstantRead:
        return "ConstantRead";
    }

    return "Unknown";
}

const char* renderGraphBindlessAccessName(render::RenderGraphBindlessAccess access)
{
    switch (access) {
    case render::RenderGraphBindlessAccess::None:
        return "None";
    case render::RenderGraphBindlessAccess::SampledImage:
        return "SampledImage";
    case render::RenderGraphBindlessAccess::Buffer:
        return "Buffer";
    }

    return "Unknown";
}

const char* renderGraphFormatName(render::Format format)
{
    switch (format) {
    case render::Format::Unknown:
        return "Unknown";
    case render::Format::R8Unorm:
        return "R8Unorm";
    case render::Format::R8Snorm:
        return "R8Snorm";
    case render::Format::R8Uint:
        return "R8Uint";
    case render::Format::R8Sint:
        return "R8Sint";
    case render::Format::Rg8Unorm:
        return "Rg8Unorm";
    case render::Format::Rg8Snorm:
        return "Rg8Snorm";
    case render::Format::Rg8Uint:
        return "Rg8Uint";
    case render::Format::Rg8Sint:
        return "Rg8Sint";
    case render::Format::Bgra8Unorm:
        return "Bgra8Unorm";
    case render::Format::Bgra8Srgb:
        return "Bgra8Srgb";
    case render::Format::Rgba8Unorm:
        return "Rgba8Unorm";
    case render::Format::Rgba8Snorm:
        return "Rgba8Snorm";
    case render::Format::Rgba8Srgb:
        return "Rgba8Srgb";
    case render::Format::Rgba8Uint:
        return "Rgba8Uint";
    case render::Format::Rgba8Sint:
        return "Rgba8Sint";
    case render::Format::R16Unorm:
        return "R16Unorm";
    case render::Format::R16Snorm:
        return "R16Snorm";
    case render::Format::R16Uint:
        return "R16Uint";
    case render::Format::R16Sint:
        return "R16Sint";
    case render::Format::R16Sfloat:
        return "R16Sfloat";
    case render::Format::Rg16Unorm:
        return "Rg16Unorm";
    case render::Format::Rg16Snorm:
        return "Rg16Snorm";
    case render::Format::Rg16Uint:
        return "Rg16Uint";
    case render::Format::Rg16Sint:
        return "Rg16Sint";
    case render::Format::Rg16Sfloat:
        return "Rg16Sfloat";
    case render::Format::Rgba16Unorm:
        return "Rgba16Unorm";
    case render::Format::Rgba16Snorm:
        return "Rgba16Snorm";
    case render::Format::Rgba16Uint:
        return "Rgba16Uint";
    case render::Format::Rgba16Sint:
        return "Rgba16Sint";
    case render::Format::Rgba16Sfloat:
        return "Rgba16Sfloat";
    case render::Format::R32Uint:
        return "R32Uint";
    case render::Format::R32Sint:
        return "R32Sint";
    case render::Format::R32Sfloat:
        return "R32Sfloat";
    case render::Format::Rg32Uint:
        return "Rg32Uint";
    case render::Format::Rg32Sint:
        return "Rg32Sint";
    case render::Format::Rg32Sfloat:
        return "Rg32Sfloat";
    case render::Format::Rgb32Uint:
        return "Rgb32Uint";
    case render::Format::Rgb32Sint:
        return "Rgb32Sint";
    case render::Format::Rgb32Sfloat:
        return "Rgb32Sfloat";
    case render::Format::Rgba32Uint:
        return "Rgba32Uint";
    case render::Format::Rgba32Sint:
        return "Rgba32Sint";
    case render::Format::Rgba32Sfloat:
        return "Rgba32Sfloat";
    case render::Format::A2B10G10R10UnormPack32:
        return "A2B10G10R10UnormPack32";
    case render::Format::A2R10G10B10UintPack32:
        return "A2R10G10B10UintPack32";
    case render::Format::B10G11R11UfloatPack32:
        return "B10G11R11UfloatPack32";
    case render::Format::E5B9G9R9UfloatPack32:
        return "E5B9G9R9UfloatPack32";
    case render::Format::D32Sfloat:
        return "D32Sfloat";
    }

    return "Unknown";
}

std::string renderGraphFieldTag(const render::RenderGraphField& field)
{
    std::string tag = "[";
    tag += renderGraphResourceTypeName(field.resourceType);
    tag += "/";
    tag += renderGraphResourceAccessName(field.access);
    if (field.bindlessAccess != render::RenderGraphBindlessAccess::None) {
        tag += "/";
        tag += renderGraphBindlessAccessName(field.bindlessAccess);
    }
    if (field.optional) {
        tag += "/Optional";
    }
    tag += "]";
    return tag;
}

void setRenderGraphFieldTooltip(const render::RenderGraphField& field)
{
    if (!ImGui::IsItemHovered()) {
        return;
    }

    std::string text = std::string(renderGraphResourceTypeName(field.resourceType)) +
        " / " +
        renderGraphResourceAccessName(field.access);
    if (field.bindlessAccess != render::RenderGraphBindlessAccess::None) {
        text += "\nBindless: ";
        text += renderGraphBindlessAccessName(field.bindlessAccess);
    }
    if (field.resourceType == render::RenderGraphResourceType::Texture2D) {
        text += "\nFormat: ";
        text += renderGraphFormatName(field.format);
    } else {
        text += "\nSize: ";
        text += std::to_string(field.size);
        text += " bytes";
        if (field.structureStride > 0) {
            text += "\nStride: ";
            text += std::to_string(field.structureStride);
        }
    }
    if (field.optional) {
        text += "\nOptional";
    }
    if (!field.description.empty()) {
        text += "\n";
        text += field.description;
    }
    ImGui::SetTooltip("%s", text.c_str());
}

void checkVkResult(VkResult result)
{
    if (result < 0) {
        spdlog::error("Vulkan error: {}", static_cast<int>(result));
    }
}

ImVec4 nvproColor(float r, float g, float b, float a)
{
    return ImVec4(r, g, b, a);
}

ImU32 nvproColorU32(float r, float g, float b, float a)
{
    return ImGui::ColorConvertFloat4ToU32(nvproColor(r, g, b, a));
}

template <size_t Count>
void applyImGuiColorGroup(ImGuiStyle& style, const ImGuiCol (&colorIndices)[Count], const ImVec4& color)
{
    for (const ImGuiCol colorIndex : colorIndices) {
        style.Colors[colorIndex] = color;
    }
}

void applyNvproImGuiStyle()
{
    ImGui::StyleColorsDark();

    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 0.0f;
    style.WindowBorderSize = 0.0f;
    style.ColorButtonPosition = ImGuiDir_Right;
    style.FrameRounding = 2.0f;
    style.FrameBorderSize = 1.0f;
    style.GrabRounding = 4.0f;
    style.IndentSpacing = 12.0f;

    style.Colors[ImGuiCol_WindowBg] = nvproColor(0.2f, 0.2f, 0.2f, 1.0f);
    style.Colors[ImGuiCol_MenuBarBg] = nvproColor(0.2f, 0.2f, 0.2f, 1.0f);
    style.Colors[ImGuiCol_ScrollbarBg] = nvproColor(0.2f, 0.2f, 0.2f, 1.0f);
    style.Colors[ImGuiCol_PopupBg] = nvproColor(0.135f, 0.135f, 0.135f, 1.0f);
    style.Colors[ImGuiCol_Border] = nvproColor(0.4f, 0.4f, 0.4f, 0.5f);
    style.Colors[ImGuiCol_FrameBg] = nvproColor(0.05f, 0.05f, 0.05f, 0.5f);

    constexpr ImGuiCol kNormalColors[] = {
        ImGuiCol_Header,
        ImGuiCol_SliderGrab,
        ImGuiCol_Button,
        ImGuiCol_CheckMark,
        ImGuiCol_ResizeGrip,
        ImGuiCol_TextSelectedBg,
        ImGuiCol_Separator,
        ImGuiCol_FrameBgActive,
    };
    applyImGuiColorGroup(style, kNormalColors, nvproColor(0.465f, 0.465f, 0.525f, 1.0f));

    constexpr ImGuiCol kActiveColors[] = {
        ImGuiCol_HeaderActive,
        ImGuiCol_SliderGrabActive,
        ImGuiCol_ButtonActive,
        ImGuiCol_ResizeGripActive,
        ImGuiCol_SeparatorActive,
    };
    applyImGuiColorGroup(style, kActiveColors, nvproColor(0.365f, 0.365f, 0.425f, 1.0f));

    constexpr ImGuiCol kHoveredColors[] = {
        ImGuiCol_HeaderHovered,
        ImGuiCol_ButtonHovered,
        ImGuiCol_FrameBgHovered,
        ImGuiCol_ResizeGripHovered,
        ImGuiCol_SeparatorHovered,
    };
    applyImGuiColorGroup(style, kHoveredColors, nvproColor(0.565f, 0.565f, 0.625f, 1.0f));

    style.Colors[ImGuiCol_TitleBgActive] = nvproColor(0.465f, 0.465f, 0.465f, 1.0f);
    style.Colors[ImGuiCol_TitleBg] = nvproColor(0.125f, 0.125f, 0.125f, 1.0f);
    style.Colors[ImGuiCol_Tab] = nvproColor(0.05f, 0.05f, 0.05f, 0.5f);
    style.Colors[ImGuiCol_TabHovered] = nvproColor(0.465f, 0.495f, 0.525f, 1.0f);
    style.Colors[ImGuiCol_TabSelected] = nvproColor(0.282f, 0.290f, 0.302f, 1.0f);
    style.Colors[ImGuiCol_TabDimmedSelected] = style.Colors[ImGuiCol_TabSelected];
    style.Colors[ImGuiCol_DockingPreview] = nvproColor(0.465f, 0.465f, 0.525f, 0.7f);
    style.Colors[ImGuiCol_DockingEmptyBg] = nvproColor(0.2f, 0.2f, 0.2f, 1.0f);
    style.Colors[ImGuiCol_ModalWindowDimBg] = nvproColor(0.465f, 0.465f, 0.465f, 0.350f);

    ImGui::SetColorEditOptions(ImGuiColorEditFlags_Float | ImGuiColorEditFlags_PickerHueWheel);
}

void applyNvproImNodesStyle()
{
    ImNodes::StyleColorsDark();

    ImNodesStyle& style = ImNodes::GetStyle();
    style.Colors[ImNodesCol_NodeBackground] = nvproColorU32(0.135f, 0.135f, 0.135f, 1.0f);
    style.Colors[ImNodesCol_NodeBackgroundHovered] = nvproColorU32(0.2f, 0.2f, 0.2f, 1.0f);
    style.Colors[ImNodesCol_NodeBackgroundSelected] = nvproColorU32(0.2f, 0.2f, 0.2f, 1.0f);
    style.Colors[ImNodesCol_NodeOutline] = nvproColorU32(0.4f, 0.4f, 0.4f, 0.5f);
    style.Colors[ImNodesCol_TitleBar] = nvproColorU32(0.125f, 0.125f, 0.125f, 1.0f);
    style.Colors[ImNodesCol_TitleBarHovered] = nvproColorU32(0.465f, 0.495f, 0.525f, 1.0f);
    style.Colors[ImNodesCol_TitleBarSelected] = nvproColorU32(0.282f, 0.290f, 0.302f, 1.0f);
    style.Colors[ImNodesCol_Link] = nvproColorU32(0.465f, 0.465f, 0.525f, 0.85f);
    style.Colors[ImNodesCol_LinkHovered] = nvproColorU32(0.565f, 0.565f, 0.625f, 1.0f);
    style.Colors[ImNodesCol_LinkSelected] = nvproColorU32(0.565f, 0.565f, 0.625f, 1.0f);
    style.Colors[ImNodesCol_Pin] = nvproColorU32(0.465f, 0.465f, 0.525f, 0.85f);
    style.Colors[ImNodesCol_PinHovered] = nvproColorU32(0.565f, 0.565f, 0.625f, 1.0f);
    style.Colors[ImNodesCol_BoxSelector] = nvproColorU32(0.465f, 0.465f, 0.525f, 0.12f);
    style.Colors[ImNodesCol_BoxSelectorOutline] = nvproColorU32(0.565f, 0.565f, 0.625f, 0.8f);
    style.Colors[ImNodesCol_GridBackground] = nvproColorU32(0.2f, 0.2f, 0.2f, 1.0f);
    style.Colors[ImNodesCol_GridLine] = nvproColorU32(0.565f, 0.565f, 0.625f, 0.18f);
    style.Colors[ImNodesCol_GridLinePrimary] = nvproColorU32(0.565f, 0.565f, 0.625f, 0.28f);
}

} // namespace

int EditorApplication::run(
    bool smokeTest,
    bool waitForGraphicsDebugger,
    const char* startupSampleId,
    const char* startupScenePath,
    const char* startupStreamAssetPath)
{
    smokeTest_ = smokeTest;
    waitForGraphicsDebugger_ = waitForGraphicsDebugger && !smokeTest;
    startupSampleId_ = startupSampleId != nullptr ? startupSampleId : "";
    startupScenePath_ = startupScenePath != nullptr ? startupScenePath : "";
    startupStreamAssetPath_ = startupStreamAssetPath != nullptr ? startupStreamAssetPath : "";
    spdlog::info(
        "[Startup] Run requested smokeTest={} waitForGraphicsDebugger={} startupSample='{}' "
        "sceneOverride='{}' streamAssetOverride='{}'",
        smokeTest_,
        waitForGraphicsDebugger_,
        startupSampleId_,
        startupScenePath_,
        startupStreamAssetPath_);

    if (!initialize()) {
        shutdown();
        return 1;
    }

    if (smokeTest) {
        auto profileFrame = profiler_.beginFrame();
        {
            auto profileScope = profiler_.scope("Poll Events");
            pollEvents();
        }
        {
            auto profileScope = profiler_.scope("Render Frame");
            const bool rendered = renderFrame();
            shutdown();
            return rendered ? 0 : 1;
        }
    }

    while (running_) {
        auto profileFrame = profiler_.beginFrame();
        {
            auto profileScope = profiler_.scope("Poll Events");
            pollEvents();
        }

        if ((SDL_GetWindowFlags(window_) & SDL_WINDOW_MINIMIZED) != 0) {
            auto profileScope = profiler_.scope("Minimized Wait");
            SDL_Delay(10);
            continue;
        }

        {
            auto profileScope = profiler_.scope("Render Frame");
            renderFrame();
        }
    }

    shutdown();
    return 0;
}

bool EditorApplication::initialize()
{
    StartupLogScope initializeScope("Editor initialization");

    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD)) {
        spdlog::error("SDL_Init failed: {}", SDL_GetError());
        return false;
    }

#ifdef SDL_HINT_IME_SHOW_UI
    SDL_SetHint(SDL_HINT_IME_SHOW_UI, "1");
#endif

    mainScale_ = getMainDisplayScale();
    const SDL_WindowFlags windowFlags =
        SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIDDEN | SDL_WINDOW_HIGH_PIXEL_DENSITY;

    window_ = SDL_CreateWindow(
        "Metallic Engine Editor",
        static_cast<int>(kBaseWindowWidth * mainScale_),
        static_cast<int>(kBaseWindowHeight * mainScale_),
        windowFlags);
    if (window_ == nullptr) {
        spdlog::error("SDL_CreateWindow failed: {}", SDL_GetError());
        return false;
    }

    SDL_SetWindowPosition(window_, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
    SDL_ShowWindow(window_);

    if (waitForGraphicsDebugger_) {
        SDL_ShowSimpleMessageBox(
            SDL_MESSAGEBOX_INFORMATION,
            "Metallic Graphics Debugger",
            "Attach RenderDoc or Nsight Graphics now, then press OK to create the Vulkan device and swapchain.",
            window_);
    }

    if (!initializeRhi()) {
        return false;
    }

    {
        StartupLogScope scope("ImGui context and style setup");
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        imguiContextCreated_ = true;

        ImGuiIO& io = ImGui::GetIO();
        if (smokeTest_) {
            io.IniFilename = nullptr;
        }

        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
        loadDefaultImGuiLayoutIfMissing();

        applyNvproImGuiStyle();
        ImGuiStyle& style = ImGui::GetStyle();
        style.ScaleAllSizes(mainScale_);
        style.FontScaleDpi = mainScale_;

        ImNodes::CreateContext();
        imnodesContextCreated_ = true;
        applyNvproImNodesStyle();
    }

    if (!initializeImGuiBackends()) {
        return false;
    }

    {
        StartupLogScope scope("NVML monitor initialization");
        nvmlMonitor_.initialize();
    }

    {
        StartupLogScope scope("Startup render graph and scene setup");
        graphExecutor_ = std::make_unique<render::RenderGraphExecutor>();
        sceneRtx_ = std::make_unique<render::vulkan::SceneRtxBuilder>();
        if (!startupSampleId_.empty()) {
            loadBuiltInSample(startupSampleId_.c_str());
        } else {
            resetDefaultRenderGraph();
        }
    }

    return true;
}

bool EditorApplication::initializeRhi()
{
    StartupLogScope initializeScope("RHI initialization");

    render::Result result;
    {
        StartupLogScope scope("RHI createDevice");
        result = render::createDevice(
            render::DeviceDesc{
                .applicationName = "Metallic Engine Editor",
                .enableValidation = false,
                .enableBindlessDescriptorHeap = true,
                .enableShaderObject = true,
                .enableMeshShader = true,
                .enableRayTracingAccelerationStructure = true,
                .enableRayQuery = true,
                .enablePushDescriptor = true,
                .enableClusterAccelerationStructure = true,
                .enableStreamline = true,
                .enableAftermath = true,
            },
            device_);
    }
    if (!result || device_ == nullptr) {
        spdlog::error("createDevice failed with Result {}", render::resultToString(result));
        return false;
    }

    graphicsQueue_ = device_->getQueue(render::QueueType::Graphics);
    if (graphicsQueue_ == nullptr) {
        spdlog::error("RHI graphics queue is not available");
        return false;
    }

    {
        StartupLogScope scope("HistoryResourceManager initialization");
        result = historyResources_.initialize(*device_);
        if (!result) {
            spdlog::error("HistoryResourceManager initialize failed with Result {}", render::resultToString(result));
            return false;
        }
    }

    {
        StartupLogScope scope("Frame synchronization resource creation");
        result = device_->createCommandPool(*graphicsQueue_, commandPool_);
        if (!result) {
            spdlog::error("createCommandPool failed with Result {}", render::resultToString(result));
            return false;
        }
        result = commandPool_->createCommandBuffer(commandBuffer_);
        if (!result) {
            spdlog::error("createCommandBuffer failed with Result {}", render::resultToString(result));
            return false;
        }
        result = device_->createFence(true, frameFence_);
        if (!result) {
            spdlog::error("createFence failed with Result {}", render::resultToString(result));
            return false;
        }
        result = device_->createSwapchainSemaphore(imageAvailableSemaphore_);
        if (!result) {
            spdlog::error(
                "createSwapchainSemaphore(imageAvailable) failed with Result {}",
                render::resultToString(result));
            return false;
        }
    }

    int width = 0;
    int height = 0;
    if (!SDL_GetWindowSizeInPixels(window_, &width, &height)) {
        spdlog::error("SDL_GetWindowSizeInPixels failed: {}", SDL_GetError());
        return false;
    }
    {
        StartupLogScope scope("Initial swapchain creation");
        if (!createOrResizeSwapchain(
                static_cast<uint32_t>(std::max(width, 1)),
                static_cast<uint32_t>(std::max(height, 1)))) {
            return false;
        }
    }

    StartupLogScope scope("Viewport sampler creation");
    return createViewportSampler();
}

bool EditorApplication::createOrResizeSwapchain(uint32_t width, uint32_t height)
{
    StartupLogScope scope(
        "Swapchain create/resize " + std::to_string(width) + "x" + std::to_string(height));

    if (device_ == nullptr || width == 0 || height == 0) {
        return false;
    }

    if (swapchain_ != nullptr && swapchainWidth_ == width && swapchainHeight_ == height && !swapchainOutOfDate_) {
        return true;
    }

    (void)device_->waitIdle();
    destroySwapchainResources();

    render::Result result = device_->createSwapchain(
        render::SwapchainDesc{
            .window = render::WindowHandle{
                .system = render::WindowSystem::Sdl3,
                .nativeWindow = window_,
            },
            .width = width,
            .height = height,
            .imageCount = kSwapchainImageCount,
            .framesInFlight = 2,
            .format = render::Format::Bgra8Unorm,
            .vsync = true,
        },
        swapchain_);
    if (!result || swapchain_ == nullptr) {
        spdlog::error("createSwapchain failed with Result {}", render::resultToString(result));
        return false;
    }

    swapchainImageViews_.reserve(swapchain_->imageCount());
    renderFinishedSemaphores_.reserve(swapchain_->imageCount());
    swapchainImageStates_.assign(swapchain_->imageCount(), render::ResourceState::Undefined);
    for (uint32_t imageIndex = 0; imageIndex < swapchain_->imageCount(); ++imageIndex) {
        render::Texture* texture = swapchain_->texture(imageIndex);
        if (texture == nullptr) {
            spdlog::error("swapchain texture is missing at image {}", imageIndex);
            return false;
        }

        std::unique_ptr<render::TextureView> view;
        result = device_->createTextureView(
            *texture,
            render::TextureViewDesc{
                .format = swapchain_->format(),
                .baseMip = 0,
                .mipCount = 1,
                .baseLayer = 0,
                .layerCount = 1,
            },
            view);
        if (!result || view == nullptr) {
            spdlog::error("createTextureView(swapchain) failed with Result {}", render::resultToString(result));
            return false;
        }
        swapchainImageViews_.push_back(std::move(view));

        std::unique_ptr<render::SwapchainSemaphore> renderFinished;
        result = device_->createSwapchainSemaphore(renderFinished);
        if (!result || renderFinished == nullptr) {
            spdlog::error(
                "createSwapchainSemaphore(renderFinished) failed with Result {}",
                render::resultToString(result));
            return false;
        }
        renderFinishedSemaphores_.push_back(std::move(renderFinished));
    }

    swapchainWidth_ = swapchain_->width();
    swapchainHeight_ = swapchain_->height();
    swapchainOutOfDate_ = false;

    if (imguiRendererInitialized_) {
        ImGui_ImplVulkan_SetMinImageCount(kMinSwapchainImageCount);
    }
    return true;
}

void EditorApplication::destroySwapchainResources()
{
    swapchainImageViews_.clear();
    swapchainImageStates_.clear();
    renderFinishedSemaphores_.clear();
    swapchain_.reset();
    swapchainWidth_ = 0;
    swapchainHeight_ = 0;
}

bool EditorApplication::initializeImGuiBackends()
{
    StartupLogScope scope("ImGui SDL/Vulkan backend initialization");

    imguiPlatformInitialized_ = ImGui_ImplSDL3_InitForVulkan(window_);
    if (!imguiPlatformInitialized_) {
        spdlog::error("ImGui SDL3 Vulkan platform backend initialization failed");
        return false;
    }

    const render::vulkan::NativeDevice nativeDevice = render::vulkan::nativeDevice(*device_);
    const render::vulkan::NativeQueue nativeQueue = render::vulkan::nativeQueue(*graphicsQueue_);
    const VkFormat colorFormat = render::vulkan::nativeSwapchainFormat(*swapchain_);
    if (nativeDevice.instance == VK_NULL_HANDLE ||
        nativeDevice.physicalDevice == VK_NULL_HANDLE ||
        nativeDevice.device == VK_NULL_HANDLE ||
        nativeQueue.queue == VK_NULL_HANDLE ||
        colorFormat == VK_FORMAT_UNDEFINED) {
        spdlog::error("Invalid Vulkan native handles for ImGui backend");
        return false;
    }

    VkPipelineRenderingCreateInfo pipelineRenderingInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &colorFormat,
    };

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.ApiVersion = nativeDevice.apiVersion;
    initInfo.Instance = nativeDevice.instance;
    initInfo.PhysicalDevice = nativeDevice.physicalDevice;
    initInfo.Device = nativeDevice.device;
    initInfo.QueueFamily = nativeQueue.familyIndex;
    initInfo.Queue = nativeQueue.queue;
    initInfo.DescriptorPoolSize = 128;
    initInfo.MinImageCount = kMinSwapchainImageCount;
    initInfo.ImageCount = swapchain_->imageCount();
    initInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
#ifdef IMGUI_IMPL_VULKAN_HAS_DYNAMIC_RENDERING
    initInfo.UseDynamicRendering = true;
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo = pipelineRenderingInfo;
#endif
    initInfo.CheckVkResultFn = checkVkResult;

    imguiRendererInitialized_ = ImGui_ImplVulkan_Init(&initInfo);
    if (!imguiRendererInitialized_) {
        spdlog::error("ImGui Vulkan renderer backend initialization failed");
        return false;
    }
    return true;
}

bool EditorApplication::createViewportSampler()
{
    if (device_ == nullptr) {
        return false;
    }
    const render::vulkan::NativeDevice nativeDevice = render::vulkan::nativeDevice(*device_);
    if (nativeDevice.device == VK_NULL_HANDLE) {
        return false;
    }

    VkSamplerCreateInfo samplerInfo{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .maxLod = 1.0f,
    };
    const VkResult result = vkCreateSampler(nativeDevice.device, &samplerInfo, nullptr, &viewportSampler_);
    if (result != VK_SUCCESS) {
        spdlog::error("vkCreateSampler(viewport) failed with VkResult {}", static_cast<int>(result));
        return false;
    }
    return true;
}

void EditorApplication::shutdown()
{
    if (device_ != nullptr) {
        (void)device_->waitIdle();
    }

    destroyViewportTexture();
    historyResources_.reset();
    nvmlMonitor_.shutdown();

    if (imguiRendererInitialized_) {
        ImGui_ImplVulkan_Shutdown();
        imguiRendererInitialized_ = false;
    }

    if (imguiPlatformInitialized_) {
        ImGui_ImplSDL3_Shutdown();
        imguiPlatformInitialized_ = false;
    }

    if (imnodesContextCreated_) {
        ImNodes::DestroyContext();
        imnodesContextCreated_ = false;
    }

    if (imguiContextCreated_) {
        ImGui::DestroyContext();
        imguiContextCreated_ = false;
    }

    graphExecutor_.reset();
    sceneRtx_.reset();

    if (viewportSampler_ != VK_NULL_HANDLE && device_ != nullptr) {
        const render::vulkan::NativeDevice nativeDevice = render::vulkan::nativeDevice(*device_);
        if (nativeDevice.device != VK_NULL_HANDLE) {
            vkDestroySampler(nativeDevice.device, viewportSampler_, nullptr);
        }
        viewportSampler_ = VK_NULL_HANDLE;
    }

    commandBuffer_.reset();
    commandPool_.reset();
    frameFence_.reset();
    imageAvailableSemaphore_.reset();
    destroySwapchainResources();
    graphicsQueue_ = nullptr;
    device_.reset();

    if (window_ != nullptr) {
        SDL_DestroyWindow(window_);
        window_ = nullptr;
    }

    SDL_Quit();
}

void EditorApplication::pollEvents()
{
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        ImGui_ImplSDL3_ProcessEvent(&event);

        if (event.type == SDL_EVENT_QUIT) {
            running_ = false;
        }

        if (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED &&
            event.window.windowID == SDL_GetWindowID(window_)) {
            running_ = false;
        }

        if (event.type == SDL_EVENT_DROP_FILE && event.drop.data != nullptr) {
            const std::filesystem::path droppedPath(event.drop.data);
            if (isSceneFilePath(droppedPath)) {
                loadDroppedScene(droppedPath);
            } else if (isRenderGraphFilePath(droppedPath)) {
                loadDroppedRenderGraph(droppedPath);
            } else {
                sceneStatus_ = "Ignored dropped file: " + droppedPath.string();
            }
            SDL_free(const_cast<char*>(event.drop.data));
        }
    }
}

bool EditorApplication::renderFrame()
{
    int framebufferWidth = 0;
    int framebufferHeight = 0;
    if (!SDL_GetWindowSizeInPixels(window_, &framebufferWidth, &framebufferHeight)) {
        spdlog::error("SDL_GetWindowSizeInPixels failed: {}", SDL_GetError());
        running_ = false;
        return false;
    }
    if (framebufferWidth <= 0 || framebufferHeight <= 0) {
        return true;
    }

    if (frameFence_ != nullptr) {
        auto profileScope = profiler_.scope("Wait Frame Fence");
        render::Result result = frameFence_->wait();
        if (!result) {
            spdlog::error("frameFence wait before UI failed with Result {}", render::resultToString(result));
            running_ = false;
            return false;
        }
    }

    if (swapchainOutOfDate_ ||
        swapchain_ == nullptr ||
        swapchainWidth_ != static_cast<uint32_t>(framebufferWidth) ||
        swapchainHeight_ != static_cast<uint32_t>(framebufferHeight)) {
        auto profileScope = profiler_.scope("Resize Swapchain");
        if (!createOrResizeSwapchain(
                static_cast<uint32_t>(framebufferWidth),
                static_cast<uint32_t>(framebufferHeight))) {
            running_ = false;
            return false;
        }
    }

    {
        auto profileScope = profiler_.scope("ImGui NewFrame");
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();
    }

    {
        auto profileScope = profiler_.scope("Dockspace");
        drawDockspace();
    }
    {
        auto profileScope = profiler_.scope("Panels");
        drawPanels();
    }
    if (smokeTest_ && !viewportPreviewValid_) {
        spdlog::error("Smoke test did not produce a valid viewport preview: {}", renderGraphStatus_);
        running_ = false;
        return false;
    }

    {
        auto profileScope = profiler_.scope("ImGui Render");
        ImGui::Render();
    }
    if (!renderVulkanFrame()) {
        running_ = false;
        return false;
    }
    return true;
}

void EditorApplication::setupDefaultDockLayout()
{
    if (dockLayoutInitialized_) {
        return;
    }
    dockLayoutInitialized_ = true;

    const ImGuiID dockspaceId = ImGui::GetID("MetallicDockspace");
    if (ImGui::DockBuilderGetNode(dockspaceId) != nullptr) {
        return;
    }

    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::DockBuilderAddNode(dockspaceId, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dockspaceId, viewport->WorkSize);

    ImGuiID viewportDockId = dockspaceId;
    ImGuiID sideDockId = ImGui::DockBuilderSplitNode(
        viewportDockId,
        ImGuiDir_Right,
        0.24f,
        nullptr,
        &viewportDockId);
    ImGuiID bottomDockId = ImGui::DockBuilderSplitNode(
        viewportDockId,
        ImGuiDir_Down,
        0.40f,
        nullptr,
        &viewportDockId);

    ImGui::DockBuilderDockWindow("Viewport", viewportDockId);
    ImGuiID inspectorDockId = ImGui::DockBuilderSplitNode(
        sideDockId,
        ImGuiDir_Down,
        0.35f,
        nullptr,
        &sideDockId);

    ImGui::DockBuilderDockWindow("Scene Browser", sideDockId);
    ImGui::DockBuilderDockWindow("Inspector", inspectorDockId);
    ImGui::DockBuilderDockWindow("Assets", bottomDockId);
    ImGui::DockBuilderDockWindow("Console", bottomDockId);
    ImGui::DockBuilderDockWindow("NVML Monitor", bottomDockId);
    ImGui::DockBuilderDockWindow("Profiler", bottomDockId);
    ImGui::DockBuilderDockWindow("Statistics", bottomDockId);
    ImGui::DockBuilderFinish(dockspaceId);
}

void EditorApplication::drawDockspace()
{
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::SetNextWindowViewport(viewport->ID);

    ImGuiWindowFlags windowFlags = ImGuiWindowFlags_MenuBar |
        ImGuiWindowFlags_NoDocking |
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoBringToFrontOnFocus |
        ImGuiWindowFlags_NoNavFocus;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

    ImGui::Begin("MetallicEditorDockspace", nullptr, windowFlags);
    ImGui::PopStyleVar(3);

    const ImGuiID dockspaceId = ImGui::GetID("MetallicDockspace");
    setupDefaultDockLayout();
    ImGui::DockSpace(dockspaceId, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_PassthruCentralNode);

    if (!ImGui::GetIO().WantTextInput) {
        if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_O)) {
            chooseEnvironmentFile();
        } else if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_O)) {
            chooseSceneFile();
        }
    }

    if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("New Render Graph")) {
                resetDefaultRenderGraph();
            }
            if (ImGui::MenuItem("Load Render Graph")) {
                loadRenderGraph();
            }
            if (ImGui::MenuItem("Save Render Graph")) {
                saveRenderGraph();
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Open Scene...", "Ctrl+O")) {
                chooseSceneFile();
            }
            if (ImGui::MenuItem("Load Environment...", "Ctrl+Shift+O")) {
                chooseEnvironmentFile();
            }
            if (ImGui::BeginMenu("Open Recent")) {
                if (recentScenePaths_.empty()) {
                    ImGui::TextDisabled("No recent scenes");
                }
                for (const std::filesystem::path& recentPath : recentScenePaths_) {
                    if (ImGui::MenuItem(recentPath.string().c_str())) {
                        copyToBuffer(recentPath.string(), sceneFilePath_, sizeof(sceneFilePath_));
                        loadScene();
                    }
                }
                ImGui::EndMenu();
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Exit")) {
                running_ = false;
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Window")) {
            if (ImGui::MenuItem("Open Render Graph Editor")) {
                renderGraphEditorOpen_ = true;
            }
            if (ImGui::MenuItem("Reset Main Layout")) {
                ImGui::DockBuilderRemoveNode(dockspaceId);
                dockLayoutInitialized_ = false;
            }
            ImGui::Separator();
            ImGui::MenuItem("Scene Browser");
            ImGui::MenuItem("Inspector", nullptr, &inspectorOpen_);
            ImGui::MenuItem("Statistics", nullptr, &statisticsOpen_);
            ImGui::MenuItem("Viewport");
            ImGui::MenuItem("Assets");
            ImGui::MenuItem("Console");
            ImGui::MenuItem("Profiler", nullptr, &profilerOpen_);
            ImGui::MenuItem("NVML Monitor", nullptr, &nvmlMonitorOpen_);
            ImGui::EndMenu();
        }

        if (ImGui::Button("Open Render Graph Editor")) {
            renderGraphEditorOpen_ = true;
        }

        ImGui::EndMenuBar();
    }

    ImGui::End();
}

void EditorApplication::drawPanels()
{
    {
        auto profileScope = profiler_.scope("Scene Panel");
        drawScenePanel();
    }
    {
        auto profileScope = profiler_.scope("Inspector Panel");
        drawInspectorPanel();
    }
    {
        auto profileScope = profiler_.scope("Statistics Panel");
        drawStatisticsPanel();
    }
    {
        auto profileScope = profiler_.scope("Viewport Panel");
        drawViewportPanel();
    }
    {
        auto profileScope = profiler_.scope("Render Graph Editor");
        drawRenderGraphEditorWindow();
    }

    ImGui::Begin("Assets");
    ImGui::TextUnformatted(PROJECT_SOURCE_DIR);
    ImGui::End();

    ImGui::Begin("Console");
    ImGui::TextUnformatted("Metallic editor initialized with SDL3.");
    if (!renderGraphStatus_.empty()) {
        ImGui::Separator();
        ImGui::TextWrapped("%s", renderGraphStatus_.c_str());
    }
    ImGui::End();

    profiler_.drawWindow(&profilerOpen_);
    nvmlMonitor_.drawWindow(&nvmlMonitorOpen_);
}

void EditorApplication::drawScenePanel()
{
    ImGui::Begin("Scene Browser");

    ImGui::TextUnformatted("glTF Scene");
    if (scene_.valid()) {
        ImGui::TextWrapped("Path: %s", scene_.filename().string().c_str());
    } else {
        ImGui::TextDisabled("No scene loaded.");
    }

    if (ImGui::Button("Clear")) {
        clearSceneRtx();
        scene_.clear();
        sceneSelection_ = SceneSelection{};
        historyResources_.invalidateAll();
        viewportPreviewValid_ = false;
        sceneStatus_ = "No scene loaded.";
    }
    ImGui::SameLine();
    if (ImGui::Button("Build RTX AS")) {
        buildSceneRtx();
    }
    ImGui::SameLine();
    if (ImGui::Button("Clear RTX AS")) {
        clearSceneRtx();
    }

    if (!sceneStatus_.empty()) {
        ImGui::TextWrapped("%s", sceneStatus_.c_str());
    }
    if (!sceneRtxStatus_.empty()) {
        ImGui::TextWrapped("%s", sceneRtxStatus_.c_str());
    }
    if (sceneRtx_ != nullptr && sceneRtx_->valid()) {
        const render::vulkan::SceneRtxStats& rtxStats = sceneRtx_->stats();
        ImGui::Text(
            "RTX AS: %u BLAS, %u instances, %llu triangles",
            rtxStats.blasCount,
            rtxStats.instanceCount,
            static_cast<unsigned long long>(rtxStats.triangleCount));
        ImGui::Text(
            "RTX memory: geometry %llu bytes, AS %llu bytes, scratch %llu bytes",
            static_cast<unsigned long long>(rtxStats.geometryBytes),
            static_cast<unsigned long long>(rtxStats.accelerationStructureBytes),
            static_cast<unsigned long long>(rtxStats.scratchBytes));
    }
    const scene::LoadResult& loadResult = scene_.lastLoadResult();
    if (!loadResult.warning.empty()) {
        ImGui::TextWrapped("Warning: %s", loadResult.warning.c_str());
    }
    if (!loadResult.error.empty() && !loadResult.success) {
        ImGui::TextWrapped("Error: %s", loadResult.error.c_str());
    }

    ImGui::Separator();
    if (ImGui::CollapsingHeader("Runtime Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
        render::RenderGraphNode* node = activePreviewRenderGraphNode();
        if (node == nullptr) {
            ImGui::TextDisabled("No active preview render pass.");
        } else {
            ImGui::TextDisabled(
                "Active pass: %s (%s)",
                node->name.c_str(),
                node->type.c_str());
            ImGui::PushID("SceneRuntimeSettings");
            drawRuntimeSettingsForNode(
                *node,
                true,
                true);
            ImGui::PopID();
        }
    }

    ImGui::Separator();
    drawCameraControls();

    if (!scene_.valid()) {
        ImGui::Separator();
        ImGui::TextDisabled("Load a .gltf or .glb file to inspect its CPU scene graph.");
        ImGui::End();
        return;
    }

    ImGui::Separator();
    if (ImGui::CollapsingHeader("Asset Info")) {
        const scene::SceneAssetInfo& asset = scene_.assetInfo();
        ImGui::Text("Path: %s", scene_.filename().string().c_str());
        ImGui::Text("glTF Version: %s", asset.version.empty() ? "-" : asset.version.c_str());
        if (!asset.generator.empty()) {
            ImGui::TextWrapped("Generator: %s", asset.generator.c_str());
        }
        if (!asset.copyright.empty()) {
            ImGui::TextWrapped("Copyright: %s", asset.copyright.c_str());
        }
        if (!asset.minVersion.empty()) {
            ImGui::Text("Min Version: %s", asset.minVersion.c_str());
        }
    }

    if (ImGui::BeginTabBar("SceneBrowserTabs")) {
        if (ImGui::BeginTabItem("Scene Graph")) {
            drawSceneGraphTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Scene List")) {
            drawSceneListTab();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    ImGui::End();
}

void EditorApplication::drawSceneGraphTab()
{
    if (!scene_.valid()) {
        ImGui::TextDisabled("No scene loaded.");
        return;
    }

    constexpr ImGuiTableFlags tableFlags =
        ImGuiTableFlags_RowBg |
        ImGuiTableFlags_BordersOuter |
        ImGuiTableFlags_BordersV |
        ImGuiTableFlags_ScrollY |
        ImGuiTableFlags_Resizable;
    if (!ImGui::BeginTable("SceneGraphTable", 2, tableFlags, ImVec2(0.0f, 360.0f * mainScale_))) {
        return;
    }

    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("Tags", ImGuiTableColumnFlags_WidthFixed, 112.0f * mainScale_);
    ImGui::TableHeadersRow();

    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    const std::string sceneLabel = "Scene-" + std::to_string(scene_.sceneIndex()) + " " + scene_.sceneName();
    const bool sceneOpen = ImGui::TreeNodeEx(
        "SceneRoot",
        ImGuiTreeNodeFlags_DefaultOpen |
            ImGuiTreeNodeFlags_OpenOnArrow |
            ImGuiTreeNodeFlags_SpanAvailWidth,
        "%s",
        sceneLabel.c_str());
    ImGui::TableNextColumn();
    ImGui::Text("%zu roots", scene_.rootNodeIndices().size());
    if (sceneOpen) {
        for (const int32_t rootNodeIndex : scene_.rootNodeIndices()) {
            drawSceneNode(rootNodeIndex);
        }
        ImGui::TreePop();
    }

    ImGui::EndTable();
}

void EditorApplication::drawSceneListSelectable(const char* label, int32_t index, int32_t type)
{
    const SceneSelectionType selectionType = static_cast<SceneSelectionType>(type);
    const bool selected = sceneSelection_.type == selectionType && sceneSelection_.index == index;
    if (ImGui::Selectable(label, selected)) {
        sceneSelection_ = SceneSelection{
            .type = selectionType,
            .index = index,
        };
    }
}

void EditorApplication::drawSceneListTab()
{
    if (!scene_.valid()) {
        ImGui::TextDisabled("No scene loaded.");
        return;
    }

    if (ImGui::CollapsingHeader(("Nodes (" + std::to_string(scene_.nodes().size()) + ")").c_str())) {
        ImGui::BeginChild("NodesScrollRegion", ImVec2(0, 160.0f * mainScale_), false, ImGuiWindowFlags_HorizontalScrollbar);
        for (size_t index = 0; index < scene_.nodes().size(); ++index) {
            const scene::SceneNode& node = scene_.nodes()[index];
            const std::string label = "[" + std::to_string(index) + "] " + node.name;
            drawSceneListSelectable(label.c_str(), static_cast<int32_t>(index), static_cast<int32_t>(SceneSelectionType::Node));
        }
        ImGui::EndChild();
    }

    if (ImGui::CollapsingHeader(("Meshes (" + std::to_string(scene_.meshes().size()) + ")").c_str())) {
        ImGui::BeginChild("MeshesScrollRegion", ImVec2(0, 140.0f * mainScale_), false, ImGuiWindowFlags_HorizontalScrollbar);
        for (size_t index = 0; index < scene_.meshes().size(); ++index) {
            const scene::SceneMesh& mesh = scene_.meshes()[index];
            const std::string label = "[" + std::to_string(index) + "] " + mesh.name +
                " (" + std::to_string(mesh.primitiveCount) + " primitives)";
            drawSceneListSelectable(label.c_str(), static_cast<int32_t>(index), static_cast<int32_t>(SceneSelectionType::Mesh));
        }
        ImGui::EndChild();
    }

    if (ImGui::CollapsingHeader(("Render Primitives (" + std::to_string(scene_.renderPrimitives().size()) + ")").c_str())) {
        ImGui::BeginChild(
            "RenderPrimitivesScrollRegion",
            ImVec2(0, 180.0f * mainScale_),
            false,
            ImGuiWindowFlags_HorizontalScrollbar);
        for (size_t index = 0; index < scene_.renderPrimitives().size(); ++index) {
            const scene::RenderPrimitive& primitive = scene_.renderPrimitives()[index];
            const bool selected =
                sceneSelection_.type == SceneSelectionType::RenderPrimitive &&
                sceneSelection_.index == static_cast<int32_t>(index);
            const std::string label = "[" + std::to_string(index) + "] " + primitive.name +
                " / prim " + std::to_string(primitive.primitiveIndex) +
                " / mat " + std::to_string(primitive.materialIndex);
            if (ImGui::Selectable(label.c_str(), selected)) {
                sceneSelection_ = SceneSelection{
                    .type = SceneSelectionType::RenderPrimitive,
                    .index = static_cast<int32_t>(index),
                    .meshIndex = primitive.meshIndex,
                    .primitiveIndex = primitive.primitiveIndex,
                };
            }
        }
        ImGui::EndChild();
    }

    if (ImGui::CollapsingHeader(("Materials (" + std::to_string(scene_.materials().size()) + ")").c_str())) {
        ImGui::BeginChild("MaterialsScrollRegion", ImVec2(0, 160.0f * mainScale_), false, ImGuiWindowFlags_HorizontalScrollbar);
        for (size_t index = 0; index < scene_.materials().size(); ++index) {
            const scene::RenderMaterial& material = scene_.materials()[index];
            const std::string label = "[" + std::to_string(index) + "] " + material.name;
            drawSceneListSelectable(label.c_str(), static_cast<int32_t>(index), static_cast<int32_t>(SceneSelectionType::Material));
        }
        ImGui::EndChild();
    }

    if (ImGui::CollapsingHeader(("Cameras (" + std::to_string(scene_.cameras().size()) + ")").c_str())) {
        ImGui::BeginChild("CamerasScrollRegion", ImVec2(0, 120.0f * mainScale_), false, ImGuiWindowFlags_HorizontalScrollbar);
        for (size_t index = 0; index < scene_.cameras().size(); ++index) {
            const scene::RenderCamera& camera = scene_.cameras()[index];
            const std::string label = "[" + std::to_string(index) + "] " + camera.name;
            drawSceneListSelectable(label.c_str(), static_cast<int32_t>(index), static_cast<int32_t>(SceneSelectionType::Camera));
        }
        ImGui::EndChild();
    }

    if (ImGui::CollapsingHeader(("Lights (" + std::to_string(scene_.lights().size()) + ")").c_str())) {
        drawEnvironmentControls();
        ImGui::Separator();
        ImGui::BeginChild("LightsScrollRegion", ImVec2(0, 120.0f * mainScale_), false, ImGuiWindowFlags_HorizontalScrollbar);
        for (size_t index = 0; index < scene_.lights().size(); ++index) {
            const scene::RenderLight& light = scene_.lights()[index];
            const std::string label = "[" + std::to_string(index) + "] " + light.name + " (" + light.type + ")";
            drawSceneListSelectable(label.c_str(), static_cast<int32_t>(index), static_cast<int32_t>(SceneSelectionType::Light));
        }
        if (scene_.lights().empty()) {
            ImGui::TextDisabled("No punctual lights.");
        }
        ImGui::EndChild();
    }

    if (ImGui::CollapsingHeader(("Textures (" + std::to_string(scene_.textures().size()) + ")").c_str())) {
        ImGui::BeginChild("TexturesScrollRegion", ImVec2(0, 120.0f * mainScale_), false, ImGuiWindowFlags_HorizontalScrollbar);
        for (size_t index = 0; index < scene_.textures().size(); ++index) {
            const scene::RenderTexture& texture = scene_.textures()[index];
            const std::string label = "[" + std::to_string(index) + "] " + texture.name +
                " -> image " + std::to_string(texture.imageIndex);
            drawSceneListSelectable(label.c_str(), static_cast<int32_t>(index), static_cast<int32_t>(SceneSelectionType::Texture));
        }
        ImGui::EndChild();
    }

    if (ImGui::CollapsingHeader(("Images (" + std::to_string(scene_.images().size()) + ")").c_str())) {
        ImGui::BeginChild("ImagesScrollRegion", ImVec2(0, 120.0f * mainScale_), false, ImGuiWindowFlags_HorizontalScrollbar);
        for (size_t index = 0; index < scene_.images().size(); ++index) {
            const scene::RenderImage& image = scene_.images()[index];
            const std::string source = image.uri.empty() ? image.name : image.uri;
            const std::string label = "[" + std::to_string(index) + "] " + source;
            drawSceneListSelectable(label.c_str(), static_cast<int32_t>(index), static_cast<int32_t>(SceneSelectionType::Image));
        }
        ImGui::EndChild();
    }
}

void EditorApplication::drawInspectorPanel()
{
    if (!inspectorOpen_) {
        return;
    }
    if (!ImGui::Begin("Inspector", &inspectorOpen_)) {
        ImGui::End();
        return;
    }

    if (!scene_.valid()) {
        ImGui::TextDisabled("No scene loaded");
        ImGui::End();
        return;
    }
    if (sceneSelection_.type == SceneSelectionType::None || sceneSelection_.index < 0) {
        ImGui::TextDisabled("No selection");
        ImGui::Separator();
        ImGui::TextWrapped("Select an element in the Scene Browser to view its properties.");
        ImGui::End();
        return;
    }

    switch (sceneSelection_.type) {
    case SceneSelectionType::Node: {
        if (static_cast<size_t>(sceneSelection_.index) >= scene_.nodes().size()) {
            break;
        }
        const scene::SceneNode& node = scene_.nodes()[static_cast<size_t>(sceneSelection_.index)];
        ImGui::Text("Node: %s", node.name.c_str());
        ImGui::Separator();
        ImGui::Text("Index: %d", sceneSelection_.index);
        ImGui::Text("Parent: %d", node.parent);
        ImGui::Text("Mesh: %d", node.meshIndex);
        ImGui::Text("Camera: %d", node.cameraIndex);
        ImGui::Text("Light: %d", node.lightIndex);
        ImGui::Text("Visible: %s", node.visible ? "true" : "false");
        const float3 translation(node.worldMatrix.a03, node.worldMatrix.a13, node.worldMatrix.a23);
        ImGui::Text("World translation: %s", scene::formatVec3(translation).c_str());
        break;
    }
    case SceneSelectionType::Mesh: {
        if (static_cast<size_t>(sceneSelection_.index) >= scene_.meshes().size()) {
            break;
        }
        const scene::SceneMesh& mesh = scene_.meshes()[static_cast<size_t>(sceneSelection_.index)];
        ImGui::Text("Mesh: %s", mesh.name.c_str());
        ImGui::Separator();
        ImGui::Text("Index: %d", sceneSelection_.index);
        ImGui::Text("Primitives: %llu", static_cast<unsigned long long>(mesh.primitiveCount));
        break;
    }
    case SceneSelectionType::RenderPrimitive: {
        if (static_cast<size_t>(sceneSelection_.index) >= scene_.renderPrimitives().size()) {
            break;
        }
        const scene::RenderPrimitive& primitive = scene_.renderPrimitives()[static_cast<size_t>(sceneSelection_.index)];
        ImGui::Text("Primitive: %s", primitive.name.c_str());
        ImGui::Separator();
        ImGui::Text("Mesh: %d", primitive.meshIndex);
        ImGui::Text("Primitive: %d", primitive.primitiveIndex);
        ImGui::Text("Material: %d", primitive.materialIndex);
        ImGui::Text("Vertices: %llu", static_cast<unsigned long long>(primitive.vertexCount));
        ImGui::Text("Indices: %llu", static_cast<unsigned long long>(primitive.indexCount));
        ImGui::Text("Triangles: %llu", static_cast<unsigned long long>(primitive.triangleCount));
        ImGui::Text("Meshlet Clusters: %zu", primitive.meshletClusters.size());
        ImGui::Text("Meshlet Vertex References: %zu", primitive.meshletVertices.size());
        ImGui::Text("Meshlet Triangle Indices: %zu", primitive.meshletTriangles.size());
        ImGui::Text("Meshlet LOD Levels: %zu", primitive.meshletLodLevels.size());
        ImGui::Text("Meshlet LOD Groups: %zu", primitive.meshletLodGroups.size());
        ImGui::Text("Meshlet LOD Clusters: %zu", primitive.meshletLodClusters.size());
        ImGui::Text("Meshlet LOD Vertex References: %zu", primitive.meshletLodVertices.size());
        ImGui::Text("Meshlet LOD Triangle Indices: %zu", primitive.meshletLodTriangles.size());
        if (primitive.localBounds.valid) {
            ImGui::Text("Bounds min: %s", scene::formatVec3(primitive.localBounds.min).c_str());
            ImGui::Text("Bounds max: %s", scene::formatVec3(primitive.localBounds.max).c_str());
        }
        break;
    }
    case SceneSelectionType::Material: {
        if (static_cast<size_t>(sceneSelection_.index) >= scene_.materials().size()) {
            break;
        }
        const scene::RenderMaterial& material = scene_.materials()[static_cast<size_t>(sceneSelection_.index)];
        ImGui::Text("Material: %s", material.name.c_str());
        ImGui::Separator();
        ImGui::Text("Base Color: %.3f, %.3f, %.3f, %.3f",
            material.baseColorFactor.x,
            material.baseColorFactor.y,
            material.baseColorFactor.z,
            material.baseColorFactor.w);
        ImGui::Text("Metallic: %.3f", material.metallicFactor);
        ImGui::Text("Roughness: %.3f", material.roughnessFactor);
        ImGui::Text("Emissive: %s", scene::formatVec3(material.emissiveFactor).c_str());
        ImGui::Text("Alpha Mode: %s", material.alphaMode.c_str());
        ImGui::Text("Double Sided: %s", material.doubleSided ? "true" : "false");
        break;
    }
    case SceneSelectionType::Camera: {
        if (static_cast<size_t>(sceneSelection_.index) >= scene_.cameras().size()) {
            break;
        }
        const scene::RenderCamera& camera = scene_.cameras()[static_cast<size_t>(sceneSelection_.index)];
        ImGui::Text("Camera: %s", camera.name.c_str());
        ImGui::Separator();
        ImGui::Text("Type: %s", scene::cameraTypeName(camera.type));
        ImGui::Text("Eye: %s", scene::formatVec3(camera.eye).c_str());
        ImGui::Text("Center: %s", scene::formatVec3(camera.center).c_str());
        ImGui::Text("Up: %s", scene::formatVec3(camera.up).c_str());
        ImGui::Text("Z Near: %.6f", camera.znear);
        ImGui::Text("Z Far: %.6f", camera.zfar);
        if (camera.type == scene::CameraType::Perspective) {
            ImGui::Text("Y FOV: %.3f rad", camera.yfov);
            ImGui::Text("Aspect: %.6f", camera.aspectRatio);
        } else {
            ImGui::Text("X Mag: %.6f", camera.xmag);
            ImGui::Text("Y Mag: %.6f", camera.ymag);
        }
        break;
    }
    case SceneSelectionType::Light: {
        if (static_cast<size_t>(sceneSelection_.index) >= scene_.lights().size()) {
            break;
        }
        const scene::RenderLight& light = scene_.lights()[static_cast<size_t>(sceneSelection_.index)];
        ImGui::Text("Light: %s", light.name.c_str());
        ImGui::Separator();
        ImGui::Text("Type: %s", light.type.c_str());
        ImGui::Text("Color: %s", scene::formatVec3(light.color).c_str());
        ImGui::Text("Intensity: %.3f", light.intensity);
        ImGui::Text("Range: %.3f", light.range);
        break;
    }
    case SceneSelectionType::Texture: {
        if (static_cast<size_t>(sceneSelection_.index) >= scene_.textures().size()) {
            break;
        }
        const scene::RenderTexture& texture = scene_.textures()[static_cast<size_t>(sceneSelection_.index)];
        ImGui::Text("Texture: %s", texture.name.c_str());
        ImGui::Separator();
        ImGui::Text("Image: %d", texture.imageIndex);
        ImGui::Text("Sampler: %d", texture.samplerIndex);
        break;
    }
    case SceneSelectionType::Image: {
        if (static_cast<size_t>(sceneSelection_.index) >= scene_.images().size()) {
            break;
        }
        const scene::RenderImage& image = scene_.images()[static_cast<size_t>(sceneSelection_.index)];
        ImGui::Text("Image: %s", image.name.c_str());
        ImGui::Separator();
        ImGui::TextWrapped("URI: %s", image.uri.empty() ? "-" : image.uri.c_str());
        ImGui::Text("MIME: %s", image.mimeType.empty() ? "-" : image.mimeType.c_str());
        ImGui::Text("Buffer View: %d", image.bufferView);
        ImGui::Text("Embedded bytes: %zu", image.encodedData.size());
        break;
    }
    case SceneSelectionType::None:
        break;
    }

    ImGui::End();
}

void EditorApplication::drawStatisticsPanel()
{
    if (!statisticsOpen_) {
        return;
    }
    if (!ImGui::Begin("Statistics", &statisticsOpen_)) {
        ImGui::End();
        return;
    }

    if (!scene_.valid()) {
        ImGui::TextDisabled("No scene loaded");
        ImGui::End();
        return;
    }

    const scene::SceneStats& stats = scene_.stats();
    if (ImGui::BeginTable("SceneStatisticsTable", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthFixed, 96.0f * mainScale_);
        ImGui::TableHeadersRow();

        auto addStat = [](const char* label, uint64_t value) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted(label);
            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%llu", static_cast<unsigned long long>(value));
        };
        addStat("Nodes", scene_.nodes().size());
        addStat("Meshes", stats.meshCount);
        addStat("Render Nodes", stats.renderNodeCount);
        addStat("Render Primitives", stats.primitiveCount);
        addStat("Materials", stats.materialCount);
        addStat("Triangles", stats.triangleCount);
        addStat("Meshlet Clusters", stats.meshletClusterCount);
        addStat("Meshlet Vertex References", stats.meshletVertexReferenceCount);
        addStat("Meshlet Triangle Indices", stats.meshletTriangleIndexCount);
        addStat("Meshlet LOD Levels", stats.meshletLodLevelCount);
        addStat("Meshlet LOD Groups", stats.meshletLodGroupCount);
        addStat("Meshlet LOD Clusters", stats.meshletLodClusterCount);
        addStat("Meshlet LOD Vertex References", stats.meshletLodVertexReferenceCount);
        addStat("Meshlet LOD Triangle Indices", stats.meshletLodTriangleIndexCount);
        addStat("Lights", scene_.lights().size());
        addStat("Textures", stats.textureCount);
        addStat("Images", stats.imageCount);
        ImGui::EndTable();
    }

    const scene::Bounds& bounds = scene_.bounds();
    ImGui::Separator();
    if (bounds.valid) {
        ImGui::Text("Bounds min: %s", scene::formatVec3(bounds.min).c_str());
        ImGui::Text("Bounds max: %s", scene::formatVec3(bounds.max).c_str());
    } else {
        ImGui::TextDisabled("Bounds: unavailable");
    }

    if (sceneRtx_ != nullptr && sceneRtx_->valid()) {
        const render::vulkan::SceneRtxStats& rtxStats = sceneRtx_->stats();
        ImGui::Separator();
        ImGui::Text(
            "RTX AS: %u BLAS, %u instances, %llu triangles",
            rtxStats.blasCount,
            rtxStats.instanceCount,
            static_cast<unsigned long long>(rtxStats.triangleCount));
        ImGui::Text(
            "RTX memory: geometry %llu bytes, AS %llu bytes, scratch %llu bytes",
            static_cast<unsigned long long>(rtxStats.geometryBytes),
            static_cast<unsigned long long>(rtxStats.accelerationStructureBytes),
            static_cast<unsigned long long>(rtxStats.scratchBytes));
    }

    if (ImGui::Button("Copy to Clipboard")) {
        ImGui::LogToClipboard();
        ImGui::LogText("Scene Statistics:\n");
        ImGui::LogText("Nodes: %zu\n", scene_.nodes().size());
        ImGui::LogText("Meshes: %llu\n", static_cast<unsigned long long>(stats.meshCount));
        ImGui::LogText("Render Nodes: %llu\n", static_cast<unsigned long long>(stats.renderNodeCount));
        ImGui::LogText("Render Primitives: %llu\n", static_cast<unsigned long long>(stats.primitiveCount));
        ImGui::LogText("Materials: %llu\n", static_cast<unsigned long long>(stats.materialCount));
        ImGui::LogText("Triangles: %llu\n", static_cast<unsigned long long>(stats.triangleCount));
        ImGui::LogText("Meshlet Clusters: %llu\n", static_cast<unsigned long long>(stats.meshletClusterCount));
        ImGui::LogText(
            "Meshlet Vertex References: %llu\n",
            static_cast<unsigned long long>(stats.meshletVertexReferenceCount));
        ImGui::LogText(
            "Meshlet Triangle Indices: %llu\n",
            static_cast<unsigned long long>(stats.meshletTriangleIndexCount));
        ImGui::LogText("Meshlet LOD Levels: %llu\n", static_cast<unsigned long long>(stats.meshletLodLevelCount));
        ImGui::LogText("Meshlet LOD Groups: %llu\n", static_cast<unsigned long long>(stats.meshletLodGroupCount));
        ImGui::LogText("Meshlet LOD Clusters: %llu\n", static_cast<unsigned long long>(stats.meshletLodClusterCount));
        ImGui::LogText(
            "Meshlet LOD Vertex References: %llu\n",
            static_cast<unsigned long long>(stats.meshletLodVertexReferenceCount));
        ImGui::LogText(
            "Meshlet LOD Triangle Indices: %llu\n",
            static_cast<unsigned long long>(stats.meshletLodTriangleIndexCount));
        ImGui::LogText("Lights: %zu\n", scene_.lights().size());
        ImGui::LogText("Textures: %llu\n", static_cast<unsigned long long>(stats.textureCount));
        ImGui::LogText("Images: %llu\n", static_cast<unsigned long long>(stats.imageCount));
        ImGui::LogFinish();
    }

    ImGui::End();
}

render::RenderGraphNode* EditorApplication::activePreviewRenderGraphNode()
{
    render::RenderGraphNode* node = activePreviewOutput_.empty()
        ? nullptr
        : findRenderGraphNodeForOutput(renderGraph_, activePreviewOutput_);
    if (node != nullptr) {
        return node;
    }

    const std::string fallbackOutput = renderGraph_.firstOutputName();
    if (fallbackOutput.empty() || fallbackOutput == activePreviewOutput_) {
        return nullptr;
    }
    return findRenderGraphNodeForOutput(renderGraph_, fallbackOutput);
}

bool EditorApplication::drawRuntimeSettingsForNode(
    render::RenderGraphNode& node,
    bool hideCameraSettings,
    bool showEmptyMessage)
{
    std::unique_ptr<render::RenderGraphPass> pass = render::createRenderGraphPass(node.type);
    if (pass == nullptr) {
        if (showEmptyMessage) {
            ImGui::TextDisabled("No runtime settings for this pass.");
        }
        return false;
    }

    const std::vector<render::RenderGraphRuntimeSetting> settings = pass->runtimeSettings();
    if (!hasVisibleRuntimeSettings(settings, hideCameraSettings)) {
        if (showEmptyMessage) {
            ImGui::TextDisabled("No runtime settings for this pass.");
        }
        return false;
    }

    render::RenderGraphProperties runtimeProperties = node.runtimeProperties.is_object()
        ? node.runtimeProperties
        : render::RenderGraphProperties::object();
    bool changed = false;
    bool invalidateHistory = false;
    ImGui::PushID(static_cast<int>(node.id));
    for (const render::RenderGraphRuntimeSetting& setting : settings) {
        if (hideCameraSettings && isCameraRuntimeSetting(setting)) {
            continue;
        }

        ImGui::PushID(setting.key.c_str());
        render::RenderGraphProperties newValue;
        if (drawRuntimeSettingControl(setting, runtimeSettingValue(node, setting), newValue)) {
            setNestedProperty(runtimeProperties, setting.key, std::move(newValue));
            changed = true;
            invalidateHistory = invalidateHistory || setting.invalidateHistory;
        }
        ImGui::PopID();
    }
    ImGui::PopID();

    if (changed) {
        renderGraph_.setNodeRuntimeProperties(node.id, std::move(runtimeProperties));
        if (invalidateHistory) {
            historyResources_.invalidateAll();
        }
        if (graphExecutor_ != nullptr && !renderGraph_.dirty()) {
            graphExecutor_->syncRuntimeProperties(renderGraph_);
        }
        viewportPreviewNeedsRender_ = true;
        renderGraphStatus_ = "Updated runtime setting";
    }

    return true;
}

void EditorApplication::drawCameraControls()
{
    render::RenderGraphNode* node = findBunnyWireframeNode(renderGraph_);
    if (node == nullptr) {
        ImGui::TextDisabled("No render camera controls for this graph.");
        return;
    }

    render::RenderGraphProperties properties = effectiveNodeProperties(*node);
    ensureCameraProperties(properties, scene_.bounds());
    render::RenderGraphProperties& camera = properties["camera"];

    bool changed = false;
    const float labelWidth = 64.0f * mainScale_;

    if (ImGui::CollapsingHeader("Projection", ImGuiTreeNodeFlags_DefaultOpen)) {
        std::string projection = camera["projection"].get<std::string>();
        int projectionType = projection == "orthographic" ? 1 : 0;

        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("Type");
        ImGui::SameLine(labelWidth);
        if (ImGui::RadioButton("Perspective", projectionType == 0)) {
            projectionType = 0;
            changed = true;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Orthographic", projectionType == 1)) {
            projectionType = 1;
            changed = true;
        }

        float fovDegrees = camera["fovDegrees"].get<float>();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("FOV");
        ImGui::SameLine(labelWidth);
        ImGui::PushItemWidth(-1.0f);
        if (ImGui::SliderFloat("##CameraFOV", &fovDegrees, 1.0f, 120.0f, "%.1f")) {
            camera["fovDegrees"] = std::clamp(fovDegrees, 1.0f, 120.0f);
            changed = true;
        }
        ImGui::PopItemWidth();

        float zClip[2] = {
            camera["znear"].get<float>(),
            camera["zfar"].get<float>(),
        };
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("Z-Clip");
        ImGui::SameLine(labelWidth);
        ImGui::PushItemWidth(-1.0f);
        if (ImGui::InputFloat2("##CameraZClip", zClip, "%.6f")) {
            zClip[0] = std::max(zClip[0], 0.0001f);
            zClip[1] = std::max(zClip[1], zClip[0] + 0.0001f);
            camera["znear"] = zClip[0];
            camera["zfar"] = zClip[1];
            changed = true;
        }
        ImGui::PopItemWidth();

        bool reversedZ = camera["reversedZ"].get<bool>();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("Depth");
        ImGui::SameLine(labelWidth);
        if (ImGui::Checkbox("Reversed Z", &reversedZ)) {
            camera["reversedZ"] = reversedZ;
            changed = true;
        }

        camera["projection"] = projectionType == 1 ? "orthographic" : "perspective";
    }

    if (ImGui::CollapsingHeader("Position", ImGuiTreeNodeFlags_DefaultOpen)) {
        float eye[3] = {};
        float center[3] = {};
        float up[3] = {};
        readVec3Property(camera, "eye", eye);
        readVec3Property(camera, "center", center);
        readVec3Property(camera, "up", up);

        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("Eye");
        ImGui::SameLine(labelWidth);
        ImGui::PushItemWidth(-1.0f);
        if (ImGui::InputFloat3("##CameraEye", eye, "%.6f")) {
            storeVec3Property(camera, "eye", eye);
            changed = true;
        }
        ImGui::PopItemWidth();

        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("Center");
        ImGui::SameLine(labelWidth);
        ImGui::PushItemWidth(-1.0f);
        if (ImGui::InputFloat3("##CameraCenter", center, "%.6f")) {
            storeVec3Property(camera, "center", center);
            changed = true;
        }
        ImGui::PopItemWidth();

        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("Up");
        ImGui::SameLine(labelWidth);
        ImGui::PushItemWidth(-1.0f);
        if (ImGui::InputFloat3("##CameraUp", up, "%.6f")) {
            storeVec3Property(camera, "up", up);
            changed = true;
        }
        ImGui::PopItemWidth();
    }

    if (changed) {
        applyBunnyCameraProperties(std::move(properties), "Updated render camera");
    }
}

void EditorApplication::drawEnvironmentControls()
{
    render::RenderGraphNode* node = activePreviewRenderGraphNode();
    if (node == nullptr) {
        ImGui::TextDisabled("Environment: no active preview render pass.");
        return;
    }
    if (node->type != "ScenePathTracePass") {
        ImGui::TextDisabled("Environment: active pass does not consume scene environment.");
        return;
    }

    render::RenderGraphProperties staticProperties = node->properties.is_object()
        ? node->properties
        : render::RenderGraphProperties::object();
    render::RenderGraphProperties staticEnvironment =
        staticProperties.contains("environment") && staticProperties["environment"].is_object()
        ? staticProperties["environment"]
        : render::RenderGraphProperties::object();
    render::RenderGraphProperties effectiveProperties = effectiveNodeProperties(*node);
    render::RenderGraphProperties effectiveEnvironment =
        effectiveProperties.contains("environment") && effectiveProperties["environment"].is_object()
        ? effectiveProperties["environment"]
        : render::RenderGraphProperties::object();

    const std::string path = staticEnvironment.contains("path")
        ? stringValueOr(staticEnvironment["path"], "")
        : std::string();
    bool enabled = effectiveEnvironment.contains("enabled")
        ? boolValueOr(effectiveEnvironment["enabled"], true)
        : true;
    bool visible = effectiveEnvironment.contains("visible")
        ? boolValueOr(effectiveEnvironment["visible"], true)
        : true;
    float intensity = effectiveEnvironment.contains("intensity")
        ? std::max(floatValueOr(effectiveEnvironment["intensity"], 1.0f), 0.0f)
        : 1.0f;
    float rotationDegrees = effectiveEnvironment.contains("rotationDegrees")
        ? floatValueOr(effectiveEnvironment["rotationDegrees"], 0.0f)
        : 0.0f;

    render::RenderGraphProperties runtimeProperties = node->runtimeProperties.is_object()
        ? node->runtimeProperties
        : render::RenderGraphProperties::object();
    bool changedRuntime = false;
    ImGui::TextUnformatted("Environment");
    ImGui::PushID("SceneEnvironment");

    if (ImGui::Checkbox("Enabled", &enabled)) {
        setNestedProperty(runtimeProperties, "environment.enabled", enabled);
        changedRuntime = true;
    }
    ImGui::SameLine();
    if (ImGui::Checkbox("Visible", &visible)) {
        setNestedProperty(runtimeProperties, "environment.visible", visible);
        changedRuntime = true;
    }

    ImGui::TextWrapped("HDRI: %s", path.empty() ? "-" : path.c_str());

    ImGui::PushItemWidth(-1.0f);
    if (ImGui::SliderFloat("Intensity", &intensity, 0.0f, 16.0f, "%.3f")) {
        setNestedProperty(runtimeProperties, "environment.intensity", std::max(intensity, 0.0f));
        changedRuntime = true;
    }
    if (ImGui::SliderFloat("Rotation", &rotationDegrees, -180.0f, 180.0f, "%.1f deg")) {
        setNestedProperty(runtimeProperties, "environment.rotationDegrees", rotationDegrees);
        changedRuntime = true;
    }
    ImGui::PopItemWidth();
    ImGui::PopID();

    if (!changedRuntime) {
        return;
    }

    bool updated = true;
    updated = renderGraph_.setNodeRuntimeProperties(node->id, std::move(runtimeProperties));
    if (updated) {
        historyResources_.invalidateAll();
        if (graphExecutor_ != nullptr && !renderGraph_.dirty()) {
            graphExecutor_->syncRuntimeProperties(renderGraph_);
        }
        viewportPreviewNeedsRender_ = true;
    }

    if (updated) {
        renderGraphStatus_ = "Updated scene environment";
    } else {
        renderGraphStatus_ = "Environment update failed";
    }
}

void EditorApplication::applyRuntimeNodeProperties(
    uint32_t nodeId,
    render::RenderGraphProperties properties,
    const char* status)
{
    render::RenderGraphNode* node = renderGraph_.findNode(nodeId);
    if (node == nullptr) {
        return;
    }

    render::RenderGraphProperties runtimeProperties = node->runtimeProperties.is_object()
        ? node->runtimeProperties
        : render::RenderGraphProperties::object();
    if (properties.is_object() && properties.contains("camera")) {
        runtimeProperties["camera"] = properties["camera"];
    } else {
        runtimeProperties = std::move(properties);
    }
    renderGraph_.setNodeRuntimeProperties(nodeId, std::move(runtimeProperties));
    historyResources_.invalidateAll();
    if (graphExecutor_ != nullptr && !renderGraph_.dirty()) {
        graphExecutor_->syncRuntimeProperties(renderGraph_);
    }
    viewportPreviewNeedsRender_ = true;
    if (status != nullptr) {
        renderGraphStatus_ = status;
    }
}
void EditorApplication::applyBunnyCameraProperties(render::RenderGraphProperties properties, const char* status)
{
    render::RenderGraphNode* node = findBunnyWireframeNode(renderGraph_);
    if (node == nullptr) {
        return;
    }
    applyRuntimeNodeProperties(node->id, std::move(properties), status);
}

void EditorApplication::drawSceneNode(int32_t nodeIndex)
{
    const std::vector<scene::SceneNode>& nodes = scene_.nodes();
    if (nodeIndex < 0 || static_cast<size_t>(nodeIndex) >= nodes.size()) {
        return;
    }

    const scene::SceneNode& node = nodes[static_cast<size_t>(nodeIndex)];
    const std::string label = "[" + std::to_string(nodeIndex) + "] " + node.name;
    std::string tags;
    if (node.meshIndex >= 0) {
        tags += "M";
        tags += std::to_string(node.meshIndex);
        tags += " ";
    }
    if (node.cameraIndex >= 0) {
        tags += "C";
        tags += std::to_string(node.cameraIndex);
        tags += " ";
    }
    if (node.lightIndex >= 0) {
        tags += "L";
        tags += std::to_string(node.lightIndex);
        tags += " ";
    }
    if (!node.visible) {
        tags += "hidden";
    }

    ImGui::TableNextRow();
    ImGui::TableNextColumn();

    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;
    const bool leaf = node.children.empty();
    if (leaf) {
        flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
    }
    if (sceneSelection_.type == SceneSelectionType::Node && sceneSelection_.index == nodeIndex) {
        flags |= ImGuiTreeNodeFlags_Selected;
    }

    const bool open = ImGui::TreeNodeEx(
        reinterpret_cast<void*>(static_cast<intptr_t>(nodeIndex)),
        flags,
        "%s",
        label.c_str());
    if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
        sceneSelection_ = SceneSelection{
            .type = SceneSelectionType::Node,
            .index = nodeIndex,
            .nodeIndex = nodeIndex,
            .meshIndex = node.meshIndex,
        };
    }
    if (ImGui::IsItemHovered()) {
        const float3 translation(node.worldMatrix.a03, node.worldMatrix.a13, node.worldMatrix.a23);
        ImGui::SetTooltip(
            "Node %d\nParent: %d\nWorld translation: %s",
            nodeIndex,
            node.parent,
            scene::formatVec3(translation).c_str());
    }
    ImGui::TableNextColumn();
    if (tags.empty()) {
        ImGui::TextDisabled("-");
    } else {
        ImGui::TextUnformatted(tags.c_str());
    }

    if (open && !leaf) {
        for (const int32_t child : node.children) {
            drawSceneNode(child);
        }
        ImGui::TreePop();
    }
}

void EditorApplication::drawViewportPanel()
{
    ImGui::Begin("Viewport");

    ImVec2 available = ImGui::GetContentRegionAvail();
    available.x = std::max(available.x, 1.0f);
    available.y = std::max(available.y, 1.0f);
    ImGui::InvisibleButton("ViewportCanvas", available);

    const ImVec2 min = ImGui::GetItemRectMin();
    const ImVec2 max = ImGui::GetItemRectMax();
    handleViewportCameraControls(min, max);
    const float width = max.x - min.x;
    const float height = max.y - min.y;
    const uint32_t previewWidth = std::clamp(
        static_cast<uint32_t>(std::ceil(width)),
        1u,
        kMaxViewportPreviewSize);
    const uint32_t previewHeight = std::clamp(
        static_cast<uint32_t>(std::ceil(height)),
        1u,
        kMaxViewportPreviewSize);
    const bool hasRhiPreview = updateViewportPreview(previewWidth, previewHeight);

    ImDrawList* drawList = ImGui::GetWindowDrawList();
    drawList->PushClipRect(min, max, true);
    drawList->AddRectFilled(min, max, IM_COL32(16, 18, 22, 255));

    if (hasRhiPreview) {
        drawList->AddImage(
            static_cast<ImTextureID>(reinterpret_cast<std::uintptr_t>(viewportDescriptor_)),
            min,
            max,
            ImVec2(0.0f, 0.0f),
            ImVec2(1.0f, 1.0f));
    } else {
        const float gridStep = 32.0f * mainScale_;
        for (float x = min.x; x < max.x; x += gridStep) {
            drawList->AddLine(ImVec2(x, min.y), ImVec2(x, max.y), IM_COL32(32, 37, 45, 255));
        }
        for (float y = min.y; y < max.y; y += gridStep) {
            drawList->AddLine(ImVec2(min.x, y), ImVec2(max.x, y), IM_COL32(32, 37, 45, 255));
        }

        const ImVec2 center(min.x + width * 0.5f, min.y + height * 0.52f);
        const float radius = std::max(std::min(width, height) * 0.28f, 24.0f * mainScale_);
        const ImVec2 p0(center.x, center.y - radius);
        const ImVec2 p1(center.x - radius * 0.9f, center.y + radius * 0.72f);
        const ImVec2 p2(center.x + radius * 0.9f, center.y + radius * 0.72f);

        drawList->AddTriangleFilled(p0, p1, p2, IM_COL32(71, 140, 255, 255));
        drawList->AddTriangle(p0, p1, p2, IM_COL32(225, 237, 255, 255), 2.0f * mainScale_);
    }

    drawList->AddRect(min, max, IM_COL32(58, 67, 80, 255));
    drawList->PopClipRect();

    ImGui::End();
}

void EditorApplication::handleViewportCameraControls(const ImVec2& min, const ImVec2& max)
{
    render::RenderGraphNode* node = findBunnyWireframeNode(renderGraph_);
    if (node == nullptr) {
        viewportCameraDragButton_ = kNoViewportCameraDragButton;
        return;
    }

    const bool hovered = ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);
    const ImVec2 size(max.x - min.x, max.y - min.y);
    ImGuiIO& io = ImGui::GetIO();

    if (hovered) {
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            viewportCameraDragButton_ = ImGuiMouseButton_Left;
        } else if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
            viewportCameraDragButton_ = ImGuiMouseButton_Right;
        } else if (ImGui::IsMouseClicked(ImGuiMouseButton_Middle)) {
            viewportCameraDragButton_ = ImGuiMouseButton_Middle;
        }
    }

    if (viewportCameraDragButton_ != kNoViewportCameraDragButton &&
        !ImGui::IsMouseDown(viewportCameraDragButton_)) {
        viewportCameraDragButton_ = kNoViewportCameraDragButton;
    }

    render::RenderGraphProperties properties = effectiveNodeProperties(*node);
    ensureCameraProperties(properties, scene_.bounds());
    render::RenderGraphProperties& camera = properties["camera"];
    bool changed = false;

    if (hovered && io.MouseWheel != 0.0f) {
        changed = dollyCamera(io.MouseWheel, camera) || changed;
    }

    if (hovered && !io.WantTextInput) {
        const bool alt = ImGui::IsKeyDown(ImGuiKey_LeftAlt) || ImGui::IsKeyDown(ImGuiKey_RightAlt);
        if (!alt) {
            const bool shift = ImGui::IsKeyDown(ImGuiKey_LeftShift) || ImGui::IsKeyDown(ImGuiKey_RightShift);
            const bool ctrl = ImGui::IsKeyDown(ImGuiKey_LeftCtrl) || ImGui::IsKeyDown(ImGuiKey_RightCtrl);
            float speedMultiplier = 1.0f;
            if (shift) {
                speedMultiplier *= kFastCameraMoveMultiplier;
            }
            if (ctrl) {
                speedMultiplier *= kSlowCameraMoveMultiplier;
            }

            const CameraFrame frame = cameraFrameFrom(camera);
            const float baseSpeed = std::max(frame.distance, 0.1f) * kKeyboardMoveRate * speedMultiplier;
            const float step = baseSpeed * std::max(io.DeltaTime, 0.0f);
            float forwardAmount = 0.0f;
            float rightAmount = 0.0f;
            if (ImGui::IsKeyDown(ImGuiKey_W)) {
                forwardAmount += step;
            }
            if (ImGui::IsKeyDown(ImGuiKey_S)) {
                forwardAmount -= step;
            }
            if (ImGui::IsKeyDown(ImGuiKey_D)) {
                rightAmount += step;
            }
            if (ImGui::IsKeyDown(ImGuiKey_A)) {
                rightAmount -= step;
            }
            changed = keyboardMoveCamera(rightAmount, forwardAmount, camera) || changed;
        }
    }

    if (viewportCameraDragButton_ != kNoViewportCameraDragButton) {
        const ImVec2 delta = io.MouseDelta;
        if (delta.x != 0.0f || delta.y != 0.0f) {
            if (viewportCameraDragButton_ == ImGuiMouseButton_Left) {
                float eye[3] = {};
                float center[3] = {};
                float up[3] = {};
                readVec3Property(camera, "eye", eye);
                readVec3Property(camera, "center", center);
                readVec3Property(camera, "up", up);
                if (orbitCamera(delta.x, delta.y, size.x, size.y, eye, center, up)) {
                    storeVec3Property(camera, "eye", eye);
                    storeVec3Property(camera, "up", up);
                    changed = true;
                }
            } else if (viewportCameraDragButton_ == ImGuiMouseButton_Right) {
                changed = dragDollyCamera(delta.x, delta.y, size.x, size.y, camera) || changed;
            } else if (viewportCameraDragButton_ == ImGuiMouseButton_Middle) {
                changed = panCamera(delta.x, delta.y, size.x, size.y, camera) || changed;
            }
        }
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeAll);
    }

    if (changed) {
        applyBunnyCameraProperties(std::move(properties), "Updated render camera from viewport");
    }
}

bool EditorApplication::updateViewportPreview(uint32_t width, uint32_t height)
{
    if (device_ == nullptr || graphExecutor_ == nullptr || viewportSampler_ == VK_NULL_HANDLE) {
        return false;
    }

    const std::string previewOutput = activePreviewOutput_.empty()
        ? renderGraph_.firstOutputName()
        : activePreviewOutput_;
    if (previewOutput.empty()) {
        renderGraphStatus_ = "No active preview output";
        return false;
    }

    const bool textureSizeMatches =
        viewportDescriptor_ != VK_NULL_HANDLE &&
        viewportTextureWidth_ == width &&
        viewportTextureHeight_ == height;
    const bool previewResourceAvailable = graphExecutor_->compiled() &&
        graphExecutor_->outputResource(previewOutput) != nullptr;

    if (viewportPreviewValid_ &&
        textureSizeMatches &&
        previewResourceAvailable &&
        !renderGraph_.dirty()) {
        pendingViewportPreviewWidth_ = width;
        pendingViewportPreviewHeight_ = height;
        viewportResizeStableFrameCount_ = 0;
        viewportPreviewNeedsRender_ = true;
        return true;
    }

    const bool canReusePreviewDuringResize =
        viewportPreviewValid_ &&
        viewportDescriptor_ != VK_NULL_HANDLE &&
        previewResourceAvailable &&
        !textureSizeMatches &&
        !renderGraph_.dirty();
    if (canReusePreviewDuringResize) {
        if (pendingViewportPreviewWidth_ != width || pendingViewportPreviewHeight_ != height) {
            pendingViewportPreviewWidth_ = width;
            pendingViewportPreviewHeight_ = height;
            viewportResizeStableFrameCount_ = 0;
            viewportPreviewNeedsRender_ = true;
            return true;
        }

        if (viewportResizeStableFrameCount_ < kViewportResizeSettleFrames) {
            ++viewportResizeStableFrameCount_;
            viewportPreviewNeedsRender_ = true;
            return true;
        }
    }

    StartupLogScope previewScope(
        "Viewport preview prepare " + std::to_string(width) + "x" + std::to_string(height));
    destroyViewportTexture();

    std::string log;
    render::RenderGraphCompileOptions compileOptions;
    compileOptions.extraOutputs.push_back(previewOutput);
    compileOptions.enablePreviewOutputAccess = true;
    render::Result result;
    {
        StartupLogScope scope("RenderGraph compile for preview output '" + previewOutput + "'");
        result = graphExecutor_->compile(*device_, renderGraph_, width, height, compileOptions, log);
    }
    renderGraphStatus_ = log;
    if (!result) {
        spdlog::error(
            "RenderGraph compile failed with Result {}: {}",
            render::resultToString(result),
            log);
        return false;
    }

    {
        StartupLogScope scope("Bind viewport preview output '" + previewOutput + "'");
        if (!bindViewportPreviewOutput(previewOutput)) {
            return false;
        }
    }

    viewportTextureWidth_ = width;
    viewportTextureHeight_ = height;
    viewportPreviewValid_ = true;
    viewportPreviewNeedsRender_ = true;
    pendingViewportPreviewWidth_ = width;
    pendingViewportPreviewHeight_ = height;
    viewportResizeStableFrameCount_ = 0;
    return true;
}
void EditorApplication::destroyViewportDescriptor()
{
    if (viewportDescriptor_ != VK_NULL_HANDLE && imguiRendererInitialized_) {
        ImGui_ImplVulkan_RemoveTexture(viewportDescriptor_);
    }
    viewportDescriptor_ = VK_NULL_HANDLE;
}

void EditorApplication::destroyViewportTexture()
{
    destroyViewportDescriptor();
    historyResources_.invalidateAll();
    viewportTextureWidth_ = 0;
    viewportTextureHeight_ = 0;
    pendingViewportPreviewWidth_ = 0;
    pendingViewportPreviewHeight_ = 0;
    viewportResizeStableFrameCount_ = 0;
    viewportPreviewValid_ = false;
    viewportPreviewNeedsRender_ = false;
}

bool EditorApplication::renderGraphPreview()
{
    if (!viewportPreviewValid_ || !viewportPreviewNeedsRender_) {
        return true;
    }
    if (graphExecutor_ == nullptr || commandBuffer_ == nullptr) {
        return false;
    }

    auto profileScope = profiler_.scope("RenderGraph Preview");
    if (!renderGraph_.dirty()) {
        graphExecutor_->syncRuntimeProperties(renderGraph_);
    }

    commandBuffer_->beginDebugLabel(render::DebugLabelDesc{
        .name = "RenderGraph Preview",
        .color = render::ColorValue{0.78f, 0.36f, 0.92f, 1.0f},
    });
    render::Result result = graphExecutor_->execute(*commandBuffer_, &historyResources_);
    profiler_.addRenderGraphStats(graphExecutor_->executionStats());
    commandBuffer_->endDebugLabel();
    if (!result) {
        renderGraphStatus_ = std::string("RenderGraph execute failed: ") + render::resultToString(result);
        spdlog::error("{}", renderGraphStatus_);
        viewportPreviewValid_ = false;
        viewportPreviewNeedsRender_ = false;
        return false;
    }

    result = graphExecutor_->transitionOutput(
        *commandBuffer_,
        activePreviewOutput_.empty() ? renderGraph_.firstOutputName() : activePreviewOutput_,
        render::ResourceState::ShaderRead);
    if (!result) {
        renderGraphStatus_ = std::string("RenderGraph output transition failed: ") + render::resultToString(result);
        spdlog::error("{}", renderGraphStatus_);
        viewportPreviewValid_ = false;
        viewportPreviewNeedsRender_ = false;
        return false;
    }

    renderGraph_.clearDirty();
    viewportPreviewNeedsRender_ = false;
    return true;
}

bool EditorApplication::renderVulkanFrame()
{
    if (swapchain_ == nullptr ||
        commandPool_ == nullptr ||
        commandBuffer_ == nullptr ||
        frameFence_ == nullptr ||
        imageAvailableSemaphore_ == nullptr ||
        graphicsQueue_ == nullptr) {
        return false;
    }

    render::Result result = frameFence_->wait();
    if (!result) {
        spdlog::error("frameFence wait failed with Result {}", render::resultToString(result));
        return false;
    }

    uint32_t imageIndex = 0;
    {
        auto profileScope = profiler_.scope("Acquire Swapchain Image");
        result = swapchain_->acquireNextImage(*imageAvailableSemaphore_, imageIndex);
    }
    if (!result) {
        if (render::hasError(result, render::Error::OutOfDate)) {
            swapchainOutOfDate_ = true;
            return true;
        }
        spdlog::error("acquireNextImage failed with Result {}", render::resultToString(result));
        return false;
    }
    if (imageIndex >= swapchainImageViews_.size() ||
        imageIndex >= swapchainImageStates_.size() ||
        imageIndex >= renderFinishedSemaphores_.size() ||
        renderFinishedSemaphores_[imageIndex] == nullptr) {
        spdlog::error("acquireNextImage returned invalid image index {}", imageIndex);
        return false;
    }

    {
        auto profileScope = profiler_.scope("Begin Command Buffer");
        result = frameFence_->reset();
        if (!result) {
            spdlog::error("frameFence reset failed with Result {}", render::resultToString(result));
            return false;
        }
        result = commandPool_->reset();
        if (!result) {
            spdlog::error("commandPool reset failed with Result {}", render::resultToString(result));
            return false;
        }
        result = commandBuffer_->begin();
        if (!result) {
            spdlog::error("commandBuffer begin failed with Result {}", render::resultToString(result));
            return false;
        }
        historyResources_.beginFrame(historyFrameIndex_++);
    }

    bool frameLabelOpen = true;
    commandBuffer_->beginDebugLabel(render::DebugLabelDesc{
        .name = "Metallic Editor Frame",
        .color = render::ColorValue{0.24f, 0.40f, 0.95f, 1.0f},
    });
    auto endFrameLabel = [&]() {
        if (frameLabelOpen) {
            commandBuffer_->endDebugLabel();
            frameLabelOpen = false;
        }
    };

    {
        auto profileScope = profiler_.scope("Record RenderGraph");
        if (!renderGraphPreview()) {
            endFrameLabel();
            return false;
        }
    }

    render::Texture* swapchainTexture = swapchain_->texture(imageIndex);
    if (swapchainTexture == nullptr || swapchainImageViews_[imageIndex] == nullptr) {
        endFrameLabel();
        return false;
    }

    render::TextureBarrierDesc toColor{
        .texture = swapchainTexture,
        .before = swapchainImageStates_[imageIndex],
        .after = render::ResourceState::ColorAttachment,
        .baseMip = 0,
        .mipCount = 1,
        .baseLayer = 0,
        .layerCount = 1,
    };
    commandBuffer_->barrier(render::BarrierDesc{
        .textures = &toColor,
        .textureCount = 1,
    });
    swapchainImageStates_[imageIndex] = render::ResourceState::ColorAttachment;

    const render::Rect renderArea{
        .x = 0,
        .y = 0,
        .width = swapchain_->width(),
        .height = swapchain_->height(),
    };
    render::RenderingAttachmentDesc colorAttachment{
        .view = swapchainImageViews_[imageIndex].get(),
        .state = render::ResourceState::ColorAttachment,
        .loadOp = render::LoadOp::Clear,
        .storeOp = render::StoreOp::Store,
        .clearColor = render::ColorValue{
            clearColor_[0],
            clearColor_[1],
            clearColor_[2],
            clearColor_[3],
        },
    };
    commandBuffer_->beginDebugLabel(render::DebugLabelDesc{
        .name = "Editor ImGui",
        .color = render::ColorValue{0.22f, 0.70f, 0.45f, 1.0f},
    });
    commandBuffer_->beginRendering(render::RenderingDesc{
        .renderArea = renderArea,
        .colorAttachments = &colorAttachment,
        .colorAttachmentCount = 1,
    });

    {
        auto profileScope = profiler_.scope("Record ImGui Draw");
        ImGui_ImplVulkan_RenderDrawData(
            ImGui::GetDrawData(),
            render::vulkan::nativeCommandBuffer(*commandBuffer_));
    }

    commandBuffer_->endRendering();
    commandBuffer_->endDebugLabel();

    render::TextureBarrierDesc toPresent{
        .texture = swapchainTexture,
        .before = render::ResourceState::ColorAttachment,
        .after = render::ResourceState::Present,
        .baseMip = 0,
        .mipCount = 1,
        .baseLayer = 0,
        .layerCount = 1,
    };
    commandBuffer_->barrier(render::BarrierDesc{
        .textures = &toPresent,
        .textureCount = 1,
    });
    swapchainImageStates_[imageIndex] = render::ResourceState::Present;

    endFrameLabel();
    {
        auto profileScope = profiler_.scope("End Command Buffer");
        result = commandBuffer_->end();
        if (!result) {
            spdlog::error("commandBuffer end failed with Result {}", render::resultToString(result));
            return false;
        }
    }

    render::CommandBuffer* commandBuffers[] = {commandBuffer_.get()};
    render::SwapchainSemaphoreSubmitDesc waitSemaphore{
        .semaphore = imageAvailableSemaphore_.get(),
        .stages = render::PipelineStageBits::ColorAttachment,
    };
    render::SwapchainSemaphoreSubmitDesc signalSemaphore{
        .semaphore = renderFinishedSemaphores_[imageIndex].get(),
        .stages = render::PipelineStageBits::AllCommands,
    };
    {
        auto profileScope = profiler_.scope("Submit Frame");
        result = graphicsQueue_->submit(render::QueueSubmitDesc{
            .waitSwapchainSemaphores = &waitSemaphore,
            .waitSwapchainSemaphoreCount = 1,
            .commandBuffers = commandBuffers,
            .commandBufferCount = 1,
            .signalSwapchainSemaphores = &signalSemaphore,
            .signalSwapchainSemaphoreCount = 1,
            .signalFence = frameFence_.get(),
        });
    }
    if (!result) {
        spdlog::error("graphicsQueue submit failed with Result {}", render::resultToString(result));
        return false;
    }
    {
        auto profileScope = profiler_.scope("Present");
        result = swapchain_->present(*graphicsQueue_, imageIndex, *renderFinishedSemaphores_[imageIndex]);
    }
    if (!result) {
        if (render::hasError(result, render::Error::OutOfDate)) {
            swapchainOutOfDate_ = true;
            return true;
        }
        spdlog::error("swapchain present failed with Result {}", render::resultToString(result));
        return false;
    }

    return true;
}

void EditorApplication::loadBuiltInSample(const char* sampleId)
{
    StartupLogScope scope(std::string("Load built-in sample '") + (sampleId != nullptr ? sampleId : "") + "'");

    render::RenderSampleLoadResult sample;
    std::string message;
    if (!render::loadBuiltInRenderSample(sampleId, sample, message)) {
        renderGraphStatus_ = message;
        spdlog::warn("[Startup] Built-in sample load failed: {}", message);
        return;
    }
    if (!startupScenePath_.empty() &&
        !render::setRenderSampleScenePath(sample, startupScenePath_, message)) {
        renderGraphStatus_ = message;
        spdlog::warn("[Startup] Built-in sample scene override failed: {}", message);
        return;
    }
    if (!startupStreamAssetPath_.empty()) {
        for (const std::string& target : sample.desc.scenePathTargets) {
            render::RenderGraphNode* node = sample.graph.findNode(target);
            if (node == nullptr || node->type != "GPUDrivenStreamAssetPass") {
                renderGraphStatus_ =
                    "StreamAsset path override requires a GPUDrivenStreamAssetPass target: " + target;
                spdlog::warn("[Startup] {}", renderGraphStatus_);
                return;
            }
            render::RenderGraphProperties properties = node->properties.is_object()
                ? node->properties
                : render::RenderGraphProperties::object();
            properties["streamAssetPath"] = startupStreamAssetPath_;
            if (!sample.graph.setNodeProperties(node->id, std::move(properties))) {
                renderGraphStatus_ = "Failed to apply StreamAsset path override to: " + target;
                spdlog::warn("[Startup] {}", renderGraphStatus_);
                return;
            }
        }
        sample.graph.clearDirty();
    }
    spdlog::info(
        "[Startup] Loaded sample '{}' graph='{}' scene='{}' previewOutput='{}'",
        sample.desc.name,
        sample.desc.graphPath,
        sample.desc.scenePath,
        sample.desc.previewOutput);

    renderGraph_ = std::move(sample.graph);
    graphEditorPositionsInitialized_ = false;
    selectedGraphNodeId_ = -1;
    selectedGraphLinkId_ = -1;
    viewportPreviewValid_ = false;
    historyResources_.invalidateAll();
    copyToBuffer(sample.desc.graphPath, graphFilePath_, sizeof(graphFilePath_));
    copyToBuffer(sample.desc.scenePath, sceneFilePath_, sizeof(sceneFilePath_));
    copyToBuffer(renderGraph_.firstOutputName(), graphOutputBuffer_, sizeof(graphOutputBuffer_));
    activePreviewOutput_ = sample.desc.previewOutput;
    copyToBuffer(activePreviewOutput_, previewOutputBuffer_, sizeof(previewOutputBuffer_));
    renderGraphStatus_ = "Loaded Sample: " + sample.desc.name;
    if (sample.desc.loadSceneInEditor) {
        loadScene();
        return;
    }

    clearSceneRtx();
    scene_.clear();
    sceneSelection_ = SceneSelection{};
    sceneStatus_ = "StreamAsset-only sample: editor scene loading skipped for " + sample.desc.scenePath;
    spdlog::info(
        "[Startup] Skipped editor scene and static RTX loading for StreamAsset-only sample '{}'",
        sample.desc.name);
}

void EditorApplication::resetDefaultRenderGraph()
{
    loadBuiltInSample(kDefaultRenderSampleId);
}

void EditorApplication::saveRenderGraph()
{
    std::string message;
    const std::filesystem::path path = resolveGraphAssetPath(graphFilePath_);
    if (!render::saveRenderGraphToFile(renderGraph_, path, message)) {
        renderGraphStatus_ = message;
        return;
    }
    renderGraphStatus_ = message;
}

void EditorApplication::loadRenderGraph()
{
    StartupLogScope scope(std::string("Load render graph '") + graphFilePath_ + "'");

    render::RenderGraph loadedGraph;
    std::string message;
    const std::filesystem::path path = resolveGraphAssetPath(graphFilePath_);
    if (!render::loadRenderGraphFromFile(path, loadedGraph, message)) {
        renderGraphStatus_ = message;
        spdlog::warn("[Startup] Render graph load failed: {}", message);
        return;
    }
    renderGraph_ = std::move(loadedGraph);
    graphEditorPositionsInitialized_ = false;
    selectedGraphNodeId_ = -1;
    selectedGraphLinkId_ = -1;
    viewportPreviewValid_ = false;
    historyResources_.invalidateAll();
    copyToBuffer(renderGraph_.firstOutputName(), graphOutputBuffer_, sizeof(graphOutputBuffer_));
    activePreviewOutput_ = renderGraph_.firstOutputName();
    copyToBuffer(activePreviewOutput_, previewOutputBuffer_, sizeof(previewOutputBuffer_));
    renderGraphStatus_ = message;

    const std::string scenePath = firstScenePathFromGraph(renderGraph_);
    if (!scenePath.empty()) {
        copyToBuffer(scenePath, sceneFilePath_, sizeof(sceneFilePath_));
        loadScene();
    }
}

void EditorApplication::loadDroppedScene(const std::filesystem::path& path)
{
    copyToBuffer(path.string(), sceneFilePath_, sizeof(sceneFilePath_));
    loadScene();
}

void EditorApplication::loadDroppedRenderGraph(const std::filesystem::path& path)
{
    copyToBuffer(path.string(), graphFilePath_, sizeof(graphFilePath_));
    loadRenderGraph();
}

void EditorApplication::chooseSceneFile()
{
    const std::filesystem::path currentPath = resolveSceneAssetPath(sceneFilePath_);
    std::string dialogError;
    const std::filesystem::path selectedPath = openSceneFileDialog(window_, currentPath, dialogError);
    if (selectedPath.empty()) {
        if (!dialogError.empty()) {
            sceneStatus_ = dialogError;
        }
        return;
    }

    loadDroppedScene(selectedPath);
}

void EditorApplication::chooseEnvironmentFile()
{
    const std::string currentEnvironmentPath = firstEnvironmentPathFromGraph(renderGraph_);
    std::filesystem::path initialPath = currentEnvironmentPath.empty()
        ? std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset"
        : resolveSceneAssetPath(currentEnvironmentPath.c_str());
    std::string dialogError;
    const std::filesystem::path selectedPath = openEnvironmentFileDialog(window_, initialPath, dialogError);
    if (selectedPath.empty()) {
        if (!dialogError.empty()) {
            renderGraphStatus_ = dialogError;
        }
        return;
    }

    applyEnvironmentToRenderGraph(selectedPath);
}

void EditorApplication::addRecentScenePath(const std::filesystem::path& path)
{
    if (path.empty()) {
        return;
    }
    std::error_code error;
    std::filesystem::path absolutePath = std::filesystem::weakly_canonical(path, error);
    if (error) {
        absolutePath = std::filesystem::absolute(path, error);
    }
    if (error) {
        absolutePath = path;
    }
    const std::string normalizedPath = absolutePath.string();
    recentScenePaths_.erase(
        std::remove_if(
            recentScenePaths_.begin(),
            recentScenePaths_.end(),
            [&](const std::filesystem::path& existing) {
                std::error_code existingError;
                std::filesystem::path normalizedExisting = std::filesystem::weakly_canonical(existing, existingError);
                if (existingError) {
                    normalizedExisting = std::filesystem::absolute(existing, existingError);
                }
                if (existingError) {
                    normalizedExisting = existing;
                }
                return normalizedExisting.string() == normalizedPath;
            }),
        recentScenePaths_.end());
    recentScenePaths_.insert(recentScenePaths_.begin(), absolutePath);
    constexpr size_t kMaxRecentScenePaths = 16;
    if (recentScenePaths_.size() > kMaxRecentScenePaths) {
        recentScenePaths_.resize(kMaxRecentScenePaths);
    }
}

void EditorApplication::applyLoadedSceneToRenderGraph(const std::filesystem::path& path)
{
    StartupLogScope scope("Synchronize loaded scene path into RenderGraph");

    const std::string graphScenePath = displayPathForProperty(path);
    bool changed = false;

    std::vector<uint32_t> sceneNodeIds;
    for (const render::RenderGraphNode& node : renderGraph_.nodes()) {
        if (isSceneAwareRenderPassType(node.type)) {
            sceneNodeIds.push_back(node.id);
        }
    }

    for (const uint32_t nodeId : sceneNodeIds) {
        render::RenderGraphNode* node = renderGraph_.findNode(nodeId);
        if (node == nullptr) {
            continue;
        }
        render::RenderGraphProperties properties = node->properties.is_object()
            ? node->properties
            : render::RenderGraphProperties::object();
        if (!properties.contains("path") || !properties["path"].is_string() || properties["path"] != graphScenePath) {
            properties["path"] = graphScenePath;
            changed = renderGraph_.setNodeProperties(node->id, std::move(properties)) || changed;
        }
    }

    if (!changed) {
        return;
    }

    historyResources_.invalidateAll();
    viewportPreviewValid_ = false;
    viewportPreviewNeedsRender_ = true;
    renderGraphStatus_ = "Synchronized RenderGraph scene path: " + graphScenePath;
}

void EditorApplication::applyLoadedSceneCamera()
{
    StartupLogScope scope("Apply loaded scene camera to RenderGraph");

    if (!scene_.valid() || scene_.cameras().empty()) {
        return;
    }

    const scene::RenderCamera* selectedCamera = nullptr;
    for (const scene::RenderCamera& camera : scene_.cameras()) {
        if (!camera.fallback) {
            selectedCamera = &camera;
            break;
        }
    }
    if (selectedCamera == nullptr) {
        selectedCamera = &scene_.cameras().front();
    }

    std::vector<uint32_t> sceneNodeIds;
    for (const render::RenderGraphNode& node : renderGraph_.nodes()) {
        if (isSceneAwareRenderPassType(node.type)) {
            sceneNodeIds.push_back(node.id);
        }
    }
    if (sceneNodeIds.empty()) {
        return;
    }

    constexpr double kPi = 3.14159265358979323846;
    bool changed = false;
    for (const uint32_t nodeId : sceneNodeIds) {
        render::RenderGraphNode* node = renderGraph_.findNode(nodeId);
        if (node == nullptr) {
            continue;
        }

        render::RenderGraphProperties runtimeProperties = node->runtimeProperties.is_object()
            ? node->runtimeProperties
            : render::RenderGraphProperties::object();
        render::RenderGraphProperties cameraProperties = runtimeProperties.contains("camera") &&
            runtimeProperties["camera"].is_object()
            ? runtimeProperties["camera"]
            : render::RenderGraphProperties::object();
        ensureCameraProperties(cameraProperties, scene_.bounds());
        cameraProperties["projection"] =
            selectedCamera->type == scene::CameraType::Orthographic ? "orthographic" : "perspective";
        const double yfov = selectedCamera->yfov > 0.0 ? selectedCamera->yfov : 0.7853981633974483;
        const double radius = std::max(static_cast<double>(scene_.bounds().radius()), 1.0);
        const double znear = std::max(selectedCamera->znear, radius * 0.001);
        const double zfar = selectedCamera->zfar > znear
            ? selectedCamera->zfar
            : std::max(znear + 0.001, radius * 100.0);
        cameraProperties["fovDegrees"] = static_cast<float>(std::clamp(yfov * 180.0 / kPi, 1.0, 179.0));
        cameraProperties["znear"] = static_cast<float>(znear);
        cameraProperties["zfar"] = static_cast<float>(zfar);
        if (selectedCamera->type == scene::CameraType::Orthographic) {
            cameraProperties["orthoHeight"] =
                static_cast<float>(std::max(selectedCamera->ymag * 2.0, 0.0001));
        }
        storeVec3Property(cameraProperties, "eye", selectedCamera->eye);
        storeVec3Property(cameraProperties, "center", selectedCamera->center);
        storeVec3Property(cameraProperties, "up", selectedCamera->up);

        runtimeProperties["camera"] = std::move(cameraProperties);
        changed = renderGraph_.setNodeRuntimeProperties(node->id, std::move(runtimeProperties)) || changed;
    }

    if (!changed) {
        return;
    }
    historyResources_.invalidateAll();
    if (graphExecutor_ != nullptr && !renderGraph_.dirty()) {
        graphExecutor_->syncRuntimeProperties(renderGraph_);
    }
    viewportPreviewNeedsRender_ = true;
    renderGraphStatus_ = "Applied glTF camera: " + selectedCamera->name;
}

void EditorApplication::applyEnvironmentToRenderGraph(const std::filesystem::path& path)
{
    if (path.empty()) {
        renderGraphStatus_ = "Environment path is empty.";
        return;
    }

    const std::string graphEnvironmentPath = displayPathForProperty(path);
    std::vector<uint32_t> environmentNodeIds;
    for (const render::RenderGraphNode& node : renderGraph_.nodes()) {
        if (node.type == "ScenePathTracePass") {
            environmentNodeIds.push_back(node.id);
        }
    }

    if (environmentNodeIds.empty()) {
        renderGraphStatus_ = "No ScenePathTracePass found for environment map.";
        return;
    }

    size_t updatedCount = 0;
    for (const uint32_t nodeId : environmentNodeIds) {
        render::RenderGraphNode* node = renderGraph_.findNode(nodeId);
        if (node == nullptr) {
            continue;
        }

        render::RenderGraphProperties properties = node->properties.is_object()
            ? node->properties
            : render::RenderGraphProperties::object();
        render::RenderGraphProperties environment = properties.contains("environment") &&
            properties["environment"].is_object()
            ? properties["environment"]
            : render::RenderGraphProperties::object();
        if (!environment.contains("enabled")) {
            environment["enabled"] = true;
        }
        if (!environment.contains("visible")) {
            environment["visible"] = true;
        }
        if (!environment.contains("intensity")) {
            environment["intensity"] = 1.0f;
        }
        if (!environment.contains("rotationDegrees")) {
            environment["rotationDegrees"] = 0.0f;
        }
        environment["path"] = graphEnvironmentPath;
        properties["environment"] = std::move(environment);
        if (renderGraph_.setNodeProperties(node->id, std::move(properties))) {
            ++updatedCount;
        }
    }

    if (updatedCount == 0) {
        renderGraphStatus_ = "Environment update failed.";
        return;
    }

    historyResources_.invalidateAll();
    viewportPreviewValid_ = false;
    viewportPreviewNeedsRender_ = true;
    renderGraphStatus_ = "Loaded environment: " + graphEnvironmentPath +
        " (" + std::to_string(updatedCount) + " path trace pass" +
        (updatedCount == 1 ? ")" : "es)");
}

void EditorApplication::loadScene()
{
    StartupLogScope scope(std::string("Editor scene load '") + sceneFilePath_ + "'");

    clearSceneRtx();
    historyResources_.invalidateAll();

    const std::filesystem::path path = resolveSceneAssetPath(sceneFilePath_);
    if (path.empty()) {
        sceneStatus_ = "Scene path is empty.";
        return;
    }

    if (!scene_.load(path)) {
        const scene::LoadResult& loadResult = scene_.lastLoadResult();
        sceneStatus_ = loadResult.error.empty()
            ? "Failed to load scene."
            : "Failed to load scene: " + loadResult.error;
        spdlog::warn("[Startup] Scene load failed: {}", sceneStatus_);
        return;
    }

    sceneSelection_ = SceneSelection{};
    copyToBuffer(displayPathForProperty(path), sceneFilePath_, sizeof(sceneFilePath_));
    addRecentScenePath(path);
    applyLoadedSceneToRenderGraph(path);
    applyLoadedSceneCamera();
    buildSceneRtx();

    const scene::SceneStats& stats = scene_.stats();
    sceneStatus_ = "Loaded " + path.string() + " (" + std::to_string(stats.renderNodeCount) +
        " render nodes, " + std::to_string(scene_.nodes().size()) + " scene nodes).";
    spdlog::info(
        "[Startup] Editor scene loaded nodes={} renderNodes={} primitives={} triangles={} images={} textures={}",
        scene_.nodes().size(),
        stats.renderNodeCount,
        stats.primitiveCount,
        stats.triangleCount,
        stats.imageCount,
        stats.textureCount);
}

void EditorApplication::buildSceneRtx()
{
    StartupLogScope scope("Editor RTX acceleration structure build");

    if (sceneRtx_ == nullptr) {
        sceneRtx_ = std::make_unique<render::vulkan::SceneRtxBuilder>();
    }
    if (device_ == nullptr || graphicsQueue_ == nullptr) {
        sceneRtxStatus_ = "RTX AS build failed: RHI device is not initialized.";
        return;
    }
    if (!scene_.valid()) {
        sceneRtxStatus_ = "RTX AS build failed: load a glTF scene first.";
        return;
    }

    std::string log;
    const render::Result result = sceneRtx_->build(*device_, *graphicsQueue_, scene_, log);
    sceneRtxStatus_ = log.empty()
        ? std::string("RTX AS build returned ") + render::resultToString(result)
        : log;
    if (result) {
        const render::vulkan::SceneRtxStats& stats = sceneRtx_->stats();
        spdlog::info(
            "[Startup] Editor RTX build completed blas={} instances={} triangles={} asBytes={} scratchBytes={}",
            stats.blasCount,
            stats.instanceCount,
            stats.triangleCount,
            stats.accelerationStructureBytes,
            stats.scratchBytes);
    } else {
        spdlog::warn("[Startup] Editor RTX build failed: {}", sceneRtxStatus_);
    }
}

void EditorApplication::clearSceneRtx()
{
    if (sceneRtx_ != nullptr) {
        sceneRtx_->clear();
    }
    sceneRtxStatus_ = "RTX AS not built.";
}

void EditorApplication::addRenderGraphNode(std::string type, ImVec2 screenPosition)
{
    if (type.empty()) {
        return;
    }

    const std::string nodeName = makeUniqueNodeName(renderGraph_, type);
    render::RenderGraphProperties properties = defaultPropertiesForPass(type);
    const float fallbackOffset = static_cast<float>(renderGraph_.nodes().size()) * 34.0f * mainScale_;
    render::RenderGraphNode* node = renderGraph_.addNode(
        std::move(type),
        nodeName,
        std::move(properties),
        72.0f * mainScale_ + fallbackOffset,
        96.0f * mainScale_ + fallbackOffset);
    if (node == nullptr) {
        renderGraphStatus_ = "Failed to add render pass node";
        return;
    }

    if (screenPosition.x >= 0.0f && screenPosition.y >= 0.0f) {
        ImNodes::SetNodeScreenSpacePos(static_cast<int>(node->id), screenPosition);
        const ImVec2 gridPosition = ImNodes::GetNodeGridSpacePos(static_cast<int>(node->id));
        renderGraph_.setNodePosition(node->id, gridPosition.x, gridPosition.y);
    } else {
        graphEditorPositionsInitialized_ = false;
    }

    selectedGraphNodeId_ = static_cast<int>(node->id);
    selectedGraphLinkId_ = -1;
    viewportPreviewValid_ = false;
    renderGraphStatus_ = std::string("Added ") + node->name;
}

void EditorApplication::markRenderGraphOutput(std::string outputName)
{
    if (outputName.empty()) {
        return;
    }

    renderGraph_.clearOutputs();
    if (!renderGraph_.markOutput(outputName)) {
        renderGraphStatus_ = std::string("Invalid graph output '") + outputName + "'";
        return;
    }
    copyToBuffer(outputName, graphOutputBuffer_, sizeof(graphOutputBuffer_));
    viewportPreviewValid_ = false;
    renderGraphStatus_ = std::string("Graph output set to ") + outputName;
}

void EditorApplication::setActivePreviewOutput(std::string outputName)
{
    if (outputName.empty()) {
        return;
    }

    activePreviewOutput_ = std::move(outputName);
    copyToBuffer(activePreviewOutput_, previewOutputBuffer_, sizeof(previewOutputBuffer_));
    if (graphExecutor_ != nullptr &&
        graphExecutor_->compiled() &&
        graphExecutor_->outputResource(activePreviewOutput_) != nullptr &&
        bindViewportPreviewOutput(activePreviewOutput_)) {
        viewportPreviewValid_ = true;
        viewportPreviewNeedsRender_ = true;
    } else {
        destroyViewportDescriptor();
        viewportPreviewValid_ = false;
        viewportPreviewNeedsRender_ = true;
    }
    renderGraphStatus_ = std::string("Preview output set to ") + activePreviewOutput_;
}

bool EditorApplication::bindViewportPreviewOutput(std::string_view outputName)
{
    if (graphExecutor_ == nullptr || viewportSampler_ == VK_NULL_HANDLE) {
        return false;
    }
    render::RenderGraphResource* output = graphExecutor_->outputResource(outputName);
    if (output == nullptr || output->view == nullptr) {
        renderGraphStatus_ = std::string("RenderGraph preview output texture is not available: ") + std::string(outputName);
        return false;
    }

    const VkImageView imageView = render::vulkan::nativeImageView(*output->view);
    if (imageView == VK_NULL_HANDLE) {
        renderGraphStatus_ = "RenderGraph preview output image view is not available";
        return false;
    }

    destroyViewportDescriptor();
    viewportDescriptor_ = ImGui_ImplVulkan_AddTexture(
        viewportSampler_,
        imageView,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    if (viewportDescriptor_ == VK_NULL_HANDLE) {
        renderGraphStatus_ = "ImGui failed to allocate viewport descriptor";
        return false;
    }
    return true;
}
int EditorApplication::graphInputAttributeId(const render::RenderGraphNode& node, uint32_t fieldIndex) const
{
    return static_cast<int>(node.id * 100u + 1u + fieldIndex);
}

int EditorApplication::graphOutputAttributeId(const render::RenderGraphNode& node, uint32_t fieldIndex) const
{
    return static_cast<int>(node.id * 100u + 51u + fieldIndex);
}

void EditorApplication::drawRenderGraphNode(const render::RenderGraphNode& node)
{
    std::unique_ptr<render::RenderGraphPass> pass = render::createRenderGraphPass(node.type);
    render::RenderPassReflection reflection;
    if (pass != nullptr) {
        pass->setProperties(node.properties);
        reflection = pass->reflect(render::RenderGraphCompileContext{});
    }

    ImNodes::BeginNode(static_cast<int>(node.id));
    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted(node.name.c_str());
    ImGui::SameLine();
    ImGui::TextDisabled("(%s)", node.type.c_str());
    ImNodes::EndNodeTitleBar();

    uint32_t inputIndex = 0;
    uint32_t outputIndex = 0;
    for (const render::RenderGraphField& field : reflection.fields()) {
        if (field.visibility == render::RenderGraphFieldVisibility::Input) {
            ImNodes::BeginInputAttribute(graphInputAttributeId(node, inputIndex++));
            ImGui::TextUnformatted(field.name.c_str());
            ImGui::SameLine();
            ImGui::TextDisabled("%s", renderGraphFieldTag(field).c_str());
            setRenderGraphFieldTooltip(field);
            ImNodes::EndInputAttribute();
        }
    }

    if (node.type == "ClearColorPass" && node.properties.contains("color")) {
        ImNodes::BeginStaticAttribute(static_cast<int>(node.id * 100u + 90u));
        const auto& color = node.properties["color"];
        ImGui::ColorButton(
            "##ClearPreview",
            ImVec4(
                color.size() > 0 ? color[0].get<float>() : 0.0f,
                color.size() > 1 ? color[1].get<float>() : 0.0f,
                color.size() > 2 ? color[2].get<float>() : 0.0f,
                color.size() > 3 ? color[3].get<float>() : 1.0f),
            ImGuiColorEditFlags_NoTooltip,
            ImVec2(34.0f * mainScale_, 16.0f * mainScale_));
        ImNodes::EndStaticAttribute();
    }

    for (const render::RenderGraphField& field : reflection.fields()) {
        if (field.visibility == render::RenderGraphFieldVisibility::Output) {
            const std::string fullName = render::makeRenderGraphFieldName(node.name, field.name);
            const bool markedOutput = isMarkedRenderGraphOutput(renderGraph_, fullName);
            const bool previewOutput = fullName == activePreviewOutput_;
            const int attributeId = graphOutputAttributeId(node, outputIndex++);
            if (markedOutput) {
                ImNodes::PushColorStyle(ImNodesCol_Pin, IM_COL32(231, 65, 65, 255));
                ImNodes::PushColorStyle(ImNodesCol_PinHovered, IM_COL32(255, 112, 112, 255));
            } else if (previewOutput) {
                ImNodes::PushColorStyle(ImNodesCol_Pin, IM_COL32(82, 196, 126, 255));
                ImNodes::PushColorStyle(ImNodesCol_PinHovered, IM_COL32(125, 230, 158, 255));
            }
            ImNodes::BeginOutputAttribute(
                attributeId,
                markedOutput || previewOutput ? ImNodesPinShape_QuadFilled : ImNodesPinShape_CircleFilled);
            std::string label = field.name;
            label += "  ";
            label += renderGraphFieldTag(field);
            if (markedOutput) {
                label += "  [Graph Output]";
            }
            if (previewOutput) {
                label += "  [Preview]";
            }
            const float textWidth = ImGui::CalcTextSize(label.c_str()).x;
            ImGui::Indent(std::max(90.0f * mainScale_ - textWidth, 0.0f));
            ImGui::TextUnformatted(label.c_str());
            setRenderGraphFieldTooltip(field);
            ImNodes::EndOutputAttribute();
            if (markedOutput || previewOutput) {
                ImNodes::PopColorStyle();
                ImNodes::PopColorStyle();
            }
        }
    }

    ImNodes::EndNode();
}

void EditorApplication::drawRenderGraphEditorWindow()
{
    if (!renderGraphEditorOpen_) {
        return;
    }

    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(
        ImVec2(viewport->WorkPos.x + 48.0f * mainScale_, viewport->WorkPos.y + 48.0f * mainScale_),
        ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(1220.0f * mainScale_, 760.0f * mainScale_), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("Render Graph Editor", &renderGraphEditorOpen_, ImGuiWindowFlags_NoDocking)) {
        ImGui::End();
        return;
    }

    if (ImGui::Button("New Graph")) {
        resetDefaultRenderGraph();
    }
    ImGui::SameLine();
    if (ImGui::Button("Load")) {
        loadRenderGraph();
    }
    ImGui::SameLine();
    if (ImGui::Button("Save")) {
        saveRenderGraph();
    }
    ImGui::SameLine();
    if (ImGui::Button("Validate")) {
        std::string log;
        renderGraph_.validate(log);
        renderGraphStatus_ = log;
    }
    if (!renderGraphStatus_.empty()) {
        ImGui::SameLine();
        ImGui::TextDisabled("%s", renderGraphStatus_.c_str());
    }

    ImGui::Separator();

    const ImGuiStyle& style = ImGui::GetStyle();
    const float spacing = style.ItemSpacing.x;
    const ImVec2 available = ImGui::GetContentRegionAvail();
    const float bottomHeight = std::min(
        218.0f * mainScale_,
        std::max(150.0f * mainScale_, available.y * 0.42f));
    const float topHeight = std::max(260.0f * mainScale_, available.y - bottomHeight - spacing);
    const float sideWidth = std::min(
        330.0f * mainScale_,
        std::max(260.0f * mainScale_, available.x * 0.38f));
    const float canvasWidth = std::max(360.0f * mainScale_, available.x - sideWidth - spacing);

    ImGui::BeginChild("GraphCanvasPanel", ImVec2(canvasWidth, topHeight), true);
    ImGui::TextUnformatted("Graph Editor");
    ImGui::Separator();
    drawRenderGraphPanel();
    ImGui::EndChild();

    ImGui::SameLine();

    ImGui::BeginChild("RenderUiPanel", ImVec2(0.0f, topHeight), true);
    ImGui::TextUnformatted("Render UI");
    ImGui::Separator();
    drawRenderGraphRenderUiPanel();
    ImGui::EndChild();

    ImGui::BeginChild("GraphSettingsPanel", ImVec2(sideWidth, 0.0f), true);
    ImGui::TextUnformatted("Graph Editor Settings");
    ImGui::Separator();
    drawRenderGraphSettingsPanel();
    ImGui::EndChild();

    ImGui::SameLine();

    ImGui::BeginChild("RenderPassesPanel", ImVec2(0.0f, 0.0f), true);
    ImGui::TextUnformatted("Render Passes");
    ImGui::Separator();
    drawRenderPassesPanel();
    ImGui::EndChild();

    ImGui::End();
}

void EditorApplication::drawRenderGraphPanel()
{
    struct AttributeInfo {
        std::string fullName;
        render::RenderGraphFieldVisibility visibility = render::RenderGraphFieldVisibility::Output;
        render::RenderGraphResourceType resourceType = render::RenderGraphResourceType::Texture2D;
        render::RenderGraphResourceAccess access = render::RenderGraphResourceAccess::None;
        std::string tag;
    };
    std::unordered_map<int, AttributeInfo> attributes;
    std::unordered_map<std::string, int> inputAttributeIds;
    std::unordered_map<std::string, int> outputAttributeIds;

    for (const render::RenderGraphNode& node : renderGraph_.nodes()) {
        std::unique_ptr<render::RenderGraphPass> pass = render::createRenderGraphPass(node.type);
        if (pass == nullptr) {
            continue;
        }
        pass->setProperties(node.properties);
        const render::RenderPassReflection reflection = pass->reflect(render::RenderGraphCompileContext{});
        uint32_t inputIndex = 0;
        uint32_t outputIndex = 0;
        for (const render::RenderGraphField& field : reflection.fields()) {
            const std::string fullName = render::makeRenderGraphFieldName(node.name, field.name);
            AttributeInfo info{
                .fullName = fullName,
                .visibility = field.visibility,
                .resourceType = field.resourceType,
                .access = field.access,
                .tag = renderGraphFieldTag(field),
            };
            if (field.visibility == render::RenderGraphFieldVisibility::Input) {
                const int attrId = graphInputAttributeId(node, inputIndex++);
                attributes.emplace(attrId, info);
                inputAttributeIds.emplace(fullName, attrId);
            } else {
                const int attrId = graphOutputAttributeId(node, outputIndex++);
                attributes.emplace(attrId, info);
                outputAttributeIds.emplace(fullName, attrId);
            }
        }
    }

    ImNodes::BeginNodeEditor();
    ImNodes::PushAttributeFlag(ImNodesAttributeFlags_EnableLinkDetachWithDragClick);
    if (!graphEditorPositionsInitialized_) {
        for (const render::RenderGraphNode& node : renderGraph_.nodes()) {
            ImNodes::SetNodeGridSpacePos(
                static_cast<int>(node.id),
                ImVec2(node.uiX, node.uiY));
        }
        graphEditorPositionsInitialized_ = true;
    }

    for (const render::RenderGraphNode& node : renderGraph_.nodes()) {
        drawRenderGraphNode(node);
    }
    ImNodes::PopAttributeFlag();

    for (const render::RenderGraphEdge& edge : renderGraph_.edges()) {
        const std::string src = render::makeRenderGraphFieldName(edge.srcPass, edge.srcField);
        const std::string dst = render::makeRenderGraphFieldName(edge.dstPass, edge.dstField);
        const auto srcAttr = outputAttributeIds.find(src);
        const auto dstAttr = inputAttributeIds.find(dst);
        if (srcAttr != outputAttributeIds.end() && dstAttr != inputAttributeIds.end()) {
            ImNodes::Link(static_cast<int>(edge.id), srcAttr->second, dstAttr->second);
        }
    }

    ImNodes::MiniMap(0.16f, ImNodesMiniMapLocation_BottomRight);
    ImNodes::EndNodeEditor();

    if (ImGui::BeginDragDropTarget()) {
        if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload(kRenderPassDragPayload)) {
            const char* payloadData = static_cast<const char*>(payload->Data);
            if (payloadData != nullptr && payload->DataSize > 1) {
                addRenderGraphNode(
                    std::string(payloadData, static_cast<size_t>(payload->DataSize - 1)),
                    ImGui::GetMousePos());
            }
        }
        ImGui::EndDragDropTarget();
    }

    for (const render::RenderGraphNode& node : renderGraph_.nodes()) {
        const ImVec2 position = ImNodes::GetNodeGridSpacePos(static_cast<int>(node.id));
        renderGraph_.setNodePosition(node.id, position.x, position.y);
    }

    int hoveredAttribute = 0;
    if (ImNodes::IsPinHovered(&hoveredAttribute)) {
        const auto hovered = attributes.find(hoveredAttribute);
        if (hovered != attributes.end()) {
            ImGui::SetTooltip(
                "%s\n%s",
                hovered->second.fullName.c_str(),
                hovered->second.tag.c_str());
            if (hovered->second.visibility == render::RenderGraphFieldVisibility::Output &&
                ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
                copyToBuffer(hovered->second.fullName, graphOutputBuffer_, sizeof(graphOutputBuffer_));
                ImGui::OpenPopup("RenderGraphOutputPinMenu");
            }
        }
    }

    if (ImGui::BeginPopup("RenderGraphOutputPinMenu")) {
        ImGui::TextUnformatted(graphOutputBuffer_);
        ImGui::Separator();
        if (ImGui::MenuItem("Preview This Output")) {
            setActivePreviewOutput(graphOutputBuffer_);
        }
        if (ImGui::MenuItem("Set Graph Output")) {
            markRenderGraphOutput(graphOutputBuffer_);
        }
        ImGui::EndPopup();
    }

    int startedAttribute = 0;
    int endedAttribute = 0;
    if (ImNodes::IsLinkCreated(&startedAttribute, &endedAttribute)) {
        const auto started = attributes.find(startedAttribute);
        const auto ended = attributes.find(endedAttribute);
        if (started != attributes.end() && ended != attributes.end()) {
            const AttributeInfo* src = &started->second;
            const AttributeInfo* dst = &ended->second;
            if (src->visibility == render::RenderGraphFieldVisibility::Input &&
                dst->visibility == render::RenderGraphFieldVisibility::Output) {
                std::swap(src, dst);
            }
            if (src->visibility == render::RenderGraphFieldVisibility::Output &&
                dst->visibility == render::RenderGraphFieldVisibility::Input) {
                if (src->resourceType != dst->resourceType) {
                    renderGraphStatus_ = std::string("Cannot link ") +
                        renderGraphResourceTypeName(src->resourceType) +
                        " output to " +
                        renderGraphResourceTypeName(dst->resourceType) +
                        " input";
                } else if (renderGraph_.addEdge(src->fullName, dst->fullName) == nullptr) {
                    renderGraphStatus_ = "Link already exists or endpoint is invalid";
                } else {
                    viewportPreviewValid_ = false;
                }
            }
        }
    }

    int destroyedLink = 0;
    if (ImNodes::IsLinkDestroyed(&destroyedLink)) {
        renderGraph_.removeEdge(static_cast<uint32_t>(destroyedLink));
        viewportPreviewValid_ = false;
    }

    const int selectedNodeCount = ImNodes::NumSelectedNodes();
    if (selectedNodeCount > 0) {
        std::vector<int> selectedNodes(static_cast<size_t>(selectedNodeCount));
        ImNodes::GetSelectedNodes(selectedNodes.data());
        selectedGraphNodeId_ = selectedNodes.front();
        selectedGraphLinkId_ = -1;
    }
    const int selectedLinkCount = ImNodes::NumSelectedLinks();
    if (selectedLinkCount > 0) {
        std::vector<int> selectedLinks(static_cast<size_t>(selectedLinkCount));
        ImNodes::GetSelectedLinks(selectedLinks.data());
        selectedGraphLinkId_ = selectedLinks.front();
        selectedGraphNodeId_ = -1;
    }

    if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) &&
        ImGui::IsKeyPressed(ImGuiKey_Delete)) {
        if (selectedGraphLinkId_ >= 0 && renderGraph_.removeEdge(static_cast<uint32_t>(selectedGraphLinkId_))) {
            selectedGraphLinkId_ = -1;
            viewportPreviewValid_ = false;
        } else if (selectedGraphNodeId_ >= 0 &&
            renderGraph_.removeNode(static_cast<uint32_t>(selectedGraphNodeId_))) {
            selectedGraphNodeId_ = -1;
            viewportPreviewValid_ = false;
        }
    }
}

void EditorApplication::drawRenderGraphSettingsPanel()
{
    const std::string graphName = renderGraph_.name();
    if (ImGui::BeginCombo("Open Graph", graphName.c_str())) {
        ImGui::Selectable(graphName.c_str(), true);
        ImGui::EndCombo();
    }

    ImGui::PushItemWidth(-1.0f);
    ImGui::InputText("##GraphPath", graphFilePath_, sizeof(graphFilePath_));
    ImGui::PopItemWidth();

    if (ImGui::Button("New Graph")) {
        resetDefaultRenderGraph();
    }
    ImGui::SameLine();
    if (ImGui::Button("Load")) {
        loadRenderGraph();
    }
    ImGui::SameLine();
    if (ImGui::Button("Save")) {
        saveRenderGraph();
    }

    if (ImGui::Button("Validate Graph")) {
        std::string log;
        renderGraph_.validate(log);
        renderGraphStatus_ = log;
    }

    if (ImGui::BeginCombo("Built-in Sample", "Load Sample...")) {
        for (const render::RenderSampleDesc& desc : render::listBuiltInRenderSamples()) {
            if (ImGui::Selectable(desc.name.c_str())) {
                loadBuiltInSample(desc.id.c_str());
            }
            if (!desc.description.empty() && ImGui::IsItemHovered()) {
                ImGui::SetTooltip("%s", desc.description.c_str());
            }
        }
        ImGui::EndCombo();
    }
    ImGui::Separator();
    ImGui::TextUnformatted("Graph Output");
    ImGui::PushItemWidth(-1.0f);
    ImGui::InputText("##GraphOutput", graphOutputBuffer_, sizeof(graphOutputBuffer_));
    ImGui::PopItemWidth();
    if (ImGui::Button("Set Graph Output")) {
        markRenderGraphOutput(graphOutputBuffer_);
    }

    for (const render::RenderGraphOutput& output : renderGraph_.outputs()) {
        const std::string outputName = render::makeRenderGraphFieldName(output.passName, output.fieldName);
        ImGui::BulletText("%s", outputName.c_str());
    }

    ImGui::Separator();
    ImGui::TextUnformatted("Preview Output");
    ImGui::PushItemWidth(-1.0f);
    ImGui::InputText("##PreviewOutput", previewOutputBuffer_, sizeof(previewOutputBuffer_));
    ImGui::PopItemWidth();
    if (ImGui::Button("Set Preview Output")) {
        setActivePreviewOutput(previewOutputBuffer_);
    }
    if (!activePreviewOutput_.empty()) {
        ImGui::TextDisabled("Active: %s", activePreviewOutput_.c_str());
    }

    for (const render::RenderGraphOutput& output : renderGraph_.outputs()) {
        const std::string outputName = render::makeRenderGraphFieldName(output.passName, output.fieldName);
        const bool selected = outputName == activePreviewOutput_;
        if (ImGui::Selectable(outputName.c_str(), selected)) {
            setActivePreviewOutput(outputName);
        }
    }
    ImGui::Separator();
    ImGui::Text("Nodes: %zu", renderGraph_.nodes().size());
    ImGui::Text("Edges: %zu", renderGraph_.edges().size());
    if (!renderGraphStatus_.empty()) {
        ImGui::Separator();
        ImGui::TextWrapped("%s", renderGraphStatus_.c_str());
    }
}

void EditorApplication::drawRenderPassesPanel()
{
    const float cardWidth = 148.0f * mainScale_;
    const float cardHeight = 86.0f * mainScale_;
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    const int columnCount = std::max(
        1,
        static_cast<int>((ImGui::GetContentRegionAvail().x + spacing) / (cardWidth + spacing)));

    if (ImGui::BeginTable("RenderPassPalette", columnCount, ImGuiTableFlags_SizingStretchSame)) {
        int index = 0;
        for (const render::RenderGraphPassInfo& passInfo : render::listRenderGraphPassTypes()) {
            ImGui::TableNextColumn();
            ImGui::PushID(index++);

            const ImVec2 cardMin = ImGui::GetCursorScreenPos();
            const bool clicked = ImGui::InvisibleButton("RenderPassCard", ImVec2(cardWidth, cardHeight));
            const ImVec2 cardMax = ImGui::GetItemRectMax();
            const bool hovered = ImGui::IsItemHovered();

            ImDrawList* drawList = ImGui::GetWindowDrawList();
            const ImU32 accent = colorForPassType(passInfo.type);
            const ImU32 border = hovered ? IM_COL32(235, 235, 235, 255) : accent;
            drawList->PushClipRect(cardMin, cardMax, true);
            drawList->AddRectFilled(cardMin, cardMax, IM_COL32(25, 28, 34, 255), 3.0f * mainScale_);
            drawList->AddRect(cardMin, cardMax, border, 3.0f * mainScale_, 0, 1.5f * mainScale_);

            const ImVec2 iconMin(cardMin.x + 10.0f * mainScale_, cardMin.y + 8.0f * mainScale_);
            const ImVec2 iconMax(cardMax.x - 10.0f * mainScale_, cardMin.y + 42.0f * mainScale_);
            drawList->AddRectFilled(iconMin, iconMax, accent, 2.0f * mainScale_);
            drawList->AddRect(iconMin, iconMax, IM_COL32(8, 10, 14, 255), 2.0f * mainScale_);
            const char letter[2] = {passInfo.type.empty() ? '?' : passInfo.type.front(), '\0'};
            drawList->AddText(
                ImVec2(iconMin.x + 8.0f * mainScale_, iconMin.y + 5.0f * mainScale_),
                IM_COL32(230, 255, 200, 255),
                letter);
            drawList->AddText(
                ImVec2(cardMin.x + 2.0f * mainScale_, cardMin.y + 49.0f * mainScale_),
                IM_COL32(235, 238, 242, 255),
                passInfo.type.c_str());
            drawList->AddText(
                ImVec2(cardMin.x + 2.0f * mainScale_, cardMin.y + 66.0f * mainScale_),
                IM_COL32(155, 164, 178, 255),
                render::renderGraphPassKindName(passInfo.kind));
            drawList->PopClipRect();

            if (clicked) {
                addRenderGraphNode(passInfo.type, ImVec2(-1.0f, -1.0f));
            }

            if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
                ImGui::SetDragDropPayload(
                    kRenderPassDragPayload,
                    passInfo.type.c_str(),
                    passInfo.type.size() + 1);
                ImGui::TextUnformatted(passInfo.type.c_str());
                ImGui::EndDragDropSource();
            }

            if (hovered) {
                if (passInfo.description.empty()) {
                    ImGui::SetTooltip("%s", render::renderGraphPassKindName(passInfo.kind));
                } else {
                    ImGui::SetTooltip(
                        "%s\n%s",
                        render::renderGraphPassKindName(passInfo.kind),
                        passInfo.description.c_str());
                }
            }

            ImGui::PopID();
        }
        ImGui::EndTable();
    }
}

void EditorApplication::drawRenderGraphRenderUiPanel()
{
    const render::RenderGraphEdge* edge = selectedGraphLinkId_ >= 0
        ? renderGraph_.findEdge(static_cast<uint32_t>(selectedGraphLinkId_))
        : nullptr;
    if (edge != nullptr) {
        ImGui::TextUnformatted("Selected Edge");
        ImGui::Separator();
        ImGui::Text(
            "%s -> %s",
            render::makeRenderGraphFieldName(edge->srcPass, edge->srcField).c_str(),
            render::makeRenderGraphFieldName(edge->dstPass, edge->dstField).c_str());
        if (ImGui::Button("Delete Edge")) {
            renderGraph_.removeEdge(edge->id);
            selectedGraphLinkId_ = -1;
            viewportPreviewValid_ = false;
        }
        return;
    }

    ImGui::TextUnformatted("Render Graph Selection");
    ImGui::Separator();

    render::RenderGraphNode* node = selectedGraphNodeId_ >= 0
        ? renderGraph_.findNode(static_cast<uint32_t>(selectedGraphNodeId_))
        : nullptr;
    if (node == nullptr) {
        ImGui::Text("Graph: %s", renderGraph_.name().c_str());
        ImGui::Text("Graph Output: %s", renderGraph_.firstOutputName().c_str());
        ImGui::Text("Preview Output: %s", activePreviewOutput_.c_str());
        ImGui::Text("Nodes: %zu", renderGraph_.nodes().size());
        ImGui::Text("Edges: %zu", renderGraph_.edges().size());
        return;
    }

    static int editingNodeId = -1;
    if (editingNodeId != static_cast<int>(node->id)) {
        copyToBuffer(node->name, graphNodeNameBuffer_, sizeof(graphNodeNameBuffer_));
        editingNodeId = static_cast<int>(node->id);
    }

    std::unique_ptr<render::RenderGraphPass> pass = render::createRenderGraphPass(node->type);
    if (pass != nullptr) {
        ImGui::Text(
            "Type: %s (%s)",
            node->type.c_str(),
            render::renderGraphPassKindName(pass->kind()));
    } else {
        ImGui::Text("Type: %s", node->type.c_str());
    }
    ImGui::InputText("Name", graphNodeNameBuffer_, sizeof(graphNodeNameBuffer_));
    if (ImGui::IsItemDeactivatedAfterEdit() && std::strlen(graphNodeNameBuffer_) > 0) {
        if (!renderGraph_.renameNode(node->id, graphNodeNameBuffer_)) {
            renderGraphStatus_ = "Node rename failed";
        } else {
            copyToBuffer(renderGraph_.firstOutputName(), graphOutputBuffer_, sizeof(graphOutputBuffer_));
            viewportPreviewValid_ = false;
        }
    }

    const bool hasStaticScenePath =
        node->type == "GPUDrivenPreviewPass" ||
        node->type == "GPUDrivenStreamAssetPass" ||
        node->type == "SceneRayQueryVisualizationPass" ||
        node->type == "SceneMaterialVisualizationPass" ||
        node->type == "ScenePathTracePass";
    if (hasStaticScenePath) {
        static int editingScenePathNodeId = -1;
        static char scenePathBuffer[260] = {};
        render::RenderGraphProperties properties = node->properties.is_object()
            ? node->properties
            : render::RenderGraphProperties::object();
        render::RenderGraphProperties defaults = defaultPropertiesForPass(node->type);
        if (!properties.contains("path") || !properties["path"].is_string()) {
            properties["path"] = defaults.contains("path") && defaults["path"].is_string()
                ? defaults["path"]
                : "";
        }
        if (editingScenePathNodeId != static_cast<int>(node->id)) {
            copyToBuffer(properties["path"].get<std::string>(), scenePathBuffer, sizeof(scenePathBuffer));
            editingScenePathNodeId = static_cast<int>(node->id);
        }

        ImGui::InputText("Scene Path", scenePathBuffer, sizeof(scenePathBuffer));
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            properties["path"] = scenePathBuffer;
            renderGraph_.setNodeProperties(node->id, std::move(properties));
            historyResources_.invalidateAll();
            viewportPreviewValid_ = false;
            renderGraphStatus_ = "Updated static scene path";
        }
    }

    bool drewRuntimeSettings = false;
    if (pass != nullptr) {
        const std::vector<render::RenderGraphRuntimeSetting> settings = pass->runtimeSettings();
        if (!settings.empty()) {
            ImGui::Separator();
            ImGui::TextUnformatted("Runtime Settings");
            drewRuntimeSettings = drawRuntimeSettingsForNode(*node, false, false);
        }
    }

    if (!drewRuntimeSettings && !node->properties.empty()) {
        const std::string propertiesText = node->properties.dump(2);
        ImGui::TextWrapped("%s", propertiesText.c_str());
    }
    if (pass != nullptr) {
        pass->setProperties(node->properties);
        const render::RenderPassReflection reflection = pass->reflect(render::RenderGraphCompileContext{});
        ImGui::Separator();
        ImGui::TextUnformatted("Fields");
        for (const render::RenderGraphField& field : reflection.fields()) {
            const std::string fullName = render::makeRenderGraphFieldName(node->name, field.name);
            ImGui::PushID(fullName.c_str());
            if (field.visibility == render::RenderGraphFieldVisibility::Output) {
                bool output = isMarkedRenderGraphOutput(renderGraph_, fullName);
                if (ImGui::Checkbox("Graph Output", &output)) {
                    if (output) {
                        markRenderGraphOutput(fullName);
                    } else {
                        renderGraph_.clearOutputs();
                        copyToBuffer("", graphOutputBuffer_, sizeof(graphOutputBuffer_));
                        viewportPreviewValid_ = false;
                        renderGraphStatus_ = "Graph output cleared";
                    }
                }
                ImGui::SameLine();
            }
            ImGui::Text(
                "%s %s %s",
                renderGraphFieldVisibilityName(field.visibility),
                field.name.c_str(),
                renderGraphFieldTag(field).c_str());
            setRenderGraphFieldTooltip(field);
            ImGui::PopID();
        }
    }

    ImGui::Separator();
    if (ImGui::Button("Delete Node")) {
        renderGraph_.removeNode(node->id);
        selectedGraphNodeId_ = -1;
        copyToBuffer(renderGraph_.firstOutputName(), graphOutputBuffer_, sizeof(graphOutputBuffer_));
        viewportPreviewValid_ = false;
    }
}

} // namespace metallic
