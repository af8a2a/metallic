#include "Editor/EditorApplication.h"

#include "Runtime/Render/GAPI/Rhi.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Task/TaskSystem.h"
#include "imnodes.h"
#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_vulkan.h"
#include "imgui_internal.h"
#include "ImGuizmo.h"

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
#include <cstdlib>
#include <filesystem>
#include <iterator>
#include <limits>
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
constexpr float kMinViewportCameraSpeed = 0.01f;
constexpr float kMaxViewportCameraSpeed = 100.0f;
constexpr float kViewportCameraWheelSpeedStep = 1.25f;
constexpr float kMaxDollyDisplacement = 0.99f;
constexpr const char* kDefaultRenderSampleId = "pathtracing-sample";

bool environmentFlagEnabled(const char* name)
{
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' &&
        std::string_view(value) != "0" &&
        std::string_view(value) != "false" &&
        std::string_view(value) != "FALSE";
}

constexpr const char* kDefaultImGuiIni = R"ini([Window][Viewport]
Pos=0,28
Size=1588,794
Collapsed=0
DockId=0x00000003,0

[Window][Scene]
Pos=1824,28
Size=576,1322
Collapsed=0
DockId=0x00000002,0

[Window][Assets]
Pos=0,825
Size=1588,525
Collapsed=0
DockId=0x00000004,0

[Window][Console]
Pos=0,825
Size=1588,525
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
Pos=206,372
Size=1828,1070
Collapsed=0

[Window][Profiler]
Pos=0,825
Size=1588,525
Collapsed=0
DockId=0x00000004,2

[Window][NVML Monitor]
Pos=0,825
Size=1588,525
Collapsed=0
DockId=0x00000004,3

[Window][Scene Browser]
Pos=1591,28
Size=809,1073
Collapsed=0
DockId=0x00000007,1

[Window][Inspector]
Pos=1591,1104
Size=809,246
Collapsed=0
DockId=0x00000008,0

[Window][Statistics]
Pos=1591,28
Size=809,1073
Collapsed=0
DockId=0x00000007,0

[Window][Unsaved Scene Changes]
Pos=942,3
Size=498,172
Collapsed=0

[Table][0x2AFDBD75,2]
RefScale=20
Column 0  Weight=1.0000
Column 1  Width=168

[Table][0x331D395F,6]
RefScale=20
Column 0  Weight=1.0000
Column 1  Width=209
Column 2  Width=139
Column 3  Width=72
Column 4  Width=72
Column 5  Width=72

[Docking][Data]
DockSpace       ID=0xB0446515 Window=0x3660BDC2 Pos=0,28 Size=2400,1322 Split=X
  DockNode      ID=0x00000006 Parent=0xB0446515 SizeRef=1588,1322 Split=X
    DockNode    ID=0x00000001 Parent=0x00000006 SizeRef=1821,1350 Split=Y
      DockNode  ID=0x00000003 Parent=0x00000001 SizeRef=1821,794 CentralNode=1 Selected=0xC450F867
      DockNode  ID=0x00000004 Parent=0x00000001 SizeRef=1821,525 Selected=0x9B5D3198
    DockNode    ID=0x00000002 Parent=0x00000006 SizeRef=576,1350 Selected=0xE601B12F
  DockNode      ID=0x00000005 Parent=0xB0446515 SizeRef=809,1322 Split=Y Selected=0x2732FE10
    DockNode    ID=0x00000007 Parent=0x00000005 SizeRef=892,1073 Selected=0x2732FE10
    DockNode    ID=0x00000008 Parent=0x00000005 SizeRef=892,246 Selected=0x36DC96AB

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
    std::string filename = path.filename().string();
    std::transform(filename.begin(), filename.end(), filename.begin(), [](unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    return extension == ".gltf" || extension == ".glb" ||
        filename.ends_with(".metallic_scene.json");
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
        type == "SceneMaterialShaderObjectPass" ||
        type == "SceneMaterialVisualizationPass" ||
        type == "ScenePathTracePass" ||
        type == "SceneRtxdiPass";
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
        L"Metallic Scene Files (*.gltf;*.glb;*.metallic_scene.json)\0*.gltf;*.glb;*.metallic_scene.json\0"
        L"Metallic Scene Document (*.metallic_scene.json)\0*.metallic_scene.json\0"
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
            {"pageLoadConcurrency", 2},
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
            {"relaxHistoryLength", 30},
            {"relaxFastHistoryLength", 6},
            {"relaxAtrousIterations", 5},
            {"relaxDiffusePrepassRadius", 30.0f},
            {"relaxSpecularPrepassRadius", 50.0f},
            {"relaxMinHitDistanceWeight", 0.1f},
            {"relaxAntiFirefly", true},
            {"resetSerial", 0},
            {"camera", {
                {"projection", "perspective"},
                {"fovDegrees", 50.0f},
                {"znear", 0.001f},
                {"zfar", 10000.0f},
                {"reversedZ", true},
                {"eye", {0.0f, 0.25f, 3.0f}},
                {"center", {0.0f, 0.15f, 0.0f}},
                {"up", {0.0f, 1.0f, 0.0f}},
            }},
        };
    }
    if (type == "RtxdiCompositePass") {
        return render::RenderGraphProperties{{"exposure", 1.0f}};
    }
    return render::RenderGraphProperties::object();
}

render::RenderGraphNode* findSceneCameraNode(
    render::RenderGraph& graph,
    const render::RenderGraphNode* outputNode)
{
    std::vector<std::string> pendingNodes;
    if (outputNode != nullptr) {
        pendingNodes.push_back(outputNode->name);
    }
    for (size_t pendingIndex = 0; pendingIndex < pendingNodes.size(); ++pendingIndex) {
        const std::string nodeName = pendingNodes[pendingIndex];
        render::RenderGraphNode* node = graph.findNode(nodeName);
        if (node != nullptr && isSceneAwareRenderPassType(node->type)) {
            return node;
        }
        for (const render::RenderGraphEdge& edge : graph.edges()) {
            if (edge.dstPass != nodeName ||
                std::find(pendingNodes.begin(), pendingNodes.end(), edge.srcPass) != pendingNodes.end()) {
                continue;
            }
            pendingNodes.push_back(edge.srcPass);
        }
    }

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

struct ViewportCameraMatrices {
    CameraFrame frame;
    float4x4 view = float4x4::Identity();
    float4x4 projection = float4x4::Identity();
    float nearPlane = 0.1f;
    float farPlane = 10000.0f;
    bool orthographic = false;
};

ViewportCameraMatrices viewportCameraMatrices(
    const render::RenderGraphProperties& camera,
    float viewportWidth,
    float viewportHeight)
{
    constexpr float kPi = 3.14159265358979323846f;
    ViewportCameraMatrices result;
    result.frame = cameraFrameFrom(camera);
    result.view.a00 = result.frame.right.x;
    result.view.a01 = result.frame.right.y;
    result.view.a02 = result.frame.right.z;
    result.view.a03 = -dot(result.frame.right, result.frame.eye);
    result.view.a10 = result.frame.viewUp.x;
    result.view.a11 = result.frame.viewUp.y;
    result.view.a12 = result.frame.viewUp.z;
    result.view.a13 = -dot(result.frame.viewUp, result.frame.eye);
    result.view.a20 = -result.frame.forward.x;
    result.view.a21 = -result.frame.forward.y;
    result.view.a22 = -result.frame.forward.z;
    result.view.a23 = dot(result.frame.forward, result.frame.eye);

    const float znear = std::max(propertyFloatOr(camera, "znear", 0.1f), 0.00001f);
    const float zfar = std::max(propertyFloatOr(camera, "zfar", 10000.0f), znear + 0.00001f);
    result.nearPlane = znear;
    result.farPlane = zfar;
    const bool reversedZ = camera.value("reversedZ", true);
    const uint32_t projectionFlags = reversedZ ? PROJ_REVERSED_Z : 0;
    const float aspect = std::max(viewportWidth / std::max(viewportHeight, 1.0f), 0.001f);
    result.orthographic = camera.value("projection", std::string("perspective")) == "orthographic";
    if (result.orthographic) {
        const float height = cameraOrthoHeight(camera, result.frame.distance);
        const float width = height * aspect;
        result.projection.SetupByOrthoProjection(
            -width * 0.5f,
            width * 0.5f,
            -height * 0.5f,
            height * 0.5f,
            znear,
            zfar,
            projectionFlags);
    } else {
        const float fov = std::clamp(propertyFloatOr(camera, "fovDegrees", 60.0f), 1.0f, 179.0f);
        result.projection.SetupByHalfFovy(fov * (kPi / 360.0f), aspect, znear, zfar, projectionFlags);
    }
    return result;
}

float affineDeterminant(const float4x4& matrix)
{
    const float3 column0(matrix.a00, matrix.a10, matrix.a20);
    const float3 column1(matrix.a01, matrix.a11, matrix.a21);
    const float3 column2(matrix.a02, matrix.a12, matrix.a22);
    return dot(column0, cross(column1, column2));
}

bool matrixIsFinite(const float4x4& matrix)
{
    for (size_t index = 0; index < 16; ++index) {
        if (!std::isfinite(matrix.a[index])) {
            return false;
        }
    }
    return true;
}

bool matrixHasShear(const float4x4& matrix)
{
    const float3 column0(matrix.a00, matrix.a10, matrix.a20);
    const float3 column1(matrix.a01, matrix.a11, matrix.a21);
    const float3 column2(matrix.a02, matrix.a12, matrix.a22);
    const float length0 = length(column0);
    const float length1 = length(column1);
    const float length2 = length(column2);
    if (std::min({length0, length1, length2}) <= 0.000001f) {
        return false;
    }
    constexpr float kShearEpsilon = 0.0005f;
    return std::abs(dot(column0 / length0, column1 / length1)) > kShearEpsilon ||
        std::abs(dot(column0 / length0, column2 / length2)) > kShearEpsilon ||
        std::abs(dot(column1 / length1, column2 / length2)) > kShearEpsilon;
}

bool matrixHasNonUniformScale(const float4x4& matrix)
{
    const float length0 = length(float3(matrix.a00, matrix.a10, matrix.a20));
    const float length1 = length(float3(matrix.a01, matrix.a11, matrix.a21));
    const float length2 = length(float3(matrix.a02, matrix.a12, matrix.a22));
    const float maximum = std::max({length0, length1, length2});
    const float minimum = std::min({length0, length1, length2});
    return !std::isfinite(maximum) || maximum <= 0.000001f ||
        maximum - minimum > maximum * 0.0005f;
}

std::pair<uint32_t, uint32_t> constrainedPreviewExtent(
    float width,
    float height,
    uint32_t maximumDimension)
{
    width = std::max(width, 1.0f);
    height = std::max(height, 1.0f);
    const float scale = std::min({
        1.0f,
        static_cast<float>(maximumDimension) / width,
        static_cast<float>(maximumDimension) / height,
    });
    return {
        std::max(static_cast<uint32_t>(std::floor(width * scale)), 1u),
        std::max(static_cast<uint32_t>(std::floor(height * scale)), 1u),
    };
}

bool projectWorldToViewport(
    const float3& worldPosition,
    const ViewportCameraMatrices& matrices,
    const ImVec2& min,
    const ImVec2& max,
    ImVec2& screenPosition)
{
    const float4 viewPosition = matrices.view * float4(worldPosition, 1.0f);
    const float viewDepth = -viewPosition.z;
    if (!std::isfinite(viewDepth) || viewDepth < matrices.nearPlane ||
        viewDepth > matrices.farPlane) {
        return false;
    }
    const float4 clipPosition = matrices.projection * viewPosition;
    if (!std::isfinite(clipPosition.w) || std::abs(clipPosition.w) <= 0.000001f) {
        return false;
    }
    const float normalizedX = clipPosition.x / clipPosition.w;
    const float normalizedY = clipPosition.y / clipPosition.w;
    if (!std::isfinite(normalizedX) || !std::isfinite(normalizedY) ||
        normalizedX < -1.0f || normalizedX > 1.0f ||
        normalizedY < -1.0f || normalizedY > 1.0f) {
        return false;
    }
    screenPosition = ImVec2(
        min.x + (normalizedX * 0.5f + 0.5f) * (max.x - min.x),
        min.y + (1.0f - (normalizedY * 0.5f + 0.5f)) * (max.y - min.y));
    return true;
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

    const float pitch = -displacement.y * 2.0f * kPi;
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
    const char* startupStreamAssetPath,
    bool enableNsightGraphicsCapture)
{
    const auto taskInitialization = task::initializeTaskSystem();
    if (!taskInitialization) {
        spdlog::error("TaskSystem initialization failed: {}", taskInitialization.error().message);
        return 1;
    }
    struct TaskSystemShutdownGuard {
        ~TaskSystemShutdownGuard()
        {
            task::shutdownTaskSystem();
        }
    } taskSystemShutdownGuard;

    smokeTest_ = smokeTest;
    waitForGraphicsDebugger_ = waitForGraphicsDebugger && !smokeTest;
    nsightGraphicsCaptureRequested_ =
        enableNsightGraphicsCapture ||
        environmentFlagEnabled("METALLIC_NSIGHT_GRAPHICS_CAPTURE");
    startupSampleId_ = startupSampleId != nullptr ? startupSampleId : "";
    if (smokeTest_ && startupSampleId_.empty()) {
        // Allow headless verification of a specific built-in sample, e.g.
        // METALLIC_SMOKE_TEST_SAMPLE=pathtracing-sharc-meet-mat
        const char* smokeSample = std::getenv("METALLIC_SMOKE_TEST_SAMPLE");
        startupSampleId_ =
            smokeSample != nullptr && *smokeSample != '\0'
                ? std::string(smokeSample)
                : std::string("material-visualization-abeautiful-game");
    }
    startupScenePath_ = startupScenePath != nullptr ? startupScenePath : "";
    startupStreamAssetPath_ = startupStreamAssetPath != nullptr ? startupStreamAssetPath : "";
    spdlog::info(
        "[Startup] Run requested smokeTest={} waitForGraphicsDebugger={} nsightCapture={} startupSample='{}' "
        "sceneOverride='{}' streamAssetOverride='{}'",
        smokeTest_,
        waitForGraphicsDebugger_,
        nsightGraphicsCaptureRequested_,
        startupSampleId_,
        startupScenePath_,
        startupStreamAssetPath_);

    initializeNsightGraphicsCapture();

    if (!initialize()) {
        shutdown();
        return 1;
    }

    if (smokeTest) {
        if (!waitForPendingSceneLoad(30000)) {
            spdlog::error("Smoke test scene load did not complete: {}", sceneStatus_);
            shutdown();
            return 1;
        }
        if (render::RenderGraphNode* previewNode = activePreviewRenderGraphNode();
            previewNode != nullptr && previewNode->type == "ScenePathTracePass") {
            previewNode->properties["samples"] = 1;
            previewNode->properties["maxDepth"] = 1;
            previewNode->properties["accumulate"] = false;
            previewNode->properties["bsdf"] = "standard";
            (void)renderGraph_.setNodeRuntimeProperty(previewNode->id, "samples", 1);
            (void)renderGraph_.setNodeRuntimeProperty(previewNode->id, "maxDepth", 1);
            (void)renderGraph_.setNodeRuntimeProperty(previewNode->id, "accumulate", false);
        }
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

void EditorApplication::initializeNsightGraphicsCapture()
{
    if (!nsightGraphicsCaptureRequested_) {
        return;
    }
    if (!render::profiling::NsightGraphicsCapture::compiledAvailable()) {
        spdlog::warn(
            "Nsight Graphics Capture was requested, but the SDK was not available when Metallic was built.");
        return;
    }

    render::profiling::NsightGraphicsCaptureConfig config;
    config.installationRoot =
        render::profiling::NsightGraphicsCapture::defaultInstallationRoot();
    config.outputDirectory =
        std::filesystem::path(PROJECT_SOURCE_DIR) / "Captures" / "NsightGraphics";
    config.showHud = true;

    std::string error;
    if (!nsightGraphicsCapture_.initializeBeforeGraphics(config, error)) {
        spdlog::warn("Nsight Graphics Capture initialization failed: {}", error);
        return;
    }

    spdlog::info(
        "Nsight Graphics Capture initialized; captures will be written under '{}'",
        config.outputDirectory.string());
}

void EditorApplication::requestNsightGraphicsCapture()
{
    if (!nsightGraphicsCaptureRequested_ || !viewportPreviewValid_) {
        return;
    }

    std::string error;
    if (!nsightGraphicsCapture_.requestCapture(
            render::profiling::NsightGraphicsCaptureRequest{
                .framesBeforeStart = 0,
                .framesToCapture = 1,
            },
            error)) {
        spdlog::warn("Nsight Graphics Capture request failed: {}", error);
        return;
    }

    nsightGraphicsCaptureFramePhase_ =
        NsightGraphicsCaptureFramePhase::WaitingForStartPresent;
    spdlog::info("Nsight Graphics Capture queued for the next full View frame.");
}

void EditorApplication::pollNsightGraphicsCapture()
{
    if (!nsightGraphicsCapture_.hasOutstandingCapture()) {
        return;
    }

    const render::profiling::NsightGraphicsCapturePollResult result =
        nsightGraphicsCapture_.poll();
    if (result.state == render::profiling::NsightGraphicsCaptureState::CaptureCompleted) {
        nsightGraphicsCaptureFramePhase_ = NsightGraphicsCaptureFramePhase::Idle;
        spdlog::info("Nsight Graphics Capture completed: {}", result.capturePath.string());
    } else if (result.state == render::profiling::NsightGraphicsCaptureState::Error) {
        nsightGraphicsCaptureFramePhase_ = NsightGraphicsCaptureFramePhase::Idle;
        spdlog::warn("Nsight Graphics Capture failed: {}", result.message);
    }
}

void EditorApplication::advanceNsightGraphicsCaptureAfterPresent()
{
    if (nsightGraphicsCaptureFramePhase_ ==
        NsightGraphicsCaptureFramePhase::WaitingForStartPresent) {
        nsightGraphicsCaptureFramePhase_ =
            NsightGraphicsCaptureFramePhase::CapturingNextFrame;
        viewportPreviewNeedsRender_ = true;
    } else if (nsightGraphicsCaptureFramePhase_ ==
               NsightGraphicsCaptureFramePhase::CapturingNextFrame) {
        nsightGraphicsCaptureFramePhase_ = NsightGraphicsCaptureFramePhase::Idle;
    }
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
        renderWorld_.setScene(&scene_);
        graphExecutor_ = std::make_unique<render::RenderGraphExecutor>(subsystemHost_, renderWorld_);
        graphExecutor_->bindRuntimeScene(&scene_);
        sceneAccelerationStructure_ =
            std::make_unique<render::SceneAccelerationStructureBuilder>();
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

    bool enableStreamline = !smokeTest_;
    if (enableStreamline && !startupSampleId_.empty()) {
        bool sampleRequiresStreamline = false;
        if (render::queryBuiltInRenderSampleStreamlineRequirement(
                startupSampleId_,
                sampleRequiresStreamline)) {
            enableStreamline = sampleRequiresStreamline;
        }
    }
    spdlog::info(
        "[Startup] Streamline device integration {} for startup sample '{}'",
        enableStreamline ? "enabled" : "disabled",
        startupSampleId_.empty() ? "<generic-editor>" : startupSampleId_);

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
                .enableStreamline = enableStreamline,
                .enableAftermath = !smokeTest_,
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
    cancelSceneLoad();
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
    subsystemHost_.shutdown();
    sceneAccelerationStructure_.reset();

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
            requestPendingSceneAction(PendingSceneAction::Exit);
        }

        if (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED &&
            event.window.windowID == SDL_GetWindowID(window_)) {
            requestPendingSceneAction(PendingSceneAction::Exit);
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

    {
        auto profileScope = profiler_.scope("Poll Nsight Capture");
        pollNsightGraphicsCapture();
    }

    if (frameFence_ != nullptr) {
        auto frameFenceScope = profiler_.scope("Wait Frame Fence");
        render::Result result;
        {
            auto profileScope = profiler_.scope("Wait Fence Signal");
            result = frameFence_->wait();
        }
        if (!result) {
            spdlog::error("frameFence wait before UI failed with Result {}", render::resultToString(result));
            running_ = false;
            return false;
        }

        if (graphExecutor_ != nullptr) {
            std::vector<render::RenderGraphExecutionStats> completedGpuStats;
            {
                auto profileScope = profiler_.scope("Resolve GPU Queries");
                result = graphExecutor_->collectCompletedGpuExecutionStats(completedGpuStats);
            }
            if (!result) {
                spdlog::warn(
                    "RenderGraph GPU timestamp query read failed with Result {}",
                    render::resultToString(result));
            } else {
                auto profileScope = profiler_.scope("Update GPU Profiler");
                for (const render::RenderGraphExecutionStats& stats : completedGpuStats) {
                    profiler_.updateRenderGraphGpuStats(stats);
                }
            }
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
        ImGuizmo::BeginFrame();
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
        const bool transformEditing = gizmoWasUsing_ || inspectorTransformEditing_ ||
            inspectorPropertyEditing_ ||
            ImGuizmo::IsUsingAny();
        if (!transformEditing) {
            if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_S)) {
                saveScene();
            } else if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_Z)) {
                undoTransform();
            } else if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_Y)) {
                redoTransform();
            } else if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_O)) {
                chooseEnvironmentFile();
            } else if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_O)) {
                chooseSceneFile();
            }
        }
    }

    pollSceneLoad();

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
            if (ImGui::MenuItem("Save Scene", "Ctrl+S", false, scene_.valid())) {
                saveScene();
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
                        loadDroppedScene(recentPath);
                    }
                }
                ImGui::EndMenu();
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Exit")) {
                requestPendingSceneAction(PendingSceneAction::Exit);
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

        if (ImGui::BeginMenu("Edit")) {
            if (ImGui::MenuItem("Undo Scene Edit", "Ctrl+Z", false, transformCommandCursor_ > 0)) {
                undoTransform();
            }
            if (ImGui::MenuItem(
                    "Redo Scene Edit",
                    "Ctrl+Y",
                    false,
                    transformCommandCursor_ < transformCommands_.size())) {
                redoTransform();
            }
            ImGui::EndMenu();
        }

        if (ImGui::Button("Open Render Graph Editor")) {
            renderGraphEditorOpen_ = true;
        }

        ImGui::EndMenuBar();
    }

    drawUnsavedSceneModal();

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

    const auto captureState = nsightGraphicsCapture_.state();
    const bool captureReady =
        captureState == render::profiling::NsightGraphicsCaptureState::Ready ||
        captureState == render::profiling::NsightGraphicsCaptureState::CaptureCompleted;
    std::string captureStatus = nsightGraphicsCapture_.statusText();
    if (!render::profiling::NsightGraphicsCapture::compiledAvailable()) {
        captureStatus = "Nsight Graphics SDK is not available in this build.";
    } else if (!nsightGraphicsCaptureRequested_) {
        captureStatus = "Restart with --nsight-capture to enable Graphics Capture.";
    } else if (captureReady && !viewportPreviewValid_) {
        captureStatus = "The current View is not ready for capture.";
    }
    const std::string capturePath = nsightGraphicsCapture_.capturePath().string();
    const EditorProfiler::GraphicsCaptureControls captureControls{
        .sdkCompiled = render::profiling::NsightGraphicsCapture::compiledAvailable(),
        .runtimeEnabled = nsightGraphicsCaptureRequested_,
        .canCapture = captureReady && viewportPreviewValid_ &&
            !nsightGraphicsCapture_.hasOutstandingCapture(),
        .capturePending = nsightGraphicsCapture_.hasOutstandingCapture(),
        .statusText = captureStatus.c_str(),
        .capturePath = capturePath.c_str(),
    };
    if (profiler_.drawWindow(&profilerOpen_, captureControls)) {
        requestNsightGraphicsCapture();
    }
    nvmlMonitor_.drawWindow(&nvmlMonitorOpen_);
}

void EditorApplication::drawScenePanel()
{
    ImGui::Begin("Scene Browser");

    ImGui::TextUnformatted(scene_.sources().size() > 1u ? "Composite Scene" : "glTF Scene");
    if (scene_.valid()) {
        ImGui::TextWrapped("Path: %s", scene_.sourcePath().string().c_str());
        ImGui::Text("Document: %s%s", scene_.documentPath().string().c_str(), scene_.dirty() ? " *" : "");
    } else {
        ImGui::TextDisabled("No scene loaded.");
    }

    if (ImGui::Button("Clear")) {
        requestPendingSceneAction(PendingSceneAction::Clear);
    }
    ImGui::SameLine();
    if (ImGui::Button("Build RTAS")) {
        buildSceneAccelerationStructure();
    }
    ImGui::SameLine();
    if (ImGui::Button("Clear RTAS")) {
        clearSceneAccelerationStructure();
    }

    if (pendingSceneLoad_.valid() || pendingSceneResourcePreparation_) {
        const scene::SceneLoadProgress progress = pendingSceneResourcePreparation_
            ? pendingSceneResourceProgress_
            : pendingSceneLoad_.progress();
        const double elapsedSeconds = std::chrono::duration<double>(progress.elapsed).count();
        ImGui::Separator();
        ImGui::Text(
            "Loading: %s (%.1fs)",
            scene::sceneLoadPhaseName(progress.phase),
            elapsedSeconds);
        ImGui::ProgressBar(progress.fraction, ImVec2(-1.0f, 0.0f));
        if (!progress.currentItem.empty()) {
            ImGui::TextWrapped("%s", progress.currentItem.c_str());
        }
        if (progress.status == scene::SceneLoadStatus::Running && ImGui::Button("Cancel Load")) {
            cancelSceneLoad();
            sceneStatus_ = "Scene load cancellation requested.";
        }
    }

    if (!sceneStatus_.empty()) {
        ImGui::TextWrapped("%s", sceneStatus_.c_str());
    }
    if (!sceneAccelerationStructureStatus_.empty()) {
        ImGui::TextWrapped("%s", sceneAccelerationStructureStatus_.c_str());
    }
    if (sceneAccelerationStructure_ != nullptr && sceneAccelerationStructure_->valid()) {
        const render::SceneAccelerationStructureStats& accelerationStructureStats =
            sceneAccelerationStructure_->stats();
        ImGui::Text(
            "RTAS: %u BLAS, %u instances, %llu triangles",
            accelerationStructureStats.blasCount,
            accelerationStructureStats.instanceCount,
            static_cast<unsigned long long>(accelerationStructureStats.triangleCount));
        ImGui::Text(
            "RTAS memory: geometry %llu bytes, AS %llu bytes, scratch %llu bytes",
            static_cast<unsigned long long>(accelerationStructureStats.geometryBytes),
            static_cast<unsigned long long>(accelerationStructureStats.accelerationStructureBytes),
            static_cast<unsigned long long>(accelerationStructureStats.scratchBytes));
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
        ImGui::TextDisabled(
            "Load a .gltf, .glb, or .metallic_scene.json file to inspect its scene graph.");
        ImGui::End();
        return;
    }

    ImGui::Separator();
    if (scene_.sources().size() > 1u &&
        ImGui::CollapsingHeader(
            ("Sources (" + std::to_string(scene_.sources().size()) + ")").c_str(),
            ImGuiTreeNodeFlags_DefaultOpen)) {
        for (size_t sourceIndex = 0; sourceIndex < scene_.sources().size(); ++sourceIndex) {
            const scene::SceneSourceDesc& source = scene_.sources()[sourceIndex];
            ImGui::PushID(static_cast<int>(sourceIndex));
            bool enabled = source.enabled;
            if (ImGui::Checkbox("##SourceEnabled", &enabled) &&
                scene_.setSourceEnabled(source.id, enabled)) {
                sceneNonTransformDirty_ = true;
                updateSceneDirtyState();
                notifyScenePropertiesChanged();
                sceneStatus_ = enabled
                    ? "Enabled scene source: " + source.id
                    : "Disabled scene source: " + source.id;
            }
            ImGui::SameLine();
            ImGui::TextUnformatted(source.id.c_str());
            ImGui::Indent();
            ImGui::TextWrapped("%s", source.path.string().c_str());
            ImGui::TextDisabled(
                "Mount translation: %.3f, %.3f, %.3f",
                source.mountMatrix.a03,
                source.mountMatrix.a13,
                source.mountMatrix.a23);
            ImGui::Unindent();
            ImGui::PopID();
        }
    }

    if (ImGui::CollapsingHeader("Asset Info")) {
        const scene::SceneAssetInfo& asset = scene_.assetInfo();
        ImGui::Text("Path: %s", scene_.sourcePath().string().c_str());
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
    const std::string sceneLabel = scene_.sources().size() > 1u
        ? scene_.sceneName()
        : "Scene-" + std::to_string(scene_.sceneIndex()) + " " + scene_.sceneName();
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
        SceneSelection selection{
            .type = selectionType,
            .index = index,
        };
        if (selectionType == SceneSelectionType::Node) {
            selection.nodeIndex = index;
            selection.object = scene_.objectForNode(index).entity();
        } else if (selectionType == SceneSelectionType::Camera &&
            index >= 0 && static_cast<size_t>(index) < scene_.cameras().size()) {
            const scene::RenderCamera& camera = scene_.cameras()[static_cast<size_t>(index)];
            selection.nodeIndex = camera.nodeIndex;
            selection.object = camera.object;
        } else if (selectionType == SceneSelectionType::Light &&
            index >= 0 && static_cast<size_t>(index) < scene_.lights().size()) {
            const scene::RenderLight& light = scene_.lights()[static_cast<size_t>(index)];
            selection.nodeIndex = light.nodeIndex;
            selection.object = light.object;
        }
        if (scene_.sceneGraph().object(selection.object)) {
            selection.sceneLifetimeRevision = scene_.sceneGraph().lifetimeRevision();
        }
        sceneSelection_ = selection;
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
            const std::string label = "[" + std::to_string(index) + "] " + camera.name +
                " (" + scene::cameraTypeName(camera.type) + ")";
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
    const scene::ConstSceneObject currentObject = selectedSceneObject();
    if (inspectorPropertyEditing_ &&
        (!inspectorOpen_ || !currentObject ||
            inspectorPropertyEditingObject_ != currentObject.entity() ||
            inspectorPropertyEditingSceneLifetime_ != scene_.sceneGraph().lifetimeRevision())) {
        finishActiveInspectorPropertyTransaction();
    }
    if (inspectorTransformEditing_ &&
        (!inspectorOpen_ || !currentObject ||
            inspectorEditingObject_ != currentObject.entity() ||
            inspectorEditingSceneLifetime_ != scene_.sceneGraph().lifetimeRevision())) {
        const scene::ConstSceneObject editedObject =
            inspectorEditingSceneLifetime_ == scene_.sceneGraph().lifetimeRevision()
            ? scene_.sceneGraph().object(inspectorEditingObject_)
            : scene::ConstSceneObject{};
        if (const scene::TransformComponent* editedTransform =
                editedObject.tryGetComponent<scene::TransformComponent>()) {
            pushTransformCommand(
                inspectorEditingObject_,
                inspectorEditingSceneLifetime_,
                inspectorStartLocalMatrix_,
                editedTransform->localMatrix);
        }
        inspectorTransformEditing_ = false;
        inspectorEditingObject_ = scene::kNullSceneEntity;
        inspectorEditingSceneLifetime_ = 0;
    }
    if (!inspectorOpen_) {
        return;
    }
    if (!ImGui::Begin("Inspector", &inspectorOpen_)) {
        finishActiveInspectorPropertyTransaction();
        if (inspectorTransformEditing_) {
            const scene::ConstSceneObject editedObject =
                inspectorEditingSceneLifetime_ == scene_.sceneGraph().lifetimeRevision()
                ? scene_.sceneGraph().object(inspectorEditingObject_)
                : scene::ConstSceneObject{};
            if (const scene::TransformComponent* editedTransform =
                    editedObject.tryGetComponent<scene::TransformComponent>()) {
                pushTransformCommand(
                    inspectorEditingObject_,
                    inspectorEditingSceneLifetime_,
                    inspectorStartLocalMatrix_,
                    editedTransform->localMatrix);
            }
            inspectorTransformEditing_ = false;
            inspectorEditingObject_ = scene::kNullSceneEntity;
            inspectorEditingSceneLifetime_ = 0;
        }
        ImGui::End();
        return;
    }

    if (ImGui::CollapsingHeader("Viewport Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::PushItemWidth(-1.0f);
        ImGui::DragFloat(
            "Movement Speed",
            &viewportCameraSpeed_,
            0.05f,
            kMinViewportCameraSpeed,
            kMaxViewportCameraSpeed,
            "%.2fx",
            ImGuiSliderFlags_AlwaysClamp);
        ImGui::PopItemWidth();
        ImGui::TextDisabled("Hold RMB and scroll to adjust speed quickly.");
    }
    ImGui::Separator();

    if (!scene_.valid()) {
        ImGui::TextDisabled("No scene loaded");
        ImGui::End();
        return;
    }
    const scene::ConstSceneObject selectedObject = currentObject;
    if (sceneSelection_.type == SceneSelectionType::None ||
        (sceneSelection_.index < 0 && !selectedObject)) {
        ImGui::TextDisabled("No selection");
        ImGui::Separator();
        ImGui::TextWrapped("Select an element in the Scene Browser to view its properties.");
        ImGui::End();
        return;
    }

    if (selectedObject) {
        drawSelectedNodeTransformInspector();
        ImGui::Separator();

        bool drewComponentInspector = false;
        if (selectedObject.hasComponent<scene::CameraComponent>()) {
            drawSelectedCameraComponentInspector();
            drewComponentInspector = true;
        }
        if (selectedObject.hasComponent<scene::LightComponent>()) {
            if (drewComponentInspector) {
                ImGui::Separator();
            }
            drawSelectedLightComponentInspector();
            drewComponentInspector = true;
        }
        if (drewComponentInspector) {
            ImGui::Separator();
        }
    }

    switch (sceneSelection_.type) {
    case SceneSelectionType::Node: {
        if (sceneSelection_.index < 0 ||
            static_cast<size_t>(sceneSelection_.index) >= scene_.nodes().size()) {
            if (const scene::TagComponent* tag =
                    selectedObject.tryGetComponent<scene::TagComponent>()) {
                ImGui::Text("Object: %s", tag->name.c_str());
            } else {
                ImGui::Text("Object: %u", static_cast<uint32_t>(sceneSelection_.object));
            }
            ImGui::TextDisabled("Runtime object (no source node)");
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
            ImGui::TextDisabled("Camera render snapshot is unavailable; properties are read-only.");
            break;
        }
        const scene::RenderCamera& camera = scene_.cameras()[static_cast<size_t>(sceneSelection_.index)];
        ImGui::TextUnformatted("Derived camera pose (read-only)");
        ImGui::Text("Eye: %s", scene::formatVec3(camera.eye).c_str());
        ImGui::Text("Center: %s", scene::formatVec3(camera.center).c_str());
        ImGui::Text("Up: %s", scene::formatVec3(camera.up).c_str());
        if (!selectedObject || !selectedObject.hasComponent<scene::CameraComponent>()) {
            ImGui::TextDisabled("The render snapshot has no editable CameraComponent owner.");
        }
        break;
    }
    case SceneSelectionType::Light: {
        if (static_cast<size_t>(sceneSelection_.index) >= scene_.lights().size()) {
            ImGui::TextDisabled("Light render snapshot is unavailable; properties are read-only.");
            break;
        }
        const scene::RenderLight& light = scene_.lights()[static_cast<size_t>(sceneSelection_.index)];
        ImGui::TextUnformatted("Derived light pose (read-only)");
        ImGui::Text(
            "World position: %.3f, %.3f, %.3f",
            light.worldMatrix.a03,
            light.worldMatrix.a13,
            light.worldMatrix.a23);
        if (!selectedObject || !selectedObject.hasComponent<scene::LightComponent>()) {
            ImGui::TextDisabled("The render snapshot has no editable LightComponent owner.");
        }
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

    if (sceneAccelerationStructure_ != nullptr && sceneAccelerationStructure_->valid()) {
        const render::SceneAccelerationStructureStats& accelerationStructureStats =
            sceneAccelerationStructure_->stats();
        ImGui::Separator();
        ImGui::Text(
            "RTAS: %u BLAS, %u instances, %llu triangles",
            accelerationStructureStats.blasCount,
            accelerationStructureStats.instanceCount,
            static_cast<unsigned long long>(accelerationStructureStats.triangleCount));
        ImGui::Text(
            "RTAS memory: geometry %llu bytes, AS %llu bytes, scratch %llu bytes",
            static_cast<unsigned long long>(accelerationStructureStats.geometryBytes),
            static_cast<unsigned long long>(accelerationStructureStats.accelerationStructureBytes),
            static_cast<unsigned long long>(accelerationStructureStats.scratchBytes));
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

render::RenderGraphNode* EditorApplication::viewportCameraRenderGraphNode()
{
    return findSceneCameraNode(renderGraph_, activePreviewRenderGraphNode());
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
    render::RenderGraphNode* node = viewportCameraRenderGraphNode();
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
    render::EnvironmentSettings environment = renderWorld_.environment();
    bool changed = false;
    ImGui::TextUnformatted("Environment");
    ImGui::PushID("SceneEnvironment");

    if (ImGui::Checkbox("Enabled", &environment.enabled)) {
        changed = true;
    }
    ImGui::SameLine();
    if (ImGui::Checkbox("Visible", &environment.visible)) {
        changed = true;
    }

    const std::string displayPath = environment.path.empty()
        ? std::string("-")
        : displayPathForProperty(environment.path);
    ImGui::TextWrapped("HDRI: %s", displayPath.c_str());

    ImGui::PushItemWidth(-1.0f);
    if (ImGui::SliderFloat("Intensity", &environment.intensity, 0.0f, 16.0f, "%.3f")) {
        environment.intensity = std::max(environment.intensity, 0.0f);
        changed = true;
    }
    if (ImGui::SliderFloat("Rotation", &environment.rotationDegrees, -180.0f, 180.0f, "%.1f deg")) {
        changed = true;
    }
    ImGui::PopItemWidth();
    ImGui::PopID();

    if (!changed) {
        return;
    }
    beginEnvironmentEdit();
    renderWorld_.setEnvironment(environment);
    if (scene_.valid()) {
        if (scene_.setEnvironment(environment)) {
            sceneNonTransformDirty_ = true;
            updateSceneDirtyState();
        }
    }
    environmentUserEdited_ = true;
    environmentFromSample_ = false;
    preserveSampleEnvironmentForNextSceneLoad_ = false;
    viewportPreviewNeedsRender_ = true;
    renderGraphStatus_ = "Updated scene environment";
}

void EditorApplication::beginEnvironmentEdit()
{
    if (environmentEditBaselineValid_) {
        return;
    }
    environmentEditBaseline_ = renderWorld_.environment();
    environmentEditBaselineValid_ = true;
    environmentEditBaselineUserEdited_ = environmentUserEdited_;
    environmentEditBaselineFromSample_ = environmentFromSample_;
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
    render::RenderGraphNode* node = viewportCameraRenderGraphNode();
    if (node == nullptr) {
        return;
    }
    const render::RenderGraphProperties camera = properties.is_object() &&
        properties.contains("camera") &&
        properties["camera"].is_object()
        ? properties["camera"]
        : render::RenderGraphProperties::object();
    applyRuntimeNodeProperties(node->id, std::move(properties), status);

    if (camera.empty()) {
        return;
    }
    bool companionUpdated = false;
    for (const render::RenderGraphNode& candidate : renderGraph_.nodes()) {
        if (candidate.type != "NrdDenoisePass" && candidate.type != "StreamlineDlssRrPass") {
            continue;
        }
        render::RenderGraphProperties runtimeProperties = candidate.runtimeProperties.is_object()
            ? candidate.runtimeProperties
            : render::RenderGraphProperties::object();
        const render::RenderGraphProperties effectiveProperties = effectiveNodeProperties(candidate);
        const uint32_t resetSerial = effectiveProperties.value("resetSerial", 0u);
        runtimeProperties["camera"] = camera;
        runtimeProperties["resetSerial"] = resetSerial + 1u;
        companionUpdated = renderGraph_.setNodeRuntimeProperties(
            candidate.id,
            std::move(runtimeProperties)) || companionUpdated;
    }
    if (companionUpdated && graphExecutor_ != nullptr && !renderGraph_.dirty()) {
        graphExecutor_->syncRuntimeProperties(renderGraph_);
    }
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
    if (selectedNodeIndex() == nodeIndex) {
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
            .object = scene_.objectForNode(nodeIndex).entity(),
            .sceneLifetimeRevision = scene_.sceneGraph().lifetimeRevision(),
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

scene::ConstSceneObject EditorApplication::selectedSceneObject() const
{
    if (!scene_.valid() || sceneSelection_.object == scene::kNullSceneEntity ||
        sceneSelection_.sceneLifetimeRevision != scene_.sceneGraph().lifetimeRevision()) {
        return {};
    }
    const scene::ConstSceneObject object =
        scene_.sceneGraph().object(sceneSelection_.object);
    return object && object.hasComponent<scene::ActiveSceneComponent>()
        ? object
        : scene::ConstSceneObject{};
}

int32_t EditorApplication::selectedNodeIndex() const
{
    if (sceneSelection_.object != scene::kNullSceneEntity) {
        const scene::ConstSceneObject object = selectedSceneObject();
        if (!object) {
            return scene::kInvalidSceneIndex;
        }
        if (const scene::SourceNodeComponent* source =
                object.tryGetComponent<scene::SourceNodeComponent>()) {
            return source->nodeIndex >= 0 &&
                    static_cast<size_t>(source->nodeIndex) < scene_.nodes().size()
                ? source->nodeIndex
                : scene::kInvalidSceneIndex;
        }
        return scene::kInvalidSceneIndex;
    }
    int32_t nodeIndex = sceneSelection_.nodeIndex;
    if (nodeIndex == scene::kInvalidSceneIndex && sceneSelection_.type == SceneSelectionType::Node) {
        nodeIndex = sceneSelection_.index;
    }
    return nodeIndex >= 0 && static_cast<size_t>(nodeIndex) < scene_.nodes().size()
        ? nodeIndex
        : scene::kInvalidSceneIndex;
}

void EditorApplication::notifySceneTransformChanged()
{
    renderWorld_.notifySceneChanged();
    historyResources_.invalidateAll();
    historyFrameIndex_ = 0;
    if (graphExecutor_ == nullptr || !graphExecutor_->compiled()) {
        viewportPreviewValid_ = false;
    }
    viewportPreviewNeedsRender_ = true;
    if (sceneAccelerationStructure_ != nullptr && sceneAccelerationStructure_->valid()) {
        sceneAccelerationStructure_->clear();
        sceneAccelerationStructureStatus_ =
            "Static RTAS cleared after a scene transform edit; rebuild it if needed.";
    } else {
        sceneAccelerationStructureStatus_ =
            "Runtime ray tracing passes will synchronize transforms on the next frame.";
    }
}

void EditorApplication::notifyScenePropertiesChanged()
{
    renderWorld_.notifySceneChanged();
    historyResources_.invalidateAll();
    historyFrameIndex_ = 0;
    if (graphExecutor_ == nullptr || !graphExecutor_->compiled()) {
        viewportPreviewValid_ = false;
    }
    viewportPreviewNeedsRender_ = true;
}

bool EditorApplication::setSelectedObjectWorldMatrix(
    const float4x4& worldMatrix,
    std::string& reason)
{
    const scene::ConstSceneObject object = selectedSceneObject();
    if (!object || !matrixIsFinite(worldMatrix)) {
        reason = "Selected transform is invalid.";
        return false;
    }
    if (object.hasComponent<scene::GeneratedComponent>()) {
        reason = "Generated scene objects are read-only.";
        return false;
    }
    if (!object.hasComponent<scene::SourceNodeComponent>()) {
        reason = "Runtime-only scene objects are read-only until document serialization is available.";
        return false;
    }
    const scene::TransformComponent* transform =
        object.tryGetComponent<scene::TransformComponent>();
    if (transform == nullptr) {
        reason = "The selected object has no transform component.";
        return false;
    }
    const float currentDeterminant = affineDeterminant(transform->worldMatrix);
    const float editedDeterminant = affineDeterminant(worldMatrix);
    if (!std::isfinite(editedDeterminant) || std::abs(editedDeterminant) <= 0.0000001f ||
        (std::isfinite(currentDeterminant) && currentDeterminant * editedDeterminant < 0.0f)) {
        reason = "The transform edit would become singular or change handedness.";
        return false;
    }
    if (scene::matrixNearlyEqual(transform->worldMatrix, worldMatrix)) {
        return true;
    }

    if (!scene_.setObjectWorldMatrix(object.entity(), worldMatrix)) {
        reason = "The world transform could not be converted through the current parent hierarchy.";
        return false;
    }
    notifySceneTransformChanged();
    return true;
}

void EditorApplication::pushSceneEditCommand(
    scene::SceneEntity object,
    uint64_t sceneLifetimeRevision,
    SceneEditValue before,
    SceneEditValue after)
{
    bool valuesEqual = false;
    if (const float4x4* beforeMatrix = std::get_if<float4x4>(&before)) {
        const float4x4* afterMatrix = std::get_if<float4x4>(&after);
        valuesEqual = afterMatrix != nullptr && scene::matrixNearlyEqual(*beforeMatrix, *afterMatrix);
    } else if (const scene::CameraProperties* beforeCamera =
                   std::get_if<scene::CameraProperties>(&before)) {
        const scene::CameraProperties* afterCamera =
            std::get_if<scene::CameraProperties>(&after);
        valuesEqual = afterCamera != nullptr &&
            scene::cameraPropertiesNearlyEqual(*beforeCamera, *afterCamera);
    } else if (const scene::LightProperties* beforeLight =
                   std::get_if<scene::LightProperties>(&before)) {
        const scene::LightProperties* afterLight =
            std::get_if<scene::LightProperties>(&after);
        valuesEqual = afterLight != nullptr &&
            scene::lightPropertiesNearlyEqual(*beforeLight, *afterLight);
    }
    if (object == scene::kNullSceneEntity ||
        sceneLifetimeRevision != scene_.sceneGraph().lifetimeRevision() ||
        !scene_.sceneGraph().object(object) ||
        valuesEqual) {
        updateSceneDirtyState();
        return;
    }
    if (transformCommandCursor_ < transformCommands_.size()) {
        transformCommands_.erase(
            transformCommands_.begin() + static_cast<ptrdiff_t>(transformCommandCursor_),
            transformCommands_.end());
        if (savedTransformCommandCursor_ > static_cast<int64_t>(transformCommandCursor_)) {
            savedTransformCommandCursor_ = -1;
        }
    }
    transformCommands_.push_back(SceneEditCommand{
        .object = object,
        .sceneLifetimeRevision = sceneLifetimeRevision,
        .before = std::move(before),
        .after = std::move(after),
    });
    ++transformCommandCursor_;
    constexpr size_t kTransformCommandCapacity = 256;
    if (transformCommands_.size() > kTransformCommandCapacity) {
        transformCommands_.erase(transformCommands_.begin());
        --transformCommandCursor_;
        if (savedTransformCommandCursor_ > 0) {
            --savedTransformCommandCursor_;
        } else if (savedTransformCommandCursor_ == 0) {
            savedTransformCommandCursor_ = -1;
        }
    }
    updateSceneDirtyState();
}

void EditorApplication::pushTransformCommand(
    scene::SceneEntity object,
    uint64_t sceneLifetimeRevision,
    const float4x4& before,
    const float4x4& after)
{
    pushSceneEditCommand(object, sceneLifetimeRevision, before, after);
}

void EditorApplication::beginInspectorPropertyEdit(
    scene::SceneEntity object,
    uint64_t sceneLifetimeRevision,
    SceneEditValue before)
{
    if (inspectorPropertyEditing_ &&
        (inspectorPropertyEditingObject_ != object ||
            inspectorPropertyEditingSceneLifetime_ != sceneLifetimeRevision ||
            inspectorPropertyStartValue_.index() != before.index())) {
        finishActiveInspectorPropertyTransaction();
    }
    if (inspectorPropertyEditing_) {
        return;
    }
    inspectorPropertyEditing_ = true;
    inspectorPropertyEditingObject_ = object;
    inspectorPropertyEditingSceneLifetime_ = sceneLifetimeRevision;
    inspectorPropertyStartValue_ = std::move(before);
}

void EditorApplication::finishActiveInspectorPropertyTransaction()
{
    if (!inspectorPropertyEditing_) {
        return;
    }

    SceneEditValue after = inspectorPropertyStartValue_;
    const scene::ConstSceneObject object =
        inspectorPropertyEditingSceneLifetime_ == scene_.sceneGraph().lifetimeRevision()
        ? scene_.sceneGraph().object(inspectorPropertyEditingObject_)
        : scene::ConstSceneObject{};
    bool hasAfter = false;
    if (std::holds_alternative<scene::CameraProperties>(inspectorPropertyStartValue_)) {
        if (const scene::CameraComponent* camera =
                object.tryGetComponent<scene::CameraComponent>()) {
            after = camera->properties;
            hasAfter = true;
        }
    } else if (std::holds_alternative<scene::LightProperties>(inspectorPropertyStartValue_)) {
        if (const scene::LightComponent* light =
                object.tryGetComponent<scene::LightComponent>()) {
            after = light->properties;
            hasAfter = true;
        }
    }

    if (hasAfter) {
        pushSceneEditCommand(
            inspectorPropertyEditingObject_,
            inspectorPropertyEditingSceneLifetime_,
            inspectorPropertyStartValue_,
            std::move(after));
    }
    inspectorPropertyEditing_ = false;
    inspectorPropertyEditingObject_ = scene::kNullSceneEntity;
    inspectorPropertyEditingSceneLifetime_ = 0;
    inspectorPropertyStartValue_ = float4x4::Identity();
}

bool EditorApplication::applySceneEditValue(
    scene::SceneEntity object,
    const SceneEditValue& value)
{
    if (const float4x4* matrix = std::get_if<float4x4>(&value)) {
        if (!scene_.setObjectLocalMatrix(object, *matrix)) {
            return false;
        }
        notifySceneTransformChanged();
        return true;
    }
    if (const scene::CameraProperties* camera =
            std::get_if<scene::CameraProperties>(&value)) {
        if (!scene_.setObjectCameraProperties(object, *camera)) {
            return false;
        }
        notifyScenePropertiesChanged();
        return true;
    }
    const scene::LightProperties* light = std::get_if<scene::LightProperties>(&value);
    if (light == nullptr || !scene_.setObjectLightProperties(object, *light)) {
        return false;
    }
    notifyScenePropertiesChanged();
    return true;
}

void EditorApplication::updateSceneDirtyState()
{
    const bool transformDirty = savedTransformCommandCursor_ < 0 ||
        static_cast<int64_t>(transformCommandCursor_) != savedTransformCommandCursor_;
    scene_.setDirty(sceneNonTransformDirty_ || transformDirty);
}

void EditorApplication::finishActiveTransformTransactions()
{
    if (gizmoWasUsing_) {
        const scene::ConstSceneObject editedObject =
            gizmoEditingSceneLifetime_ == scene_.sceneGraph().lifetimeRevision()
            ? scene_.sceneGraph().object(gizmoEditingObject_)
            : scene::ConstSceneObject{};
        if (const scene::TransformComponent* transform =
                editedObject.tryGetComponent<scene::TransformComponent>()) {
            pushTransformCommand(
                gizmoEditingObject_,
                gizmoEditingSceneLifetime_,
                gizmoStartLocalMatrix_,
                transform->localMatrix);
        }
        if (ImGuizmo::IsUsingAny()) {
            ImGuizmo::Enable(false);
            ImGuizmo::Enable(true);
        }
        gizmoWasUsing_ = false;
        gizmoEditingObject_ = scene::kNullSceneEntity;
        gizmoEditingSceneLifetime_ = 0;
        viewportGizmoCapturingMouse_ = false;
    }

    if (inspectorTransformEditing_) {
        const scene::ConstSceneObject editedObject =
            inspectorEditingSceneLifetime_ == scene_.sceneGraph().lifetimeRevision()
            ? scene_.sceneGraph().object(inspectorEditingObject_)
            : scene::ConstSceneObject{};
        if (const scene::TransformComponent* transform =
                editedObject.tryGetComponent<scene::TransformComponent>()) {
            pushTransformCommand(
                inspectorEditingObject_,
                inspectorEditingSceneLifetime_,
                inspectorStartLocalMatrix_,
                transform->localMatrix);
        }
        inspectorTransformEditing_ = false;
        inspectorEditingObject_ = scene::kNullSceneEntity;
        inspectorEditingSceneLifetime_ = 0;
    }
    finishActiveInspectorPropertyTransaction();
}

void EditorApplication::undoTransform()
{
    finishActiveTransformTransactions();
    if (transformCommandCursor_ == 0) {
        return;
    }
    --transformCommandCursor_;
    const SceneEditCommand& command = transformCommands_[transformCommandCursor_];
    if (command.sceneLifetimeRevision != scene_.sceneGraph().lifetimeRevision() ||
        !applySceneEditValue(command.object, command.before)) {
        ++transformCommandCursor_;
        return;
    }
    updateSceneDirtyState();
}

void EditorApplication::redoTransform()
{
    finishActiveTransformTransactions();
    if (transformCommandCursor_ >= transformCommands_.size()) {
        return;
    }
    const SceneEditCommand& command = transformCommands_[transformCommandCursor_];
    if (command.sceneLifetimeRevision != scene_.sceneGraph().lifetimeRevision() ||
        !applySceneEditValue(command.object, command.after)) {
        return;
    }
    ++transformCommandCursor_;
    updateSceneDirtyState();
}

void EditorApplication::resetTransformHistory()
{
    transformCommands_.clear();
    transformCommandCursor_ = 0;
    savedTransformCommandCursor_ = 0;
    gizmoWasUsing_ = false;
    gizmoEditingObject_ = scene::kNullSceneEntity;
    gizmoEditingSceneLifetime_ = 0;
    inspectorTransformEditing_ = false;
    inspectorEditingObject_ = scene::kNullSceneEntity;
    inspectorEditingSceneLifetime_ = 0;
    inspectorPropertyEditing_ = false;
    inspectorPropertyEditingObject_ = scene::kNullSceneEntity;
    inspectorPropertyEditingSceneLifetime_ = 0;
    inspectorPropertyStartValue_ = float4x4::Identity();
    sceneNonTransformDirty_ = false;
    environmentEditBaselineValid_ = false;
    environmentEditBaselineUserEdited_ = false;
    environmentEditBaselineFromSample_ = false;
    scenePicker_.clear();
    scene_.setDirty(false);
}

void EditorApplication::saveScene()
{
    finishActiveTransformTransactions();
    if (!scene_.valid()) {
        sceneStatus_ = "No scene is open.";
        return;
    }
    std::string message;
    if (!scene_.save(message)) {
        sceneStatus_ = message.empty() ? "Failed to save scene document." : message;
        return;
    }
    const bool persistedEnvironmentEdit = sceneNonTransformDirty_ &&
        environmentEditBaselineValid_;
    savedTransformCommandCursor_ = static_cast<int64_t>(transformCommandCursor_);
    sceneNonTransformDirty_ = false;
    if (persistedEnvironmentEdit) {
        environmentUserEdited_ = false;
    }
    environmentEditBaselineValid_ = false;
    environmentEditBaselineUserEdited_ = false;
    environmentEditBaselineFromSample_ = false;
    scene_.setDirty(false);
    sceneStatus_ = "Saved scene overrides: " + scene_.documentPath().string();
}

void EditorApplication::requestPendingSceneAction(
    PendingSceneAction action,
    std::filesystem::path path,
    std::string value)
{
    finishActiveTransformTransactions();
    pendingSceneAction_ = action;
    pendingScenePath_ = std::move(path);
    pendingSceneValue_ = std::move(value);
    if (!scene_.dirty()) {
        executePendingSceneAction();
    }
}

void EditorApplication::executePendingSceneAction()
{
    const PendingSceneAction action = pendingSceneAction_;
    const std::filesystem::path path = std::move(pendingScenePath_);
    const std::string value = std::move(pendingSceneValue_);
    pendingSceneAction_ = PendingSceneAction::None;
    pendingScenePath_.clear();
    pendingSceneValue_.clear();

    switch (action) {
    case PendingSceneAction::Clear:
        cancelSceneLoad();
        clearSceneAccelerationStructure();
        scene_.clear();
        renderWorld_.notifySceneChanged();
        resetTransformHistory();
        sceneSelection_ = SceneSelection{};
        historyResources_.invalidateAll();
        viewportPreviewValid_ = false;
        sceneStatus_ = "No scene loaded.";
        break;
    case PendingSceneAction::Exit:
        running_ = false;
        break;
    case PendingSceneAction::LoadScene:
        copyToBuffer(path.string(), sceneFilePath_, sizeof(sceneFilePath_));
        loadScene();
        break;
    case PendingSceneAction::LoadRenderGraph:
        copyToBuffer(path.string(), graphFilePath_, sizeof(graphFilePath_));
        loadRenderGraph();
        break;
    case PendingSceneAction::LoadSample:
        loadBuiltInSample(value.c_str());
        break;
    case PendingSceneAction::CommitLoadedScene:
        if (readySceneLoad_ != nullptr) {
            commitLoadedScene(std::move(readySceneLoad_));
        }
        break;
    case PendingSceneAction::None:
        break;
    }
}

void EditorApplication::drawUnsavedSceneModal()
{
    if (pendingSceneAction_ == PendingSceneAction::None || !scene_.dirty()) {
        return;
    }
    ImGui::OpenPopup("Unsaved Scene Changes");
    if (!ImGui::BeginPopupModal(
            "Unsaved Scene Changes",
            nullptr,
            ImGuiWindowFlags_AlwaysAutoResize)) {
        return;
    }
    ImGui::TextWrapped(
        "Save changes to %s before continuing?",
        scene_.documentPath().string().c_str());
    ImGui::Spacing();
    if (ImGui::Button("Save", ImVec2(100.0f * mainScale_, 0.0f))) {
        saveScene();
        if (!scene_.dirty()) {
            ImGui::CloseCurrentPopup();
            executePendingSceneAction();
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Discard", ImVec2(100.0f * mainScale_, 0.0f))) {
        std::string message;
        if (!scene_.revert(message)) {
            sceneStatus_ = message.empty()
                ? "Failed to discard scene changes."
                : "Failed to discard scene changes: " + message;
        } else {
            sceneSelection_ = SceneSelection{};
            renderWorld_.notifySceneChanged();
            if (environmentEditBaselineValid_) {
                renderWorld_.setEnvironment(environmentEditBaseline_);
                environmentUserEdited_ = environmentEditBaselineUserEdited_;
                environmentFromSample_ = environmentEditBaselineFromSample_;
                preserveSampleEnvironmentForNextSceneLoad_ = environmentFromSample_;
            }
            resetTransformHistory();
            historyResources_.invalidateAll();
            historyFrameIndex_ = 0;
            viewportPreviewValid_ = false;
            viewportPreviewNeedsRender_ = true;
            clearSceneAccelerationStructure();
            sceneStatus_ = std::move(message);
            ImGui::CloseCurrentPopup();
            executePendingSceneAction();
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel", ImVec2(100.0f * mainScale_, 0.0f))) {
        if (pendingSceneAction_ == PendingSceneAction::CommitLoadedScene) {
            if (graphExecutor_ != nullptr) {
                graphExecutor_->cancelSceneResourcePreparation();
            }
            readySceneLoad_.reset();
            pendingSceneLoad_ = {};
            pendingSceneLoadPath_.clear();
            sceneStatus_ = "Loaded scene switch cancelled.";
        }
        pendingSceneAction_ = PendingSceneAction::None;
        pendingScenePath_.clear();
        pendingSceneValue_.clear();
        ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
}

void EditorApplication::drawSelectedNodeTransformInspector()
{
    const scene::ConstSceneObject object = selectedSceneObject();
    if (!object) {
        return;
    }
    if (object.hasComponent<scene::GeneratedComponent>()) {
        ImGui::TextDisabled("Generated scene objects are read-only.");
        return;
    }
    if (!object.hasComponent<scene::SourceNodeComponent>()) {
        ImGui::TextDisabled("Runtime-only object transforms are not serialized and are read-only.");
        return;
    }
    if (inspectorPropertyEditing_) {
        ImGui::TextDisabled("Finish the component property edit before using transform controls.");
        return;
    }
    if (gizmoWasUsing_ || ImGuizmo::IsUsingAny()) {
        ImGui::TextDisabled("Finish the viewport gizmo edit before using Inspector controls.");
        return;
    }
    if (inspectorTransformEditing_ &&
        (inspectorEditingObject_ != object.entity() ||
            inspectorEditingSceneLifetime_ != scene_.sceneGraph().lifetimeRevision())) {
        const scene::ConstSceneObject editedObject =
            inspectorEditingSceneLifetime_ == scene_.sceneGraph().lifetimeRevision()
            ? scene_.sceneGraph().object(inspectorEditingObject_)
            : scene::ConstSceneObject{};
        if (const scene::TransformComponent* editedTransform =
                editedObject.tryGetComponent<scene::TransformComponent>()) {
            pushTransformCommand(
                inspectorEditingObject_,
                inspectorEditingSceneLifetime_,
                inspectorStartLocalMatrix_,
                editedTransform->localMatrix);
        }
        inspectorTransformEditing_ = false;
        inspectorEditingObject_ = scene::kNullSceneEntity;
        inspectorEditingSceneLifetime_ = 0;
    }
    const scene::TransformComponent* transform =
        object.tryGetComponent<scene::TransformComponent>();
    if (transform == nullptr) {
        return;
    }

    const float4x4 originalLocal = transform->localMatrix;
    const bool finite = matrixIsFinite(originalLocal);
    const float determinant = affineDeterminant(originalLocal);
    const bool singular = !finite ||
        !std::isfinite(determinant) || std::abs(determinant) <= 0.0000001f;
    const bool shear = !singular && matrixHasShear(originalLocal);
    const bool reflection = !singular && determinant < 0.0f;
    float translation[3] = {
        originalLocal.a03,
        originalLocal.a13,
        originalLocal.a23,
    };
    float rotation[3] = {};
    float scale[3] = {1.0f, 1.0f, 1.0f};
    if (!singular) {
        ImGuizmo::DecomposeMatrixToComponents(originalLocal.a, translation, rotation, scale);
        translation[0] = originalLocal.a03;
        translation[1] = originalLocal.a13;
        translation[2] = originalLocal.a23;
    }

    ImGui::TextUnformatted("Local Transform");
    if (!finite) {
        ImGui::TextDisabled("Read-only: the local matrix contains non-finite values.");
    } else if (singular) {
        ImGui::TextDisabled("Singular transform: only translation can be edited safely.");
    } else if (shear) {
        ImGui::TextDisabled("Shear detected: only translation can be edited safely.");
    } else if (reflection) {
        ImGui::TextDisabled("Reflected transform: rotation and scale are read-only.");
    }

    const auto applyEdit = [&](bool changed, bool translationOnly) {
        if (ImGui::IsItemActivated()) {
            inspectorStartLocalMatrix_ = originalLocal;
            inspectorTransformEditing_ = true;
            inspectorEditingObject_ = object.entity();
            inspectorEditingSceneLifetime_ = scene_.sceneGraph().lifetimeRevision();
        }
        if (changed) {
            float4x4 edited = originalLocal;
            bool validEdit = true;
            if (translationOnly) {
                edited.a03 = translation[0];
                edited.a13 = translation[1];
                edited.a23 = translation[2];
            } else {
                constexpr float kMinimumEditableScale = 0.0001f;
                if (scale[0] <= kMinimumEditableScale ||
                    scale[1] <= kMinimumEditableScale ||
                    scale[2] <= kMinimumEditableScale) {
                    sceneStatus_ = "Scale edit rejected: scale components must remain positive and non-zero.";
                    validEdit = false;
                } else {
                    ImGuizmo::RecomposeMatrixFromComponents(translation, rotation, scale, edited.a);
                    const float editedDeterminant = affineDeterminant(edited);
                    if (!matrixIsFinite(edited) || !std::isfinite(editedDeterminant) ||
                        editedDeterminant <= 0.0000001f) {
                        sceneStatus_ = "Transform edit rejected: the result would be singular or reflected.";
                        validEdit = false;
                    }
                }
            }
            if (validEdit && scene_.setObjectLocalMatrix(object.entity(), edited)) {
                notifySceneTransformChanged();
            }
        }
        if (ImGui::IsItemDeactivatedAfterEdit() && inspectorTransformEditing_) {
            const scene::ConstSceneObject editedObject =
                inspectorEditingSceneLifetime_ == scene_.sceneGraph().lifetimeRevision()
                ? scene_.sceneGraph().object(inspectorEditingObject_)
                : scene::ConstSceneObject{};
            if (const scene::TransformComponent* editedTransform =
                    editedObject.tryGetComponent<scene::TransformComponent>()) {
                pushTransformCommand(
                    inspectorEditingObject_,
                    inspectorEditingSceneLifetime_,
                    inspectorStartLocalMatrix_,
                    editedTransform->localMatrix);
            }
            inspectorTransformEditing_ = false;
            inspectorEditingObject_ = scene::kNullSceneEntity;
            inspectorEditingSceneLifetime_ = 0;
        }
    };

    ImGui::BeginDisabled(!finite);
    const bool translationChanged = ImGui::DragFloat3("Position", translation, 0.01f);
    applyEdit(translationChanged, true);
    ImGui::EndDisabled();

    ImGui::BeginDisabled(singular || shear || reflection);
    const bool rotationChanged = ImGui::DragFloat3("Rotation", rotation, 0.25f);
    applyEdit(rotationChanged, false);
    const bool scaleChanged = ImGui::DragFloat3("Scale", scale, 0.01f);
    applyEdit(scaleChanged, false);
    ImGui::EndDisabled();

    ImGui::Spacing();
    ImGui::TextUnformatted("World Transform (read-only)");
    for (uint32_t row = 0; row < 4; ++row) {
        const float4 values = transform->worldMatrix.Row(row);
        ImGui::TextDisabled("% .5f  % .5f  % .5f  % .5f", values.x, values.y, values.z, values.w);
    }
}

void EditorApplication::drawSelectedCameraComponentInspector()
{
    const scene::ConstSceneObject object = selectedSceneObject();
    const scene::CameraComponent* component =
        object.tryGetComponent<scene::CameraComponent>();
    if (component == nullptr) {
        ImGui::TextDisabled("CameraComponent is unavailable; properties are read-only.");
        return;
    }

    const scene::RenderCamera* snapshot = nullptr;
    if (component->renderCameraIndex >= 0 &&
        static_cast<size_t>(component->renderCameraIndex) < scene_.cameras().size()) {
        const scene::RenderCamera& candidate =
            scene_.cameras()[static_cast<size_t>(component->renderCameraIndex)];
        if (candidate.object == object.entity()) {
            snapshot = &candidate;
        }
    }

    const scene::CameraProperties& properties = component->properties;
    const bool supportedType = properties.type == scene::CameraType::Perspective ||
        properties.type == scene::CameraType::Orthographic;
    const char* readOnlyReason = nullptr;
    if (object.hasComponent<scene::GeneratedComponent>()) {
        readOnlyReason = "Generated cameras are runtime-owned and cannot be saved.";
    } else if (!object.hasComponent<scene::SourceNodeComponent>()) {
        readOnlyReason = "Runtime-only cameras are read-only until document serialization is available.";
    } else if (!scene::validCameraProperties(properties)) {
        readOnlyReason = "The source camera contains values unsupported by the editor and is read-only.";
    } else if (snapshot == nullptr) {
        readOnlyReason = "Camera render snapshot mapping is unavailable; properties are read-only.";
    } else if (!supportedType) {
        readOnlyReason = "This camera projection type is not supported by the Inspector.";
    } else if (gizmoWasUsing_ || inspectorTransformEditing_ || ImGuizmo::IsUsingAny()) {
        readOnlyReason = "Finish the active transform edit before changing camera properties.";
    } else if (inspectorPropertyEditing_ &&
        (inspectorPropertyEditingObject_ != object.entity() ||
            inspectorPropertyEditingSceneLifetime_ != scene_.sceneGraph().lifetimeRevision() ||
            !std::holds_alternative<scene::CameraProperties>(inspectorPropertyStartValue_))) {
        readOnlyReason = "Finish the active component edit before changing camera properties.";
    }

    ImGui::PushID("CameraComponentInspector");
    ImGui::TextUnformatted("CameraComponent");
    if (snapshot != nullptr) {
        ImGui::Text("Name: %s", snapshot->name.c_str());
    }
    ImGui::Text("Type: %s (read-only)", scene::cameraTypeName(properties.type));
    if (readOnlyReason != nullptr) {
        ImGui::TextDisabled("%s", readOnlyReason);
    }

    scene::CameraProperties edited = properties;
    bool changed = false;
    bool editDeactivated = false;
    const auto trackEditItem = [this, &object, &properties, &editDeactivated]() {
        if (ImGui::IsItemActivated()) {
            beginInspectorPropertyEdit(
                object.entity(),
                scene_.sceneGraph().lifetimeRevision(),
                properties);
        }
        editDeactivated = ImGui::IsItemDeactivated() || editDeactivated;
    };
    constexpr double kRadiansToDegrees = 57.2957795130823208768;
    constexpr double kDegreesToRadians = 0.01745329251994329577;
    constexpr float kSmallStep = 0.01f;
    constexpr float kAngleStep = 0.1f;
    constexpr double kPositiveMinimum = 0.000001;
    constexpr double kMaximumValue = 1000000000000.0;
    constexpr double kMinimumFovDegrees = 1.0;
    constexpr double kMaximumFovDegrees = 179.0;
    constexpr double kMaximumAspectRatio = 100.0;
    constexpr double kAutomaticValue = 0.0;

    ImGui::BeginDisabled(readOnlyReason != nullptr);
    if (properties.type == scene::CameraType::Perspective) {
        double fovDegrees = edited.yfov * kRadiansToDegrees;
        if (ImGui::DragScalar(
                "Vertical FOV (deg)",
                ImGuiDataType_Double,
                &fovDegrees,
                kAngleStep,
                &kMinimumFovDegrees,
                &kMaximumFovDegrees,
                "%.3f",
                ImGuiSliderFlags_AlwaysClamp)) {
            edited.yfov = fovDegrees * kDegreesToRadians;
            changed = true;
        }
        trackEditItem();
        const bool aspectChanged = ImGui::DragScalar(
            "Aspect Ratio",
            ImGuiDataType_Double,
            &edited.aspectRatio,
            kSmallStep,
            &kAutomaticValue,
            &kMaximumAspectRatio,
            "%.6f",
            ImGuiSliderFlags_AlwaysClamp);
        trackEditItem();
        changed = aspectChanged || changed;
        if (edited.aspectRatio == 0.0) {
            ImGui::TextDisabled("Aspect: auto (uses the render target aspect)");
        } else {
            ImGui::TextDisabled("Set Aspect Ratio to 0 for auto.");
        }
    } else if (properties.type == scene::CameraType::Orthographic) {
        const bool xmagChanged = ImGui::DragScalar(
            "X Magnification",
            ImGuiDataType_Double,
            &edited.xmag,
            kSmallStep,
            &kPositiveMinimum,
            &kMaximumValue,
            "%.6f",
            ImGuiSliderFlags_AlwaysClamp);
        trackEditItem();
        changed = xmagChanged || changed;
        const bool ymagChanged = ImGui::DragScalar(
            "Y Magnification",
            ImGuiDataType_Double,
            &edited.ymag,
            kSmallStep,
            &kPositiveMinimum,
            &kMaximumValue,
            "%.6f",
            ImGuiSliderFlags_AlwaysClamp);
        trackEditItem();
        changed = ymagChanged || changed;
    }

    const double nearMinimum = properties.type == scene::CameraType::Perspective
        ? kPositiveMinimum
        : kAutomaticValue;
    const bool nearChanged = ImGui::DragScalar(
        "Z Near",
        ImGuiDataType_Double,
        &edited.znear,
        kSmallStep,
        &nearMinimum,
        &kMaximumValue,
        "%.6f",
        ImGuiSliderFlags_AlwaysClamp);
    trackEditItem();
    changed = nearChanged || changed;
    const double farMinimum = properties.type == scene::CameraType::Perspective
        ? kAutomaticValue
        : std::min(edited.znear + kPositiveMinimum, kMaximumValue);
    const bool farChanged = ImGui::DragScalar(
        "Z Far",
        ImGuiDataType_Double,
        &edited.zfar,
        kSmallStep,
        &farMinimum,
        &kMaximumValue,
        "%.6f",
        ImGuiSliderFlags_AlwaysClamp);
    trackEditItem();
    changed = farChanged || changed;
    if (properties.type == scene::CameraType::Perspective) {
        if (edited.zfar == 0.0) {
            ImGui::TextDisabled("Z Far: infinite");
        } else {
            ImGui::TextDisabled("Set Z Far to 0 for an infinite far plane.");
        }
    }
    ImGui::EndDisabled();

    if (changed) {
        if (!inspectorPropertyEditing_) {
            beginInspectorPropertyEdit(
                object.entity(),
                scene_.sceneGraph().lifetimeRevision(),
                properties);
        }
        if (scene_.setObjectCameraProperties(object.entity(), edited)) {
            notifyScenePropertiesChanged();
            sceneStatus_ = "Updated CameraComponent properties.";
        } else {
            sceneStatus_ = "Camera property edit rejected; the previous values were preserved.";
        }
    }
    if (editDeactivated) {
        finishActiveInspectorPropertyTransaction();
    }
    ImGui::PopID();
}

void EditorApplication::drawSelectedLightComponentInspector()
{
    const scene::ConstSceneObject object = selectedSceneObject();
    const scene::LightComponent* component =
        object.tryGetComponent<scene::LightComponent>();
    if (component == nullptr) {
        ImGui::TextDisabled("LightComponent is unavailable; properties are read-only.");
        return;
    }

    const scene::RenderLight* snapshot = nullptr;
    if (component->renderLightIndex >= 0 &&
        static_cast<size_t>(component->renderLightIndex) < scene_.lights().size()) {
        const scene::RenderLight& candidate =
            scene_.lights()[static_cast<size_t>(component->renderLightIndex)];
        if (candidate.object == object.entity()) {
            snapshot = &candidate;
        }
    }

    const scene::LightProperties& properties = component->properties;
    const bool directional = properties.type == "directional";
    const bool point = properties.type == "point";
    const bool spot = properties.type == "spot";
    const bool supportedType = directional || point || spot;
    const char* readOnlyReason = nullptr;
    if (object.hasComponent<scene::GeneratedComponent>()) {
        readOnlyReason = "Generated lights are runtime-owned and cannot be saved.";
    } else if (!object.hasComponent<scene::SourceNodeComponent>()) {
        readOnlyReason = "Runtime-only lights are read-only until document serialization is available.";
    } else if (!scene::validLightProperties(properties)) {
        readOnlyReason = "The source light contains values unsupported by the editor and is read-only.";
    } else if (snapshot == nullptr) {
        readOnlyReason = "Light render snapshot mapping is unavailable; properties are read-only.";
    } else if (!supportedType) {
        readOnlyReason = "This light type is not supported by the Inspector.";
    } else if (gizmoWasUsing_ || inspectorTransformEditing_ || ImGuizmo::IsUsingAny()) {
        readOnlyReason = "Finish the active transform edit before changing light properties.";
    } else if (inspectorPropertyEditing_ &&
        (inspectorPropertyEditingObject_ != object.entity() ||
            inspectorPropertyEditingSceneLifetime_ != scene_.sceneGraph().lifetimeRevision() ||
            !std::holds_alternative<scene::LightProperties>(inspectorPropertyStartValue_))) {
        readOnlyReason = "Finish the active component edit before changing light properties.";
    }

    ImGui::PushID("LightComponentInspector");
    ImGui::TextUnformatted("LightComponent");
    if (snapshot != nullptr) {
        ImGui::Text("Name: %s", snapshot->name.c_str());
    }
    ImGui::Text("Type: %s (read-only)", properties.type.c_str());
    if (readOnlyReason != nullptr) {
        ImGui::TextDisabled("%s", readOnlyReason);
    }

    scene::LightProperties edited = properties;
    bool changed = false;
    bool editDeactivated = false;
    const auto trackEditItem = [this, &object, &properties, &editDeactivated]() {
        if (ImGui::IsItemActivated()) {
            beginInspectorPropertyEdit(
                object.entity(),
                scene_.sceneGraph().lifetimeRevision(),
                properties);
        }
        editDeactivated = ImGui::IsItemDeactivated() || editDeactivated;
    };
    float color[3] = {edited.color.x, edited.color.y, edited.color.z};
    constexpr double kRadiansToDegrees = 57.2957795130823208768;
    constexpr double kDegreesToRadians = 0.01745329251994329577;
    constexpr float kSmallStep = 0.01f;
    constexpr float kIntensityStep = 0.1f;
    constexpr float kAngleStep = 0.1f;
    constexpr double kMinimumValue = 0.0;
    constexpr double kMaximumValue = 1000000000000.0;
    constexpr double kMaximumConeDegrees = 90.0;

    ImGui::BeginDisabled(readOnlyReason != nullptr);
    if (ImGui::ColorEdit3("Color", color, ImGuiColorEditFlags_Float)) {
        edited.color = float3(color[0], color[1], color[2]);
        changed = true;
    }
    trackEditItem();
    const bool intensityChanged = ImGui::DragScalar(
        "Intensity",
        ImGuiDataType_Double,
        &edited.intensity,
        kIntensityStep,
        &kMinimumValue,
        &kMaximumValue,
        "%.3f",
        ImGuiSliderFlags_AlwaysClamp);
    trackEditItem();
    changed = intensityChanged || changed;

    if (point || spot) {
        const bool rangeChanged = ImGui::DragScalar(
            "Range",
            ImGuiDataType_Double,
            &edited.range,
            kSmallStep,
            &kMinimumValue,
            &kMaximumValue,
            "%.3f",
            ImGuiSliderFlags_AlwaysClamp);
        trackEditItem();
        changed = rangeChanged || changed;
        if (edited.range == 0.0) {
            ImGui::TextDisabled("Range: unbounded");
        } else {
            ImGui::TextDisabled("Set Range to 0 for an unbounded light.");
        }
    } else if (directional) {
        ImGui::TextDisabled("Range is not supported by directional lights.");
    }

    if (spot) {
        double innerDegrees = edited.innerConeAngle * kRadiansToDegrees;
        double outerDegrees = edited.outerConeAngle * kRadiansToDegrees;
        if (ImGui::DragScalar(
                "Inner Cone (deg)",
                ImGuiDataType_Double,
                &innerDegrees,
                kAngleStep,
                &kMinimumValue,
                &kMaximumConeDegrees,
                "%.3f",
                ImGuiSliderFlags_AlwaysClamp)) {
            edited.innerConeAngle = innerDegrees * kDegreesToRadians;
            changed = true;
        }
        trackEditItem();
        if (ImGui::DragScalar(
                "Outer Cone (deg)",
                ImGuiDataType_Double,
                &outerDegrees,
                kAngleStep,
                &kMinimumValue,
                &kMaximumConeDegrees,
                "%.3f",
                ImGuiSliderFlags_AlwaysClamp)) {
            edited.outerConeAngle = outerDegrees * kDegreesToRadians;
            changed = true;
        }
        trackEditItem();
    } else if (point) {
        ImGui::TextDisabled("Cone angles are not supported by point lights.");
    } else if (directional) {
        ImGui::TextDisabled("Cone angles are not supported by directional lights.");
    }
    ImGui::EndDisabled();

    if (changed) {
        if (!inspectorPropertyEditing_) {
            beginInspectorPropertyEdit(
                object.entity(),
                scene_.sceneGraph().lifetimeRevision(),
                properties);
        }
        if (scene_.setObjectLightProperties(object.entity(), edited)) {
            notifyScenePropertiesChanged();
            sceneStatus_ = "Updated LightComponent properties.";
        } else {
            sceneStatus_ = "Light property edit rejected; the previous values were preserved.";
        }
    }
    if (editDeactivated) {
        finishActiveInspectorPropertyTransaction();
    }
    ImGui::PopID();
}

void EditorApplication::drawViewportGizmo(const ImVec2& min, const ImVec2& max)
{
    viewportGizmoCapturingMouse_ = false;
    const scene::ConstSceneObject object = selectedSceneObject();
    render::RenderGraphNode* renderNode = viewportCameraRenderGraphNode();

    const auto finishInterruptedTransaction = [this]() {
        if (gizmoWasUsing_ &&
            gizmoEditingSceneLifetime_ == scene_.sceneGraph().lifetimeRevision()) {
            const scene::ConstSceneObject editedObject =
                scene_.sceneGraph().object(gizmoEditingObject_);
            if (const scene::TransformComponent* transform =
                    editedObject.tryGetComponent<scene::TransformComponent>()) {
                pushTransformCommand(
                    gizmoEditingObject_,
                    gizmoEditingSceneLifetime_,
                    gizmoStartLocalMatrix_,
                    transform->localMatrix);
            }
        }
        if (ImGuizmo::IsUsingAny()) {
            ImGuizmo::Enable(false);
            ImGuizmo::Enable(true);
        }
        gizmoWasUsing_ = false;
        gizmoEditingObject_ = scene::kNullSceneEntity;
        gizmoEditingSceneLifetime_ = 0;
    };

    if (gizmoWasUsing_ &&
        (gizmoEditingObject_ != object.entity() ||
            gizmoEditingSceneLifetime_ != scene_.sceneGraph().lifetimeRevision())) {
        finishInterruptedTransaction();
    }
    if (!viewportInteractionEnabled_ || inspectorTransformEditing_ ||
        inspectorPropertyEditing_ || !object ||
        object.hasComponent<scene::GeneratedComponent>() ||
        !object.hasComponent<scene::SourceNodeComponent>() || renderNode == nullptr) {
        finishInterruptedTransaction();
        return;
    }

    const scene::TransformComponent* transform =
        object.tryGetComponent<scene::TransformComponent>();
    if (transform == nullptr) {
        finishInterruptedTransaction();
        return;
    }
    const float localDeterminant = affineDeterminant(transform->localMatrix);
    const bool singular = !matrixIsFinite(transform->localMatrix) ||
        !std::isfinite(localDeterminant) || std::abs(localDeterminant) <= 0.0000001f;
    const bool shear = !singular && matrixHasShear(transform->localMatrix);
    const bool reflection = !singular && localDeterminant < 0.0f;
    const float worldDeterminant = affineDeterminant(transform->worldMatrix);
    const bool worldInvalid = !matrixIsFinite(transform->worldMatrix) ||
        !std::isfinite(worldDeterminant) || std::abs(worldDeterminant) <= 0.0000001f;
    const bool worldShear = !worldInvalid && matrixHasShear(transform->worldMatrix);
    const bool worldReflection = !worldInvalid && worldDeterminant < 0.0f;
    bool parentSingular = false;
    bool parentDistortsLinearTransform = false;
    if (!object.hasComponent<scene::RootComponent>()) {
        if (const scene::RelationshipComponent* relationship =
            object.tryGetComponent<scene::RelationshipComponent>();
            relationship != nullptr && relationship->parent != scene::kNullSceneEntity) {
            const scene::ConstSceneObject parent =
                scene_.sceneGraph().object(relationship->parent);
            const scene::TransformComponent* parentTransform =
                parent.tryGetComponent<scene::TransformComponent>();
            if (parentTransform == nullptr) {
                parentSingular = true;
            } else {
                const float parentDeterminant = affineDeterminant(parentTransform->worldMatrix);
                parentSingular = !std::isfinite(parentDeterminant) ||
                    std::abs(parentDeterminant) <= 0.0000001f;
                parentDistortsLinearTransform = !parentSingular &&
                    (parentDeterminant < 0.0f ||
                        matrixHasShear(parentTransform->worldMatrix) ||
                        matrixHasNonUniformScale(parentTransform->worldMatrix));
            }
        }
    }
    const bool editsLinearTransform = gizmoOperation_ != GizmoOperation::Translate;
    const bool unsafeLinearEdit = editsLinearTransform &&
        (shear || reflection || worldShear || worldReflection ||
            parentDistortsLinearTransform);
    const bool operationBlocked = singular || worldInvalid || parentSingular || unsafeLinearEdit;
    if (operationBlocked) {
        const char* reason = singular
            ? "Gizmo disabled: singular local transform"
            : (worldInvalid
                    ? "Gizmo disabled: invalid world transform"
                    : (parentSingular
                            ? "Gizmo disabled: singular parent transform"
                            : "Rotate/scale disabled: hierarchy contains reflection, shear, or non-uniform parent scale"));
        ImGui::GetWindowDrawList()->AddText(
            ImVec2(min.x + 10.0f, max.y - 28.0f * mainScale_),
            IM_COL32(255, 190, 90, 255),
            reason);
        finishInterruptedTransaction();
        return;
    }

    render::RenderGraphProperties properties = effectiveNodeProperties(*renderNode);
    ensureCameraProperties(properties, scene_.bounds());
    const ViewportCameraMatrices matrices = viewportCameraMatrices(
        properties["camera"],
        max.x - min.x,
        max.y - min.y);
    ImGuizmo::SetDrawlist(ImGui::GetWindowDrawList());
    ImGuizmo::SetRect(min.x, min.y, max.x - min.x, max.y - min.y);
    ImGuizmo::SetOrthographic(matrices.orthographic);

    ImGuizmo::OPERATION operation = ImGuizmo::TRANSLATE;
    if (gizmoOperation_ == GizmoOperation::Rotate) {
        operation = ImGuizmo::ROTATE;
    } else if (gizmoOperation_ == GizmoOperation::Scale) {
        operation = ImGuizmo::SCALE;
    }
    const ImGuizmo::MODE mode = gizmoOperation_ == GizmoOperation::Scale || gizmoLocal_
        ? ImGuizmo::LOCAL
        : ImGuizmo::WORLD;
    const bool useSnap = snapEnabled_ || ImGui::GetIO().KeyCtrl;
    const float step = gizmoOperation_ == GizmoOperation::Translate
        ? translateSnap_
        : (gizmoOperation_ == GizmoOperation::Rotate ? rotateSnap_ : scaleSnap_);
    const float snap[3] = {step, step, step};
    float4x4 editedWorld = transform->worldMatrix;
    const float4x4 localBeforeManipulate = transform->localMatrix;
    ImGuizmo::PushID(static_cast<int>(object.entity()));
    const bool manipulated = ImGuizmo::Manipulate(
        matrices.view.a,
        matrices.projection.a,
        operation,
        mode,
        editedWorld.a,
        nullptr,
        useSnap ? snap : nullptr);
    const bool usingNow = ImGuizmo::IsUsing();
    const bool overNow = ImGuizmo::IsOver();
    ImGuizmo::PopID();
    viewportGizmoCapturingMouse_ = usingNow || overNow || ImGuizmo::IsUsingAny();
    if (usingNow && !gizmoWasUsing_) {
        gizmoStartLocalMatrix_ = localBeforeManipulate;
        gizmoEditingObject_ = object.entity();
        gizmoEditingSceneLifetime_ = scene_.sceneGraph().lifetimeRevision();
    }
    if (manipulated) {
        std::string reason;
        if (!setSelectedObjectWorldMatrix(editedWorld, reason) && !reason.empty()) {
            sceneStatus_ = reason;
        }
    }
    if (!usingNow && gizmoWasUsing_) {
        const scene::ConstSceneObject editedObject =
            gizmoEditingSceneLifetime_ == scene_.sceneGraph().lifetimeRevision()
            ? scene_.sceneGraph().object(gizmoEditingObject_)
            : scene::ConstSceneObject{};
        if (const scene::TransformComponent* editedTransform =
                editedObject.tryGetComponent<scene::TransformComponent>()) {
            pushTransformCommand(
                gizmoEditingObject_,
                gizmoEditingSceneLifetime_,
                gizmoStartLocalMatrix_,
                editedTransform->localMatrix);
        }
        gizmoEditingObject_ = scene::kNullSceneEntity;
        gizmoEditingSceneLifetime_ = 0;
    }
    gizmoWasUsing_ = usingNow;
}

void EditorApplication::drawViewportObjectHandles(const ImVec2& min, const ImVec2& max)
{
    viewportHoveredObject_ = scene::kNullSceneEntity;
    if (!viewportInteractionEnabled_ || !scene_.valid()) {
        return;
    }
    render::RenderGraphNode* renderNode = viewportCameraRenderGraphNode();
    if (renderNode == nullptr) {
        return;
    }

    render::RenderGraphProperties properties = effectiveNodeProperties(*renderNode);
    ensureCameraProperties(properties, scene_.bounds());
    const ViewportCameraMatrices matrices = viewportCameraMatrices(
        properties["camera"],
        max.x - min.x,
        max.y - min.y);
    const scene::ConstSceneObject selectedObject = selectedSceneObject();
    const ImVec2 mouse = ImGui::GetIO().MousePos;
    const float radius = 9.0f * mainScale_;
    float nearestDistanceSquared = radius * radius;
    ImDrawList* drawList = ImGui::GetWindowDrawList();

    const auto drawHandle = [&](scene::SceneEntity entity, ImU32 color, const char* label) {
        const scene::ConstSceneObject object = scene_.sceneGraph().object(entity);
        if (!object || object.hasComponent<scene::GeneratedComponent>() ||
            !object.hasComponent<scene::ActiveSceneComponent>()) {
            return;
        }
        const scene::TransformComponent* transform =
            object.tryGetComponent<scene::TransformComponent>();
        if (transform == nullptr) {
            return;
        }
        ImVec2 screenPosition;
        if (!projectWorldToViewport(
                float3(
                    transform->worldMatrix.a03,
                    transform->worldMatrix.a13,
                    transform->worldMatrix.a23),
                matrices,
                min,
                max,
                screenPosition)) {
            return;
        }

        const float deltaX = mouse.x - screenPosition.x;
        const float deltaY = mouse.y - screenPosition.y;
        const float distanceSquared = deltaX * deltaX + deltaY * deltaY;
        const bool hovered = viewportHovered_ && distanceSquared <= radius * radius;
        if (hovered && distanceSquared <= nearestDistanceSquared) {
            nearestDistanceSquared = distanceSquared;
            viewportHoveredObject_ = entity;
        }

        const bool selected = selectedObject && selectedObject.entity() == entity;
        drawList->AddCircleFilled(
            screenPosition,
            radius,
            hovered ? IM_COL32(255, 225, 110, 245) : color,
            20);
        drawList->AddCircle(
            screenPosition,
            radius + (selected ? 3.0f : 1.0f) * mainScale_,
            selected ? IM_COL32(255, 210, 70, 255) : IM_COL32(20, 22, 28, 230),
            20,
            selected ? 2.0f * mainScale_ : 1.0f * mainScale_);
        const ImVec2 textSize = ImGui::CalcTextSize(label);
        drawList->AddText(
            ImVec2(screenPosition.x - textSize.x * 0.5f, screenPosition.y - textSize.y * 0.5f),
            IM_COL32(20, 22, 28, 255),
            label);
    };

    for (const scene::RenderLight& light : scene_.lights()) {
        if (light.visible) {
            drawHandle(light.object, IM_COL32(255, 196, 72, 230), "L");
        }
    }
    for (const scene::RenderCamera& camera : scene_.cameras()) {
        if (!camera.fallback && camera.visible) {
            drawHandle(camera.object, IM_COL32(92, 184, 255, 230), "C");
        }
    }
}

void EditorApplication::selectViewportObject(const ImVec2& min, const ImVec2& max)
{
    if (!viewportInteractionEnabled_ || !viewportHovered_ || !scene_.valid() ||
        !ImGui::IsMouseClicked(ImGuiMouseButton_Left) || ImGui::GetIO().KeyAlt ||
        viewportGizmoCapturingMouse_ || gizmoWasUsing_ || ImGuizmo::IsUsingAny()) {
        return;
    }
    const ImVec2 mouse = ImGui::GetIO().MousePos;

    if (viewportHoveredObject_ != scene::kNullSceneEntity) {
        const scene::ConstSceneObject object =
            scene_.sceneGraph().object(viewportHoveredObject_);
        if (!object || !object.hasComponent<scene::ActiveSceneComponent>()) {
            viewportHoveredObject_ = scene::kNullSceneEntity;
            return;
        }
        SceneSelection selection{
            .type = SceneSelectionType::Node,
            .object = viewportHoveredObject_,
            .sceneLifetimeRevision = scene_.sceneGraph().lifetimeRevision(),
        };
        if (const scene::SourceNodeComponent* source =
                object.tryGetComponent<scene::SourceNodeComponent>()) {
            selection.index = source->nodeIndex;
            selection.nodeIndex = source->nodeIndex;
        }
        if (const scene::MeshComponent* mesh = object.tryGetComponent<scene::MeshComponent>()) {
            selection.meshIndex = mesh->meshIndex;
        }
        if (const scene::LightComponent* light = object.tryGetComponent<scene::LightComponent>();
            light != nullptr && light->renderLightIndex >= 0 &&
            static_cast<size_t>(light->renderLightIndex) < scene_.lights().size()) {
            selection.type = SceneSelectionType::Light;
            selection.index = light->renderLightIndex;
        } else if (const scene::CameraComponent* camera =
                       object.tryGetComponent<scene::CameraComponent>();
            camera != nullptr && camera->renderCameraIndex >= 0 &&
            static_cast<size_t>(camera->renderCameraIndex) < scene_.cameras().size()) {
            selection.type = SceneSelectionType::Camera;
            selection.index = camera->renderCameraIndex;
        }
        sceneSelection_ = selection;
        return;
    }

    render::RenderGraphNode* renderNode = viewportCameraRenderGraphNode();
    if (renderNode == nullptr) {
        return;
    }
    render::RenderGraphProperties properties = effectiveNodeProperties(*renderNode);
    ensureCameraProperties(properties, scene_.bounds());
    const render::RenderGraphProperties& camera = properties["camera"];
    const ViewportCameraMatrices matrices = viewportCameraMatrices(
        camera,
        max.x - min.x,
        max.y - min.y);
    const float normalizedX = ((mouse.x - min.x) / std::max(max.x - min.x, 1.0f)) * 2.0f - 1.0f;
    const float normalizedY = 1.0f - ((mouse.y - min.y) / std::max(max.y - min.y, 1.0f)) * 2.0f;
    const float aspect = std::max((max.x - min.x) / std::max(max.y - min.y, 1.0f), 0.001f);
    float3 rayOrigin = matrices.frame.eye;
    float3 rayDirection = matrices.frame.forward;
    if (matrices.orthographic) {
        const float height = cameraOrthoHeight(camera, matrices.frame.distance);
        rayOrigin += matrices.frame.right * (normalizedX * height * aspect * 0.5f) +
            matrices.frame.viewUp * (normalizedY * height * 0.5f);
    } else {
        constexpr float kPi = 3.14159265358979323846f;
        const float tangent = std::tan(
            std::clamp(propertyFloatOr(camera, "fovDegrees", 60.0f), 1.0f, 179.0f) *
            (kPi / 360.0f));
        rayDirection = normalizedOr(
            matrices.frame.forward +
                matrices.frame.right * (normalizedX * tangent * aspect) +
                matrices.frame.viewUp * (normalizedY * tangent),
            matrices.frame.forward);
    }
    const float viewDepthAtOrigin = dot(
        rayOrigin - matrices.frame.eye,
        matrices.frame.forward);
    const float viewDepthPerDistance = dot(rayDirection, matrices.frame.forward);
    if (!std::isfinite(viewDepthPerDistance) || viewDepthPerDistance <= 0.000001f) {
        sceneSelection_ = SceneSelection{};
        return;
    }
    const float minimumPickDistance = std::max(
        (matrices.nearPlane - viewDepthAtOrigin) / viewDepthPerDistance,
        0.0f);
    const float maximumPickDistance =
        (matrices.farPlane - viewDepthAtOrigin) / viewDepthPerDistance;
    if (!std::isfinite(minimumPickDistance) || !std::isfinite(maximumPickDistance) ||
        maximumPickDistance < minimumPickDistance) {
        sceneSelection_ = SceneSelection{};
        return;
    }

    const scene::ScenePickResult pick = scenePicker_.pick(
        scene_,
        scene::ScenePickRay{
            .origin = rayOrigin,
            .direction = rayDirection,
            .minimumDistance = minimumPickDistance,
            .maximumDistance = maximumPickDistance,
        });
    if (!pick.hit()) {
        sceneSelection_ = SceneSelection{};
        return;
    }
    const scene::ConstSceneObject object = scene_.sceneGraph().object(pick.object);
    if (!object || !object.hasComponent<scene::ActiveSceneComponent>()) {
        sceneSelection_ = SceneSelection{};
        return;
    }
    SceneSelection selection{
        .type = SceneSelectionType::Node,
        .object = pick.object,
        .sceneLifetimeRevision = scene_.sceneGraph().lifetimeRevision(),
        .index = pick.nodeIndex,
        .nodeIndex = pick.nodeIndex,
    };
    if (const scene::MeshComponent* mesh = object.tryGetComponent<scene::MeshComponent>()) {
        selection.meshIndex = mesh->meshIndex;
    } else if (pick.renderPrimitiveIndex >= 0 &&
        static_cast<size_t>(pick.renderPrimitiveIndex) < scene_.renderPrimitives().size()) {
        selection.meshIndex = scene_.renderPrimitives()[
            static_cast<size_t>(pick.renderPrimitiveIndex)].meshIndex;
    }
    sceneSelection_ = selection;
}

void EditorApplication::drawViewportPanel()
{
    ImGui::Begin("Viewport");

    const bool gizmoTransactionActive = inspectorTransformEditing_ ||
        inspectorPropertyEditing_ || gizmoWasUsing_ ||
        ImGuizmo::IsUsingAny();
    ImGui::BeginDisabled(gizmoTransactionActive);
    const auto drawToolButton = [this](const char* label, GizmoOperation operation) {
        const bool selected = gizmoOperation_ == operation;
        if (selected) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
        }
        if (ImGui::Button(label)) {
            gizmoOperation_ = operation;
        }
        if (selected) {
            ImGui::PopStyleColor();
        }
    };
    drawToolButton("W Move", GizmoOperation::Translate);
    ImGui::SameLine();
    drawToolButton("E Rotate", GizmoOperation::Rotate);
    ImGui::SameLine();
    drawToolButton("R Scale", GizmoOperation::Scale);
    ImGui::SameLine();
    if (ImGui::Button(gizmoLocal_ ? "Local" : "World")) {
        gizmoLocal_ = !gizmoLocal_;
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::Checkbox("Snap", &snapEnabled_);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(72.0f * mainScale_);
    float* snapValue = gizmoOperation_ == GizmoOperation::Translate
        ? &translateSnap_
        : (gizmoOperation_ == GizmoOperation::Rotate ? &rotateSnap_ : &scaleSnap_);
    ImGui::DragFloat("##SnapStep", snapValue, 0.05f, 0.001f, 1000.0f, "%.3f");

    ImVec2 available = ImGui::GetContentRegionAvail();
    available.x = std::max(available.x, 1.0f);
    available.y = std::max(available.y, 1.0f);
    const ImVec2 panelMin = ImGui::GetCursorScreenPos();
    const ImVec2 panelMax(panelMin.x + available.x, panelMin.y + available.y);
    const float panelWidth = panelMax.x - panelMin.x;
    const float panelHeight = panelMax.y - panelMin.y;
    constexpr uint32_t kSmokeTestPreviewSize = 256;
    const auto [previewWidth, previewHeight] = constrainedPreviewExtent(
        panelWidth,
        panelHeight,
        smokeTest_ ? kSmokeTestPreviewSize : kMaxViewportPreviewSize);
    const bool hasRhiPreview = updateViewportPreview(previewWidth, previewHeight);
    const uint32_t displayWidth = hasRhiPreview && viewportTextureWidth_ > 0
        ? viewportTextureWidth_
        : previewWidth;
    const uint32_t displayHeight = hasRhiPreview && viewportTextureHeight_ > 0
        ? viewportTextureHeight_
        : previewHeight;
    const float previewAspect = static_cast<float>(displayWidth) /
        static_cast<float>(std::max(displayHeight, 1u));
    float imageWidth = panelWidth;
    float imageHeight = imageWidth / previewAspect;
    if (imageHeight > panelHeight) {
        imageHeight = panelHeight;
        imageWidth = imageHeight * previewAspect;
    }
    const ImVec2 min(
        panelMin.x + (panelWidth - imageWidth) * 0.5f,
        panelMin.y + (panelHeight - imageHeight) * 0.5f);
    const ImVec2 max(min.x + imageWidth, min.y + imageHeight);
    const float width = max.x - min.x;
    const float height = max.y - min.y;
    const bool loadingScene = pendingSceneLoad_.valid() || pendingSceneResourcePreparation_;
    const bool previewMatchesRequestedExtent = hasRhiPreview &&
        viewportTextureWidth_ == previewWidth && viewportTextureHeight_ == previewHeight;
    const bool popupOpen = ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopup);
    viewportInteractionEnabled_ = previewMatchesRequestedExtent && !loadingScene && !popupOpen;
    const ImVec2 mouse = ImGui::GetIO().MousePos;
    const bool mouseInsideImage = mouse.x >= min.x && mouse.x < max.x &&
        mouse.y >= min.y && mouse.y < max.y;
    viewportHovered_ = viewportInteractionEnabled_ && mouseInsideImage &&
        ImGui::IsWindowHovered();

    ImDrawList* drawList = ImGui::GetWindowDrawList();
    drawList->PushClipRect(panelMin, panelMax, true);
    drawList->AddRectFilled(panelMin, panelMax, IM_COL32(12, 14, 18, 255));
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
    drawViewportObjectHandles(min, max);
    drawViewportGizmo(min, max);

    if (loadingScene) {
        const scene::SceneLoadProgress progress = pendingSceneResourcePreparation_
            ? pendingSceneResourceProgress_
            : pendingSceneLoad_.progress();
        const float panelWidth = std::min(width * 0.70f, 520.0f * mainScale_);
        const float panelHeight = 72.0f * mainScale_;
        const ImVec2 panelMin(
            min.x + (width - panelWidth) * 0.5f,
            min.y + 24.0f * mainScale_);
        const ImVec2 panelMax(panelMin.x + panelWidth, panelMin.y + panelHeight);
        drawList->AddRectFilled(panelMin, panelMax, IM_COL32(10, 12, 16, 220), 5.0f * mainScale_);
        drawList->AddRect(panelMin, panelMax, IM_COL32(100, 115, 138, 255), 5.0f * mainScale_);
        const std::string label = std::string("Loading scene - ") + scene::sceneLoadPhaseName(progress.phase);
        drawList->AddText(
            ImVec2(panelMin.x + 12.0f * mainScale_, panelMin.y + 10.0f * mainScale_),
            IM_COL32(235, 238, 242, 255),
            label.c_str());
        const ImVec2 barMin(
            panelMin.x + 12.0f * mainScale_,
            panelMin.y + 42.0f * mainScale_);
        const ImVec2 barMax(
            panelMax.x - 12.0f * mainScale_,
            panelMin.y + 58.0f * mainScale_);
        drawList->AddRectFilled(barMin, barMax, IM_COL32(35, 40, 48, 255), 3.0f * mainScale_);
        const ImVec2 fillMax(
            barMin.x + (barMax.x - barMin.x) * scene::clampSceneLoadFraction(progress.fraction),
            barMax.y);
        drawList->AddRectFilled(barMin, fillMax, IM_COL32(71, 140, 255, 255), 3.0f * mainScale_);
    }
    drawList->PopClipRect();

    ImGui::Dummy(available);
    viewportHovered_ = viewportInteractionEnabled_ && mouseInsideImage &&
        ImGui::IsItemHovered();
    const ImGuiIO& io = ImGui::GetIO();
    const bool cameraGestureActive =
        viewportCameraDragButton_ != kNoViewportCameraDragButton ||
        ImGui::IsMouseDown(ImGuiMouseButton_Right) ||
        ImGui::IsMouseDown(ImGuiMouseButton_Middle) ||
        (io.KeyAlt && ImGui::IsMouseDown(ImGuiMouseButton_Left));
    const bool transformEditing = gizmoWasUsing_ || inspectorTransformEditing_ ||
        inspectorPropertyEditing_ ||
        viewportGizmoCapturingMouse_ || ImGuizmo::IsUsingAny();
    if (viewportHovered_ && !io.WantTextInput && !io.KeyCtrl && !io.KeyAlt &&
        !cameraGestureActive && !transformEditing) {
        if (ImGui::IsKeyPressed(ImGuiKey_W, false)) {
            gizmoOperation_ = GizmoOperation::Translate;
        } else if (ImGui::IsKeyPressed(ImGuiKey_E, false)) {
            gizmoOperation_ = GizmoOperation::Rotate;
        } else if (ImGui::IsKeyPressed(ImGuiKey_R, false)) {
            gizmoOperation_ = GizmoOperation::Scale;
        } else if (ImGui::IsKeyPressed(ImGuiKey_X, false)) {
            gizmoLocal_ = !gizmoLocal_;
        }
    }
    selectViewportObject(min, max);
    handleViewportCameraControls(min, max);

    ImGui::End();
}

void EditorApplication::handleViewportCameraControls(const ImVec2& min, const ImVec2& max)
{
    render::RenderGraphNode* node = viewportCameraRenderGraphNode();
    if (node == nullptr || !viewportInteractionEnabled_) {
        viewportCameraDragButton_ = kNoViewportCameraDragButton;
        return;
    }

    const ImVec2 size(max.x - min.x, max.y - min.y);
    ImGuiIO& io = ImGui::GetIO();
    const bool hovered = viewportHovered_;
    const bool gizmoCapturingMouse = viewportGizmoCapturingMouse_ || gizmoWasUsing_ ||
        ImGuizmo::IsUsingAny();
    const bool alt = ImGui::IsKeyDown(ImGuiKey_LeftAlt) || ImGui::IsKeyDown(ImGuiKey_RightAlt);

    if (hovered && !gizmoCapturingMouse) {
        if (alt && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
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

    if (!gizmoCapturingMouse && io.MouseWheel != 0.0f) {
        if (viewportCameraDragButton_ == ImGuiMouseButton_Right) {
            viewportCameraSpeed_ = std::clamp(
                viewportCameraSpeed_ * std::pow(kViewportCameraWheelSpeedStep, io.MouseWheel),
                kMinViewportCameraSpeed,
                kMaxViewportCameraSpeed);
            ImGui::SetTooltip("Camera speed: %.2fx", viewportCameraSpeed_);
        } else if (hovered) {
            changed = dollyCamera(io.MouseWheel, camera) || changed;
        }
    }

    if (viewportCameraDragButton_ == ImGuiMouseButton_Right && !io.WantTextInput) {
        if (!alt) {
            const bool shift = ImGui::IsKeyDown(ImGuiKey_LeftShift) || ImGui::IsKeyDown(ImGuiKey_RightShift);
            const bool ctrl = ImGui::IsKeyDown(ImGuiKey_LeftCtrl) || ImGui::IsKeyDown(ImGuiKey_RightCtrl);
            float speedMultiplier = viewportCameraSpeed_;
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
                float eye[3] = {};
                float center[3] = {};
                float up[3] = {};
                readVec3Property(camera, "eye", eye);
                readVec3Property(camera, "center", center);
                readVec3Property(camera, "up", up);
                if (orbitCamera(delta.x, delta.y, size.x, size.y, center, eye, up)) {
                    storeVec3Property(camera, "center", center);
                    storeVec3Property(camera, "up", up);
                    changed = true;
                }
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
    if ((pendingSceneLoad_.valid() || pendingSceneResourcePreparation_) &&
        !scene_.valid() && !viewportPreviewValid_) {
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

    if (smokeTest_) {
        spdlog::info("[Smoke] Begin editor Vulkan frame");
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
    if (smokeTest_) {
        spdlog::info("[Smoke] Acquired swapchain image {}", imageIndex);
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
    if (smokeTest_) {
        spdlog::info("[Smoke] Recorded RenderGraph preview");
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
    if (smokeTest_) {
        spdlog::info("[Smoke] Submitted editor frame");
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

    if (smokeTest_) {
        spdlog::info("[Smoke] Presented editor frame");
    }

    advanceNsightGraphicsCaptureAfterPresent();

    return true;
}

void EditorApplication::loadBuiltInSample(const char* sampleId)
{
    if (scene_.dirty()) {
        requestPendingSceneAction(
            PendingSceneAction::LoadSample,
            {},
            sampleId != nullptr ? sampleId : kDefaultRenderSampleId);
        return;
    }
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

    if (!environmentUserEdited_) {
        if (sample.desc.environment.has_value()) {
            const render::RenderSampleEnvironmentDesc& sourceEnvironment =
                *sample.desc.environment;
            environmentFromSample_ = true;
            render::EnvironmentSettings environment{
                .enabled = sourceEnvironment.enabled,
                .path = sourceEnvironment.path,
                .intensity = sourceEnvironment.intensity,
                .rotationDegrees = sourceEnvironment.rotationDegrees,
                .visible = sourceEnvironment.visible,
            };
            if (!environment.path.empty() && environment.path.is_relative()) {
                environment.path = std::filesystem::path(PROJECT_SOURCE_DIR) / environment.path;
            }
            renderWorld_.setEnvironment(std::move(environment));
        } else {
            environmentFromSample_ = false;
        }
    }
    preserveSampleEnvironmentForNextSceneLoad_ = environmentFromSample_;

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

    clearSceneAccelerationStructure();
    scene_.clear();
    renderWorld_.notifySceneChanged();
    resetTransformHistory();
    sceneSelection_ = SceneSelection{};
    sceneStatus_ = "StreamAsset-only sample: editor scene loading skipped for " + sample.desc.scenePath;
    spdlog::info(
        "[Startup] Skipped editor scene and static RTAS loading for StreamAsset-only sample '{}'",
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
    if (scene_.dirty()) {
        requestPendingSceneAction(
            PendingSceneAction::LoadRenderGraph,
            resolveGraphAssetPath(graphFilePath_));
        return;
    }
    StartupLogScope scope(std::string("Load render graph '") + graphFilePath_ + "'");

    render::RenderGraph loadedGraph;
    std::string message;
    const std::filesystem::path path = resolveGraphAssetPath(graphFilePath_);
    if (!render::loadRenderGraphFromFile(path, loadedGraph, message)) {
        renderGraphStatus_ = message;
        spdlog::warn("[Startup] Render graph load failed: {}", message);
        return;
    }
    environmentFromSample_ = false;
    if (!environmentUserEdited_) {
        if (scene_.hasEnvironmentSettings()) {
            renderWorld_.setEnvironment(scene_.environment());
        } else {
            renderWorld_.setEnvironment(render::EnvironmentSettings{});
        }
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
    requestPendingSceneAction(PendingSceneAction::LoadScene, path);
}

void EditorApplication::loadDroppedRenderGraph(const std::filesystem::path& path)
{
    requestPendingSceneAction(PendingSceneAction::LoadRenderGraph, path);
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
    const std::filesystem::path currentEnvironmentPath = renderWorld_.environment().path;
    std::filesystem::path initialPath = currentEnvironmentPath.empty()
        ? std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset"
        : currentEnvironmentPath;
    std::string dialogError;
    const std::filesystem::path selectedPath = openEnvironmentFileDialog(window_, initialPath, dialogError);
    if (selectedPath.empty()) {
        if (!dialogError.empty()) {
            renderGraphStatus_ = dialogError;
        }
        return;
    }

    setEnvironmentPath(selectedPath);
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
        render::RenderGraphProperties properties = node->runtimeProperties.is_object()
            ? node->runtimeProperties
            : render::RenderGraphProperties::object();
        if (!properties.contains("path") || !properties["path"].is_string() || properties["path"] != graphScenePath) {
            properties["path"] = graphScenePath;
            changed = renderGraph_.setNodeRuntimeProperties(node->id, std::move(properties)) || changed;
        }
    }

    if (!changed) {
        return;
    }

    historyResources_.invalidateAll();
    if (graphExecutor_ != nullptr && graphExecutor_->compiled()) {
        graphExecutor_->syncRuntimeProperties(renderGraph_);
    }
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
        if (!camera.fallback && camera.visible) {
            selectedCamera = &camera;
            break;
        }
    }
    if (selectedCamera == nullptr) {
        for (const scene::RenderCamera& camera : scene_.cameras()) {
            if (camera.fallback || camera.visible) {
                selectedCamera = &camera;
                break;
            }
        }
        if (selectedCamera == nullptr) {
            return;
        }
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

void EditorApplication::setEnvironmentPath(const std::filesystem::path& path)
{
    if (path.empty()) {
        renderGraphStatus_ = "Environment path is empty.";
        return;
    }

    render::EnvironmentSettings environment = renderWorld_.environment();
    environment.enabled = true;
    environment.path = path;
    beginEnvironmentEdit();
    renderWorld_.setEnvironment(environment);
    if (scene_.valid()) {
        if (scene_.setEnvironment(environment)) {
            sceneNonTransformDirty_ = true;
            updateSceneDirtyState();
        }
    }
    environmentUserEdited_ = true;
    environmentFromSample_ = false;
    preserveSampleEnvironmentForNextSceneLoad_ = false;
    viewportPreviewNeedsRender_ = true;
    renderGraphStatus_ = "Loaded environment: " + displayPathForProperty(path);
}

void EditorApplication::loadScene()
{
    StartupLogScope scope(std::string("Editor scene load '") + sceneFilePath_ + "'");

    const std::filesystem::path path = resolveSceneAssetPath(sceneFilePath_);
    if (scene_.dirty()) {
        requestPendingSceneAction(PendingSceneAction::LoadScene, path);
        return;
    }

    if (preserveSampleEnvironmentForNextSceneLoad_) {
        preserveSampleEnvironmentForNextSceneLoad_ = false;
    } else {
        environmentFromSample_ = false;
    }

    if (path.empty()) {
        sceneStatus_ = "Scene path is empty.";
        return;
    }

    cancelSceneLoad();
    ++sceneLoadGeneration_;
    pendingSceneLoadPath_ = path;
    pendingSceneLoad_ = sceneLoader_.request(path);
    if (!pendingSceneLoad_.valid()) {
        sceneStatus_ = "Failed to schedule scene load.";
        return;
    }
    sceneStatus_ = "Loading scene: " + path.string();
    spdlog::info(
        "[Startup] Scheduled asynchronous editor scene load generation={} path='{}'",
        sceneLoadGeneration_,
        path.string());
}

void EditorApplication::pollSceneLoad()
{
    if (pendingSceneResourcePreparation_) {
        bool resourcesComplete = false;
        std::string log;
        const render::Result result = graphExecutor_->pumpSceneResourcePreparation(
            *readySceneLoad_,
            2.0,
            resourcesComplete,
            pendingSceneResourceProgress_,
            log);
        if (!result) {
            pendingSceneResourcePreparation_ = false;
            graphExecutor_->cancelSceneResourcePreparation();
            readySceneLoad_.reset();
            pendingSceneLoadPath_.clear();
            sceneStatus_ = log.empty()
                ? "Failed to prepare scene GPU resources."
                : "Failed to prepare scene GPU resources: " + log;
            return;
        }
        if (!resourcesComplete) {
            return;
        }
        pendingSceneResourcePreparation_ = false;
        pendingSceneResourceProgress_.phase = scene::SceneLoadPhase::Finalizing;
        pendingSceneResourceProgress_.fraction = 0.97f;
        finishActiveTransformTransactions();
        if (scene_.dirty()) {
            pendingSceneAction_ = PendingSceneAction::CommitLoadedScene;
            sceneStatus_ = "New scene is ready; resolve unsaved changes before switching.";
            return;
        }
        commitLoadedScene(std::move(readySceneLoad_));
        return;
    }

    if (!pendingSceneLoad_.valid() || !pendingSceneLoad_.complete()) {
        return;
    }

    const scene::SceneLoadProgress progress = pendingSceneLoad_.progress();
    if (progress.status == scene::SceneLoadStatus::Cancelled) {
        sceneStatus_ = "Scene load cancelled.";
        pendingSceneLoad_ = {};
        pendingSceneLoadPath_.clear();
        return;
    }
    if (progress.status == scene::SceneLoadStatus::Failed) {
        sceneStatus_ = progress.error.empty()
            ? "Failed to load scene."
            : "Failed to load scene: " + progress.error;
        spdlog::warn("[Startup] Scene load failed: {}", sceneStatus_);
        pendingSceneLoad_ = {};
        pendingSceneLoadPath_.clear();
        return;
    }
    if (progress.status != scene::SceneLoadStatus::Succeeded) {
        return;
    }

    readySceneLoad_ = pendingSceneLoad_.takeResult();
    if (readySceneLoad_ == nullptr) {
        sceneStatus_ = "Scene load completed without a scene result.";
        pendingSceneLoad_ = {};
        pendingSceneLoadPath_.clear();
        return;
    }
    pendingSceneLoad_ = {};

    if (device_ != nullptr &&
        graphExecutor_ != nullptr &&
        device_->capabilities().rayTracingAccelerationStructure &&
        device_->capabilities().rayQuery) {
        render::RenderGraphProperties properties = render::RenderGraphProperties::object();
        properties["path"] = displayPathForProperty(readySceneLoad_->sourcePath());
        std::string log;
        const render::Result result = graphExecutor_->beginSceneResourcePreparation(
            *device_,
            properties,
            *readySceneLoad_,
            log);
        if (!result) {
            readySceneLoad_.reset();
            pendingSceneLoadPath_.clear();
            sceneStatus_ = log.empty()
                ? "Failed to begin scene GPU resource preparation."
                : "Failed to begin scene GPU resource preparation: " + log;
            return;
        }
        pendingSceneResourcePreparation_ = true;
        pendingSceneResourceProgress_ = progress;
        pendingSceneResourceProgress_.status = scene::SceneLoadStatus::Running;
        pendingSceneResourceProgress_.phase = scene::SceneLoadPhase::GpuUpload;
        pendingSceneResourceProgress_.fraction = 0.65f;
        sceneStatus_ = "Preparing scene GPU resources while the current scene remains active.";
        return;
    }
    finishActiveTransformTransactions();
    if (scene_.dirty()) {
        pendingSceneAction_ = PendingSceneAction::CommitLoadedScene;
        sceneStatus_ = "New scene is ready; resolve unsaved changes before switching.";
        return;
    }
    commitLoadedScene(std::move(readySceneLoad_));
}

void EditorApplication::cancelSceneLoad()
{
    if (pendingSceneLoad_.valid() && !pendingSceneLoad_.complete()) {
        (void)pendingSceneLoad_.cancel();
    }
    pendingSceneLoad_ = {};
    if (pendingSceneResourcePreparation_ && graphExecutor_ != nullptr) {
        graphExecutor_->cancelSceneResourcePreparation();
    }
    pendingSceneResourcePreparation_ = false;
    pendingSceneResourceProgress_ = {};
    readySceneLoad_.reset();
    pendingSceneLoadPath_.clear();
    if (pendingSceneAction_ == PendingSceneAction::CommitLoadedScene) {
        pendingSceneAction_ = PendingSceneAction::None;
    }
}

bool EditorApplication::waitForPendingSceneLoad(uint32_t timeoutMilliseconds)
{
    if (!pendingSceneLoad_.valid() && !pendingSceneResourcePreparation_) {
        return true;
    }
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::milliseconds(timeoutMilliseconds);
    while ((pendingSceneLoad_.valid() || pendingSceneResourcePreparation_) &&
           std::chrono::steady_clock::now() < deadline) {
        pollSceneLoad();
        if (!pendingSceneLoad_.valid() && !pendingSceneResourcePreparation_) {
            return scene_.valid();
        }
        SDL_Delay(1);
    }
    return !pendingSceneLoad_.valid() && !pendingSceneResourcePreparation_ && scene_.valid();
}

void EditorApplication::commitLoadedScene(std::unique_ptr<scene::SceneDocument> loadedScene)
{
    if (loadedScene == nullptr || !loadedScene->valid()) {
        sceneStatus_ = "Cannot commit an invalid loaded scene.";
        return;
    }

    clearSceneAccelerationStructure();
    if (graphExecutor_ != nullptr) {
        graphExecutor_->acceptSceneResourcePreparation();
    }
    historyResources_.invalidateAll();
    scene_ = std::move(*loadedScene);
    if (renderWorld_.scene() == &scene_) {
        renderWorld_.notifySceneChanged();
    } else {
        renderWorld_.setScene(&scene_);
    }
    if (graphExecutor_ != nullptr) {
        graphExecutor_->bindRuntimeScene(&scene_);
    }
    if (!environmentUserEdited_ && !environmentFromSample_) {
        if (scene_.hasEnvironmentSettings()) {
            renderWorld_.setEnvironment(scene_.environment());
        } else {
            renderWorld_.setEnvironment(render::EnvironmentSettings{});
        }
    }
    pendingSceneLoad_ = {};
    pendingSceneResourcePreparation_ = false;
    pendingSceneResourceProgress_ = {};
    pendingSceneLoadPath_.clear();

    sceneSelection_ = SceneSelection{};
    resetTransformHistory();
    const std::filesystem::path sourcePath = scene_.sourcePath();
    copyToBuffer(displayPathForProperty(sourcePath), sceneFilePath_, sizeof(sceneFilePath_));
    addRecentScenePath(sourcePath);
    // A same-path reload still replaces the runtime scene identity and may change
    // resident geometry. Force the preview passes to compile against the new scene
    // even when no RenderGraph property needs to change.
    viewportPreviewValid_ = false;
    viewportPreviewNeedsRender_ = true;
    applyLoadedSceneToRenderGraph(sourcePath);
    applyLoadedSceneCamera();
    sceneAccelerationStructureStatus_ = "RTAS will be prepared by the active render pass.";

    const scene::SceneStats& stats = scene_.stats();
    sceneStatus_ = "Loaded " + sourcePath.string() + " (" + std::to_string(stats.renderNodeCount) +
        " render nodes, " + std::to_string(scene_.nodes().size()) + " scene nodes).";
    if (!scene_.documentWarning().empty()) {
        sceneStatus_ += " Warning: " + scene_.documentWarning();
    }
    spdlog::info(
        "[Startup] Editor scene loaded nodes={} renderNodes={} primitives={} triangles={} images={} textures={}",
        scene_.nodes().size(),
        stats.renderNodeCount,
        stats.primitiveCount,
        stats.triangleCount,
        stats.imageCount,
        stats.textureCount);
}

void EditorApplication::buildSceneAccelerationStructure()
{
    StartupLogScope scope("Editor ray tracing acceleration structure build");

    if (sceneAccelerationStructure_ == nullptr) {
        sceneAccelerationStructure_ =
            std::make_unique<render::SceneAccelerationStructureBuilder>();
    }
    if (device_ == nullptr || graphicsQueue_ == nullptr) {
        sceneAccelerationStructureStatus_ = "RTAS build failed: RHI device is not initialized.";
        return;
    }
    if (!scene_.valid()) {
        sceneAccelerationStructureStatus_ = "RTAS build failed: load a glTF scene first.";
        return;
    }

    std::string log;
    const render::Result result =
        sceneAccelerationStructure_->build(*device_, *graphicsQueue_, scene_, log);
    sceneAccelerationStructureStatus_ = log.empty()
        ? std::string("RTAS build returned ") + render::resultToString(result)
        : log;
    if (result) {
        const render::SceneAccelerationStructureStats& stats =
            sceneAccelerationStructure_->stats();
        spdlog::info(
            "[Startup] Editor RTAS build completed blas={} instances={} triangles={} asBytes={} scratchBytes={}",
            stats.blasCount,
            stats.instanceCount,
            stats.triangleCount,
            stats.accelerationStructureBytes,
            stats.scratchBytes);
    } else {
        spdlog::warn(
            "[Startup] Editor RTAS build failed: {}",
            sceneAccelerationStructureStatus_);
    }
}

void EditorApplication::clearSceneAccelerationStructure()
{
    if (sceneAccelerationStructure_ != nullptr) {
        sceneAccelerationStructure_->clear();
    }
    sceneAccelerationStructureStatus_ = "RTAS not built.";
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
