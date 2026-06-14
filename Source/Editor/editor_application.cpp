#include "Editor/editor_application.h"

#include "Runtime/Render/GAPI/rhi.h"
#include "Runtime/Render/GAPI/Vulkan/vulkan_native.h"
#include "Runtime/Render/RenderGraph/render_graph.h"
#include "imnodes.h"
#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_vulkan.h"
#include "imgui_internal.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_map>
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
                {"eye", {-0.0168404f, 0.110154f, 0.22f}},
                {"center", {-0.0168404f, 0.110154f, -0.00153695f}},
                {"up", {0.0f, 1.0f, 0.0f}},
            }},
        };
    }
    return render::RenderGraphProperties::object();
}

render::RenderGraphNode* findBunnyWireframeNode(render::RenderGraph& graph)
{
    for (const render::RenderGraphNode& node : graph.nodes()) {
        if (node.type == "BunnyWireframePass") {
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

void readVec3Property(const render::RenderGraphProperties& object, const char* key, float outValues[3])
{
    const render::RenderGraphProperties& value = object.at(key);
    outValues[0] = value[0].get<float>();
    outValues[1] = value[1].get<float>();
    outValues[2] = value[2].get<float>();
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
    case render::Format::Bgra8Unorm:
        return "Bgra8Unorm";
    case render::Format::Bgra8Srgb:
        return "Bgra8Srgb";
    case render::Format::Rgba8Unorm:
        return "Rgba8Unorm";
    case render::Format::Rgba8Srgb:
        return "Rgba8Srgb";
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
        std::cerr << "Vulkan error: " << static_cast<int>(result) << '\n';
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

int EditorApplication::run(bool smokeTest, bool waitForGraphicsDebugger)
{
    smokeTest_ = smokeTest;
    waitForGraphicsDebugger_ = waitForGraphicsDebugger && !smokeTest;

    if (!initialize()) {
        shutdown();
        return 1;
    }

    if (smokeTest) {
        pollEvents();
        renderFrame();
        shutdown();
        return 0;
    }

    while (running_) {
        pollEvents();

        if ((SDL_GetWindowFlags(window_) & SDL_WINDOW_MINIMIZED) != 0) {
            SDL_Delay(10);
            continue;
        }

        renderFrame();
    }

    shutdown();
    return 0;
}

bool EditorApplication::initialize()
{
    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD)) {
        SDL_Log("SDL_Init failed: %s", SDL_GetError());
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
        SDL_Log("SDL_CreateWindow failed: %s", SDL_GetError());
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

    applyNvproImGuiStyle();
    ImGuiStyle& style = ImGui::GetStyle();
    style.ScaleAllSizes(mainScale_);
    style.FontScaleDpi = mainScale_;

    ImNodes::CreateContext();
    imnodesContextCreated_ = true;
    applyNvproImNodesStyle();

    if (!initializeImGuiBackends()) {
        return false;
    }

    graphExecutor_ = std::make_unique<render::RenderGraphExecutor>();
    resetDefaultRenderGraph();
    loadScene();

    return true;
}

bool EditorApplication::initializeRhi()
{
    render::Result result = render::createDevice(
        render::DeviceDesc{
            .applicationName = "Metallic Engine Editor",
            .enableValidation = false,
            .enableBindlessDescriptorHeap = true,
        },
        device_);
    if (!result || device_ == nullptr) {
        std::cerr << "createDevice failed with Result " << render::resultToString(result) << '\n';
        return false;
    }

    graphicsQueue_ = device_->getQueue(render::QueueType::Graphics);
    if (graphicsQueue_ == nullptr) {
        std::cerr << "RHI graphics queue is not available\n";
        return false;
    }

    result = device_->createCommandPool(*graphicsQueue_, commandPool_);
    if (!result) {
        std::cerr << "createCommandPool failed with Result " << render::resultToString(result) << '\n';
        return false;
    }
    result = commandPool_->createCommandBuffer(commandBuffer_);
    if (!result) {
        std::cerr << "createCommandBuffer failed with Result " << render::resultToString(result) << '\n';
        return false;
    }
    result = device_->createFence(true, frameFence_);
    if (!result) {
        std::cerr << "createFence failed with Result " << render::resultToString(result) << '\n';
        return false;
    }
    result = device_->createSemaphore(imageAvailableSemaphore_);
    if (!result) {
        std::cerr << "createSemaphore(imageAvailable) failed with Result " << render::resultToString(result) << '\n';
        return false;
    }
    result = device_->createSemaphore(renderFinishedSemaphore_);
    if (!result) {
        std::cerr << "createSemaphore(renderFinished) failed with Result " << render::resultToString(result) << '\n';
        return false;
    }

    int width = 0;
    int height = 0;
    if (!SDL_GetWindowSizeInPixels(window_, &width, &height)) {
        SDL_Log("SDL_GetWindowSizeInPixels failed: %s", SDL_GetError());
        return false;
    }
    if (!createOrResizeSwapchain(
            static_cast<uint32_t>(std::max(width, 1)),
            static_cast<uint32_t>(std::max(height, 1)))) {
        return false;
    }

    return createViewportSampler();
}

bool EditorApplication::createOrResizeSwapchain(uint32_t width, uint32_t height)
{
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
        std::cerr << "createSwapchain failed with Result " << render::resultToString(result) << '\n';
        return false;
    }

    swapchainImageViews_.reserve(swapchain_->imageCount());
    swapchainImageStates_.assign(swapchain_->imageCount(), render::ResourceState::Undefined);
    for (uint32_t imageIndex = 0; imageIndex < swapchain_->imageCount(); ++imageIndex) {
        render::Texture* texture = swapchain_->texture(imageIndex);
        if (texture == nullptr) {
            std::cerr << "swapchain texture is missing at image " << imageIndex << '\n';
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
            std::cerr << "createTextureView(swapchain) failed with Result " << render::resultToString(result) << '\n';
            return false;
        }
        swapchainImageViews_.push_back(std::move(view));
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
    swapchain_.reset();
    swapchainWidth_ = 0;
    swapchainHeight_ = 0;
}

bool EditorApplication::initializeImGuiBackends()
{
    imguiPlatformInitialized_ = ImGui_ImplSDL3_InitForVulkan(window_);
    if (!imguiPlatformInitialized_) {
        SDL_Log("ImGui SDL3 Vulkan platform backend initialization failed");
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
        std::cerr << "Invalid Vulkan native handles for ImGui backend\n";
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
        SDL_Log("ImGui Vulkan renderer backend initialization failed");
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
        std::cerr << "vkCreateSampler(viewport) failed with VkResult " << static_cast<int>(result) << '\n';
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
    renderFinishedSemaphore_.reset();
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
    }
}

void EditorApplication::renderFrame()
{
    int framebufferWidth = 0;
    int framebufferHeight = 0;
    if (!SDL_GetWindowSizeInPixels(window_, &framebufferWidth, &framebufferHeight)) {
        SDL_Log("SDL_GetWindowSizeInPixels failed: %s", SDL_GetError());
        running_ = false;
        return;
    }
    if (framebufferWidth <= 0 || framebufferHeight <= 0) {
        return;
    }

    if (frameFence_ != nullptr) {
        render::Result result = frameFence_->wait();
        if (!result) {
            std::cerr << "frameFence wait before UI failed with Result " << render::resultToString(result) << '\n';
            running_ = false;
            return;
        }
    }

    if (swapchainOutOfDate_ ||
        swapchain_ == nullptr ||
        swapchainWidth_ != static_cast<uint32_t>(framebufferWidth) ||
        swapchainHeight_ != static_cast<uint32_t>(framebufferHeight)) {
        if (!createOrResizeSwapchain(
                static_cast<uint32_t>(framebufferWidth),
                static_cast<uint32_t>(framebufferHeight))) {
            running_ = false;
            return;
        }
    }

    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();

    drawDockspace();
    drawPanels();

    ImGui::Render();
    if (!renderVulkanFrame()) {
        running_ = false;
    }
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
        0.28f,
        nullptr,
        &viewportDockId);

    ImGui::DockBuilderDockWindow("Viewport", viewportDockId);
    ImGui::DockBuilderDockWindow("Scene", sideDockId);
    ImGui::DockBuilderDockWindow("Assets", bottomDockId);
    ImGui::DockBuilderDockWindow("Console", bottomDockId);
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
            ImGui::MenuItem("Scene");
            ImGui::MenuItem("Viewport");
            ImGui::MenuItem("Assets");
            ImGui::MenuItem("Console");
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
    drawScenePanel();
    drawViewportPanel();
    drawRenderGraphEditorWindow();

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
}

void EditorApplication::drawScenePanel()
{
    ImGui::Begin("Scene");

    ImGui::TextUnformatted("glTF Scene");
    ImGui::PushItemWidth(-1.0f);
    ImGui::InputText("##ScenePath", sceneFilePath_, sizeof(sceneFilePath_));
    ImGui::PopItemWidth();

    if (ImGui::Button("Load")) {
        loadScene();
    }
    ImGui::SameLine();
    if (ImGui::Button("Clear")) {
        scene_.clear();
        sceneStatus_ = "No scene loaded.";
    }

    if (!sceneStatus_.empty()) {
        ImGui::TextWrapped("%s", sceneStatus_.c_str());
    }
    const scene::LoadResult& loadResult = scene_.lastLoadResult();
    if (!loadResult.warning.empty()) {
        ImGui::TextWrapped("Warning: %s", loadResult.warning.c_str());
    }
    if (!loadResult.error.empty() && !loadResult.success) {
        ImGui::TextWrapped("Error: %s", loadResult.error.c_str());
    }

    ImGui::Separator();
    drawCameraControls();

    if (!scene_.valid()) {
        ImGui::Separator();
        ImGui::TextDisabled("Load a .gltf or .glb file to inspect its CPU scene graph.");
        ImGui::End();
        return;
    }

    const scene::SceneStats& stats = scene_.stats();
    ImGui::Separator();
    ImGui::Text("Scene: %s", scene_.sceneName().c_str());
    ImGui::Text("Scene index: %d", scene_.sceneIndex());
    ImGui::Text(
        "Meshes: %llu  Primitives: %llu  Render nodes: %llu",
        static_cast<unsigned long long>(stats.meshCount),
        static_cast<unsigned long long>(stats.primitiveCount),
        static_cast<unsigned long long>(stats.renderNodeCount));
    ImGui::Text(
        "Materials: %llu  Triangles: %llu",
        static_cast<unsigned long long>(stats.materialCount),
        static_cast<unsigned long long>(stats.triangleCount));

    const scene::Bounds& bounds = scene_.bounds();
    if (bounds.valid) {
        ImGui::Text("Bounds min: %s", scene::formatVec3(bounds.min).c_str());
        ImGui::Text("Bounds max: %s", scene::formatVec3(bounds.max).c_str());
    } else {
        ImGui::TextDisabled("Bounds: unavailable");
    }

    if (ImGui::CollapsingHeader("Cameras", ImGuiTreeNodeFlags_DefaultOpen)) {
        int cameraIndex = 0;
        for (const scene::RenderCamera& camera : scene_.cameras()) {
            ImGui::PushID(cameraIndex++);
            const char* suffix = camera.fallback ? " (fallback)" : "";
            ImGui::Text("%s%s", camera.name.c_str(), suffix);
            ImGui::Text("  Type: %s", scene::cameraTypeName(camera.type));
            ImGui::Text("  Eye: %s", scene::formatVec3(camera.eye).c_str());
            ImGui::Text("  Center: %s", scene::formatVec3(camera.center).c_str());
            ImGui::PopID();
        }
    }

    if (ImGui::CollapsingHeader("Lights")) {
        int lightIndex = 0;
        for (const scene::RenderLight& light : scene_.lights()) {
            ImGui::PushID(lightIndex++);
            ImGui::Text("%s  [%s]", light.name.c_str(), light.type.c_str());
            ImGui::Text("  Color: %s", scene::formatVec3(light.color).c_str());
            ImGui::Text("  Intensity: %.3f", light.intensity);
            ImGui::PopID();
        }
        if (scene_.lights().empty()) {
            ImGui::TextDisabled("No punctual lights.");
        }
    }

    if (ImGui::CollapsingHeader("Scene Graph", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (const int32_t rootNodeIndex : scene_.rootNodeIndices()) {
            drawSceneNode(rootNodeIndex);
        }
    }

    ImGui::End();
}

void EditorApplication::drawCameraControls()
{
    render::RenderGraphNode* node = findBunnyWireframeNode(renderGraph_);
    if (node == nullptr) {
        ImGui::TextDisabled("No render camera controls for this graph.");
        return;
    }

    render::RenderGraphProperties properties = node->properties;
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

void EditorApplication::applyBunnyCameraProperties(render::RenderGraphProperties properties, const char* status)
{
    render::RenderGraphNode* node = findBunnyWireframeNode(renderGraph_);
    if (node == nullptr) {
        return;
    }

    node->properties = std::move(properties);
    if (graphExecutor_ != nullptr && !renderGraph_.dirty()) {
        graphExecutor_->syncProperties(renderGraph_);
    }
    viewportPreviewNeedsRender_ = true;
    if (status != nullptr) {
        renderGraphStatus_ = status;
    }
}

void EditorApplication::drawSceneNode(int32_t nodeIndex)
{
    const std::vector<scene::SceneNode>& nodes = scene_.nodes();
    if (nodeIndex < 0 || static_cast<size_t>(nodeIndex) >= nodes.size()) {
        return;
    }

    const scene::SceneNode& node = nodes[static_cast<size_t>(nodeIndex)];
    std::string label = node.name;
    if (node.meshIndex >= 0) {
        label += " [Mesh ";
        label += std::to_string(node.meshIndex);
        label += "]";
    }
    if (node.cameraIndex >= 0) {
        label += " [Camera ";
        label += std::to_string(node.cameraIndex);
        label += "]";
    }
    if (node.lightIndex >= 0) {
        label += " [Light ";
        label += std::to_string(node.lightIndex);
        label += "]";
    }
    if (!node.visible) {
        label += " [Hidden]";
    }

    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;
    const bool leaf = node.children.empty();
    if (leaf) {
        flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
    }

    const bool open = ImGui::TreeNodeEx(
        reinterpret_cast<void*>(static_cast<intptr_t>(nodeIndex)),
        flags,
        "%s",
        label.c_str());
    if (ImGui::IsItemHovered()) {
        const float3 translation(node.worldMatrix.a03, node.worldMatrix.a13, node.worldMatrix.a23);
        ImGui::SetTooltip(
            "Node %d\nParent: %d\nWorld translation: %s",
            nodeIndex,
            node.parent,
            scene::formatVec3(translation).c_str());
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
        viewportCameraDragging_ = false;
        return;
    }

    const bool hovered = ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);
    const ImVec2 size(max.x - min.x, max.y - min.y);
    ImGuiIO& io = ImGui::GetIO();

    if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Middle)) {
        viewportCameraDragging_ = true;
    }
    if (!ImGui::IsMouseDown(ImGuiMouseButton_Middle)) {
        viewportCameraDragging_ = false;
    }

    render::RenderGraphProperties properties = node->properties;
    ensureCameraProperties(properties, scene_.bounds());
    render::RenderGraphProperties& camera = properties["camera"];
    bool changed = false;

    if (hovered && io.MouseWheel != 0.0f) {
        changed = dollyCamera(io.MouseWheel, camera) || changed;
    }

    if (viewportCameraDragging_) {
        const ImVec2 delta = io.MouseDelta;
        if (delta.x != 0.0f || delta.y != 0.0f) {
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

    const bool textureSizeMatches =
        viewportDescriptor_ != VK_NULL_HANDLE &&
        viewportTextureWidth_ == width &&
        viewportTextureHeight_ == height;

    if (viewportPreviewValid_ &&
        textureSizeMatches &&
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

    destroyViewportTexture();

    std::string log;
    render::Result result = graphExecutor_->compile(*device_, renderGraph_, width, height, log);
    renderGraphStatus_ = log;
    if (!result) {
        std::cerr << "RenderGraph compile failed with Result "
                  << render::resultToString(result) << '\n';
        return false;
    }

    render::RenderGraphResource* output = graphExecutor_->outputResource(renderGraph_.firstOutputName());
    if (output == nullptr || output->view == nullptr) {
        renderGraphStatus_ = "RenderGraph output texture is not available";
        return false;
    }

    const VkImageView imageView = render::vulkan::nativeImageView(*output->view);
    if (imageView == VK_NULL_HANDLE) {
        renderGraphStatus_ = "RenderGraph output image view is not available";
        return false;
    }

    viewportDescriptor_ = ImGui_ImplVulkan_AddTexture(
        viewportSampler_,
        imageView,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    if (viewportDescriptor_ == VK_NULL_HANDLE) {
        renderGraphStatus_ = "ImGui failed to allocate viewport descriptor";
        return false;
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

    if (!renderGraph_.dirty()) {
        graphExecutor_->syncProperties(renderGraph_);
    }

    commandBuffer_->beginDebugLabel(render::DebugLabelDesc{
        .name = "RenderGraph Preview",
        .color = render::ColorValue{0.78f, 0.36f, 0.92f, 1.0f},
    });
    render::Result result = graphExecutor_->execute(*commandBuffer_);
    commandBuffer_->endDebugLabel();
    if (!result) {
        renderGraphStatus_ = std::string("RenderGraph execute failed: ") + render::resultToString(result);
        std::cerr << renderGraphStatus_ << '\n';
        viewportPreviewValid_ = false;
        viewportPreviewNeedsRender_ = false;
        return false;
    }

    result = graphExecutor_->transitionOutput(
        *commandBuffer_,
        renderGraph_.firstOutputName(),
        render::ResourceState::ShaderRead);
    if (!result) {
        renderGraphStatus_ = std::string("RenderGraph output transition failed: ") + render::resultToString(result);
        std::cerr << renderGraphStatus_ << '\n';
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
        renderFinishedSemaphore_ == nullptr ||
        graphicsQueue_ == nullptr) {
        return false;
    }

    render::Result result = frameFence_->wait();
    if (!result) {
        std::cerr << "frameFence wait failed with Result " << render::resultToString(result) << '\n';
        return false;
    }

    uint32_t imageIndex = 0;
    result = swapchain_->acquireNextImage(*imageAvailableSemaphore_, imageIndex);
    if (!result) {
        if (render::hasError(result, render::Error::OutOfDate)) {
            swapchainOutOfDate_ = true;
            return true;
        }
        std::cerr << "acquireNextImage failed with Result " << render::resultToString(result) << '\n';
        return false;
    }
    if (imageIndex >= swapchainImageViews_.size() || imageIndex >= swapchainImageStates_.size()) {
        std::cerr << "acquireNextImage returned invalid image index " << imageIndex << '\n';
        return false;
    }

    result = frameFence_->reset();
    if (!result) {
        std::cerr << "frameFence reset failed with Result " << render::resultToString(result) << '\n';
        return false;
    }
    result = commandPool_->reset();
    if (!result) {
        std::cerr << "commandPool reset failed with Result " << render::resultToString(result) << '\n';
        return false;
    }
    result = commandBuffer_->begin();
    if (!result) {
        std::cerr << "commandBuffer begin failed with Result " << render::resultToString(result) << '\n';
        return false;
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

    if (!renderGraphPreview()) {
        endFrameLabel();
        return false;
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

    ImGui_ImplVulkan_RenderDrawData(
        ImGui::GetDrawData(),
        render::vulkan::nativeCommandBuffer(*commandBuffer_));

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
    result = commandBuffer_->end();
    if (!result) {
        std::cerr << "commandBuffer end failed with Result " << render::resultToString(result) << '\n';
        return false;
    }

    render::CommandBuffer* commandBuffers[] = {commandBuffer_.get()};
    render::SemaphoreSubmitDesc waitSemaphore{
        .semaphore = imageAvailableSemaphore_.get(),
        .stages = render::PipelineStageBits::ColorAttachment,
    };
    render::SemaphoreSubmitDesc signalSemaphore{
        .semaphore = renderFinishedSemaphore_.get(),
        .stages = render::PipelineStageBits::AllCommands,
    };
    result = graphicsQueue_->submit(render::QueueSubmitDesc{
        .waitSemaphores = &waitSemaphore,
        .waitSemaphoreCount = 1,
        .commandBuffers = commandBuffers,
        .commandBufferCount = 1,
        .signalSemaphores = &signalSemaphore,
        .signalSemaphoreCount = 1,
        .signalFence = frameFence_.get(),
    });
    if (!result) {
        std::cerr << "graphicsQueue submit failed with Result " << render::resultToString(result) << '\n';
        return false;
    }
    result = swapchain_->present(*graphicsQueue_, imageIndex, *renderFinishedSemaphore_);
    if (!result) {
        if (render::hasError(result, render::Error::OutOfDate)) {
            swapchainOutOfDate_ = true;
            return true;
        }
        std::cerr << "swapchain present failed with Result " << render::resultToString(result) << '\n';
        return false;
    }

    return true;
}

void EditorApplication::resetDefaultRenderGraph()
{
    renderGraph_ = render::RenderGraph::createDefaultBunnyGraph();
    graphEditorPositionsInitialized_ = false;
    selectedGraphNodeId_ = -1;
    selectedGraphLinkId_ = -1;
    viewportPreviewValid_ = false;
    copyToBuffer(renderGraph_.firstOutputName(), graphOutputBuffer_, sizeof(graphOutputBuffer_));
    renderGraphStatus_ = "Created Stanford Bunny wireframe RenderGraph";
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
    render::RenderGraph loadedGraph;
    std::string message;
    const std::filesystem::path path = resolveGraphAssetPath(graphFilePath_);
    if (!render::loadRenderGraphFromFile(path, loadedGraph, message)) {
        renderGraphStatus_ = message;
        return;
    }
    renderGraph_ = std::move(loadedGraph);
    graphEditorPositionsInitialized_ = false;
    selectedGraphNodeId_ = -1;
    selectedGraphLinkId_ = -1;
    viewportPreviewValid_ = false;
    copyToBuffer(renderGraph_.firstOutputName(), graphOutputBuffer_, sizeof(graphOutputBuffer_));
    renderGraphStatus_ = message;
}

void EditorApplication::loadScene()
{
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
        return;
    }

    const scene::SceneStats& stats = scene_.stats();
    sceneStatus_ = "Loaded " + path.string() + " (" + std::to_string(stats.renderNodeCount) +
        " render nodes, " + std::to_string(scene_.nodes().size()) + " scene nodes).";
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
            const int attributeId = graphOutputAttributeId(node, outputIndex++);
            if (markedOutput) {
                ImNodes::PushColorStyle(ImNodesCol_Pin, IM_COL32(231, 65, 65, 255));
                ImNodes::PushColorStyle(ImNodesCol_PinHovered, IM_COL32(255, 112, 112, 255));
            }
            ImNodes::BeginOutputAttribute(
                attributeId,
                markedOutput ? ImNodesPinShape_QuadFilled : ImNodesPinShape_CircleFilled);
            std::string label = field.name;
            label += "  ";
            label += renderGraphFieldTag(field);
            if (markedOutput) {
                label += "  [Graph Output]";
            }
            const float textWidth = ImGui::CalcTextSize(label.c_str()).x;
            ImGui::Indent(std::max(90.0f * mainScale_ - textWidth, 0.0f));
            ImGui::TextUnformatted(label.c_str());
            setRenderGraphFieldTooltip(field);
            ImNodes::EndOutputAttribute();
            if (markedOutput) {
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

    ImGui::Separator();
    ImGui::TextUnformatted("Graph Output");
    ImGui::PushItemWidth(-1.0f);
    ImGui::InputText("##GraphOutput", graphOutputBuffer_, sizeof(graphOutputBuffer_));
    ImGui::PopItemWidth();
    if (ImGui::Button("Add Output")) {
        markRenderGraphOutput(graphOutputBuffer_);
    }

    for (const render::RenderGraphOutput& output : renderGraph_.outputs()) {
        const std::string outputName = render::makeRenderGraphFieldName(output.passName, output.fieldName);
        const bool selected = outputName == renderGraph_.firstOutputName();
        if (ImGui::Selectable(outputName.c_str(), selected)) {
            markRenderGraphOutput(outputName);
        }
    }

    if (ImGui::Button("Open Preview")) {
        renderGraphStatus_ = "Viewport previews the current graph output";
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
    const float cardHeight = 74.0f * mainScale_;
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

            if (hovered && !passInfo.description.empty()) {
                ImGui::SetTooltip("%s", passInfo.description.c_str());
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
        ImGui::Text("Output: %s", renderGraph_.firstOutputName().c_str());
        ImGui::Text("Nodes: %zu", renderGraph_.nodes().size());
        ImGui::Text("Edges: %zu", renderGraph_.edges().size());
        return;
    }

    static int editingNodeId = -1;
    if (editingNodeId != static_cast<int>(node->id)) {
        copyToBuffer(node->name, graphNodeNameBuffer_, sizeof(graphNodeNameBuffer_));
        editingNodeId = static_cast<int>(node->id);
    }

    ImGui::Text("Type: %s", node->type.c_str());
    ImGui::InputText("Name", graphNodeNameBuffer_, sizeof(graphNodeNameBuffer_));
    if (ImGui::IsItemDeactivatedAfterEdit() && std::strlen(graphNodeNameBuffer_) > 0) {
        if (!renderGraph_.renameNode(node->id, graphNodeNameBuffer_)) {
            renderGraphStatus_ = "Node rename failed";
        } else {
            copyToBuffer(renderGraph_.firstOutputName(), graphOutputBuffer_, sizeof(graphOutputBuffer_));
            viewportPreviewValid_ = false;
        }
    }

    if (node->type == "ClearColorPass") {
        render::RenderGraphProperties properties = node->properties;
        if (!properties.contains("color") || !properties["color"].is_array() || properties["color"].size() < 4) {
            properties["color"] = {0.04f, 0.06f, 0.09f, 1.0f};
        }
        float color[4] = {
            properties["color"][0].get<float>(),
            properties["color"][1].get<float>(),
            properties["color"][2].get<float>(),
            properties["color"][3].get<float>(),
        };
        if (ImGui::ColorEdit4("Color", color)) {
            properties["color"] = {color[0], color[1], color[2], color[3]};
            renderGraph_.setNodeProperties(node->id, std::move(properties));
            viewportPreviewValid_ = false;
        }
    } else if (!node->properties.empty()) {
        const std::string propertiesText = node->properties.dump(2);
        ImGui::TextWrapped("%s", propertiesText.c_str());
    }

    std::unique_ptr<render::RenderGraphPass> pass = render::createRenderGraphPass(node->type);
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
