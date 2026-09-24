#include "Editor/EditorApplication.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanStreamline.h"

#include <imgui.h>

namespace metallic {
namespace {

const char* dlssModeName(render::vulkan::StreamlineDlssRrMode mode)
{
    using Mode = render::vulkan::StreamlineDlssRrMode;
    switch (mode) {
    case Mode::Off: return "Off";
    case Mode::Dlaa: return "DLAA";
    case Mode::Quality: return "Quality";
    case Mode::Balanced: return "Balanced";
    case Mode::Performance: return "Performance";
    case Mode::UltraPerformance: return "Ultra Performance";
    case Mode::UltraQuality: return "Ultra Quality";
    }
    return "Unknown";
}

const char* resourceFormatName(render::Format format)
{
    switch (format) {
    case render::Format::Rgba16Sfloat: return "RGBA16_FLOAT";
    case render::Format::Rg16Sfloat: return "RG16_FLOAT";
    case render::Format::R32Sfloat: return "R32_FLOAT";
    case render::Format::D32Sfloat: return "D32_FLOAT";
    default: return "Other / unknown";
    }
}

void drawDlssStatus(const char* title, bool supported, const render::vulkan::StreamlineDlssDebugStatus& status)
{
    if (!ImGui::CollapsingHeader(title, ImGuiTreeNodeFlags_DefaultOpen)) { return; }
    ImGui::PushID(title);
    ImGui::Text("Device support: %s", supported ? "Yes" : "No");
    if (!status.attempts) {
        ImGui::TextDisabled("No evaluation recorded in this device session.");
        ImGui::PopID();
        return;
    }
    ImGui::Text("Last attempt: frame %u, %.2f seconds ago", status.frameIndex, status.ageSeconds);
    ImGui::TextDisabled("Last recorded values; support does not imply the feature is running.");
    ImGui::Text("Attempts: %llu   Successful: %llu",
        static_cast<unsigned long long>(status.attempts), static_cast<unsigned long long>(status.successes));
    ImGui::TextColored(status.succeeded ? ImVec4(0.3f, 0.9f, 0.4f, 1.0f) : ImVec4(1.0f, 0.4f, 0.3f, 1.0f),
        "%s", status.succeeded ? "Last evaluation succeeded" : "Last evaluation failed");
    if (!status.succeeded) { ImGui::TextWrapped("%s", status.message.c_str()); }
    ImGui::Text("Mode: %s", dlssModeName(status.mode));
    ImGui::Text("Render: %u x %u   Output: %u x %u",
        status.renderWidth, status.renderHeight, status.outputWidth, status.outputHeight);
    ImGui::Text("CPU evaluation: %.3f ms (not GPU time)", status.cpuMs);
    ImGui::Text("History reset: %s   Previous camera: %s",
        status.reset ? "Yes" : "No", status.camera.previousValid ? "Valid" : "Unavailable");
    if (ImGui::TreeNode("Camera / common constants")) {
        const auto& camera = status.camera;
        ImGui::Text("Jitter: %.4f, %.4f", camera.jitterOffset[0], camera.jitterOffset[1]);
        ImGui::Text("Position: %.3f, %.3f, %.3f", camera.eye[0], camera.eye[1], camera.eye[2]);
        ImGui::Text("Near / far: %.5f / %.1f", camera.zNear, camera.zFar);
        ImGui::Text("FOV: %.4f rad   Aspect: %.4f", camera.fovRadians, camera.aspectRatio);
        ImGui::Text("Projection: %s", camera.orthographic ? "Orthographic" : "Perspective");
        ImGui::TextUnformatted("MV scale: (1, 1); camera motion included; unjittered 2D vectors");
        ImGui::TextUnformatted("Depth inverted: No; HDR color: Yes; pre-exposure: 1");
        ImGui::TreePop();
    }
    if (ImGui::TreeNode("Input / output resources")) {
        ImGui::TextDisabled("Resources supplied to the last attempt; a failed attempt may not tag them.");
        if (ImGui::BeginTable("Resources", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
            ImGui::TableSetupColumn("Resource");
            ImGui::TableSetupColumn("Bound");
            ImGui::TableSetupColumn("Extent");
            ImGui::TableSetupColumn("Format");
            ImGui::TableHeadersRow();
            for (uint32_t i = 0; i < status.resourceCount; ++i) {
                const auto& resource = status.resources[i];
                ImGui::TableNextRow();
                ImGui::TableNextColumn(); ImGui::TextUnformatted(resource.name);
                ImGui::TableNextColumn(); ImGui::TextUnformatted(resource.bound ? "Yes" : "Missing");
                ImGui::TableNextColumn(); ImGui::Text("%u x %u", resource.width, resource.height);
                ImGui::TableNextColumn(); ImGui::TextUnformatted(resourceFormatName(resource.format));
            }
            ImGui::EndTable();
        }
        ImGui::TreePop();
    }
    ImGui::PopID();
}

} // namespace

void EditorApplication::drawStreamlineDebugPanel()
{
    if (!streamlineDebugOpen_) { return; }
    ImGui::SetNextWindowSize(ImVec2(650.0f, 720.0f), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Streamline Debug", &streamlineDebugOpen_)) {
        ImGui::End();
        return;
    }
    const auto status = render::vulkan::streamlineDebugStatus();
    if (!status.sdkAvailable) {
        ImGui::TextWrapped("Streamline SDK is not available in this build.");
        ImGui::End();
        return;
    }
    ImGui::Text("Streamline SDK (compiled): %s   API: Vulkan", status.sdkVersion.c_str());
    ImGui::Text("Initialized: %s   Device registered: %s   Frame: %u",
        status.initialized ? "Yes" : "No", status.deviceSet ? "Yes" : "No", status.frameIndex);
    ImGui::Text("NGX descriptor heap workaround: %s", status.descriptorHeapWorkaround ? "Enabled" : "Disabled");
    drawDlssStatus("DLSS Super Resolution", status.dlssSrSupported, status.sr);
    drawDlssStatus("DLSS Ray Reconstruction", status.dlssRrSupported, status.rr);
    if (ImGui::CollapsingHeader("NVIDIA Reflex", ImGuiTreeNodeFlags_DefaultOpen)) {
        const auto& reflex = status.reflex;
        ImGui::Text("Available: %s   Suspended: %s", reflex.available ? "Yes" : "No", reflex.suspended ? "Yes" : "No");
        if (reflex.suspended) { ImGui::TextWrapped("Pacing is paused while minimized or using detached windows."); }
        auto options = reflex.options;
        int mode = static_cast<int>(options.mode);
        // Keep the stored interval exact until the user changes this control.
        int interval = static_cast<int>(options.frameLimitUs);
        ImGui::BeginDisabled(!reflex.available);
        bool changed = ImGui::Combo("Requested mode", &mode, "Off\0On\0On + Boost\0");
        changed |= ImGui::SliderInt("Frame interval (us)", &interval, 0, 100000, "%d", ImGuiSliderFlags_AlwaysClamp);
        ImGui::EndDisabled();
        if (changed) {
            options.mode = static_cast<render::vulkan::StreamlineReflexMode>(mode);
            options.frameLimitUs = static_cast<uint32_t>(interval);
            if (!render::vulkan::setStreamlineReflexOptions(options)) {
                ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.3f, 1.0f), "Failed to update Reflex options.");
            }
        }
        if (options.frameLimitUs) { ImGui::Text("Requested cap: %.1f FPS", 1000000.0 / options.frameLimitUs); }
        else { ImGui::TextDisabled("Frame interval 0 disables the Reflex frame cap."); }
        if (reflex.latencyReportAvailable && !reflex.suspended) {
            ImGui::Text("Report frame: %llu", static_cast<unsigned long long>(reflex.reportFrameId));
            ImGui::Text("Render latency: %.3f ms   GPU render: %.3f ms", reflex.renderLatencyMs, reflex.gpuRenderMs);
            ImGui::TextDisabled("Simulation start to GPU end; excludes display latency. Cached every 60 frames.");
        } else { ImGui::TextDisabled("No current latency report."); }
    }
    if (ImGui::CollapsingHeader("Streamline SDK overlay / buffer visualizer")) {
        ImGui::TextWrapped("This panel uses Metallic integration data. NVIDIA's sl.imgui overlay requires non-production Streamline binaries. Its documented overlay shortcut is Ctrl+Shift+Home; consult the SDK overlay for current shortcuts.");
        ImGui::TextWrapped("The SDK buffer visualizer documented for DLSS Frame Generation is not available here: Metallic currently integrates SR, RR and Reflex, not DLSS-G.");
        ImGui::TextWrapped("Use the existing NVML Monitor for system and GPU memory information, and Profiler for GPU pass timings.");
    }
    ImGui::End();
}

} // namespace metallic
