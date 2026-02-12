// Pipeline Editor - Standalone tool for editing render pipeline configurations
// This is a separate executable from the Metallic renderer

#include "pipeline_editor.h"
#include "pipeline_asset.h"
#include "pass_registry.h"

#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#include <OpenGL/gl3.h>
#endif

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <imnodes.h>
#include <spdlog/spdlog.h>

#include <string>
#include <filesystem>

// Register all known pass types for the editor
// These are metadata-only - no factory functions needed since we don't instantiate passes
static void registerPassTypes() {
    using Type = PassTypeInfo::Type;

    auto& reg = PassRegistry::instance();

    // Geometry passes
    reg.registerPass("VisibilityPass", nullptr, {
        "VisibilityPass", "Visibility Pass", "Geometry",
        {}, {"visibility", "depth"}, {}, Type::Render
    });
    reg.registerPass("ForwardPass", nullptr, {
        "ForwardPass", "Forward Pass", "Geometry",
        {"skyOutput"}, {"forwardColor", "depth"}, {}, Type::Render
    });

    // Lighting passes
    reg.registerPass("ShadowRayPass", nullptr, {
        "ShadowRayPass", "Shadow Ray Pass", "Lighting",
        {"depth"}, {"shadowMap"}, {}, Type::Compute
    });
    reg.registerPass("DeferredLightingPass", nullptr, {
        "DeferredLightingPass", "Deferred Lighting", "Lighting",
        {"visibility", "depth", "shadowMap", "skyOutput"}, {"lightingOutput"}, {}, Type::Compute
    });

    // Environment passes
    reg.registerPass("SkyPass", nullptr, {
        "SkyPass", "Sky Pass", "Environment",
        {}, {"skyOutput"}, {}, Type::Render
    });

    // Post-process passes
    reg.registerPass("TonemapPass", nullptr, {
        "TonemapPass", "Tonemap", "Post-Process",
        {"lightingOutput"}, {"$backbuffer"}, {}, Type::Render
    });

    // Utility passes
    reg.registerPass("BlitPass", nullptr, {
        "BlitPass", "Blit", "Utility",
        {"source"}, {"destination"}, {}, Type::Blit
    });

    // UI passes
    reg.registerPass("ImGuiOverlayPass", nullptr, {
        "ImGuiOverlayPass", "ImGui Overlay", "UI",
        {"depth"}, {"$backbuffer"}, {}, Type::Render
    });
}

int main(int argc, char* argv[]) {
    spdlog::set_level(spdlog::level::info);
    spdlog::info("Pipeline Editor starting...");

    // Register pass types
    registerPassTypes();

    // Initialize GLFW
    if (!glfwInit()) {
        spdlog::error("Failed to initialize GLFW");
        return 1;
    }

    // OpenGL 3.3 for cross-platform compatibility
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(1280, 800, "Pipeline Editor", nullptr, nullptr);
    if (!window) {
        spdlog::error("Failed to create GLFW window");
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // Initialize imnodes
    ImNodes::CreateContext();
    ImNodes::StyleColorsDark();

    // Pipeline state
    PipelineAsset currentPipeline;
    std::string currentFilePath;
    bool unsavedChanges = false;

    // Load pipeline from command line or default
    if (argc > 1) {
        currentFilePath = argv[1];
        currentPipeline = PipelineAsset::load(currentFilePath);
        if (!currentPipeline.name.empty()) {
            spdlog::info("Loaded pipeline: {}", currentPipeline.name);
        }
    }

    PipelineEditor editor;
    editor.setVisible(true);

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Main menu bar
        if (ImGui::BeginMainMenuBar()) {
            if (ImGui::BeginMenu("File")) {
                if (ImGui::MenuItem("New")) {
                    currentPipeline = PipelineAsset{};
                    currentPipeline.name = "NewPipeline";
                    currentFilePath.clear();
                    unsavedChanges = true;
                }
                if (ImGui::MenuItem("Open...", "Ctrl+O")) {
                    // Simple file dialog - in production use native file dialog
                    // For now, just show a text input
                }
                if (ImGui::MenuItem("Save", "Ctrl+S", false, !currentFilePath.empty())) {
                    currentPipeline.save(currentFilePath);
                    unsavedChanges = false;
                    spdlog::info("Saved: {}", currentFilePath);
                }
                if (ImGui::MenuItem("Save As...")) {
                    // Would show save dialog
                }
                ImGui::Separator();
                if (ImGui::MenuItem("Exit")) {
                    glfwSetWindowShouldClose(window, GLFW_TRUE);
                }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("View")) {
                ImGui::MenuItem("Pipeline Editor", nullptr, &editor.m_visible);
                ImGui::EndMenu();
            }
            ImGui::EndMainMenuBar();
        }

        // Dockspace
        ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);

        // File browser panel
        ImGui::Begin("Files");
        ImGui::Text("Pipeline Files:");
        ImGui::Separator();

        // List JSON files in Pipelines directory
        std::string pipelinesDir = "Pipelines";
        if (std::filesystem::exists(pipelinesDir)) {
            for (const auto& entry : std::filesystem::directory_iterator(pipelinesDir)) {
                if (entry.path().extension() == ".json") {
                    std::string filename = entry.path().filename().string();
                    bool isSelected = (currentFilePath == entry.path().string());
                    if (ImGui::Selectable(filename.c_str(), isSelected)) {
                        currentFilePath = entry.path().string();
                        currentPipeline = PipelineAsset::load(currentFilePath);
                        if (!currentPipeline.name.empty()) {
                            spdlog::info("Loaded: {}", currentPipeline.name);
                        }
                        unsavedChanges = false;
                    }
                }
            }
        } else {
            ImGui::TextDisabled("Pipelines/ directory not found");
        }

        if (ImGui::Button("New Pipeline")) {
            currentPipeline = PipelineAsset{};
            currentPipeline.name = "NewPipeline";
            currentFilePath.clear();
            unsavedChanges = true;
        }
        ImGui::End();

        // Pipeline editor
        editor.render(currentPipeline);
        if (editor.isDirty()) {
            unsavedChanges = true;
            editor.markClean();
        }

        // Status bar
        ImGui::Begin("Status");
        if (!currentFilePath.empty()) {
            ImGui::Text("File: %s", currentFilePath.c_str());
        } else {
            ImGui::Text("File: (unsaved)");
        }
        if (unsavedChanges) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "[Modified]");
        }

        // Quick save button
        if (!currentFilePath.empty() && unsavedChanges) {
            ImGui::SameLine();
            if (ImGui::Button("Save")) {
                currentPipeline.save(currentFilePath);
                unsavedChanges = false;
                spdlog::info("Saved: {}", currentFilePath);
            }
        }
        ImGui::End();

        // Render
        ImGui::Render();
        int displayW, displayH;
        glfwGetFramebufferSize(window, &displayW, &displayH);
        glViewport(0, 0, displayW, displayH);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    ImNodes::DestroyContext();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
