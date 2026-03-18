// Pipeline Editor - Standalone tool for editing render pipeline configurations
// This is a separate executable from the Metallic renderer

#include "pipeline_editor.h"
#include "pipeline_asset.h"

#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#include <OpenGL/gl3.h>
#else
#include <imgui_impl_opengl3_loader.h>
#endif

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <imnodes.h>
#include <spdlog/spdlog.h>

#include <string>
#include <cstring>
#include <filesystem>

void registerEditorPassTypes();

int main(int argc, char* argv[]) {
    spdlog::set_level(spdlog::level::info);
    spdlog::info("Pipeline Editor starting...");

    // Register metadata for all pass types shown in the Add menu.
    registerEditorPassTypes();

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

    // Save As dialog state
    bool showSaveAsPopup = false;
    char saveAsBuffer[256] = {};

    // Open dialog state
    bool showOpenPopup = false;
    char openBuffer[256] = {};

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
                    editor.resetLayout();
                    unsavedChanges = true;
                }
                if (ImGui::MenuItem("Open...", "Ctrl+O")) {
                    std::string defaultDir = std::string(PROJECT_SOURCE_DIR) + "/Pipelines/";
                    std::strncpy(openBuffer, defaultDir.c_str(), sizeof(openBuffer) - 1);
                    openBuffer[sizeof(openBuffer) - 1] = '\0';
                    showOpenPopup = true;
                }
                if (ImGui::MenuItem("Save", "Ctrl+S", false, !currentFilePath.empty())) {
                    editor.collectNodePositions(currentPipeline);
                    currentPipeline.save(currentFilePath);
                    unsavedChanges = false;
                    spdlog::info("Saved: {}", currentFilePath);
                }
                if (ImGui::MenuItem("Save As...")) {
                    std::string defaultName = currentFilePath.empty()
                        ? std::string(PROJECT_SOURCE_DIR) + "/Pipelines/new_pipeline.json"
                        : currentFilePath;
                    std::strncpy(saveAsBuffer, defaultName.c_str(), sizeof(saveAsBuffer) - 1);
                    saveAsBuffer[sizeof(saveAsBuffer) - 1] = '\0';
                    showSaveAsPopup = true;
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

        // Save As popup
        if (showSaveAsPopup) {
            ImGui::OpenPopup("Save As");
            showSaveAsPopup = false;
        }
        if (ImGui::BeginPopupModal("Save As", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("File path:");
            ImGui::SetNextItemWidth(400);
            bool enterPressed = ImGui::InputText("##saveas_path", saveAsBuffer, sizeof(saveAsBuffer),
                ImGuiInputTextFlags_EnterReturnsTrue);
            if (enterPressed || ImGui::Button("Save")) {
                std::string path(saveAsBuffer);
                if (!path.empty()) {
                    editor.collectNodePositions(currentPipeline);
                    currentPipeline.save(path);
                    currentFilePath = path;
                    unsavedChanges = false;
                    spdlog::info("Saved as: {}", path);
                }
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel")) {
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }

        // Open popup
        if (showOpenPopup) {
            ImGui::OpenPopup("Open Pipeline");
            showOpenPopup = false;
        }
        if (ImGui::BeginPopupModal("Open Pipeline", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("File path:");
            ImGui::SetNextItemWidth(400);
            bool enterPressed = ImGui::InputText("##open_path", openBuffer, sizeof(openBuffer),
                ImGuiInputTextFlags_EnterReturnsTrue);
            if (enterPressed || ImGui::Button("Open")) {
                std::string path(openBuffer);
                if (!path.empty() && std::filesystem::exists(path)) {
                    currentFilePath = path;
                    currentPipeline = PipelineAsset::load(currentFilePath);
                    editor.resetLayout();
                    unsavedChanges = false;
                    if (!currentPipeline.name.empty()) {
                        spdlog::info("Loaded: {}", currentPipeline.name);
                    }
                } else if (!path.empty()) {
                    spdlog::warn("File not found: {}", path);
                }
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel")) {
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }

        // Dockspace
        ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);

        // File browser panel
        ImGui::Begin("Files");
        ImGui::Text("Pipeline Files:");
        ImGui::Separator();

        // List JSON files in source tree Pipelines directory
        std::string pipelinesDir = std::string(PROJECT_SOURCE_DIR) + "/Pipelines";
        if (std::filesystem::exists(pipelinesDir)) {
            for (const auto& entry : std::filesystem::directory_iterator(pipelinesDir)) {
                if (entry.path().extension() == ".json") {
                    std::string filename = entry.path().filename().string();
                    bool isSelected = (currentFilePath == entry.path().string());
                    if (ImGui::Selectable(filename.c_str(), isSelected)) {
                        currentFilePath = entry.path().string();
                        currentPipeline = PipelineAsset::load(currentFilePath);
                        editor.resetLayout();
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
                editor.collectNodePositions(currentPipeline);
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
