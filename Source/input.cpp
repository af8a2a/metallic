#include "input.h"
#include "camera.h"
#include "imgui.h"

static void mouseButtonCallback(GLFWwindow* window, int button, int action, int /*mods*/) {
    if (ImGui::GetIO().WantCaptureMouse)
        return;
    auto* state = static_cast<InputState*>(glfwGetWindowUserPointer(window));
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        state->mouseDown = (action == GLFW_PRESS);
        if (state->mouseDown) {
            glfwGetCursorPos(window, &state->lastMouseX, &state->lastMouseY);
        }
    }
}

static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    if (ImGui::GetIO().WantCaptureMouse)
        return;
    auto* state = static_cast<InputState*>(glfwGetWindowUserPointer(window));
    if (state->mouseDown && state->camera) {
        double dx = xpos - state->lastMouseX;
        double dy = ypos - state->lastMouseY;
        state->camera->rotate((float)dx * 0.005f, (float)dy * 0.005f);
        state->lastMouseX = xpos;
        state->lastMouseY = ypos;
    }
}

static void scrollCallback(GLFWwindow* window, double /*xoffset*/, double yoffset) {
    if (ImGui::GetIO().WantCaptureMouse)
        return;
    auto* state = static_cast<InputState*>(glfwGetWindowUserPointer(window));
    if (state->camera) {
        state->camera->zoom((float)yoffset);
    }
}

void setupInputCallbacks(GLFWwindow* window, InputState* state) {
    glfwSetWindowUserPointer(window, state);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetScrollCallback(window, scrollCallback);
}
