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

    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        state->rightMouseDown = (action == GLFW_PRESS);
        if (state->rightMouseDown) {
            glfwGetCursorPos(window, &state->lastMouseX, &state->lastMouseY);
        }
    }
}

static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    if (ImGui::GetIO().WantCaptureMouse)
        return;
    auto* state = static_cast<InputState*>(glfwGetWindowUserPointer(window));
    if (!state->camera) return;

    double dx = xpos - state->lastMouseX;
    double dy = ypos - state->lastMouseY;

    if (state->camera->mode == CameraMode::FPS) {
        if (state->rightMouseDown) {
            state->camera->lookFPS(static_cast<float>(dx), static_cast<float>(dy));
            state->lastMouseX = xpos;
            state->lastMouseY = ypos;
        }
    } else {
        if (state->mouseDown) {
            state->camera->rotate(static_cast<float>(dx) * 0.005f, static_cast<float>(dy) * 0.005f);
            state->lastMouseX = xpos;
            state->lastMouseY = ypos;
        }
    }
}

static void scrollCallback(GLFWwindow* window, double /*xoffset*/, double yoffset) {
    if (ImGui::GetIO().WantCaptureMouse)
        return;
    auto* state = static_cast<InputState*>(glfwGetWindowUserPointer(window));
    if (!state->camera) return;

    if (state->camera->mode == CameraMode::FPS) {
        // Scroll adjusts move speed in FPS mode
        state->camera->moveSpeed *= (1.0f + static_cast<float>(yoffset) * 0.1f);
        if (state->camera->moveSpeed < 0.01f) state->camera->moveSpeed = 0.01f;
    } else {
        state->camera->zoom(static_cast<float>(yoffset));
    }
}

void setupInputCallbacks(GLFWwindow* window, InputState* state) {
    state->window = window;
    glfwSetWindowUserPointer(window, state);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetScrollCallback(window, scrollCallback);
}

void updateCameraInput(GLFWwindow* window, InputState* state, float deltaTime) {
    if (!state->camera) return;
    if (ImGui::GetIO().WantCaptureKeyboard) return;

    // Tab to toggle mode (with debounce)
    static bool tabWasDown = false;
    bool tabDown = glfwGetKey(window, GLFW_KEY_TAB) == GLFW_PRESS;
    if (tabDown && !tabWasDown) {
        if (state->camera->mode == CameraMode::Orbit) {
            state->camera->switchToFPS();
        } else {
            state->camera->switchToOrbit();
        }
    }
    tabWasDown = tabDown;

    // WASD + QE movement in FPS mode
    if (state->camera->mode == CameraMode::FPS) {
        float speed = state->camera->moveSpeed * deltaTime;

        float3 fwd = state->camera->forwardDirection();
        float3 worldUp(0.f, 1.f, 0.f);
        float3 right = normalize(cross(fwd, worldUp));

        float3 move(0.f);
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) move = move + fwd * speed;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) move = move - fwd * speed;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) move = move + right * speed;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) move = move - right * speed;
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) move = move + worldUp * speed;
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) move = move - worldUp * speed;

        state->camera->moveFPS(move);
    }
}
