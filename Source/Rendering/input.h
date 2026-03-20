#pragma once

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

struct OrbitCamera;

struct InputState {
    OrbitCamera* camera = nullptr;
    GLFWwindow* window = nullptr;
    bool mouseDown = false;       // left mouse (orbit rotate)
    bool rightMouseDown = false;  // right mouse (FPS look)
    double lastMouseX = 0.0;
    double lastMouseY = 0.0;
};

void setupInputCallbacks(GLFWwindow* window, InputState* state);
void updateCameraInput(GLFWwindow* window, InputState* state, float deltaTime);
