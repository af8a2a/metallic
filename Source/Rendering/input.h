#pragma once

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

struct OrbitCamera;

struct InputState {
    OrbitCamera* camera = nullptr;
    bool mouseDown = false;
    double lastMouseX = 0.0;
    double lastMouseY = 0.0;
};

void setupInputCallbacks(GLFWwindow* window, InputState* state);
