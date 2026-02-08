#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#import <QuartzCore/CAMetalLayer.h>
#import <AppKit/AppKit.h>

#include "glfw_metal_bridge.h"

extern "C" void* attachMetalLayerToGLFWWindow(void* glfwWindow) {
    NSWindow* nsWindow = glfwGetCocoaWindow(static_cast<GLFWwindow*>(glfwWindow));
    CAMetalLayer* metalLayer = [CAMetalLayer layer];
    NSView* contentView = [nsWindow contentView];
    [contentView setWantsLayer:YES];
    [contentView setLayer:metalLayer];
    return metalLayer;
}
