# metal-cpp
add_library(metal-cpp INTERFACE)
target_include_directories(metal-cpp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/metal-cpp)
target_link_libraries(metal-cpp INTERFACE
    "-framework Metal"
    "-framework Foundation"
    "-framework QuartzCore"
)

# meshoptimizer
add_library(meshoptimizer STATIC
    meshoptimizer/src/clusterizer.cpp
    meshoptimizer/src/indexgenerator.cpp
    meshoptimizer/src/vcacheoptimizer.cpp
    meshoptimizer/src/vfetchoptimizer.cpp
)
target_include_directories(meshoptimizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/meshoptimizer/src)

# GLFW
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
add_subdirectory(glfw)

# Dear ImGui
add_library(imgui STATIC
    imgui/imgui.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_tables.cpp
    imgui/imgui_widgets.cpp
    imgui/imgui_demo.cpp
    imgui/backends/imgui_impl_glfw.cpp
    imgui/backends/imgui_impl_metal.mm
)
target_include_directories(imgui PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui/backends
)
target_link_libraries(imgui PUBLIC glfw metal-cpp "-framework Metal" "-framework AppKit")


add_subdirectory(MathLib)