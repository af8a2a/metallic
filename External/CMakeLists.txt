# metal-cpp
add_library(metal-cpp INTERFACE)
target_include_directories(metal-cpp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/metal-cpp)
target_link_libraries(metal-cpp INTERFACE
    "-framework Metal"
    "-framework Foundation"
    "-framework QuartzCore"
)

# GLFW
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
add_subdirectory(glfw)
