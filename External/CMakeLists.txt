if(APPLE)
    add_library(metal-cpp INTERFACE)
    target_include_directories(metal-cpp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/metal-cpp)
    target_link_libraries(metal-cpp INTERFACE
        "-framework Metal"
        "-framework Foundation"
        "-framework QuartzCore"
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/meshoptimizer_build/CMakeLists.txt")
    add_subdirectory(meshoptimizer_build)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/stdexec/include/stdexec/execution.hpp")
    find_package(Threads REQUIRED)

    add_library(stdexec INTERFACE)
    add_library(STDEXEC::stdexec ALIAS stdexec)
    target_include_directories(stdexec SYSTEM INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/stdexec/include"
    )
    target_link_libraries(stdexec INTERFACE Threads::Threads)
    target_compile_features(stdexec INTERFACE cxx_std_20)
    target_compile_options(stdexec INTERFACE
        $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-fcoroutines;-fconcepts-diagnostics-depth=10>
        $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/Zc:__cplusplus;/Zc:preprocessor;/Zc:externConstexpr;/bigobj>
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/entt/entt/entity/registry.hpp")
    add_library(metallic_entt INTERFACE)
    add_library(EnTT::EnTT ALIAS metallic_entt)
    target_include_directories(metallic_entt SYSTEM INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/entt"
    )
else()
    find_package(EnTT CONFIG REQUIRED)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/volk/volk.c")
    add_library(volk STATIC
        volk/volk.c
    )
    add_library(metallic::volk ALIAS volk)
    target_include_directories(volk PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/volk"
    )

    set(METALLIC_VOLK_VULKAN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/SDL3/src/video/khronos")
    if(EXISTS "${METALLIC_VOLK_VULKAN_INCLUDE_DIR}/vulkan/vulkan.h")
        target_include_directories(volk SYSTEM PUBLIC
            "${METALLIC_VOLK_VULKAN_INCLUDE_DIR}"
        )
        target_compile_definitions(volk PUBLIC
            "VOLK_VULKAN_H_PATH=\"${METALLIC_VOLK_VULKAN_INCLUDE_DIR}/vulkan/vulkan.h\""
        )
    endif()
endif()

if(METALLIC_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/googletest/CMakeLists.txt")
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    if(MSVC)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    endif()
    add_subdirectory(googletest)
endif()
# SDL3: prefer a complete vendored checkout, but allow package managers to
# provide SDL3 while the repository is being restructured.
if(METALLIC_DEPENDENCY_MODE STREQUAL "PREBUILT")
    set(SDL3_DIR "${METALLIC_DEPENDENCY_ROOT}/cmake")
    find_package(SDL3 CONFIG REQUIRED GLOBAL PATHS "${METALLIC_DEPENDENCY_ROOT}" NO_DEFAULT_PATH)
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/SDL3/CMakeLists.txt")
    metallic_sdl_options()
    add_subdirectory(SDL3)
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/SDL/CMakeLists.txt")
    metallic_sdl_options()
    add_subdirectory(SDL)
else()
    find_package(SDL3 CONFIG REQUIRED)
endif()

set(METALLIC_SDL_TARGET "")
foreach(METALLIC_SDL_CANDIDATE IN ITEMS
    SDL3::SDL3
    SDL3::SDL3-static
    SDL3::SDL3-shared
    SDL3
    SDL3-static
    SDL3-shared
)
    if(NOT METALLIC_SDL_TARGET AND TARGET ${METALLIC_SDL_CANDIDATE})
        set(METALLIC_SDL_TARGET ${METALLIC_SDL_CANDIDATE})
    endif()
endforeach()

if(NOT METALLIC_SDL_TARGET)
    message(FATAL_ERROR "SDL3 was found, but no usable SDL3 CMake target is available.")
endif()

add_library(metallic_sdl3 INTERFACE)
add_library(metallic::sdl3 ALIAS metallic_sdl3)
target_link_libraries(metallic_sdl3 INTERFACE ${METALLIC_SDL_TARGET})

# Dear ImGui
add_library(imgui STATIC
    imgui/imgui.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_tables.cpp
    imgui/imgui_widgets.cpp
    imgui/imgui_demo.cpp
    imgui/backends/imgui_impl_sdl3.cpp
)
target_include_directories(imgui PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui/backends
)
target_link_libraries(imgui PUBLIC metallic::sdl3)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ImGuizmo/src/ImGuizmo.cpp")
    add_library(imguizmo STATIC
        ImGuizmo/src/ImGuizmo.cpp
    )
    add_library(imguizmo::imguizmo ALIAS imguizmo)
    target_include_directories(imguizmo PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/ImGuizmo/src"
    )
    target_link_libraries(imguizmo PUBLIC imgui)
endif()

if(APPLE)
    target_sources(imgui PRIVATE imgui/backends/imgui_impl_metal.mm)
    target_link_libraries(imgui PUBLIC metal-cpp "-framework Metal" "-framework AppKit")
elseif(WIN32)
    include("${CMAKE_SOURCE_DIR}/cmake/PrepareImGuiVulkanBackend.cmake")
    set(METALLIC_IMGUI_VULKAN_BACKEND "${CMAKE_CURRENT_BINARY_DIR}/imgui/imgui_impl_vulkan.cpp")
    metallic_prepare_imgui_vulkan_backend(
        "${CMAKE_CURRENT_SOURCE_DIR}/imgui/backends/imgui_impl_vulkan.cpp"
        "${METALLIC_IMGUI_VULKAN_BACKEND}")
    if(TARGET metallic::volk)
        target_sources(imgui PRIVATE "${METALLIC_IMGUI_VULKAN_BACKEND}")
        target_compile_definitions(imgui PUBLIC IMGUI_IMPL_VULKAN_USE_VOLK)
        target_link_libraries(imgui PUBLIC metallic::volk)
    else()
        find_package(Vulkan QUIET)
        if(Vulkan_FOUND)
            target_sources(imgui PRIVATE "${METALLIC_IMGUI_VULKAN_BACKEND}")
            target_link_libraries(imgui PUBLIC Vulkan::Vulkan)
        endif()
    endif()
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/MathLib/CMakeLists.txt")
    add_subdirectory(MathLib)
endif()

# NRD kernels are vendored under Shaders/Libraries/Denoising/NRD and compiled
# on demand by Metallic's Slang cache. No NRD SDK, ShaderMake, DXC, or shader
# permutation build is required here.

if(METALLIC_ENABLE_TRACY)
    option(TRACY_ENABLE "Enable Tracy profiler" ON)
    option(TRACY_ON_DEMAND "Tracy on-demand mode" ON)
    set(TRACY_STATIC ON CACHE BOOL "Build Tracy into Metallic" FORCE)
    add_subdirectory(tracy)
endif()

if(METALLIC_DEPENDENCY_MODE STREQUAL "PREBUILT")
    set(spdlog_DIR "${METALLIC_DEPENDENCY_ROOT}/lib/cmake/spdlog")
    find_package(spdlog CONFIG REQUIRED GLOBAL PATHS "${METALLIC_DEPENDENCY_ROOT}" NO_DEFAULT_PATH)
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/spdlog/CMakeLists.txt")
    add_subdirectory(spdlog)
endif()

add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/nlohmann)

add_library(tinygltf INTERFACE)
target_include_directories(tinygltf INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/tinygltf)
target_compile_definitions(tinygltf INTERFACE
    TINYGLTF_NO_INCLUDE_JSON
    TINYGLTF_NO_INCLUDE_STB_IMAGE
    TINYGLTF_NO_INCLUDE_STB_IMAGE_WRITE
    TINYGLTF_NO_STB_IMAGE_WRITE
)

include("${CMAKE_SOURCE_DIR}/cmake/SetupOpenUsd.cmake")

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/imnodes/CMakeLists.txt")
    add_library(imnodes STATIC
        imnodes/imnodes.cpp
    )
    target_include_directories(imnodes PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/imnodes)
    target_link_libraries(imnodes PUBLIC imgui)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/microprofile/CMakeLists.txt")
    set(MICROPROFILE_USE_CONFIG_FILE OFF CACHE BOOL "Use microprofile.config.h" FORCE)
    add_subdirectory(microprofile)
    target_compile_definitions(microprofile PUBLIC MICROPROFILE_USE_CONFIG=0)
endif()
