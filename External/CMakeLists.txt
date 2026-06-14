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

# SDL3: prefer a complete vendored checkout, but allow package managers to
# provide SDL3 while the repository is being restructured.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/SDL3/CMakeLists.txt")
    set(SDL_TESTS OFF CACHE BOOL "" FORCE)
    set(SDL_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(SDL_RENDER OFF CACHE BOOL "" FORCE)
    set(SDL_RENDER_D3D OFF CACHE BOOL "" FORCE)
    set(SDL_RENDER_D3D11 OFF CACHE BOOL "" FORCE)
    set(SDL_RENDER_D3D12 OFF CACHE BOOL "" FORCE)
    set(SDL_RENDER_GPU OFF CACHE BOOL "" FORCE)
    set(SDL_RENDER_VULKAN OFF CACHE BOOL "" FORCE)
    set(SDL_GPU OFF CACHE BOOL "" FORCE)
    set(SDL_GPU_OPENXR OFF CACHE BOOL "" FORCE)
    set(SDL_VULKAN ON CACHE BOOL "" FORCE)
    add_subdirectory(SDL3)
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/SDL/CMakeLists.txt")
    set(SDL_TESTS OFF CACHE BOOL "" FORCE)
    set(SDL_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(SDL_RENDER OFF CACHE BOOL "" FORCE)
    set(SDL_RENDER_D3D OFF CACHE BOOL "" FORCE)
    set(SDL_RENDER_D3D11 OFF CACHE BOOL "" FORCE)
    set(SDL_RENDER_D3D12 OFF CACHE BOOL "" FORCE)
    set(SDL_RENDER_GPU OFF CACHE BOOL "" FORCE)
    set(SDL_RENDER_VULKAN OFF CACHE BOOL "" FORCE)
    set(SDL_GPU OFF CACHE BOOL "" FORCE)
    set(SDL_GPU_OPENXR OFF CACHE BOOL "" FORCE)
    set(SDL_VULKAN ON CACHE BOOL "" FORCE)
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

if(APPLE)
    target_sources(imgui PRIVATE imgui/backends/imgui_impl_metal.mm)
    target_link_libraries(imgui PUBLIC metal-cpp "-framework Metal" "-framework AppKit")
elseif(WIN32)
    if(TARGET metallic::volk)
        target_sources(imgui PRIVATE imgui/backends/imgui_impl_vulkan.cpp)
        target_compile_definitions(imgui PUBLIC IMGUI_IMPL_VULKAN_USE_VOLK)
        target_link_libraries(imgui PUBLIC metallic::volk)
    else()
        find_package(Vulkan QUIET)
        if(Vulkan_FOUND)
            target_sources(imgui PRIVATE imgui/backends/imgui_impl_vulkan.cpp)
            target_link_libraries(imgui PUBLIC Vulkan::Vulkan)
        endif()
    endif()
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/MathLib/CMakeLists.txt")
    add_subdirectory(MathLib)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tracy/CMakeLists.txt")
    option(TRACY_ENABLE "Enable Tracy profiler" ON)
    option(TRACY_ON_DEMAND "Tracy on-demand mode" ON)
    add_subdirectory(tracy)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/spdlog/CMakeLists.txt")
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
