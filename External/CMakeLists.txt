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

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/googletest/CMakeLists.txt")
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    if(MSVC)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    endif()
    add_subdirectory(googletest)
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

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ShaderMakeBlob/ShaderBlob.cpp")
    add_library(ShaderMakeBlob STATIC
        ShaderMakeBlob/ShaderBlob.cpp
    )
    target_include_directories(ShaderMakeBlob PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/ShaderMakeBlob"
        "${CMAKE_CURRENT_SOURCE_DIR}/ShaderMakeBlob/ShaderMake"
    )
    target_compile_features(ShaderMakeBlob PUBLIC cxx_std_17)

    add_executable(ShaderMake
        ShaderMakeBlob/ShaderMake.cpp
        ShaderMakeBlob/argparse.c
        ShaderMakeBlob/ShaderBlob.cpp
    )
    target_include_directories(ShaderMake PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/ShaderMakeBlob"
        "${CMAKE_CURRENT_SOURCE_DIR}/ShaderMakeBlob/ShaderMake"
    )
    target_compile_features(ShaderMake PRIVATE cxx_std_17)
    if(WIN32)
        target_compile_definitions(ShaderMake PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    endif()
endif()

set(METALLIC_NRD_ROOT "" CACHE PATH "Path to the NVIDIA NRD SDK checkout")
if(NOT METALLIC_NRD_ROOT)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/nrd/CMakeLists.txt")
        set(METALLIC_NRD_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/nrd")
    elseif(EXISTS "E:/vk_denoise_nrd/externals/nrd/CMakeLists.txt")
        set(METALLIC_NRD_ROOT "E:/vk_denoise_nrd/externals/nrd")
    endif()
endif()

if(METALLIC_NRD_ROOT AND EXISTS "${METALLIC_NRD_ROOT}/CMakeLists.txt")
    set(NRD_STATIC_LIBRARY ON CACHE BOOL "" FORCE)
    set(NRD_NRI OFF CACHE BOOL "" FORCE)
    set(NRD_EMBEDS_SPIRV_SHADERS ON CACHE BOOL "" FORCE)
    set(NRD_EMBEDS_DXIL_SHADERS OFF CACHE BOOL "" FORCE)
    set(NRD_EMBEDS_DXBC_SHADERS OFF CACHE BOOL "" FORCE)
    set(NRD_NORMAL_ENCODING "2" CACHE STRING "" FORCE)
    set(NRD_ROUGHNESS_ENCODING "1" CACHE STRING "" FORCE)
    # Generate embedded shader blobs into the build tree; the source-tree
    # default may be write-protected in some deployments.
    set(NRD_SHADERS_PATH "${CMAKE_BINARY_DIR}/External/nrd-shaders" CACHE STRING "" FORCE)
    if(TARGET ShaderMake)
        set(SHADERMAKE_PATH "$<TARGET_FILE:ShaderMake>" CACHE STRING "" FORCE)
    endif()
    if(EXISTS "E:/vk_denoise_nrd/out/build/x64-Debug/_deps/dxc-src/bin/x64/dxc.exe")
        set(DXC_SPIRV_PATH "E:/vk_denoise_nrd/out/build/x64-Debug/_deps/dxc-src/bin/x64/dxc.exe" CACHE FILEPATH "" FORCE)
        set(SHADERMAKE_DXC_VK_PATH "E:/vk_denoise_nrd/out/build/x64-Debug/_deps/dxc-src/bin/x64/dxc.exe" CACHE FILEPATH "" FORCE)
    endif()
    add_subdirectory("${METALLIC_NRD_ROOT}" "${CMAKE_BINARY_DIR}/External/nrd")
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
