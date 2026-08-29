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

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/oneTBB/CMakeLists.txt")
    set(_METALLIC_BUILD_SHARED_LIBS "${BUILD_SHARED_LIBS}")
    set(BUILD_SHARED_LIBS ON)
    set(TBB_TEST OFF CACHE BOOL "Build oneTBB tests" FORCE)
    set(TBB_EXAMPLES OFF CACHE BOOL "Build oneTBB examples" FORCE)
    set(TBB_STRICT OFF CACHE BOOL "Treat oneTBB warnings as errors" FORCE)
    set(TBBMALLOC_BUILD OFF CACHE BOOL "Build oneTBB allocator" FORCE)
    set(TBBMALLOC_PROXY_BUILD OFF CACHE BOOL "Build oneTBB allocator proxy" FORCE)
    set(TBB4PY_BUILD OFF CACHE BOOL "Build oneTBB Python bindings" FORCE)
    set(TBB_ENABLE_IPO OFF CACHE BOOL "Enable oneTBB IPO" FORCE)
    set(_METALLIC_CMAKE_POLICY_VERSION_MINIMUM "${CMAKE_POLICY_VERSION_MINIMUM}")
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
    add_subdirectory(oneTBB)
    set(CMAKE_POLICY_VERSION_MINIMUM "${_METALLIC_CMAKE_POLICY_VERSION_MINIMUM}")
    unset(_METALLIC_CMAKE_POLICY_VERSION_MINIMUM)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/OpenUSD/CMakeLists.txt")
    if(NOT TARGET TBB::tbb)
        message(FATAL_ERROR "OpenUSD requires the vendored oneTBB target")
    endif()

    set(TBB_FOUND TRUE)
    set(PXR_FIND_TBB_IN_CONFIG OFF CACHE BOOL "Use the in-tree oneTBB target" FORCE)
    set(PXR_BUILD_TESTS OFF CACHE BOOL "Build OpenUSD tests" FORCE)
    set(PXR_BUILD_EXAMPLES OFF CACHE BOOL "Build OpenUSD examples" FORCE)
    set(PXR_BUILD_TUTORIALS OFF CACHE BOOL "Build OpenUSD tutorials" FORCE)
    set(PXR_BUILD_USD_TOOLS OFF CACHE BOOL "Build OpenUSD command-line tools" FORCE)
    set(PXR_BUILD_IMAGING OFF CACHE BOOL "Build OpenUSD imaging" FORCE)
    set(PXR_BUILD_USD_IMAGING OFF CACHE BOOL "Build OpenUSD imaging adapters" FORCE)
    set(PXR_BUILD_USD_VALIDATION OFF CACHE BOOL "Build OpenUSD validation" FORCE)
    set(PXR_BUILD_EXEC OFF CACHE BOOL "Build OpenUSD execution framework" FORCE)
    set(PXR_BUILD_USDVIEW OFF CACHE BOOL "Build usdview" FORCE)
    set(PXR_ENABLE_PYTHON_SUPPORT OFF CACHE BOOL "Build OpenUSD Python bindings" FORCE)
    set(PXR_ENABLE_GL_SUPPORT OFF CACHE BOOL "Build OpenUSD OpenGL support" FORCE)
    set(PXR_ENABLE_METAL_SUPPORT OFF CACHE BOOL "Build OpenUSD Metal support" FORCE)
    set(PXR_ENABLE_VULKAN_SUPPORT OFF CACHE BOOL "Build OpenUSD Vulkan support" FORCE)
    set(PXR_ENABLE_MATERIALX_SUPPORT OFF CACHE BOOL "Build OpenUSD MaterialX support" FORCE)
    set(PXR_BUILD_MONOLITHIC ON CACHE BOOL "Build one OpenUSD shared library" FORCE)
    set(PXR_PREFER_SAFETY_OVER_SPEED ON CACHE BOOL "Validate malformed USD input" FORCE)
    add_subdirectory(OpenUSD)

    set(BUILD_SHARED_LIBS "${_METALLIC_BUILD_SHARED_LIBS}")
    unset(_METALLIC_BUILD_SHARED_LIBS)

    add_library(metallic_openusd INTERFACE)
    add_library(metallic::openusd ALIAS metallic_openusd)
    target_link_libraries(metallic_openusd INTERFACE usd_m)
    target_include_directories(metallic_openusd SYSTEM INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/OpenUSD"
        "${CMAKE_CURRENT_BINARY_DIR}/OpenUSD/include"
    )

    function(metallic_copy_openusd_runtime target_name)
        if(NOT TARGET ${target_name} OR NOT TARGET usd_m OR NOT TARGET tbb)
            return()
        endif()

        set(openusd_binary_root "${CMAKE_BINARY_DIR}/External/OpenUSD")
        set(openusd_source_root "${CMAKE_SOURCE_DIR}/External/OpenUSD")
        set(runtime_root "$<TARGET_FILE_DIR:${target_name}>")
        set(core_resource_root "${runtime_root}/usd")
        set(plugin_resource_root "${runtime_root}/../plugin/usd")

        set(runtime_commands
            COMMAND ${CMAKE_COMMAND} -E make_directory "${runtime_root}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:usd_m>"
                "${runtime_root}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:tbb>"
                "${runtime_root}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${core_resource_root}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${openusd_binary_root}/plugins_plugInfo.json"
                "${core_resource_root}/plugInfo.json"
        )

        file(GLOB openusd_core_plug_infos
            RELATIVE "${openusd_binary_root}/pxr/usd"
            "${openusd_binary_root}/pxr/usd/*/plugInfo.json"
        )
        foreach(relative_plug_info IN LISTS openusd_core_plug_infos)
            get_filename_component(module_name "${relative_plug_info}" DIRECTORY)
            set(module_resource_root "${core_resource_root}/${module_name}/resources")
            list(APPEND runtime_commands
                COMMAND ${CMAKE_COMMAND} -E make_directory "${module_resource_root}"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${openusd_binary_root}/pxr/usd/${relative_plug_info}"
                    "${module_resource_root}/plugInfo.json"
            )

            set(module_source_root "${openusd_source_root}/pxr/usd/${module_name}")
            if(EXISTS "${module_source_root}/generatedSchema.usda")
                list(APPEND runtime_commands
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${module_source_root}/generatedSchema.usda"
                        "${module_resource_root}/generatedSchema.usda"
                )
            endif()
            if(EXISTS "${module_source_root}/schema.usda")
                list(APPEND runtime_commands
                    COMMAND ${CMAKE_COMMAND} -E make_directory
                        "${module_resource_root}/${module_name}"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${module_source_root}/schema.usda"
                        "${module_resource_root}/${module_name}/schema.usda"
                )
            endif()
        endforeach()

        set(usd_shaders_binary_root
            "${openusd_binary_root}/pxr/usd/plugin/usdShaders")
        set(usd_shaders_source_root
            "${openusd_source_root}/pxr/usd/plugin/usdShaders")
        if(EXISTS "${usd_shaders_binary_root}/plugInfo.json")
            list(APPEND runtime_commands
                COMMAND ${CMAKE_COMMAND} -E make_directory "${plugin_resource_root}"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${openusd_binary_root}/usd_plugInfo.json"
                    "${plugin_resource_root}/plugInfo.json"
                COMMAND ${CMAKE_COMMAND} -E make_directory
                    "${plugin_resource_root}/usdShaders/resources"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${usd_shaders_binary_root}/plugInfo.json"
                    "${plugin_resource_root}/usdShaders/resources/plugInfo.json"
                COMMAND ${CMAKE_COMMAND} -E copy_directory
                    "${usd_shaders_source_root}/shaders"
                    "${plugin_resource_root}/usdShaders/resources/shaders"
            )
        endif()

        add_custom_target(${target_name}OpenUsdRuntime
            ${runtime_commands}
            DEPENDS usd_m tbb
            COMMENT "Deploying OpenUSD runtime for ${target_name}"
            VERBATIM
        )
        add_dependencies(${target_name} ${target_name}OpenUsdRuntime)
    endfunction()
endif()

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
