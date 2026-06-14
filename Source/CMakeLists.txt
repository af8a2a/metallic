set(METALLIC_SLANG_ROOT "${CMAKE_SOURCE_DIR}/External/slang")
if(DEFINED SLANG_ROOT)
    set(METALLIC_SLANG_ROOT "${SLANG_ROOT}")
endif()
find_package(slang CONFIG REQUIRED PATHS "${METALLIC_SLANG_ROOT}" NO_DEFAULT_PATH)

add_library(MetallicRuntimeRender STATIC
    Runtime/Render/slang_compiler.cpp
    Runtime/Render/RenderGraph/render_graph.cpp
    Runtime/Render/RenderPass/builtin_render_passes.cpp
    Runtime/Render/GAPI/Vulkan/vulkan_rhi.cpp
)

target_include_directories(MetallicRuntimeRender PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_include_directories(MetallicRuntimeRender SYSTEM PRIVATE
    "${CMAKE_SOURCE_DIR}/External/VulkanMemoryAllocator/include"
)

target_link_libraries(MetallicRuntimeRender PUBLIC
    metallic::volk
    metallic::sdl3
    metallic::nsight_events
    nlohmann_json
    slang::slang
)

target_compile_definitions(MetallicRuntimeRender PRIVATE
    PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

if(WIN32)
    target_compile_definitions(MetallicRuntimeRender PRIVATE NOMINMAX)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(MetallicRuntimeRender PRIVATE -Wno-nullability-completeness)
endif()

add_executable(Metallic
    main.cpp
    Editor/editor_application.cpp
)

target_include_directories(Metallic PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(Metallic PRIVATE
    imgui
    imnodes
    metallic::sdl3
    MetallicRuntimeRender
)

target_compile_definitions(Metallic PRIVATE
    PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

if(WIN32)
    target_compile_definitions(Metallic PRIVATE NOMINMAX)

    file(GLOB METALLIC_SLANG_RUNTIME_DLLS "${METALLIC_SLANG_ROOT}/bin/*.dll")
    foreach(METALLIC_SLANG_RUNTIME_DLL IN LISTS METALLIC_SLANG_RUNTIME_DLLS)
        add_custom_command(TARGET Metallic POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${METALLIC_SLANG_RUNTIME_DLL}"
                "$<TARGET_FILE_DIR:Metallic>"
        )
    endforeach()

    if(TARGET SDL3-shared)
        add_custom_command(TARGET Metallic POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:SDL3-shared>"
                "$<TARGET_FILE_DIR:Metallic>"
        )
    elseif(TARGET SDL3::SDL3-shared)
        add_custom_command(TARGET Metallic POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:SDL3::SDL3-shared>"
                "$<TARGET_FILE_DIR:Metallic>"
        )
    endif()
endif()
