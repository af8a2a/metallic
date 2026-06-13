find_package(Vulkan REQUIRED)

add_library(MetallicRuntimeRender STATIC
    Runtime/Render/GAPI/Vulkan/vulkan_rhi.cpp
)

target_include_directories(MetallicRuntimeRender PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_include_directories(MetallicRuntimeRender SYSTEM PRIVATE
    "${CMAKE_SOURCE_DIR}/External/VulkanMemoryAllocator/include"
)

target_link_libraries(MetallicRuntimeRender PUBLIC
    Vulkan::Vulkan
    metallic::sdl3
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
    metallic::sdl3
    MetallicRuntimeRender
)

target_compile_definitions(Metallic PRIVATE
    PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

if(WIN32)
    target_compile_definitions(Metallic PRIVATE NOMINMAX)

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
