add_executable(rendergraph main.cpp metal_impl.cpp glfw_metal_bridge.mm imgui_metal_bridge.mm mesh_loader.cpp input.cpp meshlet_builder.cpp material_loader.cpp)
target_include_directories(rendergraph PRIVATE "${CMAKE_SOURCE_DIR}/External/cgltf" "${CMAKE_SOURCE_DIR}/External/stb")
target_link_libraries(rendergraph PRIVATE metal-cpp glfw meshoptimizer imgui slang::slang MathLib "-framework AppKit")

# Copy Slang dylibs next to executable at build time
set(SLANG_LIB_DIR "${CMAKE_SOURCE_DIR}/External/slang/lib")
add_custom_command(TARGET rendergraph POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${SLANG_LIB_DIR}/libslang-compiler.dylib"
        "$<TARGET_FILE_DIR:rendergraph>/libslang-compiler.dylib"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${SLANG_LIB_DIR}/libslang-rt.dylib"
        "$<TARGET_FILE_DIR:rendergraph>/libslang-rt.dylib"
)

# Copy shader files next to executable
add_custom_command(TARGET rendergraph POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/Shaders"
        "$<TARGET_FILE_DIR:rendergraph>/Shaders"
)

# Copy asset files next to executable
add_custom_command(TARGET rendergraph POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/Asset"
        "$<TARGET_FILE_DIR:rendergraph>/Asset"
)
