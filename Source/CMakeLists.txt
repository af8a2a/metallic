add_executable(rendergraph main.cpp metal_impl.cpp glfw_metal_bridge.mm)
target_link_libraries(rendergraph PRIVATE metal-cpp glfw "-framework AppKit")
