add_library(metal-cpp INTERFACE)
target_include_directories(metal-cpp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/metal-cpp)
target_link_libraries(metal-cpp INTERFACE
    "-framework Metal"
    "-framework Foundation"
    "-framework QuartzCore"
)
target_compile_definitions(metal-cpp INTERFACE
    NS_PRIVATE_IMPLEMENTATION
    MTL_PRIVATE_IMPLEMENTATION
    CA_PRIVATE_IMPLEMENTATION
)
