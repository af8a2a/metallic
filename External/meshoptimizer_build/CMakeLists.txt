# meshoptimizer — needs optimization even in Debug for performance
# Built in a subdirectory so we can override CMAKE_CXX_FLAGS_DEBUG without affecting other targets
if(MSVC)
    string(REPLACE "/RTC1" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
endif()

add_library(meshoptimizer STATIC
    ../meshoptimizer/src/allocator.cpp
    ../meshoptimizer/src/clusterizer.cpp
    ../meshoptimizer/src/indexanalyzer.cpp
    ../meshoptimizer/src/indexcodec.cpp
    ../meshoptimizer/src/indexgenerator.cpp
    ../meshoptimizer/src/meshletcodec.cpp
    ../meshoptimizer/src/meshletutils.cpp
    ../meshoptimizer/src/opacitymap.cpp
    ../meshoptimizer/src/overdrawoptimizer.cpp
    ../meshoptimizer/src/partition.cpp
    ../meshoptimizer/src/quantization.cpp
    ../meshoptimizer/src/rasterizer.cpp
    ../meshoptimizer/src/simplifier.cpp
    ../meshoptimizer/src/spatialorder.cpp
    ../meshoptimizer/src/stripifier.cpp
    ../meshoptimizer/src/vcacheoptimizer.cpp
    ../meshoptimizer/src/vertexcodec.cpp
    ../meshoptimizer/src/vertexfilter.cpp
    ../meshoptimizer/src/vfetchoptimizer.cpp
)
target_include_directories(meshoptimizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../meshoptimizer/src)
if(MSVC)
    target_compile_options(meshoptimizer PRIVATE
        $<$<CONFIG:Debug>:/O2>
        $<$<CONFIG:Debug>:/Ob2>
    )
else()
    target_compile_options(meshoptimizer PRIVATE $<$<CONFIG:Debug>:-O3>)
endif()
target_compile_definitions(meshoptimizer PRIVATE $<$<CONFIG:Debug>:NDEBUG>)
