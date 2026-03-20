# meshoptimizer — needs optimization even in Debug for performance
# Built in a subdirectory so we can override CMAKE_CXX_FLAGS_DEBUG without affecting other targets
if(MSVC)
    string(REPLACE "/RTC1" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
endif()

add_library(meshoptimizer STATIC
    ../meshoptimizer/src/allocator.cpp
    ../meshoptimizer/src/clusterizer.cpp
    ../meshoptimizer/src/indexgenerator.cpp
    ../meshoptimizer/src/partition.cpp
    ../meshoptimizer/src/simplifier.cpp
    ../meshoptimizer/src/vcacheoptimizer.cpp
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
