include_guard(GLOBAL)

option(METALLIC_ENABLE_ISPC "Build optional ISPC CPU kernels when the compiler is installed" ON)
option(METALLIC_CPU_KERNEL_VECTOR_REPORT "Print C++ CPU kernel vectorization diagnostics" OFF)
set(METALLIC_ISPC_INSTRUCTION_SETS "sse4.2-i32x4;avx2-i32x8" CACHE STRING
    "CPU kernel ISPC variants (SSE4.2, AVX2, or both for runtime dispatch)")

set(_metallic_cpu_kernel_dir "${CMAKE_CURRENT_LIST_DIR}/../Source/Runtime/CPUKernels")
set(_metallic_cpu_has_ispc OFF)
set(_metallic_cpu_ispc_requires_avx2 OFF)
if(METALLIC_ENABLE_ISPC AND CMAKE_SIZEOF_VOID_P EQUAL 8 AND
        (CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64|X86_64)$" OR
         CMAKE_CXX_COMPILER_ARCHITECTURE_ID STREQUAL "x64"))
    include(CheckLanguage)
    check_language(ISPC)
    if(CMAKE_ISPC_COMPILER)
        enable_language(ISPC)
        set(_metallic_cpu_has_ispc ON)
    else()
        message(STATUS "ISPC compiler unavailable; CPU kernels use the C++ fallback")
    endif()
endif()

add_library(MetallicCPUKernels STATIC
    "${_metallic_cpu_kernel_dir}/AABBTransform.cpp"
    "${_metallic_cpu_kernel_dir}/AABBTransformScalar.cpp"
    "${_metallic_cpu_kernel_dir}/AABBTransformAutoVectorized.cpp")
target_compile_features(MetallicCPUKernels PUBLIC cxx_std_23)
target_include_directories(MetallicCPUKernels PUBLIC "${CMAKE_CURRENT_LIST_DIR}/../Source")
target_compile_definitions(MetallicCPUKernels PRIVATE METALLIC_HAS_ISPC=$<BOOL:${_metallic_cpu_has_ispc}>)

# Keep the reference deliberately scalar without raising the application's ISA.
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    if(METALLIC_CPU_KERNEL_VECTOR_REPORT)
        target_compile_options(MetallicCPUKernels PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/Qvec-report:2>")
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set_source_files_properties("${_metallic_cpu_kernel_dir}/AABBTransformScalar.cpp"
        PROPERTIES COMPILE_OPTIONS "-fno-vectorize;-fno-slp-vectorize")
    if(METALLIC_CPU_KERNEL_VECTOR_REPORT)
        target_compile_options(MetallicCPUKernels PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Rpass=loop-vectorize>")
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set_source_files_properties("${_metallic_cpu_kernel_dir}/AABBTransformScalar.cpp"
        PROPERTIES COMPILE_OPTIONS "-fno-tree-vectorize")
    if(METALLIC_CPU_KERNEL_VECTOR_REPORT)
        target_compile_options(MetallicCPUKernels PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-fopt-info-vec-optimized>")
    endif()
endif()

if(_metallic_cpu_has_ispc)
    foreach(_metallic_cpu_isa IN LISTS METALLIC_ISPC_INSTRUCTION_SETS)
        if(NOT _metallic_cpu_isa MATCHES "^(sse4\\.2-i32x4|avx2-i32x8)$")
            message(FATAL_ERROR "Unsupported CPU kernel ISPC target: ${_metallic_cpu_isa}")
        endif()
    endforeach()
    if(NOT METALLIC_ISPC_INSTRUCTION_SETS)
        message(FATAL_ERROR "METALLIC_ISPC_INSTRUCTION_SETS must contain at least one target")
    endif()
    if(NOT "sse4.2-i32x4" IN_LIST METALLIC_ISPC_INSTRUCTION_SETS)
        set(_metallic_cpu_ispc_requires_avx2 ON)
    endif()
    target_compile_definitions(MetallicCPUKernels PRIVATE
        METALLIC_ISPC_REQUIRES_AVX2=$<BOOL:${_metallic_cpu_ispc_requires_avx2}>)
    # ISPC names this variant's object *_sse4.obj. CMake derives the suffix
    # from the target spelling, so use the equivalent legacy SSE4.2 alias.
    set(_metallic_cpu_ispc_targets "${METALLIC_ISPC_INSTRUCTION_SETS}")
    list(TRANSFORM _metallic_cpu_ispc_targets REPLACE "^sse4\\.2-" "sse4-")
    target_sources(MetallicCPUKernels PRIVATE "${_metallic_cpu_kernel_dir}/AABBTransform.ispc")
    set_target_properties(MetallicCPUKernels PROPERTIES
        ISPC_INSTRUCTION_SETS "${_metallic_cpu_ispc_targets}"
        ISPC_HEADER_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/cpu-kernels-generated")
    target_include_directories(MetallicCPUKernels PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/cpu-kernels-generated")
    target_compile_options(MetallicCPUKernels PRIVATE
        "$<$<COMPILE_LANGUAGE:ISPC>:--opt=disable-fma>"
        "$<$<COMPILE_LANGUAGE:ISPC>:--addressing=64>")
endif()
