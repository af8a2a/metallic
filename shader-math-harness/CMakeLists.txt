# Metallic ShaderMathHarness: standalone certified-error validation for
# shader numerics. Dual-mode CMake: it configures standalone
#   cmake -S shader-math-harness -B build-shader-math-harness -G Ninja
# or is consumed from the repository root behind
# METALLIC_BUILD_SHADER_MATH_HARNESS=ON without touching default builds.

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.20)
    project(MetallicShaderMathHarness LANGUAGES CXX)
    set(CMAKE_CXX_STANDARD 23)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    if(MSVC AND NOT DEFINED CMAKE_MSVC_RUNTIME_LIBRARY)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    endif()
endif()

add_executable(MetallicShaderMathHarness
    src/Main.cpp
    src/Numeric.cpp
    src/Contract.cpp
    src/Model.cpp
    src/Certify.cpp
    src/Sweep.cpp
    src/Export.cpp
    src/Report.cpp
)
target_compile_features(MetallicShaderMathHarness PRIVATE cxx_std_23)

if(MSVC)
    target_compile_options(MetallicShaderMathHarness PRIVATE /utf-8 /W4)
    # AVX2/FMA3 only inlines the sweep hot loop (std::fma becomes one vfmadd
    # instruction); correctness never depends on it because std::fma is a
    # single-rounding operation either way. Disable for pre-AVX2 CPUs.
    option(METALLIC_SHADER_MATH_HARNESS_AVX2 "Inline FMA with AVX2 codegen" ON)
    if(METALLIC_SHADER_MATH_HARNESS_AVX2)
        target_compile_options(MetallicShaderMathHarness PRIVATE /arch:AVX2)
    endif()
else()
    target_compile_options(MetallicShaderMathHarness PRIVATE -Wall -Wextra)
endif()

# Registering tests from this subdirectory keeps the self-check runnable under
# ctest both standalone and inside a repository build.
enable_testing()
add_test(NAME MetallicShaderMathHarnessSelfTest
    COMMAND MetallicShaderMathHarness --mode self-test)
set_tests_properties(MetallicShaderMathHarnessSelfTest PROPERTIES TIMEOUT 600)
