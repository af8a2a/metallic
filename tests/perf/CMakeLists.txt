# Can run without configuring the renderer:
# cmake -S tests/perf -B build/perf-tests && ctest --test-dir build/perf-tests -C Debug
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.24)
    project(MetallicPerfEvidenceTests NONE)
    enable_testing()
endif()

find_package(Python3 QUIET COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    add_test(NAME MetallicWorkloadCaseTests
        COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/TestWorkloadCase.py")
    set_tests_properties(MetallicWorkloadCaseTests PROPERTIES LABELS "tools;perf" TIMEOUT 30)
    add_test(NAME MetallicPerfEvidenceTests
        COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/TestBaseline.py")
    set_tests_properties(MetallicPerfEvidenceTests PROPERTIES LABELS "tools;perf" TIMEOUT 30)
    add_test(NAME MetallicNsightSourceTests
        COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/TestNsightSource.py")
    set_tests_properties(MetallicNsightSourceTests PROPERTIES LABELS "tools;perf" TIMEOUT 30)
    add_test(NAME MetallicNativeNsightSourceTests
        COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/TestNativeNsightSource.py")
    set_tests_properties(MetallicNativeNsightSourceTests PROPERTIES LABELS "tools;perf" TIMEOUT 30)
endif()
