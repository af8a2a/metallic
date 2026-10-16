# Can run without configuring the renderer:
# cmake -S tests/perf -B build/perf-tests && ctest --test-dir build/perf-tests -C Debug
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.24)
    project(MetallicPerfEvidenceTests NONE)
    enable_testing()
endif()

find_package(Python3 QUIET COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    add_test(NAME MetallicSPIRVComplexityTests
        COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/TestSPIRVComplexity.py")
    set_tests_properties(MetallicSPIRVComplexityTests PROPERTIES LABELS "tools;shader;perf" TIMEOUT 30)
    add_test(NAME MetallicComputeShaderFrontendTests
        COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/TestComputeShaderFrontend.py")
    set_tests_properties(MetallicComputeShaderFrontendTests PROPERTIES LABELS "tools;shader;perf" TIMEOUT 30)
    add_test(NAME MetallicShaderWarmupEvidenceTests
        COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/TestMeasureShaderWarmup.py")
    set_tests_properties(MetallicShaderWarmupEvidenceTests PROPERTIES LABELS "tools;shader;perf" TIMEOUT 30)
    add_test(NAME MetallicSlangLinkTimeSpecializationTests
        COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/TestSlangLinkTimeSpecialization.py")
    set_tests_properties(MetallicSlangLinkTimeSpecializationTests PROPERTIES LABELS "tools;shader;perf" TIMEOUT 30)
    add_test(NAME MetallicComputePipelineAnalysisTests
        COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/TestComputePipelineAnalysis.py")
    set_tests_properties(MetallicComputePipelineAnalysisTests PROPERTIES LABELS "tools;shader;perf" TIMEOUT 30)
    add_test(NAME MetallicComputePipelineExperimentsTests
        COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/TestComputePipelineExperiments.py")
    set_tests_properties(MetallicComputePipelineExperimentsTests PROPERTIES LABELS "tools;shader;perf" TIMEOUT 30)
    add_test(NAME MetallicComputePipelineExperimentsValidationTests
        COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/TestComputePipelineExperimentsValidation.py")
    set_tests_properties(MetallicComputePipelineExperimentsValidationTests PROPERTIES LABELS "tools;shader;perf" TIMEOUT 30)
    add_test(NAME MetallicShaderPipelineDriverToolsTests
        COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/TestShaderPipelineDriverTools.py")
    set_tests_properties(MetallicShaderPipelineDriverToolsTests PROPERTIES LABELS "tools;shader;perf" TIMEOUT 30)
    if(TARGET MetallicShaderPipelineBench)
        add_test(NAME MetallicShaderPipelineBenchTests
            COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/TestShaderPipelineBench.py"
                --executable "$<TARGET_FILE:MetallicShaderPipelineBench>")
        set_tests_properties(MetallicShaderPipelineBenchTests PROPERTIES LABELS "tools;shader;perf" TIMEOUT 180)
    endif()
    add_test(NAME MetallicPathTraceQueryEvidenceTests
        COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/TestPathTraceQueries.py")
    set_tests_properties(MetallicPathTraceQueryEvidenceTests PROPERTIES LABELS "tools;perf" TIMEOUT 30)
    add_test(NAME MetallicPrimaryGuidesSplitTests
        COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/TestPrimaryGuidesSplit.py")
    set_tests_properties(MetallicPrimaryGuidesSplitTests PROPERTIES LABELS "tools;perf" TIMEOUT 30)
    add_test(NAME MetallicClosureSchedulingTests
        COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/TestClosureScheduling.py")
    set_tests_properties(MetallicClosureSchedulingTests PROPERTIES LABELS "tools;perf" TIMEOUT 30)
    add_test(NAME MetallicMaterialBaselineTests
        COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/TestMaterialBaseline.py")
    set_tests_properties(MetallicMaterialBaselineTests PROPERTIES LABELS "tools;perf" TIMEOUT 30)
    add_test(NAME MetallicWorkControlReplayTests
        COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/TestWorkControlReplay.py")
    set_tests_properties(MetallicWorkControlReplayTests PROPERTIES LABELS "tools;perf" TIMEOUT 30)
    add_test(NAME MetallicNvPerfTests
        COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/TestNvPerf.py")
    set_tests_properties(MetallicNvPerfTests PROPERTIES LABELS "tools;perf" TIMEOUT 30)
    add_test(NAME MetallicDeepProfileTests
        COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/TestDeepProfile.py")
    set_tests_properties(MetallicDeepProfileTests PROPERTIES LABELS "tools;perf" TIMEOUT 30)
    add_test(NAME MetallicExperimentRunnerTests
        COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/TestExperimentRunner.py")
    set_tests_properties(MetallicExperimentRunnerTests PROPERTIES LABELS "tools;perf" TIMEOUT 30)
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
