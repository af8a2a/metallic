# CMake's Ninja generator matches /showIncludes output byte-for-byte. Some
# localized MSVC installations emit the prefix in the active OEM code page,
# which can leave Ninja with an empty dependency list when configure and build
# use different code pages. Decode compiler output as OEM and emit a stable
# ASCII prefix while preserving the compiler's exit status.

if(CMAKE_ARGC LESS 5)
    message(FATAL_ERROR "RunMsvcCompiler.cmake did not receive a compiler command")
endif()

math(EXPR compilerArgumentLast "${CMAKE_ARGC} - 1")
set(compilerCommand)
foreach(argumentIndex RANGE 4 ${compilerArgumentLast})
    list(APPEND compilerCommand "${CMAKE_ARGV${argumentIndex}}")
endforeach()

execute_process(
    COMMAND ${compilerCommand}
    RESULT_VARIABLE compilerResult
    OUTPUT_VARIABLE compilerOutput
    ERROR_VARIABLE compilerError
    ENCODING OEM
)

string(REPLACE
    "注意: 包含文件:"
    "Note: including file:"
    compilerOutput
    "${compilerOutput}"
)
# Under Ninja, recent MSVC builds may switch redirected output to UTF-8. If
# that stream is reported as OEM, CMake exposes this deterministic mojibake.
string(REPLACE
    "娉ㄦ剰: 鍖呭惈鏂囦欢:"
    "Note: including file:"
    compilerOutput
    "${compilerOutput}"
)
string(REPLACE
    "注意: 包含文件:"
    "Note: including file:"
    compilerError
    "${compilerError}"
)
string(REPLACE
    "娉ㄦ剰: 鍖呭惈鏂囦欢:"
    "Note: including file:"
    compilerError
    "${compilerError}"
)

if(NOT compilerOutput STREQUAL "")
    message("${compilerOutput}")
endif()
if(NOT compilerError STREQUAL "")
    message("${compilerError}")
endif()
if(NOT compilerResult EQUAL 0)
    message(FATAL_ERROR "MSVC compiler exited with code ${compilerResult}")
endif()
