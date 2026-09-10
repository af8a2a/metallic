# Loaded in each dependency's project(), after command-line cache arguments.
include("${CMAKE_CURRENT_LIST_DIR}/../DependencyOptions.cmake")
if(MSVC AND CMAKE_GENERATOR MATCHES "Ninja")
    # Each standalone child detects its own localized prefix. Match the wrapper
    # passed through InitialCache.cmake, just as the application does.
    set(CMAKE_C_CL_SHOWINCLUDES_PREFIX "Note: including file:  ")
    set(CMAKE_CXX_CL_SHOWINCLUDES_PREFIX "Note: including file:  ")
    set(CMAKE_CL_SHOWINCLUDES_PREFIX "Note: including file:  ")
    # SDL enables CXX later in its CMakeLists, which reloads the detected prefix.
    # Set it once more after all enable_language calls, before generation.
    cmake_language(DEFER CALL set CMAKE_C_CL_SHOWINCLUDES_PREFIX "Note: including file:  ")
    cmake_language(DEFER CALL set CMAKE_CXX_CL_SHOWINCLUDES_PREFIX "Note: including file:  ")
    cmake_language(DEFER CALL set CMAKE_CL_SHOWINCLUDES_PREFIX "Note: including file:  ")
endif()
if(PROJECT_NAME STREQUAL "SDL3")
    metallic_sdl_options()
elseif(PROJECT_NAME STREQUAL "TBB")
    metallic_tbb_options()
elseif(PROJECT_NAME STREQUAL "usd")
    metallic_openusd_options()
endif()
