option(METALLIC_ENABLE_AFTERMATH
    "Enable Nsight Aftermath GPU crash dump tracking when the SDK is available"
    ON)

if(NOT TARGET metallic_aftermath)
    add_library(metallic_aftermath INTERFACE)
    add_library(metallic::aftermath ALIAS metallic_aftermath)
endif()

set(METALLIC_HAS_AFTERMATH OFF)

if(NOT METALLIC_ENABLE_AFTERMATH)
    message(STATUS "Nsight Aftermath disabled via METALLIC_ENABLE_AFTERMATH=OFF")
    return()
endif()

if(NOT WIN32)
    message(STATUS "Nsight Aftermath is only supported on Windows")
    return()
endif()

set(NsightAftermath_SDK "" CACHE PATH
    "Path to the Nsight Aftermath SDK root directory")

if(NOT NsightAftermath_SDK AND DEFINED ENV{NSIGHT_AFTERMATH_SDK})
    set(NsightAftermath_SDK "$ENV{NSIGHT_AFTERMATH_SDK}")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
find_package(NsightAftermath)

if(NsightAftermath_FOUND)
    target_link_libraries(metallic_aftermath INTERFACE NsightAftermath::NsightAftermath)
    target_compile_definitions(metallic_aftermath INTERFACE METALLIC_HAS_AFTERMATH=1)
    set(METALLIC_HAS_AFTERMATH ON)
    set(METALLIC_AFTERMATH_DLL "${NsightAftermath_DLLS}" CACHE INTERNAL "")
    message(STATUS "Nsight Aftermath enabled: ${NsightAftermath_INCLUDE_DIRS}")
else()
    message(STATUS "Nsight Aftermath SDK not found. Aftermath crash tracking will be disabled.")
endif()
