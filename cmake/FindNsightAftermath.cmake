# FindNsightAftermath.cmake
#
# Try to find the NVIDIA Nsight Aftermath SDK.
# Set NSIGHT_AFTERMATH_SDK environment variable or CMake cache variable to the SDK root.
#
# IMPORTED Targets:
#   NsightAftermath::NsightAftermath
#
# Result Variables:
#   NsightAftermath_FOUND
#   NsightAftermath_INCLUDE_DIRS
#   NsightAftermath_LIBRARIES
#   NsightAftermath_DLLS

if(WIN32 OR UNIX)
    find_path(NsightAftermath_INCLUDE_DIR
        NAMES GFSDK_Aftermath.h
        PATHS "${NsightAftermath_SDK}/include"
    )

    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        find_library(NsightAftermath_LIBRARY
            NAMES GFSDK_Aftermath_Lib.x64
            PATHS "${NsightAftermath_SDK}/lib/x64"
            NO_SYSTEM_ENVIRONMENT_PATH
        )
    elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
        find_library(NsightAftermath_LIBRARY
            NAMES GFSDK_Aftermath_Lib.x86
            PATHS "${NsightAftermath_SDK}/lib/x86"
            NO_SYSTEM_ENVIRONMENT_PATH
        )
    endif()
else()
    find_path(NsightAftermath_INCLUDE_DIR
        NAMES GFSDK_Aftermath.h
        PATHS "${NsightAftermath_SDK}/include"
    )
    find_library(NsightAftermath_LIBRARY
        NAMES GFSDK_Aftermath_Lib
        PATHS "${NsightAftermath_SDK}/lib"
    )
endif()

if(NsightAftermath_LIBRARY)
    string(REPLACE ".lib" ".dll" NsightAftermath_DLLS "${NsightAftermath_LIBRARY}")
else()
    set(NsightAftermath_DLLS "")
endif()

set(NsightAftermath_LIBRARIES ${NsightAftermath_LIBRARY})
set(NsightAftermath_INCLUDE_DIRS ${NsightAftermath_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NsightAftermath
    DEFAULT_MSG
    NsightAftermath_LIBRARY NsightAftermath_INCLUDE_DIR)

mark_as_advanced(NsightAftermath_INCLUDE_DIR NsightAftermath_LIBRARY)

if(NsightAftermath_FOUND AND NOT TARGET NsightAftermath::NsightAftermath)
    add_library(NsightAftermath::NsightAftermath UNKNOWN IMPORTED)
    set_target_properties(NsightAftermath::NsightAftermath PROPERTIES
        IMPORTED_LOCATION "${NsightAftermath_LIBRARIES}"
        INTERFACE_INCLUDE_DIRECTORIES "${NsightAftermath_INCLUDE_DIRS}")
endif()

set(NsightAftermath_FOUND ${NsightAftermath_FOUND} CACHE INTERNAL "Nsight Aftermath found")
set(NsightAftermath_DLLS "${NsightAftermath_DLLS}" CACHE INTERNAL "Nsight Aftermath DLL")

if(NOT NsightAftermath_FOUND)
    message(STATUS "Nsight Aftermath SDK not found (NSIGHT_AFTERMATH_SDK=${NsightAftermath_SDK})")
endif()
