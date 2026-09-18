include(FetchContent)
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()
# NVIDIA's CPU reference codec. Pin both revision and archive contents; callers
# may set FETCHCONTENT_SOURCE_DIR_METALLIC_GDEFLATE for an offline checkout.
FetchContent_Declare(metallic_gdeflate
    URL https://codeload.github.com/NVIDIA/libdeflate/tar.gz/8ba9502fb30d2bf728592d121f0d402e40c8cb05
    URL_HASH SHA256=d1b4c38dce43e68a5f4c28d0fbb3f81a01953039a3dea63f4bd1a84d7ff80592
)
FetchContent_MakeAvailable(metallic_gdeflate)
configure_file(${metallic_gdeflate_SOURCE_DIR}/COPYING
    ${CMAKE_BINARY_DIR}/licenses/GDeflate-COPYING.txt COPYONLY)
add_library(MetallicGDeflate STATIC
    ${metallic_gdeflate_SOURCE_DIR}/lib/deflate_compress.c
    ${metallic_gdeflate_SOURCE_DIR}/lib/gdeflate_compress.c
    ${metallic_gdeflate_SOURCE_DIR}/lib/gdeflate_decompress.c
    ${metallic_gdeflate_SOURCE_DIR}/lib/utils.c
    ${metallic_gdeflate_SOURCE_DIR}/lib/x86/cpu_features.c
    ${metallic_gdeflate_SOURCE_DIR}/lib/arm/cpu_features.c
)
target_include_directories(MetallicGDeflate PUBLIC ${metallic_gdeflate_SOURCE_DIR})
set_target_properties(MetallicGDeflate PROPERTIES POSITION_INDEPENDENT_CODE ON)
