# One recipe shared by the in-tree fallback and the installed dependency build.
macro(metallic_sdl_options)
    foreach(feature IN ITEMS TESTS TEST_LIBRARY EXAMPLES RENDER RENDER_D3D RENDER_D3D11
            RENDER_D3D12 RENDER_GPU RENDER_VULKAN GPU GPU_OPENXR)
        set(SDL_${feature} OFF CACHE BOOL "" FORCE)
    endforeach()
    set(SDL_VULKAN ON CACHE BOOL "" FORCE)
endmacro()

macro(metallic_tbb_options)
    foreach(feature IN ITEMS TBB_TEST TBB_EXAMPLES TBB_STRICT TBBMALLOC_BUILD
            TBBMALLOC_PROXY_BUILD TBB4PY_BUILD TBB_ENABLE_IPO)
        set(${feature} OFF CACHE BOOL "" FORCE)
    endforeach()
endmacro()

macro(metallic_openusd_options)
    foreach(feature IN ITEMS BUILD_TESTS BUILD_EXAMPLES BUILD_TUTORIALS BUILD_USD_TOOLS
            BUILD_IMAGING BUILD_USD_IMAGING BUILD_USD_VALIDATION BUILD_EXEC BUILD_USDVIEW
            ENABLE_PYTHON_SUPPORT ENABLE_GL_SUPPORT ENABLE_METAL_SUPPORT ENABLE_VULKAN_SUPPORT
            ENABLE_MATERIALX_SUPPORT)
        set(PXR_${feature} OFF CACHE BOOL "" FORCE)
    endforeach()
    set(PXR_BUILD_MONOLITHIC ON CACHE BOOL "" FORCE)
    set(PXR_PREFER_SAFETY_OVER_SPEED ON CACHE BOOL "" FORCE)
endmacro()
