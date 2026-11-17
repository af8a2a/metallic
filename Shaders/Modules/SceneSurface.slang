#language slang 2026
module SceneSurface;

#if METALLIC_CUSTOM_MATERIALS
__include "MaterialValueDispatch.slang";
#endif

__include "SceneSurface/SceneResources.slang";
__include "SceneSurface/EnvironmentSampling.slang";
__include "SceneSurface/MaterialTextureSampling.slang";
__include "SceneSurface/SurfaceMath.slang";
