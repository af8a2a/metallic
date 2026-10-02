#include "../../../Source/Runtime/Render/Core/PathTraceInlineParameters.h"
#define METALLIC_PATH_TRACE_TYPED 1
#define METALLIC_PATH_TRACE_INLINE 1
[[vk::push_constant]] ConstantBuffer<PathTraceInlineParameters> gPathTraceInline;
PathTraceParameters inlinePathTraceResources()
{
    PathTraceParameters resources = *gPathTraceInline.resources;
    resources.settings = gPathTraceInline.settings;
    resources.output = gPathTraceInline.output;
    resources.historyCurrent = gPathTraceInline.historyCurrent;
    resources.historyPrevious = gPathTraceInline.historyPrevious;
    return resources;
}
#define gPathTraceParameters inlinePathTraceResources()
#define METALLIC_LOAD_MATERIAL_VALUE(index) resolveDescriptor(gPathTraceParameters.materialValues)[index]
#include "ScenePathTrace.slang"
