#include "../../../Source/Runtime/Render/Core/SharcTraceParameters.h"
#define METALLIC_PATH_TRACE_SHARC_INLINE 1
[[vk::push_constant]] ConstantBuffer<SharcTraceParameters> gSharcTrace;
#define gPathTraceInline gSharcTrace.path
#include "ScenePathTraceInline.slang"
