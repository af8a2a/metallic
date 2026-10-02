#include "../../../Source/Runtime/Render/Core/NRCTraceParameters.h"
#define METALLIC_PATH_TRACE_NRC_INLINE 1
[[vk::push_constant]] ConstantBuffer<NRCTraceParameters> gNRCTrace;
#define gPathTraceInline gNRCTrace.path
#include "ScenePathTraceInline.slang"
