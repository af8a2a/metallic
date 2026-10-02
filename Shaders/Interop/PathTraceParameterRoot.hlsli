#pragma once
#ifndef METALLIC_PATH_TRACE_TYPED
#define METALLIC_PATH_TRACE_TYPED 1
#endif
#include "../../Source/Runtime/Render/Core/PathTraceParameters.h"
import ParameterRoot;
#define gPathTraceParameters getParameters<PathTraceParameters>()
#define METALLIC_LOAD_MATERIAL_VALUE(index) resolveDescriptor(gPathTraceParameters.materialValues)[index]
