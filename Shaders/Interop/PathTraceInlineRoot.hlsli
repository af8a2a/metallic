#pragma once
#define METALLIC_PATH_TRACE_TYPED 1
#if defined(METALLIC_PATH_TRACE_GUIDES_INLINE)
#include "../../Source/Runtime/Render/Core/PathTraceGuidesInlineParameters.h"
[[vk::push_constant]] ConstantBuffer<PathTraceGuidesInlineParameters> gPathTraceGuidesInline;
#define gPathTraceInline gPathTraceGuidesInline.path
#else
#include "../../Source/Runtime/Render/Core/PathTraceInlineParameters.h"
#ifndef gPathTraceInline
[[vk::push_constant]] ConstantBuffer<PathTraceInlineParameters> gPathTraceInline;
#endif
#endif

PathTraceParameters inlinePathTraceResources()
{
    PathTraceParameters resources = *gPathTraceInline.resources;
    resources.settings = gPathTraceInline.settings;
    resources.output = gPathTraceInline.output;
    resources.historyCurrent = gPathTraceInline.historyCurrent;
    resources.historyPrevious = gPathTraceInline.historyPrevious;
#if defined(METALLIC_PATH_TRACE_GUIDES_INLINE)
    resources.albedo = gPathTraceGuidesInline.albedo;
    resources.specularAlbedo = gPathTraceGuidesInline.specularAlbedo;
    resources.normalRoughness = gPathTraceGuidesInline.normalRoughness;
    resources.motionVectors = gPathTraceGuidesInline.motionVectors;
    resources.linearDepth = gPathTraceGuidesInline.linearDepth;
    resources.specularHitDistance = gPathTraceGuidesInline.specularHitDistance;
    resources.depth = gPathTraceGuidesInline.depth;
#endif
    return resources;
}
#define gPathTraceParameters inlinePathTraceResources()
#define METALLIC_LOAD_MATERIAL_VALUE(index) resolveDescriptor(gPathTraceParameters.materialValues)[index]
