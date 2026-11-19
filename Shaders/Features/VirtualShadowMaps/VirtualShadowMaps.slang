#language slang 2026
module VirtualShadowMaps;

#if VSM_STATIC_PAGE_HZB
__include "VirtualShadowMapStaticHZB.slang";
#endif

#if VSM_PROPAGATED_MAPPING_PROBE
__include "../../../tests/rhi/shaders/VirtualShadowMapPropagatedMappingProbe.slang";
#endif

import ShaderCore;
import Material;
import MaterialProgram;
import VirtualShadowMapGeometry;
import VirtualShadowMapReceiver;
import VirtualShadowMapCommon;
import Lighting;

#include "../../../Source/Runtime/Render/Shadow/VirtualShadowMapProfiling.h"
#include "../../../Source/Runtime/Render/Shadow/VirtualShadowMapCacheDiagnostics.h"
#include "../../../Source/Runtime/Render/Shadow/VirtualShadowMapProjectionGrid.h"
#include "../../../Source/Runtime/Render/Shadow/VirtualShadowMapPhysicalPoolInterop.h"
#include "../../../Source/Runtime/Render/Shadow/VirtualShadowMapBatchInterop.h"
#include "../../../Source/Runtime/Render/Shadow/VirtualShadowMapLODInterop.h"

// Incompatible diagnostic permutations are rejected at the module boundary.
#ifndef VSM_DIRECTIONAL_MARK
#define VSM_DIRECTIONAL_MARK 0
#endif
#ifndef VSM_RECEIVER_COVERAGE_DIAGNOSTICS
#define VSM_RECEIVER_COVERAGE_DIAGNOSTICS 0
#endif
#if VSM_DIRECTIONAL_MARK && VSM_RECEIVER_COVERAGE_DIAGNOSTICS
#error The full-cone Mark candidate does not support the legacy per-ray receiver coverage records.
#endif

using Metallic;
using Metallic.Material;
using Metallic.Shadow;
using Metallic.Lighting;

// Keep pass entry points and shared implementation under one stable module.
__include "VirtualShadowMaps/ParametersAndDiagnostics.slang";
__include "VirtualShadowMaps/BatchExecution.slang";
__include "VirtualShadowMaps/ProjectionGeometry.slang";
__include "VirtualShadowMaps/ShadowViewLOD.slang";
__include "VirtualShadowMaps/ReceiverReconstruction.slang";
__include "VirtualShadowMaps/ShadowSampling.slang";
__include "VirtualShadowMaps/PageLookup.slang";
__include "VirtualShadowMaps/DirectionalRayLookup.slang";
__include "VirtualShadowMaps/ShadowFiltering.slang";
__include "VirtualShadowMaps/PageMarking.slang";
__include "VirtualShadowMaps/PageReset.slang";
__include "VirtualShadowMaps/ReceiverMarking.slang";
__include "VirtualShadowMaps/PageAllocation.slang";
__include "VirtualShadowMaps/SharedPhysicalPool.slang";
__include "VirtualShadowMaps/ClusterCulling.slang";
__include "VirtualShadowMaps/BuildWork.slang";
__include "VirtualShadowMaps/PageFinalize.slang";
__include "VirtualShadowMaps/ShadowRaster.slang";
#if VSM_TWO_PASS_HZB
__include "VirtualShadowMaps/RasterHZB.slang";
#endif
#if VSM_HIERARCHY_HZB && VSM_INSTANCE_BUILD && VSM_COMPACT_BUILD
__include "VirtualShadowMaps/HierarchyHZB.slang";
#endif
#if VSM_CLUSTER_HYBRID_RASTER || VSM_TWO_PASS_HZB
__include "VirtualShadowMaps/HybridRaster.slang";
#endif
__include "VirtualShadowMaps/ShadowProjection.slang";
