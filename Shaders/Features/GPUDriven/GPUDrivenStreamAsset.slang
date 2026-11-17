#language slang 2026
module GPUDrivenStreamAsset;

import ShaderCore;
using Metallic;

import GPUDriven;
using Metallic.GPUDriven;
import Material;
using Metallic.Material;
import WaveWorkDistribution;
import AlphaCoverage;
#include "../../Interop/MeshletStreamBLAS.h"
// Canonical interop macros stay in this owner; implementation units use the typed constant.
static const uint kStreamBLASHeaderWords = METALLIC_STREAM_BLAS_HEADER_WORDS;
import VisibilityTessellation;

// Keep program identity and registered entry points stable; implementations are grouped by responsibility.
__include "GPUDrivenStreamAsset/Types.slang";
__include "GPUDrivenStreamAsset/CameraCulling.slang";
__include "GPUDrivenStreamAsset/PageRequests.slang";
__include "GPUDrivenStreamAsset/TraversalSelection.slang";
__include "GPUDrivenStreamAsset/Traversal.slang";
__include "GPUDrivenStreamAsset/LODFrontier.slang";
__include "GPUDrivenStreamAsset/DistributedDemand.slang";
__include "GPUDrivenStreamAsset/CooperativeLOD.slang";
__include "GPUDrivenStreamAsset/ActiveGroupBuild.slang";
__include "GPUDrivenStreamAsset/BLASCommon.slang";
__include "GPUDrivenStreamAsset/RTRepresentations.slang";
__include "GPUDrivenStreamAsset/BLASInput.slang";
__include "GPUDrivenStreamAsset/RasterCluster.slang";
__include "GPUDrivenStreamAsset/HardwareRaster.slang";
__include "GPUDrivenStreamAsset/ClusterCulling.slang";
__include "GPUDrivenStreamAsset/SoftwareRaster.slang";
__include "GPUDrivenStreamAsset/Deferred.slang";

// Existing implementations share the same module and streaming contracts.
__include "StreamClusterClassify.slang";
__include "StreamRasterCooperative.slang";
__include "../VisibilityBuffer/VisibilityTessellationStream.slang";
