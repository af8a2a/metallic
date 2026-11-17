implementing VisibilityBufferShading;

import GPUDriven;
using Metallic.GPUDriven;
import Core;
import MaterialTextureDecode;
using Metallic;
import Material;
using Metallic.Material;

import OpenPBR;
import OpenPBRClosure;
import OpenPBRTextureLUT;
import MaterialProgram;
using Metallic.OpenPBR;

static const uint kModeMeshlet = 0u;
static const uint kModePrimitive = 1u;
static const uint kModeLOD = 2u;
static const uint kModeShaded = 3u;
static const uint kModeBaseColor = 4u;
static const uint kEnvironmentModeProcedural = 0u;
static const uint kEnvironmentModeMap = 1u;
static const uint kEnvironmentModeDisabled = 2u;
static const uint kStreamMaxMeshletVertices = 128u;
static const uint kStreamMaxMeshletTriangles = 128u;
static const uint kStreamPayloadHeaderBytes = 112u;
static const uint kStreamPayloadClusterBytes = 96u;
static const uint kStreamPayloadClusterWords = kStreamPayloadClusterBytes / 4u;
static const uint kStreamPayloadAttributePosition = 1u << 0u;
static const uint kStreamPayloadAttributeNormal = 1u << 1u;
static const uint kStreamPayloadAttributeTexcoord0 = 1u << 2u;
static const uint kStreamPayloadAttributeTangent = 1u << 4u;
static const uint kStreamPayloadFormatFloat32x2 = 1u;
static const uint kStreamPayloadFormatFloat32x4 = 2u;
static const uint kStreamPayloadFormatFloat32x3 = 4u;
static const uint kStreamInvalidDeviceOffset = 0xffffffffu;
static const uint kStreamStateResident = 2u;
static const uint kStreamStateLockedFallback = 3u;
static const uint kStreamPageStateMask = 0x00000007u;
static const uint kStreamPageDeviceOffsetMask = 0xfffffff8u;
static const float kPi = 3.14159265358979323846;

struct GPUDrivenStreamDeferredBindings
{
    uint pageBuffer;
    uint activeGroupBuffer;
    uint pageTableBuffer;
    uint activeHeaderBuffer;
    uint paramsBuffer;
    uint visibleClusterBuffer;
    uint visibleRecordBase;
    uint visibleRecordCapacity;
};

struct GPUDrivenStreamDeferredActiveHeader
{
    uint activeGroupCount;
    uint activeGroupCapacity;
    uint maxActiveGroupClusters;
    uint overflowCount;
    uint frameIndex;
    uint padding0;
    uint padding1;
    uint padding2;
};

struct GPUDrivenStreamDeferredActiveGroup
{
    uint pageDeviceOffsetBytes;
    uint pageIndex;
    uint clusterCount;
    uint primitiveIndex;
    uint lodLevel;
    uint materialIndex;
    uint clusterSelectionMask;
    uint flags;
    uint instanceIndex;
    uint gpuSceneInstanceIndex;
    uint padding1;
    uint padding2;
    float4 world0;
    float4 world1;
    float4 world2;
    float4 world3;
};

struct GPUDrivenStreamDeferredPageTableEntry
{
    uint deviceOffsetAndState;
    uint lastRequestFrame;
};

struct GPUDrivenStreamDeferredParams
{
    float4 eye;
    float4 center;
    float4 upProjection;
    float4 viewport;
    float4 clipOrtho;
    float4 clearColor;
    uint debugColorMode;
    uint pageBufferBytes;
    uint drawTaskCount;
    uint frameIndex;
    uint maxGpuPageRequests;
    uint maxGpuPageUnloadRequests;
    uint activeGroupCount;
    uint maxActiveGroupClusters;
    uint sceneInstanceCount;
    uint scenePrimitiveCount;
    uint sceneLodLevelCount;
    uint scenePageCount;
    uint selectedLodLevel;
    uint enableGpuLodSelection;
    uint enableGpuUnloadRequests;
    uint sceneGroupCount;
    uint maxPrimitiveGroupCount;
    uint sceneNodeCount;
    uint traversalWorkerCount;
    uint traversalWorkCapacity;
    uint blasClusterReferenceAddressLow;
    uint blasClusterReferenceAddressHigh;
    uint blasClusterReferenceCapacity;
    uint blasBuildCapacity;
    float4 previousEye;
    float4 previousCenter;
    float4 previousUpProjection;
    float4 previousViewport;
    float4 previousClipOrtho;
    float lodPixelError;
    uint lodTopologyBuffer;
    uint lodStateBuffer;
    uint lodInstanceOffsetsOffset;
    float4 renderEye;
    float4 renderCenter;
    float4 renderUpProjection;
    float4 renderViewport;
    float4 renderClipOrtho;
};
struct GPUDrivenDeferredSurface
{
    float3 worldPosition;
    float3 geometryNormal;
    float3 normal;
    float3 tangent;
    float2 texcoord;
    float tangentSign;
    uint materialIndex;
    uint primitiveIndex;
    uint lodLevel;
    uint lodGroupIndex;
};
