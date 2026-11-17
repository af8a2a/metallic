module VisibilityBufferShading;

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

__include "VisibilityBufferShading/Common.slang";
__include "VisibilityBufferShading/Geometry.slang";
__include "VisibilityBufferShading/StreamDecode.slang";
__include "VisibilityBufferShading/MaterialShading.slang";

// Resolve the winning triangle once per pixel. Rasterization never evaluates
// OpenPBR or exports normal/tangent/material G-buffers.
[shader("compute")]
[numthreads(8, 8, 1)]
void visibilityBufferShadeMain(
    uint3 dispatchThreadId : SV_DispatchThreadID,
    uniform GPUDrivenPreviewPush push)
{
    StructuredBuffer<GPUDrivenPreviewParams> paramsHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenPreviewParams>>(push.paramsBuffer));
    GPUDrivenPreviewParams params = paramsHandle[0];
    if (dispatchThreadId.x >= params.width || dispatchThreadId.y >= params.height) {
        return;
    }

    Texture2D<uint> visibilityHandle =
        resolveUniform(ResourceHandle<Texture2D<uint>>(push.visibilityImage));
    RWStructuredBuffer<uint> colorHandle =
        resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.deferredColorBuffer));
    Texture2D<uint> visibility = visibilityHandle;
    RWStructuredBuffer<uint> colors = colorHandle;
    uint packedVisibility = visibility.Load(int3(dispatchThreadId.xy, 0));
    uint outputIndex = dispatchThreadId.y * params.width + dispatchThreadId.x;
    float2 pixelCenter = float2(dispatchThreadId.xy) + 0.5;
    uint background = packColor(params.clearColor.xyz);
    if (packedVisibility == 0u) {
        if (params.mode == kModeShaded && params.environmentVisible != 0u) {
            background = packHdrColor(environmentRadiance(
                cameraRayDirection(pixelCenter, params), params, push));
        }
        colors[outputIndex] = background;
        return;
    }

    uint recordIndex = 0u;
    uint triangleIndex = 0u;
    if (!unpackVisibilityId(packedVisibility, recordIndex, triangleIndex)) {
        colors[outputIndex] = background;
        return;
    }

    GPUDrivenDeferredSurface surface;
    bool decoded = decodeStreamVisibilitySurface(
        recordIndex, triangleIndex, pixelCenter, params, push, surface);
    if (!decoded) {
        colors[outputIndex] = background;
        return;
    }

    if (params.mode != kModeShaded && params.mode != kModeBaseColor) {
        uint colorSeed = recordIndex;
        if (params.mode == kModePrimitive) {
            colorSeed = surface.primitiveIndex;
        } else if (params.mode == kModeLOD) {
            colorSeed = surface.primitiveIndex * 1009u +
                surface.lodLevel * 131u + surface.lodGroupIndex * 17u + 1u;
        }
        colors[outputIndex] = packColor(hashColor(colorSeed + 1u));
        return;
    }
    colors[outputIndex] = shadeDeferredSurface(surface, pixelCenter, params, push);
}
