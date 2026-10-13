module SceneMaterialReadProbe;

import Core;
import Material;
import RTXCRMaterialParameters;
import SceneSurface;

using Metallic;
using Metallic.Material;
using Metallic.SceneSurface;

void writeMaterialReadProbe(RWByteAddressBuffer output, uint base,
    SceneMaterialReadHeader header, uint domain, RTXCRHairMaterialParameters fiber,
    MaterialCoverageParameters coverage, float scalarProgramId)
{
    output.Store2(base, asuint(float2(header.programId, header.alphaMode)));
    output.Store(base + 8u, domain);
    output.Store4(base + 12u, asuint(fiber.baseColor));
    output.Store4(base + 28u, asuint(fiber.roughnessIOR));
    output.Store4(base + 44u, asuint(fiber.melanin));
    output.Store4(base + 60u, asuint(fiber.diffuseTint));
    output.Store3(base + 76u, asuint(float3(coverage.baseAlpha, coverage.cutoff, coverage.alphaMode)));
    output.Store(base + 88u, coverage.hasProgram ? 1u : 0u);
    output.Store4(base + 92u, uint4(coverage.baseColorTexture.textureIndex,
        coverage.baseColorTexture.texCoord, coverage.baseColorTexture.alignmentPadding));
    output.Store4(base + 108u, asuint(coverage.baseColorTexture.transform0));
    output.Store4(base + 124u, asuint(coverage.baseColorTexture.transform1));
    output.Store(base + 140u, asuint(scalarProgramId));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 thread : SV_DispatchThreadID)
{
    if (thread.x >= 2u) { return; }
    let output = resolveBuffer<RWByteAddressBuffer>(getConstants<ResourceHandle<ByteAddressBuffer>>());
    let original = loadSceneMaterial(thread.x);
    SceneMaterialReadHeader originalHeader;
    originalHeader.programId = original.textureParams.z;
    originalHeader.alphaMode = original.textureParams.w;
    writeMaterialReadProbe(output, thread.x * 144u, originalHeader, materialDomain(original),
        sceneFiberParameters(original), sceneMaterialCoverageParameters(original), original.textureParams.z);

    let header = loadSceneMaterialReadHeader(thread.x);
    writeMaterialReadProbe(output, (thread.x + 2u) * 144u, header,
        sceneMaterialReadDomain(header.programId), loadSceneMaterialReadFiber(thread.x),
        loadSceneMaterialReadCoverage(thread.x, header.alphaMode), loadSceneMaterialReadProgramId(thread.x));
}
