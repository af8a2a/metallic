module GlobalDistanceFieldCompose;

import ShaderCore;
import ParameterRoot;
import MeshDistanceField;
import GlobalDistanceField;
using Metallic;
using Metallic.DistanceField;

struct GlobalDistanceFieldParams
{
    ResourceHandle<StructuredBuffer<MeshDistanceFieldInstance>> instances;
    ResourceHandle<StructuredBuffer<MeshDistanceFieldMip>> mips;
    ResourceHandle<StructuredBuffer<uint>> indirection;
    ResourceHandle<Texture3D<float>> atlas;
    SamplerHandle atlasSampler;
    ResourceHandle<RWTexture3D<float>> distanceField;
    ResourceHandle<RWStructuredBuffer<GlobalDistanceFieldDomain>> domainOutput;
    uint instanceCount;
    GlobalDistanceFieldDomain domain;
};

[shader("compute")]
[numthreads(4, 4, 4)]
void globalDistanceFieldMain(uint3 id : SV_DispatchThreadID)
{
    GlobalDistanceFieldParams params = getParameters<GlobalDistanceFieldParams>();
    RWTexture3D<float> distanceField = resolveUniform(params.distanceField);
    uint3 dimensions;
    distanceField.GetDimensions(dimensions.x, dimensions.y, dimensions.z);
    if (any(id >= dimensions)) {
        return;
    }
    if (all(id == 0u)) {
        RWStructuredBuffer<GlobalDistanceFieldDomain> domainOutput = resolveUniform(params.domainOutput);
        domainOutput[0] = params.domain;
    }

    float influenceRadius = params.domain.maxAndInfluenceRadius.w;
    float distance = influenceRadius;
    if (params.instanceCount != 0u) {
        Texture3D<float> atlas = resolveUniform(params.atlas);
        uint atlasWidth, atlasHeight, atlasDepth;
        atlas.GetDimensions(atlasWidth, atlasHeight, atlasDepth);
        MeshDistanceFieldResources meshes = {resolveUniform(params.mips), resolveUniform(params.indirection),
            atlas, resolveUniform(params.atlasSampler), atlasWidth / 8u};
        StructuredBuffer<MeshDistanceFieldInstance> instances = resolveUniform(params.instances);
        float3 worldPosition = globalDistanceFieldVoxelCenter(params.domain, id);
        distance = compositeGlobalDistanceField(meshes, instances, params.instanceCount, worldPosition, influenceRadius);
    }
    distanceField[id] = distance;
}
