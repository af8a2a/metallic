#language slang 2026
implementing SceneRTXDI;

import Core;
using Metallic;

#define gPositionPrevious resolveUniform(getResourceParameters<SceneResourceParameters>().positionPrevious)
#define gNormalPrevious resolveUniform(getResourceParameters<SceneResourceParameters>().normalPrevious)
#define gEnvironmentMap resolveUniform(getResourceParameters<SceneResourceParameters>().environmentMap)

DIReservoir emptyReservoir()
{
    DIReservoir reservoir;
    reservoir.lightIndex = 0xffffffffu;
    reservoir.weightSum = 0.0;
    reservoir.selectedTarget = 0.0;
    reservoir.M = 0u;
    return reservoir;
}

bool reservoirValid(DIReservoir reservoir, SceneRTXDIPush push)
{
    uint environmentWidth = 0u;
    uint environmentHeight = 0u;
    gEnvironmentMap.GetDimensions(environmentWidth, environmentHeight);
    bool sampleIndexValid = isEnvironmentSample(reservoir.lightIndex)
        ? behaviorEnabled(push, kBehaviorEnvironmentEnabled) &&
            environmentTexelIndex(reservoir.lightIndex) < environmentWidth * environmentHeight
        : reservoir.lightIndex < push.lightCount;
    return sampleIndexValid &&
        reservoir.M > 0u &&
        reservoir.weightSum > 0.0 &&
        reservoir.selectedTarget > 0.0 &&
        isfinite(reservoir.weightSum) &&
        isfinite(reservoir.selectedTarget);
}

void updateReservoir(
    inout DIReservoir reservoir,
    uint lightIndex,
    float candidateWeight,
    float candidateTarget,
    uint candidateM,
    float randomValue)
{
    if (candidateM == 0u) {
        return;
    }
    reservoir.M += candidateM;
    if (candidateWeight <= 0.0 || candidateTarget <= 0.0 ||
        !isfinite(candidateWeight) || !isfinite(candidateTarget)) {
        return;
    }
    reservoir.weightSum += candidateWeight;
    if (randomValue * reservoir.weightSum < candidateWeight) {
        reservoir.lightIndex = lightIndex;
        reservoir.selectedTarget = candidateTarget;
    }
}

void clampReservoirHistory(inout DIReservoir reservoir, uint maximumM)
{
    if (reservoir.M > maximumM && maximumM > 0u) {
        float scale = float(maximumM) / float(reservoir.M);
        reservoir.weightSum *= scale;
        reservoir.M = maximumM;
    }
}

void combineReservoir(
    inout DIReservoir reservoir,
    DIReservoir candidate,
    float currentTarget,
    uint maximumM,
    float randomValue,
    SceneRTXDIPush push)
{
    if (!reservoirValid(candidate, push) || currentTarget <= 0.0) {
        return;
    }
    uint candidateM = min(candidate.M, maximumM);
    float historyScale = float(candidateM) / float(max(candidate.M, 1u));
    float candidateWeight = currentTarget *
        candidate.weightSum /
        max(candidate.selectedTarget, 1e-30) *
        historyScale;
    updateReservoir(
        reservoir,
        candidate.lightIndex,
        candidateWeight,
        currentTarget,
        candidateM,
        randomValue);
    clampReservoirHistory(reservoir, maximumM);
}

uint4 packReservoir(DIReservoir reservoir)
{
    return uint4(
        reservoir.lightIndex,
        asuint(reservoir.weightSum),
        asuint(reservoir.selectedTarget),
        reservoir.M);
}

DIReservoir unpackReservoir(uint4 packed)
{
    DIReservoir reservoir;
    reservoir.lightIndex = packed.x;
    reservoir.weightSum = asfloat(packed.y);
    reservoir.selectedTarget = asfloat(packed.z);
    reservoir.M = packed.w;
    return reservoir;
}

bool previousSurfaceMatches(
    int2 pixel,
    Surface surface,
    float positionThreshold,
    SceneRTXDIPush push)
{
    if (pixel.x < 0 || pixel.y < 0 || pixel.x >= int(push.width) || pixel.y >= int(push.height)) {
        return false;
    }
    float4 previousPosition = gPositionPrevious[pixel];
    float4 previousNormal = gNormalPrevious[pixel];
    if (previousPosition.w < 0.5 || previousNormal.w < 0.5) {
        return false;
    }
    return distance(previousPosition.xyz, surface.position) <= positionThreshold &&
        dot(safeNormalize(previousNormal.xyz, surface.normal), surface.normal) >= push.normalThreshold;
}
