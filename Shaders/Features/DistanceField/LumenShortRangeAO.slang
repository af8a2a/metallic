module LumenShortRangeAO;

import ShaderCore;
import ParameterRoot;
import ScenePathMath;
import LumenProbeMath;
import LumenAdaptiveProbes;
import LumenShortRangeAOPacking;

using Metallic;
using Metallic.Lumen;

typealias PackedBentNormalTexture = _Texture<uint, __Shape2D, 0, 0, 0, 1, 0, 0, 37>;

struct LumenShortRangeAOParams
{
    ResourceHandle<Texture2D<float>> depth;
    ResourceHandle<Texture2D<float4>> normal;
    ResourceHandle<Texture2D<float2>> blueNoiseVec2;
    ResourceHandle<StructuredBuffer<ViewConstants>> view;
    ResourceHandle<PackedBentNormalTexture> shortRangeAORaw;
};

struct ShortRangeAOProjection
{
    uint2 ViewSize;
    float2 InvViewSize;
    float2 ScreenToViewScale;
    float2 ScreenToViewBias;
    float NearPlane;
    float FarPlane;
    bool bOrthographic;
};

float convertFromDeviceZ(ShortRangeAOProjection Projection, float DeviceZ)
{
    return Projection.bOrthographic ? lerp(Projection.FarPlane, Projection.NearPlane, DeviceZ)
        : Projection.NearPlane * Projection.FarPlane /
            (Projection.NearPlane + DeviceZ * (Projection.FarPlane - Projection.NearPlane));
}

// UE ScreenToViewPos with the zero-origin viewport and Metallic's camera projection parameters.
float3 screenToViewPos(ShortRangeAOProjection Projection, float2 ScreenUV, float SceneDepth)
{
    float2 ProjViewPos = ScreenUV * Projection.ScreenToViewScale + Projection.ScreenToViewBias;
    return float3(ProjViewPos * (Projection.bOrthographic ? 1.0f : SceneDepth), SceneDepth);
}

// UE FastMathThirdParty.ush acosFast.
float acosFast(float inX)
{
    float x = abs(inX);
    float res = -0.156583f * x + (0.5f * kLumenProbePi);
    res *= sqrt(1.0f - x);
    return (inX >= 0) ? res : kLumenProbePi - res;
}

// UE LumenScreenSpaceBentNormal.usf BuildRotationMatrix.
float3x3 buildRotationMatrix(float3 FromDirection, float3 ToDirection)
{
    const float e = dot(FromDirection, ToDirection);
    const float f = abs(e);

    if (f > float(1.0f - 0.0003f)) {
        return float3x3(1, 0, 0, 0, 1, 0, 0, 0, 1);
    }

    const float3 v = cross(FromDirection, ToDirection);
    const float h = (1.0f) / (1.0f + e);
    const float hvx = h * v.x;
    const float hvz = h * v.z;
    const float hvxy = hvx * v.y;
    const float hvxz = hvx * v.z;
    const float hvyz = hvz * v.y;

    float3x3 mtx;
    mtx[0][0] = e + hvx * v.x;
    mtx[0][1] = hvxy - v.z;
    mtx[0][2] = hvxz + v.y;

    mtx[1][0] = hvxy + v.z;
    mtx[1][1] = e + h * v.y * v.y;
    mtx[1][2] = hvyz - v.x;

    mtx[2][0] = hvxz - v.y;
    mtx[2][1] = hvyz + v.x;
    mtx[2][2] = e + hvz * v.z;
    return mtx;
}

// UE 5.7.4 Epic no-HZB, opaque horizon permutation with Short Range GI disabled.
void updateOccludedHorizonForStep(Texture2D<float> Depth, ShortRangeAOProjection Projection,
    float2 SampleScreenPos, float3 ViewSpacePosition, float3 ViewVector, float InvForegroundDistance,
    float LowHorizonCos, inout float HorizonCos)
{
    float2 SampleScreenPosUV = SampleScreenPos * float2(0.5f, -0.5f) + 0.5f;
    uint2 SamplePixel = min(uint2(SampleScreenPosUV * float2(Projection.ViewSize)), Projection.ViewSize - 1u);
    float SampleDepth = convertFromDeviceZ(Projection, Depth.Load(int3(SamplePixel, 0)));
    float3 SampleViewPos = screenToViewPos(Projection, SampleScreenPosUV, SampleDepth);
    float3 SampleDelta = SampleViewPos - ViewSpacePosition;

    float SampleDist = length(SampleDelta);
    float3 ToSample = SampleDelta / SampleDist;
    float NewHorizonCos = dot(ToSample, ViewVector);

    const float ForegroundSampleRejectPower = 1.0f;
    NewHorizonCos = lerp(NewHorizonCos, LowHorizonCos,
        saturate(pow(abs(SampleDelta.z) * InvForegroundDistance, ForegroundSampleRejectPower)));
    HorizonCos = max(HorizonCos, NewHorizonCos);
}

float3 calculateAOHorizonSearch(Texture2D<float> Depth, ShortRangeAOProjection Projection,
    uint2 SvPosition, float SceneDepth, float3 WorldNormal, float3x3 WorldToView, float2 UniformRandom)
{
    const float2 ScreenUV = (SvPosition + 0.5f) * Projection.InvViewSize;
    float2 ScreenPos = (ScreenUV - 0.5f) / float2(0.5f, -0.5f);

    const float ForegroundSampleRejectDistanceFraction = 0.3f;
    float InvForegroundDistance = 1.0f / (SceneDepth * ForegroundSampleRejectDistanceFraction);
    float3 ViewSpacePosition = screenToViewPos(Projection, ScreenUV, SceneDepth);
    float DistanceToCamera = length(ViewSpacePosition);
    float3 ViewVector = -ViewSpacePosition / DistanceToCamera;
    float3 ViewNormal = mul(WorldToView, WorldNormal);

    float Visibility = 0;
    float3 ViewBentNormal = 0;
    const float EffectiveSliceCount = 2;
    const float StepsPerSliceForAO = 3;
    const float2 MaxScreenFractionForAO = (float(kLumenProbeDownsampleFactor) * 2.0f) /
        float2(Projection.ViewSize);

    for (float SliceIndex = 0; SliceIndex < EffectiveSliceCount; SliceIndex++) {
        float SliceFraction = (SliceIndex + UniformRandom.x) / EffectiveSliceCount;
        float Phi = SliceFraction * kLumenProbePi;
        float CosPhi = cos(Phi);
        float SinPhi = sin(Phi);
        float3 SliceDirection = float3(CosPhi, SinPhi, 0);

        float3 OrthogonalSliceDirection = SliceDirection - (dot(SliceDirection, ViewVector) * ViewVector);
        float3 AxisVector = normalize(cross(OrthogonalSliceDirection, ViewVector));
        float3 ProjectedNormal = ViewNormal - AxisVector * dot(ViewNormal, AxisVector);
        float SignNorm = sign(dot(OrthogonalSliceDirection, ProjectedNormal));
        float ProjectedNormalLength = length(ProjectedNormal);
        float CosNormal = (float)saturate(dot(ProjectedNormal, ViewVector) / ProjectedNormalLength);
        float NormalAngle = SignNorm * acosFast(CosNormal);

        float LowHorizonCos0 = cos(NormalAngle + .5f * kLumenProbePi);
        float LowHorizonCos1 = cos(NormalAngle - .5f * kLumenProbePi);
        float HorizonCos0 = LowHorizonCos0;
        float HorizonCos1 = LowHorizonCos1;

        for (float StepIndex = 0; StepIndex < StepsPerSliceForAO; StepIndex++) {
            float StepBaseNoise = float(SliceIndex + StepIndex * StepsPerSliceForAO) * 0.6180339887498948482f;
            float StepNoise = frac(UniformRandom.y + StepBaseNoise);
            float StepFraction = (StepIndex + StepNoise) / StepsPerSliceForAO;
            float2 StepRadius = StepFraction * StepFraction * MaxScreenFractionForAO;
            float2 SampleOffset = 2.0f * (StepRadius + Projection.InvViewSize) * SliceDirection.xy;

            float2 SampleScreenPos0 = ScreenPos + SampleOffset;
            if (all(SampleScreenPos0 > -1) && all(SampleScreenPos0 < 1)) {
                updateOccludedHorizonForStep(Depth, Projection, SampleScreenPos0, ViewSpacePosition, ViewVector,
                    InvForegroundDistance, LowHorizonCos0, HorizonCos0);
            }

            float2 SampleScreenPos1 = ScreenPos - SampleOffset;
            if (all(SampleScreenPos1 > -1) && all(SampleScreenPos1 < 1)) {
                updateOccludedHorizonForStep(Depth, Projection, SampleScreenPos1, ViewSpacePosition, ViewVector,
                    InvForegroundDistance, LowHorizonCos1, HorizonCos1);
            }
        }

        float H0 = -acosFast(HorizonCos1);
        float H1 = acosFast(HorizonCos0);

        float IntegralArc0 = (CosNormal + 2 * H0 * sin(NormalAngle) - cos(2 * H0 - NormalAngle)) / 4.0f;
        float IntegralArc1 = (CosNormal + 2 * H1 * sin(NormalAngle) - cos(2 * H1 - NormalAngle)) / 4.0f;
        float SliceVisibility = ProjectedNormalLength * (IntegralArc0 + IntegralArc1);

        float t0 = (6 * sin(H0 - NormalAngle) - sin(3 * H0 - NormalAngle) + 6 * sin(H1 - NormalAngle) -
            sin(3 * H1 - NormalAngle) + 16 * sin(NormalAngle) - 3 * (sin(H0 + NormalAngle) + sin(H1 + NormalAngle))) / 12.0f;
        float t1 = (-cos(3 * H0 - NormalAngle) - cos(3 * H1 - NormalAngle) + 8 * cos(NormalAngle) -
            3 * (cos(H0 + NormalAngle) + cos(H1 + NormalAngle))) / 12.0f;
        float3 LocalBentNormal = float3(SliceDirection.x * t0, SliceDirection.y * t0, -t1);
        LocalBentNormal = mul(buildRotationMatrix(float3(0, 0, -1), ViewVector), LocalBentNormal) * ProjectedNormalLength;
        ViewBentNormal += LocalBentNormal;

        Visibility += SliceVisibility;
    }

    Visibility = max(Visibility / EffectiveSliceCount, 0.03f);
    return mul(normalize(ViewBentNormal), WorldToView) * Visibility;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void lumenShortRangeAOMain(uint3 DispatchThreadId : SV_DispatchThreadID)
{
    LumenShortRangeAOParams Params = getParameters<LumenShortRangeAOParams>();
    ViewConstants View = resolveUniform(Params.view)[0];
    uint2 ViewSize = uint2(View.current.viewport.yz);
    uint2 ShortRangeAOCoord = DispatchThreadId.xy;
    if (any(ShortRangeAOCoord >= (ViewSize + 1u) / 2u)) {
        return;
    }

    uint2 SvPosition = min(ShortRangeAOCoord * 2u + lumenShortRangeAOJitter(ShortRangeAOCoord, View.frame.x), ViewSize - 1u);
    Texture2D<float> Depth = resolveUniform(Params.depth);
    float DeviceZ = Depth.Load(int3(SvPosition, 0));
    float3 ReceiverNormal = resolveUniform(Params.normal).Load(int3(SvPosition, 0)).xyz;
    // Match UE's decoded PF_A2B10G10R10 normal from the selected 2x2 receiver.
    float3 WorldNormal = normalize(lumenShortRangeAODecodeNormal(ReceiverNormal));
    float3 BentNormal = WorldNormal;

    // Deferred exports zero for pixels without an opaque Lumen receiver.
    if (DeviceZ > 0.0f && any(ReceiverNormal != 0.0f)) {
        float3 Forward = ScenePath.cameraForwardFrom(View.current.eye, View.current.center);
        float3 Right = ScenePath.cameraRightFrom(Forward, View.current.upProjection.xyz);
        float3 Up = ScenePath.cameraUpFrom(Forward, Right);
        float3x3 WorldToView = float3x3(Right, Up, Forward);

        ShortRangeAOProjection Projection;
        Projection.ViewSize = ViewSize;
        Projection.InvViewSize = 1.0f / float2(ViewSize);
        Projection.bOrthographic = View.current.upProjection.w > 0.5f;
        float HalfHeight = Projection.bOrthographic ? View.current.clipOrtho.z * 0.5f : tan(View.current.viewport.w * 0.5f);
        float2 HalfExtent = float2(View.current.viewport.x * HalfHeight, HalfHeight);
        Projection.ScreenToViewScale = HalfExtent * float2(2.0f, -2.0f);
        Projection.ScreenToViewBias = HalfExtent * float2(-1.0f, 1.0f);
        Projection.NearPlane = View.current.clipOrtho.x;
        Projection.FarPlane = View.current.clipOrtho.y;

        float SceneDepth = convertFromDeviceZ(Projection, DeviceZ);
        float2 UniformRandom = lumenProbeBlueNoiseVec2(resolveUniform(Params.blueNoiseVec2), SvPosition, View.frame.x);
        BentNormal = calculateAOHorizonSearch(Depth, Projection, SvPosition, SceneDepth, WorldNormal,
            WorldToView, UniformRandom);
    }

    resolveUniform(Params.shortRangeAORaw)[ShortRangeAOCoord] = packLumenScreenBentNormal(BentNormal);
}
