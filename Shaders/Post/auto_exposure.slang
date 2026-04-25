// Auto-exposure via luminance histogram, adapted from nvpro_core2 (Apache-2.0).

struct AutoExposureUniforms {
    float evMinValue;
    float evMaxValue;
    float adaptationSpeed;
    float deltaTime;
    uint screenWidth;
    uint screenHeight;
    float lowPercentile;
    float highPercentile;
};

[[vk::push_constant]] ConstantBuffer<AutoExposureUniforms> params;

// --- Histogram kernel ---
Texture2D<float4> hdrInput;
RWByteAddressBuffer histogramBuffer; // 256 * 4 bytes

groupshared uint sharedHistogram[256];

float bt709Luminance(float3 c) {
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

float luminanceToEv100(float L) {
    return log2(L * 8.0);
}

float ev100ToLuminance(float ev) {
    return exp2(ev) / 8.0;
}

uint luminanceToBucket(float luminance) {
    if (luminance <= 0.0) return 0;
    float ev = luminanceToEv100(luminance);
    float t = saturate((ev - params.evMinValue) / (params.evMaxValue - params.evMinValue));
    return uint(t * 254.0 + 1.0);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void histogramMain(uint3 dtid : SV_DispatchThreadID, uint localIdx : SV_GroupIndex) {
    // Clear shared histogram
    if (localIdx < 256)
        sharedHistogram[localIdx] = 0;
    GroupMemoryBarrierWithGroupSync();

    // Accumulate into shared histogram
    if (dtid.x < params.screenWidth && dtid.y < params.screenHeight) {
        float3 color = hdrInput[uint2(dtid.x, dtid.y)].rgb;
        float lum = bt709Luminance(color);
        uint bucket = luminanceToBucket(lum);
        InterlockedAdd(sharedHistogram[bucket], 1);
    }
    GroupMemoryBarrierWithGroupSync();

    // Merge shared histogram into global buffer
    if (localIdx < 256) {
        uint count = sharedHistogram[localIdx];
        if (count > 0) {
            histogramBuffer.InterlockedAdd(localIdx * 4, count);
        }
    }
}

// --- Exposure kernel ---
RWTexture2D<float> exposureLut;

groupshared uint sharedHist[256];
groupshared float sharedCdf[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void exposureMain(uint localIdx : SV_GroupIndex) {
    // Load histogram into shared memory
    sharedHist[localIdx] = histogramBuffer.Load(localIdx * 4);
    GroupMemoryBarrierWithGroupSync();

    // Prefix sum (Hillis-Steele)
    sharedCdf[localIdx] = float(sharedHist[localIdx]);
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint offset = 1; offset < 256; offset <<= 1) {
        float val = 0.0;
        if (localIdx >= offset)
            val = sharedCdf[localIdx - offset];
        GroupMemoryBarrierWithGroupSync();
        sharedCdf[localIdx] += val;
        GroupMemoryBarrierWithGroupSync();
    }

    // Compute weighted average EV within percentile range
    if (localIdx == 0) {
        float totalPixels = sharedCdf[255];
        if (totalPixels < 1.0) {
            // No valid pixels — write previous value unchanged
            // Clear histogram for next frame
            for (uint i = 0; i < 256; i++)
                histogramBuffer.Store(i * 4, 0);
            return;
        }

        float lowCount = totalPixels * params.lowPercentile;
        float highCount = totalPixels * params.highPercentile;

        float weightedSum = 0.0;
        float weightTotal = 0.0;

        for (uint i = 1; i < 256; i++) {
            float cdfPrev = sharedCdf[i - 1];
            float cdfCurr = sharedCdf[i];
            float binCount = float(sharedHist[i]);

            // Clamp contribution to percentile range
            float lo = max(cdfPrev, lowCount);
            float hi = min(cdfCurr, highCount);
            float contribution = max(hi - lo, 0.0);

            // Bucket center EV
            float t = (float(i) - 0.5) / 254.0;
            float ev = params.evMinValue + t * (params.evMaxValue - params.evMinValue);

            weightedSum += ev * contribution;
            weightTotal += contribution;
        }

        float targetEv = (weightTotal > 0.0) ? (weightedSum / weightTotal) : 0.0;
        float targetLum = ev100ToLuminance(targetEv);

        // Read previous adapted luminance
        float prevLum = exposureLut[uint2(0, 0)];
        if (prevLum <= 0.0) prevLum = targetLum; // first frame

        // Temporal adaptation
        float speed = params.adaptationSpeed;
        float alpha = 1.0 - exp2(-speed * params.deltaTime);
        float adaptedLum = lerp(prevLum, targetLum, alpha);
        adaptedLum = max(adaptedLum, 0.001);

        exposureLut[uint2(0, 0)] = adaptedLum;

        // Clear histogram for next frame
        for (uint j = 0; j < 256; j++)
            histogramBuffer.Store(j * 4, 0);
    }
}
