// Input/output arrays and numeric contracts match AABBTransform.h. This kernel
// processes only the current thread's range; Metallic owns multicore execution.
export void metallicTransformAABBs(
    uniform const float * uniform minX, uniform const float * uniform minY,
    uniform const float * uniform minZ, uniform const float * uniform maxX,
    uniform const float * uniform maxY, uniform const float * uniform maxZ,
    uniform const float * uniform m00, uniform const float * uniform m01,
    uniform const float * uniform m02, uniform const float * uniform m03,
    uniform const float * uniform m10, uniform const float * uniform m11,
    uniform const float * uniform m12, uniform const float * uniform m13,
    uniform const float * uniform m20, uniform const float * uniform m21,
    uniform const float * uniform m22, uniform const float * uniform m23,
    uniform float * uniform outMinX, uniform float * uniform outMinY,
    uniform float * uniform outMinZ, uniform float * uniform outMaxX,
    uniform float * uniform outMaxY, uniform float * uniform outMaxZ,
    uniform uint32 begin, uniform uint32 end)
{
    // foreach accepts signed 32-bit indices; offset from begin using 64-bit
    // counters so a valid public uint32 range near UINT32_MAX cannot wrap.
    for (uniform uint64 base = begin; base < (uniform uint64)end; base += programCount) {
        uint64 index = base + programIndex;
        // Full gangs use an unmasked fast path; only the final partial gang
        // needs masked accesses. This avoids AVX2 maskmov on every iteration.
        cif (index < (uniform uint64)end) {
            float x0 = m00[index] * minX[index], x1 = m00[index] * maxX[index];
            float y0 = m01[index] * minY[index], y1 = m01[index] * maxY[index];
            float z0 = m02[index] * minZ[index], z1 = m02[index] * maxZ[index];
            outMinX[index] = min(x0, x1) + min(y0, y1) + min(z0, z1) + m03[index];
            outMaxX[index] = max(x0, x1) + max(y0, y1) + max(z0, z1) + m03[index];
            x0 = m10[index] * minX[index]; x1 = m10[index] * maxX[index];
            y0 = m11[index] * minY[index]; y1 = m11[index] * maxY[index];
            z0 = m12[index] * minZ[index]; z1 = m12[index] * maxZ[index];
            outMinY[index] = min(x0, x1) + min(y0, y1) + min(z0, z1) + m13[index];
            outMaxY[index] = max(x0, x1) + max(y0, y1) + max(z0, z1) + m13[index];
            x0 = m20[index] * minX[index]; x1 = m20[index] * maxX[index];
            y0 = m21[index] * minY[index]; y1 = m21[index] * maxY[index];
            z0 = m22[index] * minZ[index]; z1 = m22[index] * maxZ[index];
            outMinZ[index] = min(x0, x1) + min(y0, y1) + min(z0, z1) + m23[index];
            outMaxZ[index] = max(x0, x1) + max(y0, y1) + max(z0, z1) + m23[index];
        }
    }
}

export uniform int32 metallicAABBTransformISA()
{
#if defined(ISPC_TARGET_AVX2)
    return 2;
#else
    return 1;
#endif
}
