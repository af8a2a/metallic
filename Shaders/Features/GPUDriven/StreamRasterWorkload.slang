int workloadEdge(int2 a, int2 b, int2 p)
{
    int2 d = b - a, q = p - a;
    return d.x * q.y - d.y * q.x;
}
int workloadBias(int2 a, int2 b)
{
    int2 d = b - a;
    return d.y < 0 || (d.y == 0 && d.x > 0) ? 0 : -1;
}

// Integer setup and edge stepping match hybridRasterPreparedTriangle. The final
// coverage count equals InterlockedMax attempts in that kernel (no depth reject).
uint64_t4 workloadTriangle(int2 p0, int2 p1, int2 p2, bool doubleSided, uint width, uint height, uint bits)
{
    int area = workloadEdge(p0, p1, p2);
    if (area == 0 || (!doubleSided && area > 0)) { return uint64_t4(0); }
    if (area < 0) { int2 t = p1; p1 = p2; p2 = t; }
    int scale = 1 << bits;
    int2 lower = max(int2(0), int2(ceil(float2(min(p0, min(p1, p2))) / scale - 0.5)));
    int2 upper = min(int2(width, height) - 1, int2(floor(float2(max(p0, max(p1, p2))) / scale - 0.5)));
    if (any(lower > upper)) { return uint64_t4(0); }
    float2 s0 = float2(p0) / scale;
    float2 e1 = float2(p1) / scale - s0, e2 = float2(p2) / scale - s0;
    if (abs(e1.x * e2.y - e1.y * e2.x) < 1e-12) { return uint64_t4(0); }
    int2 point = lower * scale + scale / 2;
    int3 edges = int3(workloadEdge(p1,p2,point), workloadEdge(p2,p0,point), workloadEdge(p0,p1,point)) +
        int3(workloadBias(p1,p2), workloadBias(p2,p0), workloadBias(p0,p1));
    int3 stepX = -int3(p2.y-p1.y, p0.y-p2.y, p1.y-p0.y) * scale;
    int3 stepY = int3(p2.x-p1.x, p0.x-p2.x, p1.x-p0.x) * scale;
    uint64_t covered = 0;
    for (int y = lower.y; y <= upper.y; ++y) {
        int3 pixelEdges = edges;
        for (int x = lower.x; x <= upper.x; ++x) {
            covered += all(pixelEdges >= 0) ? 1u : 0u;
            pixelEdges += stepX;
        }
        edges += stepY;
    }
    uint64_t w = uint(upper.x - lower.x + 1), h = uint(upper.y - lower.y + 1);
    return uint64_t4(w * h, covered, h, w);
}

