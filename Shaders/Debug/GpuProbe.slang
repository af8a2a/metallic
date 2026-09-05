// Fixed read-only probes. No floating-point atomics, wave-size assumptions or
// user shader text. CPU merges at most 256 associative group summaries.
struct ProbePush
{
    uint byteOffset;
    uint stride;
    uint fieldOffset;
    uint count;
    uint scalarType; // 0 u32, 1 i32, 2 f32
    uint operation; // 0 count, 1 outOfBounds, 2 nonFinite, 3 minMax
    uint predicate; // all, eq, ne, lt, le, gt, ge
    uint lowerBits;
    uint upperBits;
    uint groupCount;
    uint bitOffset;
    uint bitMask;
    uint scale;
};
struct Partial
{
    uint finiteCount;
    uint matchedCount;
    uint nanCount;
    uint infCount;
    uint minBits;
    uint maxBits;
    uint firstIndex;
    uint firstBits;
};
[[vk::binding(0, 0)]] ByteAddressBuffer gInput;
[[vk::binding(1, 0)]] RWStructuredBuffer<Partial> gOutput;
groupshared Partial partials[128];

bool lessValue(uint a, uint b, uint type)
{
    if (type == 2) return asfloat(a) < asfloat(b);
    if (type == 1) return asint(a) < asint(b);
    return a < b;
}

bool equalValue(uint a, uint b, uint type)
{
    return type == 2 ? asfloat(a) == asfloat(b) : a == b;
}

Partial merge(Partial a, Partial b, uint type)
{
    if (b.finiteCount != 0) {
        if (a.finiteCount == 0 || lessValue(b.minBits, a.minBits, type)) a.minBits = b.minBits;
        if (a.finiteCount == 0 || lessValue(a.maxBits, b.maxBits, type)) a.maxBits = b.maxBits;
    }
    a.finiteCount += b.finiteCount;
    a.matchedCount += b.matchedCount;
    a.nanCount += b.nanCount;
    a.infCount += b.infCount;
    if (b.firstIndex < a.firstIndex) { a.firstIndex = b.firstIndex; a.firstBits = b.firstBits; }
    return a;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void probe(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupID,
    uint3 local : SV_GroupThreadID, uniform ProbePush push)
{
    Partial result = {};
    result.firstIndex = 0xffffffff;
    for (uint i = tid.x; i < push.count; i += push.groupCount * 128) {
        uint bits = gInput.Load(push.byteOffset + i * push.stride + push.fieldOffset);
        if (push.scalarType == 0) bits = ((bits >> push.bitOffset) & push.bitMask) * push.scale;
        bool nan = push.scalarType == 2 && isnan(asfloat(bits));
        bool inf = push.scalarType == 2 && isinf(asfloat(bits));
        bool matched = false;
        if (push.operation == 2) matched = nan || inf;
        else if (push.operation == 1) matched = lessValue(bits, push.lowerBits, push.scalarType) ||
            !lessValue(bits, push.upperBits, push.scalarType);
        else if (push.operation == 0) {
            bool lt = lessValue(bits, push.lowerBits, push.scalarType);
            bool gt = lessValue(push.lowerBits, bits, push.scalarType);
            bool eq = equalValue(bits, push.lowerBits, push.scalarType);
            matched = push.predicate == 0 || (push.predicate == 1 && eq) ||
                (push.predicate == 2 && !eq) || (push.predicate == 3 && lt) ||
                (push.predicate == 4 && (lt || eq)) || (push.predicate == 5 && gt) ||
                (push.predicate == 6 && (gt || eq));
        }
        Partial item = {};
        item.finiteCount = (!nan && !inf) ? 1 : 0;
        item.nanCount = nan ? 1 : 0;
        item.infCount = inf ? 1 : 0;
        item.minBits = bits; item.maxBits = bits;
        item.matchedCount = matched ? 1 : 0;
        item.firstIndex = matched ? i : 0xffffffff;
        item.firstBits = bits;
        result = merge(result, item, push.scalarType);
    }
    partials[local.x] = result;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 64; stride != 0; stride >>= 1) {
        if (local.x < stride) partials[local.x] = merge(partials[local.x], partials[local.x + stride], push.scalarType);
        GroupMemoryBarrierWithGroupSync();
    }
    if (local.x == 0) gOutput[gid.x] = partials[0];
}
