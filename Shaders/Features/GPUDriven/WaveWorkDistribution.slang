// Wave-local expansion of variable-sized sources. Compact once, then count
// source boundaries in each output batch instead of searching via shuffles.
// All lanes must participate, including zero sources and the final batch tail.
// Counts must have a representable wave sum (stream leaves contain <= 64 items).

uint waveWorkSelectBit(uint bits, uint rank)
{
    uint lane = 0u;
    [unroll] for (uint half = 16u; half != 0u; half >>= 1u) {
        uint lower = bits & ((1u << half) - 1u);
        uint count = countbits(lower);
        if (rank >= count) {
            rank -= count;
            bits >>= half;
            lane += half;
        } else {
            bits = lower;
        }
    }
    return lane;
}

// Ballots can contain up to 128 lanes. Never shift a uint by 32 or use a
// workgroup index as a wave index (a 64-thread group can contain two waves).
uint waveWorkSelectSource(uint4 ballot, uint rank)
{
    uint base = 0u;
    [unroll] for (uint word = 0u; word < 3u; ++word) {
        uint count = countbits(ballot[word]);
        if (rank < count) { return base + waveWorkSelectBit(ballot[word], rank); }
        rank -= count;
        base += 32u;
    }
    return base + waveWorkSelectBit(ballot.w, rank);
}

struct WaveWorkDistribution
{
    uint source;
    uint end;
    uint total;
    uint sourceCount;
    uint sourceHead;
};

WaveWorkDistribution waveWorkBegin(uint count)
{
    WaveWorkDistribution work;
    uint prefix = WavePrefixSum(count);
    work.total = WaveActiveSum(count);
    work.sourceCount = WaveActiveCountBits(count != 0u);
    work.sourceHead = 0u;
    // Clamp unused compact lanes to a valid source even for an empty wave;
    // they participate in every shuffle but never publish a boundary.
    uint rank = min(WaveGetLaneIndex(), max(work.sourceCount, 1u) - 1u);
    work.source = WaveGetLaneIndex();
    work.end = prefix + count;
    if (!WaveActiveAllTrue(count != 0u)) {
        uint4 sources = WaveActiveBallot(count != 0u);
        work.source = work.sourceCount == 0u ? 0u : WaveGetLaneCount() <= 32u ?
            waveWorkSelectBit(sources.x, rank) : waveWorkSelectSource(sources, rank);
        work.end = WaveReadLaneAt(work.end, work.source);
    }
    return work;
}

uint waveWorkNext(inout WaveWorkDistribution work, uint batch, out uint localItem)
{
    uint lane = WaveGetLaneIndex();
    uint last = work.end - batch - 1u;
    bool boundary = lane < work.sourceCount && last < WaveGetLaneCount();
    uint bit = 1u << (last & 31u);
    uint low = WaveActiveBitOr(boundary && last < 32u ? bit : 0u);
    uint before = countbits(low & ((1u << (lane & 31u)) - 1u));
    uint finished = countbits(low);
    if (WaveGetLaneCount() > 32u) {
        uint4 masks;
        masks.x = low;
        masks.y = WaveActiveBitOr(boundary && last >= 32u && last < 64u ? bit : 0u);
        masks.z = WaveActiveBitOr(boundary && last >= 64u && last < 96u ? bit : 0u);
        masks.w = WaveActiveBitOr(boundary && last >= 96u ? bit : 0u);
        before = 0u;
        finished = 0u;
        [unroll] for (uint word = 0u; word < 4u; ++word) {
            if (word < (lane >> 5u)) { before += countbits(masks[word]); }
            if (word == (lane >> 5u)) { before += countbits(masks[word] & ((1u << (lane & 31u)) - 1u)); }
            finished += countbits(masks[word]);
        }
    }
    uint slot = min(work.sourceHead + before, max(work.sourceCount, 1u) - 1u);
    uint previousEnd = WaveReadLaneAt(work.end, slot == 0u ? 0u : slot - 1u);
    localItem = batch + lane - (slot == 0u ? 0u : previousEnd);
    uint source = WaveReadLaneAt(work.source, slot);
    work.sourceHead += finished;
    return source;
}
