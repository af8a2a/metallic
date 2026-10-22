#language slang 2026
module StreamCLASAllocator;

// Only allocatorMain phase 2 mutates the gap topology. Work is page-granular:
// CLAS size reduction and move-address generation run independently in parallel.
struct Gap
{
    uint offset, size, previous, next;
    uint binPrevious, binNext, free, unusedNext;
};
struct Operation
{
    uint page, kind, count, addressOffset;
    uint sourceOffset, sizeOffset, generation, reserved;
};
struct PageAllocation { uint node, bytes, encodedBytes, count; };
struct Outcome { uint success, bytes, encodedBytes, kind; };
struct Work { uint units, encodedBytes, moveOffset, success; };
struct Parameters
{
    uint* state;
    Gap* gaps;
    PageAllocation* allocations;
    uint2* pageTable;
    uint64_t* addresses;
    Operation* operations;
    uint* sizes;
    uint64_t* sourceAddresses;
    uint64_t* moveSources;
    uint64_t* moveDestinations;
    uint* moveCount;
    Work* work;
    Outcome* outcomes;
    uint64_t arena;
    uint operationCount, pageCount, alignment, stride;
    uint committedUnits, nodeCapacity, initialize, reserved;
};
struct Push { Parameters* parameters; uint phase, reserved; };
[[vk::push_constant]] ConstantBuffer<Push> gPush;

static const uint kInvalid = 0xffffffffu;
static const uint kNextNode = 32u;
static const uint kFreeNode = 33u;
static const uint kTail = 34u;
static const uint kCommitted = 35u;

uint sizeBin(uint size) { return uint(firstbithigh(size)); }

void removeFree(Parameters p, uint id)
{
    Gap gap = p.gaps[id];
    uint bin = sizeBin(gap.size);
    if (gap.binPrevious != kInvalid) p.gaps[gap.binPrevious].binNext = gap.binNext;
    else p.state[bin] = gap.binNext;
    if (gap.binNext != kInvalid) p.gaps[gap.binNext].binPrevious = gap.binPrevious;
    p.gaps[id].free = 0;
}

void insertFree(Parameters p, uint id)
{
    uint bin = sizeBin(p.gaps[id].size);
    uint head = p.state[bin];
    p.gaps[id].binPrevious = kInvalid;
    p.gaps[id].binNext = head;
    p.gaps[id].free = 1;
    if (head != kInvalid) p.gaps[head].binPrevious = id;
    p.state[bin] = id;
}

uint newNode(Parameters p)
{
    uint id = p.state[kFreeNode];
    if (id != kInvalid) p.state[kFreeNode] = p.gaps[id].unusedNext;
    else { id = p.state[kNextNode]; p.state[kNextNode] = id + 1u; }
    return id;
}

void releaseNode(Parameters p, uint id)
{
    p.gaps[id].unusedNext = p.state[kFreeNode];
    p.state[kFreeNode] = id;
}

void releaseAllocation(Parameters p, uint id)
{
    uint previous = p.gaps[id].previous;
    if (previous != kInvalid && p.gaps[previous].free != 0u)
    {
        removeFree(p, previous);
        p.gaps[previous].size += p.gaps[id].size;
        p.gaps[previous].next = p.gaps[id].next;
        if (p.gaps[id].next != kInvalid) p.gaps[p.gaps[id].next].previous = previous;
        else p.state[kTail] = previous;
        releaseNode(p, id);
        id = previous;
    }
    uint next = p.gaps[id].next;
    if (next != kInvalid && p.gaps[next].free != 0u)
    {
        removeFree(p, next);
        p.gaps[id].size += p.gaps[next].size;
        p.gaps[id].next = p.gaps[next].next;
        if (p.gaps[next].next != kInvalid) p.gaps[p.gaps[next].next].previous = id;
        else p.state[kTail] = id;
        releaseNode(p, next);
    }
    insertFree(p, id);
}

void extendArena(Parameters p)
{
    uint oldUnits = p.state[kCommitted];
    if (oldUnits == p.committedUnits) return;
    uint tail = p.state[kTail];
    if (tail != kInvalid && p.gaps[tail].free != 0u)
    {
        removeFree(p, tail);
        p.gaps[tail].size += p.committedUnits - oldUnits;
        insertFree(p, tail);
    }
    else
    {
        uint id = newNode(p);
        Gap gap = { oldUnits, p.committedUnits - oldUnits, tail, kInvalid,
                    kInvalid, kInvalid, 0u, kInvalid };
        p.gaps[id] = gap;
        if (tail != kInvalid) p.gaps[tail].next = id;
        p.state[kTail] = id;
        insertFree(p, id);
    }
    p.state[kCommitted] = p.committedUnits;
}

uint allocate(Parameters p, uint units)
{
    uint found = kInvalid;
    for (uint bin = sizeBin(units); bin < 32u; ++bin)
    {
        uint id = p.state[bin];
        // Only the first size class can contain undersized gaps. Higher bins
        // guarantee a fit, so no arena-sized per-cluster scan is necessary.
        while (id != kInvalid && p.gaps[id].size < units) id = p.gaps[id].binNext;
        if (id != kInvalid) { found = id; break; }
    }
    if (found == kInvalid) return found;
    Gap gap = p.gaps[found];
    removeFree(p, found);
    if (gap.size > units)
    {
        uint remainder = newNode(p);
        Gap rest = { gap.offset + units, gap.size - units, found, gap.next,
                     kInvalid, kInvalid, 0u, kInvalid };
        p.gaps[remainder] = rest;
        if (gap.next != kInvalid) p.gaps[gap.next].previous = remainder;
        else p.state[kTail] = remainder;
        p.gaps[found].next = remainder;
        p.gaps[found].size = units;
        insertFree(p, remainder);
    }
    return found;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void allocatorMain(uint3 groupID : SV_GroupID, uint lane : SV_GroupIndex)
{
    Parameters p = *gPush.parameters;
    uint id = (groupID.x + groupID.y * 65535u) * 64u + lane;
    if (gPush.phase == 0u)
    {
        if (id < p.pageCount)
        {
            p.pageTable[id] = uint2(0u, 0u);
            PageAllocation empty = { kInvalid, 0u, 0u, 0u };
            p.allocations[id] = empty;
        }
        if (id < 32u) p.state[id] = kInvalid;
        if (id == 0u)
        {
            p.state[kNextNode] = 0u;
            p.state[kFreeNode] = kInvalid;
            p.state[kTail] = kInvalid;
            p.state[kCommitted] = 0u;
        }
        return;
    }
    if (gPush.phase == 1u)
    {
        if (id >= p.operationCount) return;
        Operation op = p.operations[id];
        Work work = {};
        if (op.kind == 0u)
        {
            bool valid = true;
            for (uint c = 0u; c < op.count; ++c)
            {
                uint bytes = p.sizes[op.sizeOffset + c];
                valid = valid && bytes != 0u && bytes <= p.stride;
                work.units += (bytes + p.alignment - 1u) / p.alignment;
                work.encodedBytes += bytes;
            }
            if (!valid) work.units = 0u;
        }
        p.work[id] = work;
        return;
    }
    if (gPush.phase == 2u)
    {
        if (id != 0u) return;
        extendArena(p);
        uint count = 0u;
        for (uint i = 0u; i < p.operationCount; ++i)
        {
            Operation op = p.operations[i];
            PageAllocation page = p.allocations[op.page];
            Outcome result = { 1u, page.bytes, page.encodedBytes, op.kind };
            if (op.kind == 1u)
            {
                if (page.node != kInvalid) releaseAllocation(p, page.node);
                PageAllocation empty = { kInvalid, 0u, 0u, 0u };
                p.allocations[op.page] = empty;
                p.pageTable[op.page] = uint2(0u, op.generation);
            }
            else if (op.kind == 2u || op.kind == 3u)
            {
                uint state = op.kind == 2u ? 2u : 1u;
                p.pageTable[op.page] = uint2(page.node != kInvalid ? op.addressOffset | (state << 30u) : 0u, op.generation);
            }
            else
            {
                Work work = p.work[i];
                uint node = work.units != 0u ? allocate(p, work.units) : kInvalid;
                work.success = node != kInvalid ? 1u : 0u;
                work.moveOffset = count;
                p.work[i] = work;
                result.success = work.units == 0u ? 2u : work.success;
                result.bytes = work.success != 0u ? work.units * p.alignment : 0u;
                result.encodedBytes = work.success != 0u ? work.encodedBytes : 0u;
                p.pageTable[op.page] = uint2(0u, op.generation);
                if (work.success != 0u)
                {
                    PageAllocation allocation = { node, result.bytes, result.encodedBytes, op.count };
                    p.allocations[op.page] = allocation;
                    count += op.count;
                }
            }
            p.outcomes[i] = result;
        }
        *p.moveCount = count;
        return;
    }
    if (id >= p.operationCount) return;
    Operation op = p.operations[id];
    Work work = p.work[id];
    if (op.kind != 0u || work.success == 0u) return;
    if (gPush.phase == 3u)
    {
        uint64_t address = p.arena + uint64_t(p.gaps[p.allocations[op.page].node].offset) * p.alignment;
        for (uint c = 0u; c < op.count; ++c)
        {
            p.moveSources[work.moveOffset + c] = p.sourceAddresses[op.sourceOffset + c];
            p.moveDestinations[work.moveOffset + c] = address;
            p.addresses[op.addressOffset + c] = address;
            address += uint64_t((p.sizes[op.sizeOffset + c] + p.alignment - 1u) / p.alignment) * p.alignment;
        }
    }
    else if (gPush.phase == 4u)
    {
        p.pageTable[op.page] = uint2(op.addressOffset | (1u << 30u), op.generation);
    }
}
