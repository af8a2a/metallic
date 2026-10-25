#include "Sweep.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

#include "Contract.hpp"

namespace metallic::shadermath
{

namespace
{

struct LocalWorst
{
    std::uint32_t inBits = 0;
    std::uint32_t outBits = 0;
    double error = -1.0;
};

struct ChunkResult
{
    double maxError = -1.0;
    LocalWorst worst{0, 0, -1.0};
    double maxEvalError = 0.0;
    std::uint64_t monotonicityViolations = 0;
    std::uint32_t monotonicityFirstLo = 0;
    std::uint32_t monotonicityFirstHi = 0;
    bool monotonicityRecorded = false;
    float minY = 3.4e38f;
    float maxY = -3.4e38f;
    std::vector<LocalWorst> top;
    std::uint64_t evaluated = 0;
};

void InsertTop(std::vector<LocalWorst>& top, std::uint32_t inBits, std::uint32_t outBits,
    double error, std::size_t capacity)
{
    if (capacity == 0)
    {
        return;
    }
    if (top.size() < capacity)
    {
        top.push_back(LocalWorst{inBits, outBits, error});
        for (std::size_t i = top.size() - 1; i > 0 && top[i].error > top[i - 1].error; --i)
        {
            std::swap(top[i], top[i - 1]);
        }
        return;
    }
    if (error <= top.back().error)
    {
        return;
    }
    top.back() = LocalWorst{inBits, outBits, error};
    for (std::size_t i = top.size() - 1; i > 0 && top[i].error > top[i - 1].error; --i)
    {
        std::swap(top[i], top[i - 1]);
    }
}

// Per-input hot path: bit-exact model chain, double reference, and a double
// Horner evaluation of the quantized polynomial for the evaluation-error
// cross-check (its ~1e-16 error is irrelevant at the 1e-7 scale).
void EvaluateRange(const CandidateModel& model, std::uint64_t beginBits, std::uint64_t endBits,
    std::size_t stride, ChunkResult& out, std::size_t topCapacity, bool monotonicityCheck)
{
    float previousY = 0.0f;
    bool havePrevious = false;
    for (std::uint64_t bits = beginBits; bits <= endBits && bits <= 0x7F7FFFFFULL; bits += stride)
    {
        const float x = FloatFromBits(static_cast<std::uint32_t>(bits));
        const double xd = static_cast<double>(x);
        const float raw = EvaluateModelF32Raw(model, x);
        const float y = model.clamp01 ? std::min(1.0f, std::max(0.0f, raw)) : raw;

        const double reference = std::exp(-xd);
        const double error = std::fabs(static_cast<double>(y) - reference);

        double poly = static_cast<double>(model.coefficients[model.degree]);
        for (std::size_t k = model.degree; k-- > 0;)
        {
            poly = std::fma(poly, xd, static_cast<double>(model.coefficients[k]));
        }
        const double evalError = std::fabs(static_cast<double>(raw) - poly);

        if (error > out.maxError)
        {
            out.maxError = error;
            out.worst = LocalWorst{static_cast<std::uint32_t>(bits), BitsFromFloat(y), error};
        }
        if (evalError > out.maxEvalError)
        {
            out.maxEvalError = evalError;
        }
        InsertTop(out.top, static_cast<std::uint32_t>(bits), BitsFromFloat(y), error, topCapacity);
        if (y < out.minY)
        {
            out.minY = y;
        }
        if (y > out.maxY)
        {
            out.maxY = y;
        }
        if (monotonicityCheck)
        {
            if (havePrevious && y > previousY)
            {
                if (!out.monotonicityRecorded)
                {
                    out.monotonicityFirstLo = static_cast<std::uint32_t>(bits - 1);
                    out.monotonicityFirstHi = static_cast<std::uint32_t>(bits);
                    out.monotonicityRecorded = true;
                }
                ++out.monotonicityViolations;
            }
            previousY = y;
            havePrevious = true;
        }
        ++out.evaluated;
    }
}

void MergeChunk(ChunkResult& merged, const ChunkResult& chunk, std::size_t topCapacity)
{
    if (chunk.maxError > merged.maxError)
    {
        merged.maxError = chunk.maxError;
        merged.worst = chunk.worst;
    }
    if (chunk.maxEvalError > merged.maxEvalError)
    {
        merged.maxEvalError = chunk.maxEvalError;
    }
    if (chunk.minY < merged.minY)
    {
        merged.minY = chunk.minY;
    }
    if (chunk.maxY > merged.maxY)
    {
        merged.maxY = chunk.maxY;
    }
    merged.monotonicityViolations += chunk.monotonicityViolations;
    if (!merged.monotonicityRecorded && chunk.monotonicityRecorded)
    {
        merged.monotonicityFirstLo = chunk.monotonicityFirstLo;
        merged.monotonicityFirstHi = chunk.monotonicityFirstHi;
        merged.monotonicityRecorded = true;
    }
    for (const LocalWorst& entry : chunk.top)
    {
        InsertTop(merged.top, entry.inBits, entry.outBits, entry.error, topCapacity);
    }
    merged.evaluated += chunk.evaluated;
}

} // namespace

SweepResults RunSweep(const CandidateModel& model, const Contract& contract, const SweepConfig& config)
{
    SweepResults results;

    // Legal grid: largest float <= domain lo/hi. Positive binary32 values sort
    // by bit pattern, so the legal set is the contiguous range [gridLo, gridHi].
    const double loReal =
        static_cast<double>(contract.domainLo.num) / static_cast<double>(contract.domainLo.den);
    const double hiReal =
        static_cast<double>(contract.domainHi.num) / static_cast<double>(contract.domainHi.den);
    float loF = static_cast<float>(loReal);
    if (static_cast<double>(loF) > loReal)
    {
        loF = NextBelowF(loF);
    }
    float hiF = static_cast<float>(hiReal);
    if (static_cast<double>(hiF) > hiReal)
    {
        hiF = NextBelowF(hiF);
    }
    const std::uint32_t gridLo = BitsFromFloat(loF);
    const std::uint32_t gridHi = BitsFromFloat(hiF);
    results.gridLoBits = gridLo;
    results.gridHiBits = gridHi;
    const std::uint64_t gridCount = static_cast<std::uint64_t>(gridHi) - gridLo + 1;

    results.exhaustive =
        config.tryExhaustive && (config.maxInputs == 0 || gridCount <= config.maxInputs);
    std::size_t stride = 1;
    if (!results.exhaustive)
    {
        std::uint64_t planned = config.maxInputs != 0 ? config.maxInputs : 1'000'000;
        if (planned > gridCount)
        {
            planned = gridCount;
        }
        stride = static_cast<std::size_t>(gridCount / planned);
        if (stride < 1)
        {
            stride = 1;
        }
    }
    results.plannedInputs = results.exhaustive
        ? gridCount
        : ((gridHi - gridLo) / stride) + 1;

    std::uint32_t threads = config.threads;
    if (threads == 0)
    {
        const unsigned hardware = std::thread::hardware_concurrency();
        threads = hardware == 0 ? 1u : static_cast<std::uint32_t>(std::min(hardware, 32u));
    }
    results.threadsUsed = threads;

    const auto startTime = std::chrono::steady_clock::now();

    const std::size_t topCapacity = config.worstK;
    const std::size_t chunkCount = static_cast<std::size_t>(threads) * 4;
    std::vector<ChunkResult> chunks(chunkCount);
    std::atomic<std::size_t> activeWorkers{0};
    std::atomic<std::uint64_t> progress{0};
    std::vector<std::thread> workers;

    const bool monotonicity = results.exhaustive && stride == 1;
    for (std::size_t c = 0; c < chunkCount; ++c)
    {
        const std::uint64_t span = gridCount;
        const std::uint64_t begin = gridLo + span * c / chunkCount;
        const std::uint64_t end = gridLo + span * (c + 1) / chunkCount - 1;
        if (begin > end)
        {
            continue;
        }
        chunks[c].top.reserve(topCapacity);
        ++activeWorkers;
        workers.emplace_back([&model, begin, end, stride, monotonicity, topCapacity, c, &chunks,
            &activeWorkers, &progress]()
        {
            EvaluateRange(model, begin, end, stride, chunks[c], topCapacity, monotonicity);
            progress.fetch_add(chunks[c].evaluated);
            --activeWorkers;
        });
    }

    if (!config.quiet)
    {
        while (activeWorkers.load() > 0)
        {
            std::printf("sweep: %llu / %llu inputs (%.1f%%)\n",
                static_cast<unsigned long long>(progress.load()),
                static_cast<unsigned long long>(results.plannedInputs),
                100.0 * static_cast<double>(progress.load()) /
                    static_cast<double>(results.plannedInputs));
            std::fflush(stdout);
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
    }
    for (std::thread& worker : workers)
    {
        worker.join();
    }

    ChunkResult merged;
    for (const ChunkResult& chunk : chunks)
    {
        MergeChunk(merged, chunk, topCapacity);
    }

    // Exhaustive mode: re-check the chunk boundary pairs, each worker only
    // tracked adjacency inside its own range.
    if (monotonicity)
    {
        for (std::size_t c = 1; c < chunkCount; ++c)
        {
            const std::uint64_t span = gridCount;
            const std::uint64_t boundary = gridLo + span * c / chunkCount;
            if (boundary <= gridLo || boundary > gridHi)
            {
                continue;
            }
            const float left =
                EvaluateModelF32(model, FloatFromBits(static_cast<std::uint32_t>(boundary - 1)));
            const float right =
                EvaluateModelF32(model, FloatFromBits(static_cast<std::uint32_t>(boundary)));
            if (right > left)
            {
                if (!merged.monotonicityRecorded)
                {
                    merged.monotonicityFirstLo = static_cast<std::uint32_t>(boundary - 1);
                    merged.monotonicityFirstHi = static_cast<std::uint32_t>(boundary);
                    merged.monotonicityRecorded = true;
                }
                ++merged.monotonicityViolations;
            }
        }
    }

    // Sampled mode: additionally probe representational boundaries.
    if (!results.exhaustive)
    {
        std::vector<std::uint32_t> specials;
        specials.push_back(gridLo);
        if (gridLo + 1 <= gridHi)
        {
            specials.push_back(gridLo + 1);
        }
        specials.push_back(0x00000001);
        specials.push_back(0x007FFFFF);
        specials.push_back(0x00800000);
        specials.push_back(0x00800001);
        if (gridHi >= 1)
        {
            specials.push_back(gridHi - 1);
        }
        specials.push_back(gridHi);
        for (const std::uint32_t bits : specials)
        {
            if (bits < gridLo || bits > gridHi)
            {
                continue;
            }
            ChunkResult scratch;
            scratch.top.reserve(1);
            EvaluateRange(model, bits, bits, 1, scratch, topCapacity, false);
            MergeChunk(merged, scratch, topCapacity);
        }
    }

    results.evaluatedInputs = merged.evaluated;
    results.maxObservedError = merged.maxError;
    results.worstInBits = merged.worst.inBits;
    results.worstOutBits = merged.worst.outBits;
    results.maxObservedEvalError = merged.maxEvalError;
    results.monotonicityViolations = merged.monotonicityViolations;
    results.monotonicityFirstPairLo = merged.monotonicityFirstLo;
    results.monotonicityFirstPairHi = merged.monotonicityFirstHi;
    results.observedMinY = merged.minY;
    results.observedMaxY = merged.maxY;

    // Re-verify the worst inputs against the rigorous interval oracle so the
    // reported worst cases carry certified errors, not double-exp estimates.
    results.worst.reserve(merged.top.size());
    for (const LocalWorst& entry : merged.top)
    {
        const float x = FloatFromBits(entry.inBits);
        const float y = FloatFromBits(entry.outBits);
        const Interval reference =
            ExpNegativeSmall(Interval{static_cast<double>(x), static_cast<double>(x)});
        const Interval certified = Abs(Sub(IntervalExact(static_cast<double>(y)), reference));
        results.worst.push_back(SweepWorstEntry{entry.inBits, entry.outBits, entry.error, certified});
    }

    const auto endTime = std::chrono::steady_clock::now();
    results.seconds = std::chrono::duration<double>(endTime - startTime).count();
    return results;
}

} // namespace metallic::shadermath
