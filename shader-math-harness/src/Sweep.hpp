#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "Model.hpp"
#include "Numeric.hpp"

namespace metallic::shadermath
{

// ---------------------------------------------------------------------------
// Host sweep of the bit-exact proof-model simulation over the legal binary32
// inputs of the contract domain. Positive binary32 values sort by bit pattern,
// so the legal set is one contiguous bit range and can be exhausted.
//
// The reference for observed errors is double std::exp (about 1 ulp); the
// worst inputs are afterwards re-verified against the rigorous interval
// oracle, and the observed maxima are cross-checked against the certified
// bounds so a mismatch fails loudly instead of hiding an engine bug.
// ---------------------------------------------------------------------------

struct SweepWorstEntry
{
    std::uint32_t inBits = 0;
    std::uint32_t outBits = 0;
    double observedError = 0.0;
    Interval certifiedError{0.0, 0.0};
};

struct SweepConfig
{
    bool tryExhaustive = true;
    std::uint64_t maxInputs = 0; // 0 = unlimited when tryExhaustive
    std::uint32_t threads = 0;   // 0 = hardware concurrency
    std::uint32_t worstK = 32;
    bool quiet = false;
};

struct SweepResults
{
    bool exhaustive = false;
    std::uint64_t plannedInputs = 0;
    std::uint64_t evaluatedInputs = 0;
    std::uint32_t gridLoBits = 0;
    std::uint32_t gridHiBits = 0;

    double maxObservedError = 0.0;
    std::uint32_t worstInBits = 0;
    std::uint32_t worstOutBits = 0;
    std::vector<SweepWorstEntry> worst;

    double maxObservedEvalError = 0.0; // |fl(p) - p_double| before clamp
    std::uint64_t monotonicityViolations = 0;
    std::uint32_t monotonicityFirstPairLo = 0;
    std::uint32_t monotonicityFirstPairHi = 0;
    float observedMinY = 0.0f;
    float observedMaxY = 0.0f;
    double seconds = 0.0;
    std::uint32_t threadsUsed = 1;
};

SweepResults RunSweep(const CandidateModel& model, const Contract& contract, const SweepConfig& config);

} // namespace metallic::shadermath
