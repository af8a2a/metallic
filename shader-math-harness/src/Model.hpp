#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "Contract.hpp"
#include "Numeric.hpp"

namespace metallic::shadermath
{

// ---------------------------------------------------------------------------
// The candidate numeric graph. A single source of truth describes the
// candidate so that the bit-exact host simulation, the interval proofs, and
// the generated Slang all refer to the same operation order.
// ---------------------------------------------------------------------------

struct CandidateModel
{
    std::uint32_t degree = 4;
    std::vector<float> coefficients; // c0..cN, exact values from the contract bits
    std::vector<std::uint32_t> coefficientBits;
    bool clamp01 = true;
};

std::optional<CandidateModel> MakeModel(const Contract& contract, std::string& error);

// Bit-exact host simulation of the proof model: binary32 Horner with one
// std::fma (single correctly rounded multiply-add) per step, inputs used as-is
// (subnormals preserved), optional clamp to [0, 1] applied afterwards.
float EvaluateModelF32(const CandidateModel& model, float x);

// Same chain without the postprocess clamp.
float EvaluateModelF32Raw(const CandidateModel& model, float x);

// Real-arithmetic polynomial evaluation over interval boxes (Horner).
using CoefficientList = std::vector<Interval>;
Interval EvaluatePolynomialReal(const CoefficientList& coefficients, Interval x);

// Coefficients of the derivative polynomial.
CoefficientList PolynomialDerivative(const CoefficientList& coefficients);

// Coefficient lists for the exact-real polynomial from the contract.
CoefficientList IdealCoefficients(const Contract& contract);
CoefficientList QuantizedCoefficients(const CandidateModel& model);

// Certified enclosure of the binary32 FMA chain evaluated over real boxes,
// including every intermediate FMA range (used for range proofs and for the
// "no subnormal intermediates" premise of the gamma evaluation bound).
struct F32ChainTrace
{
    std::vector<F32Interval> steps; // one entry per FMA, in evaluation order
    F32Interval rawResult;          // after the final FMA
    F32Interval result;             // after the postprocess clamp
};

F32ChainTrace EvaluateModelIntervalF32(const CandidateModel& model, Interval x);

} // namespace metallic::shadermath
