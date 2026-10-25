#include "Model.hpp"

#include <algorithm>
#include <cassert>

namespace metallic::shadermath
{

std::optional<CandidateModel> MakeModel(const Contract& contract, std::string& error)
{
    if (contract.form != "horner_degree_4" || contract.coefficientBits.size() != 5)
    {
        error = "unsupported candidate form '" + contract.form + "'";
        return std::nullopt;
    }
    CandidateModel model;
    model.degree = 4;
    model.coefficientBits = contract.coefficientBits;
    model.coefficients.reserve(contract.coefficientBits.size());
    for (const std::uint32_t bits : contract.coefficientBits)
    {
        model.coefficients.push_back(FloatFromBits(bits));
    }
    model.clamp01 = (contract.postprocess == "clamp_0_1");
    return model;
}

float EvaluateModelF32Raw(const CandidateModel& model, float x)
{
    assert(model.coefficients.size() == model.degree + 1);
    float h = model.coefficients[model.degree];
    for (std::size_t k = model.coefficients.size() - 1; k-- > 0;)
    {
        h = std::fma(h, x, model.coefficients[k]);
    }
    return h;
}

float EvaluateModelF32(const CandidateModel& model, float x)
{
    float y = EvaluateModelF32Raw(model, x);
    if (model.clamp01)
    {
        y = std::min(1.0f, std::max(0.0f, y));
    }
    return y;
}

Interval EvaluatePolynomialReal(const CoefficientList& coefficients, Interval x)
{
    assert(!coefficients.empty());
    Interval h = coefficients.back();
    for (std::size_t k = coefficients.size() - 1; k-- > 0;)
    {
        h = Add(Mul(h, x), coefficients[k]);
    }
    return h;
}

CoefficientList PolynomialDerivative(const CoefficientList& coefficients)
{
    CoefficientList derivative;
    const std::size_t n = coefficients.size();
    if (n < 2)
    {
        return CoefficientList{IntervalExact(0.0)};
    }
    derivative.resize(n - 1);
    for (std::size_t k = 1; k < n; ++k)
    {
        // d/dx (c_k x^k) = k * c_k x^(k-1)
        derivative[k - 1] = Mul(coefficients[k], IntervalFromDouble(static_cast<double>(k)));
    }
    return derivative;
}

CoefficientList IdealCoefficients(const Contract& contract)
{
    CoefficientList coefficients;
    if (contract.hasIdealCoefficients)
    {
        coefficients.reserve(contract.idealCoefficients.size());
        for (const RationalValue& rational : contract.idealCoefficients)
        {
            coefficients.push_back(IntervalFromRational(rational.num, rational.den));
        }
    }
    else
    {
        // Without declared ideal coefficients the quantized values are treated
        // as the exact-real target, so the coefficient-quantization term is 0.
        for (const std::uint32_t bits : contract.coefficientBits)
        {
            coefficients.push_back(IntervalExact(static_cast<double>(FloatFromBits(bits))));
        }
    }
    return coefficients;
}

CoefficientList QuantizedCoefficients(const CandidateModel& model)
{
    CoefficientList coefficients;
    coefficients.reserve(model.coefficients.size());
    for (const float c : model.coefficients)
    {
        coefficients.push_back(IntervalExact(static_cast<double>(c)));
    }
    return coefficients;
}

F32ChainTrace EvaluateModelIntervalF32(const CandidateModel& model, Interval x)
{
    F32ChainTrace trace;
    F32Interval h{model.coefficients[model.degree], model.coefficients[model.degree]};
    for (std::size_t k = model.coefficients.size() - 1; k-- > 0;)
    {
        // The input box stays a real interval: the proof covers every real x in
        // the box, and the FMA rounding step is the only binary32 rounding.
        h = FmaIntervalF32(h, x, model.coefficients[k]);
        trace.steps.push_back(h);
    }
    trace.rawResult = h;
    trace.result = model.clamp01 ? Clamp01F32(h) : h;
    return trace;
}

} // namespace metallic::shadermath
