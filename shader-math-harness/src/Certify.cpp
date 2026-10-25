#include "Certify.hpp"

#include <algorithm>
#include <cassert>
#include <utility>

namespace metallic::shadermath
{

const char* ToString(EvidenceLevel level)
{
    switch (level)
    {
    case EvidenceLevel::NotRun: return "NOT_RUN";
    case EvidenceLevel::ProvedUnderModel: return "PROVED_UNDER_MODEL";
    case EvidenceLevel::CertifiedBound: return "CERTIFIED_BOUND";
    case EvidenceLevel::ExhaustiveModelSweep: return "EXHAUSTIVE_MODEL_SWEEP";
    case EvidenceLevel::SampledModelSweep: return "SAMPLED_MODEL_SWEEP";
    case EvidenceLevel::Inconclusive: return "INCONCLUSIVE";
    case EvidenceLevel::Failed: return "FAILED";
    }
    return "?";
}

namespace
{

// The derivative enclosure is requested over a box inflated by two ulps so the
// rounded midpoint used by the mean-value form is always interior.
Interval Inflate(Interval box)
{
    return Interval{
        std::nextafter(box.lo, -std::numeric_limits<double>::infinity()),
        std::nextafter(box.hi, std::numeric_limits<double>::infinity())};
}

// Exact point interval when the rational is exactly representable as a double
// (verified through the exact fma residual), widened otherwise. Requirement
// endpoints like 0, 1, and 1/4 are dyadic; comparing against their exact
// values keeps outputs that sit exactly on a requirement bound acceptable.
// The exactness test is exact for the magnitudes used by contracts
// (|num|, |den| < 2^53, quotient well inside the double exponent range).
Interval RationalIntervalExactAware(std::int64_t num, std::int64_t den)
{
    assert(den != 0);
    if (den < 0)
    {
        num = -num;
        den = -den;
    }
    const double quotient = static_cast<double>(num) / static_cast<double>(den);
    const double residual = std::fma(quotient, static_cast<double>(den), -static_cast<double>(num));
    if (std::isfinite(quotient) && residual == 0.0)
    {
        return IntervalExact(quotient);
    }
    return IntervalFromRational(num, den);
}

} // namespace

HullResult CertifiedHull(const FunctionPair& function, Interval domain, double targetWidth,
    std::uint32_t maxDepth, std::uint64_t maxBoxes)
{
    HullResult result;
    result.completed = true;
    Interval hull{std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()};

    std::vector<std::pair<Interval, std::uint32_t>> pending;
    pending.emplace_back(domain, 0);

    while (!pending.empty())
    {
        const auto [box, depth] = pending.back();
        pending.pop_back();

        const double midpoint = 0.5 * (box.lo + box.hi);
        const Interval offset = Sub(box, IntervalExact(midpoint));
        const Interval value = function.value(IntervalExact(midpoint));
        const Interval slope = function.derivative(Inflate(box));
        const Interval enclosure = Add(value, Mul(slope, offset));

        ++result.boxes;
        result.depthUsed = std::max(result.depthUsed, depth);

        const double width = enclosure.hi - enclosure.lo;
        if (width <= targetWidth)
        {
            // Only accepted leaf enclosures form the hull; a subdivided box is
            // fully covered by its children and would only loosen the bound.
            hull = Hull(hull, enclosure);
            continue;
        }
        if (depth >= maxDepth || result.boxes + pending.size() >= maxBoxes)
        {
            result.completed = false;
            hull = Hull(hull, enclosure);
            continue;
        }
        const double middle = midpoint;
        pending.emplace_back(Interval{box.lo, middle}, depth + 1);
        pending.emplace_back(Interval{middle, box.hi}, depth + 1);
    }

    result.hull = hull;
    return result;
}

CertificationReport RunCertification(const Contract& contract, const CandidateModel& model)
{
    CertificationReport report;

    const Interval domainLo = RationalIntervalExactAware(contract.domainLo.num, contract.domainLo.den);
    const Interval domainHi = RationalIntervalExactAware(contract.domainHi.num, contract.domainHi.den);
    report.domainBox = Interval{domainLo.lo, domainHi.hi};

    const CoefficientList ideal = IdealCoefficients(contract);
    const CoefficientList quantized = QuantizedCoefficients(model);
    const CoefficientList idealDerivative = PolynomialDerivative(ideal);
    const CoefficientList quantizedDerivative = PolynomialDerivative(quantized);

    // --- Error chain ------------------------------------------------------

    // P(ideal) - exp(-x); derivative P'(ideal)(x) + exp(-x).
    {
        FunctionPair pair;
        pair.value = [&ideal](const Interval& x)
        {
            return Sub(EvaluatePolynomialReal(ideal, x), ExpNegativeSmall(x));
        };
        pair.derivative = [&ideal, &idealDerivative](const Interval& box)
        {
            return Add(EvaluatePolynomialReal(idealDerivative, box), ExpNegativeSmall(box));
        };
        report.approximationError = CertifiedHull(pair, report.domainBox, 2e-8, 48, 2'000'000);
    }

    // P(quantized) - P(ideal); derivative of the difference.
    {
        FunctionPair pair;
        pair.value = [&ideal, &quantized](const Interval& x)
        {
            return Sub(EvaluatePolynomialReal(quantized, x), EvaluatePolynomialReal(ideal, x));
        };
        pair.derivative = [&idealDerivative, &quantizedDerivative](const Interval& box)
        {
            return Sub(EvaluatePolynomialReal(quantizedDerivative, box),
                EvaluatePolynomialReal(idealDerivative, box));
        };
        report.coefficientError = CertifiedHull(pair, report.domainBox, 1e-12, 48, 2'000'000);
    }

    // P(quantized) - exp(-x).
    {
        FunctionPair pair;
        pair.value = [&quantized](const Interval& x)
        {
            return Sub(EvaluatePolynomialReal(quantized, x), ExpNegativeSmall(x));
        };
        pair.derivative = [&quantized, &quantizedDerivative](const Interval& box)
        {
            return Add(EvaluatePolynomialReal(quantizedDerivative, box), ExpNegativeSmall(box));
        };
        report.candidateRealError = CertifiedHull(pair, report.domainBox, 2e-8, 48, 2'000'000);
    }

    // --- Floating point evaluation bound (gamma form) ----------------------
    // With n single-rounding FMA steps and unit roundoff u = 2^-24 for
    // binary32, |fl(p) - p| <= gamma_n * sum_k |c~_k| * hi^k on x in [lo, hi],
    // provided no intermediate is subnormal (checked below with the F32 chain).
    {
        const std::uint32_t n = static_cast<std::uint32_t>(model.degree);
        const Interval unitRoundoff = IntervalFromRational(1, 1 << 24);
        const Interval nU = Scale(unitRoundoff, static_cast<double>(n));
        const Interval denominator = Sub(IntervalExact(1.0), nU);
        assert(denominator.lo > 0.0);
        const Interval gamma = Div(nU, denominator);

        Interval weightedSum = IntervalExact(0.0);
        Interval power = IntervalExact(1.0);
        const Interval hiBox = IntervalExact(report.domainBox.hi);
        for (std::size_t k = 0; k < quantized.size(); ++k)
        {
            weightedSum = Add(weightedSum, Mul(Abs(quantized[k]), power));
            power = Mul(power, hiBox);
        }
        report.gammaBound = Mul(gamma, weightedSum);
    }

    // --- Reference and polynomial ranges ----------------------------------

    report.referenceRange = ExpNegativeSmall(report.domainBox);

    // Monotonicity of the real polynomial via its derivative hull, then tight
    // endpoints; otherwise a coarse hull.
    {
        FunctionPair derivativePair;
        derivativePair.value = [&idealDerivative](const Interval& x)
        {
            return EvaluatePolynomialReal(idealDerivative, x);
        };
        derivativePair.derivative = [&ideal](const Interval& box)
        {
            CoefficientList second = PolynomialDerivative(PolynomialDerivative(ideal));
            return EvaluatePolynomialReal(second, box);
        };
        const HullResult derivativeHull =
            CertifiedHull(derivativePair, report.domainBox, 5e-3, 48, 500'000);
        if (derivativeHull.completed && derivativeHull.hull.hi < 0.0)
        {
            report.realPolynomialMonotoneDecreasing = true;
            report.realPolynomialRange = Hull(
                EvaluatePolynomialReal(ideal, IntervalExact(report.domainBox.hi)),
                EvaluatePolynomialReal(ideal, IntervalExact(report.domainBox.lo)));
        }
        else
        {
            FunctionPair pair;
            pair.value = [&ideal](const Interval& x)
            {
                return EvaluatePolynomialReal(ideal, x);
            };
            pair.derivative = [&idealDerivative](const Interval& box)
            {
                return EvaluatePolynomialReal(idealDerivative, box);
            };
            report.realPolynomialRange = CertifiedHull(pair, report.domainBox, 1e-6, 48, 100'000).hull;
        }
    }

    // --- Binary32 model range over the real domain -------------------------

    {
        constexpr std::uint32_t kBoxes = 1u << 20;
        F32Interval raw{std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()};
        F32Interval clamped{std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()};
        bool noUnderflow = true;
        const float kMinNormal = 1.17549435e-38f;
        const double span = report.domainBox.hi - report.domainBox.lo;
        for (std::uint32_t i = 0; i < kBoxes; ++i)
        {
            const double a = report.domainBox.lo + span * static_cast<double>(i) / kBoxes;
            const double b = report.domainBox.lo + span * static_cast<double>(i + 1) / kBoxes;
            const F32ChainTrace trace = EvaluateModelIntervalF32(model, Interval{a, b});
            raw = F32Interval{(raw.lo < trace.rawResult.lo) ? raw.lo : trace.rawResult.lo,
                (raw.hi > trace.rawResult.hi) ? raw.hi : trace.rawResult.hi};
            clamped = F32Interval{(clamped.lo < trace.result.lo) ? clamped.lo : trace.result.lo,
                (clamped.hi > trace.result.hi) ? clamped.hi : trace.result.hi};
            for (const F32Interval& step : trace.steps)
            {
                const bool normal = (step.lo >= kMinNormal || step.hi <= -kMinNormal);
                if (!normal)
                {
                    noUnderflow = false;
                }
            }
        }
        report.modelOutputRangeRaw = raw;
        report.modelOutputRange = clamped;
        report.modelRangeBoxes = kBoxes;
        report.gammaNoUnderflow = noUnderflow;
    }

    // --- Error-sign statement ----------------------------------------------
    // Near tau = 0 the true error is below the interval noise floor, so global
    // one-sidedness is certified only on an upper sub-interval; below that the
    // hull provides the (tiny) certified floor everywhere.
    report.globalErrorFloor = Interval{report.candidateRealError.hull.lo, report.candidateRealError.hull.lo};
    {
        const double hi = report.domainBox.hi;
        const double step = hi / 16.0;
        double certifiedFrom = hi;
        bool any = false;
        for (int i = 15; i >= 1; --i)
        {
            const double from = step * static_cast<double>(i);
            FunctionPair pair;
            pair.value = [&quantized](const Interval& x)
            {
                return Sub(EvaluatePolynomialReal(quantized, x), ExpNegativeSmall(x));
            };
            pair.derivative = [&quantized, &quantizedDerivative](const Interval& box)
            {
                return Add(EvaluatePolynomialReal(quantizedDerivative, box), ExpNegativeSmall(box));
            };
            const HullResult upper = CertifiedHull(pair, Interval{from, hi}, 2e-8, 48, 500'000);
            if (upper.completed && upper.hull.lo >= 0.0)
            {
                certifiedFrom = from;
                any = true;
            }
        }
        report.signNonNegativeOnUpperPart = any;
        report.signCertifiedFrom = certifiedFrom;
    }

    // --- Total and requirement verdicts ------------------------------------

    const Interval totalUpper = Add(IntervalExact(report.candidateRealError.hull.hi), report.gammaBound);
    report.totalErrorBound = totalUpper;
    report.totalErrorInfimum = Interval{report.candidateRealError.hull.lo,
        report.candidateRealError.hull.lo};

    {
        const Interval requirement = RationalIntervalExactAware(contract.maxAbsoluteError.num,
            contract.maxAbsoluteError.den);
        if (report.completed() && report.gammaNoUnderflow && totalUpper.hi <= requirement.lo)
        {
            report.totalErrorVerdict = EvidenceLevel::ProvedUnderModel;
        }
        else if (report.candidateRealError.hull.lo > requirement.hi)
        {
            report.totalErrorVerdict = EvidenceLevel::Failed;
        }
        else
        {
            report.totalErrorVerdict = EvidenceLevel::Inconclusive;
        }
    }

    {
        const Interval requiredLo = RationalIntervalExactAware(contract.outputRangeLo.num,
            contract.outputRangeLo.den);
        const Interval requiredHi = RationalIntervalExactAware(contract.outputRangeHi.num,
            contract.outputRangeHi.den);
        const bool inside = static_cast<double>(report.modelOutputRange.lo) >= requiredLo.hi &&
            static_cast<double>(report.modelOutputRange.hi) <= requiredHi.lo;
        const bool outside = static_cast<double>(report.modelOutputRange.hi) < requiredLo.lo ||
            static_cast<double>(report.modelOutputRange.lo) > requiredHi.hi;
        report.outputRangeVerdict = inside ? EvidenceLevel::ProvedUnderModel
            : (outside ? EvidenceLevel::Failed : EvidenceLevel::Inconclusive);
    }

    {
        const float x0 = 0.0f;
        report.endpointValue = EvaluateModelF32(model, x0);
        report.endpointBits = BitsFromFloat(report.endpointValue);
        const Interval expected = RationalIntervalExactAware(contract.endpointAtZero.num,
            contract.endpointAtZero.den);
        const double value = static_cast<double>(report.endpointValue);
        report.endpointVerdict = (value >= expected.lo && value <= expected.hi)
            ? EvidenceLevel::ProvedUnderModel
            : EvidenceLevel::Failed;
    }

    return report;
}

} // namespace metallic::shadermath
