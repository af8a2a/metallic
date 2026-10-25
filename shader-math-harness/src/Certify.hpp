#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "Model.hpp"
#include "Numeric.hpp"

namespace metallic::shadermath
{

// ---------------------------------------------------------------------------
// Evidence levels. The distinctions matter: a proof under the stated floating
// point model is not a GPU observation, an exhaustive host sweep of the binary32
// grid is not a proof about arbitrary reals, and an observed maximum error is
// not a certified bound.
// ---------------------------------------------------------------------------

enum class EvidenceLevel
{
    NotRun,
    ProvedUnderModel,
    CertifiedBound,
    ExhaustiveModelSweep,
    SampledModelSweep,
    Inconclusive,
    Failed
};

const char* ToString(EvidenceLevel level);

// ---------------------------------------------------------------------------
// Certified hull of a real function over an interval using the mean-value
// form: for a box B with midpoint m,
//     g(x) in g(m) + g'(B)*(x - m)   for every x in B,
// where g'(B) must be an interval enclosure of g' over (slightly more than) B.
// The value term is evaluated on point boxes, so cancellation-heavy functions
// whose derivative is cheap to enclose certify at far coarser subdivision than
// naive interval evaluation. Subdivision is adaptive on enclosure width.
// ---------------------------------------------------------------------------

struct FunctionPair
{
    std::function<Interval(const Interval&)> value;      // evaluated at point boxes
    std::function<Interval(const Interval&)> derivative; // enclosure over boxes
};

struct HullResult
{
    Interval hull{0.0, 0.0};
    bool completed = false;
    std::uint64_t boxes = 0;
    std::uint32_t depthUsed = 0;
};

HullResult CertifiedHull(const FunctionPair& function, Interval domain, double targetWidth,
    std::uint32_t maxDepth, std::uint64_t maxBoxes);

// ---------------------------------------------------------------------------
// Full certification pass for the pilot contract family (exp(-tau)).
// ---------------------------------------------------------------------------

struct CertificationReport
{
    Interval domainBox{0.0, 0.0};

    // Certified hulls of the error-chain terms over the whole domain.
    HullResult approximationError; // P(ideal) - exp(-x)
    HullResult coefficientError;   // P(quantized) - P(ideal)
    HullResult candidateRealError; // P(quantized) - exp(-x)

    // gamma_n * sum_k |c~_k| * hi^k for the FMA Horner evaluation bound.
    Interval gammaBound{0.0, 0.0};
    bool gammaNoUnderflow = false; // every intermediate certified normal

    // Certified total: candidateRealError hull supremum + gamma bound.
    Interval totalErrorBound{0.0, 0.0};
    Interval totalErrorInfimum{0.0, 0.0};

    Interval referenceRange{0.0, 0.0};  // exp(-x) over the domain
    Interval realPolynomialRange{0.0, 0.0};
    bool realPolynomialMonotoneDecreasing = false;

    F32Interval modelOutputRange{0.0f, 0.0f};
    F32Interval modelOutputRangeRaw{0.0f, 0.0f};
    std::uint32_t modelRangeBoxes = 0;

    // Error-sign statement: one-sidedness certified on [signCertifiedFrom, hi]
    // when signNonNegativeOnUpperPart holds; everywhere below that the hull
    // still bounds the error from below by globalErrorFloor.
    bool signNonNegativeOnUpperPart = false;
    double signCertifiedFrom = 0.0;
    Interval globalErrorFloor{0.0, 0.0};

    // Requirement verdicts.
    EvidenceLevel totalErrorVerdict = EvidenceLevel::NotRun;
    EvidenceLevel outputRangeVerdict = EvidenceLevel::NotRun;
    EvidenceLevel endpointVerdict = EvidenceLevel::NotRun;
    float endpointValue = 0.0f;
    std::uint32_t endpointBits = 0;

    bool completed() const
    {
        return approximationError.completed && coefficientError.completed && candidateRealError.completed;
    }
};

CertificationReport RunCertification(const Contract& contract, const CandidateModel& model);

} // namespace metallic::shadermath
