#include "Report.hpp"

#include <cstdio>
#include <sstream>

namespace metallic::shadermath
{

namespace
{

std::string JsonEscape(const std::string& text)
{
    std::string escaped;
    escaped.reserve(text.size());
    for (const char c : text)
    {
        switch (c)
        {
        case '"': escaped += "\\\""; break;
        case '\\': escaped += "\\\\"; break;
        case '\n': escaped += "\\n"; break;
        case '\r': escaped += "\\r"; break;
        case '\t': escaped += "\\t"; break;
        default:
            if (static_cast<unsigned char>(c) < 0x20)
            {
                char buffer[8];
                std::snprintf(buffer, sizeof(buffer), "\\u%04x", c);
                escaped += buffer;
            }
            else
            {
                escaped += c;
            }
        }
    }
    return escaped;
}

std::string Quote(const std::string& text)
{
    return "\"" + JsonEscape(text) + "\"";
}

std::string Number(double value)
{
    if (value != value)
    {
        return std::string("null");
    }
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.17g", value);
    return std::string(buffer);
}

std::string IntervalText(Interval interval)
{
    std::ostringstream stream;
    stream << "[" << Number(interval.lo) << ", " << Number(interval.hi) << "]";
    return stream.str();
}

std::string IntervalJson(Interval interval)
{
    return "{\"lo\": " + Number(interval.lo) + ", \"hi\": " + Number(interval.hi) + "}";
}

std::string HullJson(const HullResult& hull)
{
    std::ostringstream stream;
    stream << "{\"hull\": " << IntervalJson(hull.hull) << ", \"completed\": "
           << (hull.completed ? "true" : "false") << ", \"boxes\": " << hull.boxes
           << ", \"depthUsed\": " << hull.depthUsed << "}";
    return stream.str();
}

} // namespace

ConsistencyChecks RunConsistencyChecks(const CertificationReport& certification,
    const SweepResults& sweep)
{
    ConsistencyChecks checks;
    // The double std::exp reference carries up to ~1 ulp (~1.1e-16 at 1.0);
    // allow a fixed 1e-12 observation slack that is still five orders of
    // magnitude below the certified margin.
    checks.observedErrorWithinCertified =
        sweep.maxObservedError <= certification.totalErrorBound.hi + 1e-12;
    checks.observedEvalWithinGamma = sweep.maxObservedEvalError <= certification.gammaBound.hi + 1e-12;
    checks.observedRangeWithinCertified =
        sweep.observedMinY >= certification.modelOutputRange.lo &&
        sweep.observedMaxY <= certification.modelOutputRange.hi;
    checks.worstEntryCertifiedContainsObserved = !sweep.worst.empty() &&
        sweep.worst.front().certifiedError.lo - 1e-12 <= sweep.worst.front().observedError &&
        sweep.worst.front().observedError <= sweep.worst.front().certifiedError.hi + 1e-12;
    return checks;
}

std::string BuildReportJson(const ValidationRun& run)
{
    const Contract& contract = *run.contract;
    const CertificationReport& cert = *run.certification;
    const SweepResults& sweep = *run.sweep;
    const ExportArtifacts& exports = *run.exports;

    std::ostringstream json;
    json << "{\n";

    json << "  \"identity\": {\n";
    json << "    \"id\": " << Quote(contract.id) << ",\n";
    json << "    \"contractHash\": " << Quote(contract.sourceHash) << ",\n";
    json << "    \"source\": " << Quote(contract.sourceName) << ",\n";
    json << "    \"referenceExpression\": " << Quote(contract.referenceExpression) << ",\n";
    json << "    \"referenceInterpretation\": " << Quote(contract.referenceInterpretation) << ",\n";
    json << "    \"model\": " << Quote(DescribeProofModel(contract)) << ",\n";
    json << "    \"environment\": " << Quote(run.environment) << ",\n";
    json << "    \"gitRevision\": " << Quote(run.gitRevision) << "\n";
    json << "  },\n";

    json << "  \"domain\": " << IntervalJson(cert.domainBox) << ",\n";

    json << "  \"certifiedErrorChain\": {\n";
    json << "    \"approximation\": " << HullJson(cert.approximationError) << ",\n";
    json << "    \"coefficients\": " << HullJson(cert.coefficientError) << ",\n";
    json << "    \"evaluation\": {\"bound\": " << IntervalJson(cert.gammaBound)
         << ", \"noUnderflow\": " << (cert.gammaNoUnderflow ? "true" : "false") << "},\n";
    json << "    \"candidateReal\": " << HullJson(cert.candidateRealError) << ",\n";
    json << "    \"total\": " << IntervalJson(cert.totalErrorBound) << ",\n";
    json << "    \"totalInfimum\": " << IntervalJson(cert.totalErrorInfimum) << "\n";
    json << "  },\n";

    json << "  \"certifiedProperties\": {\n";
    json << "    \"referenceRange\": " << IntervalJson(cert.referenceRange) << ",\n";
    json << "    \"realPolynomialRange\": " << IntervalJson(cert.realPolynomialRange) << ",\n";
    json << "    \"realPolynomialMonotoneDecreasing\": "
         << (cert.realPolynomialMonotoneDecreasing ? "true" : "false") << ",\n";
    json << "    \"modelOutputRangeRaw\": {\"lo\": " << Number(cert.modelOutputRangeRaw.lo)
         << ", \"hi\": " << Number(cert.modelOutputRangeRaw.hi) << "},\n";
    json << "    \"modelOutputRange\": {\"lo\": " << Number(cert.modelOutputRange.lo)
         << ", \"hi\": " << Number(cert.modelOutputRange.hi) << "},\n";
    json << "    \"modelRangeBoxes\": " << cert.modelRangeBoxes << ",\n";
    json << "    \"errorSign\": {\"nonNegativeFrom\": "
         << (cert.signNonNegativeOnUpperPart ? Number(cert.signCertifiedFrom)
                                             : std::string("null"))
         << ", \"globalFloor\": " << IntervalJson(cert.globalErrorFloor) << "}\n";
    json << "  },\n";

    json << "  \"requirements\": {\n";
    json << "    \"maxAbsoluteError\": {\"value\": " << Number(static_cast<double>(
        contract.maxAbsoluteError.num) / static_cast<double>(contract.maxAbsoluteError.den))
         << ", \"verdict\": " << Quote(ToString(cert.totalErrorVerdict)) << "},\n";
    json << "    \"outputRange\": {\"verdict\": " << Quote(ToString(cert.outputRangeVerdict)) << "},\n";
    json << "    \"endpointAtZero\": {\"verdict\": " << Quote(ToString(cert.endpointVerdict))
         << ", \"bits\": " << cert.endpointBits << ", \"value\": " << Number(cert.endpointValue)
         << "}\n";
    json << "  },\n";

    json << "  \"sweep\": {\n";
    json << "    \"mode\": " << Quote(sweep.exhaustive ? "exhaustive_binary32_grid" : "sampled") << ",\n";
    json << "    \"status\": " << Quote(ToString(sweep.exhaustive ? EvidenceLevel::ExhaustiveModelSweep
                                                                    : EvidenceLevel::SampledModelSweep))
         << ",\n";
    json << "    \"grid\": [" << sweep.gridLoBits << ", " << sweep.gridHiBits << "],\n";
    json << "    \"evaluatedInputs\": " << sweep.evaluatedInputs << ",\n";
    json << "    \"threads\": " << sweep.threadsUsed << ",\n";
    json << "    \"seconds\": " << Number(sweep.seconds) << ",\n";
    json << "    \"observedMaxAbsoluteError\": " << Number(sweep.maxObservedError) << ",\n";
    json << "    \"observedMaxEvalError\": " << Number(sweep.maxObservedEvalError) << ",\n";
    json << "    \"observedRange\": [" << Number(sweep.observedMinY) << ", "
         << Number(sweep.observedMaxY) << "],\n";
    json << "    \"monotonicityViolations\": " << sweep.monotonicityViolations << ",\n";
    json << "    \"worst\": [\n";
    for (std::size_t i = 0; i < sweep.worst.size() && i < 32; ++i)
    {
        const SweepWorstEntry& entry = sweep.worst[i];
        json << "      {\"inBits\": " << entry.inBits << ", \"outBits\": " << entry.outBits
             << ", \"observed\": " << Number(entry.observedError)
             << ", \"certified\": " << IntervalJson(entry.certifiedError) << "}"
             << (i + 1 < sweep.worst.size() && i + 1 < 32 ? "," : "") << "\n";
    }
    json << "    ]\n";
    json << "  },\n";

    json << "  \"consistencyChecks\": {\n";
    json << "    \"observedErrorWithinCertified\": "
         << (run.checks.observedErrorWithinCertified ? "true" : "false") << ",\n";
    json << "    \"observedEvalWithinGamma\": "
         << (run.checks.observedEvalWithinGamma ? "true" : "false") << ",\n";
    json << "    \"observedRangeWithinCertified\": "
         << (run.checks.observedRangeWithinCertified ? "true" : "false") << ",\n";
    json << "    \"worstEntryCertifiedContainsObserved\": "
         << (run.checks.worstEntryCertifiedContainsObserved ? "true" : "false") << "\n";
    json << "  },\n";

    json << "  \"exports\": {\n";
    json << "    \"sollya\": {\"path\": " << Quote(exports.sollya.path) << ", \"hash\": "
         << Quote(exports.sollya.contentHash) << ", \"executed\": "
         << (exports.sollya.executed ? "true" : "false") << ", \"note\": "
         << Quote(exports.sollya.note) << "},\n";
    json << "    \"gappa\": {\"path\": " << Quote(exports.gappa.path) << ", \"hash\": "
         << Quote(exports.gappa.contentHash) << ", \"executed\": "
         << (exports.gappa.executed ? "true" : "false") << ", \"note\": "
         << Quote(exports.gappa.note) << "},\n";
    json << "    \"slang\": {\"path\": " << Quote(exports.slang.path) << ", \"hash\": "
         << Quote(exports.slang.contentHash) << ", \"note\": " << Quote(exports.slang.note) << "},\n";
    json << "    \"gpuManifest\": {\"path\": " << Quote(exports.manifestPath) << ", \"hash\": "
         << Quote(exports.manifestHash) << ", \"entries\": " << exports.manifestEntryCount << "}\n";
    json << "  },\n";

    json << "  \"gpuValidation\": {\n";
    json << "    \"status\": \"NOT_RUN\",\n";
    json << "    \"note\": \"phase 2: run the generated Slang module through the RHI test runner and compare output bits against gpu-manifest.txt\"\n";
    json << "  }\n";

    json << "}";
    return json.str();
}

void PrintSummary(const ValidationRun& run)
{
    const Contract& contract = *run.contract;
    const CertificationReport& cert = *run.certification;
    const SweepResults& sweep = *run.sweep;

    std::printf("\n=== Metallic ShaderMathHarness ===\n");
    std::printf("contract: %s (hash %s)\n", contract.id.c_str(), contract.sourceHash.c_str());
    std::printf("model:    %s\n", DescribeProofModel(contract).c_str());
    std::printf("domain:   %s\n", IntervalText(cert.domainBox).c_str());

    std::printf("\ncertified error chain (rigorous, whole domain):\n");
    std::printf("  approximation  |P(c)-exp(-x)|  sup in %s   boxes=%llu completed=%s\n",
        IntervalText(cert.approximationError.hull).c_str(),
        static_cast<unsigned long long>(cert.approximationError.boxes),
        cert.approximationError.completed ? "yes" : "NO");
    std::printf("  coefficients   |P(c~)-P(c)|    sup in %s   boxes=%llu\n",
        IntervalText(cert.coefficientError.hull).c_str(),
        static_cast<unsigned long long>(cert.coefficientError.boxes));
    std::printf("  evaluation     |fl(P)-P(c~)|   bound %s   no-underflow=%s\n",
        IntervalText(cert.gammaBound).c_str(), cert.gammaNoUnderflow ? "yes" : "NO");
    std::printf("  total          |fl(P)-exp(-x)| bound <= %.6e   requirement %.6e   [%s]\n",
        cert.totalErrorBound.hi,
        static_cast<double>(contract.maxAbsoluteError.num) /
            static_cast<double>(contract.maxAbsoluteError.den),
        ToString(cert.totalErrorVerdict));

    std::printf("\nproperties:\n");
    std::printf("  model output range: [%.9g, %.9g]   verdict [%s]\n",
        cert.modelOutputRange.lo, cert.modelOutputRange.hi,
        ToString(cert.outputRangeVerdict));
    std::printf("  real polynomial range: %s   monotone decreasing: %s\n",
        IntervalText(cert.realPolynomialRange).c_str(),
        cert.realPolynomialMonotoneDecreasing ? "proved" : "not proved");
    std::printf("  endpoint tau=0 -> %.9g (bits 0x%08x)   [%s]\n", cert.endpointValue,
        cert.endpointBits, ToString(cert.endpointVerdict));
    if (cert.signNonNegativeOnUpperPart)
    {
        std::printf("  error sign: non-negative on [%g, domain hi]; global floor %s\n",
            cert.signCertifiedFrom, IntervalText(cert.globalErrorFloor).c_str());
    }
    else
    {
        std::printf("  error sign: not certified on any upper sub-interval (global floor %s)\n",
            IntervalText(cert.globalErrorFloor).c_str());
    }

    std::printf("\nsweep (%s, %llu inputs, %u threads, %.2fs):\n",
        sweep.exhaustive ? "exhaustive binary32 grid" : "sampled",
        static_cast<unsigned long long>(sweep.evaluatedInputs), sweep.threadsUsed, sweep.seconds);
    std::printf("  observed max |error|     = %.6e at tau bits 0x%08x\n", sweep.maxObservedError,
        sweep.worstInBits);
    std::printf("  observed max |eval err|  = %.6e\n", sweep.maxObservedEvalError);
    std::printf("  observed output range    = [%.9g, %.9g]\n", sweep.observedMinY, sweep.observedMaxY);
    std::printf("  monotonicity violations  = %llu\n",
        static_cast<unsigned long long>(sweep.monotonicityViolations));
    if (!sweep.worst.empty())
    {
        const SweepWorstEntry& worst = sweep.worst.front();
        std::printf("  worst input re-verified (rigorous oracle): observed %.6e in %s\n",
            worst.observedError, IntervalText(worst.certifiedError).c_str());
    }
    std::printf("  consistency: observed<=certified %s, eval<=gamma %s, range<=certified %s\n",
        run.checks.observedErrorWithinCertified ? "OK" : "FAIL",
        run.checks.observedEvalWithinGamma ? "OK" : "FAIL",
        run.checks.observedRangeWithinCertified ? "OK" : "FAIL");

    std::printf("\nexports: %s\n", run.exports->outDirectory.c_str());
    std::printf("  sollya script %s (%s)\n", run.exports->sollya.contentHash.c_str(),
        run.exports->sollya.executed ? "executed" : "NOT executed");
    std::printf("  gappa script  %s (%s)\n", run.exports->gappa.contentHash.c_str(),
        run.exports->gappa.executed ? "executed" : "NOT executed");
    std::printf("  slang module  %s\n", run.exports->slang.contentHash.c_str());
    std::printf("  gpu manifest  %s (%llu entries)\n", run.exports->manifestHash.c_str(),
        static_cast<unsigned long long>(run.exports->manifestEntryCount));

    std::printf("\ngpu validation: NOT_RUN (phase 2)\n");
    const bool proved = cert.totalErrorVerdict == EvidenceLevel::ProvedUnderModel &&
        cert.outputRangeVerdict == EvidenceLevel::ProvedUnderModel &&
        cert.endpointVerdict == EvidenceLevel::ProvedUnderModel && run.checks.AllPassed();
    std::printf("overall: %s%s\n", proved ? "PROVED_UNDER_MODEL" : "NOT PROVED",
        proved ? "" : " (see verdicts and consistency checks above)");
}

} // namespace metallic::shadermath
