#pragma once

#include <string>

#include "Certify.hpp"
#include "Contract.hpp"
#include "Export.hpp"
#include "Model.hpp"
#include "Sweep.hpp"

namespace metallic::shadermath
{

struct ConsistencyChecks
{
    bool observedErrorWithinCertified = false;
    bool observedEvalWithinGamma = false;
    bool observedRangeWithinCertified = false;
    bool worstEntryCertifiedContainsObserved = false;

    bool AllPassed() const
    {
        return observedErrorWithinCertified && observedEvalWithinGamma && observedRangeWithinCertified &&
            worstEntryCertifiedContainsObserved;
    }
};

struct ValidationRun
{
    const Contract* contract = nullptr;
    const CandidateModel* model = nullptr;
    const CertificationReport* certification = nullptr;
    const SweepResults* sweep = nullptr;
    const ExportArtifacts* exports = nullptr;
    ConsistencyChecks checks;
    std::string environment;
    std::string gitRevision;
    double totalSeconds = 0.0;
};

ConsistencyChecks RunConsistencyChecks(const CertificationReport& certification,
    const SweepResults& sweep);

std::string BuildReportJson(const ValidationRun& run);
void PrintSummary(const ValidationRun& run);

} // namespace metallic::shadermath
