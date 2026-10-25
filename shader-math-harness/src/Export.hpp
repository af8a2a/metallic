#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "Contract.hpp"
#include "Model.hpp"
#include "Sweep.hpp"

namespace metallic::shadermath
{

// ---------------------------------------------------------------------------
// Artifact export: Sollya/Gappa proof scripts, a Slang module generated from
// the same candidate graph, and a GPU expectation manifest (input bits ->
// expected output bits) for the phase-2 RHI runner.
//
// Sollya and Gappa are optional external tools. When absent the scripts are
// still written and their status is reported as NOT_RUN; nothing is claimed
// from tools that did not execute.
// ---------------------------------------------------------------------------

struct ExportArtifact
{
    std::string path;
    std::string contentHash;
    bool executed = false;
    std::string toolOutput;
    std::string note;
};

struct ExportArtifacts
{
    std::string outDirectory;
    ExportArtifact sollya;
    ExportArtifact gappa;
    ExportArtifact slang;
    std::string manifestPath;
    std::string manifestHash;
    std::uint64_t manifestEntryCount = 0;
};

struct ExportOptions
{
    bool attemptSollya = false;
    bool attemptGappa = false;
    std::uint32_t randomEntryCount = 10000;
    std::uint32_t worstEntryCount = 100;
};

ExportArtifacts WriteExports(const Contract& contract, const CandidateModel& model,
    const SweepResults& sweep, const std::string& outDirectory, const ExportOptions& options);

// Deterministic manifest entries (worst inputs, representational boundaries,
// stratified random); exposed for the determinism self-check.
std::string BuildManifestEntriesText(const Contract& contract, const CandidateModel& model,
    const SweepResults& sweep, std::uint32_t randomEntryCount, std::uint32_t worstEntryCount);

std::string BuildSlangModule(const Contract& contract, const CandidateModel& model);
std::string BuildSollyaScript(const Contract& contract);
std::string BuildGappaScript(const Contract& contract, const CandidateModel& model);

} // namespace metallic::shadermath
