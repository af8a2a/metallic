#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "Numeric.hpp"

namespace metallic::shadermath
{

// ---------------------------------------------------------------------------
// Versioned numeric validation contracts (see contracts/*.yaml).
//
// The loader parses a deliberately small YAML subset: nested mappings with
// two-space indentation, block sequences of scalars ("- item"), flow sequences
// of scalars ("[a, b]"), quoted or bare scalar values, and '#' comments.
// Unknown keys are rejected so contract typos fail loudly instead of silently
// changing what gets proven.
// ---------------------------------------------------------------------------

struct YamlNode
{
    enum class Kind
    {
        Scalar,
        Sequence,
        Mapping
    };

    Kind kind = Kind::Scalar;
    std::string scalar;
    std::vector<std::string> sequence;
    std::map<std::string, YamlNode> mapping;
    int line = 0;
};

struct YamlParseResult
{
    std::optional<YamlNode> root;
    std::string error;
};

YamlParseResult ParseYamlSubset(const std::string& text);

struct Contract
{
    std::string id;

    std::string referenceExpression;
    std::string referenceInterpretation;

    std::string inputName;
    std::string inputFormat;
    RationalValue domainLo;
    RationalValue domainHi;
    bool finiteOnly = true;

    std::string form;
    std::vector<std::uint32_t> coefficientBits;
    bool hasIdealCoefficients = false;
    std::vector<RationalValue> idealCoefficients;
    std::string postprocess;

    std::string proofOperations;
    std::string proofRounding;
    std::string proofSubnormals;
    std::string proofReassociation;

    RationalValue maxAbsoluteError;
    RationalValue outputRangeLo;
    RationalValue outputRangeHi;
    RationalValue endpointAtZero;

    std::string outsideDomainAction;

    std::string sourceName;
    std::string sourceHash;
};

std::optional<Contract> LoadContractText(const std::string& text, const std::string& sourceName,
    std::string& error);
std::optional<Contract> LoadContractFile(const std::string& path, std::string& error);

// Human-readable description of the proof model implied by the contract.
std::string DescribeProofModel(const Contract& contract);

} // namespace metallic::shadermath
