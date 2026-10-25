#include "Contract.hpp"

#include <cassert>
#include <fstream>
#include <sstream>

namespace metallic::shadermath
{

namespace
{

struct SourceLine
{
    int indent = 0;
    std::string text;
    int number = 0;
};

bool IsSpace(char c)
{
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

std::string Trim(std::string_view text)
{
    size_t begin = 0;
    size_t end = text.size();
    while (begin < end && IsSpace(text[begin]))
    {
        ++begin;
    }
    while (end > begin && IsSpace(text[end - 1]))
    {
        --end;
    }
    return std::string(text.substr(begin, end - begin));
}

std::string StripComment(std::string_view line)
{
    bool inSingleQuote = false;
    bool inDoubleQuote = false;
    for (size_t i = 0; i < line.size(); ++i)
    {
        const char c = line[i];
        if (c == '\'' && !inDoubleQuote)
        {
            inSingleQuote = !inSingleQuote;
        }
        else if (c == '"' && !inSingleQuote)
        {
            inDoubleQuote = !inDoubleQuote;
        }
        else if (c == '#' && !inSingleQuote && !inDoubleQuote)
        {
            return std::string(line.substr(0, i));
        }
    }
    return std::string(line);
}

std::optional<std::string> Unquote(const std::string& text)
{
    if (text.size() >= 2 && text.front() == text.back() &&
        (text.front() == '"' || text.front() == '\''))
    {
        return text.substr(1, text.size() - 2);
    }
    if (text.size() >= 2 && ((text.front() == '"' && text.back() == '\'') ||
        (text.front() == '\'' && text.back() == '"')))
    {
        return std::nullopt;
    }
    return text;
}

size_t FindTopLevel(std::string_view text, char separator)
{
    bool inSingleQuote = false;
    bool inDoubleQuote = false;
    for (size_t i = 0; i < text.size(); ++i)
    {
        const char c = text[i];
        if (c == '\'' && !inDoubleQuote)
        {
            inSingleQuote = !inSingleQuote;
        }
        else if (c == '"' && !inSingleQuote)
        {
            inDoubleQuote = !inDoubleQuote;
        }
        else if (c == separator && !inSingleQuote && !inDoubleQuote)
        {
            return i;
        }
    }
    return std::string_view::npos;
}

class YamlParser
{
public:
    explicit YamlParser(const std::vector<SourceLine>& lines)
        : lines_(lines)
    {
    }

    bool Parse(YamlNode& root)
    {
        if (lines_.empty())
        {
            error_ = "document is empty";
            return false;
        }
        const int indent = lines_[0].indent;
        if (indent != 0)
        {
            error_ = "document must start at column 0";
            return false;
        }
        return ParseBlock(indent, root);
    }

    const std::string& Error() const
    {
        return error_;
    }

private:
    bool Fail(const std::string& message, int line)
    {
        std::ostringstream stream;
        stream << "line " << line << ": " << message;
        error_ = stream.str();
        return false;
    }

    bool ParseBlock(int indent, YamlNode& out)
    {
        const std::string& first = lines_[next_].text;
        const bool isSequence = first == "-" || (first.size() >= 2 && first[0] == '-' && first[1] == ' ');
        return isSequence ? ParseSequence(indent, out) : ParseMapping(indent, out);
    }

    bool ParseSequence(int indent, YamlNode& out)
    {
        out.kind = YamlNode::Kind::Sequence;
        while (next_ < lines_.size())
        {
            const SourceLine& line = lines_[next_];
            if (line.indent < indent)
            {
                return true;
            }
            if (line.indent > indent)
            {
                return Fail("unexpected indentation inside sequence", line.number);
            }
            if (!(line.text == "-" || (line.text.size() >= 2 && line.text[0] == '-' && line.text[1] == ' ')))
            {
                return true;
            }
            std::string item = (line.text == "-") ? std::string() : Trim(std::string_view(line.text).substr(2));
            if (item.empty())
            {
                return Fail("nested sequence items are not supported", line.number);
            }
            if (item.front() == '[')
            {
                return Fail("nested sequences are not supported", line.number);
            }
            std::optional<std::string> value = Unquote(item);
            if (!value)
            {
                return Fail("mismatched quotes", line.number);
            }
            out.sequence.push_back(*value);
            ++next_;
        }
        return true;
    }

    bool ParseMapping(int indent, YamlNode& out)
    {
        out.kind = YamlNode::Kind::Mapping;
        while (next_ < lines_.size())
        {
            const SourceLine& line = lines_[next_];
            if (line.indent < indent)
            {
                return true;
            }
            if (line.indent > indent)
            {
                return Fail("unexpected indentation", line.number);
            }
            if (line.text == "-" || (line.text.size() >= 2 && line.text[0] == '-' && line.text[1] == ' '))
            {
                return Fail("sequence item where a mapping key was expected", line.number);
            }
            const size_t colon = FindTopLevel(line.text, ':');
            if (colon == std::string_view::npos || colon == 0)
            {
                return Fail("expected 'key: value' or 'key:'", line.number);
            }
            const std::string key = Trim(std::string_view(line.text).substr(0, colon));
            std::string value = Trim(std::string_view(line.text).substr(colon + 1));
            std::optional<std::string> unquotedKey = Unquote(key);
            if (!unquotedKey || unquotedKey->empty())
            {
                return Fail("invalid mapping key", line.number);
            }
            if (out.mapping.count(*unquotedKey) != 0)
            {
                return Fail("duplicate key '" + *unquotedKey + "'", line.number);
            }

            YamlNode child;
            child.line = line.number;
            if (value.empty())
            {
                ++next_;
                if (next_ >= lines_.size() || lines_[next_].indent <= indent)
                {
                    return Fail("expected an indented block after '" + *unquotedKey + ":'", line.number);
                }
                if (!ParseBlock(lines_[next_].indent, child))
                {
                    return false;
                }
            }
            else if (value.front() == '[')
            {
                if (value.back() != ']')
                {
                    return Fail("unterminated flow sequence", line.number);
                }
                child.kind = YamlNode::Kind::Sequence;
                const std::string body = value.substr(1, value.size() - 2);
                size_t position = 0;
                while (position <= body.size())
                {
                    size_t comma = FindTopLevel(std::string_view(body).substr(position), ',');
                    std::string item;
                    if (comma == std::string_view::npos)
                    {
                        item = Trim(std::string_view(body).substr(position));
                        position = body.size() + 1;
                    }
                    else
                    {
                        item = Trim(std::string_view(body).substr(position, comma));
                        position += comma + 1;
                    }
                    if (item.empty())
                    {
                        if (position > body.size())
                        {
                            break;
                        }
                        return Fail("empty flow sequence item", line.number);
                    }
                    std::optional<std::string> unquoted = Unquote(item);
                    if (!unquoted)
                    {
                        return Fail("mismatched quotes in flow sequence", line.number);
                    }
                    child.sequence.push_back(*unquoted);
                }
                ++next_;
            }
            else
            {
                std::optional<std::string> unquoted = Unquote(value);
                if (!unquoted)
                {
                    return Fail("mismatched quotes", line.number);
                }
                child.kind = YamlNode::Kind::Scalar;
                child.scalar = *unquoted;
                ++next_;
            }
            out.mapping[*unquotedKey] = child;
        }
        return true;
    }

    const std::vector<SourceLine>& lines_;
    size_t next_ = 0;
    std::string error_;
};

bool IsHexDigit(char c)
{
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

std::optional<std::uint32_t> ParseBits32(const std::string& text)
{
    if (text.size() < 3 || text.size() > 10 || text[0] != '0' || (text[1] != 'x' && text[1] != 'X'))
    {
        return std::nullopt;
    }
    std::uint64_t value = 0;
    for (size_t i = 2; i < text.size(); ++i)
    {
        if (!IsHexDigit(text[i]))
        {
            return std::nullopt;
        }
        char c = text[i];
        int digit = (c <= '9') ? (c - '0') : ((c | 0x20) - 'a' + 10);
        value = value * 16 + static_cast<std::uint32_t>(digit);
        if (value > 0xFFFFFFFFULL)
        {
            return std::nullopt;
        }
    }
    return static_cast<std::uint32_t>(value);
}

const YamlNode& RequireMapping(const YamlNode& parent, const std::string& name,
    const std::vector<std::string>& allowedKeys, std::string& error)
{
    static const YamlNode kEmpty{};
    auto it = parent.mapping.find(name);
    if (it == parent.mapping.end())
    {
        error = "missing section '" + name + "'";
        return kEmpty;
    }
    if (it->second.kind != YamlNode::Kind::Mapping)
    {
        error = "section '" + name + "' must be a mapping";
        return kEmpty;
    }
    for (const auto& [key, value] : it->second.mapping)
    {
        bool known = false;
        for (const std::string& allowed : allowedKeys)
        {
            if (key == allowed)
            {
                known = true;
                break;
            }
        }
        if (!known)
        {
            std::ostringstream stream;
            stream << "unknown key '" << key << "' in '" << name
                   << "' (allowed keys:";
            for (const std::string& allowed : allowedKeys)
            {
                stream << " " << allowed;
            }
            stream << ")";
            error = stream.str();
            return kEmpty;
        }
    }
    return it->second;
}

std::optional<std::string> ScalarOf(const YamlNode& parent, const std::string& name, std::string& error)
{
    auto it = parent.mapping.find(name);
    if (it == parent.mapping.end())
    {
        error = "missing key '" + name + "' in '" + parent.scalar + "'";
        return std::nullopt;
    }
    if (it->second.kind != YamlNode::Kind::Scalar)
    {
        error = "key '" + name + "' must be a scalar";
        return std::nullopt;
    }
    return it->second.scalar;
}

std::optional<std::vector<std::string>> SequenceOf(const YamlNode& parent, const std::string& name,
    std::string& error)
{
    auto it = parent.mapping.find(name);
    if (it == parent.mapping.end())
    {
        error = "missing key '" + name + "'";
        return std::nullopt;
    }
    if (it->second.kind != YamlNode::Kind::Sequence)
    {
        error = "key '" + name + "' must be a sequence";
        return std::nullopt;
    }
    return it->second.sequence;
}

std::optional<RationalValue> RationalOf(const YamlNode& parent, const std::string& name, std::string& error)
{
    std::optional<std::string> text = ScalarOf(parent, name, error);
    if (!text)
    {
        return std::nullopt;
    }
    std::optional<RationalValue> value = ParseRational(*text);
    if (!value)
    {
        error = "key '" + name + "' is not a supported number: '" + *text + "'";
        return std::nullopt;
    }
    return value;
}

std::optional<RationalValue> RationalFromSequence(const std::vector<std::string>& items,
    const std::string& name, size_t index, std::string& error)
{
    if (items.size() <= index)
    {
        error = "sequence '" + name + "' needs at least " + std::to_string(index + 1) + " entries";
        return std::nullopt;
    }
    std::optional<RationalValue> value = ParseRational(items[index]);
    if (!value)
    {
        error = "entry " + std::to_string(index) + " of '" + name + "' is not a supported number: '"
            + items[index] + "'";
        return std::nullopt;
    }
    return value;
}

bool ExpectValue(const std::string& actual, const std::string& expected, const std::string& name,
    std::string& error)
{
    if (actual != expected)
    {
        error = "key '" + name + "' must be '" + expected + "' but is '" + actual + "'";
        return false;
    }
    return true;
}

} // namespace

YamlParseResult ParseYamlSubset(const std::string& text)
{
    YamlParseResult result;
    std::vector<SourceLine> lines;
    std::istringstream stream(text);
    std::string raw;
    int number = 0;
    while (std::getline(stream, raw))
    {
        ++number;
        std::string content = StripComment(raw);
        size_t indent = 0;
        while (indent < content.size() && content[indent] == ' ')
        {
            ++indent;
        }
        if (indent < content.size() && content[indent] == '\t')
        {
            result.error = "line " + std::to_string(number) + ": tabs are not allowed for indentation";
            return result;
        }
        std::string rest = Trim(std::string_view(content).substr(indent));
        if (rest.empty())
        {
            continue;
        }
        lines.push_back(SourceLine{static_cast<int>(indent), rest, number});
    }

    if (lines.empty())
    {
        result.error = "document is empty";
        return result;
    }

    YamlParser parser(lines);
    YamlNode root;
    if (!parser.Parse(root))
    {
        result.error = parser.Error();
        return result;
    }
    if (parser.Error().empty() == false)
    {
        result.error = parser.Error();
        return result;
    }
    result.root = std::move(root);
    return result;
}

std::optional<Contract> LoadContractText(const std::string& text, const std::string& sourceName,
    std::string& error)
{
    YamlParseResult parsed = ParseYamlSubset(text);
    if (!parsed.root)
    {
        error = parsed.error;
        return std::nullopt;
    }
    if (parsed.root->kind != YamlNode::Kind::Mapping)
    {
        error = "contract root must be a mapping";
        return std::nullopt;
    }

    const std::vector<std::string> kRootKeys = {"id", "reference", "inputs", "candidate",
        "proof_model", "requirements", "outside_domain"};
    const YamlNode& root = *parsed.root;
    for (const auto& [key, value] : root.mapping)
    {
        bool known = false;
        for (const std::string& allowed : kRootKeys)
        {
            if (key == allowed)
            {
                known = true;
                break;
            }
        }
        if (!known)
        {
            error = "unknown key '" + key + "' at contract root";
            return std::nullopt;
        }
    }

    Contract contract;
    contract.sourceName = sourceName;
    contract.sourceHash = HashToHex(Fnv1a64(text));

    std::optional<std::string> id = ScalarOf(root, "id", error);
    if (!id || id->empty())
    {
        return std::nullopt;
    }
    contract.id = *id;

    const YamlNode& reference = RequireMapping(root, "reference", {"expression", "interpretation"}, error);
    if (!error.empty())
    {
        return std::nullopt;
    }
    std::optional<std::string> expression = ScalarOf(reference, "expression", error);
    if (!expression)
    {
        return std::nullopt;
    }
    std::optional<std::string> interpretation = ScalarOf(reference, "interpretation", error);
    if (!interpretation)
    {
        return std::nullopt;
    }
    contract.referenceExpression = *expression;
    contract.referenceInterpretation = *interpretation;
    if (!ExpectValue(contract.referenceExpression, "exp(-tau)", "reference.expression", error) ||
        !ExpectValue(contract.referenceInterpretation, "exact_real_at_binary32_input",
            "reference.interpretation", error))
    {
        return std::nullopt;
    }

    const YamlNode& inputs = RequireMapping(root, "inputs", {"tau"}, error);
    if (!error.empty())
    {
        return std::nullopt;
    }
    const YamlNode& tau = RequireMapping(inputs, "tau", {"format", "domain", "finite_only"}, error);
    if (!error.empty())
    {
        return std::nullopt;
    }
    contract.inputName = "tau";
    std::optional<std::string> format = ScalarOf(tau, "format", error);
    if (!format || !ExpectValue(*format, "binary32", "inputs.tau.format", error))
    {
        return std::nullopt;
    }
    contract.inputFormat = *format;
    std::optional<std::vector<std::string>> domain = SequenceOf(tau, "domain", error);
    if (!domain)
    {
        return std::nullopt;
    }
    std::optional<RationalValue> domainLo = RationalFromSequence(*domain, "domain", 0, error);
    if (!domainLo)
    {
        return std::nullopt;
    }
    std::optional<RationalValue> domainHi = RationalFromSequence(*domain, "domain", 1, error);
    if (!domainHi)
    {
        return std::nullopt;
    }
    if (static_cast<double>(domainLo->num) / static_cast<double>(domainLo->den) >
        static_cast<double>(domainHi->num) / static_cast<double>(domainHi->den))
    {
        error = "inputs.tau.domain lower bound exceeds upper bound";
        return std::nullopt;
    }
    contract.domainLo = *domainLo;
    contract.domainHi = *domainHi;
    std::optional<std::string> finiteOnly = ScalarOf(tau, "finite_only", error);
    if (!finiteOnly || !ExpectValue(*finiteOnly, "true", "inputs.tau.finite_only", error))
    {
        return std::nullopt;
    }
    contract.finiteOnly = true;

    const YamlNode& candidate = RequireMapping(root, "candidate",
        {"form", "coefficient_bits", "ideal_coefficients", "postprocess"}, error);
    if (!error.empty())
    {
        return std::nullopt;
    }
    std::optional<std::string> form = ScalarOf(candidate, "form", error);
    if (!form || !ExpectValue(*form, "horner_degree_4", "candidate.form", error))
    {
        return std::nullopt;
    }
    contract.form = *form;
    std::optional<std::vector<std::string>> bits = SequenceOf(candidate, "coefficient_bits", error);
    if (!bits)
    {
        return std::nullopt;
    }
    if (bits->size() != 5)
    {
        error = "candidate.coefficient_bits must contain 5 entries for horner_degree_4";
        return std::nullopt;
    }
    for (const std::string& item : *bits)
    {
        std::optional<std::uint32_t> value = ParseBits32(item);
        if (!value)
        {
            error = "candidate.coefficient_bits entry '" + item + "' is not a 0x-prefixed bit pattern";
            return std::nullopt;
        }
        contract.coefficientBits.push_back(*value);
    }
    auto ideal = candidate.mapping.find("ideal_coefficients");
    if (ideal != candidate.mapping.end())
    {
        if (ideal->second.kind != YamlNode::Kind::Sequence || ideal->second.sequence.size() != 5)
        {
            error = "candidate.ideal_coefficients must be a sequence of 5 exact numbers";
            return std::nullopt;
        }
        for (const std::string& item : ideal->second.sequence)
        {
            std::optional<RationalValue> value = ParseRational(item);
            if (!value)
            {
                error = "candidate.ideal_coefficients entry '" + item + "' is not a supported number";
                return std::nullopt;
            }
            contract.idealCoefficients.push_back(*value);
        }
        contract.hasIdealCoefficients = true;
    }
    std::optional<std::string> postprocess = ScalarOf(candidate, "postprocess", error);
    if (!postprocess)
    {
        return std::nullopt;
    }
    if (*postprocess != "clamp_0_1" && *postprocess != "none")
    {
        error = "candidate.postprocess must be 'clamp_0_1' or 'none'";
        return std::nullopt;
    }
    contract.postprocess = *postprocess;

    const YamlNode& proofModel = RequireMapping(root, "proof_model",
        {"operations", "rounding", "subnormal_inputs", "reassociation"}, error);
    if (!error.empty())
    {
        return std::nullopt;
    }
    std::optional<std::string> operations = ScalarOf(proofModel, "operations", error);
    std::optional<std::string> rounding = ScalarOf(proofModel, "rounding", error);
    std::optional<std::string> subnormals = ScalarOf(proofModel, "subnormal_inputs", error);
    std::optional<std::string> reassociation = ScalarOf(proofModel, "reassociation", error);
    if (!operations || !rounding || !subnormals || !reassociation)
    {
        return std::nullopt;
    }
    if (!ExpectValue(*operations, "binary32_fma", "proof_model.operations", error) ||
        !ExpectValue(*rounding, "nearest_even", "proof_model.rounding", error) ||
        !ExpectValue(*subnormals, "preserve", "proof_model.subnormal_inputs", error) ||
        !ExpectValue(*reassociation, "forbidden", "proof_model.reassociation", error))
    {
        return std::nullopt;
    }
    contract.proofOperations = *operations;
    contract.proofRounding = *rounding;
    contract.proofSubnormals = *subnormals;
    contract.proofReassociation = *reassociation;

    const YamlNode& requirements = RequireMapping(root, "requirements",
        {"max_absolute_error", "output_range", "endpoint_at_zero"}, error);
    if (!error.empty())
    {
        return std::nullopt;
    }
    std::optional<RationalValue> maxError = RationalOf(requirements, "max_absolute_error", error);
    if (!maxError)
    {
        return std::nullopt;
    }
    contract.maxAbsoluteError = *maxError;
    std::optional<std::vector<std::string>> outputRange = SequenceOf(requirements, "output_range", error);
    if (!outputRange)
    {
        return std::nullopt;
    }
    std::optional<RationalValue> rangeLo = RationalFromSequence(*outputRange, "output_range", 0, error);
    if (!rangeLo)
    {
        return std::nullopt;
    }
    std::optional<RationalValue> rangeHi = RationalFromSequence(*outputRange, "output_range", 1, error);
    if (!rangeHi)
    {
        return std::nullopt;
    }
    contract.outputRangeLo = *rangeLo;
    contract.outputRangeHi = *rangeHi;
    std::optional<RationalValue> endpoint = RationalOf(requirements, "endpoint_at_zero", error);
    if (!endpoint)
    {
        return std::nullopt;
    }
    contract.endpointAtZero = *endpoint;

    const YamlNode& outsideDomain = RequireMapping(root, "outside_domain", {"action"}, error);
    if (!error.empty())
    {
        return std::nullopt;
    }
    std::optional<std::string> action = ScalarOf(outsideDomain, "action", error);
    if (!action || !ExpectValue(*action, "reference_fallback", "outside_domain.action", error))
    {
        return std::nullopt;
    }
    contract.outsideDomainAction = *action;

    return contract;
}

std::optional<Contract> LoadContractFile(const std::string& path, std::string& error)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        error = "cannot open contract file: " + path;
        return std::nullopt;
    }
    std::ostringstream content;
    content << stream.rdbuf();
    return LoadContractText(content.str(), path, error);
}

std::string DescribeProofModel(const Contract& contract)
{
    std::string text = "binary32 Horner degree 4; 4 x FMA (single rounding, nearest-even); "
        "reassociation forbidden; subnormal inputs preserved; coefficients fixed bit patterns";
    if (contract.postprocess == "clamp_0_1")
    {
        text += "; postprocess clamp to [0, 1]";
    }
    return text;
}

} // namespace metallic::shadermath
