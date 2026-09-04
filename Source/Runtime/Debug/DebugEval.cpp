#include "Runtime/Debug/DebugTypes.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <memory>
#include <set>

namespace metallic::debug {
namespace {

[[noreturn]] void fail(std::string code, std::string message)
{
    throw DebugError{std::move(code), std::move(message)};
}

struct Expression {
    enum class Kind { Literal, Name, Field, Index, Unary, Binary, Call, Lambda };
    Kind kind = Kind::Literal;
    std::string text;
    DebugValue literal;
    std::vector<std::unique_ptr<Expression>> children;
};

class Parser {
public:
    explicit Parser(std::string_view source)
    {
        if (source.size() > 16384) { fail("BudgetExceeded", "Expression exceeds 16 KiB"); }
        for (size_t i = 0; i < source.size();) {
            const unsigned char c = source[i];
            if (std::isspace(c)) { ++i; continue; }
            const size_t begin = i++;
            if (std::isalpha(c) || c == '_') {
                while (i < source.size() && (std::isalnum(static_cast<unsigned char>(source[i])) || source[i] == '_')) { ++i; }
            } else if (std::isdigit(c)) {
                while (i < source.size() && (std::isdigit(static_cast<unsigned char>(source[i])) || source[i] == '.')) { ++i; }
                if (i < source.size() && (source[i] == 'e' || source[i] == 'E')) {
                    ++i;
                    if (i < source.size() && (source[i] == '+' || source[i] == '-')) { ++i; }
                    while (i < source.size() && std::isdigit(static_cast<unsigned char>(source[i]))) { ++i; }
                }
            } else if (c == '"') {
                bool closed = false;
                while (i < source.size()) {
                    if (source[i] == '\\') { i += 2; continue; }
                    if (source[i++] == '"') { closed = true; break; }
                }
                if (!closed || i > source.size()) { fail("ParseError", "Unterminated string"); }
            } else if (i < source.size()) {
                const auto two = source.substr(begin, 2);
                if (two == "&&" || two == "||" || two == "==" || two == "!=" || two == "<=" || two == ">=" || two == "=>") { ++i; }
            }
            tokens_.emplace_back(source.substr(begin, i - begin));
            if (tokens_.size() > 2048) { fail("BudgetExceeded", "Too many tokens"); }
        }
        tokens_.push_back("");
    }

    std::unique_ptr<Expression> parse()
    {
        auto result = expression(0, 0);
        if (!peek().empty()) { fail("ParseError", "Unexpected token: " + peek()); }
        return result;
    }

private:
    std::vector<std::string> tokens_;
    size_t position_ = 0;
    const std::string& peek() const { return tokens_.at(position_); }
    bool take(std::string_view token)
    {
        if (peek() != token) { return false; }
        ++position_;
        return true;
    }
    void require(std::string_view token)
    {
        if (!take(token)) { fail("ParseError", "Expected " + std::string(token)); }
    }
    static int precedence(std::string_view op)
    {
        if (op == "||") { return 1; }
        if (op == "&&") { return 2; }
        if (op == "==" || op == "!=") { return 3; }
        if (op == "<" || op == ">" || op == "<=" || op == ">=") { return 4; }
        if (op == "+" || op == "-") { return 5; }
        if (op == "*" || op == "/" || op == "%") { return 6; }
        return -1;
    }
    static bool identifier(std::string_view token)
    {
        return !token.empty() && (std::isalpha(static_cast<unsigned char>(token[0])) || token[0] == '_');
    }
    std::unique_ptr<Expression> expression(int minimum, uint32_t depth)
    {
        if (depth > 64) { fail("BudgetExceeded", "Expression nesting exceeds 64"); }
        auto left = std::make_unique<Expression>();
        const std::string token = peek();
        if (token.empty()) { fail("ParseError", "Expected expression"); }
        ++position_;
        if (token == "(" ) {
            left = expression(0, depth + 1);
            require(")");
        } else if (token == "!" || token == "-" || token == "+") {
            left->kind = Expression::Kind::Unary;
            left->text = token;
            left->children.push_back(expression(7, depth + 1));
        } else if (token == "true" || token == "false" || token == "null" || token[0] == '"' || std::isdigit(static_cast<unsigned char>(token[0]))) {
            left->literal = DebugValue::parse(token);
            if (left->literal.is_number_float() && token.find_first_of(".eE") == std::string::npos) {
                fail("Overflow", "Integer literal is outside 64-bit range");
            }
        } else if (identifier(token)) {
            left->kind = Expression::Kind::Name;
            left->text = token;
            if (take("=>")) {
                left->kind = Expression::Kind::Lambda;
                left->children.push_back(expression(0, depth + 1));
            } else if (take("(")) {
                static const std::set<std::string> kFunctions{"count", "min", "max", "mean", "findFirst", "isnan", "isinf", "length"};
                if (!kFunctions.contains(token)) { fail("Unsupported", "Function is not allowed: " + token); }
                left->kind = Expression::Kind::Call;
                if (!take(")")) {
                    do { left->children.push_back(expression(0, depth + 1)); } while (take(","));
                    require(")");
                }
                const size_t size = left->children.size();
                if (!size || size > 2 || (size == 2 && left->children[1]->kind != Expression::Kind::Lambda) ||
                    ((token == "isnan" || token == "isinf" || token == "length") && size != 1)) {
                    fail("TypeError", "Invalid arguments to " + token);
                }
            }
        } else { fail("ParseError", "Unexpected token: " + token); }

        while (true) {
            if (take(".")) {
                auto field = std::make_unique<Expression>();
                field->kind = Expression::Kind::Field;
                field->text = peek();
                if (!identifier(peek())) { fail("ParseError", "Expected field name"); }
                ++position_;
                field->children.push_back(std::move(left));
                left = std::move(field);
            } else if (take("[")) {
                auto index = std::make_unique<Expression>();
                index->kind = Expression::Kind::Index;
                index->children.push_back(std::move(left));
                index->children.push_back(expression(0, depth + 1));
                require("]");
                left = std::move(index);
            } else { break; }
        }
        while (precedence(peek()) >= minimum) {
            auto binary = std::make_unique<Expression>();
            binary->kind = Expression::Kind::Binary;
            binary->text = peek();
            ++position_;
            binary->children.push_back(std::move(left));
            binary->children.push_back(expression(precedence(binary->text) + 1, depth + 1));
            left = std::move(binary);
        }
        return left;
    }
};

struct Integer {
    bool negative = false;
    uint64_t magnitude = 0;
};

Integer integer(const DebugValue& v)
{
    if (!v.is_number_integer()) { fail("TypeError", "Expected integer"); }
    if (v.is_number_unsigned()) { return {false, v.get<uint64_t>()}; }
    const int64_t n = v.get<int64_t>();
    return {n < 0, n < 0 ? uint64_t(-(n + 1)) + 1 : uint64_t(n)};
}

DebugValue value(Integer n)
{
    if (!n.negative || !n.magnitude) { return n.magnitude; }
    if (n.magnitude > (uint64_t(1) << 63)) { fail("Overflow", "Signed integer underflow"); }
    return -int64_t(n.magnitude - 1) - 1;
}

double real(const DebugValue& v)
{
    if (!v.is_number()) { fail("TypeError", "Expected number"); }
    return v.get<double>();
}

bool boolean(const DebugValue& v)
{
    if (!v.is_boolean()) { fail("TypeError", "Expected bool"); }
    return v.get<bool>();
}

int compare(const DebugValue& a, const DebugValue& b)
{
    if (a.is_number_integer() && b.is_number_integer()) {
        const auto x = integer(a), y = integer(b);
        if (x.negative != y.negative) { return x.negative ? -1 : 1; }
        const int order = x.magnitude < y.magnitude ? -1 : x.magnitude > y.magnitude ? 1 : 0;
        return x.negative ? -order : order;
    }
    if (a.is_number() && b.is_number()) {
        const double x = real(a), y = real(b);
        // Reject implicit integer rounding in mixed comparisons.
        const auto check = [](const DebugValue& v) {
            if (v.is_number_integer() && integer(v).magnitude > (uint64_t(1) << 53)) {
                fail("PrecisionLoss", "Compare large integers with integer operands");
            }
        };
        check(a); check(b);
        if (std::isnan(x) || std::isnan(y)) { return 2; }
        return x < y ? -1 : x > y ? 1 : 0;
    }
    if (a.type() != b.type() || a.is_structured()) { fail("TypeError", "Incompatible comparison"); }
    return a == b ? 0 : a < b ? -1 : 1;
}

DebugValue arithmetic(std::string_view op, const DebugValue& a, const DebugValue& b)
{
    if (!a.is_number() || !b.is_number()) { fail("TypeError", "Arithmetic requires numbers"); }
    if (a.is_number_float() || b.is_number_float() || op == "/") {
        if (op == "%") { fail("TypeError", "Remainder requires integers"); }
        for (const auto* operand : {&a, &b}) {
            if (operand->is_number_integer() && integer(*operand).magnitude > (uint64_t(1) << 53)) {
                fail("PrecisionLoss", "Floating arithmetic cannot implicitly round a large integer");
            }
        }
        const double x = real(a), y = real(b);
        if (op == "/" && y == 0) { fail("DivisionByZero", "Division by zero"); }
        if (op == "+") { return x + y; }
        if (op == "-") { return x - y; }
        if (op == "*") { return x * y; }
        return x / y;
    }
    auto x = integer(a), y = integer(b);
    if (op == "%") {
        if (!y.magnitude) { fail("DivisionByZero", "Remainder by zero"); }
        return value({x.negative, x.magnitude % y.magnitude});
    }
    if (op == "*") {
        if (y.magnitude && x.magnitude > UINT64_MAX / y.magnitude) { fail("Overflow", "Integer multiplication overflow"); }
        return value({x.negative != y.negative, x.magnitude * y.magnitude});
    }
    if (op == "-") { y.negative = !y.negative; }
    if (x.negative == y.negative) {
        if (x.magnitude > UINT64_MAX - y.magnitude) { fail("Overflow", "Integer addition overflow"); }
        return value({x.negative, x.magnitude + y.magnitude});
    }
    if (x.magnitude >= y.magnitude) { return value({x.negative, x.magnitude - y.magnitude}); }
    return value({y.negative, y.magnitude - x.magnitude});
}

class Evaluator {
public:
    Evaluator(const DebugValue& root, uint64_t budget) : root_(root), budget_(budget) {}
    DebugValue run(const Expression& e, std::string_view variable = {}, const DebugValue* item = nullptr, uint32_t depth = 0)
    {
        if (!budget_-- || depth > 128) { fail("BudgetExceeded", "Evaluation limit reached"); }
        const auto child = [&](size_t i) { return run(*e.children.at(i), variable, item, depth + 1); };
        using Kind = Expression::Kind;
        if (e.kind == Kind::Literal) { return e.literal; }
        if (e.kind == Kind::Name) {
            if (item && e.text == variable) { chargeCopy(*item); return *item; }
            if (!root_.contains(e.text)) { fail("NotFound", "Unknown object: " + e.text); }
            chargeCopy(root_.at(e.text));
            return root_.at(e.text);
        }
        if (e.kind == Kind::Lambda) { fail("TypeError", "Lambda is only allowed as an aggregate argument"); }
        if (e.kind == Kind::Field) {
            auto object = child(0);
            if (object.is_array() && e.text == "count") { return uint64_t(object.size()); }
            if (!object.is_object() || !object.contains(e.text)) {
                fail(object.is_object() && object.value("captured", true) == false ? "NotCaptured" : "NotFound", "Unavailable field: " + e.text);
            }
            return object.at(e.text);
        }
        if (e.kind == Kind::Index) {
            auto array = child(0), index = child(1);
            if (array.is_object() && index.is_string()) {
                if (!array.contains(index.get<std::string>())) { fail("NotFound", "Unknown object key"); }
                return array.at(index.get<std::string>());
            }
            if (!array.is_array()) { fail("NotCaptured", "Indexing requires a captured array"); }
            const auto n = integer(index);
            if (n.negative || n.magnitude >= array.size()) { fail("OutOfRange", "Index exceeds captured range"); }
            return array.at(static_cast<size_t>(n.magnitude));
        }
        if (e.kind == Kind::Unary) {
            auto operand = child(0);
            if (e.text == "!") { return !boolean(operand); }
            if (!operand.is_number()) { fail("TypeError", "Unary arithmetic requires a number"); }
            if (e.text == "+") { return operand; }
            if (operand.is_number_float()) { return -real(operand); }
            auto n = integer(operand); n.negative = !n.negative;
            return value(n);
        }
        if (e.kind == Kind::Binary) {
            const auto a = child(0);
            if (e.text == "&&") { return boolean(a) && boolean(child(1)); }
            if (e.text == "||") { return boolean(a) || boolean(child(1)); }
            const auto b = child(1);
            if (e.text == "+" || e.text == "-" || e.text == "*" || e.text == "/" || e.text == "%") { return arithmetic(e.text, a, b); }
            const int c = compare(a, b);
            if (e.text == "==") { return c == 0; }
            if (e.text == "!=") { return c != 0; }
            if (c == 2) { return false; }
            if (e.text == "<") { return c < 0; }
            if (e.text == ">") { return c > 0; }
            if (e.text == "<=") { return c <= 0; }
            return c >= 0;
        }
        auto array = child(0);
        if (e.text == "isnan") { return std::isnan(real(array)); }
        if (e.text == "isinf") { return std::isinf(real(array)); }
        if (e.text == "length" && array.is_string()) { return uint64_t(array.get_ref<const std::string&>().size()); }
        if (!array.is_array()) { fail("NotCaptured", "Aggregate requires a captured array"); }
        uint64_t count = 0;
        double sum = 0;
        bool hasNan = false;
        DebugValue best;
        for (size_t i = 0; i < array.size(); ++i) {
            if (!budget_--) { fail("BudgetExceeded", "Evaluation limit reached"); }
            DebugValue projected = array[i];
            if (e.children.size() == 2) {
                const auto& lambda = *e.children[1];
                projected = run(*lambda.children[0], lambda.text, &array[i], depth + 1);
            }
            if (e.text == "count" || e.text == "findFirst") {
                const bool matches = e.children.size() == 1 ? true : boolean(projected);
                if (matches) {
                    ++count;
                    if (e.text == "findFirst") { return uint64_t(i); }
                }
            } else if (e.text == "length") {
                const double n = real(projected); sum += n * n;
            } else {
                if (!projected.is_number()) { fail("TypeError", "Numeric aggregate requires numbers"); }
                hasNan = hasNan || std::isnan(real(projected));
                if (best.is_null() || (e.text == "min" ? compare(projected, best) < 0 : compare(projected, best) == 1)) { best = projected; }
                sum += real(projected);
            }
        }
        if (e.text == "count") { return count; }
        if (e.text == "findFirst") { return nullptr; }
        if (e.text == "length") { return std::sqrt(sum); }
        if (e.text == "mean") { return array.empty() ? DebugValue(nullptr) : DebugValue(sum / array.size()); }
        if (hasNan) { return std::numeric_limits<double>::quiet_NaN(); }
        return best;
    }
private:
    void chargeCopy(const DebugValue& value)
    {
        if (!budget_--) { fail("BudgetExceeded", "Evaluation value-copy limit reached"); }
        if (value.is_structured()) { for (const auto& child : value) { chargeCopy(child); } }
    }
    const DebugValue& root_;
    uint64_t budget_;
};

} // namespace

DebugResult<DebugValue> evaluate(std::string_view expression, const DebugValue& root, uint64_t operationBudget)
{
    try {
        auto ir = Parser(expression).parse();
        return Evaluator(root, operationBudget).run(*ir);
    } catch (const DebugError& error) {
        return std::unexpected(error);
    } catch (const std::exception& error) {
        return std::unexpected(DebugError{"ParseError", error.what()});
    }
}

} // namespace metallic::debug
