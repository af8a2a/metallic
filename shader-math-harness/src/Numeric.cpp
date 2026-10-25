#include "Numeric.hpp"

#include <cassert>
#include <cstring>

namespace metallic::shadermath
{

namespace
{

constexpr int kIntervalUlpInflation = 2;
constexpr int kF32GuardUlp = 1;

double WidenDown(double v)
{
    if (!std::isfinite(v))
    {
        return v;
    }
    for (int i = 0; i < kIntervalUlpInflation; ++i)
    {
        v = std::nextafter(v, -std::numeric_limits<double>::infinity());
    }
    return v;
}

double WidenUp(double v)
{
    if (!std::isfinite(v))
    {
        return v;
    }
    for (int i = 0; i < kIntervalUlpInflation; ++i)
    {
        v = std::nextafter(v, std::numeric_limits<double>::infinity());
    }
    return v;
}

float WidenDownF(float v)
{
    if (!std::isfinite(v))
    {
        return v;
    }
    for (int i = 0; i < kF32GuardUlp; ++i)
    {
        v = std::nextafterf(v, -std::numeric_limits<float>::infinity());
    }
    return v;
}

float WidenUpF(float v)
{
    if (!std::isfinite(v))
    {
        return v;
    }
    for (int i = 0; i < kF32GuardUlp; ++i)
    {
        v = std::nextafterf(v, std::numeric_limits<float>::infinity());
    }
    return v;
}

bool IsDigit(char c)
{
    return c >= '0' && c <= '9';
}

} // namespace

Interval IntervalFromDouble(double v)
{
    return Interval{WidenDown(v), WidenUp(v)};
}

Interval IntervalFromRational(std::int64_t num, std::int64_t den)
{
    assert(den != 0);
    assert(num > -(1LL << 53) && num < (1LL << 53));
    assert(den > -(1LL << 53) && den < (1LL << 53));
    if (den < 0)
    {
        num = -num;
        den = -den;
    }
    const double quotient = static_cast<double>(num) / static_cast<double>(den);
    return Interval{WidenDown(quotient), WidenUp(quotient)};
}

Interval Add(Interval a, Interval b)
{
    return Interval{WidenDown(a.lo + b.lo), WidenUp(a.hi + b.hi)};
}

Interval Sub(Interval a, Interval b)
{
    return Interval{WidenDown(a.lo - b.hi), WidenUp(a.hi - b.lo)};
}

Interval Mul(Interval a, Interval b)
{
    const double p1 = a.lo * b.lo;
    const double p2 = a.lo * b.hi;
    const double p3 = a.hi * b.lo;
    const double p4 = a.hi * b.hi;
    double lo = p1;
    if (p2 < lo) lo = p2;
    if (p3 < lo) lo = p3;
    if (p4 < lo) lo = p4;
    double hi = p1;
    if (p2 > hi) hi = p2;
    if (p3 > hi) hi = p3;
    if (p4 > hi) hi = p4;
    return Interval{WidenDown(lo), WidenUp(hi)};
}

Interval Div(Interval a, Interval b)
{
    assert(b.lo > 0.0 || b.hi < 0.0);
    const double q1 = a.lo / b.lo;
    const double q2 = a.lo / b.hi;
    const double q3 = a.hi / b.lo;
    const double q4 = a.hi / b.hi;
    double lo = q1;
    if (q2 < lo) lo = q2;
    if (q3 < lo) lo = q3;
    if (q4 < lo) lo = q4;
    double hi = q1;
    if (q2 > hi) hi = q2;
    if (q3 > hi) hi = q3;
    if (q4 > hi) hi = q4;
    return Interval{WidenDown(lo), WidenUp(hi)};
}

Interval Scale(Interval a, double k)
{
    return Mul(a, IntervalFromDouble(k));
}

Interval Abs(Interval a)
{
    if (a.lo >= 0.0)
    {
        return a;
    }
    if (a.hi <= 0.0)
    {
        return Interval{-a.hi, -a.lo};
    }
    const double hi = (-a.lo > a.hi) ? -a.lo : a.hi;
    return Interval{0.0, hi};
}

Interval Hull(Interval a, Interval b)
{
    return Interval{(a.lo < b.lo) ? a.lo : b.lo, (a.hi > b.hi) ? a.hi : b.hi};
}

bool Contains(Interval a, double v)
{
    return a.lo <= v && v <= a.hi;
}

double Sup(Interval a)
{
    const double loAbs = -a.lo;
    return (loAbs > a.hi) ? loAbs : a.hi;
}

std::optional<RationalValue> ParseRational(std::string_view text)
{
    size_t begin = 0;
    size_t end = text.size();
    while (begin < end && (text[begin] == ' ' || text[begin] == '\t'))
    {
        ++begin;
    }
    while (end > begin && (text[end - 1] == ' ' || text[end - 1] == '\t'))
    {
        --end;
    }
    if (begin >= end)
    {
        return std::nullopt;
    }

    bool negative = false;
    if (text[begin] == '-' || text[begin] == '+')
    {
        negative = (text[begin] == '-');
        ++begin;
    }

    constexpr std::int64_t kDigitLimit = (1LL << 53) / 10 - 9;
    auto accumulate = [&](std::int64_t& value, char digit, int& digitCount) -> bool
    {
        if (value > kDigitLimit)
        {
            return false;
        }
        value = value * 10 + (digit - '0');
        ++digitCount;
        return value < (1LL << 53);
    };

    // "a/b" fraction form.
    const size_t slash = text.find('/', begin);
    if (slash != std::string_view::npos && slash < end)
    {
        if (slash == begin || slash + 1 >= end)
        {
            return std::nullopt;
        }
        std::int64_t num = 0;
        std::int64_t den = 0;
        int numDigits = 0;
        int denDigits = 0;
        for (size_t i = begin; i < slash; ++i)
        {
            if (!IsDigit(text[i]) || !accumulate(num, text[i], numDigits))
            {
                return std::nullopt;
            }
        }
        for (size_t i = slash + 1; i < end; ++i)
        {
            if (!IsDigit(text[i]) || !accumulate(den, text[i], denDigits))
            {
                return std::nullopt;
            }
        }
        if (num == 0 && den == 0)
        {
            return std::nullopt;
        }
        if (den == 0)
        {
            return std::nullopt;
        }
        if (negative)
        {
            num = -num;
        }
        return RationalValue{num, den};
    }

    // Decimal form with optional exponent.
    std::int64_t mantissa = 0;
    int mantissaDigits = 0;
    int fractionDigits = 0;
    std::int64_t exponent = 0;
    bool seenDigit = false;
    bool seenDot = false;
    size_t i = begin;
    for (; i < end; ++i)
    {
        const char c = text[i];
        if (IsDigit(c))
        {
            if (!accumulate(mantissa, c, mantissaDigits))
            {
                return std::nullopt;
            }
            if (seenDot)
            {
                ++fractionDigits;
            }
            seenDigit = true;
        }
        else if (c == '.' && !seenDot)
        {
            seenDot = true;
        }
        else
        {
            break;
        }
    }
    if (!seenDigit)
    {
        return std::nullopt;
    }
    if (i < end && (text[i] == 'e' || text[i] == 'E'))
    {
        ++i;
        bool exponentNegative = false;
        if (i < end && (text[i] == '-' || text[i] == '+'))
        {
            exponentNegative = (text[i] == '-');
            ++i;
        }
        if (i >= end)
        {
            return std::nullopt;
        }
        std::int64_t exponentMagnitude = 0;
        int exponentDigits = 0;
        for (; i < end; ++i)
        {
            if (!IsDigit(text[i]) || !accumulate(exponentMagnitude, text[i], exponentDigits))
            {
                return std::nullopt;
            }
        }
        exponent = exponentNegative ? -exponentMagnitude : exponentMagnitude;
    }
    else if (i != end)
    {
        return std::nullopt;
    }

    // value = mantissa * 10^(exponent - fractionDigits)
    std::int64_t num = mantissa;
    std::int64_t den = 1;
    std::int64_t power = exponent - fractionDigits;
    if (power > 0)
    {
        for (std::int64_t k = 0; k < power; ++k)
        {
            if (num > kDigitLimit || den > kDigitLimit)
            {
                return std::nullopt;
            }
            num *= 10;
        }
    }
    else
    {
        for (std::int64_t k = 0; k < -power; ++k)
        {
            if (num > kDigitLimit || den > kDigitLimit)
            {
                return std::nullopt;
            }
            den *= 10;
        }
    }
    if (negative)
    {
        num = -num;
    }
    return RationalValue{num, den};
}

namespace
{

// Rigorous e^s for a single point s in [-0.25, 0.26] using a directed Taylor
// series. The remainder satisfies 0 <= R <= (s^K / K!) * e^s * tail where the
// geometric tail ratio is s/(K+1) <= 0.26/K; the constant 4/3 bounds e^s on
// this range because e^0.26 < 1.3 < 4/3.
Interval ExpPoint(double s)
{
    assert(s >= -0.25 && s <= 0.26);
    if (s < 0.0)
    {
        // e^s = 1 / e^{-s} with -s in (0, 0.25].
        return Div(IntervalExact(1.0), ExpPoint(-s));
    }

    constexpr int kTerms = 40;
    Interval argument = IntervalExact(s);
    Interval term = IntervalExact(1.0);
    Interval sum = IntervalExact(1.0);
    for (int k = 1; k <= kTerms; ++k)
    {
        term = Mul(term, argument);
        term = Div(term, IntervalExact(static_cast<double>(k)));
        sum = Add(sum, term);
    }
    // Tail bound: |term_{K+1} + term_{K+2} + ...| <= term_K * (s/(K+1)) / (1 - s/(K+1)).
    const double ratio = s / static_cast<double>(kTerms + 1);
    const double tailFactor = ratio / (1.0 - ratio);
    Interval tail = Mul(term, IntervalFromDouble(tailFactor));
    Interval remainder = Mul(tail, IntervalFromRational(4, 3));
    sum = Add(sum, remainder);
    return sum;
}

} // namespace

Interval ExpSmall(Interval t)
{
    assert(t.lo >= -0.02 && t.hi <= 0.26);
    const Interval low = ExpPoint(t.lo);
    const Interval high = ExpPoint(t.hi);
    return Interval{low.lo, high.hi};
}

Interval ExpNegativeSmall(Interval x)
{
    assert(x.lo >= -0.02 && x.hi <= 0.26);
    // e^{-x} is decreasing in x: value over [a, b] is [e^{-b}, e^{-a}].
    const Interval upper = ExpPoint(-x.hi);
    const Interval lower = ExpPoint(-x.lo);
    return Interval{upper.lo, lower.hi};
}

float FloatFromBits(std::uint32_t bits)
{
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

std::uint32_t BitsFromFloat(float value)
{
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float NextBelowF(float value)
{
    return std::nextafterf(value, -std::numeric_limits<float>::infinity());
}

float NextAboveF(float value)
{
    return std::nextafterf(value, std::numeric_limits<float>::infinity());
}

F32Interval RoundOutwardF32(Interval a)
{
    float lo = static_cast<float>(a.lo);
    if (static_cast<double>(lo) > a.lo)
    {
        lo = NextBelowF(lo);
    }
    lo = WidenDownF(lo);
    float hi = static_cast<float>(a.hi);
    if (static_cast<double>(hi) < a.hi)
    {
        hi = NextAboveF(hi);
    }
    hi = WidenUpF(hi);
    return F32Interval{lo, hi};
}

F32Interval FmaIntervalF32(F32Interval a, Interval x, float c)
{
    const Interval accumulated{static_cast<double>(a.lo), static_cast<double>(a.hi)};
    const Interval product = Mul(accumulated, x);
    const Interval sum = Add(product, IntervalExact(static_cast<double>(c)));
    return RoundOutwardF32(sum);
}

F32Interval Clamp01F32(F32Interval a)
{
    float lo = (a.lo < 0.0f) ? 0.0f : a.lo;
    float hi = (a.hi > 1.0f) ? 1.0f : a.hi;
    return F32Interval{lo, hi};
}

std::uint64_t Fnv1a64(std::string_view text)
{
    std::uint64_t hash = 14695981039346656037ULL;
    for (const char c : text)
    {
        hash ^= static_cast<std::uint8_t>(c);
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::string HashToHex(std::uint64_t hash)
{
    static const char* kDigits = "0123456789abcdef";
    std::string text(16, '0');
    for (int i = 15; i >= 0; --i)
    {
        text[static_cast<size_t>(i)] = kDigits[hash & 0xF];
        hash >>= 4;
    }
    return text;
}

} // namespace metallic::shadermath
