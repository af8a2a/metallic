#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>

namespace metallic::shadermath
{

// ---------------------------------------------------------------------------
// Rigorous numeric core for the shader math harness.
//
// Interval arithmetic over double with explicit outward ulp inflation. Every
// elementary operation is evaluated in default round-to-nearest double and each
// endpoint is then widened by kIntervalUlpInflation ulps toward -/+ infinity.
// This stays sound without switching hardware rounding modes as long as the
// binary is built without fast-math reassociation (repository default) and each
// compiled operation is a single IEEE-754 rounded step, which compiler
// contraction of a*b+c into one FMA does not invalidate: a contraction removes
// one rounding the inflation already accounts for. See README.md for the full
// soundness argument and its limits.
// ---------------------------------------------------------------------------

struct Interval
{
    double lo;
    double hi;
};

// Point interval for an exactly representable value (any double literal, or a
// float promoted to double). No widening is applied.
constexpr Interval IntervalExact(double v)
{
    return Interval{v, v};
}

// Point interval with outward inflation (for values produced by rounding).
Interval IntervalFromDouble(double v);

// Interval containing the exact rational num/den (|num|, |den| < 2^53).
Interval IntervalFromRational(std::int64_t num, std::int64_t den);

Interval Add(Interval a, Interval b);
Interval Sub(Interval a, Interval b);
Interval Mul(Interval a, Interval b);

// Division; the denominator interval must not contain zero.
Interval Div(Interval a, Interval b);

Interval Scale(Interval a, double k);
Interval Abs(Interval a);
Interval Hull(Interval a, Interval b);
bool Contains(Interval a, double v);

// max(|lo|, |hi|): certified upper bound for |g| when the interval encloses g.
double Sup(Interval a);

struct RationalValue
{
    std::int64_t num;
    std::int64_t den;
};

// Parses the contract number grammar: integers, "a/b" fractions, decimals with
// optional exponent ("8.5e-6"). Values must stay below 2^53 in numerator and
// denominator; anything larger is rejected.
std::optional<RationalValue> ParseRational(std::string_view text);

// Rigorous e^t for t boxes contained in [-0.02, 0.26]. Uses monotonicity plus
// a directed Taylor series with a certified remainder; negative arguments are
// evaluated through the reciprocal of the positive-argument series.
Interval ExpSmall(Interval t);

// Rigorous e^{-x} for x boxes contained in [-0.02, 0.26].
Interval ExpNegativeSmall(Interval x);

// ---------------------------------------------------------------------------
// Binary32 helpers.
// ---------------------------------------------------------------------------

float FloatFromBits(std::uint32_t bits);
std::uint32_t BitsFromFloat(float value);
float NextBelowF(float value);
float NextAboveF(float value);

struct F32Interval
{
    float lo;
    float hi;
};

// Smallest float interval (plus one guard ulp) containing the double interval.
F32Interval RoundOutwardF32(Interval a);

// Interval model of fl32(a * x + c): a is a binary32 enclosure of the
// accumulated value, x is a real (double) input box, c is an exact binary32
// coefficient. The double interval product and sum contain every exact real
// value, and the outward rounding to binary32 covers the single FMA rounding
// of the proof model.
F32Interval FmaIntervalF32(F32Interval a, Interval x, float c);

F32Interval Clamp01F32(F32Interval a);

// ---------------------------------------------------------------------------
// Content hashing (FNV-1a, 64 bit) used for contract/artifact identity.
// ---------------------------------------------------------------------------

std::uint64_t Fnv1a64(std::string_view text);
std::string HashToHex(std::uint64_t hash);

} // namespace metallic::shadermath
