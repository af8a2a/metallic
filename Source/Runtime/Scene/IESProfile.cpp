#include "Runtime/Scene/IESProfile.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>

namespace metallic::scene {
namespace {

constexpr std::streamoff kMaxFileBytes = 64 * 1024 * 1024;
constexpr uint32_t kMaxCandelaSamples = 1024 * 1024;
constexpr std::string_view kWhitespace = " \t\r\n\v\f";
constexpr std::string_view kSeparators = " \t\r\n\v\f,";

std::string_view trim(std::string_view value)
{
    const auto first = value.find_first_not_of(kWhitespace);
    if (first == std::string_view::npos) { return {}; }
    return value.substr(first, value.find_last_not_of(kWhitespace) - first + 1);
}

template<typename T>
bool readNumber(std::string_view& input, T& number)
{
    const auto first = input.find_first_not_of(kSeparators);
    if (first == std::string_view::npos) { return false; }
    input.remove_prefix(first);
    const auto end = input.find_first_of(kSeparators);
    auto token = input.substr(0, end);
    input.remove_prefix(token.size());
    if (token.starts_with('+')) {
        token.remove_prefix(1);
        if (token.starts_with('-')) { return false; }
    }
    const auto* last = token.data() + token.size();
    const auto parsed = std::from_chars(token.data(), last, number);
    if (parsed.ec != std::errc{} || parsed.ptr != last) { return false; }
    if constexpr (std::is_floating_point_v<T>) { return std::isfinite(number); }
    return true;
}

bool readAngles(std::string_view& input, std::vector<float>& angles)
{
    for (size_t i = 0; i < angles.size(); ++i) {
        if (!readNumber(input, angles[i]) || (i != 0 && angles[i - 1] >= angles[i])) { return false; }
    }
    return true;
}

} // namespace

IESProfile::IESProfile(std::filesystem::path path, std::vector<float> vertical, std::vector<float> horizontal,
    std::vector<float> samples, float peak)
    : sourcePath(std::move(path)), verticalAngles(std::move(vertical)), horizontalAngles(std::move(horizontal)),
      candela(std::move(samples)), peakCandela(peak)
{}

std::shared_ptr<const IESProfile> loadIESProfile(const std::filesystem::path& path, std::string& reason)
{
    reason.clear();
    const auto reject = [&](const char* message) -> std::shared_ptr<const IESProfile> {
        reason = message;
        return {};
    };
    std::error_code error;
    auto sourcePath = std::filesystem::absolute(path, error).lexically_normal();
    if (path.empty() || error) { return reject("Invalid IES profile path."); }
    std::ifstream file(sourcePath, std::ios::binary | std::ios::ate);
    if (!file) { return reject("Could not open the IES profile."); }
    const auto fileSize = static_cast<std::streamoff>(file.tellg());
    if (fileSize <= 0 || fileSize > kMaxFileBytes) { return reject("IES file size is outside the supported range."); }
    std::string contents(static_cast<size_t>(fileSize), '\0');
    if (!file.seekg(0).read(contents.data(), static_cast<std::streamsize>(fileSize))) {
        return reject("Could not read the complete IES profile.");
    }

    std::string_view input = contents;
    if (input.starts_with("\xef\xbb\xbf")) { input.remove_prefix(3); }
    const auto version = trim(input.substr(0, input.find_first_of("\r\n")));
    // The second field after the ballast factor is the lamp photometric factor
    // in LM-63-1986/1991 and was reserved or repurposed by later versions.
    const bool hasBallastLampFactor =
        (!version.starts_with("IESNA:LM-63-") && !version.starts_with("IES:LM-63-")) || version.ends_with("-1991");
    bool tiltFound = false;
    while (!input.empty()) {
        const auto end = input.find_first_of("\r\n");
        const auto line = trim(input.substr(0, end));
        input.remove_prefix(end == std::string_view::npos ? input.size() : end + 1);
        const auto equal = line.find('=');
        if (equal == std::string_view::npos || trim(line.substr(0, equal)) != "TILT") { continue; }
        if (trim(line.substr(equal + 1)) != "NONE") { return reject("Only TILT=NONE IES profiles are supported."); }
        tiltFound = true;
        break;
    }
    if (!tiltFound) { return reject("IES profile has no TILT declaration."); }

    uint32_t lampCount = 0, verticalCount = 0, horizontalCount = 0, photometricType = 0, units = 0;
    double lumensPerLamp = 0.0, multiplier = 0.0, ballastFactor = 0.0, ballastLampFactor = 0.0;
    double width = 0.0, length = 0.0, height = 0.0, inputWatts = 0.0;
    if (!readNumber(input, lampCount) || !readNumber(input, lumensPerLamp) || !readNumber(input, multiplier) ||
        !readNumber(input, verticalCount) || !readNumber(input, horizontalCount) ||
        !readNumber(input, photometricType) || !readNumber(input, units) ||
        !readNumber(input, width) || !readNumber(input, length) || !readNumber(input, height) ||
        !readNumber(input, ballastFactor) || !readNumber(input, ballastLampFactor) || !readNumber(input, inputWatts)) {
        return reject("Invalid or truncated IES photometric header.");
    }
    if (photometricType != 1) { return reject("Only Type C IES photometry is supported."); }
    if ((units != 1 && units != 2) || (lumensPerLamp < 0.0 && lumensPerLamp != -1.0) ||
        multiplier < 0.0 || ballastFactor < 0.0 || (hasBallastLampFactor && ballastLampFactor < 0.0)) {
        return reject("IES photometric units or intensity multipliers are invalid.");
    }
    const uint64_t count = uint64_t(verticalCount) * horizontalCount;
    if (verticalCount < 2 || horizontalCount == 0 || count > kMaxCandelaSamples) {
        return reject("IES angle counts are outside the supported range.");
    }
    std::vector<float> verticalAngles(verticalCount);
    std::vector<float> horizontalAngles(horizontalCount);
    if (!readAngles(input, verticalAngles) ||
        (verticalAngles.front() != 0.0f && verticalAngles.front() != 90.0f) ||
        (verticalAngles.back() != 90.0f && verticalAngles.back() != 180.0f)) {
        return reject("Invalid Type C vertical angles. Expected increasing 0-90, 0-180, or 90-180 degree samples.");
    }
    if (!readAngles(input, horizontalAngles)) { return reject("Invalid or non-increasing IES horizontal angles."); }
    const float first = horizontalAngles.front();
    const float last = horizontalAngles.back();
    if (!((first == 0.0f && (last == 0.0f || last == 90.0f || last == 180.0f || last == 360.0f)) ||
        (first == 90.0f && last == 270.0f))) {
        return reject("Unsupported Type C horizontal angle range.");
    }

    std::vector<float> candela(static_cast<size_t>(count));
    float unscaledPeak = 0.0f;
    for (float& sample : candela) {
        if (!readNumber(input, sample) || sample < 0.0f) { return reject("Missing or invalid IES candela sample."); }
        unscaledPeak = std::max(unscaledPeak, sample);
    }
    if (input.find_first_not_of(kSeparators) != std::string_view::npos) {
        return reject("IES file has data after its last candela sample.");
    }
    const double scaledPeak = double(unscaledPeak) * multiplier * ballastFactor *
        (hasBallastLampFactor ? ballastLampFactor : 1.0);
    if (!std::isfinite(scaledPeak) || scaledPeak > std::numeric_limits<float>::max() ||
        (scaledPeak > 0.0 && scaledPeak < std::numeric_limits<float>::denorm_min())) {
        return reject("IES peak candela exceeds the supported precision.");
    }
    if (scaledPeak > 0.0) {
        for (float& sample : candela) { sample /= unscaledPeak; }
    } else {
        std::ranges::fill(candela, 0.0f);
    }
    return std::shared_ptr<const IESProfile>(new IESProfile(std::move(sourcePath), std::move(verticalAngles),
        std::move(horizontalAngles), std::move(candela), static_cast<float>(scaledPeak)));
}

} // namespace metallic::scene
