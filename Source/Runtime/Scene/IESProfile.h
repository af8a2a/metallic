#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace metallic::scene {

// Immutable Type C photometry. Angles are in degrees and each horizontal plane
// stores a row of samples in vertical-angle order. Samples are normalized to
// peakCandela, which includes the file's photometric multipliers.
struct IESProfile final {
    const std::filesystem::path sourcePath;
    const std::vector<float> verticalAngles;
    const std::vector<float> horizontalAngles;
    const std::vector<float> candela;
    const float peakCandela;

private:
    IESProfile(std::filesystem::path path, std::vector<float> vertical, std::vector<float> horizontal,
        std::vector<float> samples, float peak);
    friend std::shared_ptr<const IESProfile> loadIESProfile(const std::filesystem::path&, std::string&);
};

std::shared_ptr<const IESProfile> loadIESProfile(const std::filesystem::path& path, std::string& reason);

} // namespace metallic::scene
