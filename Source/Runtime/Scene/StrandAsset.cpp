#include "StrandAsset.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <json.hpp>
#include <set>
#include <stdexcept>

namespace metallic::scene {
bool loadStrandAsset(const std::filesystem::path& path, StrandAsset& output, std::string& error)
{
    try {
        std::ifstream stream(path);
        const auto json = nlohmann::json::parse(stream);
        auto require = [](bool valid, const char* reason) {
            if (!valid) {
                throw std::runtime_error(reason);
            }
        };
        require(json.at("version") == 1, "Unsupported strand version");
        StrandAsset candidate;
        candidate.materialRoot =
            std::filesystem::absolute(path.parent_path() / json.at("materialRoot").get<std::string>());
        candidate.materials = json.at("materials").get<std::vector<std::string>>();
        require(!candidate.materials.empty() && candidate.materials.size() <= 64, "Strand material capacity is 1..64");
        std::set<uint32_t> ids;
        for (const auto& value : json.at("strands")) {
            NativeStrand strand;
            require(value.at("id").is_number_integer() && value.at("material").is_number_integer(),
                    "Strand identities and material indices must be integers");
            const auto id = value.at("id").get<int64_t>();
            const auto material = value.at("material").get<int64_t>();
            require(id >= 0 && id < UINT32_MAX && ids.insert(uint32_t(id)).second, "Invalid/duplicate strand identity");
            require(material >= 0 && material < candidate.materials.size(), "Invalid strand material index");
            strand.id = uint32_t(id);
            strand.material = uint32_t(material);
            strand.opacity = value.value("opacity", 1.0f);
            require(std::isfinite(strand.opacity) && strand.opacity >= 0 && strand.opacity <= 1,
                    "Invalid strand opacity");
            strand.normal = value.value("normal", std::array<float, 3>{0, 0, 1});
            float normalLength = 0;
            for (float v : strand.normal) {
                require(std::isfinite(v), "Nonfinite strand normal");
                normalLength += v * v;
            }
            require(std::isfinite(normalLength) && normalLength > 1e-12f, "Degenerate/overflowing strand normal");
            for (float& v : strand.normal) {
                v /= std::sqrt(normalLength);
            }
            for (const auto& point : value.at("points")) {
                auto p = point.at("position").get<std::array<float, 3>>();
                auto previous = point.value("previousPosition", p);
                const float radius = point.at("radius").get<float>();
                require(std::isfinite(radius) && radius > 0, "Strand radius must be positive and finite");
                for (size_t i = 0; i < 3; ++i) {
                    require(std::isfinite(p[i]) && std::isfinite(previous[i]), "Nonfinite strand position");
                }
                if (!strand.points.empty()) {
                    float length = 0, previousLength = 0;
                    for (size_t i = 0; i < 3; ++i) {
                        const float d = p[i] - strand.points.back().position[i];
                        length += d * d;
                        const float old = previous[i] - strand.points.back().previousPosition[i];
                        previousLength += old * old;
                    }
                    require(std::isfinite(length) && std::isfinite(previousLength) && length > 1e-14f &&
                                previousLength > 1e-14f,
                            "Degenerate/overflowing strand segment");
                    require(++candidate.segmentCount <= 4096, "Native strand backend capacity is 4096 segments");
                }
                strand.points.push_back({p, previous, radius});
            }
            require(strand.points.size() >= 2, "Strand needs at least two control points");
            candidate.strands.push_back(std::move(strand));
        }
        require(!candidate.strands.empty(), "Empty strand asset");
        std::ranges::sort(candidate.strands, {}, &NativeStrand::id);
        output = std::move(candidate);
        error.clear();
        return true;
    }
    catch (const std::exception& exception) {
        error = "Strand asset: " + std::string(exception.what());
        return false;
    }
}
} // namespace metallic::scene
