#include "Runtime/Scene/MeshCards.h"

#include "json.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>

namespace metallic::scene {
namespace {

using Json = nlohmann::json;

bool validCard(const MeshCardBuildRecord& card)
{
    const auto finite = [](const auto& vector) {
        return std::ranges::all_of(vector, [](float value) { return std::isfinite(value); });
    };
    const auto dot = [](const auto& a, const auto& b) {
        return std::inner_product(a.begin(), a.end(), b.begin(), 0.0f);
    };
    constexpr float tolerance = 1.0e-4f;
    return card.axisAlignedDirectionIndex < 6 && finite(card.origin) && finite(card.extent) &&
        std::ranges::all_of(card.extent, [](float value) { return value > 0.0f; }) &&
        std::abs(dot(card.axisX, card.axisX) - 1.0f) < tolerance &&
        std::abs(dot(card.axisY, card.axisY) - 1.0f) < tolerance &&
        std::abs(dot(card.axisZ, card.axisZ) - 1.0f) < tolerance &&
        std::abs(dot(card.axisX, card.axisY)) < tolerance &&
        std::abs(dot(card.axisX, card.axisZ)) < tolerance &&
        std::abs(dot(card.axisY, card.axisZ)) < tolerance;
}

bool validCards(const MeshCards& cards)
{
    return cards.sourceMeshIndex >= 0 && std::ranges::all_of(cards.cards, validCard);
}

} // namespace

bool loadMeshCards(const std::filesystem::path& path, MeshCards& result, std::string& error)
{
    std::ifstream file(path);
    if (!file) {
        error = "Could not open mesh cards: " + path.string();
        return false;
    }

    MeshCards cards;
    try {
        const auto json = Json::parse(file);
        if (json.at("version") != 1 || !json.at("cards").is_array()) {
            error = "Unsupported mesh cards format: " + path.string();
            return false;
        }
        cards.sourceMeshIndex = json.at("sourceMeshIndex").get<int32_t>();
        cards.sourceFingerprint = json.at("sourceFingerprint").get<uint64_t>();
        for (const auto& card : json.at("cards")) {
            cards.cards.push_back({card.at("origin"), card.at("axisX"), card.at("axisY"), card.at("axisZ"),
                card.at("extent"), card.at("axisAlignedDirectionIndex")});
        }
    } catch (const Json::exception& e) {
        error = "Invalid mesh cards in " + path.string() + ": " + e.what();
        return false;
    }
    if (!validCards(cards)) {
        error = "Invalid mesh card bounds or directions: " + path.string();
        return false;
    }
    result = std::move(cards);
    return true;
}

bool saveMeshCards(const std::filesystem::path& path, const MeshCards& cards, std::string& error)
{
    if (!validCards(cards)) {
        error = "Invalid mesh card bounds or directions: " + path.string();
        return false;
    }
    auto records = Json::array();
    for (const auto& card : cards.cards) {
        records.push_back({{"origin", card.origin}, {"axisX", card.axisX}, {"axisY", card.axisY},
            {"axisZ", card.axisZ}, {"extent", card.extent}, {"axisAlignedDirectionIndex", card.axisAlignedDirectionIndex}});
    }
    const Json json{{"version", 1}, {"sourceMeshIndex", cards.sourceMeshIndex},
        {"sourceFingerprint", cards.sourceFingerprint}, {"cards", std::move(records)}};
    std::ofstream file(path);
    file << json.dump(4) << '\n';
    file.close();
    if (!file) {
        error = "Could not write mesh cards: " + path.string();
        return false;
    }
    return true;
}

} // namespace metallic::scene
