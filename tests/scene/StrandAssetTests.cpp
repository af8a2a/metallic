#include "Runtime/Scene/StrandAsset.h"
#include <fstream>
#include <gtest/gtest.h>
#include <json.hpp>

namespace metallic::tests {
TEST(StrandAssets, ValidationAndTransactionalPublication)
{
    const auto root = std::filesystem::path(PROJECT_SOURCE_DIR) / "build/strand-assets-cpu";
    std::filesystem::create_directories(root);
    const auto path = root / "fixture.strands.json";
    auto data = nlohmann::json::parse(
        R"({"version":1,"materialRoot":".","materials":["asset://Fiber.material"],"strands":[{"id":17,"material":0,"points":[{"position":[0,0,0],"radius":0.01},{"position":[0,1,0],"radius":0.001}]}]})");
    std::string error;
    scene::StrandAsset asset;
    const auto load = [&](const auto& value) {
        std::ofstream(path) << value.dump();
        return scene::loadStrandAsset(path, asset, error);
    };
    ASSERT_TRUE(load(data)) << error;
    ASSERT_EQ(asset.segmentCount, 1u);
    EXPECT_EQ(asset.strands[0].id, 17u);
    EXPECT_EQ(asset.strands[0].points[1].position, asset.strands[0].points[1].previousPosition);
    for (int mutation = 0; mutation < 11; ++mutation) {
        auto bad = data;
        if (mutation == 0) {
            bad["version"] = 2;
        }
        if (mutation == 1) {
            bad["strands"].push_back(bad["strands"][0]);
        }
        if (mutation == 2) {
            bad["strands"][0]["points"][0]["radius"] = 0;
        }
        if (mutation == 3) {
            bad["strands"][0]["points"][1]["position"] = {0, 0, 0};
        }
        if (mutation == 4) {
            bad["strands"][0]["material"] = 1;
        }
        if (mutation == 5) {
            bad["strands"][0]["normal"] = {0, 0, 0};
        }
        if (mutation == 6) {
            bad["strands"][0]["points"][0]["previousPosition"] = {0, 1, 0};
        }
        if (mutation == 7) {
            bad["strands"][0]["id"] = 17.5;
        }
        if (mutation == 8) {
            bad["strands"][0]["material"] = 0.5;
        }
        if (mutation == 9) {
            bad["strands"][0]["normal"] = {1e30, 0, 0};
        }
        if (mutation == 10) {
            bad["strands"][0]["points"][1]["position"] = {1e30, 0, 0};
        }
        EXPECT_FALSE(load(bad)) << mutation;
        EXPECT_FALSE(error.empty());
        EXPECT_EQ(asset.segmentCount, 1u);
        EXPECT_EQ(asset.strands[0].id, 17u);
    }
}
TEST(StrandAssets, GroomAndStableOrdering)
{
    scene::StrandAsset asset;
    std::string error;
    ASSERT_TRUE(scene::loadStrandAsset(
        std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/Strands/NativeGroom.strands.json", asset, error))
        << error;
    EXPECT_EQ(asset.strands.size(), 72u);
    EXPECT_EQ(asset.segmentCount, 576u);
    for (size_t i = 1; i < asset.strands.size(); ++i) {
        EXPECT_LT(asset.strands[i - 1].id, asset.strands[i].id);
    }
}
} // namespace metallic::tests
