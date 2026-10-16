#include "Runtime/Render/Core/PipelineRecipe.h"
#include "Runtime/Render/GAPI/PipelineStateHash.h"

#include <gtest/gtest.h>
#include <json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace metallic::tests {
namespace {

using namespace render;

TEST(ShaderSpecialization, CanonicalScalar32AndMalformedModuleContracts)
{
    const ShaderSpecializationConstant zero{1000, ShaderSpecializationType::Int32, 0};
    const ShaderSpecializationConstant truth{1001, ShaderSpecializationType::Bool32, 1};
    const std::array reversed{truth, zero};
    const std::vector<ShaderSpecializationConstant> expected{zero, truth};
    EXPECT_EQ(detail::canonicalShaderSpecializationConstants(reversed), expected);
    auto invalid = reversed;
    invalid[0].bits = 2;
    EXPECT_THROW(detail::canonicalShaderSpecializationConstants(invalid), std::invalid_argument);
    invalid = reversed; invalid[0] = invalid[1];
    EXPECT_THROW(detail::canonicalShaderSpecializationConstants(invalid), std::invalid_argument);
    invalid = reversed; invalid[0].type = static_cast<ShaderSpecializationType>(9);
    EXPECT_THROW(detail::canonicalShaderSpecializationConstants(invalid), std::invalid_argument);
    const auto bytes = detail::shaderSpecializationBytes(expected);
    EXPECT_EQ(bytes.size(), 40);
    EXPECT_EQ(bytes[24], 0xe9); // little-endian SpecId 1001
    EXPECT_EQ(bytes[25], 0x03);
    const std::array<uint8_t, 3> driver{7, 8, 9};
    const auto empty = detail::specializationCompatibilityKey(driver, {});
    EXPECT_EQ(empty, (std::vector<uint8_t>{7, 8, 9}));
    const auto key = detail::specializationCompatibilityKey(driver, expected);
    auto other = expected; other[0].bits = 1;
    EXPECT_NE(key, detail::specializationCompatibilityKey(driver, other));
    EXPECT_NE(key, detail::specializationCompatibilityKey((std::array<uint8_t, 2>{7, 8}), expected));
    EXPECT_THROW(detail::specializationCompatibilityKey(std::array<uint8_t, 256>{}, expected), std::invalid_argument);
}

// A complete small scalar-specialization compute module. These CPU tests parse
// it only; native validity/readback is exercised separately, never inferred here.
std::vector<uint32_t> scalarSpecializationModule()
{
    return {0x07230203u, 0x00010600u, 0, 9, 0,
        0x00020011u, 1, 0x0003000eu, 0, 1,
        0x0005000fu, 5, 1, 0x6e69616du, 0,
        0x00060010u, 1, 17, 1, 1, 1,
        0x00040047u, 3, 1, 1000, 0x00040047u, 4, 1, 1001,
        0x00020013u, 5, 0x00030021u, 6, 5,
        0x00040015u, 2, 32, 1, 0x00020014u, 7,
        0x00040032u, 2, 3, 0xffffffffu, 0x00030031u, 7, 4,
        0x00050036u, 5, 1, 0, 6, 0x000200f8u, 8, 0x000100fdu, 0x00010038u};
}

TEST(ShaderSpecialization, ActualDeclarationTypeAndRecipeRoundtrip)
{
    const auto words = scalarSpecializationModule();
    const std::vector<ShaderSpecializationConstant> values{{1000, ShaderSpecializationType::Int32, 0xffffffffu},
        {1001, ShaderSpecializationType::Bool32, 1}};
    std::string error;
    ASSERT_TRUE(detail::validateShaderSpecializationModule(words, values, error)) << error;
    auto wrong = values; wrong[0].id = 9;
    EXPECT_FALSE(detail::validateShaderSpecializationModule(words, wrong, error));
    wrong = values; wrong[0].type = ShaderSpecializationType::Bool32; wrong[0].bits = 1;
    EXPECT_FALSE(detail::validateShaderSpecializationModule(words, wrong, error));
    auto malformed = words; malformed[5] = 0;
    EXPECT_FALSE(detail::validateShaderSpecializationModule(malformed, values, error));
    auto wrongWidth = words;
    const auto integer = std::find(wrongWidth.begin() + 5, wrongWidth.end(), 0x00040015u);
    ASSERT_NE(integer, wrongWidth.end());
    *(integer + 2) = 64;
    EXPECT_FALSE(detail::validateShaderSpecializationModule(wrongWidth, values, error));

    const auto directory = std::filesystem::temp_directory_path() /
        ("metallic-specialization-recipe-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(directory);
    struct RemoveFixture {
        std::filesystem::path path;
        ~RemoveFixture() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
    } cleanup{directory};
    PipelineRecipe recipe;
    recipe.computeShader.spirv = words;
    recipe.computeShader.specializationConstants = values;
    const auto path = directory / "compute.recipe.json";
    ASSERT_TRUE(writePipelineRecipe(recipe, path, error)) << error;
    auto loaded = readPipelineRecipe(path, error);
    ASSERT_TRUE(loaded) << error;
    EXPECT_EQ(loaded->computeShader.specializationConstants, values);
    EXPECT_EQ(loaded->computeShader.spirv, words);
    EXPECT_EQ(loaded->computeDesc(nullptr).computeShader.specializationConstants.size(), 2);
    const auto json = nlohmann::json::parse(std::ifstream(path));
    EXPECT_EQ(json.at("version"), 2);
    const auto rejects = [&](nlohmann::json changed) {
        const auto invalidPath = directory / "invalid.recipe.json";
        { std::ofstream output(invalidPath); output << changed.dump(2); }
        return !readPipelineRecipe(invalidPath, error);
    };
    auto changed = json; changed["shaders"]["compute"].erase("specialization_constants");
    EXPECT_TRUE(rejects(changed));
    changed = json; changed["shaders"]["compute"]["specialization_constants"][0]["size"] = 8;
    EXPECT_TRUE(rejects(changed));
    changed = json; changed["shaders"]["compute"]["specialization_constants"][1]["bits"] = 2;
    EXPECT_TRUE(rejects(changed));
    changed = json; changed["shaders"]["compute"]["specialization_constants"][1]["id"] = 1000;
    EXPECT_TRUE(rejects(changed));
    changed = json; std::swap(changed["shaders"]["compute"]["specialization_constants"][0],
        changed["shaders"]["compute"]["specialization_constants"][1]);
    EXPECT_TRUE(rejects(changed));
    recipe.computeShader.specializationConstants.clear();
    ASSERT_TRUE(writePipelineRecipe(recipe, path, error));
    const auto legacy = nlohmann::json::parse(std::ifstream(path));
    EXPECT_EQ(legacy.at("version"), 1);
    EXPECT_FALSE(legacy["shaders"]["compute"].contains("specialization_constants"));
    ASSERT_TRUE(readPipelineRecipe(path, error));
}

TEST(ShaderSpecialization, PipelineHashUsesCanonicalValuesNotPointers)
{
    std::array values{ShaderSpecializationConstant{1001, ShaderSpecializationType::Bool32, 1},
        ShaderSpecializationConstant{1000, ShaderSpecializationType::Int32, 0}};
    ComputePipelineDesc first;
    first.computeShader.specializationConstants = values;
    const auto firstHash = detail::computePipelineStateHash(first);
    auto copy = detail::canonicalShaderSpecializationConstants(values);
    auto second = first; second.computeShader.specializationConstants = copy;
    EXPECT_EQ(firstHash, detail::computePipelineStateHash(second));
    copy[0].bits = 1;
    EXPECT_NE(firstHash, detail::computePipelineStateHash(second));
    copy[0].bits = 0; copy[1].bits = 0;
    EXPECT_NE(firstHash, detail::computePipelineStateHash(second));
    second.computeShader.specializationConstants = {};
    EXPECT_NE(firstHash, detail::computePipelineStateHash(second));
}

} // namespace
} // namespace metallic::tests
