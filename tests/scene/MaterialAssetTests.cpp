#include "Runtime/Material/MaterialAsset.h"
#include "Runtime/Material/MaterialAssetFields.h"
#include "Runtime/Scene/SceneDocument.h"
#include <gtest/gtest.h>
#include <fstream>

namespace metallic::tests {
namespace {

using namespace material;
using Json = nlohmann::json;

class MaterialAssets : public ::testing::Test
{
protected:
    std::filesystem::path root;
    std::string error;
    MaterialDefinition definition = defaultOpenPBRDefinition();

    void SetUp() override
    {
        root = std::filesystem::path(PROJECT_SOURCE_DIR) / "build/material-phase1-cpu" /
            ::testing::UnitTest::GetInstance()->current_test_info()->name();
        std::filesystem::create_directories(root);
        write("OpenPBR.materialdef", serializeMaterialDefinition(definition));
    }

    void write(const char* name, const std::string& text)
    {
        std::ofstream(root / name, std::ios::binary) << text;
    }

    MaterialInstance instance()
    {
        MaterialInstance result;
        result.definition = "asset://OpenPBR.materialdef";
        return result;
    }

    void loadScene(scene::SceneDocument& document)
    {
        const auto source = std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/LookDev/OpenPBRDefault";
        std::ifstream stream(source / "OpenPbrDefault.gltf");
        auto gltf = Json::parse(stream);
        for (auto& buffer : gltf["buffers"]) {
            const auto name = buffer["uri"].get<std::string>();
            std::filesystem::copy_file(source / name, root / name, std::filesystem::copy_options::overwrite_existing);
        }
        write("Scene.gltf", gltf.dump());
        std::filesystem::remove(root / "Scene.metallic_scene.json");
        ASSERT_TRUE(document.load(root / "Scene.gltf")) << document.lastLoadResult().error;
    }
};

TEST_F(MaterialAssets, FeatureSignaturesSeparateProgramVisibilityAndPipeline)
{
    scene::RenderMaterial value;
    value.metallicFactor = 0;
    const auto original = resolveMaterialFeatures(value);
    value.baseColorFactor.x = 0.2f;
    value.roughnessFactor = 0.3f;
    value.baseColorTexture.textureIndex = 42;
    EXPECT_EQ(resolveMaterialFeatures(value).programSignature, original.programSignature);
    for (const auto* mode : {"MASK", "BLEND"}) {
        value.alphaMode = mode;
        const auto changed = resolveMaterialFeatures(value);
        EXPECT_EQ(changed.programSignature, original.programSignature);
        EXPECT_NE(changed.visibilitySignature, original.visibilitySignature);
        EXPECT_EQ(changed.pipelineSignature, original.pipelineSignature);
    }
    value.alphaMode = "OPAQUE";
    value.doubleSided = true;
    const auto twoSided = resolveMaterialFeatures(value);
    EXPECT_EQ(twoSided.programSignature, original.programSignature);
    EXPECT_EQ(twoSided.visibilitySignature, original.visibilitySignature);
    EXPECT_NE(twoSided.pipelineSignature, original.pipelineSignature);
    uint32_t categories = 0;
    for (const auto& descriptor : materialFeatureDescriptors()) { categories |= descriptor.categories; }
    EXPECT_EQ(categories, 63u);
}

TEST_F(MaterialAssets, FeatureAutoConservativelyClassifiesClosures)
{
    scene::RenderMaterial value;
    EXPECT_EQ(resolveMaterialFeatures(value).surfaceProgram, SurfaceProgramClass::Conductor);
    value.metallicRoughnessTexture.textureIndex = 12;
    EXPECT_EQ(resolveMaterialFeatures(value).surfaceProgram, SurfaceProgramClass::Opaque);
    value.featurePolicies.metalness = FeaturePolicy::Specialization;
    EXPECT_EQ(resolveMaterialFeatures(value).surfaceProgram, SurfaceProgramClass::Opaque);
    value.metallicFactor = 0;
    EXPECT_EQ(resolveMaterialFeatures(value).surfaceProgram, SurfaceProgramClass::Dielectric);
    value.featurePolicies.metalness = FeaturePolicy::Dynamic;
    EXPECT_EQ(resolveMaterialFeatures(value).surfaceProgram, SurfaceProgramClass::Opaque);
    value.featurePolicies.transmission = FeaturePolicy::Dynamic;
    EXPECT_EQ(resolveMaterialFeatures(value).surfaceProgram, SurfaceProgramClass::General);
    value.featurePolicies = {};
    value.transmissionFactor = 0.5f;
    EXPECT_EQ(resolveMaterialFeatures(value).surfaceProgram, SurfaceProgramClass::General);
    value.transmissionFactor = 0;
    value.valueProgram = R"({"version":1,"metallic":{"op":"parameter","index":0}})";
    const auto graph = resolveMaterialFeatures(value);
    EXPECT_EQ(graph.surfaceProgram, SurfaceProgramClass::General);
    value.valueParameters[0] = 0.9f;
    value.valueProgram = R"({ "metallic": {"index":0,"op":"parameter"}, "version":1 })";
    EXPECT_EQ(resolveMaterialFeatures(value).programSignature, graph.programSignature);
    value.valueProgram = R"({"version":1,"metallic":{"op":"parameter","index":1}})";
    EXPECT_NE(resolveMaterialFeatures(value).programSignature, graph.programSignature);
    value.valueProgram.clear();
    EXPECT_EQ(resolveMaterialFeatures(value, FeatureCompileTarget::RayHit).surfaceProgram, SurfaceProgramClass::General);
}

TEST_F(MaterialAssets, CoverageAndTransmissionHaveSeparateSignatures)
{
    scene::RenderMaterial value;
    value.alphaMode = "MASK";
    const auto baseline = resolveMaterialFeatures(value);
    value.valueProgram = R"({"version":1,"coverage":{"op":"parameter","index":0}})";
    const auto coverage = resolveMaterialFeatures(value);
    EXPECT_EQ(coverage.programSignature, baseline.programSignature);
    EXPECT_NE(coverage.visibilitySignature, baseline.visibilitySignature);
    value.valueParameters[0] = 0.4f;
    EXPECT_EQ(resolveMaterialFeatures(value).visibilitySignature, coverage.visibilitySignature);
    value.transmissionFactor = 0.9f;
    EXPECT_EQ(resolveMaterialFeatures(value).visibilitySignature, coverage.visibilitySignature);
    EXPECT_NE(resolveMaterialFeatures(value).programSignature, coverage.programSignature);
}

TEST_F(MaterialAssets, ValueIRSignaturesIgnoreDeadNodesAndCoverageDefinitions)
{
    scene::RenderMaterial value;
    value.alphaMode = "MASK";
    value.valueProgram = R"({"version":1,"roughness":0.5,"coverage":{"op":"parameter","index":0}})";
    const auto nested = resolveMaterialFeatures(value);
    value.valueProgram = R"({"version":2,"nodes":{"dead":{"op":"textureSample"},"r":{"op":"add","args":[0.25,0.25]},"c":{"op":"parameter","index":0}},"outputs":{"roughness":{"ref":"r"},"coverage":{"ref":"c"}}})";
    const auto graph = resolveMaterialFeatures(value);
    EXPECT_EQ(nested.programSignature, graph.programSignature);
    EXPECT_EQ(nested.visibilitySignature, graph.visibilitySignature);
    value.valueProgram = R"({"version":2,"nodes":{"dead":{"op":"textureSample"}},"outputs":{"coverage":{"op":"parameter","index":0}}})";
    const auto coverageOnly = resolveMaterialFeatures(value);
    value.valueProgram.clear();
    EXPECT_EQ(coverageOnly.programSignature, resolveMaterialFeatures(value).programSignature);
}

TEST_F(MaterialAssets, FeaturePoliciesInheritWithoutPersistingCompilerDecisions)
{
    definition.featurePolicies.metalness = FeaturePolicy::Dynamic;
    write("OpenPBR.materialdef", serializeMaterialDefinition(definition));
    MaterialAssetLibrary library(root);
    auto parent = instance();
    parent.featurePolicies = {{"transmission", "Dynamic"}};
    ASSERT_TRUE(library.save("asset://Parent.material", parent, error)) << error;
    auto child = instance();
    child.parent = "asset://Parent.material";
    child.featurePolicies = {{"transmission", "Auto"}};
    ResolvedMaterialInstance resolved;
    ASSERT_TRUE(library.resolve(child, resolved, error)) << error;
    EXPECT_EQ(resolved.featurePolicies.metalness, FeaturePolicy::Dynamic);
    EXPECT_EQ(resolved.featurePolicies.transmission, FeaturePolicy::Auto);
    EXPECT_EQ(resolved.featureResolution.surfaceProgram, SurfaceProgramClass::Opaque);
    scene::RenderMaterial lowered;
    ASSERT_TRUE(lowerMaterialInstance(resolved, {}, lowered, error)) << error;
    EXPECT_EQ(lowered.featurePolicies, resolved.featurePolicies);
    EXPECT_EQ(resolveMaterialFeatures(lowered).programSignature, resolved.featureResolution.programSignature);
    const auto text = serializeMaterialInstance(child);
    EXPECT_EQ(text.find("Signature"), std::string::npos);
    EXPECT_EQ(text.find("variant"), std::string::npos);
    auto invalid = Json::parse(text);
    invalid["variantId"] = 3;
    MaterialInstance parsed = child;
    EXPECT_FALSE(deserializeMaterialInstance(invalid.dump(), parsed, error));
    EXPECT_EQ(parsed.featurePolicies, child.featurePolicies);
    invalid.erase("variantId");
    invalid["featurePolicies"]["roughness"] = "Specialization";
    EXPECT_FALSE(deserializeMaterialInstance(invalid.dump(), parsed, error));
    invalid["featurePolicies"] = {{"metalness", "Keyword123"}};
    EXPECT_FALSE(deserializeMaterialInstance(invalid.dump(), parsed, error));
    MaterialInstance exported;
    ASSERT_TRUE(createMaterialInstance(lowered, child.definition, definition, {}, exported, error)) << error;
    ASSERT_TRUE(library.resolve(exported, resolved, error)) << error;
    EXPECT_EQ(resolved.featurePolicies, lowered.featurePolicies);
}

TEST_F(MaterialAssets, FeaturePolicyEditsPersistAndAdvanceMaterialRevision)
{
    scene::SceneDocument document;
    loadScene(document);
    auto edited = document.materials()[0];
    const auto revision = document.materialRevision();
    edited.featurePolicies.transmission = FeaturePolicy::Dynamic;
    ASSERT_TRUE(document.setMaterialProperties(0, edited));
    EXPECT_GT(document.materialRevision(), revision);
    ASSERT_TRUE(document.save(error)) << error;
    scene::SceneDocument loaded;
    ASSERT_TRUE(loaded.load(document.documentPath())) << loaded.documentWarning();
    EXPECT_EQ(loaded.materials()[0].featurePolicies, edited.featurePolicies);
    edited.featurePolicies.transmission = static_cast<FeaturePolicy>(999);
    EXPECT_FALSE(loaded.setMaterialProperties(0, edited));
}

TEST_F(MaterialAssets, LocalFeaturePolicyDoesNotFreezeOtherInheritedPolicies)
{
    scene::SceneDocument document;
    loadScene(document);
    MaterialAssetLibrary library(root);
    ASSERT_TRUE(library.save("asset://Surface.material", instance(), error)) << error;
    ASSERT_TRUE(document.setMaterialAsset(0, "asset://Surface.material", root, error)) << error;
    auto edited = document.materials()[0];
    edited.featurePolicies.metalness = FeaturePolicy::Dynamic;
    ASSERT_TRUE(document.setMaterialProperties(0, edited));
    ASSERT_TRUE(document.save(error)) << error;
    definition.featurePolicies.transmission = FeaturePolicy::Dynamic;
    write("OpenPBR.materialdef", serializeMaterialDefinition(definition));
    ASSERT_TRUE(document.reloadMaterialAsset(0, error)) << error;
    EXPECT_EQ(document.materials()[0].featurePolicies.metalness, FeaturePolicy::Dynamic);
    EXPECT_EQ(document.materials()[0].featurePolicies.transmission, FeaturePolicy::Dynamic);
    scene::SceneDocument loaded;
    ASSERT_TRUE(loaded.load(document.documentPath())) << loaded.documentWarning();
    EXPECT_EQ(loaded.materials()[0].featurePolicies, document.materials()[0].featurePolicies);
}

TEST_F(MaterialAssets, SparseRoundTripAndInheritedDefaults)
{
    MaterialAssetLibrary library(root);
    auto parent = instance();
    parent.parameters = {{"roughness", 0.25}, {"baseColor", {0.8, 0.2, 0.1}}};
    parent.features = {{"doubleSided", true}};
    parent.resources = {{"baseColorTexture", "asset://Wood.png"}};
    ASSERT_TRUE(library.save("asset://Parent.material", parent, error)) << error;
    auto child = instance();
    child.parent = "asset://Parent.material";
    child.parameters = {{"roughness", 0.5}};
    child.resources = {{"baseColorTexture", nullptr}};
    ASSERT_TRUE(library.save("asset://Child.material", child, error)) << error;
    MaterialInstance reloaded;
    ASSERT_TRUE(deserializeMaterialInstance(serializeMaterialInstance(child), reloaded, error)) << error;
    EXPECT_EQ(reloaded.parameters.size(), 1);
    EXPECT_EQ(reloaded.resources, child.resources);
    EXPECT_EQ(serializeMaterialInstance(reloaded), serializeMaterialInstance(child));
    ResolvedMaterialInstance resolved;
    ASSERT_TRUE(library.resolve("asset://Child.material", resolved, error)) << error;
    EXPECT_EQ(resolved.parameters["roughness"], 0.5);
    EXPECT_EQ(resolved.parameters["baseColor"], parent.parameters["baseColor"]);
    EXPECT_EQ(resolved.features["doubleSided"], true);
    EXPECT_TRUE(resolved.resources["baseColorTexture"].is_null());
    for (auto& property : definition.schema.parameters) {
        if (property.name == "metalness") { property.defaultValue = 0.2; }
    }
    write("OpenPBR.materialdef", serializeMaterialDefinition(definition));
    ASSERT_TRUE(library.resolve("asset://Child.material", resolved, error)) << error;
    EXPECT_EQ(resolved.parameters["metalness"], 0.2);
    EXPECT_EQ(resolved.parameters["roughness"], 0.5);
    // Explicit equal-to-default overrides are retained, not silently discarded.
    child.parameters["metalness"] = 0.2;
    EXPECT_TRUE(Json::parse(serializeMaterialInstance(child))["parameters"].contains("metalness"));
}

TEST_F(MaterialAssets, AllOpenPBRFieldsAndTextureTransformsRoundTrip)
{
    scene::RenderMaterial source;
    source.name = "Imported identity";
    source.baseColorFactor = float4(0.13f, 0.27f, 0.82f, 0.4f);
    source.metallicFactor = 0.3f;
    source.roughnessFactor = 0.6f;
    source.emissiveFactor = float3(2, 5, 8);
    source.normalTextureScale = -0.7f;
    source.displacementMagnitude = -2.0f;
    source.displacementCenter = 0.35f;
    source.occlusionTextureStrength = 0.8f;
    source.specularFactor = 0.6f;
    source.specularColorFactor = float3(0.9f, 0.8f, 0.7f);
    source.transmissionFactor = 0.4f;
    source.ior = 1.7f;
    source.thicknessFactor = 2.1f;
    source.attenuationDistance = 3.2f;
    source.attenuationColor = float3(0.4f, 0.5f, 0.6f);
    source.diffuseTransmissionFactor = 0.2f;
    source.diffuseTransmissionColor = float3(0.1f, 0.2f, 0.3f);
    source.alphaMode = "BLEND";
    source.alphaCutoff = 0.7f;
    source.doubleSided = true;
    source.unlit = true;
    for (const auto& [name, member] : detail::kTextures) {
        source.*member = {.textureIndex = 7, .texCoord = 1, .uvTransform = {2, 0, 0.25f, 0, -3, 0.75f}};
    }
    MaterialInstance asset;
    ASSERT_TRUE(createMaterialInstance(source, "asset://OpenPBR.materialdef", definition,
        [](auto, auto) { return "asset://Textures/Imported.ktx2"; }, asset, error)) << error;
    const auto text = serializeMaterialInstance(asset);
    EXPECT_EQ(text.find("textureIndex"), std::string::npos);
    EXPECT_EQ(text.find("programId"), std::string::npos);
    MaterialAssetLibrary library(root);
    ResolvedMaterialInstance resolved;
    ASSERT_TRUE(library.resolve(asset, resolved, error)) << error;
    scene::RenderMaterial output;
    output.name = source.name;
    ASSERT_TRUE(lowerMaterialInstance(resolved, [](auto uri) { return uri == "asset://Textures/Imported.ktx2" ? 7 : -1; }, output, error)) << error;
    EXPECT_TRUE(scene::materialPropertiesEqual(source, output));
    for (const auto& [name, member] : detail::kTextures) {
        EXPECT_EQ((source.*member).textureIndex, (output.*member).textureIndex) << name;
        EXPECT_EQ((source.*member).texCoord, (output.*member).texCoord) << name;
        EXPECT_EQ((source.*member).uvTransform, (output.*member).uvTransform) << name;
    }
}

TEST_F(MaterialAssets, ColorTextureTagsSurviveExportAndRejectDataUsage)
{
    scene::RenderMaterial source;
    source.baseColorTexture.textureIndex = 7;
    source.baseColorTexture.colorMetadata.source = render::kACEScg;
    source.normalTexture.textureIndex = 8;
    MaterialInstance asset;
    ASSERT_TRUE(createMaterialInstance(source, "asset://OpenPBR.materialdef", definition,
        [](auto, auto) { return "asset://Textures/Tagged.hdr"; }, asset, error)) << error;
    EXPECT_EQ(asset.resources["baseColorTexture"]["colorSpace"], "acescg");
    EXPECT_FALSE(asset.resources["normalTexture"].contains("colorSpace"));
    MaterialInstance parsed;
    ASSERT_TRUE(deserializeMaterialInstance(serializeMaterialInstance(asset), parsed, error)) << error;
    MaterialAssetLibrary library(root);
    ResolvedMaterialInstance resolved;
    ASSERT_TRUE(library.resolve(parsed, resolved, error)) << error;
    scene::RenderMaterial output;
    ASSERT_TRUE(lowerMaterialInstance(resolved, [](auto) { return 7; }, output, error)) << error;
    EXPECT_EQ(output.baseColorTexture.colorMetadata.source, render::kACEScg);
    EXPECT_EQ(output.normalTexture.colorMetadata.semantic, render::TextureSemantic::Data);
    resolved.resources["normalTexture"]["colorSpace"] = "acescg";
    EXPECT_FALSE(lowerMaterialInstance(resolved, [](auto) { return 7; }, output, error));
    EXPECT_NE(error.find("Data textures"), std::string::npos);
}

TEST_F(MaterialAssets, RejectsCyclesMissingParentsAndDefinitionMismatch)
{
    MaterialAssetLibrary library(root);
    auto child = instance();
    child.parent = "asset://Missing.material";
    ResolvedMaterialInstance output;
    output.definitionUri = "unchanged";
    EXPECT_FALSE(library.resolve(child, output, error));
    EXPECT_EQ(output.definitionUri, "unchanged");
    child.parent = "asset://Loop.material";
    write("Loop.material", serializeMaterialInstance(child));
    EXPECT_FALSE(library.resolve(child, output, error));
    EXPECT_NE(error.find("cycle"), std::string::npos);
    auto parent = instance();
    parent.definition = "asset://Other.materialdef";
    write("Parent.material", serializeMaterialInstance(parent));
    child.parent = "asset://Parent.material";
    EXPECT_FALSE(library.resolve(child, output, error));
    EXPECT_NE(error.find("different definition"), std::string::npos);
    child.parent.reset();
    child.definitionVersion = 2;
    EXPECT_FALSE(library.resolve(child, output, error));
    EXPECT_NE(error.find("version mismatch"), std::string::npos);
    EXPECT_THROW(library.pathFor("asset://../escape.material"), std::exception);
}

TEST_F(MaterialAssets, MigrationAndStrictSemanticSerialization)
{
    auto document = Json::parse(serializeMaterialInstance(instance()));
    document["version"] = 0;
    document["parameters"]["metallic"] = 0.2;
    ASSERT_TRUE(upgradeMaterial(document, 0, 1, error)) << error;
    EXPECT_FALSE(document["parameters"].contains("metallic"));
    EXPECT_EQ(document["parameters"]["metalness"], 0.2);
    const auto before = document;
    EXPECT_FALSE(upgradeMaterial(document, 1, 99, error));
    EXPECT_EQ(document, before);
    MaterialInstance output = instance();
    const auto original = serializeMaterialInstance(output);
    for (const auto* name : {"shaderVariant", "pipelineHandle", "descriptorIndex", "parameterOffset"}) {
        auto invalid = before;
        invalid[name] = 0;
        EXPECT_FALSE(deserializeMaterialInstance(invalid.dump(), output, error)) << name;
        EXPECT_EQ(serializeMaterialInstance(output), original);
    }
    document["parameters"]["unknown"] = 1;
    EXPECT_FALSE(deserializeMaterialInstance(document.dump(), output, error));
    document = before;
    document["parameters"]["roughness"] = -1;
    EXPECT_FALSE(deserializeMaterialInstance(document.dump(), output, error));
    document = before;
    document["resources"]["normalTexture"] = {{"textureIndex", 7}};
    EXPECT_FALSE(deserializeMaterialInstance(document.dump(), output, error));
    document = before;
    document["features"]["doubleSided"] = "false";
    EXPECT_FALSE(deserializeMaterialInstance(document.dump(), output, error));
}

TEST_F(MaterialAssets, SaveFailurePreservesOldAsset)
{
    MaterialAssetLibrary library(root);
    auto asset = instance();
    ASSERT_TRUE(library.save("asset://Existing.material", asset, error)) << error;
    const auto original = serializeMaterialInstance(asset);
    asset.parameters["roughness"] = -1;
    EXPECT_FALSE(library.save("asset://Existing.material", asset, error));
    std::ifstream stream(root / "Existing.material");
    EXPECT_EQ(std::string(std::istreambuf_iterator<char>(stream), {}), original);
    asset = instance();
    asset.parent = "asset://Existing.material";
    EXPECT_FALSE(library.save("asset://Existing.material", asset, error));
}

TEST_F(MaterialAssets, SceneBindingPersistsReferenceAndInheritsNewDefaults)
{
    MaterialAssetLibrary library(root);
    auto asset = instance();
    asset.parameters["roughness"] = 0.25;
    ASSERT_TRUE(library.save("asset://Surface.material", asset, error)) << error;
    scene::SceneDocument document;
    loadScene(document);
    ASSERT_TRUE(document.valid());
    const auto identity = document.resourceIdentity();
    ASSERT_TRUE(document.setMaterialAsset(0, "asset://Surface.material", root, error)) << error;
    EXPECT_EQ(document.resourceIdentity(), identity);
    auto edited = document.materials()[0];
    edited.metallicFactor = 0.4f;
    ASSERT_TRUE(document.setMaterialProperties(0, edited));
    ASSERT_TRUE(document.save(error)) << error;
    std::ifstream saved(document.documentPath());
    const auto sidecar = Json::parse(saved);
    EXPECT_EQ(sidecar["materials"][0]["materialAsset"]["uri"], "asset://Surface.material");
    EXPECT_EQ(sidecar["materials"][0]["properties"].size(), 1);
    EXPECT_EQ(sidecar["materials"][0]["properties"]["metallicFactor"], edited.metallicFactor);
    for (auto& field : definition.schema.parameters) {
        if (field.name == "baseColor") { field.defaultValue = {0.15, 0.25, 0.35}; }
    }
    write("OpenPBR.materialdef", serializeMaterialDefinition(definition));
    ASSERT_TRUE(document.reloadMaterialAsset(0, error)) << error;
    EXPECT_FLOAT_EQ(document.materials()[0].baseColorFactor.x, 0.15f);
    EXPECT_FLOAT_EQ(document.materials()[0].metallicFactor, 0.4f);
    scene::SceneDocument reloaded;
    ASSERT_TRUE(reloaded.load(document.documentPath())) << reloaded.documentWarning();
    EXPECT_TRUE(reloaded.documentWarning().empty()) << reloaded.documentWarning();
    EXPECT_FLOAT_EQ(reloaded.materials()[0].baseColorFactor.x, 0.15f);
    EXPECT_FLOAT_EQ(reloaded.materials()[0].metallicFactor, 0.4f);
    EXPECT_FLOAT_EQ(reloaded.materials()[0].roughnessFactor, 0.25f);
    const auto revision = document.materialRevision();
    write("OpenPBR.materialdef", "invalid");
    EXPECT_FALSE(document.reloadMaterialAsset(0, error));
    EXPECT_EQ(document.materialRevision(), revision);
    EXPECT_FLOAT_EQ(document.materials()[0].baseColorFactor.x, 0.15f);
}

TEST_F(MaterialAssets, SceneOwnedValueProgramSurvivesAssetBindingAndReload)
{
    scene::SceneDocument document;
    loadScene(document);
    ASSERT_TRUE(document.valid());
    auto edited = document.materials()[0];
    edited.valueProgram = R"({"version":1,"baseColor":{"op":"parameter","index":0}})";
    edited.valueParameters[0] = 0.3f;
    ASSERT_TRUE(document.setMaterialProperties(0, edited));
    MaterialAssetLibrary library(root);
    ASSERT_TRUE(library.save("asset://Surface.material", instance(), error)) << error;
    ASSERT_TRUE(document.setMaterialAsset(0, "asset://Surface.material", root, error)) << error;
    ASSERT_TRUE(document.reloadMaterialAsset(0, error)) << error;
    ASSERT_TRUE(document.save(error)) << error;
    scene::SceneDocument reloaded;
    ASSERT_TRUE(reloaded.load(document.documentPath())) << reloaded.documentWarning();
    EXPECT_EQ(reloaded.materials()[0].valueProgram, edited.valueProgram);
    EXPECT_EQ(reloaded.materials()[0].valueParameters, edited.valueParameters);
}

TEST_F(MaterialAssets, TexturePublicationAndMissingResourceAreTransactional)
{
    scene::SceneDocument document;
    loadScene(document);
    ASSERT_TRUE(document.valid());
    MaterialAssetLibrary library(root);
    auto asset = instance();
    asset.resources["baseColorTexture"] = "asset://Albedo.png";
    asset.resources["normalTexture"] = "asset://Missing.png";
    write("Albedo.png", "CPU resource identity fixture; no GPU decode in this test");
    ASSERT_TRUE(library.save("asset://Surface.material", asset, error)) << error;
    const auto identity = document.resourceIdentity();
    const auto count = document.textures().size();
    EXPECT_FALSE(document.setMaterialAsset(0, "asset://Surface.material", root, error));
    EXPECT_EQ(document.resourceIdentity(), identity);
    EXPECT_EQ(document.textures().size(), count);
    asset.resources.erase("normalTexture");
    ASSERT_TRUE(library.save("asset://Surface.material", asset, error)) << error;
    ASSERT_TRUE(document.setMaterialAsset(0, "asset://Surface.material", root, error)) << error;
    EXPECT_NE(document.resourceIdentity(), identity);
    EXPECT_EQ(document.textures().size(), count + 1);
    const auto published = document.resourceIdentity();
    ASSERT_TRUE(document.reloadMaterialAsset(0, error)) << error;
    EXPECT_EQ(document.resourceIdentity(), published);
    EXPECT_EQ(document.textures().size(), count + 1);
}

} // namespace
} // namespace metallic::tests
