#include "Runtime/Material/MaterialGraph.h"
#include "Runtime/Scene/SceneDocument.h"
#include <gtest/gtest.h>
#include <fstream>

namespace metallic::tests {
namespace {
using namespace material;
using Json = nlohmann::json;

TEST(MaterialGraph, DefaultGraphAndSDKHaveIdenticalCanonicalIR)
{
    CompiledMaterialFrontend graph, text;
    std::string error;
    ASSERT_TRUE(compileMaterialGraph(defaultMaterialGraph(),graph,error)) << error;
    ASSERT_TRUE(compileSlangMaterial("import MaterialAuthoringSDK; Material evaluate() { let c = parameter(0); return openPBR(c,0,.35); }",
        graph.definition.valueParameters,text,error)) << error;
    EXPECT_EQ(graph.signature,text.signature);
    EXPECT_EQ(graph.definition.implementation,"OpenPBR.Value");
    EXPECT_EQ(graph.reflection["parameterMask"],1);
    EXPECT_EQ(graph.reflection["sideEffects"],false);
    const auto ir = render::MaterialValueIR::parse(graph.definition.surfaceProgram);
    EXPECT_TRUE(ir.outputs().contains("surfaceBaseColor"));
    EXPECT_TRUE(ir.outputs().contains("normalTS"));
    EXPECT_FALSE(ir.outputs().contains("coverage"));
}

TEST(MaterialGraph, NontrivialCoverageSelectsMaskAndKeepsSeparateSlice)
{
    auto graph=defaultMaterialGraph(); graph["nodes"][2]["inputs"]["coverage"]={{"node",1}};
    CompiledMaterialFrontend compiled; std::string error;
    ASSERT_TRUE(compileMaterialGraph(graph,compiled,error)) << error;
    const auto ir=render::MaterialValueIR::parse(compiled.definition.surfaceProgram);
    EXPECT_TRUE(ir.outputs().contains("coverage"));
    EXPECT_EQ(ir.slice(true).usage().parameterMask,1u);
    for (const auto& feature : compiled.definition.schema.features) {
        if (feature.name=="alphaMode") { EXPECT_EQ(feature.defaultValue,"mask"); }
    }
}

TEST(MaterialGraph, PositionsNamesIDsAndDefaultsDoNotChangeCodeIdentity)
{
    auto source = defaultMaterialGraph(); CompiledMaterialFrontend a,b; std::string error;
    ASSERT_TRUE(compileMaterialGraph(source,a,error)) << error;
    source["nodes"][0]["position"]={321,654}; source["nodes"][0]["label"]="Albedo";
    source["nodes"][0]["default"]={.8,.1,.1,1.}; source["nodes"][0]["id"]=40;
    source["nodes"][1]["inputs"]["baseColor"]={{"node",40}};
    std::reverse(source["nodes"].begin(),source["nodes"].end());
    ASSERT_TRUE(compileMaterialGraph(source,b,error)) << error;
    EXPECT_EQ(a.signature,b.signature); EXPECT_NE(a.definition.valueParameters,b.definition.valueParameters);
    source["nodes"][2]["slot"]=1;
    ASSERT_TRUE(compileMaterialGraph(source,b,error)) << error;
    EXPECT_NE(a.signature,b.signature);
}

TEST(MaterialGraph, RejectsInvalidPinsCyclesAndDefaultsTransactionally)
{
    CompiledMaterialFrontend good, result; std::string error;
    ASSERT_TRUE(compileMaterialGraph(defaultMaterialGraph(),good,error));
    const auto reject = [&](Json source) {
        result=good;
        EXPECT_FALSE(compileMaterialGraph(source,result,error)); EXPECT_FALSE(error.empty());
        EXPECT_EQ(result.signature,good.signature);
        EXPECT_EQ(result.definition.surfaceProgram,good.definition.surfaceProgram);
    };
    auto source=defaultMaterialGraph(); source["nodes"][1]["inputs"]["baseColor"]={{"node",2}}; reject(source);
    source=defaultMaterialGraph(); source["nodes"][2]["inputs"]["surface"]={{"node",1}}; reject(source);
    source=defaultMaterialGraph(); source["nodes"][1]["inputs"]["baseColor"]={{"node",99}}; reject(source);
    source=defaultMaterialGraph(); source["nodes"][0]["slot"]=4; reject(source);
    source=defaultMaterialGraph(); source["nodes"][0]["default"]={1,2}; reject(source);
    source=defaultMaterialGraph(); source["nodes"][0]["id"]=2; reject(source);
    source=defaultMaterialGraph(); source["version"]=2; reject(source);
    source=defaultMaterialGraph(); source["nodes"][1]["inputs"]["typo"]=0; reject(source);
    source=defaultMaterialGraph(); source["nodes"][0]=makeMaterialGraphNode("Math",1);
    source["nodes"][0]["inputs"]["a"]={{"node",1}}; reject(source);
    source=defaultMaterialGraph(); auto parameter=makeMaterialGraphNode("Parameter",4); parameter["default"]={1,1,1,1};
    source["nodes"].push_back(parameter); source["nodes"][1]["inputs"]["roughness"]={{"node",4}}; reject(source);
}

TEST(MaterialGraph, TextureNormalMathReflectionAndDeadCode)
{
    auto source=defaultMaterialGraph();
    auto texture=makeMaterialGraphNode("Texture",4); texture["texture"]="normal";
    auto normal=makeMaterialGraphNode("NormalMap",5); normal["inputs"]["value"]={{"node",4}};
    source["nodes"].push_back(texture); source["nodes"].push_back(normal);
    source["nodes"][1]["inputs"]["normalTS"]={{"node",5}};
    CompiledMaterialFrontend a,b; std::string error;
    ASSERT_TRUE(compileMaterialGraph(source,a,error)) << error;
    EXPECT_EQ(a.reflection["textureMask"],4); EXPECT_EQ(a.reflection["resources"][0],"normal");
    EXPECT_NE(a.reflection["features"].get<uint32_t>() & render::ValueNormalMapping,0u);
    auto unused=makeMaterialGraphNode("Math",6); unused["inputs"]["a"]={{"node",6}};
    source["nodes"].push_back(unused);
    ASSERT_TRUE(compileMaterialGraph(source,b,error)) << error;
    EXPECT_EQ(a.signature,b.signature);
    ASSERT_TRUE(compileSlangMaterial("Material evaluate() { return withNormal(openPBR(parameter(0),0,.35),normalMap(sampleNormal(uv(),0))); }",
        a.definition.valueParameters,b,error)) << error;
    EXPECT_EQ(a.signature,b.signature);
}

TEST(MaterialGraph, SlabMixLayerShareBackendBudgetsAndSDK)
{
    for (const auto* operation : {"Mix","Layer"}) {
        auto source=defaultMaterialGraph(true);
        auto second=makeMaterialGraphNode("Slab",4); second["inputs"]["reflectance"]=.5;
        auto composite=makeMaterialGraphNode(operation,5);
        composite["inputs"]["a"]={{"node",2}}; composite["inputs"]["b"]={{"node",4}};
        source["nodes"].push_back(second); source["nodes"].push_back(composite);
        source["nodes"][2]["inputs"]["surface"]={{"node",5}};
        CompiledMaterialFrontend graph,text; std::string error;
        ASSERT_TRUE(compileMaterialGraph(source,graph,error)) << error;
        const std::string expr = std::string(operation)=="Mix" ? "mix(slab(parameter(0),0),slab(.5,0),0)" : "layer(slab(parameter(0),0),slab(.5,0))";
        ASSERT_TRUE(compileSlangMaterial("Material evaluate() { return " + expr + "; }",graph.definition.valueParameters,text,error)) << error;
        EXPECT_EQ(graph.signature,text.signature); EXPECT_EQ(graph.definition.implementation,"Slab.Surface");
        source["nodes"][4]["inputs"]["b"]={{"node",5}};
        EXPECT_FALSE(compileMaterialGraph(source,text,error));
    }
    CompiledMaterialFrontend result; std::string error;
    EXPECT_FALSE(compileSlangMaterial("Material evaluate() { return layer(slab(1,0),layer(slab(1,0),slab(1,0))); }",Json::object(),result,error));
}

TEST(MaterialGraph, SDKRejectsSideEffectsUnknownSyntaxAndBudgets)
{
    CompiledMaterialFrontend result; std::string error;
    for (const char* source : {"Material evaluate() { return textureStore(0,1); }", "Material evaluate() { while (1) {} return slab(1,0); }",
        "import Evil; Material evaluate() { return slab(1,0); }", "Material evaluate() { return openPBR(parameter(4),0,1); }",
        "Material evaluate() { return slab(1e999,0); }", "Material evaluate() { return mix(openPBR(1,0,1),slab(1,0),.5); }",
        "Material evaluate() { return slab(1,0); } garbage", "Material evaluate() { let a=slab(1,0); return a+a; }"}) {
        EXPECT_FALSE(compileSlangMaterial(source,Json::object(),result,error)) << source; EXPECT_FALSE(error.empty());
    }
    EXPECT_FALSE(compileSlangMaterial(std::string(17000,' '),Json::object(),result,error));
    EXPECT_TRUE(compileSlangMaterial("/*sdk*/ Material evaluate() { return openPBR(float4(-.1,.2,.3,1),0,.35); }",Json::object(),result,error)) << error;
}

TEST(MaterialGraph, MathPaletteAndPackedTextureChannelsLowerToExistingIR)
{
    for (const auto* op : {"add","multiply","dot","lerp","clamp","select","sin","fract","abs","saturate","normalize"}) {
        auto graph=defaultMaterialGraph(); auto math=makeMaterialGraphNode("Math",4);
        math["operation"]=op; math["inputs"]["a"]={{"node",1}};
        graph["nodes"].push_back(math); graph["nodes"][1]["inputs"]["baseColor"]={{"node",4}};
        CompiledMaterialFrontend compiled; std::string error;
        EXPECT_TRUE(compileMaterialGraph(graph,compiled,error)) << op << ": " << error;
    }
    auto graph=defaultMaterialGraph(); auto texture=makeMaterialGraphNode("Texture",4); texture["texture"]="metallicRoughness";
    auto channel=makeMaterialGraphNode("Swizzle",5); channel["components"]="yyyy"; channel["inputs"]["value"]={{"node",4}};
    graph["nodes"].push_back(texture); graph["nodes"].push_back(channel);
    graph["nodes"][1]["inputs"]["roughness"]={{"node",5}};
    CompiledMaterialFrontend compiled,sdk; std::string error;
    ASSERT_TRUE(compileMaterialGraph(graph,compiled,error)) << error;
    ASSERT_TRUE(compileSlangMaterial("Material evaluate() { return openPBR(parameter(0),0,y(sampleMetallicRoughness(uv(),0))); }",
        compiled.definition.valueParameters,sdk,error)) << error;
    EXPECT_EQ(compiled.signature,sdk.signature);
    EXPECT_TRUE(compileSlangMaterial("Material evaluate() { let a=parameter(0)*2-1; return openPBR(lerp(a,float4(.1,.2,.3,1),.5),0,.35); }",
        Json::object(),sdk,error)) << error;
}

TEST(MaterialGraph, CoverageBaseAlphaAndInvalidStageAccess)
{
    auto graph=defaultMaterialGraph(); graph["nodes"].push_back(makeMaterialGraphNode("BaseAlpha",4));
    graph["nodes"][2]["inputs"]["coverage"]={{"node",4}};
    CompiledMaterialFrontend compiled,sdk; std::string error;
    ASSERT_TRUE(compileMaterialGraph(graph,compiled,error)) << error;
    ASSERT_TRUE(compileSlangMaterial("Material evaluate() { return surface(openPBR(parameter(0),0,.35),0,alpha()); }",
        compiled.definition.valueParameters,sdk,error)) << error;
    EXPECT_EQ(compiled.signature,sdk.signature);
    graph["nodes"][1]["inputs"]["baseColor"]={{"node",4}};
    EXPECT_FALSE(compileMaterialGraph(graph,compiled,error));
    EXPECT_NE(error.find("Coverage-only"),std::string::npos);
    EXPECT_FALSE(compileSlangMaterial("Material evaluate() { return surface(slab(1,0),0,sampleBaseColor(uv(),0)); }",
        Json::object(),compiled,error));
}

TEST(MaterialGraph, DefinitionInstanceSceneSaveReloadClosesAuthoringLoop)
{
    CompiledMaterialFrontend compiled; std::string error;
    ASSERT_TRUE(compileMaterialGraph(defaultMaterialGraph(),compiled,error)) << error;
    const auto root=std::filesystem::path(PROJECT_SOURCE_DIR)/"build/material-graph-cpu";
    std::filesystem::create_directories(root);
    { std::ofstream file(root/"Graph.materialdef"); file<<serializeMaterialDefinition(compiled.definition); }
    MaterialInstance instance; instance.definition="asset://Graph.materialdef";
    instance.valueParameters={{"0",{.2,.4,.7,1.}}};
    MaterialAssetLibrary library(root);
    ASSERT_TRUE(library.save("asset://Graph.material",instance,error)) << error;
    const auto source=std::filesystem::path(PROJECT_SOURCE_DIR)/"Asset/LookDev/OpenPBRDefault";
    std::filesystem::copy_file(source/"OpenPbrDefault.gltf",root/"Scene.gltf",std::filesystem::copy_options::overwrite_existing);
    std::filesystem::copy_file(source/"Shaderball.bin",root/"Shaderball.bin",std::filesystem::copy_options::overwrite_existing);
    std::filesystem::remove(root/"Scene.metallic_scene.json");
    scene::SceneDocument document; ASSERT_TRUE(document.load(root/"Scene.gltf"));
    ASSERT_TRUE(document.setMaterialAsset(0,"asset://Graph.material",root,error)) << error;
    EXPECT_EQ(document.materials()[0].valueProgram,compiled.definition.surfaceProgram);
    EXPECT_FLOAT_EQ(document.materials()[0].valueParameters[1],.4f);
    ASSERT_TRUE(document.save(error)) << error;
    scene::SceneDocument reloaded; ASSERT_TRUE(reloaded.load(document.documentPath()));
    EXPECT_TRUE(reloaded.documentWarning().empty()) << reloaded.documentWarning();
    EXPECT_EQ(reloaded.materials()[0].valueProgram,compiled.definition.surfaceProgram);
    EXPECT_EQ(reloaded.materials()[0].valueParameters,document.materials()[0].valueParameters);
}
} // namespace
} // namespace metallic::tests
