#include "RHITest.h"
#include "Runtime/Material/MaterialGraph.h"
#include "Runtime/Render/Material/MaterialValueProgram.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Scene/SceneDocument.h"
#include <cmath>
#include <cstring>
#include <fstream>

namespace metallic::tests {
namespace {
using namespace render;
using Json = nlohmann::json;
class MaterialGraphSceneTest final : public RHITest
{
public:
    MaterialGraphSceneTest() { name="material_graph_scene"; type=RHITestType::Rendering; }
    RHITestResult run(RHITestContext& context) override
    {
        try {
            std::string log;
            const auto check=[&](const auto& condition,const std::string& message) { if (!condition) { throw std::runtime_error(message+": "+log); } };
            const auto root=std::filesystem::absolute(context.outputDirectory);
            std::filesystem::create_directories(root);
            const auto source=std::filesystem::path(PROJECT_SOURCE_DIR)/"Asset/LookDev/OpenPBRDefault";
            std::filesystem::copy_file(source/"OpenPbrDefault.gltf",root/"GraphScene.gltf",std::filesystem::copy_options::overwrite_existing);
            std::filesystem::copy_file(source/"Shaderball.bin",root/"Shaderball.bin",std::filesystem::copy_options::overwrite_existing);
            std::filesystem::remove(root/"GraphScene.metallic_scene.json");
            const std::array<uint8_t,4> gray{128,166,199,255}, normal{166,153,246,255};
            check(saveRgba8Png(root/"Color.png",gray.data(),1,1,log),"Write color texture");
            check(saveRgba8Png(root/"Normal.png",normal.data(),1,1,log),"Write normal texture");
            material::MaterialAssetLibrary library(root);
            const auto publish=[&](const char* name,const material::MaterialDefinition& definition) {
                { std::ofstream file(root/(std::string(name)+".materialdef")); file<<material::serializeMaterialDefinition(definition); }
                material::MaterialInstance instance; instance.definition="asset://"+std::string(name)+".materialdef";
                instance.parameters["baseColor"]={1.,1.,1.}; instance.parameters["metalness"]=0.; instance.parameters["roughness"]=.35;
                instance.resources["baseColorTexture"]="asset://Color.png";
                instance.resources["normalTexture"]="asset://Normal.png";
                check(library.save("asset://"+std::string(name)+".material",instance,log),"Publish asset");
            };
            publish("Legacy",material::defaultOpenPBRDefinition());
            auto graph=material::defaultMaterialGraph();
            graph["nodes"][0]=material::makeMaterialGraphNode("Texture",1);
            auto normalTexture=material::makeMaterialGraphNode("Texture",4); normalTexture["texture"]="normal";
            auto normalMap=material::makeMaterialGraphNode("NormalMap",5); normalMap["inputs"]["value"]={{"node",4}};
            graph["nodes"].push_back(normalTexture); graph["nodes"].push_back(normalMap);
            graph["nodes"][1]["inputs"]["normalTS"]={{"node",5}};
            material::CompiledMaterialFrontend compiled,sdk;
            check(material::compileMaterialGraph(graph,compiled,log),"Compile textured graph");
            check(material::compileSlangMaterial("Material evaluate() { return withNormal(openPBR(sampleBaseColor(uv(),0),0,.35),normalMap(sampleNormal(uv(),0))); }",
                Json::object(),sdk,log),"Compile matching SDK");
            check(compiled.signature==sdk.signature,"Graph/SDK canonical mismatch");
            publish("Graph",compiled.definition); publish("SDK",sdk.definition);
            scene::SceneDocument document; check(document.load(root/"GraphScene.gltf"),"Load scene");
            check(document.setMaterialAsset(0,"asset://Legacy.material",root,log),"Bind reference asset");
            RenderSampleLoadResult sample; check(loadBuiltInRenderSample("lookdev-vbuffer",sample,log),"Load sample");
            for (const char* pass : {"Reference","Deferred","VBuffer"}) { sample.graph.findNode(pass)->properties["path"]=(root/"GraphScene.gltf").string(); }
            for (const char* pass : {"Reference","Deferred"}) {
                auto& props=sample.graph.findNode(pass)->properties;
                props["samples"]=1; props["maxDepth"]=1; props["accumulate"]=false; props["outputLinear"]=true; props["debugDisableShadows"]=true;
            }
            scene::EnvironmentSettings environment; environment.enabled=false;
            scene::LightingSettings lighting; lighting.autoExposure.enabled=false;
            environment::WorldEnvironment celestial;
            auto& sun=celestial.sun; sun.enabled=true; sun.illuminance=2;
            sun.direction=float3(-.4f,-.6f,-1);
            const auto capture=[&](const char* output,const std::string& label) {
                // Restart replays frame zero, including camera/path sample seeds.
                RenderGraphPreviewRenderer preview;
                preview.bindRuntimeScene(&document); preview.setWorldEnvironment(celestial); preview.setEnvironment(environment); preview.setLighting(lighting); preview.setRawReadbackEnabled(true);
                check(preview.initialize(context.enableValidation,true,false),"Initialize preview");
                check(preview.render(sample.graph,128,128,output),label+": "+preview.lastLog());
                check(preview.readbackFormat()==Format::RGBA32Sfloat && preview.readbackBytes().size()==128*128*16,"HDR readback format");
                const auto& bytes=preview.readbackBytes(); std::vector<float> pixels(bytes.size()/4);
                std::memcpy(pixels.data(),bytes.data(),bytes.size());
                double energy=0; for (size_t i=0;i<pixels.size();++i) { check(std::isfinite(pixels[i]),"Nonfinite pixel"); if (i%4!=3) { energy+=std::abs(pixels[i]); } }
                check(energy>1,"Empty image");
                check(saveRgba8Png(root/(label+".png"),reinterpret_cast<const uint8_t*>(preview.pixels().data()),128,128,log),"Save image");
                return pixels;
            };
            const auto difference=[](const auto& a,const auto& b) {
                double sum=0,energy=0; for (size_t i=0;i<a.size();++i) { if (i%4!=3) { sum+=std::abs(a[i]-b[i]); energy+=std::abs(a[i]); } }
                return sum/std::max(energy,1e-20);
            };
            const auto legacy=capture("Deferred.color","legacy-textured");
            const auto legacyPT=capture("Reference.color","legacy-textured-pt");
            check(document.setMaterialAsset(0,"asset://Graph.material",root,log),"Bind graph");
            const auto graphPixels=capture("Deferred.color","graph-textured");
            const auto graphPT=capture("Reference.color","graph-textured-pt");
            const auto error=difference(legacy,graphPixels), errorPT=difference(legacyPT,graphPT);
            check(error<1e-5,"Graph texture/normal differs from legacy oracle: "+std::to_string(error));
            check(errorPT<1e-5,"PT graph texture/normal differs from legacy oracle: "+std::to_string(errorPT));
            check(document.setMaterialAsset(0,"asset://SDK.material",root,log),"Bind SDK");
            const auto sdkPixels=capture("Deferred.color","sdk-textured");
            check(difference(graphPixels,sdkPixels)<1e-6,"Graph and SDK scene output mismatch");
            sample.graph.findNode("Deferred")->properties["materialBinning"]=false; sample.graph.markDirty();
            const auto unbinned=capture("Deferred.color","graph-unbinned");
            check(difference(graphPixels,unbinned)<1e-5,"Graph program binning changes lighting");
            // Parameter edits must change runtime bytes/output while retaining executable identity.
            check(material::compileMaterialGraph(material::defaultMaterialGraph(),compiled,log),"Compile parameter graph");
            auto material=document.materials()[0]; material.valueProgram=compiled.definition.surfaceProgram;
            material.valueParameters.fill(.18f); check(document.setMaterialProperties(0,material),"Apply parameter graph");
            const auto first=MaterialValueProgramSet::create(document.materials(),log); check(bool(first),"Build graph program set");
            const auto firstImage=capture("Deferred.color","graph-parameter-dark");
            material.valueParameters[0]=.8f; material.valueParameters[1]=.4f;
            check(document.setMaterialProperties(0,material),"Update instance");
            const auto second=MaterialValueProgramSet::create(document.materials(),log);
            check(second && first->key()==second->key() && first->source()==second->source(),"Parameter edit changed executable");
            check(first->instances()[0].parameters!=second->instances()[0].parameters,"Parameter bytes unchanged");
            const auto secondImage=capture("Deferred.color","graph-parameter-color");
            check(difference(firstImage,secondImage)>.05,"Instance update did not reach lighting");
            std::ofstream(root/"MaterialGraphAcceptance.json")<<Json{{"textureNormalRelativeError",error},{"ptTextureNormalRelativeError",errorPT},
                {"sdkRelativeError",difference(graphPixels,sdkPixels)},{"binRelativeError",difference(graphPixels,unbinned)},
                {"parameterImageChange",difference(firstImage,secondImage)}}.dump(2);
            return RHITestResult::pass("Graph/SDK asset execution, texture single application, authored TBN normal oracle in PT+Deferred, bin equivalence and dynamic instance updates");
        } catch (const std::exception& error) { return RHITestResult::fail(error.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(MaterialGraphSceneTest);
} // namespace
} // namespace metallic::tests
