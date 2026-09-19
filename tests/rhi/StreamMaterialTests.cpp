#include "RhiTest.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Scene/SceneDocument.h"
#include "Runtime/Scene/MeshletStreamAsset.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <map>
#include <stdexcept>

namespace metallic::tests {
namespace {
using namespace render;
using Json = nlohmann::json;
void require(bool value, const std::string& message)
{
    if (!value) { throw std::runtime_error(message); }
}

std::filesystem::path materialFixture(const std::filesystem::path& directory)
{
    // Oblique authored normals/tangents distinguish linear from normal transforms.
    const float vertices[] = {
        -.45f,-.7f,0, .45f,-.7f,0, .45f,.7f,0, -.45f,.7f,0,
        .3f,.4f,.8660254f, .3f,.4f,.8660254f, .3f,.4f,.8660254f, .3f,.4f,.8660254f,
        0,0, 2,0, 2,2, 0,2,
        .8f,-.6f,0,1, .8f,-.6f,0,1, .8f,-.6f,0,1, .8f,-.6f,0,1};
    const uint16_t indices[] = {0,1,2,0,2,3};
    { std::ofstream out(directory / "Z4.bin", std::ios::binary);
        out.write(reinterpret_cast<const char*>(vertices), sizeof(vertices));
        out.write(reinterpret_cast<const char*>(indices), sizeof(indices)); }
    std::array<uint8_t, 16 * 16 * 4> pixels{};
    for (uint32_t y = 0; y < 16; ++y) {
        for (uint32_t x = 0; x < 16; ++x) {
            const uint32_t i = (y * 16 + x) * 4;
            pixels[i] = uint8_t(40 + x * 12); pixels[i + 1] = uint8_t(40 + y * 12); pixels[i + 2] = 160;
            pixels[i + 3] = ((x / 4 + y / 4) & 1) ? 255 : 0;
        }
    }
    std::string log;
    require(saveRgba8Png(directory / "Z4Color.png", pixels.data(), 16, 16, log), log);
    for (size_t i = 0; i < pixels.size(); i += 4) { pixels[i] = 170; pixels[i + 1] = 190; pixels[i + 2] = 238; pixels[i + 3] = 255; }
    require(saveRgba8Png(directory / "Z4Normal.png", pixels.data(), 16, 16, log), log);
    Json materials = Json::array();
    for (uint32_t i = 0; i < 260; ++i) { materials.push_back({{"name", "unused-" + std::to_string(i)}}); }
    Json texture = {{"index", 0}, {"extensions", {{"KHR_texture_transform", {{"offset", {.2,.1}}, {"scale", {.7,1.1}}, {"rotation", .3}}}}}};
    materials[257] = {{"name", "textured-257"}, {"doubleSided", true},
        {"pbrMetallicRoughness", {{"baseColorTexture", texture}, {"metallicFactor", .3}, {"roughnessFactor", .65}}},
        {"normalTexture", {{"index", 1}, {"scale", .65}}},
        {"extensions", {{"KHR_materials_specular", {{"specularFactor", .4}, {"specularColorFactor", {.2,.6,.9}},
            {"specularTexture", {{"index", 0}}}, {"specularColorTexture", texture}}}}}};
    materials[258] = materials[257]; materials[258]["alphaMode"] = "MASK"; materials[258]["alphaCutoff"] = .5;
    materials[259] = materials[257]; materials[259]["extensions"] = {{"KHR_materials_unlit", Json::object()}};
    Json meshes = Json::array();
    for (int material : {257,258,259}) { meshes.push_back({{"primitives", {{{"attributes", {{"POSITION",0},{"NORMAL",1},{"TEXCOORD_0",2},{"TANGENT",3}}}, {"indices",4}, {"material",material}}}}}); }
    Json gltf = {{"asset", {{"version", "2.0"}}}, {"scene", 0}, {"scenes", {{{"nodes", {0,1,2}}}}},
        {"nodes", {{{"mesh",0},{"translation",{-1.15,0,0}},{"scale",{1.1,.8,1.5}}},
            {{"mesh",1}}, {{"mesh",2},{"translation",{1.15,0,0}},{"scale",{-1,.9,1.2}}}}},
        {"buffers", {{{"uri","Z4.bin"},{"byteLength",sizeof(vertices)+sizeof(indices)}}}},
        {"bufferViews", {{{"buffer",0},{"byteOffset",0},{"byteLength",48}}, {{"buffer",0},{"byteOffset",48},{"byteLength",48}},
            {{"buffer",0},{"byteOffset",96},{"byteLength",32}}, {{"buffer",0},{"byteOffset",128},{"byteLength",64}},
            {{"buffer",0},{"byteOffset",192},{"byteLength",12}}}},
        {"accessors", {{{"bufferView",0},{"componentType",5126},{"count",4},{"type","VEC3"},{"min",{-.45,-.7,0}},{"max",{.45,.7,0}}},
            {{"bufferView",1},{"componentType",5126},{"count",4},{"type","VEC3"}},
            {{"bufferView",2},{"componentType",5126},{"count",4},{"type","VEC2"}},
            {{"bufferView",3},{"componentType",5126},{"count",4},{"type","VEC4"}},
            {{"bufferView",4},{"componentType",5123},{"count",6},{"type","SCALAR"}}}},
        {"meshes",meshes}, {"materials",materials}, {"images", {{{"uri","Z4Color.png"}},{{"uri","Z4Normal.png"}}}},
        {"textures", {{{"source",0}},{{"source",1}}}}, {"extensionsUsed", {"KHR_texture_transform","KHR_materials_specular","KHR_materials_unlit"}}};
    const auto path = directory / "Z4.gltf";
    { std::ofstream out(path); out << gltf.dump(2); }
    return path;
}

RenderGraph materialGraph(const std::filesystem::path& path, const std::filesystem::path& cache, bool streamed)
{
    RenderGraph graph;
    Json camera = {{"eye", {0,0,3}}, {"center",{0,0,0}}, {"up",{0,1,0}}, {"fovDegrees",60}, {"znear",.001}, {"zfar",20}, {"reversedZ",true}};
    graph.addNode("VisibilityBufferPass", "Raster", {{"path",path.generic_string()}, {"streamAssetPath",cache.generic_string()},
        {"enableMeshletStreaming",streamed}, {"streamAssetOnly",streamed}, {"autoBuildStreamAsset",false},
        {"maxResidentPages",32}, {"maxLockedFallbackPages",32}, {"maxActiveGroups",128}, {"maxGpuPageRequests",128},
        {"maxTraversalWorkers",32}, {"maxTraversalWorkItems",256}, {"autoLod",false}, {"lodLevel",0},
        {"instanceHzbCull",false}, {"meshletHzbCull",false}, {"meshletNormalConeCull",false}, {"hybridRaster",false},
        {"camera",camera}});
    graph.addNode("VisibilityBufferDeferredPass", "Deferred", {{"lightingMode","realtime"}, {"debugView","baseColor"},
        {"materialBinning",false}, {"shadow.enabled",false}, {"accumulate",false}});
    graph.addEdge("Raster.visibility", "Deferred.visibility");
    graph.addEdge("Raster.depth", "Deferred.depth");
    graph.addEdge("Raster.rasterInfo", "Deferred.rasterInfo");
    graph.markOutput("Deferred.color");
    return graph;
}

class StreamMaterialShadingTest final : public RhiTest {
public:
    StreamMaterialShadingTest() { type = RhiTestType::Rendering; name = "stream_material_shading"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        try {
            const auto directory = std::filesystem::absolute(context.outputDirectory);
            const auto path = materialFixture(directory);
            const auto cache = directory / "Z4.meshstream.bin";
            std::string log;
            require(scene::buildMeshletStreamAssetOffline({.sourcePath=path, .outputPath=cache}, log), log);
            std::map<std::string, std::vector<uint32_t>> reference;
            Json report = Json::array();
            for (bool streamed : {false,true}) {
                scene::SceneDocument scene;
                require(streamed ? scene.loadStreamMetadata(path) : scene.load(path), scene.lastLoadResult().error);
                require(scene.materials()[257].specularTexture.textureIndex == 0 && scene.materials()[259].unlit,
                    "Specular/unlit import lost material data");
                RenderGraphPreviewRenderer preview;
                preview.bindRuntimeScene(&scene);
                require(bool(preview.initialize(context.enableValidation, true, false)), preview.lastLog());
                preview.setEnvironment({.enabled=false});
                auto lighting = scene.lighting(); lighting.autoExposure.enabled=false; lighting.exposureEV100=0;
                scene::PunctualLight light;
                light.properties.type = "directional";
                light.properties.intensity = 3.0;
                light.direction = float3(-.2f, -.3f, -1.0f);
                lighting.lights = {light};
                preview.setLighting(lighting);
                auto graph = materialGraph(path, cache, streamed);
                auto render = [&]() { require(bool(preview.render(graph, 256,128,"Deferred.color")), preview.lastLog()); };
                for (const char* debug : {"baseColor","uv","shadingNormal","mappedNormal","tangent","bitangent","normalTexture","final"}) {
                    graph.setNodeRuntimeProperty(graph.findNode("Deferred")->id, "debugView",debug);
                    for (uint32_t frame=0; frame<(streamed ? 8u : 1u); ++frame) { render(); }
                    const std::string label = std::string(streamed ? "stream-" : "resident-") + debug;
                    require(saveRgba8Png(directory/(label+".png"), reinterpret_cast<const uint8_t*>(preview.pixels().data()),256,128,log),log);
                    if (!streamed) { reference[debug]=preview.pixels(); continue; }
                    const auto& expected=reference.at(debug); double error=0; uint32_t pixels=0,outliers=0;
                    for (uint32_t y=1;y<127;++y) { for (uint32_t x=1;x<255;++x) {
                        const size_t i=y*256+x;
                        if (!(expected[i]&0xffffff) || !(expected[i-1]&0xffffff) || !(expected[i+1]&0xffffff) ||
                            !(expected[i-256]&0xffffff) || !(expected[i+256]&0xffffff)) { continue; }
                        ++pixels; uint32_t maximum=0;
                        for (uint32_t shift : {0u,8u,16u}) { auto delta=uint32_t(std::abs(int((expected[i]>>shift)&255)-int((preview.pixels()[i]>>shift)&255))); error+=delta; maximum=std::max(maximum,delta); }
                        outliers+=maximum>3;
                    }}
                    report.push_back({{"view",debug},{"pixels",pixels},{"meanError",error/std::max(1u,pixels*3)},{"outliers",outliers}});
                    { std::ofstream out(directory/"Z4Comparison.json"); out<<report.dump(2); }
                    require(pixels>300 && error/(pixels*3)<.7 && double(outliers)/pixels<.03, std::string(debug)+" resident/stream mismatch: "+report.back().dump());
                    const auto hardware=preview.pixels();
                    graph.setNodeRuntimeProperty(graph.findNode("Raster")->id,"hybridRaster",true);
                    graph.setNodeRuntimeProperty(graph.findNode("Raster")->id,"softwareRasterMaxPixels",1024.f);
                    render();
                    uint32_t coverageMismatch=0;
                    for(size_t i=0;i<hardware.size();++i) { coverageMismatch+=((hardware[i]&0xffffff)==0)!=((preview.pixels()[i]&0xffffff)==0); }
                    require(coverageMismatch<12, "HW/hybrid alpha coverage mismatch");
                    graph.setNodeRuntimeProperty(graph.findNode("Raster")->id,"hybridRaster",false);
                }
            }
            return RhiTestResult::pass("Stream attributes, transformed textures, mirrored TBN, MASK and material ID 257 match resident shading");
        } catch(const std::exception& error) { return RhiTestResult::fail(error.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(StreamMaterialShadingTest);
} // namespace
} // namespace metallic::tests
