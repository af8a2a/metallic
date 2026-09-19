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
#include <cstdlib>
#include <cstring>
#include "meshoptimizer.h"

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
    std::filesystem::create_directories(directory);
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
    materials[259]["doubleSided"] = false; // Negative scale must retain front-face coverage.
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
    graph.addNode("FinalBlitPass", "Output");
    graph.addEdge("Deferred.color", "Output.source");
    graph.markOutput("Output.color");
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
                require(scene.materials()[258].alphaMode == "MASK" && scene.renderNodes()[1].materialIndex == 258,
                    "Masked fixture material assignment was lost");
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
                auto render = [&]() { require(bool(preview.render(graph, 256,128)), preview.lastLog()); };
                for (const char* debug : {"baseColor","uv","shadingNormal","mappedNormal","tangent","bitangent","normalTexture","final"}) {
                    graph.setNodeRuntimeProperty(graph.findNode("Deferred")->id, "debugView",debug);
                    for (uint32_t frame=0; frame<(streamed ? 8u : 1u); ++frame) { render(); }
                    const std::string label = std::string(streamed ? "stream-" : "resident-") + debug;
                    require(saveRgba8Png(directory/(label+".png"), reinterpret_cast<const uint8_t*>(preview.pixels().data()),256,128,log),log);
                    if (std::string_view(debug) == "mappedNormal") {
                        uint32_t holes = 0, surfaces = 0;
                        for (uint32_t y = 43; y < 85; ++y) { for (uint32_t x = 114; x < 142; ++x) {
                            if ((preview.pixels()[y * 256 + x] & 0xffffffu) == 0) { ++holes; } else { ++surfaces; }
                        }}
                        require(holes > 300 && surfaces > 300, label + " alpha mask has no holes: " + std::to_string(holes));
                    }
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

class StreamMaterialTransmissionTest final : public RhiTest {
public:
    StreamMaterialTransmissionTest() { type = RhiTestType::Rendering; name = "stream_material_transmission"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        try {
            const auto directory = std::filesystem::absolute(context.outputDirectory / "transmission");
            std::filesystem::create_directories(directory);
            const auto path = materialFixture(directory);
            Json root;
            { std::ifstream input(path); input >> root; }
            root["materials"][257] = {{"doubleSided",true}, {"extensions",{{"KHR_materials_unlit",Json::object()}}},
                {"pbrMetallicRoughness",{{"baseColorFactor",{.05,.1,.8,1}},{"metallicFactor",0}}}};
            root["materials"][258] = {{"doubleSided",true}, {"alphaMode","BLEND"},
                {"extensions",{{"KHR_materials_unlit",Json::object()}}},
                {"pbrMetallicRoughness",{{"baseColorFactor",{.8,.05,.05,.5}},{"metallicFactor",0}}}};
            root["materials"][259] = {{"doubleSided",true},
                {"pbrMetallicRoughness",{{"baseColorFactor",{1,1,1,1}},{"metallicFactor",0},{"roughnessFactor",0}}},
                {"extensions",{{"KHR_materials_transmission",{{"transmissionFactor",1}}},{"KHR_materials_ior",{{"ior",1}}}}}};
            root["nodes"][0] = {{"mesh",0},{"translation",{0,0,-.4}},{"scale",{4,2,1}}};
            root["nodes"][1] = {{"mesh",1},{"translation",{-.55,0,0}}};
            root["nodes"][2] = {{"mesh",2},{"translation",{.55,0,0}}};
            // A cutout behind glass is reached only through the CLAS ray path.
            // Both red texels and the blue background must survive the query.
            root["materials"].push_back({{"doubleSided",true},{"alphaMode","MASK"},{"alphaCutoff",.5},
                {"extensions",{{"KHR_materials_unlit",Json::object()}}},
                {"pbrMetallicRoughness",{{"baseColorFactor",{1,.05,.05,1}},
                    {"baseColorTexture",{{"index",0}}},{"metallicFactor",0}}}});
            root["meshes"].push_back(root["meshes"][1]);
            root["meshes"][3]["primitives"][0]["material"] = 260;
            root["nodes"].push_back({{"mesh",3},{"translation",{.55,0,-.2}}});
            root["scenes"][0]["nodes"].push_back(3);
            { std::ofstream output(path); output << root.dump(2); }
            const auto cache = directory / "Z4.meshstream.bin";
            std::string log;
            require(scene::buildMeshletStreamAssetOffline({.sourcePath=path,.outputPath=cache},log),log);
            scene::SceneDocument scene;
            require(scene.loadStreamMetadata(path),scene.lastLoadResult().error);
            RenderGraphPreviewRenderer preview;
            preview.bindRuntimeScene(&scene);
            require(bool(preview.initialize(context.enableValidation,true,false)),preview.lastLog());
            preview.setEnvironment({.enabled=false});
            auto lighting=scene.lighting(); lighting.lights.clear(); lighting.autoExposure.enabled=false;
            preview.setLighting(lighting);
            auto graph=materialGraph(path,cache,true);
            const auto raster=graph.findNode("Raster")->id, deferred=graph.findNode("Deferred")->id;
            graph.setNodeRuntimeProperty(raster,"enableClas",true);
            graph.setNodeRuntimeProperty(raster,"enableClusterRtx",true);
            graph.setNodeRuntimeProperty(raster,"maxClasBytes",16777216);
            graph.setNodeRuntimeProperty(deferred,"debugView","final");
            graph.setNodeRuntimeProperty(deferred,"transmissionSamples",16);
            graph.setNodeRuntimeProperty(deferred,"transmissionDepth",8);
            for(uint32_t frame=0;frame<8;++frame) { require(bool(preview.render(graph,256,128)),preview.lastLog()); }
            require(saveRgba8Png(directory/"BlendAndGlass.png",reinterpret_cast<const uint8_t*>(preview.pixels().data()),256,128,log),log);
            uint32_t blendRed=0, blendBlue=0, glassBlue=0, samples=0, cutoutRed=0, cutoutBlue=0;
            for(uint32_t y=46;y<82;++y) { for(uint32_t x=100;x<115;++x) {
                const uint32_t blend=preview.pixels()[y*256+x],glass=preview.pixels()[y*256+(256-x)];
                blendRed+=blend&255u; blendBlue+=(blend>>16)&255u; glassBlue+=(glass>>16)&255u; ++samples;
                const uint32_t red=glass&255u, blue=(glass>>16)&255u;
                cutoutRed+=red>blue+40; cutoutBlue+=blue>red+40;
            }}
            require(blendRed>samples*70 && blendBlue>samples*70,"BLEND did not retain both front and background radiance");
            require(glassBlue>samples*70 && cutoutRed>100 && cutoutBlue>100,
                "Stream CLAS glass/MASK continuation lost cutout or background coverage: " +
                std::to_string(cutoutRed) + "/" + std::to_string(cutoutBlue));
            const auto unbinned=preview.pixels();
            graph.setNodeRuntimeProperty(deferred,"materialBinning",true);
            for(uint32_t frame=0;frame<8;++frame) { require(bool(preview.render(graph,256,128)),preview.lastLog()); }
            uint32_t changed=0;
            for(size_t i=0;i<unbinned.size();++i) { changed+=unbinned[i]!=preview.pixels()[i]; }
            require(changed<16,"Material binning changed BLEND/glass continuation");
            return RhiTestResult::pass("BLEND composites background; IOR=1 glass continues through streamed CLAS with MASK holes and material ID 260");
        } catch(const std::exception& error) { return RhiTestResult::fail(error.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(StreamMaterialTransmissionTest);

class StreamMaterialShadowTest final : public RhiTest {
public:
    StreamMaterialShadowTest() { type=RhiTestType::Rendering; name="stream_material_shadow"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        try {
            const auto directory=std::filesystem::absolute(context.outputDirectory/"shadow");
            std::filesystem::create_directories(directory);
            const auto path=materialFixture(directory);
            Json root;
            { std::ifstream input(path); input>>root; }
            root["materials"][257]={{"doubleSided",true},
                {"pbrMetallicRoughness",{{"baseColorFactor",{.8,.8,.8,1}},{"metallicFactor",0}}}};
            root["nodes"]={{{"mesh",0},{"translation",{0,0,-.6}},{"scale",{5,2,1}}},{{"mesh",1}}};
            root["scenes"][0]["nodes"]={0,1};
            { std::ofstream output(path); output<<root.dump(2); }
            auto cache=directory/"Z4.meshstream.bin";
            std::string log;
            require(scene::buildMeshletStreamAssetOffline({.sourcePath=path,.outputPath=cache},log),log);
            std::vector<uint32_t> reference;
            for(bool streamed : {false,true}) {
                scene::SceneDocument scene;
                require(streamed ? scene.loadStreamMetadata(path) : scene.load(path),scene.lastLoadResult().error);
                RenderGraphPreviewRenderer preview;
                preview.bindRuntimeScene(&scene);
                require(bool(preview.initialize(context.enableValidation,true,false)),preview.lastLog());
                preview.setEnvironment({.enabled=false});
                auto lighting=scene.lighting(); lighting.autoExposure.enabled=false;
                scene::PunctualLight sun; sun.properties.type="directional"; sun.properties.intensity=3;
                sun.direction=float3(2,0,-1); lighting.lights={sun}; preview.setLighting(lighting);
                auto graph=materialGraph(path,cache,streamed);
                const auto raster=graph.findNode("Raster")->id, deferred=graph.findNode("Deferred")->id;
                graph.setNodeRuntimeProperty(raster,"enableClas",streamed);
                graph.setNodeRuntimeProperty(raster,"enableClusterRtx",streamed);
                graph.setNodeRuntimeProperty(raster,"maxClasBytes",16777216);
                graph.setNodeRuntimeProperty(deferred,"debugView","final");
                graph.addNode("RayTracedShadowPass","Shadows",{{"rayTracedShadows",true},{"sigmaDenoise",false},
                    {"shadowAngularRadius",0},{"shadowBias",.001},{"shadowDebug",true}});
                graph.addEdge("Raster.depth","Shadows.depth");
                graph.addEdge("Raster.rasterInfo","Shadows.rasterInfo");
                graph.addEdge("Shadows.shadow","Deferred.shadow");
                graph.addEdge("Shadows.parameters","Deferred.shadowParameters");
                for(uint32_t frame=0;frame<8;++frame) { require(bool(preview.render(graph,256,128)),preview.lastLog()); }
                const auto& pixels=preview.pixels();
                require(saveRgba8Png(directory/(streamed ? "Stream.png" : "Resident.png"),
                    reinterpret_cast<const uint8_t*>(pixels.data()),256,128,log),log);
                if(!streamed) { reference=pixels; continue; }
                uint32_t dark=0,lit=0,mismatch=0;
                // Projected cutout shadow lies to the right of the visible leaf.
                for(uint32_t y=43;y<85;++y) { for(uint32_t x=157;x<180;++x) {
                    const size_t i=y*256+x;
                    const uint32_t a=reference[i]&255u,b=pixels[i]&255u;
                    dark+=b<20; lit+=b>150; mismatch+=std::abs(int(a)-int(b))>8;
                }}
                require(dark>100 && lit>100 && mismatch<20,
                    "MASK shadow mismatch: dark="+std::to_string(dark)+", lit="+std::to_string(lit)+", mismatch="+std::to_string(mismatch));
            }
            return RhiTestResult::pass("Stream CLAS MASK shadow holes match resident ray-query coverage");
        } catch(const std::exception& error) { return RhiTestResult::fail(error.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(StreamMaterialShadowTest);

std::filesystem::path residentProbe(const std::filesystem::path& source, const std::filesystem::path& directory)
{
    Json root;
    { std::ifstream input(source); input >> root; }
    std::vector<uint8_t> decoded;
    for (auto& view : root["bufferViews"]) {
        const auto extension = view.value("extensions", Json::object()).value("EXT_meshopt_compression", Json());
        const auto& stored = extension.is_null() ? view : extension;
        const auto path = source.parent_path() / root["buffers"][stored["buffer"].get<size_t>()]["uri"].get<std::string>();
        std::ifstream input(path,std::ios::binary);
        std::vector<uint8_t> bytes(stored["byteLength"].get<size_t>());
        input.seekg(stored.value("byteOffset",uint64_t(0)));
        require(bool(input.read(reinterpret_cast<char*>(bytes.data()),bytes.size())),"Cannot read probe view");
        decoded.resize((decoded.size()+15)&~size_t(15));
        const size_t offset=decoded.size(), size=view["byteLength"].get<size_t>();
        decoded.resize(offset+size);
        auto* destination=decoded.data()+offset;
        if (extension.is_null()) { require(bytes.size()==size,"Raw probe view size"); std::memcpy(destination,bytes.data(),size); }
        else {
            const size_t count=extension["count"],stride=extension["byteStride"];
            require(count*stride==size,"Compressed probe view size");
            const std::string mode=extension["mode"],filter=extension.value("filter","NONE");
            int result=-1;
            if(mode=="ATTRIBUTES") { result=meshopt_decodeVertexBuffer(destination,count,stride,bytes.data(),bytes.size()); }
            if(mode=="TRIANGLES") { result=meshopt_decodeIndexBuffer(destination,count,stride,bytes.data(),bytes.size()); }
            if(mode=="INDICES") { result=meshopt_decodeIndexSequence(destination,count,stride,bytes.data(),bytes.size()); }
            require(result==0,"Probe meshopt decode failed");
            if(filter=="OCTAHEDRAL") { meshopt_decodeFilterOct(destination,count,stride); }
            else if(filter=="QUATERNION") { meshopt_decodeFilterQuat(destination,count,stride); }
            else if(filter=="EXPONENTIAL") { meshopt_decodeFilterExp(destination,count,stride); }
            else { require(filter=="NONE","Unknown meshopt filter"); }
        }
        view["buffer"]=0; view["byteOffset"]=offset; view.erase("extensions");
    }
    root["buffers"]={{{"uri","Reference.bin"},{"byteLength",decoded.size()}}};
    for(const char* field : {"extensionsUsed","extensionsRequired"}) {
        if(root.contains(field)) { auto& list=root[field]; list.erase(std::remove(list.begin(),list.end(),Json("EXT_meshopt_compression")),list.end()); }
    }
    // Legacy resident VBuffer omits BLEND draws. For attribute-only comparisons,
    // preserve positive-alpha coverage using MASK; this is not a transport reference.
    for(auto& material : root["materials"]) {
        if(material.value("alphaMode","")=="BLEND") {
            material["alphaMode"]="MASK";
            material["alphaCutoff"]=0.000001;
        }
    }
    { std::ofstream output(directory/"Reference.bin",std::ios::binary); output.write(reinterpret_cast<const char*>(decoded.data()),decoded.size()); }
    const auto path=directory/"Reference.gltf";
    { std::ofstream output(path); output<<root.dump(); }
    return path;
}

class ZorahStreamMaterialProbesTest final : public RhiTest {
public:
    ZorahStreamMaterialProbesTest() { type=RhiTestType::Rendering; name="zorah_stream_material_probes"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        const char* manifestPath=std::getenv("METALLIC_ZORAH_Z4_PROBES");
        if(!manifestPath) { return RhiTestResult::skip("Set METALLIC_ZORAH_Z4_PROBES to the cooked Z2 probes.json"); }
        try {
            Json manifest,report=Json::array();
            { std::ifstream input(manifestPath); input>>manifest; }
            for(const auto& probe : manifest["probes"]) {
                const std::string name=probe["name"];
                if(name!="StoneUdim" && name!="MaskedLeaves" && name!="TextureTransformBc4" && name!="MirroredInstances" &&
                    name!="InstancingNoTangent" && name!="Unlit" && name!="Glass" && name!="Blend") { continue; }
                const auto directory=std::filesystem::absolute(context.outputDirectory/name);
                std::filesystem::create_directories(directory);
                const std::filesystem::path source=probe["path"].get<std::string>();
                auto cache=source; cache.replace_extension("meshstream.bin");
                const auto referencePath=residentProbe(source,directory);
                std::map<std::string,std::vector<uint32_t>> reference;
                float3 probeCenter(0.0f), probeDirection(0,0,1);
                float probeRadius=1;
                for(bool streamed : {false,true}) {
                    scene::SceneDocument scene;
                    require(streamed ? scene.loadStreamMetadata(source) : scene.load(referencePath),scene.lastLoadResult().error);
                    RenderGraphPreviewRenderer preview;
                    preview.bindRuntimeScene(&scene);
                    require(bool(preview.initialize(context.enableValidation,true,false)),preview.lastLog());
                    preview.setEnvironment({.enabled=false});
                    auto lighting=scene.lighting(); lighting.autoExposure.enabled=false;
                    scene::PunctualLight light; light.properties.type="directional"; light.properties.intensity=3;
                    light.direction=float3(-.3f,-.4f,-1); lighting.lights={light}; preview.setLighting(lighting);
                    auto graph=materialGraph(streamed ? source : referencePath,cache,streamed);
                    const auto raster=graph.findNode("Raster")->id,deferred=graph.findNode("Deferred")->id;
                    if(!streamed) {
                        // City-wide instance bounds and long, thin stalk bounds can
                        // produce empty probes. Frame the largest authored triangle.
                        const auto cross3=[](float3 a,float3 b) { return float3(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x); };
                        const auto length3=[](float3 v) { return std::sqrt(v.x*v.x+v.y*v.y+v.z*v.z); };
                        float maximumArea=0;
                        std::vector<bool> visited(scene.renderPrimitives().size(),false);
                        for(const auto& instance : scene.renderNodes()) {
                            if(visited[instance.renderPrimitiveIndex]) { continue; }
                            visited[instance.renderPrimitiveIndex]=true;
                            const auto& primitive=scene.renderPrimitives()[instance.renderPrimitiveIndex];
                            const auto& m=instance.worldMatrix;
                            const float determinant=m.a00*(m.a11*m.a22-m.a12*m.a21)-m.a01*(m.a10*m.a22-m.a12*m.a20)+m.a02*(m.a10*m.a21-m.a11*m.a20);
                            for(size_t index=0;index+2<primitive.indices.size();index+=3) {
                                const float3 a=(m*float4(primitive.positions[primitive.indices[index]],1)).xyz;
                                const float3 b=(m*float4(primitive.positions[primitive.indices[index+1]],1)).xyz;
                                const float3 c=(m*float4(primitive.positions[primitive.indices[index+2]],1)).xyz;
                                const auto normal=cross3(b-a,c-a); const float area=length3(normal);
                                if(area<=maximumArea) { continue; }
                                maximumArea=area; probeCenter=(a+b+c)/3;
                                probeDirection=normal*((determinant<0 ? -1.f : 1.f)/area);
                                probeRadius=std::max({length3(a-probeCenter),length3(b-probeCenter),length3(c-probeCenter),.001f});
                            }
                        }
                        require(maximumArea>0,"Probe has no nondegenerate triangles");
                    }
                    const auto eye=probeCenter+probeDirection*(probeRadius*2.6f);
                    lighting.lights.front().direction=-probeDirection;
                    preview.setLighting(lighting);
                    graph.setNodeRuntimeProperty(raster,"camera.eye",{eye.x,eye.y,eye.z});
                    graph.setNodeRuntimeProperty(raster,"camera.center",{probeCenter.x,probeCenter.y,probeCenter.z});
                    graph.setNodeRuntimeProperty(raster,"camera.up",std::abs(probeDirection.y)>.95f ? Json({0,0,1}) : Json({0,1,0}));
                    graph.setNodeRuntimeProperty(raster,"camera.znear",probeRadius*.001f);
                    graph.setNodeRuntimeProperty(raster,"camera.zfar",probeRadius*16);
                    graph.setNodeRuntimeProperty(raster,"maxResidentPages",512);
                    graph.setNodeRuntimeProperty(raster,"maxLockedFallbackPages",512);
                    graph.setNodeRuntimeProperty(raster,"maxActiveGroups",4096);
                    graph.setNodeRuntimeProperty(raster,"maxTraversalWorkItems",8192);
                    graph.setNodeRuntimeProperty(raster,"enableClusterRtx",streamed);
                    graph.setNodeRuntimeProperty(raster,"enableClas",streamed);
                    graph.setNodeRuntimeProperty(raster,"maxClasBytes",134217728);
                    for(const char* view : {"mappedNormal","baseColor","normalTexture","final"}) {
                        graph.setNodeRuntimeProperty(deferred,"debugView",view);
                        for(uint32_t frame=0;frame<(streamed ? 24u : 1u);++frame) { require(bool(preview.render(graph,256,256)),name+": "+preview.lastLog()); }
                        std::string log;
                        const std::string label=std::string(streamed ? "stream-" : "resident-")+view;
                        require(saveRgba8Png(directory/(label+".png"),reinterpret_cast<const uint8_t*>(preview.pixels().data()),256,256,log),log);
                        if(!streamed) { reference[view]=preview.pixels(); continue; }
                        const auto& expected=reference.at(view);
                        const auto& coverage=reference.at("mappedNormal");
                        uint32_t subjects=0,outliers=0; double error=0;
                        for(uint32_t y=1;y<255;++y) { for(uint32_t x=1;x<255;++x) {
                            const uint32_t i=y*256+x;
                            if(!(coverage[i]&0xffffff) || !(coverage[i-1]&0xffffff) || !(coverage[i+1]&0xffffff) ||
                                !(coverage[i-256]&0xffffff) || !(coverage[i+256]&0xffffff)) { continue; }
                            ++subjects; uint32_t maximum=0;
                            for(uint32_t shift : {0u,8u,16u}) { const auto delta=uint32_t(std::abs(int((expected[i]>>shift)&255)-int((preview.pixels()[i]>>shift)&255))); error+=delta; maximum=std::max(maximum,delta); }
                            outliers+=maximum>8;
                        }}
                        const bool comparable=!((name=="Glass" || name=="Blend") && std::string_view(view)=="final") &&
                            !(name=="Unlit" && std::string_view(view)=="mappedNormal");
                        report.push_back({{"probe",name},{"view",view},{"pixels",subjects},{"meanError",error/std::max(subjects*3,1u)},
                            {"outliers",outliers},{"comparable",comparable}});
                        { std::ofstream output(context.outputDirectory/"ZorahZ4Probes.json"); output<<report.dump(2); }
                        // Glass/BLEND final uses explicit ray continuation, whereas the
                        // resident realtime reference above only compares the surface.
                        // The unlit source has no NORMAL: resident smooth normals
                        // and the stream geometric fallback need not match. Neither
                        // affects unlit radiance; retain coverage and color checks.
                        require(subjects>20,"Probe has no comparison coverage: "+name);
                        if(comparable) {
                            require(subjects>20 && error/(subjects*3)<2.0 && double(outliers)/subjects<.08,
                                "Probe pixel mismatch: "+report.back().dump());
                        }
                    }
                }
            }
            return RhiTestResult::pass("Zorah KTX2 attribute/material probes rendered and compared with independently decoded resident geometry");
        } catch(const std::exception& error) { return RhiTestResult::fail(error.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(ZorahStreamMaterialProbesTest);
} // namespace
} // namespace metallic::tests
