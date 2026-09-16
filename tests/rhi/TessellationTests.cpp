#include "RhiTest.h"
#include "Runtime/Render/TessellationPatterns.h"
#include "Runtime/Render/RenderGraph/RenderGraphExecutor.h"
#include "Runtime/Scene/SceneDocument.h"
#include "Runtime/Scene/MeshletStreamAsset.h"

#include <cmath>
#include <fstream>
#include <map>
#include <numeric>

namespace metallic::tests {
namespace {

class TessellationPatternTest final : public RhiTest {
public:
    TessellationPatternTest() { type = RhiTestType::Validation; name = "tessellation_pattern_coverage"; }
    RhiTestResult run(RhiTestContext&) override
    {
        for (uint32_t a = 1; a <= 8; ++a) {
            for (uint32_t b = 1; b <= 8; ++b) {
                for (uint32_t c = 1; c <= 8; ++c) {
                    const auto p = render::makeTessellationPattern({a, b, c});
                    if (p.vertices.size() > 64 || p.triangles.size() > 128) { return RhiTestResult::fail("Patch output exceeded mesh limits"); }
                    std::map<std::pair<uint32_t, uint32_t>, uint32_t> edges;
                    int64_t area = 0;
                    for (const auto& tri : p.triangles) {
                        for (auto v : tri) {
                            if (v >= p.vertices.size() || std::accumulate(p.vertices[v].begin(), p.vertices[v].end(), 0u) != 65535) {
                                return RhiTestResult::fail("Invalid pattern index/domain");
                            }
                        }
                        const auto& x = p.vertices[tri[0]]; const auto& y = p.vertices[tri[1]]; const auto& z = p.vertices[tri[2]];
                        int64_t triangleArea = (int64_t(y[1]) - x[1]) * (int64_t(z[2]) - x[2]) -
                            (int64_t(z[1]) - x[1]) * (int64_t(y[2]) - x[2]);
                        if (triangleArea <= 0) { return RhiTestResult::fail("Degenerate or reversed pattern triangle"); }
                        area += triangleArea;
                        for (uint32_t e = 0; e < 3; ++e) { ++edges[std::minmax(tri[e], tri[(e + 1) % 3])]; }
                    }
                    if (area != 65535ll * 65535ll) { return RhiTestResult::fail("Pattern does not cover the complete source triangle"); }
                    std::array<uint32_t, 3> boundary{};
                    for (const auto& [edge, uses] : edges) {
                        if (uses == 2) { continue; }
                        bool exterior = false;
                        for (uint32_t zero = 0; zero < 3; ++zero) {
                            if (p.vertices[edge.first][zero] == 0 && p.vertices[edge.second][zero] == 0) {
                                exterior = true; ++boundary[(zero + 1) % 3];
                            }
                        }
                        if (uses != 1 || !exterior) { return RhiTestResult::fail("Non-manifold interior or open patch edge"); }
                    }
                    if (boundary != std::array<uint32_t, 3>{a, b, c}) { return RhiTestResult::fail("Independent edge rate changed"); }
                }
            }
        }
        return RhiTestResult::pass("All 512 edge-rate patterns cover the domain exactly, have manifold interiors and bounded output");
    }
};

struct TessTestVertex { float x, y, z, nx, ny, nz, u, v; };
constexpr uint32_t kHeightSize = 65;
std::vector<uint8_t> tessTestImage()
{
    std::vector<uint8_t> pixels(kHeightSize * kHeightSize * 4);
    for (uint32_t y = 0; y < kHeightSize; ++y) {
        for (uint32_t x = 0; x < kHeightSize; ++x) {
            const size_t i = (y * kHeightSize + x) * 4;
            pixels[i] = uint8_t(std::lround(127.5 + 100.0 * std::sin(x * 6.283185307 / kHeightSize) * std::sin(y * 6.283185307 / kHeightSize)));
            pixels[i + 1] = uint8_t(x * 255 / (kHeightSize - 1));
            pixels[i + 2] = uint8_t(y * 255 / (kHeightSize - 1)); pixels[i + 3] = 255;
        }
    }
    return pixels;
}
float tessTestHeight(float u, float v, const std::vector<uint8_t>& pixels)
{
    float x = (u - std::floor(u)) * kHeightSize - .5f, y = (v - std::floor(v)) * kHeightSize - .5f;
    const int ix = int(std::floor(x)), iy = int(std::floor(y));
    const auto fetch = [&](int a, int b) { return pixels[((b + int(kHeightSize)) % kHeightSize * kHeightSize + (a + int(kHeightSize)) % kHeightSize) * 4] / 255.f; };
    return (std::lerp(std::lerp(fetch(ix, iy), fetch(ix + 1, iy), x - ix),
        std::lerp(fetch(ix, iy + 1), fetch(ix + 1, iy + 1), x - ix), y - iy) - .5f) * .4f;
}

bool writeTessFixture(const std::filesystem::path& path, bool baked, const std::vector<uint8_t>& pixels, float edgePixels = 1.0f, bool orthographic = false)
{
    using Json = nlohmann::json;
    const TessTestVertex corners[] = {{-1, -1, 0, 0, 0, 1, 0, 0}, {1, -1, 0, 0, 0, 1, 1, 0},
        {1, 1, 0, 0, 0, 1, 1, 1}, {-1, 1, 0, 0, 0, 1, 0, 1}};
    const uint32_t original[] = {0, 1, 2, 0, 2, 3};
    std::vector<TessTestVertex> vertices;
    std::vector<uint32_t> indices;
    if (!baked) { vertices.assign(std::begin(corners), std::end(corners)); indices.assign(std::begin(original), std::end(original)); }
    else {
        const float3 eye(0, -1.8f, 3), forward = normalize(-eye);
        const float pixelScale = 157.0f / (orthographic ? 3.2f : 2.0f * std::tan(3.14159265358979323846f / 6.0f));
        for (uint32_t patch = 0; patch < 2; ++patch) {
            std::array<uint32_t, 3> rates;
            for (uint32_t edge = 0; edge < 3; ++edge) {
                const auto& a = corners[original[patch * 3 + edge]];
                const auto& b = corners[original[patch * 3 + (edge + 1) % 3]];
                const float3 pa(a.x, a.y, a.z), pb(b.x, b.y, b.z);
                const float distance = orthographic ? 1.0f : std::max(0.02f, std::min(dot(pa - eye, forward), dot(pb - eye, forward)));
                rates[edge] = uint32_t(std::clamp(std::ceil(length(pb - pa) * pixelScale / (distance * edgePixels)), 1.0f, 8.0f));
            }
            const auto pattern = render::makeTessellationPattern(rates);
            std::vector<TessTestVertex> domain;
            for (const auto& q : pattern.vertices) {
                TessTestVertex v{};
                for (uint32_t i = 0; i < 3; ++i) {
                    float w = q[i] / 65535.f;
                    const auto& source = corners[original[patch * 3 + i]];
                    v.x += source.x * w; v.y += source.y * w; v.u += source.u * w; v.v += source.v * w;
                }
                v.z = tessTestHeight(v.u, v.v, pixels); domain.push_back(v);
            }
            for (const auto& tri : pattern.triangles) {
                const auto& a = domain[tri[0]]; const auto& b = domain[tri[1]]; const auto& c = domain[tri[2]];
                float3 normal = normalize(cross(float3(b.x-a.x,b.y-a.y,b.z-a.z), float3(c.x-a.x,c.y-a.y,c.z-a.z)));
                for (auto i : tri) {
                    auto v = domain[i]; v.nx = normal.x; v.ny = normal.y; v.nz = normal.z;
                    indices.push_back(uint32_t(vertices.size())); vertices.push_back(v);
                }
            }
        }
    }
    auto bin = path; bin.replace_extension("bin");
    std::ofstream binary(bin, std::ios::binary);
    binary.write(reinterpret_cast<const char*>(vertices.data()), vertices.size() * sizeof(TessTestVertex));
    binary.write(reinterpret_cast<const char*>(indices.data()), indices.size() * sizeof(uint32_t)); binary.close();
    const size_t vertexBytes = vertices.size() * sizeof(TessTestVertex);
    Json material = {{"doubleSided", true}, {"pbrMetallicRoughness", {{"baseColorTexture", {{"index", 0}}}, {"metallicFactor", 0}, {"roughnessFactor", 1}}}};
    if (!baked) { material["extras"]["METALLIC_displacement"] = {{"magnitude", .4}, {"center", .5}, {"texture", {{"index", 0}}}}; }
    Json doc = {{"asset", {{"version", "2.0"}}}, {"scene", 0}, {"scenes", Json::array({{{"nodes", {0}}}})},
        {"nodes", Json::array({{{"mesh", 0}}})}, {"meshes", Json::array({{{"primitives", Json::array({{{"attributes", {{"POSITION", 0}, {"NORMAL", 1}, {"TEXCOORD_0", 2}}}, {"indices", 3}, {"material", 0}}})}}})},
        {"buffers", Json::array({{{"uri", bin.filename().string()}, {"byteLength", vertexBytes + indices.size() * 4}}})},
        {"bufferViews", Json::array({{{"buffer", 0}, {"byteOffset", 0}, {"byteLength", vertexBytes}, {"byteStride", sizeof(TessTestVertex)}},
            {{"buffer", 0}, {"byteOffset", vertexBytes}, {"byteLength", indices.size() * 4}}})},
        {"accessors", Json::array({{{"bufferView", 0}, {"byteOffset", 0}, {"componentType", 5126}, {"count", vertices.size()}, {"type", "VEC3"}, {"min", {-1, -1, -.3}}, {"max", {1, 1, .3}}},
            {{"bufferView", 0}, {"byteOffset", 12}, {"componentType", 5126}, {"count", vertices.size()}, {"type", "VEC3"}},
            {{"bufferView", 0}, {"byteOffset", 24}, {"componentType", 5126}, {"count", vertices.size()}, {"type", "VEC2"}},
            {{"bufferView", 1}, {"componentType", 5125}, {"count", indices.size()}, {"type", "SCALAR"}}})},
        {"images", Json::array({{{"uri", "Height.png"}}})}, {"textures", Json::array({{{"source", 0}}})}, {"materials", Json::array({material})}};
    std::ofstream(path) << doc.dump(2);
    return bool(binary);
}

class TessellationRenderTest final : public RhiTest {
public:
    TessellationRenderTest() { type = RhiTestType::Rendering; name = "tessellation_displacement_render"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        std::filesystem::create_directories(context.outputDirectory);
        std::string log;
        const auto pixels = tessTestImage();
        if (!saveRgba8Png(context.outputDirectory / "Height.png", pixels.data(), kHeightSize, kHeightSize, log)) { return RhiTestResult::fail(log); }
        const auto source = std::filesystem::absolute(context.outputDirectory / "Displaced.gltf");
        const auto baked = std::filesystem::absolute(context.outputDirectory / "Baked.gltf");
        if (!writeTessFixture(source, false, pixels) || !writeTessFixture(baked, true, pixels)) { return RhiTestResult::fail("Could not write fixture"); }
        const auto stream = std::filesystem::absolute(context.outputDirectory / "Displaced.meshstream.bin");
        if (!scene::buildMeshletStreamAssetOffline({.sourcePath = source, .outputPath = stream}, log)) { return RhiTestResult::fail(log); }
        scene::SceneDocument document;
        render::RenderGraphPreviewRenderer preview;
        preview.bindRuntimeScene(&document);
        const auto initialized = preview.initialize(context.enableValidation, true);
        if (render::hasError(initialized, render::Error::Unsupported)) { return RhiTestResult::skip("Requires mesh shaders and bindless"); }
        if (!initialized) { return RhiTestResult::fail(preview.lastLog()); }
        preview.setEnvironment({.enabled = false});
        render::RenderGraph graph;
        graph.addNode("VisibilityBufferPass", "VBuffer", {{"autoLod", false}, {"lodLevel", 0}, {"visualization", "coverage"},
            {"tessellationEdgePixels", 1}, {"tessellationMaxFactor", 8}, {"temporalJitter", false}, {"hybridRaster", false},
            {"camera", {{"eye", {0, -1.8, 3.0}}, {"center", {0, 0, 0}}, {"fovDegrees", 60}, {"znear", .02}, {"zfar", 20}, {"orthoHeight", 3.2}}}});
        graph.addNode("VisibilityBufferDeferredPass", "Deferred", {{"lightingMode", "realtime"}, {"debugView", "baseColor"}, {"materialBinning", true}, {"accumulate", false}});
        graph.addEdge("VBuffer.visibility", "Deferred.visibility"); graph.addEdge("VBuffer.depth", "Deferred.depth");
        graph.addEdge("VBuffer.rasterInfo", "Deferred.rasterInfo"); graph.addEdge("VBuffer.domain", "Deferred.domain");
        graph.addNode("FinalBlitPass", "Final", {{"inputEncoding", "srgb"}});
        graph.addEdge("Deferred.color", "Final.source");
        graph.markOutput("Final.color");
        const uint32_t vbuffer = graph.findNode("VBuffer")->id, deferred = graph.findNode("Deferred")->id;
        size_t cases = 0;
        std::string failures;
        for (float edgePixels : {1.0f, 24.0f}) {
            for (bool mirrored : {false, true}) {
                for (bool ortho : {false, true}) {
                    const auto referencePath = std::filesystem::absolute(context.outputDirectory /
                        (std::string("Baked") + (ortho ? "Ortho" : "Perspective") + std::to_string(int(edgePixels)) + ".gltf"));
                    if (!writeTessFixture(referencePath, true, pixels, edgePixels, ortho)) { return RhiTestResult::fail("Could not write adaptive reference"); }
                    graph.setNodeRuntimeProperty(vbuffer, "tessellationEdgePixels", edgePixels);
                    for (bool reversed : {false, true}) {
                        for (const char* debug : {"baseColor", "shadingNormal"}) {
                            graph.setNodeRuntimeProperty(vbuffer, "camera.projection", ortho ? "orthographic" : "perspective");
                            graph.setNodeRuntimeProperty(vbuffer, "camera.reversedZ", reversed);
                            graph.setNodeRuntimeProperty(deferred, "debugView", debug);
                            const auto renderScene = [&](const std::filesystem::path& path, bool tess, bool streaming) {
                                if (!document.load(path)) { log = document.lastLoadResult().error; return false; }
                                if (mirrored) {
                                    auto transform = float4x4::Identity(); transform.SetupByScale(float3(-1, 1, 1));
                                    if (!document.setNodeLocalMatrix(0, transform)) { log = "Failed to mirror fixture"; return false; }
                                }
                                graph.setNodeRuntimeProperty(vbuffer, "path", path.string());
                                graph.setNodeRuntimeProperty(deferred, "path", path.string());
                                graph.setNodeRuntimeProperty(vbuffer, "tessellation", tess);
                                graph.setNodeRuntimeProperty(vbuffer, "clusterPrebin", !mirrored);
                                graph.setNodeRuntimeProperty(vbuffer, "enableMeshletStreaming", streaming);
                                graph.setNodeRuntimeProperty(vbuffer, "streamAssetPath", stream.string());
                                graph.setNodeRuntimeProperty(vbuffer, "maxActiveGroups", 32);
                                graph.markDirty();
                                for (uint32_t i = 0; i < (streaming ? 8u : 2u); ++i) {
                                    if (!preview.render(graph, 193, 157)) { log = preview.lastLog(); return false; }
                                }
                                return true;
                            };
                            if (!renderScene(referencePath, false, false)) { return RhiTestResult::fail(log); }
                            const auto reference = preview.pixels();
                            for (bool streaming : {false, true}) {
                                if (!renderScene(source, true, streaming)) { return RhiTestResult::fail(log); }
                                size_t covered = 0, outliers = 0; double error = 0;
                                for (size_t i = 0; i < reference.size(); ++i) {
                                    covered += (reference[i] & 0xffffffu) != 0;
                                    int maximum = 0;
                                    for (uint32_t shift : {0u, 8u, 16u}) {
                                        const int d = std::abs(int((reference[i] >> shift) & 255) - int((preview.pixels()[i] >> shift) & 255));
                                        error += d; maximum = std::max(maximum, d);
                                    }
                                    outliers += maximum > 4;
                                }
                                const auto name = std::to_string(int(edgePixels)) + std::string(streaming ? "Stream" : "Resident") + (mirrored ? "MirroredUnbinned" : "") + (ortho ? "Ortho" : "Perspective") + (reversed ? "Reverse" : "Standard") + debug;
                                saveRgba8Png(context.outputDirectory / (name + ".png"), reinterpret_cast<const uint8_t*>(preview.pixels().data()), 193, 157, log);
                                if (covered < 3000 || outliers > reference.size() / 100 || error / (reference.size() * 3) > 1.0) {
                                    saveRgba8Png(context.outputDirectory / (name + "Reference.png"), reinterpret_cast<const uint8_t*>(reference.data()), 193, 157, log);
                                    failures += name + " differs from explicitly displaced geometry: covered=" + std::to_string(covered) +
                                        " outliers=" + std::to_string(outliers) + " mean=" + std::to_string(error / (reference.size() * 3)) + "\n";
                                }
                                ++cases;
                            }
                        }
                    }
                }
            }
        }
        if (!failures.empty()) { return RhiTestResult::fail(failures); }
        // Changing a material must republish the immutable GPU table and bounds
        // without a graph reload, then restore exactly the same visible surface.
        const auto beforeEdit = preview.pixels();
        const auto originalMaterial = document.materials()[0];
        auto flatMaterial = originalMaterial; flatMaterial.displacementMagnitude = 0.0f;
        if (!document.setMaterialProperties(0, flatMaterial) || !preview.render(graph, 193, 157)) {
            return RhiTestResult::fail("Live displacement edit failed: " + preview.lastLog());
        }
        if (preview.pixels() == beforeEdit) { return RhiTestResult::fail("Live material edit did not reach rasterization"); }
        if (std::count_if(preview.pixels().begin(), preview.pixels().end(), [](uint32_t pixel) { return (pixel & 0xffffffu) != 0; }) < 3000) {
            return RhiTestResult::fail("Live material edit discarded the resident stream cut");
        }
        if (!document.setMaterialProperties(0, originalMaterial) || !preview.render(graph, 193, 157)) {
            return RhiTestResult::fail("Live displacement restore failed: " + preview.lastLog());
        }
        if (preview.pixels() != beforeEdit) {
            saveRgba8Png(context.outputDirectory / "BeforeMaterialEdit.png", reinterpret_cast<const uint8_t*>(beforeEdit.data()), 193, 157, log);
            saveRgba8Png(context.outputDirectory / "RestoredMaterial.png", reinterpret_cast<const uint8_t*>(preview.pixels().data()), 193, 157, log);
            return RhiTestResult::fail("Live displacement restore changed the original image");
        }
        return RhiTestResult::pass(std::to_string(cases) + " resident/stream displaced renders match baked geometry, UV and normals across projection/depth conventions");
    }
};

METALLIC_REGISTER_RHI_TEST(TessellationPatternTest);
METALLIC_REGISTER_RHI_TEST(TessellationRenderTest);
} // namespace
} // namespace metallic::tests
