#include "Runtime/Scene/UsdSceneImporter.h"

#include "pxr/base/gf/matrix4d.h"
#include "pxr/base/gf/vec2d.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec2h.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/tf/errorMark.h"
#include "pxr/base/tf/token.h"
#include "pxr/base/vt/array.h"
#include "pxr/base/vt/value.h"
#include "pxr/usd/ar/asset.h"
#include "pxr/usd/ar/packageUtils.h"
#include "pxr/usd/ar/resolvedPath.h"
#include "pxr/usd/ar/resolver.h"
#include "pxr/usd/sdf/assetPath.h"
#include "pxr/usd/usd/primFlags.h"
#include "pxr/usd/usd/primRange.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usd/usdGeom/camera.h"
#include "pxr/usd/usdGeom/imageable.h"
#include "pxr/usd/usdGeom/mesh.h"
#include "pxr/usd/usdGeom/metrics.h"
#include "pxr/usd/usdGeom/primvar.h"
#include "pxr/usd/usdGeom/primvarsAPI.h"
#include "pxr/usd/usdGeom/subset.h"
#include "pxr/usd/usdGeom/tokens.h"
#include "pxr/usd/usdGeom/xformCache.h"
#include "pxr/usd/usdShade/input.h"
#include "pxr/usd/usdShade/material.h"
#include "pxr/usd/usdShade/materialBindingAPI.h"
#include "pxr/usd/usdShade/shader.h"
#include "pxr/usd/usdShade/tokens.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace metallic::scene::detail {
namespace {

constexpr float kDegreesToRadians = 0.01745329251994329577f;

void appendWarning(std::string& warning, std::string message)
{
    if (message.empty()) {
        return;
    }
    if (!warning.empty() && warning.back() != '\n') {
        warning += '\n';
    }
    warning += std::move(message);
}

void consumeOpenUsdErrors(const TfErrorMark& errorMark, std::string& destination)
{
    for (const auto& error : errorMark) {
        appendWarning(destination, error.GetCommentary());
    }
    errorMark.Clear();
}

std::string normalizedAssetUri(std::string uri)
{
    std::replace(uri.begin(), uri.end(), '\\', '/');
    return uri;
}

std::string primDisplayName(const UsdPrim& prim, std::string fallback)
{
    const std::string displayName = prim.GetDisplayName();
    if (!displayName.empty()) {
        return displayName;
    }
    const std::string name = prim.GetName().GetString();
    return name.empty() ? std::move(fallback) : name;
}

float3 toFloat3(const GfVec3f& value)
{
    return float3(value[0], value[1], value[2]);
}

float4x4 toColumnVectorMatrix(const GfMatrix4d& matrix)
{
    float4x4 result;
    for (size_t row = 0; row < 4; ++row) {
        for (size_t column = 0; column < 4; ++column) {
            result.a[column * 4u + row] = static_cast<float>(matrix[column][row]);
        }
    }
    return result;
}

bool matrixFinite(const float4x4& matrix)
{
    return std::all_of(std::begin(matrix.a), std::end(matrix.a), [](float value) {
        return std::isfinite(value);
    });
}

float4x4 stageConversionMatrix(const TfToken& upAxis, double stageMetersPerUnit)
{
    float4x4 axis = float4x4::Identity();
    if (upAxis == UsdGeomTokens->z) {
        axis = float4x4(
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, -1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f);
    } else if (upAxis == TfToken("X")) {
        axis = float4x4(
            0.0f, -1.0f, 0.0f, 0.0f,
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f);
    }

    const float metersPerUnit = std::isfinite(stageMetersPerUnit) &&
            stageMetersPerUnit > 0.0
        ? static_cast<float>(stageMetersPerUnit)
        : 1.0f;
    float4x4 scale;
    scale.SetupByScale(float3(metersPerUnit, metersPerUnit, metersPerUnit));
    return axis * scale;
}

template <typename T>
T shaderInputValue(
    const UsdShadeShader& shader,
    const char* inputName,
    const T& fallback)
{
    T value = fallback;
    const UsdShadeInput input = shader.GetInput(TfToken(inputName));
    if (input) {
        (void)input.Get(&value);
    }
    return value;
}

uint8_t textureChannel(const TfToken& outputName)
{
    const std::string name = outputName.GetString();
    if (name == "g" || name == "green") {
        return 1;
    }
    if (name == "b" || name == "blue") {
        return 2;
    }
    if (name == "a" || name == "alpha") {
        return 3;
    }
    return 0;
}

struct TextureBinding {
    int32_t sourceIndex = kInvalidSceneIndex;
    uint8_t channel = 0;

    explicit operator bool() const
    {
        return sourceIndex != kInvalidSceneIndex;
    }
};

struct TextureSource {
    std::string key;
    std::string name;
    RenderImage::ChannelSource image;
    int32_t texCoord = 0;
    std::array<float, 6> uvTransform{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
};

class TextureBuilder {
public:
    TextureBuilder(const std::filesystem::path& scenePath, UsdImportedScene& destination)
        : scenePath_(scenePath)
        , destination_(destination)
    {
    }

    TextureBinding binding(const UsdShadeInput& input)
    {
        if (!input) {
            return {};
        }

        for (const UsdAttribute& attribute : input.GetValueProducingAttributes(true)) {
            const UsdShadeShader shader(attribute.GetPrim());
            TfToken shaderId;
            if (!shader || !shader.GetIdAttr().Get(&shaderId) ||
                shaderId != TfToken("UsdUVTexture")) {
                continue;
            }

            const int32_t sourceIndex = findOrCreateSource(shader);
            if (sourceIndex != kInvalidSceneIndex) {
                return TextureBinding{
                    .sourceIndex = sourceIndex,
                    .channel = textureChannel(attribute.GetBaseName()),
                };
            }
        }
        return {};
    }

    RenderTextureInfo directTexture(const TextureBinding& binding)
    {
        if (!validSource(binding.sourceIndex)) {
            return {};
        }
        const auto found = directTextures_.find(binding.sourceIndex);
        if (found != directTextures_.end()) {
            return makeTextureInfo(found->second, binding.sourceIndex);
        }

        const TextureSource& source = sources_[static_cast<size_t>(binding.sourceIndex)];
        RenderImage image;
        image.name = source.name;
        image.uri = source.image.uri;
        image.encodedData = source.image.encodedData;
        const int32_t imageIndex = appendImage(std::move(image));

        RenderTexture texture;
        texture.name = source.name;
        texture.imageIndex = imageIndex;
        const int32_t textureIndex = appendTexture(std::move(texture));
        directTextures_.emplace(binding.sourceIndex, textureIndex);
        return makeTextureInfo(textureIndex, binding.sourceIndex);
    }

    RenderTextureInfo baseColorOpacityTexture(
        const TextureBinding& color,
        const TextureBinding& opacity,
        std::string_view materialName)
    {
        if (!color && !opacity) {
            return {};
        }
        if (color && !opacity) {
            return directTexture(color);
        }

        std::array<ChannelBinding, 4> channels{};
        channels[0] = ChannelBinding{color.sourceIndex, 0};
        channels[1] = ChannelBinding{color.sourceIndex, 1};
        channels[2] = ChannelBinding{color.sourceIndex, 2};
        channels[3] = ChannelBinding{opacity.sourceIndex, opacity.channel};
        return composedTexture(
            std::string(materialName) + " BaseColorOpacity",
            channels,
            {255, 255, 255, 255},
            color ? color.sourceIndex : opacity.sourceIndex);
    }

    RenderTextureInfo metallicRoughnessTexture(
        const TextureBinding& metallic,
        const TextureBinding& roughness,
        std::string_view materialName)
    {
        if (!metallic && !roughness) {
            return {};
        }

        std::array<ChannelBinding, 4> channels{};
        channels[0] = ChannelBinding{kInvalidSceneIndex, 0};
        channels[1] = ChannelBinding{roughness.sourceIndex, roughness.channel};
        channels[2] = ChannelBinding{metallic.sourceIndex, metallic.channel};
        channels[3] = ChannelBinding{kInvalidSceneIndex, 3};
        return composedTexture(
            std::string(materialName) + " MetallicRoughness",
            channels,
            {255, 255, 255, 255},
            roughness ? roughness.sourceIndex : metallic.sourceIndex);
    }

private:
    struct ChannelBinding {
        int32_t sourceIndex = kInvalidSceneIndex;
        uint8_t channel = 0;
    };

    bool validSource(int32_t sourceIndex) const
    {
        return sourceIndex >= 0 &&
            static_cast<size_t>(sourceIndex) < sources_.size();
    }

    static bool readShaderId(const UsdShadeShader& shader, TfToken& id)
    {
        return shader && shader.GetIdAttr().Get(&id);
    }

    std::string readPrimvarName(const UsdShadeInput& input) const
    {
        if (!input) {
            return {};
        }
        for (const UsdAttribute& attribute : input.GetValueProducingAttributes(true)) {
            const UsdShadeShader shader(attribute.GetPrim());
            TfToken shaderId;
            if (!readShaderId(shader, shaderId) ||
                shaderId.GetString().rfind("UsdPrimvarReader_", 0) != 0) {
                continue;
            }
            TfToken tokenName;
            const UsdShadeInput varname = shader.GetInput(TfToken("varname"));
            if (varname && varname.Get(&tokenName)) {
                return tokenName.GetString();
            }
            std::string stringName;
            if (varname && varname.Get(&stringName)) {
                return stringName;
            }
        }
        return {};
    }

    void readUvMapping(const UsdShadeShader& textureShader, TextureSource& source)
    {
        const UsdShadeInput stInput = textureShader.GetInput(TfToken("st"));
        if (!stInput) {
            return;
        }

        std::string primvarName;
        for (const UsdAttribute& attribute : stInput.GetValueProducingAttributes(true)) {
            const UsdShadeShader shader(attribute.GetPrim());
            TfToken shaderId;
            if (!readShaderId(shader, shaderId)) {
                continue;
            }
            if (shaderId == TfToken("UsdTransform2d")) {
                const GfVec2f scale = shaderInputValue(
                    shader, "scale", GfVec2f(1.0f, 1.0f));
                const GfVec2f translation = shaderInputValue(
                    shader, "translation", GfVec2f(0.0f, 0.0f));
                const float rotation = shaderInputValue(shader, "rotation", 0.0f) *
                    kDegreesToRadians;
                const float cosine = std::cos(rotation);
                const float sine = std::sin(rotation);
                source.uvTransform = {
                    cosine * scale[0],
                    -sine * scale[1],
                    translation[0],
                    sine * scale[0],
                    cosine * scale[1],
                    translation[1],
                };
                primvarName = readPrimvarName(shader.GetInput(TfToken("in")));
            } else if (shaderId.GetString().rfind("UsdPrimvarReader_", 0) == 0) {
                primvarName = readPrimvarName(stInput);
            }
            if (!primvarName.empty()) {
                break;
            }
        }

        if (!primvarName.empty() && primvarName != "st") {
            appendWarning(
                destination_.warning,
                "USD texture '" + source.name + "' uses unsupported primvar '" +
                    primvarName + "'; TEXCOORD_0 is used");
        }
    }

    RenderImage::ChannelSource readImageSource(
        const UsdShadeShader& shader,
        const std::string& textureName)
    {
        RenderImage::ChannelSource result;
        SdfAssetPath assetPath;
        const UsdShadeInput fileInput = shader.GetInput(TfToken("file"));
        if (!fileInput || !fileInput.Get(&assetPath)) {
            appendWarning(
                destination_.warning,
                "USD texture '" + textureName + "' has no file input");
            return result;
        }

        const std::string authoredPath = assetPath.GetAssetPath();
        std::string resolvedPath = assetPath.GetResolvedPath();
        if (resolvedPath.empty() && !authoredPath.empty()) {
            const ArResolvedPath resolved = ArGetResolver().Resolve(authoredPath);
            resolvedPath = resolved.GetPathString();
        }

        if (ArIsPackageRelativePath(resolvedPath)) {
            result.uri = normalizedAssetUri(
                authoredPath.empty() ? resolvedPath : authoredPath);
            const std::shared_ptr<ArAsset> asset =
                ArGetResolver().OpenAsset(ArResolvedPath(resolvedPath));
            if (asset) {
                const size_t size = asset->GetSize();
                const std::shared_ptr<const char> buffer = asset->GetBuffer();
                if (buffer && size != 0) {
                    const auto* bytes = reinterpret_cast<const uint8_t*>(buffer.get());
                    result.encodedData.assign(bytes, bytes + size);
                }
            }
            if (result.encodedData.empty()) {
                appendWarning(
                    destination_.warning,
                    "OpenUSD could not read packaged texture '" + authoredPath + "'");
            }
            return result;
        }

        std::filesystem::path path = !resolvedPath.empty()
            ? std::filesystem::path(resolvedPath)
            : std::filesystem::path(authoredPath);
        if (path.is_relative()) {
            std::error_code pathError;
            std::filesystem::path absoluteScenePath =
                std::filesystem::absolute(scenePath_, pathError);
            if (pathError) {
                absoluteScenePath = scenePath_;
            }
            path = absoluteScenePath.parent_path() / path;
        }
        result.uri = normalizedAssetUri(path.lexically_normal().string());
        return result;
    }

    int32_t findOrCreateSource(const UsdShadeShader& shader)
    {
        const std::string key = shader.GetPrim().GetPath().GetString();
        const auto found = sourceIndices_.find(key);
        if (found != sourceIndices_.end()) {
            return found->second;
        }
        if (sources_.size() >= static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            return kInvalidSceneIndex;
        }

        const int32_t sourceIndex = static_cast<int32_t>(sources_.size());
        TextureSource source;
        source.key = key;
        source.name = primDisplayName(
            shader.GetPrim(),
            "USD Texture " + std::to_string(sourceIndex));
        source.image = readImageSource(shader, source.name);
        readUvMapping(shader, source);
        sources_.push_back(std::move(source));
        sourceIndices_.emplace(key, sourceIndex);
        return sourceIndex;
    }

    int32_t appendImage(RenderImage image)
    {
        if (destination_.images.size() >=
            static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            return kInvalidSceneIndex;
        }
        const int32_t index = static_cast<int32_t>(destination_.images.size());
        destination_.images.push_back(std::move(image));
        return index;
    }

    int32_t appendTexture(RenderTexture texture)
    {
        if (destination_.textures.size() >=
            static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            return kInvalidSceneIndex;
        }
        const int32_t index = static_cast<int32_t>(destination_.textures.size());
        destination_.textures.push_back(std::move(texture));
        return index;
    }

    RenderTextureInfo makeTextureInfo(int32_t textureIndex, int32_t sourceIndex) const
    {
        RenderTextureInfo result;
        result.textureIndex = textureIndex;
        if (validSource(sourceIndex)) {
            const TextureSource& source = sources_[static_cast<size_t>(sourceIndex)];
            result.texCoord = source.texCoord;
            result.uvTransform = source.uvTransform;
        }
        return result;
    }

    RenderTextureInfo composedTexture(
        std::string name,
        const std::array<ChannelBinding, 4>& bindings,
        const std::array<uint8_t, 4>& constants,
        int32_t primarySourceIndex)
    {
        RenderImage image;
        image.name = name;
        RenderImage::ChannelComposition composition;
        composition.constants = constants;
        std::unordered_map<int32_t, int32_t> compositionSources;
        for (size_t channel = 0; channel < bindings.size(); ++channel) {
            const ChannelBinding& binding = bindings[channel];
            if (!validSource(binding.sourceIndex)) {
                continue;
            }
            auto found = compositionSources.find(binding.sourceIndex);
            int32_t compositionSourceIndex = 0;
            if (found == compositionSources.end()) {
                compositionSourceIndex = static_cast<int32_t>(composition.sources.size());
                composition.sources.push_back(
                    sources_[static_cast<size_t>(binding.sourceIndex)].image);
                compositionSources.emplace(binding.sourceIndex, compositionSourceIndex);
            } else {
                compositionSourceIndex = found->second;
            }
            composition.sourceIndices[channel] = compositionSourceIndex;
            composition.sourceChannels[channel] = binding.channel;
        }
        image.channelComposition = std::move(composition);
        const int32_t imageIndex = appendImage(std::move(image));

        RenderTexture texture;
        texture.name = std::move(name);
        texture.imageIndex = imageIndex;
        const int32_t textureIndex = appendTexture(std::move(texture));
        return makeTextureInfo(textureIndex, primarySourceIndex);
    }

    std::filesystem::path scenePath_;
    UsdImportedScene& destination_;
    std::vector<TextureSource> sources_;
    std::unordered_map<std::string, int32_t> sourceIndices_;
    std::unordered_map<int32_t, int32_t> directTextures_;
};

using MaterialIndexMap = std::unordered_map<std::string, int32_t>;

void convertMaterials(
    const std::vector<UsdPrim>& prims,
    const std::filesystem::path& scenePath,
    UsdImportedScene& destination,
    MaterialIndexMap& materialIndices)
{
    TextureBuilder textures(scenePath, destination);
    for (const UsdPrim& prim : prims) {
        const UsdShadeMaterial usdMaterial(prim);
        if (!usdMaterial) {
            continue;
        }
        if (destination.materials.size() >=
            static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            break;
        }

        const int32_t materialIndex = static_cast<int32_t>(destination.materials.size());
        materialIndices.emplace(prim.GetPath().GetString(), materialIndex);

        RenderMaterial material;
        material.name = primDisplayName(
            prim,
            "USD Material " + std::to_string(materialIndex));

        const UsdShadeShader shader = usdMaterial.ComputeSurfaceSource();
        TfToken shaderId;
        if (!shader || !shader.GetIdAttr().Get(&shaderId) ||
            shaderId != TfToken("UsdPreviewSurface")) {
            appendWarning(
                destination.warning,
                "USD material '" + material.name +
                    "' has no supported UsdPreviewSurface shader");
            destination.materials.push_back(std::move(material));
            continue;
        }

        const UsdShadeInput diffuseInput = shader.GetInput(TfToken("diffuseColor"));
        const UsdShadeInput emissiveInput = shader.GetInput(TfToken("emissiveColor"));
        const UsdShadeInput metallicInput = shader.GetInput(TfToken("metallic"));
        const UsdShadeInput roughnessInput = shader.GetInput(TfToken("roughness"));
        const UsdShadeInput opacityInput = shader.GetInput(TfToken("opacity"));
        const UsdShadeInput normalInput = shader.GetInput(TfToken("normal"));
        const UsdShadeInput occlusionInput = shader.GetInput(TfToken("occlusion"));

        const TextureBinding diffuseTexture = textures.binding(diffuseInput);
        const TextureBinding emissiveTexture = textures.binding(emissiveInput);
        const TextureBinding metallicTexture = textures.binding(metallicInput);
        const TextureBinding roughnessTexture = textures.binding(roughnessInput);
        const TextureBinding opacityTexture = textures.binding(opacityInput);
        const TextureBinding normalTexture = textures.binding(normalInput);
        const TextureBinding occlusionTexture = textures.binding(occlusionInput);

        GfVec3f diffuse(0.18f, 0.18f, 0.18f);
        GfVec3f emissive(0.0f, 0.0f, 0.0f);
        float metallic = 0.0f;
        float roughness = 0.5f;
        float opacity = 1.0f;
        float opacityThreshold = 0.0f;
        float ior = 1.5f;
        if (diffuseInput) {
            (void)diffuseInput.Get(&diffuse);
        }
        if (emissiveInput) {
            (void)emissiveInput.Get(&emissive);
        }
        if (metallicInput) {
            (void)metallicInput.Get(&metallic);
        }
        if (roughnessInput) {
            (void)roughnessInput.Get(&roughness);
        }
        if (opacityInput) {
            (void)opacityInput.Get(&opacity);
        }
        const UsdShadeInput opacityThresholdInput =
            shader.GetInput(TfToken("opacityThreshold"));
        if (opacityThresholdInput) {
            (void)opacityThresholdInput.Get(&opacityThreshold);
        }
        const UsdShadeInput iorInput = shader.GetInput(TfToken("ior"));
        if (iorInput) {
            (void)iorInput.Get(&ior);
        }

        opacity = std::clamp(opacity, 0.0f, 1.0f);
        material.baseColorFactor = diffuseTexture
            ? float4(1.0f, 1.0f, 1.0f, opacityTexture ? 1.0f : opacity)
            : float4(diffuse[0], diffuse[1], diffuse[2], opacityTexture ? 1.0f : opacity);
        material.metallicFactor = metallicTexture
            ? 1.0f
            : std::clamp(metallic, 0.0f, 1.0f);
        material.roughnessFactor = roughnessTexture
            ? 1.0f
            : std::clamp(roughness, 0.0f, 1.0f);
        material.emissiveFactor = emissiveTexture
            ? float3(1.0f, 1.0f, 1.0f)
            : float3(emissive[0], emissive[1], emissive[2]);
        material.alphaCutoff = std::clamp(opacityThreshold, 0.0f, 1.0f);
        if (material.alphaCutoff > 0.0f) {
            material.alphaMode = "MASK";
        } else if (opacityTexture || opacity < 0.999f) {
            material.alphaMode = "BLEND";
        }
        material.ior = std::clamp(ior, 1.0f, 3.0f);
        material.baseColorTexture = textures.baseColorOpacityTexture(
            diffuseTexture, opacityTexture, material.name);
        material.metallicRoughnessTexture = textures.metallicRoughnessTexture(
            metallicTexture, roughnessTexture, material.name);
        material.normalTexture = textures.directTexture(normalTexture);
        material.occlusionTexture = textures.directTexture(occlusionTexture);
        material.emissiveTexture = textures.directTexture(emissiveTexture);
        destination.materials.push_back(std::move(material));
    }
}

int32_t materialIndexFor(
    const UsdShadeMaterial& material,
    const MaterialIndexMap& materialIndices)
{
    if (!material) {
        return kInvalidSceneIndex;
    }
    const auto found = materialIndices.find(material.GetPath().GetString());
    return found == materialIndices.end() ? kInvalidSceneIndex : found->second;
}

struct MeshAttributes {
    VtArray<GfVec3f> points;
    VtArray<GfVec3f> normals;
    std::vector<float2> texcoords;
    TfToken normalsInterpolation = UsdGeomTokens->constant;
    TfToken texcoordsInterpolation = UsdGeomTokens->constant;
};

bool readTexcoords(const UsdGeomMesh& mesh, MeshAttributes& attributes)
{
    const UsdGeomPrimvar st =
        UsdGeomPrimvarsAPI(mesh.GetPrim()).GetPrimvar(TfToken("st"));
    if (!st) {
        return false;
    }
    VtValue value;
    if (!st.ComputeFlattened(&value)) {
        return false;
    }

    const auto append = [&](const auto& values) {
        attributes.texcoords.reserve(values.size());
        for (const auto& item : values) {
            attributes.texcoords.emplace_back(
                static_cast<float>(item[0]),
                static_cast<float>(item[1]));
        }
    };
    if (value.IsHolding<VtArray<GfVec2f>>()) {
        append(value.UncheckedGet<VtArray<GfVec2f>>());
    } else if (value.IsHolding<VtArray<GfVec2d>>()) {
        append(value.UncheckedGet<VtArray<GfVec2d>>());
    } else if (value.IsHolding<VtArray<GfVec2h>>()) {
        append(value.UncheckedGet<VtArray<GfVec2h>>());
    } else {
        return false;
    }
    attributes.texcoordsInterpolation = st.GetInterpolation();
    return !attributes.texcoords.empty();
}

int32_t attributeElementIndex(
    const TfToken& interpolation,
    size_t faceIndex,
    int32_t pointIndex,
    size_t cornerIndex,
    size_t valueCount,
    size_t faceCount,
    size_t pointCount,
    size_t cornerCount)
{
    size_t index = 0;
    if (interpolation == UsdGeomTokens->uniform) {
        index = faceIndex;
    } else if (interpolation == UsdGeomTokens->vertex ||
        interpolation == UsdGeomTokens->varying) {
        if (pointIndex < 0) {
            return kInvalidSceneIndex;
        }
        index = static_cast<size_t>(pointIndex);
    } else if (interpolation == UsdGeomTokens->faceVarying) {
        index = cornerIndex;
    } else if (interpolation != UsdGeomTokens->constant) {
        if (valueCount == cornerCount) {
            index = cornerIndex;
        } else if (valueCount == pointCount && pointIndex >= 0) {
            index = static_cast<size_t>(pointIndex);
        } else if (valueCount == faceCount) {
            index = faceIndex;
        }
    }
    return index < valueCount ? static_cast<int32_t>(index) : kInvalidSceneIndex;
}

struct VertexReference {
    int32_t point = kInvalidSceneIndex;
    int32_t normal = kInvalidSceneIndex;
    int32_t texcoord = kInvalidSceneIndex;
};

struct VertexKey {
    int32_t point = kInvalidSceneIndex;
    int32_t normal = kInvalidSceneIndex;
    int32_t texcoord = kInvalidSceneIndex;

    bool operator==(const VertexKey&) const = default;
};

struct VertexKeyHash {
    size_t operator()(const VertexKey& key) const
    {
        size_t hash = std::hash<int32_t>{}(key.point);
        hash ^= std::hash<int32_t>{}(key.normal) + 0x9e3779b9u + (hash << 6u) + (hash >> 2u);
        hash ^= std::hash<int32_t>{}(key.texcoord) + 0x9e3779b9u + (hash << 6u) + (hash >> 2u);
        return hash;
    }
};

float3 normalizedVector(const float3& value, const float3& fallback)
{
    const float lengthSquared = value.x * value.x + value.y * value.y + value.z * value.z;
    if (!std::isfinite(lengthSquared) || lengthSquared <= 1.0e-20f) {
        return fallback;
    }
    const float inverseLength = 1.0f / std::sqrt(lengthSquared);
    return value * inverseLength;
}

void generateSmoothNormals(RenderPrimitive& primitive)
{
    primitive.normals.assign(primitive.positions.size(), float3(0.0f, 0.0f, 0.0f));
    for (size_t index = 0; index + 2 < primitive.indices.size(); index += 3) {
        const uint32_t i0 = primitive.indices[index + 0];
        const uint32_t i1 = primitive.indices[index + 1];
        const uint32_t i2 = primitive.indices[index + 2];
        if (i0 >= primitive.positions.size() || i1 >= primitive.positions.size() ||
            i2 >= primitive.positions.size()) {
            continue;
        }
        const float3 faceNormal = cross(
            primitive.positions[i1] - primitive.positions[i0],
            primitive.positions[i2] - primitive.positions[i0]);
        primitive.normals[i0] = primitive.normals[i0] + faceNormal;
        primitive.normals[i1] = primitive.normals[i1] + faceNormal;
        primitive.normals[i2] = primitive.normals[i2] + faceNormal;
    }
    for (float3& normal : primitive.normals) {
        normal = normalizedVector(normal, float3(0.0f, 1.0f, 0.0f));
    }
}

RenderPrimitive buildPrimitive(
    const UsdGeomMesh& mesh,
    const MeshAttributes& attributes,
    const std::vector<VertexReference>& sourceVertices,
    int32_t meshIndex,
    int32_t primitiveIndex,
    int32_t materialIndex)
{
    RenderPrimitive primitive;
    const std::string meshName = primDisplayName(mesh.GetPrim(), "USD Mesh");
    primitive.name = meshName + " " + std::to_string(primitiveIndex);
    primitive.meshIndex = meshIndex;
    primitive.primitiveIndex = primitiveIndex;
    primitive.materialIndex = materialIndex;
    primitive.mode = 4;

    bool allNormalsValid = !attributes.normals.empty();
    bool allTexcoordsValid = !attributes.texcoords.empty();
    std::unordered_map<VertexKey, uint32_t, VertexKeyHash> remap;
    remap.reserve(sourceVertices.size());
    primitive.indices.reserve(sourceVertices.size());
    for (const VertexReference& source : sourceVertices) {
        if (source.point < 0 ||
            static_cast<size_t>(source.point) >= attributes.points.size()) {
            continue;
        }
        const VertexKey key{source.point, source.normal, source.texcoord};
        auto [found, inserted] = remap.emplace(
            key,
            static_cast<uint32_t>(remap.size()));
        if (inserted) {
            const float3 position =
                toFloat3(attributes.points[static_cast<size_t>(source.point)]);
            primitive.positions.push_back(position);
            primitive.localBounds.include(position);

            if (allNormalsValid) {
                if (source.normal >= 0 &&
                    static_cast<size_t>(source.normal) < attributes.normals.size()) {
                    primitive.normals.push_back(
                        toFloat3(attributes.normals[static_cast<size_t>(source.normal)]));
                } else {
                    allNormalsValid = false;
                }
            }
            if (allTexcoordsValid) {
                if (source.texcoord >= 0 &&
                    static_cast<size_t>(source.texcoord) < attributes.texcoords.size()) {
                    primitive.texcoords0.push_back(
                        attributes.texcoords[static_cast<size_t>(source.texcoord)]);
                } else {
                    allTexcoordsValid = false;
                }
            }
        }
        primitive.indices.push_back(found->second);
    }

    if (!allNormalsValid || primitive.normals.size() != primitive.positions.size()) {
        primitive.normals.clear();
    }
    if (!allTexcoordsValid || primitive.texcoords0.size() != primitive.positions.size()) {
        primitive.texcoords0.clear();
    }
    primitive.hasAuthoredNormals = !primitive.normals.empty();
    if (primitive.normals.empty() && !primitive.positions.empty()) {
        generateSmoothNormals(primitive);
    } else {
        for (float3& normal : primitive.normals) {
            normal = normalizedVector(normal, float3(0.0f, 1.0f, 0.0f));
        }
    }
    primitive.vertexCount = primitive.positions.size();
    primitive.indexCount = primitive.indices.size();
    primitive.triangleCount = primitive.indices.size() / 3u;
    return primitive;
}

std::vector<RenderPrimitive> convertMesh(
    const UsdGeomMesh& mesh,
    int32_t meshIndex,
    const MaterialIndexMap& materialIndices,
    std::vector<RenderMaterial>& materials,
    std::string& warning)
{
    VtIntArray faceVertexCounts;
    VtIntArray faceVertexIndices;
    MeshAttributes attributes;
    if (!mesh.GetFaceVertexCountsAttr().Get(&faceVertexCounts) ||
        !mesh.GetFaceVertexIndicesAttr().Get(&faceVertexIndices) ||
        !mesh.GetPointsAttr().Get(&attributes.points) ||
        faceVertexCounts.empty() || faceVertexIndices.empty() ||
        attributes.points.empty()) {
        appendWarning(
            warning,
            "USD mesh '" + primDisplayName(mesh.GetPrim(), "unnamed") +
                "' has incomplete topology");
        return {};
    }

    if (mesh.GetNormalsAttr().Get(&attributes.normals)) {
        attributes.normalsInterpolation = mesh.GetNormalsInterpolation();
    }
    (void)readTexcoords(mesh, attributes);

    const int32_t defaultMaterial = materialIndexFor(
        UsdShadeMaterialBindingAPI(mesh.GetPrim()).ComputeBoundMaterial(),
        materialIndices);
    std::vector<int32_t> faceMaterials(faceVertexCounts.size(), defaultMaterial);
    for (const UsdGeomSubset& subset :
         UsdShadeMaterialBindingAPI(mesh.GetPrim()).GetMaterialBindSubsets()) {
        const int32_t subsetMaterial = materialIndexFor(
            UsdShadeMaterialBindingAPI(subset.GetPrim()).ComputeBoundMaterial(),
            materialIndices);
        VtIntArray faceIndices;
        if (!subset.GetIndicesAttr().Get(&faceIndices)) {
            continue;
        }
        for (const int faceIndex : faceIndices) {
            if (faceIndex >= 0 && static_cast<size_t>(faceIndex) < faceMaterials.size()) {
                faceMaterials[static_cast<size_t>(faceIndex)] = subsetMaterial;
            }
        }
    }

    TfToken orientation = UsdGeomTokens->rightHanded;
    bool doubleSided = false;
    (void)mesh.GetOrientationAttr().Get(&orientation);
    (void)mesh.GetDoubleSidedAttr().Get(&doubleSided);
    const bool reverseWinding = orientation == UsdGeomTokens->leftHanded;

    std::map<int32_t, std::vector<VertexReference>> materialVertices;
    size_t cornerOffset = 0;
    size_t malformedFaceCount = 0;
    for (size_t faceIndex = 0; faceIndex < faceVertexCounts.size(); ++faceIndex) {
        const int vertexCount = faceVertexCounts[faceIndex];
        if (vertexCount < 3 ||
            cornerOffset + static_cast<size_t>(std::max(vertexCount, 0)) >
                faceVertexIndices.size()) {
            cornerOffset += static_cast<size_t>(std::max(vertexCount, 0));
            ++malformedFaceCount;
            continue;
        }

        std::vector<VertexReference> face;
        face.reserve(static_cast<size_t>(vertexCount));
        bool validFace = true;
        for (int localCorner = 0; localCorner < vertexCount; ++localCorner) {
            const size_t cornerIndex = cornerOffset + static_cast<size_t>(localCorner);
            const int32_t pointIndex = faceVertexIndices[cornerIndex];
            if (pointIndex < 0 || static_cast<size_t>(pointIndex) >= attributes.points.size()) {
                validFace = false;
                break;
            }
            face.push_back(VertexReference{
                .point = pointIndex,
                .normal = attributeElementIndex(
                    attributes.normalsInterpolation,
                    faceIndex,
                    pointIndex,
                    cornerIndex,
                    attributes.normals.size(),
                    faceVertexCounts.size(),
                    attributes.points.size(),
                    faceVertexIndices.size()),
                .texcoord = attributeElementIndex(
                    attributes.texcoordsInterpolation,
                    faceIndex,
                    pointIndex,
                    cornerIndex,
                    attributes.texcoords.size(),
                    faceVertexCounts.size(),
                    attributes.points.size(),
                    faceVertexIndices.size()),
            });
        }
        cornerOffset += static_cast<size_t>(vertexCount);
        if (!validFace) {
            ++malformedFaceCount;
            continue;
        }

        std::vector<VertexReference>& destination =
            materialVertices[faceMaterials[faceIndex]];
        for (size_t corner = 1; corner + 1 < face.size(); ++corner) {
            destination.push_back(face[0]);
            if (reverseWinding) {
                destination.push_back(face[corner + 1]);
                destination.push_back(face[corner]);
            } else {
                destination.push_back(face[corner]);
                destination.push_back(face[corner + 1]);
            }
        }
    }
    if (cornerOffset != faceVertexIndices.size()) {
        ++malformedFaceCount;
    }
    if (malformedFaceCount != 0) {
        appendWarning(
            warning,
            "USD mesh '" + primDisplayName(mesh.GetPrim(), "unnamed") +
                "' skipped " + std::to_string(malformedFaceCount) +
                " malformed face(s)");
    }

    std::vector<RenderPrimitive> primitives;
    primitives.reserve(materialVertices.size());
    for (auto& [materialIndex, vertices] : materialVertices) {
        if (materialIndex >= 0 && static_cast<size_t>(materialIndex) < materials.size()) {
            materials[static_cast<size_t>(materialIndex)].doubleSided |= doubleSided;
        }
        RenderPrimitive primitive = buildPrimitive(
            mesh,
            attributes,
            vertices,
            meshIndex,
            static_cast<int32_t>(primitives.size()),
            materialIndex);
        if (primitive.triangleCount != 0) {
            primitives.push_back(std::move(primitive));
        }
    }
    return primitives;
}

int32_t convertCamera(
    const UsdGeomCamera& camera,
    double metersPerUnit,
    UsdImportedScene& destination)
{
    if (destination.cameras.size() >=
        static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        return kInvalidSceneIndex;
    }

    TfToken projection = UsdGeomTokens->perspective;
    GfVec2f clippingRange(1.0f, 1000000.0f);
    float focalLength = 50.0f;
    float horizontalAperture = 20.955f;
    float verticalAperture = 15.2908f;
    (void)camera.GetProjectionAttr().Get(&projection);
    (void)camera.GetClippingRangeAttr().Get(&clippingRange);
    (void)camera.GetFocalLengthAttr().Get(&focalLength);
    (void)camera.GetHorizontalApertureAttr().Get(&horizontalAperture);
    (void)camera.GetVerticalApertureAttr().Get(&verticalAperture);

    CameraProperties properties;
    properties.type = projection == UsdGeomTokens->orthographic
        ? CameraType::Orthographic
        : CameraType::Perspective;
    properties.aspectRatio = verticalAperture > 0.0f
        ? static_cast<double>(horizontalAperture / verticalAperture)
        : 0.0;
    properties.yfov = focalLength > 0.0f
        ? 2.0 * std::atan(
            0.5 * static_cast<double>(verticalAperture) /
            static_cast<double>(focalLength))
        : 0.0;
    properties.xmag = static_cast<double>(horizontalAperture) * 0.1 * metersPerUnit;
    properties.ymag = static_cast<double>(verticalAperture) * 0.1 * metersPerUnit;
    properties.znear = std::max(
        static_cast<double>(clippingRange[0]) * metersPerUnit,
        0.000001);
    properties.zfar = std::max(
        static_cast<double>(clippingRange[1]) * metersPerUnit,
        properties.znear * 2.0);

    const int32_t cameraIndex = static_cast<int32_t>(destination.cameras.size());
    destination.cameras.push_back(UsdImportedCamera{
        .name = primDisplayName(
            camera.GetPrim(),
            "USD Camera " + std::to_string(cameraIndex)),
        .properties = properties,
    });
    return cameraIndex;
}

float4x4 localNodeMatrix(
    const UsdPrim& prim,
    int32_t parentIndex,
    UsdGeomXformCache& xformCache,
    const float4x4& stageConversion,
    std::string& warning)
{
    const GfMatrix4d world = xformCache.GetLocalToWorldTransform(prim);
    GfMatrix4d local = world;
    if (parentIndex != kInvalidSceneIndex) {
        const UsdPrim parent = prim.GetParent();
        if (parent && !parent.IsPseudoRoot()) {
            const GfMatrix4d parentWorld = xformCache.GetLocalToWorldTransform(parent);
            local = world * parentWorld.GetInverse();
        }
    }

    float4x4 result = toColumnVectorMatrix(local);
    if (parentIndex == kInvalidSceneIndex) {
        result = stageConversion * result;
    }
    if (!matrixFinite(result)) {
        appendWarning(
            warning,
            "USD transform on '" + prim.GetPath().GetString() +
                "' is not finite; identity is used");
        return parentIndex == kInvalidSceneIndex
            ? stageConversion
            : float4x4::Identity();
    }
    return result;
}

} // namespace

bool isUsdScenePath(const std::filesystem::path& path)
{
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    return extension == ".usd" || extension == ".usda" ||
        extension == ".usdc" || extension == ".usdz";
}

bool importUsdScene(
    const std::filesystem::path& path,
    UsdImportedScene& imported)
{
    imported = {};
    TfErrorMark errorMark;
    const UsdStageRefPtr stage = UsdStage::Open(path.string(), UsdStage::LoadAll);
    if (!stage) {
        consumeOpenUsdErrors(errorMark, imported.error);
        if (imported.error.empty()) {
            imported.error = "OpenUSD failed to open the USD stage";
        }
        return false;
    }
    consumeOpenUsdErrors(errorMark, imported.warning);

    const double metersPerUnit = UsdGeomGetStageMetersPerUnit(stage);
    const TfToken upAxis = UsdGeomGetStageUpAxis(stage);
    const float4x4 stageConversion = stageConversionMatrix(upAxis, metersPerUnit);

    imported.name = path.stem().string();
    const UsdPrim defaultPrim = stage->GetDefaultPrim();
    if (defaultPrim) {
        imported.name = primDisplayName(defaultPrim, imported.name);
    }
    imported.assetInfo = SceneAssetInfo{
        .version = "USD 1.0",
        .generator = "OpenUSD 26.08",
    };

    std::vector<UsdPrim> prims;
    for (const UsdPrim& prim :
         UsdPrimRange::Stage(stage, UsdTraverseInstanceProxies())) {
        prims.push_back(prim);
    }
    if (prims.empty()) {
        imported.error = "USD stage contains no scene hierarchy";
        return false;
    }

    MaterialIndexMap materialIndices;
    convertMaterials(prims, path, imported, materialIndices);

    UsdGeomXformCache xformCache(UsdTimeCode::Default());
    std::unordered_map<std::string, int32_t> nodeIndices;
    nodeIndices.reserve(prims.size());
    for (const UsdPrim& prim : prims) {
        if (imported.nodes.size() >=
            static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            imported.error = "USD stage exceeds Metallic's node index limit";
            return false;
        }

        const std::string parentPath = prim.GetPath().GetParentPath().GetString();
        const auto parent = nodeIndices.find(parentPath);
        const int32_t parentIndex = parent == nodeIndices.end()
            ? kInvalidSceneIndex
            : parent->second;
        const int32_t nodeIndex = static_cast<int32_t>(imported.nodes.size());

        UsdImportedNode node;
        node.name = primDisplayName(
            prim,
            "USD Node " + std::to_string(nodeIndex));
        node.parent = parentIndex;
        node.localMatrix = localNodeMatrix(
            prim,
            parentIndex,
            xformCache,
            stageConversion,
            imported.warning);

        const UsdGeomImageable imageable(prim);
        if (imageable) {
            TfToken visibility = UsdGeomTokens->inherited;
            (void)imageable.GetVisibilityAttr().Get(&visibility);
            node.visible = visibility != UsdGeomTokens->invisible;
        }

        const UsdGeomMesh mesh(prim);
        if (mesh) {
            const int32_t meshIndex = static_cast<int32_t>(imported.meshes.size());
            std::vector<RenderPrimitive> primitives = convertMesh(
                mesh,
                meshIndex,
                materialIndices,
                imported.materials,
                imported.warning);
            imported.meshes.push_back(SceneMesh{
                .name = primDisplayName(
                    prim,
                    "USD Mesh " + std::to_string(meshIndex)),
                .primitiveCount = primitives.size(),
            });
            imported.meshPrimitives.push_back(std::move(primitives));
            node.meshIndex = meshIndex;
        }

        const UsdGeomCamera camera(prim);
        if (camera) {
            node.cameraIndex = convertCamera(camera, metersPerUnit, imported);
        }

        imported.nodes.push_back(std::move(node));
        nodeIndices.emplace(prim.GetPath().GetString(), nodeIndex);
        if (parentIndex == kInvalidSceneIndex) {
            imported.rootNodeIndices.push_back(nodeIndex);
        } else {
            imported.nodes[static_cast<size_t>(parentIndex)].children.push_back(nodeIndex);
        }
    }

    consumeOpenUsdErrors(errorMark, imported.warning);
    return true;
}

} // namespace metallic::scene::detail
