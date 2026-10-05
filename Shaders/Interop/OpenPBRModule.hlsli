// Renderer boundary: no vendor headers or language interop macros.
import OpenPBR;
using Metallic.OpenPBR;
struct SceneOpenPBRContext : IOpenPBRContext
{
    static bool enableSheenAndCoat() { return OPENPBR_FEATURE_EnableSheenAndCoat; }
    static bool enableDispersion() { return OPENPBR_FEATURE_EnableDispersion; }
    static bool enableTranslucency() { return OPENPBR_FEATURE_EnableTranslucency; }
    static bool enableMetallic() { return OPENPBR_FEATURE_EnableMetallic; }
    float4 sample2D(int id, float2 uv) { return OPENPBR_SAMPLE_2D_TEXTURE(id, uv); }
    float4 sample3D(int id, float3 uvw) { return OPENPBR_SAMPLE_3D_TEXTURE(id, uvw); }
};

static const float3 OpenPBR_BaseRgbWavelengths_nm = OpenPBRBaseRGBWavelengthsNm;
static const uint OpenPBR_BsdfLobeTypeDiffuse = OpenPBRBSDFLobeTypeDiffuse;
static const uint OpenPBR_BsdfLobeTypeGlossy = OpenPBRBSDFLobeTypeGlossy;
static const uint OpenPBR_BsdfLobeTypeReflection = OpenPBRBSDFLobeTypeReflection;
static const uint OpenPBR_BsdfLobeTypeSpecular = OpenPBRBSDFLobeTypeSpecular;
static const uint OpenPBR_BsdfLobeTypeTransmission = OpenPBRBSDFLobeTypeTransmission;
typealias OpenPBR_DiffuseSpecular = OpenPBRDiffuseSpecular;
typealias OpenPBR_PreparedBsdf = OpenPBRPreparedBSDF;
typealias OpenPBR_ResolvedInputs = OpenPBRResolvedInputs;
static const float OpenPBR_VacuumIor = OpenPBRVacuumIor;
OpenPBRResolvedInputs openpbr_make_default_resolved_inputs() { return openPBRMakeDefaultResolvedInputs(); }
OpenPBRBasis openpbr_make_basis(const float3 normal) { return openPBRMakeBasis(normal); }
OpenPBRBasis openpbr_make_basis(const float3 normal, const float3 tangent, const float handedness)
{
    return openPBRMakeBasis(normal, tangent, handedness);
}
OpenPBRBasis openpbr_make_basis(const float3 normal, const float3 tangent, const float3 bitangent)
{
    return openPBRMakeBasis(normal, tangent, bitangent);
}
float3 openpbr_get_sum_of_diffuse_specular(const OpenPBRDiffuseSpecular diffuse_specular)
{
    return openPBRGetSumOfDiffuseSpecular(diffuse_specular);
}
OpenPBRPreparedBSDF openpbr_prepare(const OpenPBRResolvedInputs resolved_inputs, const float3 path_throughput,
                                    const float3 rgb_wavelengths_nm, const float exterior_ior,
                                    const float3 view_direction)
{
    return openPBRPrepare(SceneOpenPBRContext(), resolved_inputs, path_throughput, rgb_wavelengths_nm, exterior_ior,
                          view_direction);
}
OpenPBRDiffuseSpecular openpbr_eval(const OpenPBRPreparedBSDF prepared, const float3 light_direction)
{
    return openPBREval(SceneOpenPBRContext(), prepared, light_direction);
}
void openpbr_sample(const OpenPBRPreparedBSDF prepared, const float3 rand, out float3 light_direction,
                    out OpenPBRDiffuseSpecular weight, out float pdf, out uint sampled_type)
{
    openPBRSample(SceneOpenPBRContext(), prepared, rand, light_direction, weight, pdf, sampled_type);
}
float openpbr_pdf(const OpenPBRPreparedBSDF prepared, const float3 light_direction)
{
    return openPBRPdf(SceneOpenPBRContext(), prepared, light_direction);
}
typealias OpenPBR_BsdfLobeType = uint;
