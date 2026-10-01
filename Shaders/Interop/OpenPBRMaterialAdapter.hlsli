#ifndef METALLIC_OPENPBR_MATERIAL_ADAPTER
#define METALLIC_OPENPBR_MATERIAL_ADAPTER

// Included by the program after its canonical vendor owner: LUT callbacks and
// feature macros remain program-specific. Prepared state is view-dependent.
OpenPBR_PreparedBsdf prepareOpenPBRMaterial(OpenPBR_ResolvedInputs inputs,
    float3 throughput, float3 wavelengths, float exteriorIor, float3 outgoingDirection)
{
    return openpbr_prepare(inputs, throughput, wavelengths, exteriorIor, outgoingDirection);
}

// Vendor Eval includes the surface projection factor. Consumers must NOT
// multiply by another cosine. Preserve diffuse/specular separation for guides.
OpenPBR_DiffuseSpecular evalOpenPBRMaterialProjected(OpenPBR_PreparedBsdf prepared, float3 wi)
{
    return openpbr_eval(prepared, wi);
}

float pdfOpenPBRMaterial(OpenPBR_PreparedBsdf prepared, float3 wi)
{
    return openpbr_pdf(prepared, wi);
}

// Preserve the vendor weight/PDF/event convention used by the integrator.
void sampleOpenPBRMaterial(OpenPBR_PreparedBsdf prepared, float3 random,
    out float3 wi, out OpenPBR_DiffuseSpecular weight, out float pdf,
    out OpenPBR_BsdfLobeType eventType)
{
    openpbr_sample(prepared, random, wi, weight, pdf, eventType);
}

#endif
