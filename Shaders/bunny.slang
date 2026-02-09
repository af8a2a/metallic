struct Uniforms {
    float4x4 mvp;
    float4x4 modelView;
    float4   lightDir; // view-space light direction (normalized), w unused
    float4   lightColorIntensity; // xyz=color, w=intensity
};

// Global uniform binding — Slang maps this to a Metal buffer binding
// without interfering with stage_in vertex attributes
ConstantBuffer<Uniforms> uniforms;

struct VertexInput {
    float3 position : POSITION;
    float3 normal   : NORMAL;
};

struct VertexOutput {
    float4 clipPos    : SV_Position;
    float3 viewNormal : NORMAL;
    float3 viewPos    : TEXCOORD0;
};

[shader("vertex")]
VertexOutput vertexMain(VertexInput input) {
    VertexOutput output;
    output.clipPos = mul(uniforms.mvp, float4(input.position, 1.0));

    float4 viewPosition = mul(uniforms.modelView, float4(input.position, 1.0));
    output.viewPos = viewPosition.xyz;

    // Transform normal by upper-left 3x3 of modelView
    float3 n;
    n.x = dot(uniforms.modelView[0].xyz, input.normal);
    n.y = dot(uniforms.modelView[1].xyz, input.normal);
    n.z = dot(uniforms.modelView[2].xyz, input.normal);
    output.viewNormal = normalize(n);

    return output;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
    float3 N = normalize(input.viewNormal);
    float3 L = normalize(uniforms.lightDir.xyz);
    float3 V = normalize(-input.viewPos);
    float3 H = normalize(L + V);

    float3 baseColor = float3(0.7, 0.7, 0.72);

    float ambient = 0.15;
    float diffuse = max(dot(N, L), 0.0);
    float specular = pow(max(dot(N, H), 0.0), 64.0);

    float3 lightColor = uniforms.lightColorIntensity.xyz * uniforms.lightColorIntensity.w;
    float3 color = baseColor * ambient +
                   baseColor * (diffuse * 0.75) * lightColor +
                   lightColor * (specular * 0.4);
    return float4(color, 1.0);
}
