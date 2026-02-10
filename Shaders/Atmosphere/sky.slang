static const int TRANSMITTANCE_TEXTURE_WIDTH = 256;
static const int TRANSMITTANCE_TEXTURE_HEIGHT = 64;
static const int SCATTERING_TEXTURE_R_SIZE = 32;
static const int SCATTERING_TEXTURE_MU_SIZE = 128;
static const int SCATTERING_TEXTURE_MU_S_SIZE = 32;
static const int SCATTERING_TEXTURE_NU_SIZE = 8;
static const int IRRADIANCE_TEXTURE_WIDTH = 64;
static const int IRRADIANCE_TEXTURE_HEIGHT = 16;

static const float kPi = 3.14159265359;
static const float kLengthUnitInMeters = 1000.0;

struct AtmosphereParameters {
    float3 solar_irradiance;
    float  sun_angular_radius;
    float  bottom_radius;
    float  top_radius;
    float3 rayleigh_scattering;
    float3 mie_scattering;
    float3 mie_extinction;
    float  mie_phase_function_g;
    float3 absorption_extinction;
    float3 ground_albedo;
    float  mu_s_min;
};

// Precomputed atmosphere constants from Bruneton's reference model.
static const AtmosphereParameters ATMOSPHERE = {
    float3(1.474000, 1.850400, 1.911980),
    0.004675,
    6360.000000,
    6420.000000,
    float3(0.005802, 0.013558, 0.033100),
    float3(0.003996, 0.003996, 0.003996),
    float3(0.004440, 0.004440, 0.004440),
    0.800000,
    float3(0.000650, 0.001881, 0.000085),
    float3(0.100000, 0.100000, 0.100000),
    -0.207912
};

struct SkyUniforms {
    float4x4 invViewProj;
    float4   cameraWorldPos;
    float4   sunDirection;
    float4   params; // x = exposure
    uint     screenWidth;
    uint     screenHeight;
    uint     pad0;
    uint     pad1;
};

ConstantBuffer<SkyUniforms> skyUniforms;

Texture2D<float4> transmittance_texture;
Texture3D<float4> scattering_texture;
Texture2D<float4> irradiance_texture;
SamplerState      linear_sampler;

float ClampCosine(float mu) {
    return clamp(mu, -1.0, 1.0);
}

float ClampDistance(float d) {
    return max(d, 0.0);
}

float ClampRadius(AtmosphereParameters atmosphere, float r) {
    return clamp(r, atmosphere.bottom_radius, atmosphere.top_radius);
}

float SafeSqrt(float a) {
    return sqrt(max(a, 0.0));
}

float DistanceToTopAtmosphereBoundary(AtmosphereParameters atmosphere, float r, float mu) {
    float discriminant = r * r * (mu * mu - 1.0) +
        atmosphere.top_radius * atmosphere.top_radius;
    return ClampDistance(-r * mu + SafeSqrt(discriminant));
}

float DistanceToBottomAtmosphereBoundary(AtmosphereParameters atmosphere, float r, float mu) {
    float discriminant = r * r * (mu * mu - 1.0) +
        atmosphere.bottom_radius * atmosphere.bottom_radius;
    return ClampDistance(-r * mu - SafeSqrt(discriminant));
}

bool RayIntersectsGround(AtmosphereParameters atmosphere, float r, float mu) {
    return mu < 0.0 && r * r * (mu * mu - 1.0) +
        atmosphere.bottom_radius * atmosphere.bottom_radius >= 0.0;
}

float GetTextureCoordFromUnitRange(float x, int texture_size) {
    return 0.5 / float(texture_size) + x * (1.0 - 1.0 / float(texture_size));
}

float2 GetTransmittanceTextureUvFromRMu(AtmosphereParameters atmosphere, float r, float mu) {
    float H = sqrt(atmosphere.top_radius * atmosphere.top_radius -
        atmosphere.bottom_radius * atmosphere.bottom_radius);
    float rho = SafeSqrt(r * r - atmosphere.bottom_radius * atmosphere.bottom_radius);
    float d = DistanceToTopAtmosphereBoundary(atmosphere, r, mu);
    float d_min = atmosphere.top_radius - r;
    float d_max = rho + H;
    float x_mu = (d - d_min) / (d_max - d_min);
    float x_r = rho / H;
    return float2(
        GetTextureCoordFromUnitRange(x_mu, TRANSMITTANCE_TEXTURE_WIDTH),
        GetTextureCoordFromUnitRange(x_r, TRANSMITTANCE_TEXTURE_HEIGHT));
}

float3 GetTransmittanceToTopAtmosphereBoundary(AtmosphereParameters atmosphere, float r, float mu) {
    float2 uv = GetTransmittanceTextureUvFromRMu(atmosphere, r, mu);
    return transmittance_texture.SampleLevel(linear_sampler, uv, 0).rgb;
}

float3 GetTransmittance(AtmosphereParameters atmosphere, float r, float mu, float d,
                        bool ray_r_mu_intersects_ground) {
    float r_d = ClampRadius(atmosphere, sqrt(d * d + 2.0 * r * mu * d + r * r));
    float mu_d = ClampCosine((r * mu + d) / r_d);
    if (ray_r_mu_intersects_ground) {
        return min(
            GetTransmittanceToTopAtmosphereBoundary(atmosphere, r_d, -mu_d) /
            GetTransmittanceToTopAtmosphereBoundary(atmosphere, r, -mu),
            float3(1.0));
    }
    return min(
        GetTransmittanceToTopAtmosphereBoundary(atmosphere, r, mu) /
        GetTransmittanceToTopAtmosphereBoundary(atmosphere, r_d, mu_d),
        float3(1.0));
}

float4 GetScatteringTextureUvwzFromRMuMuSNu(AtmosphereParameters atmosphere,
                                            float r, float mu, float mu_s, float nu,
                                            bool ray_r_mu_intersects_ground) {
    float H = sqrt(atmosphere.top_radius * atmosphere.top_radius -
        atmosphere.bottom_radius * atmosphere.bottom_radius);
    float rho = SafeSqrt(r * r - atmosphere.bottom_radius * atmosphere.bottom_radius);
    float u_r = GetTextureCoordFromUnitRange(rho / H, SCATTERING_TEXTURE_R_SIZE);
    float r_mu = r * mu;
    float discriminant =
        r_mu * r_mu - r * r + atmosphere.bottom_radius * atmosphere.bottom_radius;
    float u_mu;
    if (ray_r_mu_intersects_ground) {
        float d = -r_mu - SafeSqrt(discriminant);
        float d_min = r - atmosphere.bottom_radius;
        float d_max = rho;
        u_mu = 0.5 - 0.5 * GetTextureCoordFromUnitRange(
            d_max == d_min ? 0.0 : (d - d_min) / (d_max - d_min),
            SCATTERING_TEXTURE_MU_SIZE / 2);
    } else {
        float d = -r_mu + SafeSqrt(discriminant + H * H);
        float d_min = atmosphere.top_radius - r;
        float d_max = rho + H;
        u_mu = 0.5 + 0.5 * GetTextureCoordFromUnitRange(
            (d - d_min) / (d_max - d_min),
            SCATTERING_TEXTURE_MU_SIZE / 2);
    }
    float d = DistanceToTopAtmosphereBoundary(atmosphere, atmosphere.bottom_radius, mu_s);
    float d_min = atmosphere.top_radius - atmosphere.bottom_radius;
    float d_max = H;
    float a = (d - d_min) / (d_max - d_min);
    float D = DistanceToTopAtmosphereBoundary(atmosphere, atmosphere.bottom_radius, atmosphere.mu_s_min);
    float A = (D - d_min) / (d_max - d_min);
    float u_mu_s = GetTextureCoordFromUnitRange(
        max(1.0 - a / A, 0.0) / (1.0 + a),
        SCATTERING_TEXTURE_MU_S_SIZE);
    float u_nu = (nu + 1.0) / 2.0;
    return float4(u_nu, u_mu_s, u_mu, u_r);
}

float3 GetExtrapolatedSingleMieScattering(AtmosphereParameters atmosphere, float4 scattering) {
    if (scattering.r <= 0.0) {
        return float3(0.0);
    }
    return scattering.rgb * scattering.a / scattering.r *
        (atmosphere.rayleigh_scattering.r / atmosphere.mie_scattering.r) *
        (atmosphere.mie_scattering / atmosphere.rayleigh_scattering);
}

float3 GetCombinedScattering(AtmosphereParameters atmosphere,
                             float r, float mu, float mu_s, float nu,
                             bool ray_r_mu_intersects_ground,
                             out float3 single_mie_scattering) {
    float4 uvwz = GetScatteringTextureUvwzFromRMuMuSNu(
        atmosphere, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
    float tex_coord_x = uvwz.x * float(SCATTERING_TEXTURE_NU_SIZE - 1);
    float tex_x = floor(tex_coord_x);
    float lerp = tex_coord_x - tex_x;
    float3 uvw0 = float3((tex_x + uvwz.y) / float(SCATTERING_TEXTURE_NU_SIZE),
        uvwz.z, uvwz.w);
    float3 uvw1 = float3((tex_x + 1.0 + uvwz.y) / float(SCATTERING_TEXTURE_NU_SIZE),
        uvwz.z, uvwz.w);
    float4 combined_scattering =
        scattering_texture.SampleLevel(linear_sampler, uvw0, 0) * (1.0 - lerp) +
        scattering_texture.SampleLevel(linear_sampler, uvw1, 0) * lerp;
    single_mie_scattering = GetExtrapolatedSingleMieScattering(atmosphere, combined_scattering);
    return combined_scattering.rgb;
}

float RayleighPhaseFunction(float nu) {
    float k = 3.0 / (16.0 * kPi);
    return k * (1.0 + nu * nu);
}

float MiePhaseFunction(float g, float nu) {
    float k = 3.0 / (8.0 * kPi) * (1.0 - g * g) / (2.0 + g * g);
    return k * (1.0 + nu * nu) / pow(1.0 + g * g - 2.0 * g * nu, 1.5);
}

float3 GetSkyRadiance(AtmosphereParameters atmosphere,
                      float3 camera, float3 view_ray, float shadow_length,
                      float3 sun_direction, out float3 transmittance) {
    float r = length(camera);
    float rmu = dot(camera, view_ray);
    float distance_to_top_atmosphere_boundary = -rmu -
        sqrt(rmu * rmu - r * r + atmosphere.top_radius * atmosphere.top_radius);
    if (distance_to_top_atmosphere_boundary > 0.0) {
        camera = camera + view_ray * distance_to_top_atmosphere_boundary;
        r = atmosphere.top_radius;
        rmu += distance_to_top_atmosphere_boundary;
    } else if (r > atmosphere.top_radius) {
        transmittance = float3(1.0);
        return float3(0.0);
    }
    float mu = rmu / r;
    float mu_s = dot(camera, sun_direction) / r;
    float nu = dot(view_ray, sun_direction);
    bool ray_r_mu_intersects_ground = RayIntersectsGround(atmosphere, r, mu);
    transmittance = ray_r_mu_intersects_ground ? float3(0.0) :
        GetTransmittanceToTopAtmosphereBoundary(atmosphere, r, mu);
    float3 single_mie_scattering;
    float3 scattering;
    if (shadow_length == 0.0) {
        scattering = GetCombinedScattering(
            atmosphere, r, mu, mu_s, nu, ray_r_mu_intersects_ground,
            single_mie_scattering);
    } else {
        float d = shadow_length;
        float r_p = ClampRadius(atmosphere, sqrt(d * d + 2.0 * r * mu * d + r * r));
        float mu_p = (r * mu + d) / r_p;
        float mu_s_p = (r * mu_s + d * nu) / r_p;
        scattering = GetCombinedScattering(
            atmosphere, r_p, mu_p, mu_s_p, nu, ray_r_mu_intersects_ground,
            single_mie_scattering);
        float3 shadow_transmittance =
            GetTransmittance(atmosphere, r, mu, shadow_length, ray_r_mu_intersects_ground);
        scattering *= shadow_transmittance;
        single_mie_scattering *= shadow_transmittance;
    }
    return scattering * RayleighPhaseFunction(nu) +
        single_mie_scattering * MiePhaseFunction(atmosphere.mie_phase_function_g, nu);
}

float3 GetSolarRadiance(AtmosphereParameters atmosphere) {
    return atmosphere.solar_irradiance /
        (kPi * atmosphere.sun_angular_radius * atmosphere.sun_angular_radius);
}

struct VertexOutput {
    float4 position : SV_Position;
};

[shader("vertex")]
VertexOutput vertexMain(uint vertexID : SV_VertexID) {
    VertexOutput outV;
    float2 pos = float2(
        (vertexID == 2) ? 3.0 : -1.0,
        (vertexID == 1) ? 3.0 : -1.0);
    outV.position = float4(pos, 0.0, 1.0);
    return outV;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
    float2 pixel = input.position.xy;
    float2 ndc;
    ndc.x = (pixel.x + 0.5) / float(skyUniforms.screenWidth) * 2.0 - 1.0;
    ndc.y = 1.0 - (pixel.y + 0.5) / float(skyUniforms.screenHeight) * 2.0;
    float4 clip = float4(ndc, 1.0, 1.0);
    float4 worldPos4 = mul(skyUniforms.invViewProj, clip);
    float3 worldPos = worldPos4.xyz / worldPos4.w;
    float3 cameraWorld = skyUniforms.cameraWorldPos.xyz;
    float3 viewDir = normalize(worldPos - cameraWorld);

    float3 earthCenter = float3(0.0, -ATMOSPHERE.bottom_radius, 0.0);
    float3 camera = cameraWorld / kLengthUnitInMeters - earthCenter;
    float3 sunDirection = normalize(skyUniforms.sunDirection.xyz);

    float3 transmittance;
    float3 skyRadiance = GetSkyRadiance(ATMOSPHERE, camera, viewDir, 0.0,
        sunDirection, transmittance);

    float cosTheta = dot(viewDir, sunDirection);
    float sunCos = cos(ATMOSPHERE.sun_angular_radius);
    if (cosTheta > sunCos) {
        skyRadiance += transmittance * GetSolarRadiance(ATMOSPHERE);
    }

    float exposure = skyUniforms.params.x;
    float3 color = float3(1.0) - exp(-skyRadiance * exposure);
    return float4(color, 1.0);
}
