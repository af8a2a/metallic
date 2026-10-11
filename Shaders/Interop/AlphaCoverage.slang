#language slang 2026
module AlphaCoverage;

// Deterministic coverage at the finest resident mip; shared by raster and RT.
public float sampleAlphaCoverage(Texture2D<float4> texture, float2 uv, uint mip = 0u)
{
    uint width = 1u;
    uint height = 1u;
    uint mipCount;
    texture.GetDimensions(mip, width, height, mipCount);
    float2 texel = frac(uv) * float2(width, height) - 0.5;
    int2 baseTexel = int2(floor(texel));
    float2 fraction = frac(texel);
    int x0 = (baseTexel.x % int(width) + int(width)) % int(width);
    int y0 = (baseTexel.y % int(height) + int(height)) % int(height);
    int x1 = (x0 + 1) % int(width);
    int y1 = (y0 + 1) % int(height);
    float4 v00 = texture.Load(int3(x0, y0, mip));
    float4 v10 = texture.Load(int3(x1, y0, mip));
    float4 v01 = texture.Load(int3(x0, y1, mip));
    float4 v11 = texture.Load(int3(x1, y1, mip));
    return lerp(lerp(v00, v10, fraction.x), lerp(v01, v11, fraction.x), fraction.y).a;
}

