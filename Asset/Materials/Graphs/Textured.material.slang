import MaterialAuthoringSDK;
Material evaluate()
{
    let color = sampleBaseColor(uv(), 0.0);
    let packed = sampleMetallicRoughness(uv(), 0.0);
    let normal = normalMap(sampleNormal(uv(), 0.0));
    return withNormal(openPBR(color, z(packed), y(packed)), normal);
}
