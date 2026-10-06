import MaterialAuthoringSDK;
Material evaluate()
{
    let top = slab(parameter(0), parameter(1));
    let base = slab(float4(0.12, 0.35, 0.65, 0.0), 0.0);
    return layer(top, base);
}
