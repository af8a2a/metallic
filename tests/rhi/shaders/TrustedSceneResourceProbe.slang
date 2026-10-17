#language slang 2026
module TrustedSceneResourceProbe;

import Core;
using Metallic;

struct ProbeConstants
{
    ResourceHandle<ByteAddressBuffer> output;
    uint genericOnly;
    uint dynamicOffset;
    uint padding;
};

struct OversizedSceneField
{
    uint words[100];
};

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    let constants = getConstants<ProbeConstants>();
    let output = resolveBuffer<RWByteAddressBuffer>(constants.output);
    let checkedPacket = getResourceParameters<SceneResourceParameters>();
    output.Store(0, checkedPacket.materials.index);
    output.Store(4, getResourceParameterField<SceneResourceParameters, uint>(kSceneMaterialsOffset));
    output.Store(8, getResourceParameterField<SceneResourceParameters, uint>(380));
    // The raw-root short-packet cases exercise only the checked API, even in a
    // compilation that enables the separate trusted scene entry points.
    if (constants.genericOnly != 0) { return; }
    let packet = getSceneResourceParameters();
    output.Store(12, packet.materials.index);
    output.Store(16, getSceneResourceParameterField<uint>(kSceneMaterialsOffset));
    output.Store(20, getSceneResourceParameterField<uint>(380));
    output.Store(24, getSceneResourceParameterField<uint>(constants.dynamicOffset));
    output.Store(28, getSceneResourceParameterField<uint>(1));
    output.Store(32, getSceneResourceParameterField<uint>(384));
    output.Store(36, getSceneResourceParameterField<uint>(0xfffffffcu));
    output.Store(40, getSceneResourceParameterField<OversizedSceneField>(0).words[0]);
}
