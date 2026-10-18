module TextureResidencyProbe;

import Core;
import StreamTextureSampling;
using Metallic;
struct Push { uint slot; uint wantedMip; uint samples; uint unused; };
[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    Push push = getConstants<Push>();
    let resources = getResourceParameters<TextureFeedbackResourceParameters>();
    StructuredBuffer<TextureSamplingMetadata> metadata = resolveBuffer<StructuredBuffer<TextureSamplingMetadata>>(resources.samplingMetadata);
    RWStructuredBuffer<uint> demandMip = resolveBuffer<RWStructuredBuffer<uint>>(resources.demandMip);
    RWStructuredBuffer<uint> demandHits = resolveBuffer<RWStructuredBuffer<uint>>(resources.demandHits);
    if ((metadata[push.slot].residentFirstMipAndFlags & 0x80000000u) != 0u && push.samples != 0u) {
        uint ignored;
        InterlockedMin(demandMip[push.slot], push.wantedMip, ignored);
        InterlockedAdd(demandHits[push.slot], push.samples, ignored);
    }
}
