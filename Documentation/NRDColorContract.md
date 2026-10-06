# NRD color contract

The vendor frontend and denoiser shaders retain their linear Rec.709 numeric
luminance/packing assumptions. No vendor YCoCg or luminance coefficients change.

RTXDI computes radiance and demodulated lighting signals in the scene working
space. `NRDEncoding.packNrdRadianceHitDistance` converts the RGB signal to linear
Rec.709/D65 before RELAX frontend packing. The fourth channel keeps its original
hit-distance meaning. NRD receives and emits these Rec.709 lighting signals.
`RTXDIComposite` converts each output signal back to working RGB, then restores
the working-space diffuse/specular albedos and adds working-space emission.
`RTXDIConfidence` uses the same reconstructed basis for luminance/history.

There is no additional exposure in the adapter; scene exposure remains at the
existing graph boundary. Normals/roughness, material IDs, depth, motion, history
confidence and SIGMA penumbra are Data and never undergo gamut conversion.
The REFERENCE denoiser uses same-basis averaging and can retain working RGB.

Rec.709 compatibility makes both adapter transforms identities. Native AP1
colors outside Rec.709 can produce negative SDK components; vendor sanitization
may clip them. This is a known boundary limitation, requiring native-working-space
NRD specialization or an explicit gamut policy for a fully wide-gamut SDK path.
Denoiser runtime/chromaticity and temporal tests must be reported independently
of the frontend adapter's numeric tests.
