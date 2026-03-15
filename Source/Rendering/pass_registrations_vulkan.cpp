#ifdef _WIN32

#include "forward_pass.h"
#include "pass_registry.h"
#include "output_pass.h"
#include "sky_pass.h"
#include "tonemap_pass.h"

REGISTER_RENDER_PASS(OutputPass, "Output", "Utility",
    (std::vector<std::string>{"source"}),
    (std::vector<std::string>{"$backbuffer"}));

REGISTER_RENDER_PASS(SkyPass, "Sky Pass", "Environment",
    (std::vector<std::string>{}),
    (std::vector<std::string>{"skyOutput"}));

REGISTER_RENDER_PASS(ForwardPass, "Forward Pass", "Geometry",
    (std::vector<std::string>{"skyOutput"}),
    (std::vector<std::string>{"forwardColor", "depth"}));

REGISTER_RENDER_PASS(TonemapPass, "Tonemap", "Post-Process",
    (std::vector<std::string>{"lightingOutput"}),
    (std::vector<std::string>{"$backbuffer"}));

#endif
