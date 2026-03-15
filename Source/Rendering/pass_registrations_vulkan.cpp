#ifdef _WIN32

#include "pass_registry.h"
#include "output_pass.h"

REGISTER_RENDER_PASS(OutputPass, "Output", "Utility",
    (std::vector<std::string>{"source"}),
    (std::vector<std::string>{"$backbuffer"}));

#endif
