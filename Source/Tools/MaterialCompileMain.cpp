#include "Runtime/Material/MaterialGraph.h"
#include <filesystem>
#include <fstream>
#include <iostream>

int main(int argc, char** argv)
{
    using namespace metallic::material;
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: MetallicMaterialCompile <file.materialgraph|file.material.slang> <output.materialdef> [parameter-defaults.json]\n";
        return 2;
    }
    try {
        const auto read=[](const std::filesystem::path& path) {
            std::ifstream file(path,std::ios::binary);
            if (!file || std::filesystem::file_size(path)>262144) { throw std::runtime_error("Cannot read input (256 KiB limit): "+path.string()); }
            return std::string{std::istreambuf_iterator<char>(file),std::istreambuf_iterator<char>()};
        };
        const auto source=read(argv[1]);
        const auto parameters=argc==4 ? nlohmann::json::parse(read(argv[3])) : nlohmann::json::object();
        const bool graph=std::filesystem::path(argv[1]).extension()==".materialgraph";
        CompiledMaterialFrontend compiled; std::string error;
        const bool ok=graph ? compileMaterialGraph(nlohmann::json::parse(source),compiled,error) : compileSlangMaterial(source,parameters,compiled,error);
        if (!ok) { std::cerr<<error<<'\n'; return 1; }
        const std::filesystem::path output=argv[2];
        if (output.extension()!=".materialdef") { throw std::runtime_error("Output must have .materialdef extension"); }
        if (!output.parent_path().empty()) { std::filesystem::create_directories(output.parent_path()); }
        const auto write=[](const auto& path,const std::string& text) {
            std::ofstream file(path,std::ios::binary); file<<text; file.close();
            if (!file) { throw std::runtime_error("Cannot write output"); }
        };
        write(output,serializeMaterialDefinition(compiled.definition));
        auto instancePath=output; instancePath.replace_extension(".material");
        MaterialInstance instance; instance.definition="asset://"+output.filename().generic_string();
        write(instancePath,serializeMaterialInstance(instance));
        auto reflectionPath=output; reflectionPath.replace_extension(".reflection.json");
        write(reflectionPath,compiled.reflection.dump(2));
        std::cout<<compiled.reflection.dump(2)<<'\n'; return 0;
    } catch (const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
