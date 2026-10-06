#include "Editor/EditorApplication.h"
#include "Runtime/Render/Material/MaterialValueProgram.h"
#include "imgui.h"
#include "imgui_internal.h"
#include <fstream>
#include <spdlog/spdlog.h>

namespace metallic {
bool EditorApplication::runMaterialGraphSmokeTest()
{
    const auto expect = [](bool condition, const char* message) {
        if (!condition) { spdlog::error("[Smoke Material Graph] {}",message); }
        return condition;
    };
    std::error_code pathError;
    if (!expect(std::filesystem::equivalent(scene_.sourcePath().parent_path(),std::filesystem::current_path()/"scene",pathError)
        && !pathError,"Use isolated CTest scene")) { return false; }
    if (!expect(scene_.valid() && !scene_.materials().empty(),"Editable scene required")) { return false; }
    const auto before=scene_.materials()[0];
    sceneSelection_ = SceneSelection{.type=SceneSelectionType::Material,.sceneLifetimeRevision=scene_.sceneGraph().lifetimeRevision(),.index=0};
    materialGraphEditor_.open=true;
    if (!renderFrame() || !expect(viewportPreviewValid_,"Graph canvas and original comparison rendered")) { return false; }
    auto graph=material::defaultMaterialGraph(); graph["nodes"][0]["default"]={.12,.45,.7,1.};
    materialGraphEditor_.replaceGraph(graph);
    // Exercise the actual Apply button with ImGui input, not a production test hook.
    ImRect applyRect;
    const auto graphFrame = [&](ImVec2 mouse, bool down, bool locate = false) {
        auto& io=ImGui::GetIO(); io.AddMousePosEvent(mouse.x,mouse.y);
        io.AddMouseButtonEvent(ImGuiMouseButton_Left,down); io.DeltaTime=1.0f/60.0f;
        ImGui::NewFrame();
        auto* window=ImGui::FindWindowByName("Material Graph");
        if (locate && window) {
            ImGui::FocusWindow(window);
            ImGui::SetNavID(window->GetID("Apply to selected material"),ImGuiNavLayer_Main,window->ID,ImRect());
        }
        ImGui::SetNextWindowPos(ImGui::GetMainViewport()->WorkPos);
        ImGui::SetNextWindowSize({1120,700});
        drawMaterialGraphEditor();
        if (locate && window && GImGui->NavIdIsAlive) { applyRect=ImGui::WindowRectRelToAbs(window,window->NavRectRel[ImGuiNavLayer_Main]); }
        ImGui::Render(); ImGui::UpdatePlatformWindows();
    };
    graphFrame({-100,-100},false);
    graphFrame({-100,-100},false,true);
    if (!expect(applyRect.GetWidth()>0 && applyRect.GetHeight()>0,"Locate real Graph Apply control")) { return false; }
    graphFrame(applyRect.GetCenter(),false); graphFrame(applyRect.GetCenter(),true); graphFrame(applyRect.GetCenter(),false);
    if (!expect(!scene_.materials()[0].valueProgram.empty() && scene_.materials()[0].valueProgram==materialGraphEditor_.compiled().definition.surfaceProgram,"Apply button publishes compiled Graph") ||
        !renderFrame() || !expect(viewportPreviewValid_,"Compiled OpenPBR graph renders")) { return false; }
    const auto after=scene_.materials()[0];
    undoTransform();
    if (!expect(scene::materialPropertiesEqual(scene_.materials()[0],before),"Undo restores original program and parameters")) { return false; }
    redoTransform();
    if (!expect(scene::materialPropertiesEqual(scene_.materials()[0],after),"Redo restores graph program")) { return false; }
    const auto root=std::filesystem::current_path()/"graph";
    if (!expect(materialGraphEditor_.save(root/"Authoring.materialgraph"),"Save graph") ||
        !expect(materialGraphEditor_.load(root/"Authoring.materialgraph"),"Load graph") ||
        !expect(materialGraphEditor_.graph()==graph,"Graph round trip") ||
        !expect(materialGraphEditor_.exportAssets(root/"Authoring.materialdef"),"Export definition and instance")) { return false; }
    auto malformed=graph; malformed["nodes"][0]["default"]={"bad",1,2,3};
    { std::ofstream file(root/"Malformed.materialgraph"); file<<malformed.dump(); }
    if (!expect(!materialGraphEditor_.load(root/"Malformed.materialgraph") && materialGraphEditor_.graph()==graph,"Malformed graph load preserves draft")) { return false; }
    std::string error;
    material::ResolvedMaterialInstance resolved;
    if (!expect(material::MaterialAssetLibrary(root).resolve("asset://Authoring.material",resolved,error),error.c_str()) ||
        !expect(resolved.definition.surfaceProgram==after.valueProgram,"Export uses applied IR") ||
        !expect(scene_.save(error),error.c_str())) { return false; }
    scene::SceneDocument reloaded;
    if (!expect(reloaded.load(scene_.documentPath()),"Reload applied graph scene") ||
        !expect(reloaded.materials()[0].valueProgram==after.valueProgram,"Saved scene preserves graph") ||
        !expect(reloaded.materials()[0].valueParameters==after.valueParameters,"Saved scene preserves instance slots")) { return false; }
    graph=material::defaultMaterialGraph(true); graph["nodes"][0]["default"]={.7,.15,.05,0.};
    materialGraphEditor_.replaceGraph(graph);
    if (!expect(materialGraphEditor_.compile(),materialGraphEditor_.status().c_str()) ||
        !expect(applyMaterialGraph(0,materialGraphEditor_.compiled()),"Apply Slab graph") ||
        !renderFrame() || !expect(viewportPreviewValid_,"Slab graph comparison renders")) { return false; }
    graph["nodes"][2]["inputs"]["surface"]={{"node",1}};
    materialGraphEditor_.replaceGraph(graph);
    if (!expect(!materialGraphEditor_.compile(),"Invalid graph fails without replacing live scene") ||
        !renderFrame() || !expect(viewportPreviewValid_,"Last good scene remains renderable")) { return false; }
    spdlog::info("[Smoke Material Graph] Passed canvas draw, real Apply click, OpenPBR/Slab apply, undo/redo, graph/asset/scene round trip, invalid-load and failed-compile recovery");
    return true;
}
} // namespace metallic
