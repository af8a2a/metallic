#include "Runtime/Debug/ShaderTraceCore.h"
#include "Runtime/Debug/DebugCore.h"
#include "Runtime/Debug/DebugHash.h"
#include <gtest/gtest.h>
#include <cmath>
#include <thread>

using namespace metallic::debug;
namespace {
DebugValue site()
{
    return shaderTraceSite({{"name","fixture.echo"},{"id",1},{"fixture",true},
        {"invocation",{{"group",{1,0,0}},{"localIndex",3}}},
        {"fields", {{{"name","negativeZero"},{"type","f32"}}, {{"name","nanPayload"},{"type","f32"}},
            {{"name","large"},{"type","u64"}}, {{"name","value"},{"type","u32"}}}}});
}
DebugValue watch()
{
    return {{"version",1},{"generation",1},{"target",{{"site","fixture.echo"},{"expectedSiteSchemaHash",site()["schemaHash"]}}},
        {"invocation",{{"group",{1,0,0}},{"localIndex",3}}},{"limits",{{"targetFrames",1},{"maxRecords",16},{"timeoutMs",30000}}}};
}
DebugValue identity()
{
    return {{"graph","graph"},{"generation",1},{"execution",987u},{"pass","Fixture"},{"phase","fixture"},{"dispatchOrdinal",0}};
}
DebugValue raw(const DebugValue& plan, uint32_t kind, uint32_t seq, std::vector<uint32_t> payload = {})
{
    std::string text = "VVL prefix\nMTS1";
    std::vector<uint32_t> words;
    for (const char* key : {"sessionToken","runToken","dispatchToken"}) {
        const auto value = debugUnsigned(plan.at(key));
        words.push_back(uint32_t(value)); words.push_back(uint32_t(value >> 32));
    }
    words.insert(words.end(), {1,kind,1,0,0,3,seq,uint32_t(payload.size())});
    words.insert(words.end(), payload.begin(), payload.end());
    for (auto word : words) { text += " " + std::to_string(word); }
    return {{"id",0x4fe1fef9},{"idName","VVL-DEBUG-PRINTF"},{"severity",16},{"text",text+"\n"},{"truncated",false}};
}
DebugValue bundle(uint32_t evaluated = 1, uint32_t matched = 1, uint32_t emitted = 1, bool quota = false,
    std::string session = "session", DebugValue specification = watch())
{
    ShaderTraceCore trace(std::move(session));
    const auto plan = trace.begin(std::move(specification), site(), identity()).value();
    trace.compiledVariant({{"compilerSpirvSha256","fixture"}});
    trace.submitted({{"family",0}},{{"timelineValue",7u}});
    // Deliberately reverse arrival order. Sequence, not callback time, defines ordering.
    trace.ingest(raw(plan,2,emitted+1,{evaluated,matched,emitted,0,quota ? 1u : 0u}));
    for (uint32_t i=0; i<emitted; ++i) { trace.ingest(raw(plan,1,i+1,{0x80000000,0x7fc12345,1,0x20000000,73})); }
    trace.ingest(raw(plan,0,0));
    trace.completion(true,true,true);
    return trace.seal().value();
}
DebugValue call(DebugCore& core, std::string method, DebugValue params = DebugValue::object())
{
    return core.dispatch({{"method",method},{"params",params}});
}
void configure(DebugCore& core)
{
    core.setGraph({{"id","graph"},{"generation",1}});
    core.configureShaderTrace({{"configured",true},{"smokeVerified",true}},DebugValue::array({site()}));
}
} // namespace

TEST(ShaderTrace, Sha256KnownVectors)
{
    EXPECT_EQ(debugSha256(""),"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    EXPECT_EQ(debugSha256("abc"),"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    EXPECT_EQ(debugSha256(std::string(1000000,'a')),"cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST(ShaderTrace, TypedWordsSurviveLosslessOfflineReplay)
{
    const auto b = bundle();
    const std::string text = encodeLossless(b).dump();
    std::vector<uint8_t> bytes(text.begin(),text.end());
    const auto decoded = decodeShaderTraceArtifact(bytes,debugSha256(std::span(bytes)));
    ASSERT_TRUE(decoded) << decoded.error().message;
    EXPECT_EQ(*decoded,analyzeShaderTrace(b).value());
    EXPECT_EQ((*decoded)["outcome"],"Matched"); EXPECT_EQ((*decoded)["selectedScopeComplete"],true);
    EXPECT_EQ((*decoded)["dispatch"]["execution"],987u);
    const auto& fields = (*decoded)["records"][0]["fields"];
    EXPECT_EQ(fields["negativeZero"]["bits"],0x80000000u);
    EXPECT_TRUE(std::signbit(fields["negativeZero"]["value"].get<double>()));
    EXPECT_EQ(fields["nanPayload"]["bits"],0x7fc12345u);
    EXPECT_EQ(fields["nanPayload"]["classification"],"NaN"); EXPECT_TRUE(fields["nanPayload"]["value"].is_null());
    EXPECT_EQ(fields["large"]["value"].get<uint64_t>(),2305843009213693953ull);
    EXPECT_EQ(fields["value"]["value"],73u); EXPECT_EQ((*decoded)["performanceEligible"],false);
    bytes.back() ^= 1; EXPECT_FALSE(decodeShaderTraceArtifact(bytes,debugSha256(text)));
}

TEST(ShaderTrace, NoMatchRequiresCompleteEndAndSiteEvaluation)
{
    EXPECT_EQ(analyzeShaderTrace(bundle(1,0,0))->at("outcome"),"NoMatch");
    EXPECT_EQ(analyzeShaderTrace(bundle(0,0,0))->at("outcome"),"SiteNotReached");
    auto b = bundle(); b["rawMessages"].erase(0);
    EXPECT_EQ(analyzeShaderTrace(b)->at("outcome"),"Incomplete");
    b["rawMessages"] = DebugValue::array();
    EXPECT_EQ(analyzeShaderTrace(b)->at("outcome"),"Incomplete");
    b["health"]["submitted"] = false;
    EXPECT_EQ(analyzeShaderTrace(b)->at("outcome"),"TargetNotExecuted");
    EXPECT_EQ(analyzeShaderTrace(bundle(15,15,14,true))->at("selectedScopeComplete"),false);
}

TEST(ShaderTrace, EveryLossOrMalformedRecordInvalidatesCompleteness)
{
    for (int fault=0; fault<11; ++fault) {
        SCOPED_TRACE(fault);
        auto b = bundle();
        if (fault == 0) { b["health"]["hostDropped"] = 1; }
        if (fault == 1) { b["health"]["hostTruncated"] = 1; b["rawMessages"][1]["truncated"] = true; }
        if (fault == 2) { b["rawMessages"].push_back(b["rawMessages"][1]); }
        if (fault == 3) { b["rawMessages"].erase(1); }
        if (fault == 4) { b["rawMessages"][1]["text"] = "MTS1 4294967296"; }
        if (fault == 5) { b["rawMessages"][1]["text"] = "MTS1 -1"; }
        if (fault == 6) { b["rawMessages"][1]["text"] = "[WARNING] Debug Printf message was truncated due to the buffer size (128)"; }
        if (fault == 7) { b["rawMessages"][1]["severity"] = 4096; }
        if (fault == 8) { b["health"]["gpuComplete"] = false; }
        if (fault == 9) { b["health"]["backendClosed"] = false; }
        if (fault == 10) { b["health"]["readbackValid"] = false; }
        const auto result = analyzeShaderTrace(b);
        ASSERT_TRUE(result); EXPECT_EQ(result->at("selectedScopeComplete"),false);
        if (fault == 1) { EXPECT_EQ(result->at("hostTruncatedCount"),1u); }
        if (fault == 6) { EXPECT_EQ(result->at("gpuOverflowDetected"),true); EXPECT_TRUE(result->at("gpuDroppedRecordCount").is_null()); }
    }
}

TEST(ShaderTrace, LateTokenIsOrphanAndCannotAcquireNewExecutionIdentity)
{
    auto b = bundle();
    auto old = b["dispatch"]; old["dispatchToken"] = 99;
    b["rawMessages"].push_back(raw(old,1,1,{0x80000000,0x7fc12345,1,0x20000000,73}));
    const auto decoded = analyzeShaderTrace(b);
    ASSERT_TRUE(decoded); EXPECT_EQ(decoded->at("orphanCount"),1u);
    EXPECT_EQ(decoded->at("records").size(),1u); EXPECT_EQ(decoded->at("selectedScopeComplete"),false);
    EXPECT_EQ(decoded->at("orphans")[0]["dispatchToken"],99u);
}

TEST(ShaderTrace, CancelledSubmissionRetainsOwnershipAndTokensNeverWrap)
{
    ShaderTraceCore trace("session");
    auto first = trace.begin(watch(),site(),identity()); ASSERT_TRUE(first);
    trace.submitted({},{}); trace.stop("Cancelled");
    trace.completion(true,false,true);
    EXPECT_EQ(trace.seal().error().code,"NotReady");
    EXPECT_EQ(trace.begin(watch(),site(),identity()).error().code,"Busy");
    EXPECT_THROW(trace.compiledVariant({}),std::logic_error);
    trace.completion(true,true,true);
    EXPECT_EQ(analyzeShaderTrace(trace.seal().value())->at("outcome"),"Cancelled");
    auto second = trace.begin(watch(),site(),identity()); ASSERT_TRUE(second);
    EXPECT_GT(debugUnsigned(second->at("dispatchToken")),debugUnsigned(first->at("dispatchToken")));
    ShaderTraceCore exhausted("session",UINT64_MAX);
    EXPECT_EQ(exhausted.begin(watch(),site(),identity()).error().code,"TokenExhausted");
}

TEST(ShaderTrace, SchemaAndWatchRejectStaleOrUnboundedRequests)
{
    auto w = watch(); w["generation"] = 2;
    EXPECT_EQ(validateShaderWatch(w,site(),1).error().code,"StaleHandle");
    w = watch(); w["target"]["expectedSiteSchemaHash"] = "wrong";
    EXPECT_EQ(validateShaderWatch(w,site(),1).error().code,"StaleHandle");
    w = watch(); w["limits"]["maxRecords"] = 17; EXPECT_FALSE(validateShaderWatch(w,site(),1));
    w = watch(); w["predicate"] = "anything";
    EXPECT_EQ(validateShaderWatch(w,site(),1).error().code,"Unsupported");
    w = watch(); w["target"]["pass"] = "UnrelatedProductionPass"; EXPECT_FALSE(validateShaderWatch(w,site(),1));
    auto b = bundle(); b["site"]["fields"][0]["type"] = "u32"; EXPECT_FALSE(analyzeShaderTrace(b));
}

TEST(ShaderTrace, DebugCoreRoutesAndCancellationExportsPartialEvidence)
{
    DebugCore core;
    EXPECT_EQ(call(core,"shader.watch",watch())["error"]["code"],"RestartRequired");
    configure(core);
    core.configureShaderTrace({{"configured",true},{"smokeVerified",false}},DebugValue::array({site()}));
    EXPECT_EQ(call(core,"shader.watch",watch())["error"]["code"],"BackendUnavailable");
    configure(core);
    const auto queued = call(core,"shader.watch",watch()); ASSERT_EQ(queued["status"],"ok");
    const auto id = queued["result"]["job"].get<std::string>();
    EXPECT_TRUE(core.takeRequests("graph",1).empty());
    ASSERT_EQ(core.takeShaderRequests("graph",1).size(),1u);
    ASSERT_TRUE(core.reserve(id,kShaderTraceArtifactBudget)); core.transition(id,"Submitted");
    call(core,"jobs.cancel",{{"job",id}});
    EXPECT_EQ(call(core,"shader.watch",watch())["error"]["code"],"Busy");
    core.completeShaderTrace(id,bundle(1,1,1,false,core.session()));
    const auto job = call(core,"jobs.get",{{"job",id}})["result"];
    EXPECT_EQ(job["state"],"Cancelled"); EXPECT_EQ(job["artifactCount"],1u);
    EXPECT_EQ(job["shaderTrace"]["outcome"],"Cancelled"); EXPECT_EQ(job["shaderTrace"]["selectedScopeComplete"],false);
    EXPECT_EQ(call(core,"artifact.read",{{"job",id},{"manifest",true}})["status"],"ok");
    EXPECT_EQ(call(core,"shader.watch",watch())["status"],"ok");
}

TEST(ShaderTrace, TimeoutAndGenerationChangesRetainCorrectJobEvidence)
{
    DebugCore core; configure(core);
    auto w = watch(); w["limits"]["timeoutMs"] = 1;
    const auto id = call(core,"shader.watch",w)["result"]["job"].get<std::string>();
    ASSERT_EQ(core.takeShaderRequests("graph",1).size(),1u); ASSERT_TRUE(core.reserve(id,kShaderTraceArtifactBudget));
    core.transition(id,"Submitted"); std::this_thread::sleep_for(std::chrono::milliseconds(4)); core.expire();
    EXPECT_EQ(call(core,"shader.watch",watch())["error"]["code"],"Busy");
    core.completeShaderTrace(id,bundle(1,1,1,false,core.session(),w));
    const auto job = call(core,"jobs.get",{{"job",id}})["result"];
    EXPECT_EQ(job["state"],"Cancelled"); EXPECT_EQ(job["shaderTrace"]["outcome"],"Timeout"); EXPECT_EQ(job["artifactCount"],1u);
    const auto next = call(core,"shader.watch",watch())["result"]["job"];
    core.setGraph({{"id","graph"},{"generation",2}});
    EXPECT_TRUE(core.takeShaderRequests("graph",2).empty());
    EXPECT_EQ(call(core,"jobs.get",{{"job",next}})["result"]["error"]["code"],"StaleHandle");
}

TEST(ShaderTrace, OfflineIgnoresCachedClaimsAndRejectsTamperedRaw)
{
    auto b = bundle(); b["rawMessages"].erase(0);
    const std::string text = encodeLossless(b).dump();
    DebugCapture capture;
    capture.snapshot.values = {{"shaderTrace",{{"selectedScopeComplete",true}}}, {"shaderTraceCompletion",{{"selectedScopeComplete",true}}}};
    DebugArtifact artifact;
    artifact.bytes.assign(text.begin(),text.end());
    artifact.metadata = {{"kind","shader-trace-v1"},{"sha256",debugSha256(text)}};
    capture.artifacts.push_back(artifact);
    auto decoded = capture.evaluationRoot(); ASSERT_TRUE(decoded);
    EXPECT_EQ((*decoded)["shaderTrace"]["selectedScopeComplete"],false);
    EXPECT_EQ((*decoded)["shaderTraceCompletion"]["selectedScopeComplete"],false);
    capture.artifacts[0].bytes.back() ^= 1; EXPECT_FALSE(capture.evaluationRoot());
    capture.artifacts = {artifact,artifact}; EXPECT_FALSE(capture.evaluationRoot());
}

TEST(ShaderTrace, WrongScopeAndInconsistentEndAreNeverComplete)
{
    for (int fault=0; fault<4; ++fault) {
        auto b = bundle();
        if (fault == 0) { b["rawMessages"][0] = raw(b["dispatch"],2,2,{0,1,1,0,0}); }
        if (fault == 1) { b["rawMessages"][0] = raw(b["dispatch"],2,2,{1,1,0,0,0}); }
        if (fault == 2) { b["rawMessages"][1] = raw(b["dispatch"],1,16,{0x80000000,0x7fc12345,1,0x20000000,73}); }
        if (fault == 3) {
            auto text = b["rawMessages"][1]["text"].get<std::string>();
            const auto pos = text.find(" 1 1 1 0 0 3 1 5 "); ASSERT_NE(pos,std::string::npos);
            text.replace(pos,17," 1 1 2 0 0 3 1 5 "); b["rawMessages"][1]["text"] = text;
        }
        auto decoded = analyzeShaderTrace(b); ASSERT_TRUE(decoded);
        EXPECT_EQ(decoded->at("selectedScopeComplete"),false);
    }
}

TEST(ShaderTrace, HostArtifactQueueIsBoundedAndAccountsDroppedRecords)
{
    ShaderTraceCore trace("session");
    auto plan = trace.begin(watch(),site(),identity()).value();
    auto message = raw(plan,0,0);
    for (int i=0; i<260; ++i) { trace.ingest(message); }
    auto b = trace.seal().value();
    EXPECT_EQ(b["rawMessages"].size(),256u); EXPECT_EQ(b["health"]["hostDropped"],4u);
    EXPECT_EQ(analyzeShaderTrace(b)->at("selectedScopeComplete"),false);
}

TEST(ShaderTrace, WorkControlSelectorsAndPhaseAreBounded)
{
    auto production = site();
    production.erase("fixture"); production["name"]="stream.after-triangle-prepare";
    production["adapter"]="work-control-v1"; production["phase"]="early";
    production=shaderTraceSite(production);
    auto request=watch(); request["target"]={{"site",production["name"]},{"phase","early"},
        {"expectedSiteSchemaHash",production["schemaHash"]}};
    request["invocation"]={{"group",{0,0,0}},{"localIndex",127}};
    request["predicate"]={{"field","triangleId"},{"op","eq"},{"value",UINT32_MAX}};
    ASSERT_TRUE(validateShaderWatch(request,production,1));
    for (int fault=0;fault<9;++fault) {
        auto invalid=request;
        if (fault==0) { invalid["target"]["phase"]="late"; }
        if (fault==1) { invalid["target"].erase("phase"); }
        if (fault==2) { invalid["invocation"]["group"]={65535,0,0}; }
        if (fault==3) { invalid["invocation"]["group"]={0,1,0}; }
        if (fault==4) { invalid["invocation"]["localIndex"]=128; }
        if (fault==5) { invalid["predicate"]["value"]=-1; }
        if (fault==6) { invalid["predicate"]["op"]="expression"; }
        if (fault==7) { invalid["predicate"]["expression"]="anything"; }
        if (fault==8) { invalid["fixtureScenario"]="site-not-reached"; }
        EXPECT_FALSE(validateShaderWatch(invalid,production,1)) << fault;
    }
}

TEST(ShaderTrace, PhaseRegistrySelectsTheMatchingSchema)
{
    DebugCore core; core.setGraph({{"id","graph"},{"generation",1}});
    auto early=site(); early.erase("fixture"); early["adapter"]="work-control-v1"; early["phase"]="early";
    early=shaderTraceSite(early);
    auto late=early; late["phase"]="late"; late=shaderTraceSite(late);
    ASSERT_NE(early["schemaHash"],late["schemaHash"]);
    core.configureShaderTrace({{"configured",true},{"smokeVerified",true}},DebugValue::array({early,late}));
    auto request=watch(); request["target"]["phase"]="late";
    request["target"]["expectedSiteSchemaHash"]=early["schemaHash"];
    EXPECT_EQ(call(core,"shader.watch",request)["error"]["code"],"StaleHandle");
    request["target"]["expectedSiteSchemaHash"]=late["schemaHash"];
    ASSERT_EQ(call(core,"shader.watch",request)["status"],"ok");
    const auto queued=core.takeShaderRequests("graph",1);
    ASSERT_EQ(queued.size(),1u);
}

TEST(ShaderTrace, ActualRecordingIdentityFreezesAtSubmission)
{
    ShaderTraceCore trace("session");
    ASSERT_TRUE(trace.begin(watch(),site(),identity()));
    trace.recorded({{"execution",888u},{"commandBufferRecording",999u},{"frameSlot",2u},
        {"dispatchToken",77u},{"generation",9u}});
    trace.submitted({{"family",0}},{{"scope","tracked frame completion"}});
    EXPECT_THROW(trace.recorded({{"execution",123u}}),std::logic_error);
    trace.completion(true,true,true);
    const auto b=trace.seal().value();
    EXPECT_EQ(b["dispatch"]["execution"],888u); EXPECT_EQ(b["dispatch"]["commandBufferRecording"],999u);
    EXPECT_EQ(b["dispatch"]["frameSlot"],2u); EXPECT_EQ(b["dispatch"]["dispatchToken"],1u);
    EXPECT_EQ(b["dispatch"]["generation"],1u);
}

TEST(ShaderTrace, ProductionFallbackCannotBecomeAnEmptySuccess)
{
    for (uint32_t reason : {1u,2u}) {
        auto b=bundle(0,0,0);
        auto production=b["site"]; production.erase("fixture");
        production["adapter"]="work-control-v1"; production["phase"]="early";
        b["site"]=shaderTraceSite(production);
        b["request"]["target"]["phase"]="early";
        b["request"]["target"]["expectedSiteSchemaHash"]=b["site"]["schemaHash"];
        b["rawMessages"][0]=raw(b["dispatch"],2,1,{0,0,0,reason,0});
        const auto decoded=analyzeShaderTrace(b); ASSERT_TRUE(decoded);
        EXPECT_FALSE(decoded->at("selectedScopeComplete").get<bool>());
        EXPECT_EQ(decoded->at("outcome"),reason==1 ? "UnsupportedPath" : "Incomplete");
    }
}
