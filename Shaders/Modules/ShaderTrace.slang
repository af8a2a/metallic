// MTS1 wire v1. Fixed unsigned words only; f32 uses asuint, u64 is low/high.
// Do not change production control flow or barriers to enforce a logging quota.
public struct ShaderTraceIdentity
{
    public uint2 session;
    public uint2 run;
    public uint2 dispatch;
    public uint site;
    public uint3 group;
    public uint localIndex;
};

public void shaderTraceBegin(ShaderTraceIdentity id)
{
    printf("MTS1 %u %u %u %u %u %u %u 0 %u %u %u %u 0 0\n",
        id.session.x, id.session.y, id.run.x, id.run.y, id.dispatch.x, id.dispatch.y,
        id.site, id.group.x, id.group.y, id.group.z, id.localIndex);
}

public void shaderTraceWords5(ShaderTraceIdentity id, uint kind, uint seq, uint4 words, uint last)
{
    printf("MTS1 %u %u %u %u %u %u %u %u %u %u %u %u %u 5 %u %u %u %u %u\n",
        id.session.x, id.session.y, id.run.x, id.run.y, id.dispatch.x, id.dispatch.y,
        id.site, kind, id.group.x, id.group.y, id.group.z, id.localIndex, seq,
        words.x, words.y, words.z, words.w, last);
}
