// Qualifies the callback path on the same production Vulkan instance.
[shader("compute")]
[numthreads(1, 1, 1)]
void echoMain()
{
    printf("MTQ1 backend-ready\n");
}
