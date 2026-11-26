/*
 * T1 smoke test: call dxmt's winemetal.dll from a plain PE program under Wine.
 *
 *   device       -> WMTCopyAllDevices / NSArray_object
 *   zero copy    -> VirtualAlloc(1 MiB) wrapped with newBufferWithBytesNoCopy
 *   compute      -> MSL source compiled at runtime, one dispatch writes a
 *                   known value into the Windows-owned memory
 *   readback     -> checked on the Windows side, no copy involved
 */
#include <stdio.h>
#include <string.h>
#include <windows.h>
#include "winemetal.h"
#include "airconv_public.h"

#define BUF_SIZE (1u << 20)      /* 1 MiB */
#define N_THREADS 65536u         /* 1024 threadgroups x 64 threads */
#define FILL_PATTERN 0xCCCCCCCCu /* canary in the untouched part */

static int failures = 0;

static void ok(const char *what) { printf("[ok]   %s\n", what); }
static void fail(const char *what, const char *detail)
{
    printf("[FAIL] %s%s%s\n", what, detail ? ": " : "", detail ? detail : "");
    failures++;
}

static void handle_to_text(obj_handle_t h, char *buf, uint64_t len)
{
    buf[0] = 0;
    if (!h) return;
    obj_handle_t desc = NSObject_description(h);
    if (!desc) return;
    NSString_getCString(desc, buf, len, WMTUTF8StringEncoding);
    NSObject_release(desc);
}

int main(void)
{
    char text[1024];

    printf("victorymetal smoke test\n");
    obj_handle_t pool = NSAutoreleasePool_alloc_init();

    /* ---- device ---------------------------------------------------------- */
    obj_handle_t devices = WMTCopyAllDevices();
    if (!devices) { fail("WMTCopyAllDevices", "null array"); return 1; }
    uint64_t ndev = NSArray_count(devices);
    printf("       devices: %llu\n", (unsigned long long)ndev);
    if (!ndev) { fail("no Metal device", NULL); return 1; }

    obj_handle_t device = NSArray_object(devices, 0);
    if (!device) { fail("NSArray_object(0)", NULL); return 1; }

    handle_to_text(MTLDevice_name(device), text, sizeof(text));
    printf("       device[0]: %s\n", text);
    printf("       unified memory: %d, gpu family apple7: %d\n",
           (int)MTLDevice_hasUnifiedMemory(device), (int)MTLDevice_supportsFamily(device, WMTGPUFamilyApple7));
    printf("       recommended working set: %llu MiB\n",
           (unsigned long long)(MTLDevice_recommendedMaxWorkingSetSize(device) >> 20));
    ok("MTLDevice obtained through the unixlib");

    /* ---- zero-copy buffer ------------------------------------------------ */
    void *mem = VirtualAlloc(NULL, BUF_SIZE, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!mem) { fail("VirtualAlloc(1 MiB)", NULL); return 1; }
    memset(mem, FILL_PATTERN, BUF_SIZE);

    struct WMTBufferInfo info;
    memset(&info, 0, sizeof(info));
    info.length = BUF_SIZE;
    info.options = WMTResourceStorageModeShared;
    info.memory.ptr = mem;

    obj_handle_t buffer = MTLDevice_newBuffer(device, &info);
    if (!buffer) { fail("MTLDevice_newBuffer(noCopy)", NULL); return 1; }
    printf("       cpu addr: %p  gpu addr: 0x%llx  (gpu-cpu delta: 0x%llx)\n",
           mem, (unsigned long long)info.gpu_address,
           (unsigned long long)(info.gpu_address - (uint64_t)(uintptr_t)mem));
    if (info.gpu_address == (uint64_t)(uintptr_t)mem)
        printf("       note: gpu_address equals the cpu address\n");
    ok("VirtualAlloc memory wrapped with newBufferWithBytesNoCopy");

    /* ---- shader ---------------------------------------------------------- */
    static const char *msl =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void t1fill(device uint *out [[buffer(0)]], uint gid [[thread_position_in_grid]]) {\n"
        "    out[gid] = gid ^ 0xDEADBEEFu;\n"
        "}\n";

    obj_handle_t err = 0;
    obj_handle_t library = MTLDevice_newLibraryWithSource(device, msl, strlen(msl), &err);
    if (!library) {
        handle_to_text(err, text, sizeof(text));
        fail("newLibraryWithSource", text);
        return 1;
    }
    ok("MSL source compiled at runtime (newLibraryWithSource)");

    obj_handle_t function = MTLLibrary_newFunction(library, "t1fill");
    if (!function) { fail("MTLLibrary_newFunction(t1fill)", NULL); return 1; }

    struct WMTComputePipelineInfo pinfo;
    memset(&pinfo, 0, sizeof(pinfo));
    pinfo.compute_function = function;
    err = 0;
    obj_handle_t pso = MTLDevice_newComputePipelineState(device, &pinfo, &err);
    if (!pso) {
        handle_to_text(err, text, sizeof(text));
        fail("newComputePipelineState", text);
        return 1;
    }
    ok("compute pipeline state created");

    /* ---- command buffer -------------------------------------------------- */
    obj_handle_t queue = MTLDevice_newCommandQueue(device, 4);
    if (!queue) { fail("newCommandQueue", NULL); return 1; }
    obj_handle_t cmdbuf = MTLCommandQueue_commandBuffer(queue);
    if (!cmdbuf) { fail("commandBuffer", NULL); return 1; }
    obj_handle_t encoder = MTLCommandBuffer_computeCommandEncoder(cmdbuf, false);
    if (!encoder) { fail("computeCommandEncoder", NULL); return 1; }

    /* command chain, lives in ordinary PE memory: PE and unix share one
       address space, which is why winemetal can pass raw pointers through */
    static struct wmtcmd_compute_setpso cmd_pso;
    static struct wmtcmd_compute_setbuffer cmd_buf;
    static struct wmtcmd_compute_dispatch cmd_dispatch;

    memset(&cmd_pso, 0, sizeof(cmd_pso));
    cmd_pso.type = WMTComputeCommandSetPSO;
    cmd_pso.pso = pso;
    cmd_pso.threadgroup_size.width = 64;
    cmd_pso.threadgroup_size.height = 1;
    cmd_pso.threadgroup_size.depth = 1;
    cmd_pso.next.ptr = &cmd_buf;

    memset(&cmd_buf, 0, sizeof(cmd_buf));
    cmd_buf.type = WMTComputeCommandSetBuffer;
    cmd_buf.buffer = buffer;
    cmd_buf.offset = 0;
    cmd_buf.index = 0;
    cmd_buf.next.ptr = &cmd_dispatch;

    memset(&cmd_dispatch, 0, sizeof(cmd_dispatch));
    cmd_dispatch.type = WMTComputeCommandDispatch;
    cmd_dispatch.size.width = N_THREADS / 64;
    cmd_dispatch.size.height = 1;
    cmd_dispatch.size.depth = 1;
    cmd_dispatch.next.ptr = NULL;

    MTLComputeCommandEncoder_encodeCommands(encoder, (const struct wmtcmd_base *)&cmd_pso);
    MTLCommandEncoder_endEncoding(encoder);
    MTLCommandBuffer_commit(cmdbuf);
    MTLCommandBuffer_waitUntilCompleted(cmdbuf);

    enum WMTCommandBufferStatus status = MTLCommandBuffer_status(cmdbuf);
    printf("       command buffer status: %d (4 = completed)\n", (int)status);
    if (status != WMTCommandBufferStatusCompleted) {
        handle_to_text(MTLCommandBuffer_error(cmdbuf), text, sizeof(text));
        fail("command buffer did not complete", text);
        return 1;
    }
    ok("compute kernel committed and completed (GPU fence waited)");

    /* ---- verify on the Windows side -------------------------------------- */
    uint32_t *words = (uint32_t *)mem;
    unsigned bad = 0;
    for (unsigned i = 0; i < N_THREADS; i++) {
        if (words[i] != (i ^ 0xDEADBEEFu)) {
            if (!bad) printf("       first mismatch at %u: got 0x%08x want 0x%08x\n",
                             i, words[i], i ^ 0xDEADBEEFu);
            bad++;
        }
    }
    unsigned canary_bad = 0;
    for (unsigned i = N_THREADS; i < BUF_SIZE / 4; i++)
        if (words[i] != FILL_PATTERN) canary_bad++;

    printf("       %u/%u words written by the GPU are correct, %u mismatches, "
           "%u canary words changed outside the dispatch\n",
           N_THREADS - bad, N_THREADS, bad, canary_bad);
    if (bad) fail("GPU results wrong", NULL);
    else if (canary_bad) fail("dispatch wrote outside its range", NULL);
    else ok("GPU wrote the known value straight into VirtualAlloc memory (zero copy)");

    /* the second export family: airconv (DXBC -> AIR).  Stubbed out in this
       build, but the call must still reach the unix side (unix_call 74). */
    {
        sm50_shader_t shader = 0;
        sm50_error_t shader_err = 0;
        int rc = SM50Initialize(NULL, 0, &shader, NULL, &shader_err);
        printf("       SM50Initialize(NULL) -> %d (stub returns -1)\n", rc);
        if (rc < 0) ok("airconv thunk reaches the unix side (SM50 is stubbed)");
        else fail("SM50Initialize stub did not report failure", NULL);
    }

    NSObject_release(buffer);
    NSObject_release(pso);
    if (pool) NSObject_release(pool);

    printf("%s\n", failures ? "RESULT: FAILED" : "RESULT: ALL CHECKS PASSED");
    return failures ? 1 : 0;
}
