// Injected into the Wine process (DYLD_INSERT_LIBRARIES) so that D3DMetal's
// MTLDevice compiles pipelines on several MTLCompilerService processes instead
// of one: Metal leaves `shouldMaximizeConcurrentCompilation` off by default and
// D3DMetal does not set it, so every PSO compile goes through one compiler
// process (one core) however many threads ask.
//
// Build (x86_64: Wine runs under Rosetta):
//   clang -arch x86_64 -dynamiclib -fobjc-arc -O2 -framework Foundation \
//     -framework Metal -o metal_concurrent_compile.dylib metal_concurrent_compile.m
//
// It acts only in a process that loads D3DMetal, and only after it has.
// METALVI_MAX_CONCURRENT_COMPILE=0 disables it.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dispatch/dispatch.h>
#include <mach-o/dyld.h>
#include <stdlib.h>
#include <string.h>

static bool d3dmetal_loaded(void) {
    uint32_t n = _dyld_image_count();
    for (uint32_t i = 0; i < n; i++) {
        const char *name = _dyld_get_image_name(i);
        if (name && strstr(name, "D3DMetal")) return true;
    }
    return false;
}

static void apply(int tries_left) {
    if (!d3dmetal_loaded()) {
        if (tries_left > 0) {
            dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 2 * NSEC_PER_SEC),
                           dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{ apply(tries_left - 1); });
        }
        return;
    }
    @autoreleasepool {
        if (@available(macOS 13.3, *)) {
            NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
            for (id<MTLDevice> device in devices) {
                NSUInteger before = device.maximumConcurrentCompilationTaskCount;
                device.shouldMaximizeConcurrentCompilation = YES;
                fprintf(stderr, "metalvi: %s concurrent compilation tasks %lu -> %lu\n",
                        device.name.UTF8String, (unsigned long)before,
                        (unsigned long)device.maximumConcurrentCompilationTaskCount);
            }
        } else {
            fprintf(stderr, "metalvi: concurrent compilation needs macOS 13.3\n");
        }
    }
}

__attribute__((constructor)) static void metalvi_concurrent_compile_init(void) {
    const char *off = getenv("METALVI_MAX_CONCURRENT_COMPILE");
    if (off && strcmp(off, "0") == 0) return;
    // Polled for 5 minutes: D3DMetal loads when the title makes its device.
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{ apply(150); });
}
