/*
 * CUDA driver API compatibility shim.
 *
 * Older NVIDIA drivers (< 565) lack cuCtxGetDevice_v2 and cuCtxGetCurrent_v2,
 * which cubecl/burn-cuda tries to load via dlsym. This library provides them
 * by delegating to the original non-v2 functions.
 *
 * Build:   gcc -shared -fPIC -o libcuda_compat.so cuda_compat.c -ldl
 * Usage:   LD_PRELOAD=/usr/local/lib/libcuda_compat.so distrain-node ...
 */
#include <dlfcn.h>
#include <stddef.h>

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;

CUresult cuCtxGetDevice_v2(CUdevice *device) {
    typedef CUresult (*fn_t)(CUdevice *);
    static fn_t real_fn = NULL;
    if (!real_fn) {
        real_fn = (fn_t)dlsym(RTLD_NEXT, "cuCtxGetDevice");
    }
    return real_fn ? real_fn(device) : 1;
}

CUresult cuCtxGetCurrent_v2(CUcontext *ctx) {
    typedef CUresult (*fn_t)(CUcontext *);
    static fn_t real_fn = NULL;
    if (!real_fn) {
        real_fn = (fn_t)dlsym(RTLD_NEXT, "cuCtxGetCurrent");
    }
    return real_fn ? real_fn(ctx) : 1;
}
