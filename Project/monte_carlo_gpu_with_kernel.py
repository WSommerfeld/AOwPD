import numpy as np
import cupy as cp

# CUDA C code for two kernels:
#  - eval_f: compute f(x) and store in out_f (no randoms)
#  - mc_eval: compute f(x) and compare with provided random_y -> produce 0/1 hits
cuda_source = r'''
#include <math.h>

extern "C" {

// select function to evaluate
__device__ float eval_function(float x, int mode) {
    float v = 0.0f;
    switch (mode) {
        case 0: // x^2
            v = x * x;
            break;
        case 1: // sin(x)
            v = sinf(x);
            break;
        case 2: // x^3
            v = x * x * x;
            break;
        case 3: // exp(-x^2)
            v = expf(-x * x);
            break;
        default:
            // default to 0
            v = 0.0f;
    }
    return v;
}

// compute f(x) for each x and write to out_f
__global__ void eval_f(const float* x, float* out_f, int mode, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    out_f[idx] = eval_function(x[idx], mode);
}

// compare random_y with f(x) and write 0/1 hits
__global__ void mc_eval(const float* x, const float* random_y, unsigned char* hits, int mode, float y_min, float y_max, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    float xv = x[idx];
    float randy = random_y[idx];
    float fval = eval_function(xv, mode);

    unsigned char hit = 0;
    if (y_min >= 0.0f) {
        if (randy <= fval) hit = 1;
    } else {
        if (fval >= 0.0f) {
            if (randy >= 0.0f && randy <= fval) hit = 1;
        } else {
            if (randy <= 0.0f && randy >= fval) hit = 1;
        }
    }
    hits[idx] = hit;
}

} // extern "C"
'''

# Compile kernels
raw_mod = cp.RawModule(code=cuda_source, backend='nvcc')
eval_f_kernel = raw_mod.get_function('eval_f')
mc_eval_kernel = raw_mod.get_function('mc_eval')


def monte_carlo_integration_gpu_rawkernel(
    mode,
    a,
    b,
    n,
    batch_size=5_000_000,
    threads_per_block=256,
    sample_for_bounds=2048,
):
    """
    Całkowanie MC na gpu z użyciem kernela

    Args:
        mode: wybór funkcji co całkowania
              0 -> x^2
              1 -> sin(x)
              2 -> x^3
              3 -> exp(-x^2)
        a, b: zakres
        n: liczba próbk
        batch_size: liczba próbek na jedną iteracje
        threads_per_block: liczba wątków na blok
        sample_for_bounds: rozdzielczość sprawdzania y_max/y_min
    Return:
        estimated_integral (float)
    """

    if a >= b:
        raise ValueError("BŁĄD - warunek: a < b")
    if n <= 0:
        raise ValueError("BŁĄd - warunek: n > 0")

    # create sample points on GPU
    x_sample = cp.linspace(a, b, sample_for_bounds, dtype=cp.float32)
    out_f_sample = cp.empty_like(x_sample, dtype=cp.float32)

    # launch eval_f kernel
    blocks = (sample_for_bounds + threads_per_block - 1) // threads_per_block
    eval_f_kernel((blocks,), (threads_per_block,),
                  (x_sample, out_f_sample, np.int32(mode), np.int32(sample_for_bounds)))

    y_min = float(cp.min(out_f_sample).get())
    y_max = float(cp.max(out_f_sample).get())

    total_hits = 0
    samples_done = 0

    while samples_done < n:
        current_batch = int(min(batch_size, n - samples_done))

        rand_x = cp.random.uniform(a, b, size=current_batch, dtype=cp.float32)
        rand_y = cp.random.uniform(y_min, y_max, size=current_batch, dtype=cp.float32)

        hits = cp.empty(current_batch, dtype=cp.uint8)

        # launch mc_eval kernel
        blocks = (current_batch + threads_per_block - 1) // threads_per_block
        mc_eval_kernel((blocks,), (threads_per_block,),
                       (rand_x, rand_y, hits, np.int32(mode), np.float32(y_min), np.float32(y_max), np.int32(current_batch)))

        batch_hits = int(cp.sum(hits).get())

        total_hits += batch_hits
        samples_done += current_batch

    # compute integral
    rectangle = (b - a) * (y_max - y_min)
    estimated_integral = rectangle * (total_hits / n)

    return estimated_integral
