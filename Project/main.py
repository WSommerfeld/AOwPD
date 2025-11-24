import monte_carlo_integration_sequential_cpu as mciseqcpu
import monte_carlo_integration_parallel_cpu as mciparcpu
from monte_carlo_gpu import monte_carlo_integration_gpu
import time

def f(x):
    return x**3

if __name__ == '__main__':
    a=0
    b=2
    n=100000

    start = time.time()
    mciseqcpu.monte_carlo_integration(f,a,b,n)
    time_seq = time.time() - start

    start = time.time()
    mciparcpu.monte_carlo_integration(f,a,b,n)
    time_par = time.time() - start

    start = time.time()
    mciparcpu.monte_carlo_integration_multiprocess(f,a,b,n)
    time_mpi = time.time() - start

    start = time.time()
    monte_carlo_integration_gpu(f, a, b, n, function_code="x * x * x")
    time_gpu = time.time() - start


    print("Time sequential: ", time_seq)
    print("Time parallel (threads): ", time_par)
    print("Time parallel (multiprocessing): ", time_mpi)
    print("Time GPU (CUDA):", time_gpu)

