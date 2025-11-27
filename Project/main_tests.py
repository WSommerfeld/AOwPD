import monte_carlo_integration_sequential_cpu as mciseqcpu
import monte_carlo_integration_parallel_cpu as mciparcpu
from monte_carlo_gpu_with_kernel import monte_carlo_integration_gpu_rawkernel
from monte_carlo_gpu import monte_carlo_integration_gpu

import time
import csv
import os
import argparse


def f(x):
    return x ** 3


def measure_time(function, *args):
    start = time.time()
    result = function(*args)
    return time.time() - start, result


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo benchmark")
    parser.add_argument("N", type=int, help="Liczba próbek")
    parser.add_argument("R", type=int, help="Liczba powtórzeń każdego testu")
    parser.add_argument("--gpu-only", action="store_true", help="Testuj tylko algorytmy GPU")
    args = parser.parse_args()

    N = args.N
    R = args.R
    gpu_only = args.gpu_only

    times_seq = []
    times_par = []
    times_mpi = []
    times_gpu = []
    times_gpu_kernel = []

    a = 0
    b = 2

    print(f"N = {N}, powtórzeń = {R}")
    print("Tryb: tylko GPU" if gpu_only else "Tryb: wszystkie algorytmy")
    print("================================\n")

    # CPU sequential-
    if not gpu_only:
        for _ in range(R):
            t, _ = measure_time(mciseqcpu.monte_carlo_integration, f, a, b, N)
            times_seq.append(t)

    # CPU parallel threads
    if not gpu_only:
        for _ in range(R):
            t, _ = measure_time(mciparcpu.monte_carlo_integration, f, a, b, N)
            times_par.append(t)

    # CPU multiprocessing
    if not gpu_only:
        for _ in range(R):
            t, _ = measure_time(mciparcpu.monte_carlo_integration_multiprocess, f, a, b, N)
            times_mpi.append(t)

    # GPU
    for _ in range(R):
        t, _ = measure_time(monte_carlo_integration_gpu, f, a, b, N, "x*x*x")
        times_gpu.append(t)

    # GPU with kernel
    for _ in range(R):
        t, _ = measure_time(monte_carlo_integration_gpu_rawkernel, 2, a, b, N)
        times_gpu_kernel.append(t)

    avg_seq = sum(times_seq) / len(times_seq) if times_seq else 0
    avg_par = sum(times_par) / len(times_par) if times_par else 0
    avg_mpi = sum(times_mpi) / len(times_mpi) if times_mpi else 0
    avg_gpu = sum(times_gpu) / len(times_gpu) if times_gpu else 0
    avg_gpu_kernel = sum(times_gpu_kernel) / len(times_gpu_kernel) if times_gpu_kernel else 0

    # console output
    if not gpu_only:
        print(f"CPU sequential:           {avg_seq:.6f} s")
        print(f"CPU parallel threads:     {avg_par:.6f} s")
        print(f"CPU multiprocessing:      {avg_mpi:.6f} s")
    print(f"GPU:                      {avg_gpu:.6f} s")
    print(f"GPU with kernel):         {avg_gpu_kernel:.6f} s")

    # csv output
    filename = "results.csv"
    file_exists = os.path.exists(filename)

    with open(filename, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        if not file_exists:
            writer.writerow([
                "N",
                "avg_seq",
                "avg_par",
                "avg_mpi",
                "avg_gpu",
                "avg_gpu_kernel"
            ])

        writer.writerow([
            N,
            avg_seq,
            avg_par,
            avg_mpi,
            avg_gpu,
            avg_gpu_kernel
        ])

    print(f"\nWyniki zapisane do {filename}\n")


if __name__ == "__main__":
    main()
