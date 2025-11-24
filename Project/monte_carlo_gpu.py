import numpy as np
import cupy as cp  # used version cupy-cuda12x, for cuda 13.0 -- use cupy-cuda13x
import matplotlib.pyplot as plt


def monte_carlo_integration_gpu(f_cpu, a, b, n, function_code=None, batch_size=5000000):
    """
    Optimized GPU Monte Carlo integration with batching

    Args:
        f_cpu: Python function to integrate
        a: Lower integration bound
        b: Upper integration bound
        n: Number of random samples
        function_code: Ignored (kept for compatibility)
        batch_size: Process in batches to optimize memory (default 5M)

    Returns:
        Estimated integral value

    Version that was not optimized and was more of a prototype -- the GPU processes were slower than CPU,
    as it took the data from the cpu, hence the optimization was made
    """

    # Determine bounds on CPU (small overhead, only once)
    x_sample = np.linspace(a, b, 1000)
    y_sample = f_cpu(x_sample)
    y_min = float(np.min(y_sample))
    y_max = float(np.max(y_sample))

    total_hits = 0

    # Process in batches for better performance
    for start_idx in range(0, n, batch_size):
        current_batch = min(batch_size, n - start_idx)

        # Generate random points directly on CPU (faster for large batches)
        random_x_cpu = np.random.uniform(a, b, current_batch).astype(np.float32)
        random_y_cpu = np.random.uniform(y_min, y_max, current_batch).astype(np.float32)

        # Evaluate function on CPU (vectorized NumPy is fast)
        y_func_cpu = f_cpu(random_x_cpu).astype(np.float32)

        # Transfer everything to GPU in one go (minimize transfers)
        random_y_gpu = cp.asarray(random_y_cpu)
        y_func_gpu = cp.asarray(y_func_cpu)

        # Hit counting on GPU (vectorized, very fast)
        if y_min >= 0:
            hits = random_y_gpu <= y_func_gpu
        else:
            pos_hits = (y_func_gpu >= 0) & (random_y_gpu >= 0) & (random_y_gpu <= y_func_gpu)
            neg_hits = (y_func_gpu < 0) & (random_y_gpu <= 0) & (random_y_gpu >= y_func_gpu)
            hits = pos_hits | neg_hits

        # Accumulate hits
        total_hits += int(cp.sum(hits))

        # Free GPU memory for next batch
        del random_y_gpu, y_func_gpu, hits

    # Calculate integral
    rectangle = (b - a) * (y_max - y_min)
    estimated_integral = rectangle * (total_hits / n)

    return estimated_integral


# Example usage
if __name__ == "__main__":
    # Define your function for bounds calculation and plotting
    def f(x):
        return x ** 2


    # Test with different functions
    print("=" * 60)
    print("GPU Monte Carlo Integration Tests")
    print("=" * 60)

    # Test 1: f(x) = x^2 from 0 to 1
    print("\n1. f(x) = x^2, interval [0, 1]")
    result = monte_carlo_integration_gpu(f, 0, 1, 1000000, function_code="x * x")
    print(f"   Estimated integral: {result:.6f}")
    print(f"   True value: {1 / 3:.6f} = 0.333333")
    print(f"   Error: {abs(result - 1 / 3):.6f}")


    # Test 2: f(x) = sin(x) from 0 to π
    def f_sin(x):
        return np.sin(x)


    print("\n2. f(x) = sin(x), interval [0, π]")
    result = monte_carlo_integration_gpu(f_sin, 0, np.pi, 1000000, function_code="sin(x)")
    print(f"   Estimated integral: {result:.6f}")
    print(f"   True value: 2.0")
    print(f"   Error: {abs(result - 2.0):.6f}")


    # Test 3: f(x) = x^3 from -1 to 1
    def f_cubic(x):
        return x ** 3


    print("\n3. f(x) = x^3, interval [-1, 1]")
    result = monte_carlo_integration_gpu(f_cubic, -1, 1, 1000000, function_code="x * x * x")
    print(f"   Estimated integral: {result:.6f}")
    print(f"   True value: 0.0")
    print(f"   Error: {abs(result):.6f}")

    print("\n" + "=" * 60)
