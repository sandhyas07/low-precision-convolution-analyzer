
import numpy as np
import matplotlib.pyplot as plt
import time
import csv


def convolution(image, kernel, dtype):
    """Perform valid 2D convolution-style filtering at a chosen precision."""
    image = image.astype(dtype)
    kernel = kernel.astype(dtype)

    height, width = image.shape
    kh, kw = kernel.shape

    output = np.zeros(
        (height - kh + 1, width - kw + 1),
        dtype=dtype
    )

    for i in range(height - kh + 1):
        for j in range(width - kw + 1):
            region = image[i:i + kh, j:j + kw]

            # Round products and accumulation to the chosen precision
            products = (region * kernel).astype(dtype)
            output[i, j] = np.sum(products, dtype=dtype)

    return output


def measure_time(image, kernel, dtype, repeats=10):
    """Measure average execution time over repeated runs."""
    # Warm-up run
    convolution(image, kernel, dtype)

    timings = []

    for _ in range(repeats):
        start = time.perf_counter()
        convolution(image, kernel, dtype)
        timings.append(time.perf_counter() - start)

    return float(np.median(timings))


def main():
    np.random.seed(42)

    # Fixed input makes comparisons reproducible
    image = np.random.rand(128, 128).astype(np.float64)

    kernel = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ], dtype=np.float64)

    # FP64 is the higher-precision reference
    reference = convolution(image, kernel, np.float64).astype(np.float64)

    precision_levels = [
        ("FP16", np.float16),
        ("FP32", np.float32),
        ("FP64", np.float64)
    ]

    results = []

    for name, dtype in precision_levels:
        output = convolution(image, kernel, dtype).astype(np.float64)

        difference = np.abs(output - reference)
        max_error = float(np.max(difference))
        mean_error = float(np.mean(difference))

        elapsed = measure_time(image, kernel, dtype)

        results.append({
            "Precision": name,
            "Median Time (seconds)": elapsed,
            "Maximum Absolute Error": max_error,
            "Mean Absolute Error": mean_error
        })

    print("\nLOW-PRECISION CONVOLUTION ANALYSIS")
    print("=" * 88)
    print(
        f"{'Precision':<12}"
        f"{'Median Time (s)':<20}"
        f"{'Maximum Error':<22}"
        f"{'Mean Error':<22}"
    )

    for row in results:
        print(
            f"{row['Precision']:<12}"
            f"{row['Median Time (seconds)']:<20.6f}"
            f"{row['Maximum Absolute Error']:<22.8f}"
            f"{row['Mean Absolute Error']:<22.8f}"
        )

    with open("results.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    names = [row["Precision"] for row in results]
    times = [row["Median Time (seconds)"] for row in results]
    errors = [row["Mean Absolute Error"] for row in results]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].bar(names, times)
    axes[0].set_title("Median Convolution Execution Time")
    axes[0].set_ylabel("Time (seconds)")
    axes[0].set_xlabel("Numerical Precision")

    axes[1].bar(names, errors)
    axes[1].set_title("Mean Absolute Error vs FP64")
    axes[1].set_ylabel("Absolute Error")
    axes[1].set_xlabel("Numerical Precision")

    plt.tight_layout()
    plt.savefig("comparison_graph.png", dpi=150)

    print("\nUpdated results.csv and comparison_graph.png")
    plt.show()


if __name__ == "__main__":
    main()
