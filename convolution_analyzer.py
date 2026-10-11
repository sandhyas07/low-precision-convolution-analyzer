
import numpy as np
import matplotlib.pyplot as plt
import time
import csv

# Low-Precision Convolution Analyzer
# Compare FP32 and FP16 numerical precision

def convolution(image, kernel, dtype):
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
            # Explicitly round products and accumulation
            products = (region * kernel).astype(dtype)
            total = np.sum(products, dtype=dtype)
            output[i, j] = total

    return output


def main():
    np.random.seed(42)

    # Create a sample 128 x 128 image and a 3 x 3 kernel
    image = np.random.rand(128, 128).astype(np.float32)
    kernel = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ], dtype=np.float32)

    # Use FP32 as the reference result
    reference = convolution(image, kernel, np.float32)

    results = []
    outputs = {}

    for name, dtype in [
        ("FP32", np.float32),
        ("FP16", np.float16)
    ]:
        # Repeat the operation to obtain a more stable timing
        start = time.perf_counter()
        for _ in range(5):
            output = convolution(image, kernel, dtype)
        elapsed = (time.perf_counter() - start) / 5

        # Compare each result against the FP32 reference
        difference = output.astype(np.float64) - reference.astype(np.float64)
        absolute_error = np.abs(difference)

        max_error = float(np.max(absolute_error))
        mean_error = float(np.mean(absolute_error))

        results.append({
            "Precision": name,
            "Time (seconds)": elapsed,
            "Maximum Absolute Error": max_error,
            "Mean Absolute Error": mean_error
        })
        outputs[name] = output

    # Display the results
    print("\nLOW-PRECISION CONVOLUTION ANALYSIS")
    print("-" * 72)
    print(f"{'Precision':<12}{'Time (s)':<15}"
          f"{'Maximum Error':<20}{'Mean Error':<15}")

    for row in results:
        print(
            f"{row['Precision']:<12}"
            f"{row['Time (seconds)']:<15.6f}"
            f"{row['Maximum Absolute Error']:<20.8f}"
            f"{row['Mean Absolute Error']:<15.8f}"
        )

    # Save the measurements to a CSV file
    with open("results.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Plot numerical error and execution time
    names = [row["Precision"] for row in results]
    times = [row["Time (seconds)"] for row in results]
    errors = [row["Mean Absolute Error"] for row in results]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(names, times)
    axes[0].set_title("Convolution Execution Time")
    axes[0].set_ylabel("Average Time (seconds)")

    axes[1].bar(names, errors)
    axes[1].set_title("Mean Absolute Error")
    axes[1].set_ylabel("Error compared with FP32")

    plt.tight_layout()
    plt.savefig("comparison_graph.png", dpi=150)
    print("\nFiles saved: results.csv and comparison_graph.png")

    plt.show()


if __name__ == "__main__":
    main()
