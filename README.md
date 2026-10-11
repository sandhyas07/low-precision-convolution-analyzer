# Low-Precision Convolution Analyzer

## Problem Statement
Study the effect of reduced numerical precision on convolution results by comparing different precision levels in terms of accuracy and computation.

## Objective
To analyze the impact of FP16, FP32, and FP64 precision on convolution accuracy and execution time.

## Technologies Used
- Python
- NumPy
- Matplotlib
- CSV

## Methodology
1. Generate a reproducible 128 × 128 input image and a 3 × 3 kernel.
2. Perform convolution-style filtering using FP16, FP32, and FP64.
3. Use FP64 as the higher-precision reference.
4. Calculate maximum and mean absolute errors.
5. Measure execution time using repeated runs and report the median.
6. Save the measurements in CSV format and generate comparison graphs.

## How to Run

Install the libraries:

```bash
python -m pip install numpy matplotlib
```

Run the program:

```bash
python convolution_analyzer.py
```

## Output Files
- `results.csv` — execution-time and numerical-error measurements.
- `comparison_graph.png` — graphs comparing execution time and mean absolute error.

## Conclusion
The experiment demonstrates that FP16 produces greater numerical error than FP32 when compared with FP64 for the tested input. Execution time depends on the implementation and hardware; lower precision does not necessarily guarantee faster computation.

## References
- NumPy documentation: https://numpy.org/doc/
- Matplotlib documentation: https://matplotlib.org/stable/
