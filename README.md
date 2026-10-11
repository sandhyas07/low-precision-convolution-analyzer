# Low-Precision Convolution Analyzer

## Problem Statement
Reduced numerical precision can affect the accuracy and computational performance of convolution operations. This project compares FP32 and FP16 precision levels to study their numerical differences and execution time.

## Objective
To analyze how reduced numerical precision affects convolution results by comparing numerical error and execution time.

## Technologies Used
- Python
- NumPy
- Matplotlib
- CSV

## Methodology
1. Generate a sample 128 × 128 input image and a 3 × 3 convolution kernel.
2. Perform convolution using FP32 and FP16 precision.
3. Use FP32 as the reference result.
4. Calculate maximum and mean absolute errors.
5. Measure average execution time.
6. Save the results in CSV format and generate comparison graphs.

## How to Run
Install the required libraries:

```bash
python -m pip install numpy matplotlib
```

Run the program:

```bash
python convolution_analyzer.py
```

## Output Files
- `results.csv` — contains execution time and numerical error measurements.
- `comparison_graph.png` — compares execution time and mean absolute error.

## Conclusion
The experiment demonstrates that FP16 can introduce small numerical differences compared with FP32. Execution-time results depend on the implementation and hardware, so lower precision does not guarantee faster execution.

## References
- NumPy documentation: https://numpy.org/doc/
- Matplotlib documentation: https://matplotlib.org/stable/
