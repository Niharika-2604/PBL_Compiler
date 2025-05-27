# Calculate performance
"""
metrics.py

This module handles runtime benchmarking and performance comparison
between original and optimized code snippets.

Author: Your Name
"""

import timeit
import tracemalloc
from typing import Optional, Dict


# ===============================
# Measure Runtime of a Code Block
# ===============================

def measure_runtime(code: str, func_name: str = "benchmark", number: int = 5, warmup: int = 1) -> Optional[float]:
    """
    Measures the average runtime of a function in the given code snippet.

    Args:
        code (str): The code containing the function to test.
        func_name (str): The entry point function name in the code.
        number (int): How many times to run for averaging.
        warmup (int): How many warmup runs to perform before measurement.

    Returns:
        float: Average runtime (in milliseconds), or None on error.
    """
    try:
        setup_code = f"{code}\n"
        test_code = f"{func_name}()"

        # Warmup phase
        timeit.timeit(stmt=test_code, setup=setup_code, number=warmup)

        # Run benchmark
        total_time = timeit.timeit(stmt=test_code, setup=setup_code, number=number)
        avg_time = (total_time / number) * 1000  # ms

        return round(avg_time, 4)
    except Exception as e:
        print(f"❌ Error during runtime measurement: {e}")
        return None


# ===============================
# Measure Memory Usage (Optional)
# ===============================

def measure_memory(code: str, func_name: str = "benchmark") -> Optional[int]:
    """
    Measures the peak memory usage while executing a function.

    Args:
        code (str): Code containing the benchmark function.
        func_name (str): Entry point function name.

    Returns:
        int or None: Peak memory in KB, if successful.
    """
    try:
        global_namespace = {}
        exec(code, global_namespace)

        tracemalloc.start()
        global_namespace[func_name]()  # Call the function
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Return only peak memory in kilobytes
        return round(peak / 1024)
    except Exception as e:
        print(f"❌ Error during memory profiling: {e}")
        return None


# ===============================
# Compare Metrics — Speedup Info
# ===============================

def compare_metrics(original_time: float, optimized_time: float) -> Dict[str, float]:
    """
    Compare two runtimes and calculate improvement.

    Args:
        original_time (float): Runtime of original code (in ms)
        optimized_time (float): Runtime of optimized code (in ms)

    Returns:
        Dict: Contains 'speedup_percent', 'original_time', 'optimized_time'
    """
    try:
        improvement = original_time - optimized_time
        speedup_pct = (improvement / original_time) * 100 if original_time else 0.0
        return {
            "original_time_ms": round(original_time, 2),
            "optimized_time_ms": round(optimized_time, 2),
            "speedup_percent": round(speedup_pct, 2)
        }
    except Exception as e:
        print(f"❌ Error comparing metrics: {e}")
        return {}


# ===============================
# Example Usage (Test)
# ===============================

if __name__ == "__main__":
    # Basic example to validate
    original_code = """
def benchmark():
    result = 0
    for i in range(10000):
        result += i
    return result
"""

    optimized_code = """
def benchmark():
    return sum(range(10000))
"""

    print("🧪 Measuring original code...")
    orig_time = measure_runtime(original_code)
    print("Avg time (ms):", orig_time)

    print("🧪 Measuring optimized code...")
    opt_time = measure_runtime(optimized_code)
    print("Avg time (ms):", opt_time)

    print("\n⚖️ Comparing...")
    report = compare_metrics(orig_time, opt_time)
    print(report)