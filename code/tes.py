import os
import csv
import time
import random
import resource
from memory_profiler import memory_usage
from mysorts import mergesort, quicksort # Use the new in-place quicksort
from tqdm import tqdm

# --- 1. Dataset Generation (Good, can be more programmatic) ---
def gen_random(size):
    return [random.randint(0, size) for _ in range(size)]

def gen_nearly_sorted(size, swap_fraction=0.1):
    arr = list(range(size))
    num_swaps = int(size * swap_fraction)
    for _ in range(num_swaps):
        i, j = random.randrange(size), random.randrange(size)
        arr[i], arr[j] = arr[j], arr[i]
    return arr

def gen_reverse(size):
    return list(range(size, 0, -1))

def gen_duplicates(size, value_range=100):
    return [random.randint(0, value_range) for _ in range(size)]


# --- 2. Correct Measurement and Constraint Function ---
def run_experiment(sort_fn, data, mem_limit_mb):
    """`
    Runs a single sorting experiment with a memory limit,
    measuring both time and peak memory usage.
    """
    # Set the memory limit for this process.
    # Note: resource module works on Unix-like systems (macOS, Linux).
    # The limit is in bytes.
    # limit_bytes = mem_limit_mb * 1024 * 1024
    # resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    
    data_copy = data.copy() # Work on a copy to not affect other runs
    start_time = time.perf_counter()
    # Profile memory of the single sort execution
    mem_profile = memory_usage((sort_fn, (data_copy,)), interval=0.01)
    end_time = time.perf_counter()

    exec_time = end_time - start_time
    peak_mem = max(mem_profile)

    return exec_time, peak_mem

def process_target(q, sort_fn, dataset, limit):
                            try:
                                exec_time, peak_mem = run_experiment(sort_fn, dataset, limit)
                                q.put((exec_time, peak_mem, "SUCCESS"))
                            except MemoryError:
                                q.put((None, None, "MEMORY_ERROR"))
                            except Exception as e:
                                q.put((None, None, f"RUNTIME_ERROR: {e}"))

# --- 3. Main Orchestration Logic ---
if __name__ == "__main__":
    # Define your experimental parameters
    # Note: Python's recursion limit might be an issue for naive quicksorts on large, sorted inputs.
    # The in-place one is better but still might hit it. You may need sys.setrecursionlimit().
    import sys
    sys.setrecursionlimit(2000000) # Set a high recursion limit for large datasets

    DATASET_SIZES = [10_000, 250_000, 1_000_000, 5_000_000]
    DATASET_TYPES = {
        "Random": gen_random,
        "NearlySorted": gen_nearly_sorted,
        "Reversed": gen_reverse,
        "Duplicates": gen_duplicates,
    }
    MEMORY_LIMITS_MB = [96, 128, 256, 512]
    ALGORITHMS = {
        "MergeSort": mergesort,
        "QuickSort_InPlace": quicksort,
    }
    RESULTS_FILE = "experiment_results.csv"

    # Set a seed for reproducibility
    random.seed(42)

    # Write header to the results file
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Algorithm", "DatasetType", "DatasetSize", "MemoryLimit_MB",
            "ExecutionTime_s", "PeakMemory_MB", "Status"
        ])

    total_experiments = len(DATASET_SIZES) * len(DATASET_TYPES) * len(MEMORY_LIMITS_MB) * len(ALGORITHMS)
    progress = tqdm(total=total_experiments, desc="Running Experiments", ncols=100)

    # --- Main experimental loop ---
    for size in DATASET_SIZES:
        for type_name, gen_fn in DATASET_TYPES.items():
            print(f"\n--- Generating dataset: {type_name} {size} ---")
            dataset = gen_fn(size)

            for algo_name, sort_fn in ALGORITHMS.items():
                for limit in MEMORY_LIMITS_MB:
                    progress.set_postfix({
                        "Algo": algo_name,
                        "Type": type_name,
                        "Size": f"{size:,}",
                        "Limit_MB": limit
                    })
                    print(f"Running {algo_name} on {type_name} (Size: {size}) with {limit}MB limit...")
                    
                    try:
                        # We need to run the experiment in a separate process
                        # because `resource.setrlimit` cannot be lowered once set.
                        # Using multiprocessing is the cleanest way.
                        import multiprocessing

                        # Use a queue to get the results back from the child process
                        q = multiprocessing.Queue()

                        p = multiprocessing.Process(target=process_target, args=(q, sort_fn, dataset, limit))
                        p.start()
                        p.join() # Wait for the process to finish

                        exec_time, peak_mem, status = q.get()

                        # Write result to CSV
                        with open(RESULTS_FILE, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                algo_name, type_name, size, limit,
                                f"{exec_time:.4f}" if exec_time is not None else "N/A",
                                f"{peak_mem:.2f}" if peak_mem is not None else "N/A",
                                status
                            ])
                            progress.update(1)
                            print(f"Result: {status}")
                
                    except Exception as e:
                        print(f"FATAL ERROR in multiprocessing setup: {e}")

    progress.close()
    print("\n✅ All experiments completed successfully.")