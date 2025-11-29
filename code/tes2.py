import os
import csv
import time
import random
import resource
import statistics
import sys
import multiprocessing
from memory_profiler import memory_usage

# ==============================================================================
# --- IMPLEMENTASI ALGORITMA ---
# ==============================================================================

def mergesort(arr):
    """
    Implementasi standar MergeSort rekursif. Algoritma ini TIDAK in-place 
    dan menggunakan ruang tambahan O(n).
    """
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

        mergesort(left_half)
        mergesort(right_half)
 
        i = j = k = 0
        while i < len(left_half) and j < len(right_half):
            if left_half[i] <= right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1

def partition_naive(arr, low, high):
    """
    Fungsi bantuan untuk QuickSort: Partisi naif menggunakan elemen terakhir sebagai pivot.
    """
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

def _quicksort_naive_recursive(arr, low, high):
    """
    Bagian rekursif dari QuickSort naif.
    """
    if low < high:
        pi = partition_naive(arr, low, high)
        _quicksort_naive_recursive(arr, low, pi - 1)
        _quicksort_naive_recursive(arr, pi + 1, high)

def quicksort_naive(arr):
    """
    Fungsi pembungkus untuk QuickSort rekursif naif.
    Implementasi ini dirancang untuk menunjukkan kasus terburuk O(n^2) secara klasik.
    """
    _quicksort_naive_recursive(arr, 0, len(arr) - 1)

# ==============================================================================
# --- PENGATURAN DATASET DAN EKSPERIMEN ---
# ==============================================================================

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

def run_experiment(sort_fn, data, mem_limit_mb):
    limit_bytes = mem_limit_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    
    data_copy = data.copy()
    
    start_time = time.perf_counter()
    mem_profile = memory_usage((sort_fn, (data_copy,)), interval=0.01, max_usage=True)
    end_time = time.perf_counter()
    
    exec_time = end_time - start_time
    peak_mem = mem_profile if isinstance(mem_profile, (int, float)) else max(mem_profile)
    
    return exec_time, peak_mem

# ==============================================================================
# --- LOGIKA ORKESTRASI UTAMA ---
# ==============================================================================

if __name__ == "__main__":
    sys.setrecursionlimit(2000000)

    # --- PARAMETER EKSPERIMEN ---
    DATASET_SIZES = [10_000, 250_000, 1_000_000, 5_000_000]
    DATASET_TYPES = {
        "Random": gen_random,
        "NearlySorted": gen_nearly_sorted,
        "Reversed": gen_reverse,
        "Duplicates": gen_duplicates,
    }
    MEMORY_LIMITS_MB = [64, 96, 128, 256] 
    
    ALGORITHMS = {
        "MergeSort": mergesort,
        "QuickSort_Naive": quicksort_naive,
    }
    
    NUM_RUNS = 3
    RESULTS_FILE = "experiment_results_detailed_v4_id.csv"
    random.seed(42)

    # --- MEMBUAT HEADER BARU UNTUK FILE CSV ---
    header = [
        "Algorithm", "DatasetType", "DatasetSize", "MemoryLimit_MB"
    ]
    # Tambahkan kolom untuk setiap run
    for i in range(NUM_RUNS):
        header.append(f"ExecutionTime_s_Run{i+1}")
    for i in range(NUM_RUNS):
        header.append(f"PeakMemory_MB_Run{i+1}")
    # Tambahkan kolom untuk statistik agregat
    header.extend([
        "AvgExecutionTime_s", "MaxExecutionTime_s",
        "AvgPeakMemory_MB", "MaxPeakMemory_MB", "Status"
    ])

    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    # --- LOOP EKSPERIMEN UTAMA ---
    for size in DATASET_SIZES:
        for type_name, gen_fn in DATASET_TYPES.items():
            print(f"\n--- Generating dataset: {type_name} {size} ---")
            dataset = gen_fn(size)

            for algo_name, sort_fn in ALGORITHMS.items():
                for limit in MEMORY_LIMITS_MB:
                    print(f"Running {algo_name} on {type_name} (Size: {size}) with {limit}MB limit ({NUM_RUNS} runs)...")

                    exec_times = []
                    peak_mems = []
                    statuses = []

                    for i in range(NUM_RUNS):
                        try:
                            q = multiprocessing.Queue()

                            def process_target(q, sort_fn, dataset, limit):
                                try:
                                    exec_time, peak_mem = run_experiment(sort_fn, dataset, limit)
                                    q.put((exec_time, peak_mem, "SUCCESS"))
                                except MemoryError:
                                    q.put((None, None, "MEMORY_ERROR"))
                                except RecursionError:
                                    q.put((None, None, "RECURSION_ERROR"))
                                except Exception as e:
                                    q.put((None, None, f"RUNTIME_ERROR: {type(e).__name__}"))

                            p = multiprocessing.Process(target=process_target, args=(q, sort_fn, dataset, limit))
                            p.start()
                            p.join()

                            exec_time, peak_mem, status = q.get()
                            statuses.append(status)
                            if status == "SUCCESS":
                                exec_times.append(exec_time)
                                peak_mems.append(peak_mem)
                            else:
                                print(f"Run {i+1}/{NUM_RUNS} failed with status: {status}. Aborting for this configuration.")
                                break
                        except Exception as e:
                            statuses.append(f"FATAL_ERROR: {e}")
                            break
                    
                    # --- MENULIS HASIL KE CSV ---
                    final_status = statuses[-1]
                    row_data = [algo_name, type_name, size, limit]

                    if all(s == "SUCCESS" for s in statuses):
                        # Kalkulasi statistik
                        avg_time = statistics.mean(exec_times)
                        max_time = max(exec_times)
                        avg_mem = statistics.mean(peak_mems)
                        max_mem = max(peak_mems)
                        
                        # Susun baris data untuk ditulis
                        row_data.extend([f"{t:.4f}" for t in exec_times])
                        row_data.extend([f"{m:.2f}" for m in peak_mems])
                        row_data.extend([
                            f"{avg_time:.4f}", f"{max_time:.4f}",
                            f"{avg_mem:.2f}", f"{max_mem:.2f}",
                            "SUCCESS"
                        ])
                    else:
                        # Jika ada kegagalan, isi kolom data dengan "N/A"
                        num_data_cols = (NUM_RUNS * 2) + 4
                        row_data.extend(["N/A"] * num_data_cols)
                        row_data.append(final_status)

                    with open(RESULTS_FILE, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(row_data)

    print(f"\n--- All experiments complete. Results saved to {RESULTS_FILE} ---")