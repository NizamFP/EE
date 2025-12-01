"""
FINAL EXPERIMENT SCRIPT - IB CS Extended Essay

Experimental Design (Full Factorial):
- 2 algorithms: QuickSort, MergeSort
- 4 memory categories: 128, 256, 512, 1024 MB (organizational)
- 4 dataset types: random, nearly_sorted, reverse, duplicate_heavy
- 4 sizes: 10,000 | 50,000 | 100,000 | 500,000 elements
- 5 repetitions per configuration

Total: 2 × 4 × 4 × 4 × 5 = 640 experimental runs

Note: Memory categories are organizational (not enforced) due to 
      GitHub Codespaces container limitations.
      Focus on accurate timing measurement.

Author: [Candidate Number]
Date: November 2024
Purpose: IB Computer Science Extended Essay (May 2026)
"""

import os
import csv
import sys
import time
from datetime import datetime, timezone
from tqdm import tqdm
from statistics import mean, stdev

from mysorts import quicksort_wrapper, mergesort_wrapper
from datagen import (
    generate_random,
    generate_nearly_sorted,
    generate_reverse_sorted,
    generate_duplicate_heavy
)

# -----------------------------
# CONFIGURATION
# -----------------------------
SIZES = [10000, 50000, 100000, 500000]
STRUCTURES = ["random", "nearly_sorted", "reverse", "duplicate_heavy"]
TRIALS = 5
MEMORY_LIMITS = [128, 256, 512, 1024]  # Organizational categories
ALGORITHMS = ['QuickSort', 'MergeSort']

TOTAL_CONFIGURATIONS = len(ALGORITHMS) * len(MEMORY_LIMITS) * len(STRUCTURES) * len(SIZES)
TOTAL_RUNS = TOTAL_CONFIGURATIONS * TRIALS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "../results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RAW_CSV = os.path.join(RESULTS_DIR, 
    f"raw_results_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv")


# --------------------------------
# CSV FUNCTIONS
# --------------------------------
def write_raw_header():
    """Write header row for raw results CSV."""
    with open(RAW_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id",
            "algorithm",
            "memory_category_MB",
            "size",
            "structure",
            "trial",
            "execution_time_s",
            "comparisons",
            "success",
            "error_message"
        ])


def append_raw_row(row):
    """Append a data row to raw results CSV."""
    with open(RAW_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def write_summary_table_by_memory(memory_limit, qs_data, ms_data):
    """
    Write summary statistics table for one memory category.
    
    Args:
        memory_limit (int): Memory category in MB
        qs_data (dict): QuickSort statistics by (size, structure)
        ms_data (dict): MergeSort statistics by (size, structure)
    """
    outpath = os.path.join(RESULTS_DIR, f"table1_mem{memory_limit}MB.csv")
    
    with open(outpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "size",
            "structure",
            "trials_succeeded",
            "QuickSort_mean_time",
            "QuickSort_std_time",
            "QuickSort_mean_comparisons",
            "QuickSort_std_comparisons",
            "MergeSort_mean_time",
            "MergeSort_std_time",
            "MergeSort_mean_comparisons",
            "MergeSort_std_comparisons"
        ])
        
        all_keys = set(qs_data.keys()) | set(ms_data.keys())
        
        for key in sorted(all_keys):
            size, structure = key
            
            qs_stats = qs_data.get(key, {})
            ms_stats = ms_data.get(key, {})
            
            qs_n = qs_stats.get('n_trials', 0)
            ms_n = ms_stats.get('n_trials', 0)
            
            if qs_n < 3 and ms_n < 3:
                continue
            
            w.writerow([
                size,
                structure,
                f"{qs_n}/{ms_n}",
                qs_stats.get('mean_t', 'N/A'),
                qs_stats.get('std_t', 'N/A'),
                qs_stats.get('mean_c', 'N/A'),
                qs_stats.get('std_c', 'N/A'),
                ms_stats.get('mean_t', 'N/A'),
                ms_stats.get('std_t', 'N/A'),
                ms_stats.get('mean_c', 'N/A'),
                ms_stats.get('std_c', 'N/A'),
            ])
    
    print(f"    ✓ Wrote: table1_mem{memory_limit}MB.csv")


# --------------------------------
# MAIN EXPERIMENT LOOP
# --------------------------------

def run_experiments():
    """
    Main experimental execution.
    
    Loop Structure:
        FOR each memory category (4) - organizational grouping
            FOR each dataset size (4)
                FOR each structure type (4)
                    FOR each trial (5)
                        - Run QuickSort → measure time, comparisons
                        - Run MergeSort → measure time, comparisons
    
    Total: 4 × 4 × 4 × 5 × 2 = 640 algorithm executions
    
    Note: No tracemalloc to ensure accurate timing measurements.
          Memory analysis conducted separately.
    """
    
    print("=" * 80)
    print(" " * 15 + "SORTING ALGORITHM PERFORMANCE ANALYSIS")
    print(" " * 25 + "(Timing & Comparisons)")
    print("=" * 80)
    print(f"\nExperimental Design (Full Factorial):")
    print(f"  Algorithms:           {len(ALGORITHMS)} (QuickSort, MergeSort)")
    print(f"  Memory categories:    {len(MEMORY_LIMITS)} (organizational)")
    print(f"  Dataset structures:   {len(STRUCTURES)}")
    print(f"  Dataset sizes:        {len(SIZES)}")
    print(f"  Trials per config:    {TRIALS}")
    print(f"\n  Total configurations: {TOTAL_CONFIGURATIONS}")
    print(f"  Total experimental runs: {TOTAL_RUNS}")
    print(f"\nMeasurements:")
    print(f"  • Execution time (time.perf_counter)")
    print(f"  • Comparison counts (algorithm-embedded)")
    print(f"  • NO tracemalloc (ensures timing accuracy)")
    print(f"\nEnvironment:")
    print(f"  • Platform: GitHub Codespaces (Linux)")
    print(f"  • Python: {sys.version.split()[0]}")
    print(f"\nOutput:")
    print(f"  • Raw: {os.path.basename(RAW_CSV)}")
    print(f"  • Summary: table1_mem*.csv (one per category)")
    print("=" * 80)
    print()
    
    write_raw_header()
    
    pbar = tqdm(total=TOTAL_RUNS, desc="Experimental runs", unit="run",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    run_id = 0
    
    # ==========================================
    # MAIN EXPERIMENTAL LOOP
    # ==========================================
    
    for mem_limit in MEMORY_LIMITS:
        print(f"\n{'='*80}")
        print(f"Memory Category: {mem_limit} MB")
        print(f"{'='*80}")
        
        qs_aggregated = {}
        ms_aggregated = {}
        
        for size in SIZES:
            print(f"\n  Dataset size: {size:,} elements")
            
            for structure in STRUCTURES:
                print(f"    {structure:20s}: ", end="", flush=True)
                
                qs_results = []
                ms_results = []
                
                for trial in range(1, TRIALS + 1):
                    
                    # ==========================================
                    # GENERATE DATASET
                    # ==========================================
                    try:
                        if structure == "random":
                            dataset = generate_random(size)
                        elif structure == "nearly_sorted":
                            dataset = generate_nearly_sorted(size, disorder_fraction=0.1)
                        elif structure == "reverse":
                            dataset = generate_reverse_sorted(size)
                        elif structure == "duplicate_heavy":
                            dataset = generate_duplicate_heavy(size, duplicate_ratio=0.5)
                        else:
                            raise ValueError(f"Unknown structure: {structure}")
                    
                    except MemoryError:
                        run_id += 2
                        pbar.update(2)
                        append_raw_row([run_id-1, "QuickSort", mem_limit, size, structure, trial,
                                      "FAIL", "FAIL", False, "Dataset MemoryError"])
                        append_raw_row([run_id, "MergeSort", mem_limit, size, structure, trial,
                                      "FAIL", "FAIL", False, "Dataset MemoryError"])
                        print("✗✗", end="", flush=True)
                        continue
                    
                    except Exception as e:
                        run_id += 2
                        pbar.update(2)
                        append_raw_row([run_id-1, "QuickSort", mem_limit, size, structure, trial,
                                      "FAIL", "FAIL", False, f"Generation error: {str(e)[:30]}"])
                        append_raw_row([run_id, "MergeSort", mem_limit, size, structure, trial,
                                      "FAIL", "FAIL", False, f"Generation error: {str(e)[:30]}"])
                        print("✗✗", end="", flush=True)
                        continue
                    
                    # ==========================================
                    # QUICKSORT EXECUTION
                    # ==========================================
                    run_id += 1
                    
                    try:
                        # Time measurement
                        start = time.perf_counter()
                        qs_sorted, qs_comps = quicksort_wrapper(dataset)
                        qs_time = time.perf_counter() - start
                        
                        # Validate correctness
                        if qs_sorted == sorted(dataset):
                            qs_results.append({
                                "time": qs_time,
                                "comparisons": qs_comps
                            })
                            append_raw_row([run_id, "QuickSort", mem_limit, size, structure, trial,
                                          qs_time, qs_comps, True, ""])
                            print("✓", end="", flush=True)
                        else:
                            raise ValueError("Incorrect sort")
                    
                    except RecursionError:
                        append_raw_row([run_id, "QuickSort", mem_limit, size, structure, trial,
                                      "FAIL", "FAIL", False, "RecursionError"])
                        print("✗", end="", flush=True)
                    
                    except MemoryError:
                        append_raw_row([run_id, "QuickSort", mem_limit, size, structure, trial,
                                      "FAIL", "FAIL", False, "MemoryError"])
                        print("✗", end="", flush=True)
                    
                    except Exception as e:
                        append_raw_row([run_id, "QuickSort", mem_limit, size, structure, trial,
                                      "FAIL", "FAIL", False, f"{type(e).__name__}: {str(e)[:30]}"])
                        print("✗", end="", flush=True)
                    
                    pbar.update(1)
                    
                    # ==========================================
                    # MERGESORT EXECUTION
                    # ==========================================
                    run_id += 1
                    
                    try:
                        # Time measurement
                        start = time.perf_counter()
                        ms_sorted, ms_comps = mergesort_wrapper(dataset)
                        ms_time = time.perf_counter() - start
                        
                        # Validate correctness
                        if ms_sorted == sorted(dataset):
                            ms_results.append({
                                "time": ms_time,
                                "comparisons": ms_comps
                            })
                            append_raw_row([run_id, "MergeSort", mem_limit, size, structure, trial,
                                          ms_time, ms_comps, True, ""])
                            print("✓", end="", flush=True)
                        else:
                            raise ValueError("Incorrect sort")
                    
                    except RecursionError:
                        append_raw_row([run_id, "MergeSort", mem_limit, size, structure, trial,
                                      "FAIL", "FAIL", False, "RecursionError"])
                        print("✗", end="", flush=True)
                    
                    except MemoryError:
                        append_raw_row([run_id, "MergeSort", mem_limit, size, structure, trial,
                                      "FAIL", "FAIL", False, "MemoryError"])
                        print("✗", end="", flush=True)
                    
                    except Exception as e:
                        append_raw_row([run_id, "MergeSort", mem_limit, size, structure, trial,
                                      "FAIL", "FAIL", False, f"{type(e).__name__}: {str(e)[:30]}"])
                        print("✗", end="", flush=True)
                    
                    pbar.update(1)
                
                # End of trials for this configuration
                print(f" ({len(qs_results)}QS/{len(ms_results)}MS)")
                
                # Store results
                key = (size, structure)
                if qs_results:
                    qs_aggregated[key] = qs_results
                if ms_results:
                    ms_aggregated[key] = ms_results
        
        # ==========================================
        # COMPUTE SUMMARY STATISTICS
        # ==========================================
        print(f"\n  Computing summary statistics for {mem_limit} MB category...")
        
        qs_summary = {}
        ms_summary = {}
        
        # QuickSort statistics
        for (size, structure), results in qs_aggregated.items():
            if len(results) >= 3:
                times = [r['time'] for r in results]
                comparisons = [r['comparisons'] for r in results]
                
                qs_summary[(size, structure)] = {
                    'n_trials': len(results),
                    'mean_t': mean(times),
                    'std_t': stdev(times) if len(times) > 1 else 0,
                    'mean_c': mean(comparisons),
                    'std_c': stdev(comparisons) if len(comparisons) > 1 else 0,
                }
            else:
                print(f"    ⚠ QuickSort {size:,}/{structure}: Only {len(results)}/5 succeeded - EXCLUDED")
        
        # MergeSort statistics
        for (size, structure), results in ms_aggregated.items():
            if len(results) >= 3:
                times = [r['time'] for r in results]
                comparisons = [r['comparisons'] for r in results]
                
                ms_summary[(size, structure)] = {
                    'n_trials': len(results),
                    'mean_t': mean(times),
                    'std_t': stdev(times) if len(times) > 1 else 0,
                    'mean_c': mean(comparisons),
                    'std_c': stdev(comparisons) if len(comparisons) > 1 else 0,
                }
            else:
                print(f"    ⚠ MergeSort {size:,}/{structure}: Only {len(results)}/5 succeeded - EXCLUDED")
        
        # Write summary table for this memory category
        write_summary_table_by_memory(mem_limit, qs_summary, ms_summary)
    
    # ==========================================
    # FINAL SUMMARY
    # ==========================================
    
    pbar.close()
    
    print("\n" + "=" * 80)
    print(" " * 30 + "EXPERIMENTS COMPLETE")
    print("=" * 80)
    print(f"\nExecution Summary:")
    print(f"  Total runs executed:   {run_id}")
    print(f"  Expected runs:         {TOTAL_RUNS}")
    print(f"  Success rate:          {(run_id/TOTAL_RUNS)*100:.1f}%")
    print(f"\nOutput Files:")
    print(f"  Raw data:     {RAW_CSV}")
    print(f"  Summary:      {RESULTS_DIR}/table1_mem*.csv")
    print(f"                (4 tables, one per memory category)")
    print(f"\nNote: This run measures TIMING accurately (no tracemalloc overhead).")
    print(f"      For memory analysis, see separate memory profiling run.")
    print("=" * 80)


# --------------------------------
# ENTRY POINT
# --------------------------------
if __name__ == "__main__":
    print(f"\nExperiment started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print()
    
    try:
        run_experiments()
    except KeyboardInterrupt:
        print("\n\n✗ Experiment interrupted by user (Ctrl+C)")
        print("  Partial results saved to CSV files")
    except Exception as e:
        print(f"\n\n✗ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()