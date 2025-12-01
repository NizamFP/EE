"""
Memory Analysis Script - Separate from Timing
Uses tracemalloc for precise memory measurement
(Timing data discarded due to overhead)
"""

import os
import csv
from datetime import datetime, timezone
from statistics import mean, stdev

from mysorts import quicksort_wrapper_with_memory, mergesort_wrapper_with_memory
from datagen import (
    generate_random,
    generate_nearly_sorted,
    generate_reverse_sorted,
    generate_duplicate_heavy
)

SIZES = [10000, 50000, 100000, 500000]
STRUCTURES = ["random", "nearly_sorted", "reverse", "duplicate_heavy"]
TRIALS = 5

RESULTS_DIR = "../results"
os.makedirs(RESULTS_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(RESULTS_DIR, 
    f"memory_analysis_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv")

def run_memory_analysis():
    print("=" * 80)
    print("MEMORY USAGE ANALYSIS (tracemalloc)")
    print("=" * 80)
    print("Note: Timing data from this run is invalid due to profiling overhead")
    print("      Use only for memory consumption analysis")
    print("=" * 80)
    
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "size", "structure", "algorithm",
            "mean_memory_MB", "std_memory_MB",
            "mean_comparisons", "std_comparisons"
        ])
        
        for size in SIZES:
            print(f"\nSize {size:,}:")
            
            for structure in STRUCTURES:
                print(f"  {structure:20s}: ", end="", flush=True)
                
                qs_mems = []
                qs_comps = []
                ms_mems = []
                ms_comps = []
                
                for trial in range(TRIALS):
                    # Generate dataset
                    if structure == "random":
                        dataset = generate_random(size)
                    elif structure == "nearly_sorted":
                        dataset = generate_nearly_sorted(size, disorder_fraction=0.1)
                    elif structure == "reverse":
                        dataset = generate_reverse_sorted(size)
                    else:
                        dataset = generate_duplicate_heavy(size, duplicate_ratio=0.5)
                    
                    try:
                        # QuickSort with memory
                        _, qs_c, qs_m_dict = quicksort_wrapper_with_memory(dataset)
                        qs_mems.append(qs_m_dict['tracemalloc_mb'])
                        qs_comps.append(qs_c)
                        print("✓", end="", flush=True)
                        
                        # MergeSort with memory
                        _, ms_c, ms_m_dict = mergesort_wrapper_with_memory(dataset)
                        ms_mems.append(ms_m_dict['tracemalloc_mb'])
                        ms_comps.append(ms_c)
                        print("✓", end="", flush=True)
                        
                    except Exception as e:
                        print("✗✗", end="", flush=True)
                
                print(f" ({len(qs_mems)}QS/{len(ms_mems)}MS)")
                
                # Write statistics
                if qs_mems:
                    writer.writerow([
                        size, structure, "QuickSort",
                        mean(qs_mems), stdev(qs_mems) if len(qs_mems) > 1 else 0,
                        mean(qs_comps), stdev(qs_comps) if len(qs_comps) > 1 else 0
                    ])
                
                if ms_mems:
                    writer.writerow([
                        size, structure, "MergeSort",
                        mean(ms_mems), stdev(ms_mems) if len(ms_mems) > 1 else 0,
                        mean(ms_comps), stdev(ms_comps) if len(ms_comps) > 1 else 0
                    ])
    
    print("\n" + "=" * 80)
    print("MEMORY ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Output: {OUTPUT_CSV}")
    print("=" * 80)

if __name__ == "__main__":
    run_memory_analysis()