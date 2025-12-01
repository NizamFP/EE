"""
Sorting Algorithm Implementations for IB CS Extended Essay
QuickSort (in-place with median-of-three) and MergeSort (recursive)

Provides two wrapper variants:
1. Standard wrappers (no memory tracking) - for accurate timing
2. Memory-tracking wrappers (with tracemalloc) - for memory analysis

Author: [Candidate Number]
Date: November 2024
Purpose: IB Computer Science Extended Essay (May 2026)
Platform: GitHub Codespaces (Linux) / Python 3.12+
"""

import sys
import tracemalloc
import psutil
import os

# Increase recursion limit for large datasets
sys.setrecursionlimit(100000)


# ==========================================
# QUICKSORT IMPLEMENTATION
# ==========================================

def quicksort(arr, low=None, high=None, comparisons=None):
    """
    In-place QuickSort with median-of-three pivot selection.
    
    Args:
        arr (list): Array to sort (modified in-place)
        low (int): Starting index (default: 0)
        high (int): Ending index (default: len(arr)-1)
        comparisons (list): Mutable counter for comparisons
    
    Returns:
        tuple: (sorted array reference, total comparisons)
    
    Time Complexity:
        Best/Average: O(n log n) with median-of-three
        Worst: O(n²) - mitigated but not eliminated
    
    Space Complexity:
        Average: O(log n) - recursion stack
        Worst: O(n) - deep recursion on structured inputs
    """
    
    # Initialize parameters on first call
    if low is None:
        low = 0
    if high is None:
        high = len(arr) - 1
    if comparisons is None:
        comparisons = [0]
    
    # Base case: 0 or 1 elements already sorted
    if low >= high:
        return arr, comparisons[0]
    
    # Partition array and recursively sort
    pivot_index = partition(arr, low, high, comparisons)
    quicksort(arr, low, pivot_index - 1, comparisons)
    quicksort(arr, pivot_index + 1, high, comparisons)
    
    return arr, comparisons[0]


def partition(arr, low, high, comparisons):
    """
    Partition array around median-of-three pivot.
    
    Args:
        arr (list): Array to partition
        low (int): Starting index
        high (int): Ending index
        comparisons (list): Comparison counter
    
    Returns:
        int: Final pivot position
    """
    
    # Handle small subarrays (size < 3)
    if high - low < 2:
        comparisons[0] += 1
        if arr[low] > arr[high]:
            arr[low], arr[high] = arr[high], arr[low]
        return high
    
    mid = (low + high) // 2
    
    # Median-of-three: order first, middle, last elements
    comparisons[0] += 2
    if arr[low] > arr[mid]:
        arr[low], arr[mid] = arr[mid], arr[low]
    if arr[low] > arr[high]:
        arr[low], arr[high] = arr[high], arr[low]
    if arr[mid] > arr[high]:
        arr[mid], arr[high] = arr[high], arr[mid]
    
    # Now: arr[low] ≤ arr[mid] ≤ arr[high]
    # Use middle element as pivot
    pivot = arr[mid]
    
    # Move pivot to second-to-last position
    arr[mid], arr[high - 1] = arr[high - 1], arr[mid]
    pivot_index = high - 1
    
    # Partition remaining elements
    i = low
    j = high - 1
    
    while True:
        # Move i right until finding element ≥ pivot
        i += 1
        while i < pivot_index and arr[i] < pivot:
            comparisons[0] += 1
            i += 1
        
        # Move j left until finding element ≤ pivot
        j -= 1
        while j > low and arr[j] > pivot:
            comparisons[0] += 1
            j -= 1
        
        # If pointers crossed, partitioning complete
        if i >= j:
            break
        
        # Swap elements
        arr[i], arr[j] = arr[j], arr[i]
        comparisons[0] += 1
    
    # Place pivot in final position
    arr[i], arr[pivot_index] = arr[pivot_index], arr[i]
    
    return i


def quicksort_wrapper(arr):
    """
    Standard wrapper for QuickSort (no memory tracking).
    
    Use this for accurate timing measurements.
    
    Args:
        arr (list): Array to sort
    
    Returns:
        tuple: (sorted array, comparison count)
    """
    if not arr:
        return [], 0
    
    arr_copy = arr.copy()
    return quicksort(arr_copy)


def quicksort_wrapper_with_memory(arr):
    """
    QuickSort wrapper WITH memory tracking (tracemalloc).
    
    WARNING: Introduces ~12× performance overhead!
    Use ONLY for memory analysis, NOT for timing comparison.
    
    Args:
        arr (list): Array to sort
    
    Returns:
        tuple: (sorted array, comparison count, memory_dict)
        
    memory_dict contains:
        - 'tracemalloc_mb': Python allocations (precise)
        - 'psutil_mb': Process RSS growth (includes overhead)
    """
    if not arr:
        return [], 0, {'tracemalloc_mb': 0.0, 'psutil_mb': 0.0}
    
    arr_copy = arr.copy()
    
    # Baseline process memory
    process = psutil.Process(os.getpid())
    baseline_rss = process.memory_info().rss
    
    # Start tracemalloc
    tracemalloc.start()
    
    # Execute QuickSort
    sorted_arr, comparisons = quicksort(arr_copy)
    
    # Capture memory measurements
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    final_rss = process.memory_info().rss
    
    memory_dict = {
        'tracemalloc_mb': peak_mem / (1024 * 1024),
        'psutil_mb': (final_rss - baseline_rss) / (1024 * 1024),
    }
    
    return sorted_arr, comparisons, memory_dict


# ==========================================
# MERGESORT IMPLEMENTATION
# ==========================================

def mergesort(arr, comparisons=None):
    """
    Recursive MergeSort implementation.
    
    Args:
        arr (list): Array to sort
        comparisons (list): Mutable counter for comparisons
    
    Returns:
        tuple: (sorted array, total comparisons)
    
    Time Complexity:
        All cases: O(n log n) - always divides evenly
    
    Space Complexity:
        O(n) - auxiliary arrays for merging
        O(log n) - recursion stack
        Total: O(n) dominated by merge arrays
    """
    
    if comparisons is None:
        comparisons = [0]
    
    # Base case: 0 or 1 elements already sorted
    if len(arr) <= 1:
        return arr, comparisons[0]
    
    # Divide: split at midpoint
    mid = len(arr) // 2
    left_half = arr[:mid]
    right_half = arr[mid:]
    
    # Conquer: recursively sort halves
    left_sorted, _ = mergesort(left_half, comparisons)
    right_sorted, _ = mergesort(right_half, comparisons)
    
    # Combine: merge sorted halves
    merged = merge(left_sorted, right_sorted, comparisons)
    
    return merged, comparisons[0]


def merge(left, right, comparisons):
    """
    Merge two sorted arrays into one sorted array.
    
    Args:
        left (list): First sorted array
        right (list): Second sorted array
        comparisons (list): Comparison counter
    
    Returns:
        list: Merged sorted array
    """
    
    result = []
    i = j = 0
    
    # Merge while both arrays have elements
    while i < len(left) and j < len(right):
        comparisons[0] += 1
        
        # Use ≤ (not <) to maintain stability
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # Append remaining elements (no comparisons needed)
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result


def mergesort_wrapper(arr):
    """
    Standard wrapper for MergeSort (no memory tracking).
    
    Use this for accurate timing measurements.
    
    Args:
        arr (list): Array to sort
    
    Returns:
        tuple: (sorted array, comparison count)
    """
    if not arr:
        return [], 0
    
    arr_copy = arr.copy()
    return mergesort(arr_copy)


def mergesort_wrapper_with_memory(arr):
    """
    MergeSort wrapper WITH memory tracking (tracemalloc).
    
    WARNING: Introduces ~6× performance overhead!
    Use ONLY for memory analysis, NOT for timing comparison.
    
    Args:
        arr (list): Array to sort
    
    Returns:
        tuple: (sorted array, comparison count, memory_dict)
    """
    if not arr:
        return [], 0, {'tracemalloc_mb': 0.0, 'psutil_mb': 0.0}
    
    arr_copy = arr.copy()
    
    # Baseline process memory
    process = psutil.Process(os.getpid())
    baseline_rss = process.memory_info().rss
    
    # Start tracemalloc
    tracemalloc.start()
    
    # Execute MergeSort
    sorted_arr, comparisons = mergesort(arr_copy)
    
    # Capture memory measurements
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    final_rss = process.memory_info().rss
    
    memory_dict = {
        'tracemalloc_mb': peak_mem / (1024 * 1024),
        'psutil_mb': (final_rss - baseline_rss) / (1024 * 1024),
    }
    
    return sorted_arr, comparisons, memory_dict


# ==========================================
# VALIDATION AND TESTING
# ==========================================

def validate_sorting_algorithms():
    """
    Comprehensive validation suite for both algorithms.
    """
    import random
    
    print("=" * 70)
    print("SORTING ALGORITHMS VALIDATION SUITE")
    print("=" * 70)
    
    # Basic correctness tests
    test_cases = [
        ([], "Empty array"),
        ([1], "Single element"),
        ([2, 1], "Two elements"),
        ([1, 2, 3], "Already sorted"),
        ([3, 2, 1], "Reverse sorted"),
        ([1, 1, 1], "All duplicates"),
        ([3, 1, 4, 1, 5, 9, 2, 6], "Mixed with duplicates"),
    ]
    
    print("\n1. Basic Correctness Tests:")
    print("-" * 70)
    
    all_passed = True
    
    for test_input, description in test_cases:
        expected = sorted(test_input)
        
        qs_result, _ = quicksort_wrapper(test_input.copy())
        ms_result, _ = mergesort_wrapper(test_input.copy())
        
        qs_correct = (qs_result == expected)
        ms_correct = (ms_result == expected)
        
        if qs_correct and ms_correct:
            print(f"  ✓ {description:25s}")
        else:
            print(f"  ✗ {description:25s}: QS={qs_correct}, MS={ms_correct}")
            all_passed = False
    
    # Stress test
    print("\n2. Stress Test (100 random arrays):")
    print("-" * 70)
    
    random.seed(42)
    stress_passed = 0
    
    for i in range(100):
        size = random.randint(0, 1000)
        test_arr = [random.randint(-1000, 1000) for _ in range(size)]
        expected = sorted(test_arr)
        
        qs_result, _ = quicksort_wrapper(test_arr.copy())
        ms_result, _ = mergesort_wrapper(test_arr.copy())
        
        if qs_result == expected and ms_result == expected:
            stress_passed += 1
        else:
            print(f"  ✗ Failed on size {size}")
            all_passed = False
            break
    
    print(f"  Passed: {stress_passed}/100")
    
    # Memory tracking test
    print("\n3. Memory Tracking Test (10,000 elements):")
    print("-" * 70)
    
    random.seed(99)
    test_arr = [random.randint(0, 10000) for _ in range(10000)]
    
    qs_sorted, qs_comps, qs_mem = quicksort_wrapper_with_memory(test_arr.copy())
    ms_sorted, ms_comps, ms_mem = mergesort_wrapper_with_memory(test_arr.copy())
    
    print(f"  QuickSort:")
    print(f"    Comparisons: {qs_comps:,}")
    print(f"    Memory (tracemalloc): {qs_mem['tracemalloc_mb']:.4f} MB")
    print(f"    Memory (psutil): {qs_mem['psutil_mb']:.4f} MB")
    print(f"    Correct: {qs_sorted == sorted(test_arr)}")
    
    print(f"\n  MergeSort:")
    print(f"    Comparisons: {ms_comps:,}")
    print(f"    Memory (tracemalloc): {ms_mem['tracemalloc_mb']:.4f} MB")
    print(f"    Memory (psutil): {ms_mem['psutil_mb']:.4f} MB")
    print(f"    Correct: {ms_sorted == sorted(test_arr)}")
    
    print(f"\n  Memory Ratio: {ms_mem['tracemalloc_mb'] / qs_mem['tracemalloc_mb']:.0f}× "
          f"(MergeSort uses more)")
    
    # Final summary
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL VALIDATION TESTS PASSED")
        print("Both algorithms ready for experimental use")
    else:
        print("✗ SOME TESTS FAILED - Review implementation")
    print("=" * 70)
    
    return all_passed


# ==========================================
# PERFORMANCE COMPARISON TEST
# ==========================================

def compare_wrapper_overhead():
    """
    Demonstrate the overhead difference between standard and memory wrappers.
    """
    import random
    import time
    
    print("\n" + "=" * 70)
    print("WRAPPER OVERHEAD COMPARISON")
    print("=" * 70)
    
    random.seed(42)
    test_sizes = [1000, 10000, 50000]
    
    for size in test_sizes:
        print(f"\nDataset size: {size:,} elements")
        print("-" * 70)
        
        dataset = [random.randint(0, size) for _ in range(size)]
        
        # Standard wrapper (no memory tracking)
        start = time.perf_counter()
        qs_result1, qs_comps1 = quicksort_wrapper(dataset.copy())
        qs_time_standard = time.perf_counter() - start
        
        start = time.perf_counter()
        ms_result1, ms_comps1 = mergesort_wrapper(dataset.copy())
        ms_time_standard = time.perf_counter() - start
        
        # Memory wrapper (with tracemalloc)
        start = time.perf_counter()
        qs_result2, qs_comps2, qs_mem = quicksort_wrapper_with_memory(dataset.copy())
        qs_time_memory = time.perf_counter() - start
        
        start = time.perf_counter()
        ms_result2, ms_comps2, ms_mem = mergesort_wrapper_with_memory(dataset.copy())
        ms_time_memory = time.perf_counter() - start
        
        # Calculate overhead
        qs_overhead = qs_time_memory / qs_time_standard
        ms_overhead = ms_time_memory / ms_time_standard
        
        print(f"  QuickSort:")
        print(f"    Standard wrapper: {qs_time_standard:.4f}s")
        print(f"    Memory wrapper:   {qs_time_memory:.4f}s")
        print(f"    Overhead factor:  {qs_overhead:.1f}×")
        
        print(f"\n  MergeSort:")
        print(f"    Standard wrapper: {ms_time_standard:.4f}s")
        print(f"    Memory wrapper:   {ms_time_memory:.4f}s")
        print(f"    Overhead factor:  {ms_overhead:.1f}×")
        
        print(f"\n  Comparison:")
        print(f"    Standard: QS/MS ratio = {ms_time_standard/qs_time_standard:.2f}× "
              f"({'QS faster' if ms_time_standard > qs_time_standard else 'MS faster'})")
        print(f"    With tracemalloc: QS/MS ratio = {ms_time_memory/qs_time_memory:.2f}× "
              f"({'QS faster' if ms_time_memory > qs_time_memory else 'MS faster'})")
    
    print("\n" + "=" * 70)
    print("CONCLUSION:")
    print("  • tracemalloc introduces 6-12× overhead")
    print("  • Overhead affects QuickSort MORE (recursive allocations)")
    print("  • Use standard wrappers for timing, memory wrappers for space analysis")
    print("=" * 70)


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    # Run validation
    validate_sorting_algorithms()
    
    # Demonstrate overhead
    compare_wrapper_overhead()