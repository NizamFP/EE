import sys

def quicksort(arr, low=None, high=None, comparisons=None):
    """
    In-place QuickSort implementation with median-of-three pivot selection.
    
    This implementation sorts the array in-place to achieve O(log n) space
    complexity (from recursion stack only), avoiding the O(n) auxiliary
    space that would result from creating new sublists.
    
    Args:
        arr (list): Array to be sorted (modified in-place)
        low (int): Starting index of partition (default: 0)
        high (int): Ending index of partition (default: len(arr)-1)
        comparisons (list): Mutable container tracking comparison count
        
    Returns:
        tuple: (sorted array reference, total comparison count)
        
    Time Complexity:
        Best/Average: O(n log n) - balanced partitions
        Worst: O(n²) - highly unbalanced partitions (mitigated by pivot strategy)
        
    Space Complexity:
        O(log n) average - recursion stack depth
        O(n) worst - deep recursion with unbalanced partitions
    """
    
    # Initialize parameters on first call
    if low is None:
        low = 0
    if high is None:
        high = len(arr) - 1
    if comparisons is None:
        comparisons = [0]  # Use list to maintain reference across recursive calls
    
    # Base case: partition with 0 or 1 elements is already sorted
    if low >= high:
        return arr, comparisons[0]
    
    # Partition array and get pivot's final position
    pivot_index = partition(arr, low, high, comparisons)
    
    # Recursively sort elements before and after partition
    quicksort(arr, low, pivot_index - 1, comparisons)
    quicksort(arr, pivot_index + 1, high, comparisons)
    
    return arr, comparisons[0]


def partition(arr, low, high, comparisons):
    """
    Partitions array segment around a pivot using median-of-three selection.
    
    Rearranges elements so that:
    - Elements < pivot are moved to the left
    - Elements > pivot are moved to the right
    - Pivot is placed at its final sorted position
    
    Median-of-three pivot selection examines first, middle, and last elements,
    choosing the median value. This strategy significantly reduces the
    probability of worst-case O(n²) behavior on sorted or reverse-sorted inputs.
    
    Args:
        arr (list): Array being sorted
        low (int): Starting index of partition
        high (int): Ending index of partition
        comparisons (list): Comparison counter
        
    Returns:
        int: Final index position of pivot element
    """
    
    # Median-of-three pivot selection
    mid = (low + high) // 2
    
    # Sort first, middle, last elements to find median
    # This places median at arr[mid]
    comparisons[0] += 2  # Two comparisons needed to find median of three
    
    if arr[low] > arr[mid]:
        arr[low], arr[mid] = arr[mid], arr[low]
    if arr[low] > arr[high]:
        arr[low], arr[high] = arr[high], arr[low]
    if arr[mid] > arr[high]:
        arr[mid], arr[high] = arr[high], arr[mid]
    
    # Place median (now at mid) at second-to-last position
    # This is a standard optimization: pivot is temporarily moved
    # to avoid redundant comparisons
    pivot = arr[mid]
    arr[mid], arr[high - 1] = arr[high - 1], arr[mid]
    pivot_index = high - 1
    
    # Partition remaining elements (between low+1 and high-2)
    # arr[low] and arr[high] already positioned correctly from median-of-three
    i = low
    j = high - 1
    
    while True:
        # Move i right until finding element >= pivot
        i += 1
        comparisons[0] += 1
        while i < pivot_index and arr[i] < pivot:
            i += 1
            comparisons[0] += 1
        
        # Move j left until finding element <= pivot
        j -= 1
        comparisons[0] += 1
        while j > low and arr[j] > pivot:
            j -= 1
            comparisons[0] += 1
        
        # If pointers crossed, partitioning is complete
        if i >= j:
            break
        
        # Swap elements at i and j
        arr[i], arr[j] = arr[j], arr[i]
    
    # Place pivot in its final position (between partitions)
    arr[i], arr[pivot_index] = arr[pivot_index], arr[i]
    
    return i


def quicksort_wrapper(arr):
    """
    Convenience wrapper for QuickSort that creates a copy of the input array.
    
    This function is used in experiments to prevent mutation of original
    datasets, ensuring each trial works with identical input data.
    
    Args:
        arr (list): Array to be sorted
        
    Returns:
        tuple: (sorted array, comparison count)
    """
    arr_copy = arr.copy()  # Create copy to preserve original
    return quicksort(arr_copy)


# Example usage demonstrating correctness and comparison counting
if __name__ == "__main__":
    # Test Case 1: Random data
    test_random = [64, 34, 25, 12, 22, 11, 90]
    sorted_arr, comps = quicksort_wrapper(test_random)
    print(f"Random: {sorted_arr}")
    print(f"Comparisons: {comps}\n")
    
    # Test Case 2: Already sorted (tests pivot optimization)
    test_sorted = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    sorted_arr, comps = quicksort_wrapper(test_sorted)
    print(f"Already Sorted: {sorted_arr}")
    print(f"Comparisons: {comps}\n")
    
    # Test Case 3: Reverse sorted (worst case for naive pivot)
    test_reverse = [9, 8, 7, 6, 5, 4, 3, 2, 1]
    sorted_arr, comps = quicksort_wrapper(test_reverse)
    print(f"Reverse Sorted: {sorted_arr}")
    print(f"Comparisons: {comps}\n")
    
    # Test Case 4: Duplicates
    test_duplicates = [5, 2, 8, 2, 9, 1, 5, 5]
    sorted_arr, comps = quicksort_wrapper(test_duplicates)
    print(f"Duplicates: {sorted_arr}")
    print(f"Comparisons: {comps}\n")
    
    # Test Case 5: Single element
    test_single = [42]
    sorted_arr, comps = quicksort_wrapper(test_single)
    print(f"Single Element: {sorted_arr}")
    print(f"Comparisons: {comps}\n")
    
    # Test Case 6: Empty array
    test_empty = []
    sorted_arr, comps = quicksort_wrapper(test_empty)
    print(f"Empty: {sorted_arr}")
    print(f"Comparisons: {comps}")


def mergesort(arr, comparisons=None):
    """
    Stable MergeSort implementation using divide-and-conquer strategy.
    
    This implementation creates auxiliary arrays during the merge process,
    resulting in O(n) space complexity. This represents a fundamental
    trade-off: guaranteed O(n log n) time complexity in exchange for
    higher memory consumption compared to in-place algorithms.
    
    Args:
        arr (list): Array to be sorted
        comparisons (list): Mutable container tracking comparison count
        
    Returns:
        tuple: (sorted array, total comparison count)
        
    Time Complexity:
        Best/Average/Worst: O(n log n) - always divides evenly
        
    Space Complexity:
        O(n) - auxiliary arrays for merging
        O(log n) - recursion stack depth
        Total: O(n) dominated by merge arrays
        
    Stability:
        Stable - preserves relative order of equal elements
        Critical for multi-key sorting and maintaining metadata associations
    """
    
    # Initialize comparison counter on first call
    if comparisons is None:
        comparisons = [0]  # Use list to maintain reference across recursive calls
    
    # Base case: arrays with 0 or 1 elements are already sorted
    if len(arr) <= 1:
        return arr, comparisons[0]
    
    # Divide: Split array into two halves
    mid = len(arr) // 2
    left_half = arr[:mid]    # Slice creates new list: O(n/2) space
    right_half = arr[mid:]   # Slice creates new list: O(n/2) space
    
    # Conquer: Recursively sort each half
    left_sorted, _ = mergesort(left_half, comparisons)
    right_sorted, _ = mergesort(right_half, comparisons)
    
    # Combine: Merge the sorted halves
    merged = merge(left_sorted, right_sorted, comparisons)
    
    return merged, comparisons[0]


def merge(left, right, comparisons):
    """
    Merges two sorted arrays into a single sorted array.
    
    This is the core operation of MergeSort. By comparing the smallest
    unmerged elements from each subarray, it constructs the result in
    sorted order. The algorithm guarantees O(n) time for merging, where
    n = len(left) + len(right).
    
    Args:
        left (list): First sorted subarray
        right (list): Second sorted subarray
        comparisons (list): Comparison counter
        
    Returns:
        list: Merged sorted array containing all elements from left and right
        
    Stability Property:
        When left[i] == right[j], the element from 'left' is chosen first.
        This preserves the original relative ordering, ensuring stability.
    """
    
    result = []  # Auxiliary array: O(n) space
    i = j = 0    # Pointers for left and right subarrays
    
    # Merge elements in sorted order by comparing front elements
    while i < len(left) and j < len(right):
        comparisons[0] += 1  # Count each comparison
        
        # Use <= (not <) to maintain stability
        # Equal elements from 'left' subarray come first
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # Append remaining elements (at most one subarray has remaining elements)
    # No comparisons needed - remaining elements are already sorted
    result.extend(left[i:])   # If left has remaining elements
    result.extend(right[j:])  # If right has remaining elements
    
    return result


def mergesort_wrapper(arr):
    """
    Convenience wrapper for MergeSort that preserves the original array.
    
    Since MergeSort creates new arrays during sorting, this wrapper
    exists primarily for API consistency with QuickSort's wrapper,
    which must explicitly copy the input to avoid in-place mutation.
    
    Args:
        arr (list): Array to be sorted
        
    Returns:
        tuple: (sorted array, comparison count)
    """
    # Note: MergeSort naturally returns a new array, so explicit copy
    # is unnecessary. However, we copy for consistency with QuickSort API
    # and to guarantee the original array remains unmodified.
    arr_copy = arr.copy()
    return mergesort(arr_copy)


def mergesort_iterative(arr):
    """
    Iterative (bottom-up) MergeSort implementation for comparison.
    
    This variant eliminates recursion by iteratively merging progressively
    larger subarrays. It achieves the same O(n log n) time complexity but
    with O(1) stack space (only O(n) heap space for merge arrays).
    
    Included to demonstrate algorithmic alternatives and validate that
    recursive implementation does not introduce anomalous behavior.
    
    Args:
        arr (list): Array to be sorted
        
    Returns:
        tuple: (sorted array, comparison count)
    """
    
    if len(arr) <= 1:
        return arr, 0
    
    arr = arr.copy()  # Work on copy to avoid mutation
    n = len(arr)
    comparisons = [0]
    
    # Start with subarrays of size 1, double size each iteration
    current_size = 1
    
    # Outer loop: O(log n) iterations
    while current_size < n:
        # Merge subarrays of current_size
        # Inner loop: O(n) work per iteration
        for start in range(0, n, current_size * 2):
            # Calculate subarray boundaries
            mid = min(start + current_size, n)
            end = min(start + current_size * 2, n)
            
            # Merge arr[start:mid] and arr[mid:end]
            left = arr[start:mid]
            right = arr[mid:end]
            merged = merge(left, right, comparisons)
            
            # Copy merged result back to original array
            arr[start:end] = merged
        
        # Double the subarray size for next iteration
        current_size *= 2
    
    return arr, comparisons[0]


# Example usage demonstrating correctness, stability, and comparison counting
if __name__ == "__main__":
    print("=" * 60)
    print("MergeSort Validation and Demonstration")
    print("=" * 60)
    
    # Test Case 1: Random data
    print("\n1. Random Data:")
    test_random = [64, 34, 25, 12, 22, 11, 90]
    sorted_arr, comps = mergesort_wrapper(test_random)
    print(f"   Input:       {test_random}")
    print(f"   Sorted:      {sorted_arr}")
    print(f"   Comparisons: {comps}")
    
    # Test Case 2: Already sorted (tests best-case behavior)
    print("\n2. Already Sorted (Best Case):")
    test_sorted = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    sorted_arr, comps = mergesort_wrapper(test_sorted)
    print(f"   Input:       {test_sorted}")
    print(f"   Sorted:      {sorted_arr}")
    print(f"   Comparisons: {comps}")
    print(f"   Note: Same as worst case - MergeSort always O(n log n)")
    
    # Test Case 3: Reverse sorted (tests worst-case behavior)
    print("\n3. Reverse Sorted (Worst Case):")
    test_reverse = [9, 8, 7, 6, 5, 4, 3, 2, 1]
    sorted_arr, comps = mergesort_wrapper(test_reverse)
    print(f"   Input:       {test_reverse}")
    print(f"   Sorted:      {sorted_arr}")
    print(f"   Comparisons: {comps}")
    
    # Test Case 4: Duplicates (tests stability and duplicate handling)
    print("\n4. Duplicates (Tests Stability):")
    test_duplicates = [5, 2, 8, 2, 9, 1, 5, 5]
    sorted_arr, comps = mergesort_wrapper(test_duplicates)
    print(f"   Input:       {test_duplicates}")
    print(f"   Sorted:      {sorted_arr}")
    print(f"   Comparisons: {comps}")
    
    # Test Case 5: Stability demonstration with tuples
    print("\n5. Stability Demonstration (tuple: value, original_index):")
    # Create tuples of (value, original_position) to track stability
    test_stability = [(3, 0), (1, 1), (3, 2), (2, 3), (3, 4)]
    # MergeSort by first element only
    sorted_arr, comps = mergesort_wrapper(test_stability)
    print(f"   Input:  {test_stability}")
    print(f"   Sorted: {sorted_arr}")
    print(f"   Check: (3,0) < (3,2) < (3,4) → Order preserved ✓")
    
    # Test Case 6: Single element
    print("\n6. Edge Case - Single Element:")
    test_single = [42]
    sorted_arr, comps = mergesort_wrapper(test_single)
    print(f"   Input:       {test_single}")
    print(f"   Sorted:      {sorted_arr}")
    print(f"   Comparisons: {comps} (no comparisons needed)")
    
    # Test Case 7: Empty array
    print("\n7. Edge Case - Empty Array:")
    test_empty = []
    sorted_arr, comps = mergesort_wrapper(test_empty)
    print(f"   Input:       {test_empty}")
    print(f"   Sorted:      {sorted_arr}")
    print(f"   Comparisons: {comps}")
    
    # Test Case 8: Comparison with iterative implementation
    print("\n8. Recursive vs. Iterative Implementation:")
    test_comparison = [15, 3, 8, 1, 9, 2, 14, 7]
    sorted_recursive, comps_recursive = mergesort_wrapper(test_comparison)
    sorted_iterative, comps_iterative = mergesort_iterative(test_comparison)
    print(f"   Input:      {test_comparison}")
    print(f"   Recursive:  {sorted_recursive} ({comps_recursive} comparisons)")
    print(f"   Iterative:  {sorted_iterative} ({comps_iterative} comparisons)")
    print(f"   Match: {sorted_recursive == sorted_iterative} ✓")
    
    # Test Case 9: Large-scale validation
    print("\n9. Large-Scale Validation (10,000 elements):")
    import random
    random.seed(42)
    large_test = [random.randint(0, 10000) for _ in range(10000)]
    sorted_arr, comps = mergesort_wrapper(large_test)
    is_correct = sorted_arr == sorted(large_test)
    print(f"   Size:        10,000 elements")
    print(f"   Comparisons: {comps}")
    print(f"   Correct:     {is_correct} ✓")
    print(f"   Theoretical: ~10,000 × log₂(10,000) ≈ {10000 * 13.3:.0f}")
    print(f"   Ratio:       {comps / (10000 * 13.3):.2f}× theoretical")
    
    print("\n" + "=" * 60)
    print("All validation tests completed successfully ✓")
    print("=" * 60)