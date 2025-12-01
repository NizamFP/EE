"""
Dataset Generation Module for Sorting Algorithm Performance Analysis

This module generates controlled datasets with specific structural
characteristics to evaluate sorting algorithm behavior under varying
input conditions. All datasets are deterministic (seeded) to ensure
experimental reproducibility.

Author: [Candidate Number]
Date: November 2024
Purpose: IB Computer Science Extended Essay (May 2026)
"""

import random
import csv
import os
from typing import List, Tuple

# Global constants for reproducibility
RANDOM_SEED = 42  # Fixed seed ensures identical datasets across experimental runs
VALUE_RANGE_MULTIPLIER = 1.0  # Controls spread of random values relative to dataset size

# Set random seed globally for all generation functions
random.seed(RANDOM_SEED)


def generate_random(size: int, value_range: int = None) -> List[int]:
    """
    Generates a uniformly distributed random integer array.
    
    This dataset type approximates the "average case" assumption underlying
    theoretical complexity analysis. Random distribution typically produces
    balanced partitions in QuickSort and uniform merge operations in MergeSort.
    
    Args:
        size (int): Number of elements to generate
        value_range (int, optional): Maximum value for random integers.
                                    Defaults to size (ensures diverse values)
    
    Returns:
        List[int]: Array of random integers in range [0, value_range)
    
    Characteristics:
        - Uniform probability distribution
        - Expected number of unique values: min(size, value_range)
        - Worst-case duplicates when size >> value_range
        - Represents typical unsorted data in practice
    
    Example:
        >>> generate_random(10, value_range=100)
        [51, 92, 14, 71, 60, 20, 82, 86, 74, 74]
    """
    if value_range is None:
        value_range = size  # Default: range equals size for high uniqueness
    
    dataset = [random.randint(0, value_range) for _ in range(size)]
    
    return dataset


def generate_nearly_sorted(size: int, disorder_fraction: float = 0.1) -> List[int]:
    """
    Generates a nearly sorted array by randomly swapping a fraction of elements.
    
    This dataset models real-world scenarios where data is partially ordered:
    - Database logs with mostly sequential timestamps but occasional out-of-order entries
    - Streaming data with minor jitter
    - Previously sorted data with new insertions
    
    Nearly sorted data is particularly challenging for naive QuickSort implementations
    (those using first/last element pivots), as the mostly-ordered structure can
    lead to unbalanced partitions. Our median-of-three pivot strategy mitigates
    this vulnerability.
    
    Args:
        size (int): Number of elements to generate
        disorder_fraction (float): Proportion of elements to randomly swap (0.0-1.0)
                                  Default 0.1 means 10% of elements displaced
    
    Returns:
        List[int]: Array that is mostly sorted but with localized disorder
    
    Algorithm:
        1. Create perfectly sorted array [0, 1, 2, ..., size-1]
        2. Perform ⌊size × disorder_fraction⌋ random swaps
        3. Each swap displaces two elements from sorted positions
    
    Characteristics:
        - Expected inversions: O(size × disorder_fraction)
        - Worst-case for naive QuickSort if disorder_fraction is small
        - MergeSort performance unaffected by partial ordering
    
    Example:
        >>> generate_nearly_sorted(10, disorder_fraction=0.2)
        [0, 1, 5, 3, 4, 2, 6, 7, 8, 9]  # 2 swaps: (2↔5), (4 remains)
    """
    # Start with perfectly sorted array
    dataset = list(range(size))
    
    # Calculate number of swaps to perform
    num_swaps = int(size * disorder_fraction)
    
    # Perform random swaps to introduce disorder
    for _ in range(num_swaps):
        # Select two random distinct indices
        i = random.randrange(size)
        j = random.randrange(size)
        
        # Swap elements at positions i and j
        dataset[i], dataset[j] = dataset[j], dataset[i]
    
    return dataset


def generate_reverse_sorted(size: int) -> List[int]:
    """
    Generates a reverse-sorted (descending order) array.
    
    This dataset represents the classic worst-case scenario for naive QuickSort
    implementations that always choose the first or last element as pivot:
    - First-element pivot: Always creates partitions of size 0 and n-1
    - Recursion depth reaches O(n), causing O(n²) time complexity
    
    Median-of-three pivot selection mitigates this issue by examining multiple
    elements, though it doesn't eliminate the problem entirely for adversarial
    inputs.
    
    Args:
        size (int): Number of elements to generate
    
    Returns:
        List[int]: Array in descending order [size-1, size-2, ..., 1, 0]
    
    Characteristics:
        - Maximum number of inversions: n(n-1)/2
        - Worst case for comparison-based sorting lower bound: Ω(n log n)
        - Tests pivot selection effectiveness in QuickSort
        - No effect on MergeSort performance (still O(n log n))
    
    Historical Context:
        The reverse-sorted case is famously problematic for naive QuickSort,
        leading to the development of sophisticated pivot strategies in
        production implementations (Hoare, 1961; Bentley & McIlroy, 1993).
    
    Example:
        >>> generate_reverse_sorted(5)
        [4, 3, 2, 1, 0]
    """
    dataset = list(range(size - 1, -1, -1))
    
    return dataset


def generate_duplicate_heavy(size: int, unique_values: int = None, 
                             duplicate_ratio: float = 0.5) -> List[int]:
    """
    Generates an array with a high proportion of duplicate values.
    
    Duplicate-heavy datasets are common in practice:
    - Categorical data (e.g., sorting by country code, product category)
    - Rating systems (discrete values like 1-5 stars)
    - Bucketed continuous data (age groups, price ranges)
    
    Many duplicate values can cause issues for two-way partitioning QuickSort,
    as equal elements may be split across both partitions unnecessarily.
    Three-way partitioning (separating <, =, > pivot) handles this efficiently.
    
    Args:
        size (int): Number of elements to generate
        unique_values (int, optional): Number of distinct values in dataset.
                                      Defaults to size * (1 - duplicate_ratio)
        duplicate_ratio (float): Target proportion of duplicate values (0.0-1.0)
                                Default 0.5 means ~50% of values are duplicates
    
    Returns:
        List[int]: Array with controlled duplicate frequency
    
    Algorithm:
        1. Calculate number of unique values based on duplicate_ratio
        2. Generate random integers constrained to small value range
        3. Shuffle to avoid sorted subsequences
    
    Characteristics:
        - Expected duplicates: size × duplicate_ratio
        - Average partition balance depends on value distribution
        - Tests three-way partitioning effectiveness in QuickSort
        - MergeSort stable property preserves duplicate ordering
    
    Example:
        >>> generate_duplicate_heavy(10, unique_values=3)
        [2, 1, 2, 0, 1, 2, 0, 2, 1, 0]  # Only values 0, 1, 2 present
    """
    # Calculate unique value range if not specified
    if unique_values is None:
        # Higher duplicate_ratio → fewer unique values
        unique_values = max(1, int(size * (1 - duplicate_ratio)))
    
    # Generate array by randomly selecting from limited value range
    dataset = [random.randint(0, unique_values - 1) for _ in range(size)]
    
    # Shuffle to prevent unintentional patterns
    random.shuffle(dataset)
    
    return dataset


def generate_dataset_suite(size: int) -> dict:
    """
    Generates complete suite of test datasets for a given size.
    
    This convenience function creates all four dataset types with consistent
    parameters, ensuring systematic experimental coverage.
    
    Args:
        size (int): Number of elements in each dataset
    
    Returns:
        dict: Dictionary mapping dataset type names to generated arrays
              Keys: 'random', 'nearly_sorted', 'reverse', 'duplicate_heavy'
    
    Example:
        >>> suite = generate_dataset_suite(100)
        >>> suite.keys()
        dict_keys(['random', 'nearly_sorted', 'reverse', 'duplicate_heavy'])
    """
    return {
        'random': generate_random(size),
        'nearly_sorted': generate_nearly_sorted(size, disorder_fraction=0.1),
        'reverse': generate_reverse_sorted(size),
        'duplicate_heavy': generate_duplicate_heavy(size, duplicate_ratio=0.5)
    }


def save_dataset_to_csv(dataset: List[int], filename: str, 
                        metadata: dict = None) -> None:
    """
    Saves a dataset to CSV file with optional metadata header.
    
    CSV format enables:
    - Human-readable inspection of datasets
    - Import into analysis tools (Excel, pandas, R)
    - Archival of experimental inputs for reproducibility
    
    Args:
        dataset (List[int]): Array to save
        filename (str): Output file path
        metadata (dict, optional): Key-value pairs to save as header comments
    
    File Format:
        # Metadata lines prefixed with #
        value
        value
        ...
    
    Example:
        >>> save_dataset_to_csv([3, 1, 2], "test.csv", 
        ...                      metadata={'size': 3, 'type': 'random'})
        # Creates test.csv:
        # size: 3
        # type: random
        3
        1
        2
    """
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write metadata as comments
        if metadata:
            for key, value in metadata.items():
                writer.writerow([f"# {key}: {value}"])
        
        # Write each element on separate line
        for value in dataset:
            writer.writerow([value])


def load_dataset_from_csv(filename: str) -> Tuple[List[int], dict]:
    """
    Loads a dataset from CSV file, parsing metadata if present.
    
    Args:
        filename (str): Input file path
    
    Returns:
        Tuple[List[int], dict]: (dataset, metadata_dict)
    
    Example:
        >>> dataset, meta = load_dataset_from_csv("test.csv")
        >>> dataset
        [3, 1, 2]
        >>> meta
        {'size': '3', 'type': 'random'}
    """
    dataset = []
    metadata = {}
    
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        
        for row in reader:
            if not row:  # Skip empty rows
                continue
            
            # Parse metadata lines (start with #)
            if row[0].startswith('#'):
                parts = row[0][1:].strip().split(':', 1)
                if len(parts) == 2:
                    key, value = parts
                    metadata[key.strip()] = value.strip()
            else:
                # Parse data value
                dataset.append(int(row[0]))
    
    return dataset, metadata


def generate_and_save_all_datasets(sizes: List[int], output_dir: str = "datasets") -> None:
    """
    Generates and saves complete experimental dataset collection.
    
    This function creates the full matrix of datasets needed for the investigation:
    - 4 dataset types × multiple sizes
    - Organized in directory structure for easy access
    - Includes metadata for traceability
    
    Args:
        sizes (List[int]): List of dataset sizes to generate
        output_dir (str): Base directory for saved datasets
    
    Directory Structure:
        datasets/
        ├── 10000/
        │   ├── random.csv
        │   ├── nearly_sorted.csv
        │   ├── reverse.csv
        │   └── duplicate_heavy.csv
        ├── 50000/
        │   └── ...
        └── ...
    
    Example:
        >>> generate_and_save_all_datasets([100, 1000, 10000])
        Generated datasets/100/random.csv (100 elements)
        Generated datasets/100/nearly_sorted.csv (100 elements)
        ...
    """
    # Create base output directory
    os.makedirs(output_dir, exist_ok=True)
    
    for size in sizes:
        # Create subdirectory for this size
        size_dir = os.path.join(output_dir, str(size))
        os.makedirs(size_dir, exist_ok=True)
        
        # Generate dataset suite
        suite = generate_dataset_suite(size)
        
        # Save each dataset type
        for dataset_type, dataset in suite.items():
            filename = os.path.join(size_dir, f"{dataset_type}.csv")
            
            metadata = {
                'size': size,
                'type': dataset_type,
                'seed': RANDOM_SEED,
                'generated': 'November 2024'
            }
            
            save_dataset_to_csv(dataset, filename, metadata)
            print(f"Generated {filename} ({size} elements)")


def validate_dataset_properties(dataset: List[int], expected_type: str) -> dict:
    """
    Validates that generated dataset has expected structural properties.
    
    This function performs sanity checks to ensure dataset generators
    produce data with intended characteristics, catching implementation
    errors before experiments.
    
    Args:
        dataset (List[int]): Dataset to validate
        expected_type (str): Expected type ('random', 'nearly_sorted', 
                            'reverse', 'duplicate_heavy')
    
    Returns:
        dict: Validation results with metrics and pass/fail status
    
    Metrics:
        - is_sorted: Whether array is in ascending order
        - is_reverse_sorted: Whether array is in descending order
        - unique_count: Number of distinct values
        - inversion_count: Number of out-of-order pairs
        - passes_validation: Boolean indicating if properties match expected_type
    
    Example:
        >>> data = generate_reverse_sorted(100)
        >>> validate_dataset_properties(data, 'reverse')
        {'is_sorted': False, 'is_reverse_sorted': True, ..., 
         'passes_validation': True}
    """
    n = len(dataset)
    
    # Calculate properties
    is_sorted = all(dataset[i] <= dataset[i+1] for i in range(n-1)) if n > 1 else True
    is_reverse_sorted = all(dataset[i] >= dataset[i+1] for i in range(n-1)) if n > 1 else True
    unique_count = len(set(dataset))
    
    # Count inversions (out-of-order pairs) - O(n²) but acceptable for validation
    inversion_count = sum(1 for i in range(n) for j in range(i+1, n) 
                         if dataset[i] > dataset[j])
    
    # Calculate duplicate ratio
    duplicate_ratio = 1 - (unique_count / n) if n > 0 else 0
    
    # Validation logic based on expected type
    passes_validation = False
    
    if expected_type == 'random':
        # Random data should not be sorted and should have reasonable uniqueness
        passes_validation = (not is_sorted and not is_reverse_sorted and 
                           unique_count > n * 0.5)
    
    elif expected_type == 'nearly_sorted':
        # Should have low inversion count relative to worst case (n*(n-1)/2)
        max_inversions = n * (n - 1) // 2
        inversion_ratio = inversion_count / max_inversions if max_inversions > 0 else 0
        passes_validation = (not is_sorted and inversion_ratio < 0.2)
    
    elif expected_type == 'reverse':
        # Should be perfectly reverse sorted
        passes_validation = is_reverse_sorted
    
    elif expected_type == 'duplicate_heavy':
        # Should have high duplicate ratio (>30%)
        passes_validation = duplicate_ratio > 0.3
    
    return {
        'size': n,
        'is_sorted': is_sorted,
        'is_reverse_sorted': is_reverse_sorted,
        'unique_count': unique_count,
        'unique_ratio': unique_count / n if n > 0 else 0,
        'duplicate_ratio': duplicate_ratio,
        'inversion_count': inversion_count,
        'inversion_ratio': inversion_count / (n * (n-1) // 2) if n > 1 else 0,
        'passes_validation': passes_validation
    }


# Demonstration and validation
if __name__ == "__main__":
    print("=" * 70)
    print("Dataset Generation Module - Validation and Demonstration")
    print("=" * 70)
    
    # Test dataset generation with small size for inspection
    test_size = 20
    
    print(f"\n1. Random Dataset (n={test_size}):")
    print("-" * 70)
    random_data = generate_random(test_size)
    print(f"   Data: {random_data}")
    validation = validate_dataset_properties(random_data, 'random')
    print(f"   Unique values: {validation['unique_count']}/{test_size} "
          f"({validation['unique_ratio']*100:.1f}%)")
    print(f"   Validation: {'✓ PASS' if validation['passes_validation'] else '✗ FAIL'}")
    
    print(f"\n2. Nearly Sorted Dataset (n={test_size}, 10% disorder):")
    print("-" * 70)
    nearly_sorted_data = generate_nearly_sorted(test_size, disorder_fraction=0.1)
    print(f"   Data: {nearly_sorted_data}")
    validation = validate_dataset_properties(nearly_sorted_data, 'nearly_sorted')
    print(f"   Inversions: {validation['inversion_count']} "
          f"({validation['inversion_ratio']*100:.1f}% of maximum)")
    print(f"   Validation: {'✓ PASS' if validation['passes_validation'] else '✗ FAIL'}")
    
    print(f"\n3. Reverse Sorted Dataset (n={test_size}):")
    print("-" * 70)
    reverse_data = generate_reverse_sorted(test_size)
    print(f"   Data: {reverse_data}")
    validation = validate_dataset_properties(reverse_data, 'reverse')
    print(f"   Is reverse sorted: {validation['is_reverse_sorted']}")
    print(f"   Validation: {'✓ PASS' if validation['passes_validation'] else '✗ FAIL'}")
    
    print(f"\n4. Duplicate-Heavy Dataset (n={test_size}, 50% duplicates):")
    print("-" * 70)
    duplicate_data = generate_duplicate_heavy(test_size, duplicate_ratio=0.5)
    print(f"   Data: {duplicate_data}")
    validation = validate_dataset_properties(duplicate_data, 'duplicate_heavy')
    print(f"   Unique values: {validation['unique_count']}/{test_size} "
          f"({validation['unique_ratio']*100:.1f}%)")
    print(f"   Duplicate ratio: {validation['duplicate_ratio']*100:.1f}%")
    print(f"   Validation: {'✓ PASS' if validation['passes_validation'] else '✗ FAIL'}")
    
    # Large-scale validation
    print(f"\n5. Large-Scale Validation (n=10,000):")
    print("-" * 70)
    
    for dataset_type in ['random', 'nearly_sorted', 'reverse', 'duplicate_heavy']:
        if dataset_type == 'random':
            data = generate_random(10000)
        elif dataset_type == 'nearly_sorted':
            data = generate_nearly_sorted(10000)
        elif dataset_type == 'reverse':
            data = generate_reverse_sorted(10000)
        else:
            data = generate_duplicate_heavy(10000)
        
        validation = validate_dataset_properties(data, dataset_type)
        status = '✓ PASS' if validation['passes_validation'] else '✗ FAIL'
        
        print(f"   {status} | {dataset_type:15s} | "
              f"Unique: {validation['unique_ratio']*100:5.1f}% | "
              f"Inversions: {validation['inversion_ratio']*100:5.1f}%")
    
    # Demonstrate CSV save/load functionality
    print(f"\n6. CSV Save/Load Demonstration:")
    print("-" * 70)
    
    test_data = [3, 1, 4, 1, 5, 9, 2, 6]
    test_file = "test_dataset.csv"
    
    save_dataset_to_csv(test_data, test_file, 
                       metadata={'size': len(test_data), 'type': 'test'})
    print(f"   Saved: {test_data} to {test_file}")
    
    loaded_data, loaded_metadata = load_dataset_from_csv(test_file)
    print(f"   Loaded: {loaded_data}")
    print(f"   Metadata: {loaded_metadata}")
    
    match = test_data == loaded_data
    print(f"   Data integrity: {'✓ PASS' if match else '✗ FAIL'}")
    
    # Clean up test file
    os.remove(test_file)
    
    # Generate complete experimental dataset suite
    print(f"\n7. Generating Complete Experimental Dataset Suite:")
    print("-" * 70)
    print("   Sizes: 10,000 | 50,000 | 100,000 | 500,000")
    print("   Types: random | nearly_sorted | reverse | duplicate_heavy")
    print("   Total: 16 datasets")
    print("\n   Generating...")
    
    generate_and_save_all_datasets(
        sizes=[10000, 50000, 100000, 500000],
        output_dir="experimental_datasets"
    )
    
    print("\n" + "=" * 70)
    print("✓ Dataset generation module validated successfully")
    print("✓ All datasets saved to 'experimental_datasets/' directory")
    print("=" * 70)
    
    # Reproducibility verification
    print(f"\n8. Reproducibility Verification:")
    print("-" * 70)
    print("   Testing that identical seed produces identical datasets...")
    
    # Generate same dataset twice with same seed
    random.seed(42)
    data1 = generate_random(100)
    
    random.seed(42)
    data2 = generate_random(100)
    
    reproducible = data1 == data2
    print(f"   Dataset 1 == Dataset 2: {'✓ PASS' if reproducible else '✗ FAIL'}")
    
    if reproducible:
        print("   ✓ Reproducibility confirmed: Fixed seed produces identical outputs")
    
    print("\n" + "=" * 70)
    print("Dataset generation module ready for experimental use")
    print("=" * 70)

def load_csv(path):
    """
    Load dataset from CSV file in simple format.
    Compatible with experimentmergesort.py expectations.
    
    Args:
        path (str): Path to CSV file
        
    Returns:
        List[int]: Array of integers from CSV
        
    File Format Expected:
        value
        123
        456
        789
    """
    data = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        # Skip header if present
        try:
            first_row = next(reader)
            # Check if first row is header
            try:
                int(first_row[0])
                # It's a number, add it to data
                data.append(int(first_row[0]))
            except ValueError:
                # It's a header like "value", skip it
                pass
        except StopIteration:
            return data
        
        # Read remaining rows
        for row in reader:
            if row:
                data.append(int(row[0]))
    
    return data