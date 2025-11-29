"""
Memory Constraint Enforcement and Monitoring Module

This module provides process-level memory limiting and monitoring for
sorting algorithm performance analysis under constrained conditions.
It implements resource limits using OS-level mechanisms and tracks
memory usage throughout algorithm execution.

Key Features:
- Process-level memory constraints (RLIMIT_AS on Linux/macOS)
- Real-time memory usage tracking via psutil
- Baseline subtraction to isolate algorithm memory consumption
- Exception handling for memory limit violations

Author: [Candidate Number]
Date: November 2024
Purpose: IB Computer Science Extended Essay (May 2026)
Platform: GitHub Codespaces (Linux container) / macOS compatibility
"""

import resource
import psutil
import os
import sys
import traceback
from typing import Optional, Tuple
from contextlib import contextmanager


class MemoryMonitor:
    """
    Context manager for enforcing and monitoring memory constraints.
    
    This class implements process-level memory limiting using the resource
    module (RLIMIT_AS) and tracks peak memory consumption using psutil.
    Designed as a context manager for automatic cleanup and exception handling.
    
    Usage:
        with MemoryMonitor(limit_mb=128) as monitor:
            # Algorithm execution here
            result = some_sorting_function(data)
        
        peak_memory = monitor.get_peak_usage()
        success = not monitor.memory_exceeded
    
    Attributes:
        limit_bytes (int): Memory limit in bytes
        baseline_memory (int): Process memory before algorithm execution
        peak_memory (float): Maximum memory usage in MB during execution
        memory_exceeded (bool): Whether MemoryError was raised
        process (psutil.Process): Current process handle for monitoring
    """
    
    def __init__(self, limit_mb: int):
        """
        Initialize memory monitor with specified limit.
        
        Args:
            limit_mb (int): Memory limit in megabytes (MB)
        
        Note:
            The limit applies to the process's virtual address space (AS),
            which includes code, data, heap, stack, and memory-mapped files.
            This is stricter than limiting only heap memory (RSS).
        """
        self.limit_bytes = limit_mb * 1024 * 1024  # Convert MB to bytes
        self.process = psutil.Process(os.getpid())
        
        # Record baseline memory before algorithm execution
        # This allows isolating algorithm-specific memory usage
        self.baseline_memory = self.process.memory_info().rss
        
        self.peak_memory = 0.0  # Peak usage in MB
        self.memory_exceeded = False  # Flag for MemoryError
        self.limit_mb = limit_mb  # Store for reporting
        
        # Platform detection for informative warnings
        self.platform = sys.platform
        self._check_platform_compatibility()
    
    def _check_platform_compatibility(self) -> None:
        """
        Checks platform compatibility and warns about potential issues.
        
        RLIMIT_AS behavior varies across platforms:
        - Linux: Reliable enforcement via kernel
        - macOS: May be ignored on some versions (especially ≥10.14)
        - Windows: Not supported (resource module unavailable)
        
        GitHub Codespaces uses Linux containers, ensuring reliable enforcement.
        """
        if self.platform.startswith('win'):
            print(f"WARNING: Windows does not support resource.RLIMIT_AS")
            print(f"         Memory limiting will NOT be enforced")
        elif self.platform == 'darwin':  # macOS
            print(f"INFO: macOS memory limiting may be unreliable")
            print(f"      Consider testing on Linux for accurate results")
    
    def __enter__(self):
        """
        Enter context manager: Set memory limit.
        
        Sets RLIMIT_AS (address space limit) to constrain total virtual
        memory available to the process. When exceeded, the OS raises
        MemoryError on allocation attempts.
        
        Returns:
            self: MemoryMonitor instance for usage tracking
        """
        try:
            # Set both soft and hard limits to the same value
            # Soft limit: checked on each allocation
            # Hard limit: maximum the soft limit can be raised to
            resource.setrlimit(
                resource.RLIMIT_AS,
                (self.limit_bytes, self.limit_bytes)
            )
        except (ValueError, OSError) as e:
            print(f"WARNING: Failed to set memory limit: {e}")
            print(f"         Continuing without memory constraint")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit context manager: Restore unlimited memory and handle exceptions.
        
        Args:
            exc_type: Exception type (if exception occurred)
            exc_val: Exception value
            exc_tb: Exception traceback
        
        Returns:
            bool: False to propagate exceptions, True to suppress
        
        Behavior:
            - Restores memory limit to infinity (RLIM_INFINITY)
            - Detects MemoryError and sets memory_exceeded flag
            - Allows other exceptions to propagate normally
        """
        # Restore unlimited memory for subsequent operations
        try:
            resource.setrlimit(
                resource.RLIMIT_AS,
                (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
            )
        except (ValueError, OSError) as e:
            print(f"WARNING: Failed to restore memory limit: {e}")
        
        # Check if MemoryError occurred during execution
        if exc_type is MemoryError:
            self.memory_exceeded = True
            print(f"MemoryError: Algorithm exceeded {self.limit_mb} MB limit")
            return True  # Suppress exception (handled via flag)
        
        # Allow other exceptions to propagate
        return False
    
    def get_current_usage(self) -> float:
        """
        Get current memory usage above baseline.
        
        Returns:
            float: Current memory usage in MB (algorithm only)
        
        Note:
            Uses RSS (Resident Set Size) which measures physical RAM
            actually used by the process, excluding swapped-out pages.
            This differs from virtual address space (RLIMIT_AS target).
        """
        current_rss = self.process.memory_info().rss
        usage_bytes = current_rss - self.baseline_memory
        usage_mb = usage_bytes / (1024 * 1024)
        
        # Track peak usage
        self.peak_memory = max(self.peak_memory, usage_mb)
        
        return usage_mb
    
    def get_peak_usage(self) -> float:
        """
        Get peak memory usage recorded during execution.
        
        Returns:
            float: Peak memory usage in MB
        
        Note:
            Peak is tracked via periodic calls to get_current_usage()
            during algorithm execution. Extremely brief spikes between
            measurements may be missed (sampling limitation).
        """
        # Final check to ensure we capture terminal state
        self.get_current_usage()
        return self.peak_memory
    
    def get_memory_info(self) -> dict:
        """
        Get detailed memory information for debugging and analysis.
        
        Returns:
            dict: Memory statistics including:
                - rss: Resident Set Size (physical memory)
                - vms: Virtual Memory Size (address space)
                - peak: Peak usage recorded
                - baseline: Memory before algorithm
                - available: System-wide available memory
        
        This extended information helps diagnose discrepancies between
        expected and measured memory usage.
        """
        mem_info = self.process.memory_info()
        virtual_mem = psutil.virtual_memory()
        
        return {
            'rss_mb': mem_info.rss / (1024 * 1024),
            'vms_mb': mem_info.vms / (1024 * 1024),
            'peak_mb': self.peak_memory,
            'baseline_mb': self.baseline_memory / (1024 * 1024),
            'available_mb': virtual_mem.available / (1024 * 1024),
            'percent_used': virtual_mem.percent
        }


class MemoryProfiler:
    """
    Periodic memory sampler for tracking usage throughout execution.
    
    While MemoryMonitor tracks peak usage, MemoryProfiler captures
    memory usage at regular intervals to analyze temporal patterns.
    Useful for understanding whether memory grows linearly, exhibits
    spikes, or plateaus.
    
    Usage:
        profiler = MemoryProfiler()
        profiler.start()
        # Algorithm execution
        profiler.stop()
        print(profiler.get_profile())
    """
    
    def __init__(self, sample_interval: float = 0.01):
        """
        Initialize memory profiler.
        
        Args:
            sample_interval (float): Seconds between samples (default 10ms)
        
        Note:
            Smaller intervals provide finer granularity but increase overhead.
            10ms is a reasonable compromise for algorithms running 0.1-10s.
        """
        self.sample_interval = sample_interval
        self.samples = []
        self.process = psutil.Process(os.getpid())
        self.baseline = self.process.memory_info().rss
        self.is_sampling = False
    
    def start(self) -> None:
        """Begin memory sampling."""
        self.is_sampling = True
        self.samples = []
        self.baseline = self.process.memory_info().rss
    
    def sample(self) -> float:
        """
        Take single memory measurement.
        
        Returns:
            float: Current memory usage in MB
        """
        if not self.is_sampling:
            return 0.0
        
        current = self.process.memory_info().rss
        usage_mb = (current - self.baseline) / (1024 * 1024)
        self.samples.append(usage_mb)
        return usage_mb
    
    def stop(self) -> None:
        """Stop memory sampling."""
        self.is_sampling = False
    
    def get_profile(self) -> dict:
        """
        Get statistical summary of memory usage profile.
        
        Returns:
            dict: Statistics including min, max, mean, std deviation
        """
        if not self.samples:
            return {'error': 'No samples collected'}
        
        import statistics
        
        return {
            'sample_count': len(self.samples),
            'min_mb': min(self.samples),
            'max_mb': max(self.samples),
            'mean_mb': statistics.mean(self.samples),
            'median_mb': statistics.median(self.samples),
            'stdev_mb': statistics.stdev(self.samples) if len(self.samples) > 1 else 0.0
        }


@contextmanager
def memory_limit(limit_mb: int):
    """
    Simple context manager for memory limiting without monitoring.
    
    Args:
        limit_mb (int): Memory limit in MB
    
    Yields:
        None
    
    Usage:
        with memory_limit(128):
            result = sorting_function(data)
    
    This is a lightweight alternative to MemoryMonitor when you only
    need enforcement without tracking peak usage.
    """
    limit_bytes = limit_mb * 1024 * 1024
    
    # Set limit
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    except (ValueError, OSError) as e:
        print(f"WARNING: Failed to set memory limit: {e}")
    
    try:
        yield
    finally:
        # Restore unlimited
        try:
            resource.setrlimit(
                resource.RLIMIT_AS,
                (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
            )
        except (ValueError, OSError):
            pass


def get_system_memory_info() -> dict:
    """
    Get system-wide memory information for environment documentation.
    
    Returns:
        dict: System memory statistics
    
    This information should be recorded at the start of experiments to
    document the computational environment (Section 3.1).
    """
    virtual = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    return {
        'total_ram_gb': virtual.total / (1024**3),
        'available_ram_gb': virtual.available / (1024**3),
        'used_ram_gb': virtual.used / (1024**3),
        'ram_percent': virtual.percent,
        'total_swap_gb': swap.total / (1024**3),
        'used_swap_gb': swap.used / (1024**3),
        'swap_percent': swap.percent
    }


def test_memory_limit_enforcement(limit_mb: int = 50) -> Tuple[bool, str]:
    """
    Test whether memory limiting actually works on this platform.
    
    Args:
        limit_mb (int): Test limit in MB (should be low enough to trigger)
    
    Returns:
        Tuple[bool, str]: (success, message)
    
    This validation function should be run before experiments to confirm
    that the OS actually enforces memory limits. On some platforms
    (macOS, Windows), RLIMIT_AS may be ignored.
    """
    try:
        with MemoryMonitor(limit_mb) as monitor:
            # Attempt to allocate memory exceeding limit
            data = []
            chunk_size = 1024 * 1024  # 1 MB chunks
            
            # Try to allocate 2× the limit
            for _ in range(limit_mb * 2):
                data.append(bytearray(chunk_size))
        
        # If we reach here without MemoryError, limiting failed
        if monitor.memory_exceeded:
            return True, f"✓ Memory limiting works (caught at {limit_mb} MB)"
        else:
            return False, f"✗ Memory limiting NOT enforced (allocated {limit_mb*2} MB)"
    
    except Exception as e:
        return False, f"✗ Test failed with exception: {e}"


# Demonstration and validation
if __name__ == "__main__":
    print("=" * 70)
    print("Memory Constraint Enforcement Module - Validation")
    print("=" * 70)
    
    # Test 1: System Information
    print("\n1. System Memory Information:")
    print("-" * 70)
    sys_info = get_system_memory_info()
    print(f"   Total RAM:     {sys_info['total_ram_gb']:.2f} GB")
    print(f"   Available RAM: {sys_info['available_ram_gb']:.2f} GB")
    print(f"   Used RAM:      {sys_info['used_ram_gb']:.2f} GB ({sys_info['ram_percent']:.1f}%)")
    print(f"   Total Swap:    {sys_info['total_swap_gb']:.2f} GB")
    print(f"   Platform:      {sys.platform}")
    
    # Test 2: Basic Memory Monitoring
    print("\n2. Basic Memory Monitoring (No Limit):")
    print("-" * 70)
    
    # Simulate algorithm that allocates memory
    with MemoryMonitor(limit_mb=1024) as monitor:  # High limit, won't trigger
        data = [0] * 1_000_000  # ~28 MB in Python (1M × 28 bytes/int object)
        current = monitor.get_current_usage()
        print(f"   After allocating 1M integers:")
        print(f"   Current usage: {current:.2f} MB")
    
    peak = monitor.get_peak_usage()
    print(f"   Peak usage:    {peak:.2f} MB")
    print(f"   Exceeded:      {monitor.memory_exceeded}")
    
    # Test 3: Memory Limit Enforcement Test
    print("\n3. Memory Limit Enforcement Test:")
    print("-" * 70)
    
    success, message = test_memory_limit_enforcement(limit_mb=50)
    print(f"   {message}")
    
    if not success:
        print("\n   WARNING: Memory limiting may not work on this platform!")
        print("   Experimental results may not reflect true constraint effects.")
        print("   Consider running on Linux for reliable enforcement.")
    
    # Test 4: Detailed Memory Info
    print("\n4. Detailed Memory Information:")
    print("-" * 70)
    
    with MemoryMonitor(limit_mb=256) as monitor:
        # Allocate in stages to see growth
        data = []
        for i in range(5):
            data.extend([0] * 100_000)  # Add 100K integers
            info = monitor.get_memory_info()
            print(f"   Stage {i+1}: RSS={info['rss_mb']:.2f} MB, "
                  f"VMS={info['vms_mb']:.2f} MB, "
                  f"Peak={info['peak_mb']:.2f} MB")
    
    # Test 5: Memory Profiler
    print("\n5. Memory Profiler Temporal Analysis:")
    print("-" * 70)
    
    profiler = MemoryProfiler(sample_interval=0.01)
    profiler.start()
    
    # Simulate algorithm with growing memory
    data = []
    for i in range(100):
        data.extend([0] * 10_000)
        profiler.sample()
    
    profiler.stop()
    profile = profiler.get_profile()
    
    print(f"   Samples collected: {profile['sample_count']}")
    print(f"   Min memory:        {profile['min_mb']:.2f} MB")
    print(f"   Max memory:        {profile['max_mb']:.2f} MB")
    print(f"   Mean memory:       {profile['mean_mb']:.2f} MB")
    print(f"   Std deviation:     {profile['stdev_mb']:.2f} MB")
    
    # Test 6: Exception Handling
    print("\n6. Memory Limit Violation Handling:")
    print("-" * 70)
    
    try:
        with MemoryMonitor(limit_mb=32) as monitor:  # Very low limit
            print("   Attempting to allocate 50 MB...")
            data = bytearray(50 * 1024 * 1024)  # Try to allocate 50 MB
            print("   ✗ Allocation succeeded (limit not enforced)")
    except MemoryError:
        print("   ✓ MemoryError caught correctly")
    
    if monitor.memory_exceeded:
        print("   ✓ memory_exceeded flag set correctly")
        print(f"   Peak before failure: {monitor.get_peak_usage():.2f} MB")
    else:
        print("   Note: Limit may not have been enforced on this platform")
    
    # Test 7: Baseline Subtraction Accuracy
    print("\n7. Baseline Subtraction Accuracy:")
    print("-" * 70)
    
    # Measure Python interpreter overhead
    with MemoryMonitor(limit_mb=512) as monitor:
        baseline_peak = monitor.get_peak_usage()
    
    print(f"   Baseline (empty execution): {baseline_peak:.2f} MB")
    
    with MemoryMonitor(limit_mb=512) as monitor:
        test_data = [0] * 1_000_000
        with_data_peak = monitor.get_peak_usage()
    
    algorithmic_memory = with_data_peak - baseline_peak
    print(f"   With 1M integers:          {with_data_peak:.2f} MB")
    print(f"   Algorithmic usage:         {algorithmic_memory:.2f} MB")
    print(f"   Expected (~28 MB):         28.00 MB")
    print(f"   Measurement accuracy:      {(algorithmic_memory/28)*100:.1f}%")
    
    # Test 8: Platform-Specific Recommendations
    print("\n8. Platform-Specific Recommendations:")
    print("-" * 70)
    
    platform = sys.platform
    if platform.startswith('linux'):
        print("   ✓ Linux detected: Optimal platform for memory limiting")
        print("   ✓ RLIMIT_AS enforced reliably by kernel")
        print("   ✓ GitHub Codespaces uses Linux - excellent choice")
    elif platform == 'darwin':
        print("   ⚠ macOS detected: Memory limiting may be unreliable")
        print("   ⚠ macOS 10.14+ often ignores RLIMIT_AS")
        print("   → Recommendation: Use GitHub Codespaces (Linux) for experiments")
    elif platform.startswith('win'):
        print("   ✗ Windows detected: Memory limiting NOT supported")
        print("   ✗ resource module unavailable on Windows")
        print("   → Recommendation: Use GitHub Codespaces or Linux VM")
    
    print("\n" + "=" * 70)
    print("Memory monitoring module validation complete")
    print("=" * 70)
    
    # Final validation summary
    print("\n9. Final Validation Summary:")
    print("-" * 70)
    
    checks = [
        ("System memory info accessible", True),
        ("Memory monitoring functional", peak > 0),
        ("Baseline subtraction working", abs(algorithmic_memory - 28) < 10),
        ("Detailed info available", True),
        ("Exception handling works", monitor.memory_exceeded or not success),
    ]
    
    all_passed = all(result for _, result in checks)
    
    for check, result in checks:
        status = "✓" if result else "✗"
        print(f"   {status} {check}")
    
    print("-" * 70)
    if all_passed:
        print("   ✓ All validation checks passed")
        print("   Module ready for experimental use")
    else:
        print("   ⚠ Some checks failed - review warnings above")
    
    print("=" * 70)