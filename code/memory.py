from memory_profiler import memory_usage
import time

def heavy_function():
    a = [i for i in range(10_000_000)]  # big list
    time.sleep(1)
    return sum(a)

if __name__ == "__main__":
    mem_usage, result = memory_usage((heavy_function,), retval=True, interval=0.1)
    print("Result:", result)
    print("Peak memory (MB):", max(mem_usage))