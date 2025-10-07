from memory_profiler import memory_usage
import csv
import time
from mysorts import quicksort
import resource


def load_csv(path):
    with open(path,newline="") as f:
        reader = csv.reader(f)
        return [int(x) for row in reader for x in row]
    
def run_sort_with_limit(limit_mb):
    import psutil
    import os
    process = psutil.Process(os.getpid())

    data = load_csv("datasets/random1000k.csv")
    quicksort(data)
    used = process.memory_info().rss / (1024 * 1024)
    if used > limit_mb:
        raise MemoryError(f"Exceeded memory limit of {limit_mb} MB (used {used:.2f} MB)")
    
if __name__ == "__main__":
    for limit in [96, 96, 96]:
        print(f"\n--- Running with {limit} MB limit ---")
        start = time.perf_counter()
        try:
            mem_usage = memory_usage((run_sort_with_limit, (limit,)), interval=0.1)
            end = time.perf_counter()
            print("Execution time (s):", round(end - start, 3))
            print("Peak memory (MB):", max(mem_usage))
        except MemoryError as e:
            print("❌", e)