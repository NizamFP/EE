import os
import csv
import time
from memory_profiler import memory_usage
from mysorts import quicksort, mergesort

DATASET_DIR = "datasets"
RESULTS_FILE = "results.csv"

def load_csv(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        return [int(x) for row in reader for x in row]  # flatten rows

def measure(sort_fn, data):
    def run():
        sort_fn(data.copy())  # copy so the original stays unsorted
    mem, _ = memory_usage((run,), retval=True, interval=0.05)
    start = time.perf_counter()
    run()
    end = time.perf_counter()
    return end - start, max(mem)

def main():
    files = sorted(os.listdir(DATASET_DIR))
    algorithms = [("QuickSort", quicksort), ("MergeSort", mergesort)]

    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["algorithm", "dataset", "time_sec", "peak_memory_mb"])

        for file in files:
            path = os.path.join(DATASET_DIR, file)
            data = load_csv(path)

            for name, fn in algorithms:
                try:
                    print(f"Running {name} on {file}...")
                    t, m = measure(fn, data)
                    writer.writerow([name, file, f"{t:.3f}", f"{m:.2f}"])
                except MemoryError:
                    writer.writerow([name, file, "FAILED", "MEMORY ERROR"])
                except Exception as e:
                    writer.writerow([name, file, "FAILED", str(e)])

if __name__ == "__main__":
    main()