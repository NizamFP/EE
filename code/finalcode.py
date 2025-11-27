# finalcode.py
# Experiment orchestration for EE: runs QuickSort and MergeSort over pre-generated datasets
# Measures: time, peak memory (via MemoryMonitor), comparison count, correctness
# Produces a CSV of raw results and summary tables per memory constraint

import os
import csv
import time
import gc
import hashlib
from datetime import datetime
from tqdm import tqdm
from memconstraint import MemoryMonitor  # assumed to exist in same package
from mysorts import quicksort, mergesort    # both should return (sorted_list, comparisons)

# ==========================
# CONFIG (edit if needed)
# ==========================
DATASETS_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SIZES = [10_000] # [10_000, 50_000, 100_000, 500_000]
SIZE_LABEL = {10000: '10k', 50000: '50k', 100000: '100k', 500000: '500k'}
STRUCTURES = ['random', 'reversed', 'nearly_sorted', 'duplicate']
STRUCTURE_FILENAME_PREFIX = {
    'random': 'random',
    'reversed': 'reverse',
    'nearly_sorted': 'nearly',
    'duplicate': 'dup'
}
MEM_LIMITS_MB = [256] #[128, 256, 512, 1024]
ALGORITHMS = {'quicksort': quicksort, 'mergesort': mergesort}
REPETITIONS = 1 #5
SEED_BASE = 123456

RAW_CSV = os.path.join(RESULTS_DIR, f"raw_results_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.csv")

# ==========================
# Utilities
# ==========================

def dataset_path(size, structure):
    prefix = STRUCTURE_FILENAME_PREFIX[structure]
    label = SIZE_LABEL[size]
    filename = f"{prefix}{label}.csv"
    return os.path.join(DATASETS_DIR, filename)


def read_dataset_csv(path):
    """Read dataset CSV into Python list of ints (or floats). Assumes one column."""
    data = []
    with open(path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            # try int then float
            val = row[0].strip()
            try:
                data.append(int(val))
            except ValueError:
                try:
                    data.append(float(val))
                except ValueError:
                    # if non-numeric, keep raw
                    data.append(val)
    return data


def safe_sorted_copy(data):
    """Return a shallow copy for passing to sorting functions to avoid mutating original list."""
    return list(data)


def checksum_list(data):
    """Return a short sha1 hex digest of the data list for quick correctness fingerprinting."""
    m = hashlib.sha1()
    # convert to bytes in a deterministic way
    for x in data:
        m.update(repr(x).encode('utf-8') + b'|')
    return m.hexdigest()

# ==========================
# Core runner
# ==========================

def run_one(algo_name, algo_fn, original_data, mem_limit_mb, seed):
    """
    Run a single experiment with memory limit. Returns a dict of results.
    Assumes algo_fn(data_copy) -> (sorted_list, comparisons)
    """
    # Prepare
    data_copy = safe_sorted_copy(original_data)
    result = {
        'timestamp_utc': datetime.utcnow().isoformat(),
        'algorithm': algo_name,
        'size': len(original_data),
        'mem_limit_mb': mem_limit_mb,
        'seed': seed,
        'time_s': None,
        'peak_mem_mb': None,
        'memory_exceeded': False,
        'comparisons': None,
        'correct': False,
        'error_msg': None,
        'orig_checksum': checksum_list(original_data),
        'sorted_checksum': None
    }

    # Run under memory monitor
    with MemoryMonitor(mem_limit_mb) as mm:
        try:
            t0 = time.perf_counter()
            sorted_list, comparisons = algo_fn(data_copy)
            t1 = time.perf_counter()
            result['time_s'] = t1 - t0
            result['comparisons'] = comparisons
        except MemoryError as me:
            # memory limit exceeded inside sorting
            result['memory_exceeded'] = True
            result['error_msg'] = 'MemoryError'
            # try to capture time delta if available
            try:
                result['time_s'] = time.perf_counter() - t0
            except Exception:
                result['time_s'] = None
            sorted_list = None
            comparisons = None
        except Exception as e:
            result['error_msg'] = repr(e)
            sorted_list = None
            comparisons = None
        finally:
            # record peak memory usage measured by MemoryMonitor
            try:
                result['peak_mem_mb'] = mm.get_peak_usage()
            except Exception:
                result['peak_mem_mb'] = None

    # Correctness validation
    if not result['memory_exceeded'] and sorted_list is not None:
        try:
            correct = (sorted_list == sorted(original_data))
            result['correct'] = bool(correct)
            result['sorted_checksum'] = checksum_list(sorted_list)
        except Exception as e:
            result['error_msg'] = f"correctness-check-failed: {e}"
            result['correct'] = False

    # cleanup
    try:
        del data_copy
        del sorted_list
    except Exception:
        pass
    gc.collect()

    return result

# ==========================
# Experiment orchestration
# ==========================

def run_experiments():
    # prepare raw csv header
    header = [
        'run_id', 'timestamp_utc', 'algorithm', 'size', 'structure', 'mem_limit_mb', 'repetition', 'seed',
        'time_s', 'peak_mem_mb', 'memory_exceeded', 'comparisons', 'correct', 'error_msg',
        'orig_checksum', 'sorted_checksum'
    ]

    run_id = 0
    with open(RAW_CSV, 'w', newline='') as rf:
        writer = csv.DictWriter(rf, fieldnames=header)
        writer.writeheader()

        # nested loops
        total_runs = len(SIZES) * len(STRUCTURES) * len(MEM_LIMITS_MB) * len(ALGORITHMS) * REPETITIONS
        pbar = tqdm(total=total_runs, desc='Total runs')

        for size in SIZES:
            for structure in STRUCTURES:
                ds_path = dataset_path(size, structure)
                if not os.path.exists(ds_path):
                    print(f"Dataset missing: {ds_path} -- skipping")
                    for _ in range(len(MEM_LIMITS_MB) * len(ALGORITHMS) * REPETITIONS):
                        pbar.update(1)
                    continue

                original_data = read_dataset_csv(ds_path)

                for mem_limit in MEM_LIMITS_MB:
                    for algo_name, algo_fn in ALGORITHMS.items():
                        for rep in range(1, REPETITIONS + 1):
                            run_id += 1
                            seed = SEED_BASE + run_id
                            # run
                            res = run_one(algo_name, algo_fn, original_data, mem_limit, seed)
                            # add extra metadata
                            row = {
                                'run_id': run_id,
                                'timestamp_utc': res['timestamp_utc'],
                                'algorithm': res['algorithm'],
                                'size': res['size'],
                                'structure': structure,
                                'mem_limit_mb': res['mem_limit_mb'],
                                'repetition': rep,
                                'seed': res['seed'],
                                'time_s': res['time_s'],
                                'peak_mem_mb': res['peak_mem_mb'],
                                'memory_exceeded': res['memory_exceeded'],
                                'comparisons': res['comparisons'],
                                'correct': res['correct'],
                                'error_msg': res['error_msg'],
                                'orig_checksum': res['orig_checksum'],
                                'sorted_checksum': res['sorted_checksum']
                            }
                            writer.writerow(row)
                            rf.flush()
                            pbar.update(1)

                # free dataset memory between sizes
                del original_data
                gc.collect()

        pbar.close()

    print(f"Raw results written to: {RAW_CSV}")
    return RAW_CSV


# ==========================
# Table generation (Table 1 style)
# ==========================

def generate_table1_by_mem(raw_csv_path):
    """
    Build Table 1 including:
    QuickSort_time_s, QuickSort_memory_MB,
    MergeSort_time_s, MergeSort_memory_MB
    for each memory limit.
    """
    import collections

    # read raw results
    rows = []
    with open(raw_csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # group by mem_limit
    grouped = collections.defaultdict(list)
    for r in rows:
        grouped[r['mem_limit_mb']].append(r)

    for mem_limit, group in grouped.items():
        # nested grouping by (size, structure, repetition)
        table = {}

        for r in group:
            key = (r['size'], r['structure'], r['repetition'])
            algo = r['algorithm']  # 'quicksort' or 'mergesort'

            # -----------------------------
           # CHANGE #1:
            # struktur tabel baru yang menyimpan
            # time + mem untuk masing-masing algoritma
            # -----------------------------
            if key not in table:
                table[key] = {
                    'quicksort': {'time': '', 'mem': ''},
                    'mergesort': {'time': '', 'mem': ''}
                }

            # -----------------------------
            # CHANGE #2:
            # memasukkan dua nilai sekaligus:
            # time_s dan peak_mem_mb
            # -----------------------------
            table[key][algo]['time'] = r['time_s']
            table[key][algo]['mem'] = r['peak_mem_mb']

        # write CSV
        out_path = os.path.join(RESULTS_DIR, f"table1_mem{mem_limit}MB.csv")
        with open(out_path, 'w', newline='') as of:
            w = csv.writer(of)

            # -----------------------------
            # CHANGE #3: header baru
            # -----------------------------
            w.writerow([
                'size', 'structure', 'trial',
                'QuickSort_time_s', 'QuickSort_memory_MB',
                'MergeSort_time_s', 'MergeSort_memory_MB'
            ])

            # isi tabel
            for (size, structure, trial), vals in sorted(table.items()):
                qs_t = vals['quicksort']['time']
                qs_m = vals['quicksort']['mem']
                ms_t = vals['mergesort']['time']
                ms_m = vals['mergesort']['mem']

                # -----------------------------
                # CHANGE #4:
                # menulis 4 kolom hasil
                # -----------------------------
                w.writerow([
                    size, structure, trial,
                    qs_t, qs_m,
                    ms_t, ms_m
                ])

        print(f"Wrote table for mem {mem_limit} MB -> {out_path}")



# ==========================
# Main
# ==========================
if __name__ == '__main__':
    raw = run_experiments()
    generate_table1_by_mem(raw)
    print('Done')