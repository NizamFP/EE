import random, copy
random.seed(42)

def gen_random(size):
    return [random.randint(0, size) for _ in range(size)]

def gen_nearly_sorted(size, swap_fraction=0.1):
    arr = list(range(size))
    num_swaps = int(size * swap_fraction)
    for _ in range(num_swaps):
        i, j = random.randrange(size), random.randrange(size)
        arr[i], arr[j] = arr[j], arr[i]
    return arr

def gen_reverse(size):
    return list(range(size, 0, -1))

def gen_duplicates(size, value_range=100):
    return [random.randint(0, value_range) for _ in range(size)]