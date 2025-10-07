import csv
from mysorts import quicksort, mergesort

def load_csv(path):
    with open(path, newline="") as f:
        reader=csv.reader(f)
        return [int(row[0]) for row in reader]
data = load_csv("datasets/reverse10k.csv")

sorted_quick = quicksort(data)
sorted_merge = mergesort(data)

def is_sorted(arr):
    return all(arr[i] <= arr[i+1] for i in range (len(arr)-1)) 
print(is_sorted(sorted_quick))
print(is_sorted(sorted_merge))

