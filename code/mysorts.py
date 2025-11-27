def quicksort(arr):
    comparisons = 0
    left = []
    mid = []
    right = []
    if len(arr) <= 1:
        return arr, 0
    def median_of_three(a, b, c):
        return sorted([a, b, c])[1]
    first = arr[0]
    middle = arr[len(arr)//2]
    last = arr[-1]

    pivot = median_of_three(first, middle, last)
    
    for x in arr:
        comparisons += 1
        if x<pivot:
            left.append(x)
        elif x==pivot:
            mid.append(x)
        else:
            right.append(x)
    
    left_sorted, left_comps = quicksort(left)
    right_sorted, right_comps = quicksort(right)
    total_comparisons = comparisons + left_comps + right_comps
    sorted_list = left_sorted + mid + right_sorted
    return sorted_list, total_comparisons

def mergesort(arr):
    if len(arr) <= 1:
        return arr, 0

    mid = len(arr) // 2
    left, left_comps = mergesort(arr[:mid])
    right, right_comps = mergesort(arr[mid:])

    merged, merge_comps = merge(left, right)
    return  merged, left_comps + right_comps + merge_comps

def merge(left, right):
    result = []
    i = j = 0
    comparisons = 0
    while i < len(left) and j < len(right):
        comparisons += 1
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result, comparisons