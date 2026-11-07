"""
Quick sort and merge sort, used to compare against heap sort.

quick_sort picks a random pivot and splits into three groups
(less / equal / greater). The random pivot stops sorted or reversed input
from hitting the O(n^2) case, and the "equal" group means lots of duplicate
values don't slow it down either. merge_sort is the usual top-down version
and gives back a new list.

Both only compare with `<`, same as heapsort.py.
"""

import random


def three_way_partition(arr, low, high, rand):
    """Split arr[low..high] around a random pivot.

    After this runs:
        arr[low .. lt-1]  < pivot
        arr[lt .. gt]    == pivot
        arr[gt+1 .. high] > pivot
    and we return (lt, gt).
    """
    pick = rand.randint(low, high)
    arr[low], arr[pick] = arr[pick], arr[low]
    pivot = arr[low]

    lt, cur, gt = low, low + 1, high
    while cur <= gt:
        if arr[cur] < pivot:
            arr[lt], arr[cur] = arr[cur], arr[lt]
            lt += 1
            cur += 1
        elif pivot < arr[cur]:
            arr[cur], arr[gt] = arr[gt], arr[cur]
            gt -= 1
        else:
            cur += 1
    return lt, gt


def quick_sort(arr, rand=None):
    """Randomized 3-way quick sort, in place. Returns arr.

    Recurses into the smaller half and loops on the bigger one, which keeps
    the recursion depth around log n even when the pivots are unlucky.
    """
    rand = rand or random.Random()

    def _qs(low, high):
        while low < high:
            lt, gt = three_way_partition(arr, low, high, rand)
            if lt - low < high - gt:
                _qs(low, lt - 1)
                low = gt + 1
            else:
                _qs(gt + 1, high)
                high = lt - 1

    _qs(0, len(arr) - 1)
    return arr


def merge_lists(left, right):
    """Merge two already-sorted lists into one sorted list (stable)."""
    merged = []
    i = j = 0
    while i < len(left) and j < len(right):
        if right[j] < left[i]:
            merged.append(right[j])
            j += 1
        else:
            merged.append(left[i])
            i += 1
    # one side is used up, tack on whatever's left of the other
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged


def merge_sort(arr):
    """Top-down merge sort. Returns a new list and doesn't touch arr.

    T(n) = 2T(n/2) + O(n), so O(n log n) no matter what the input looks like.
    """
    if len(arr) <= 1:
        return list(arr)
    half = len(arr) // 2
    return merge_lists(merge_sort(arr[:half]), merge_sort(arr[half:]))


if __name__ == "__main__":
    nums = [random.randint(0, 20) for _ in range(15)]
    print("input     :", nums)
    print("merge_sort:", merge_sort(nums))
    print("quick_sort:", quick_sort(nums[:]))
