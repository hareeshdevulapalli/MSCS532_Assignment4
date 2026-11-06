"""Quicksort and Merge Sort, used as baselines for the Heapsort comparison.

Both return a sorted list. `quicksort` sorts in place; `merge_sort`
returns a new list (it needs Theta(n) scratch space anyway).

Quicksort uses a random pivot and three-way ("Dutch national flag")
partitioning. The random pivot makes sorted and reverse-sorted input
behave like random input (expected Theta(n lg n)); the three-way split
puts all keys equal to the pivot in their final place at once, so input
with many duplicates does not degrade to Theta(n^2) the way two-way
Lomuto partitioning does. The worst case is still Theta(n^2), but only
with vanishing probability.

Like heapsort.py, only `<` is used to compare elements.
"""

import random


def _partition3(a, lo, hi, rng):
    """Partition a[lo..hi] around a random pivot into  < | == | >.

    Returns (lt, gt) such that a[lo..lt-1] < pivot, a[lt..gt] == pivot,
    a[gt+1..hi] > pivot. One pass, Theta(hi - lo) comparisons.
    """
    p = rng.randint(lo, hi)
    a[lo], a[p] = a[p], a[lo]
    pivot = a[lo]
    lt, i, gt = lo, lo + 1, hi
    while i <= gt:
        if a[i] < pivot:
            a[lt], a[i] = a[i], a[lt]
            lt += 1
            i += 1
        elif pivot < a[i]:
            a[i], a[gt] = a[gt], a[i]
            gt -= 1
        else:
            i += 1
    return lt, gt


def quicksort(a, rng=None):
    """Randomized three-way Quicksort, in place. Returns `a`.

    Recurses on the smaller side and loops on the larger one, so the call
    stack is O(lg n) deep even in an unlucky run.
    """
    rng = rng or random.Random()

    def sort(lo, hi):
        while lo < hi:
            lt, gt = _partition3(a, lo, hi, rng)
            if lt - lo < hi - gt:
                sort(lo, lt - 1)
                lo = gt + 1
            else:
                sort(gt + 1, hi)
                hi = lt - 1

    sort(0, len(a) - 1)
    return a


def _merge(left, right):
    """Merge two sorted lists into a new sorted list. Stable."""
    out = []
    i = j = 0
    nl, nr = len(left), len(right)
    while i < nl and j < nr:
        if right[j] < left[i]:
            out.append(right[j])
            j += 1
        else:
            out.append(left[i])
            i += 1
    out.extend(left[i:])
    out.extend(right[j:])
    return out


def merge_sort(a):
    """Top-down Merge Sort. Returns a new sorted list; `a` is not modified.

    T(n) = 2T(n/2) + Theta(n) = Theta(n lg n) on every input.
    """
    if len(a) <= 1:
        return list(a)
    mid = len(a) // 2
    return _merge(merge_sort(a[:mid]), merge_sort(a[mid:]))


if __name__ == "__main__":
    demo = [random.randint(0, 20) for _ in range(15)]
    print("input     :", demo)
    print("merge_sort:", merge_sort(demo))
    print("quicksort :", quicksort(demo[:]))
