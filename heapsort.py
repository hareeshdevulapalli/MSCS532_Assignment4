"""Heapsort on a binary max-heap stored in a Python list.

Follows HEAPSORT, BUILD-MAX-HEAP and MAX-HEAPIFY from *Introduction to
Algorithms* (Cormen et al., 4th ed., Chapter 6), using 0-based indices:

    parent(i) = (i - 1) // 2
    left(i)   = 2i + 1
    right(i)  = 2i + 2

The max-heap property is A[parent(i)] >= A[i] for every i > 0, so the
largest element is always at A[0].

Heapsort sorts in place: the heap and the sorted suffix share one list.
After each extraction the heap shrinks by one and the freed slot at the
end receives the element just removed.

    A = [heap ......... | sorted suffix]
         0     heap_size-1  heap_size    n-1

Only the `<` operator is used on elements, so any mutually comparable
values work (ints, floats, strings, tuples, objects defining __lt__).
"""


def max_heapify(a, i, heap_size):
    """Sift a[i] down until the subtree rooted at i is a max-heap.

    Precondition: the subtrees rooted at left(i) and right(i) are already
    max-heaps; only a[i] may be out of place.

    Written as a loop rather than CLRS's tail recursion, which saves
    Python call overhead and cannot hit the recursion limit. Each pass
    moves one level down the tree, so the running time is O(h) where
    h = floor(lg heap_size) is the height of node i's subtree.
    """
    while True:
        left = 2 * i + 1
        right = left + 1
        largest = i
        if left < heap_size and a[largest] < a[left]:
            largest = left
        if right < heap_size and a[largest] < a[right]:
            largest = right
        if largest == i:
            return
        a[i], a[largest] = a[largest], a[i]
        i = largest


def build_max_heap(a):
    """Rearrange list `a` in place into a max-heap. Runs in O(n).

    Nodes at indices n//2 .. n-1 are leaves, which are trivially
    one-element heaps, so only the internal nodes n//2 - 1 down to 0 are
    heapified, bottom-up. Although each call is O(lg n), most nodes sit
    near the bottom where heapify is cheap; summing the heights gives
    sum_{h} ceil(n / 2^(h+1)) * O(h) = O(n * sum h/2^h) = O(n).
    """
    n = len(a)
    for i in range(n // 2 - 1, -1, -1):
        max_heapify(a, i, n)


def heapsort(a):
    """Sort list `a` in ascending order, in place. Also returns `a`.

    1. BUILD-MAX-HEAP:            O(n)
    2. Repeat n - 1 times:        (n - 1) * O(lg n)
         swap the max a[0] to the end of the heap region,
         shrink the heap by one,
         MAX-HEAPIFY the new root.
    Total: O(n lg n) time, O(1) extra space. Not stable.
    """
    build_max_heap(a)
    for end in range(len(a) - 1, 0, -1):
        a[0], a[end] = a[end], a[0]     # current max goes to its final slot
        max_heapify(a, 0, end)          # heap is now a[0:end]
    return a


def heapsorted(iterable):
    """Return a new sorted list, leaving the input untouched (like sorted())."""
    return heapsort(list(iterable))


if __name__ == "__main__":
    import random

    demo = [random.randint(0, 99) for _ in range(15)]
    print("input :", demo)
    heap = demo[:]
    build_max_heap(heap)
    print("heap  :", heap)
    print("sorted:", heapsort(demo))
