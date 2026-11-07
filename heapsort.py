"""
Heapsort using a max-heap kept inside a normal Python list.

Based on the HEAPSORT pseudocode in CLRS chapter 6, but with 0-based
indexing, so for a node at position i:
    parent = (i - 1) // 2
    left child = 2*i + 1
    right child = 2*i + 2

Max-heap rule: every parent is >= its children, so the biggest value
sits at index 0.

The sort works in place. The front of the list is the heap and the back
fills up with sorted values as we pull the max out each round.
"""


def heapify(arr, root, size):
    """Push arr[root] down until it's bigger than both its children.

    Only looks at the first `size` items of arr. Assumes the two subtrees
    under root are already valid heaps. I used a loop instead of recursion
    so there's no extra call overhead and no recursion limit to worry about.
    Cost is O(height of root), which is at most O(log n).
    """
    while True:
        l = 2 * root + 1
        r = l + 1
        biggest = root

        if l < size and arr[biggest] < arr[l]:
            biggest = l
        if r < size and arr[biggest] < arr[r]:
            biggest = r

        if biggest == root:
            break  # heap property holds here, done

        arr[root], arr[biggest] = arr[biggest], arr[root]
        root = biggest


def build_heap(arr):
    """Turn the whole list into a max-heap, bottom up. Runs in O(n).

    Everything from index n//2 onward is a leaf, so we only need to fix the
    nodes before that, going backwards toward the root.
    """
    n = len(arr)
    for node in range(n // 2 - 1, -1, -1):
        heapify(arr, node, n)


def heap_sort(arr):
    """Sort arr ascending in place (and return it for convenience).

    Steps:
      1. build a max-heap                         O(n)
      2. n-1 times: swap the max to the back,
         shrink the heap by one, fix the root     O(log n) each
    Overall O(n log n) time, O(1) extra memory, not stable.
    """
    build_heap(arr)
    for last in range(len(arr) - 1, 0, -1):
        # biggest item goes to its final spot at the end
        arr[0], arr[last] = arr[last], arr[0]
        heapify(arr, 0, last)
    return arr


def heap_sorted(values):
    """Like the built-in sorted(): returns a new list, input is left alone."""
    return heap_sort(list(values))


if __name__ == "__main__":
    import random

    nums = [random.randint(0, 99) for _ in range(15)]
    print("input :", nums)
    as_heap = nums[:]
    build_heap(as_heap)
    print("heap  :", as_heap)
    print("sorted:", heap_sort(nums))
