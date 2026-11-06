"""Array-backed binary-heap priority queue of Task objects.

Design
------
* Storage: one Python list used as an implicit complete binary tree
  (CLRS Chapter 6, 0-based): parent(i) = (i-1)//2, children 2i+1, 2i+2.
  A list gives O(1) index arithmetic, contiguous storage, amortized O(1)
  append/pop at the end (exactly where a heap grows and shrinks) and no
  per-node pointers. A linked tree would need parent pointers and a way to
  find the "last" node, for no asymptotic gain.

* Entries: each slot holds a small list [key, seq, task]. `key` is a cached
  copy of the task attribute the queue is ordered on (priority, deadline or
  arrival time). `seq` is an insertion counter used to break ties in FIFO
  order, so equal-priority tasks are served in the order they were queued
  and the scheduler's results are deterministic.

* Position map: `_pos[task_id] -> index in the list`. It is updated on
  every swap, which costs O(1), and lets increase_key / decrease_key /
  remove find a task in O(1) instead of an O(n) scan. Without it, changing
  a key would be O(n); with it, O(lg n).

* Order: `order="max"` serves the largest key first (a max-priority queue,
  used for priority scheduling, where 10 is most urgent); `order="min"`
  serves the smallest first (used for earliest-deadline-first and for FCFS
  by arrival time). The same code handles both through `_above()`.

Complexities (n = number of queued tasks)
-----------------------------------------
    insert            O(lg n)   append, then sift up at most height levels
    peek              O(1)      the root
    extract_max/min   O(lg n)   move last leaf to root, sift down
    increase_key      O(lg n)   O(1) lookup + sift up (max) / down (min)
    decrease_key      O(lg n)   O(1) lookup + sift down (max) / up (min)
    remove            O(lg n)   swap with last, pop, sift either way
    is_empty, len     O(1)
    build (heapify)   O(n)      bottom-up, as in BUILD-MAX-HEAP
Space: Theta(n) for the list plus Theta(n) for the position map.
"""

import itertools
from dataclasses import dataclass, field
from typing import Optional


@dataclass(eq=False)
class Task:
    """One schedulable unit of work.

    task_id       unique identifier (used as the position-map key)
    priority      larger = more urgent (1..10 in the simulation); may be
                  raised over time by aging
    arrival_time  tick at which the task becomes ready
    burst_time    CPU ticks the task needs
    deadline      tick by which it should finish (None = no deadline)
    name          optional label for printing

    The scheduler fills in the bookkeeping fields below.
    """

    task_id: int
    priority: int
    arrival_time: int = 0
    burst_time: int = 1
    deadline: Optional[int] = None
    name: str = ""

    base_priority: int = field(init=False)
    remaining: int = field(init=False)
    start_time: Optional[int] = field(default=None, init=False)
    finish_time: Optional[int] = field(default=None, init=False)

    def __post_init__(self):
        self.base_priority = self.priority
        self.remaining = self.burst_time

    # Derived metrics, valid once the task has finished.
    @property
    def turnaround(self):
        return self.finish_time - self.arrival_time

    @property
    def waiting(self):
        return self.turnaround - self.burst_time

    @property
    def response(self):
        return self.start_time - self.arrival_time

    @property
    def missed_deadline(self):
        return self.deadline is not None and self.finish_time > self.deadline

    def __repr__(self):
        label = self.name or f"T{self.task_id}"
        return (f"Task({label}, prio={self.priority}, arr={self.arrival_time}, "
                f"burst={self.burst_time}, dl={self.deadline})")


class PriorityQueue:
    """Binary-heap priority queue of Task objects keyed on one attribute.

    PriorityQueue()                               max-heap on task.priority
    PriorityQueue(order="min", key="deadline")    min-heap on task.deadline
    """

    def __init__(self, order="max", key="priority", tasks=()):
        if order not in ("max", "min"):
            raise ValueError("order must be 'max' or 'min'")
        self.order = order
        self.key = key
        self._heap = []
        self._pos = {}
        self._counter = itertools.count()
        if tasks:
            self._build(tasks)

    # ----------------------------------------------------------- basics
    def __len__(self):
        return len(self._heap)

    def is_empty(self):
        """True if no tasks are queued. O(1)."""
        return not self._heap

    def __contains__(self, task):
        return task.task_id in self._pos

    def peek(self):
        """Return the highest-priority task without removing it. O(1)."""
        if not self._heap:
            raise IndexError("peek from an empty priority queue")
        return self._heap[0][2]

    def key_of(self, task):
        return self._heap[self._pos[task.task_id]][0]

    # ------------------------------------------------- heap primitives
    def _above(self, a, b):
        """True if entry a belongs closer to the root than entry b."""
        if a[0] != b[0]:
            return a[0] > b[0] if self.order == "max" else a[0] < b[0]
        return a[1] < b[1]                  # FIFO among equal keys

    def _swap(self, i, j):
        h = self._heap
        h[i], h[j] = h[j], h[i]
        self._pos[h[i][2].task_id] = i
        self._pos[h[j][2].task_id] = j

    def _sift_up(self, i):
        """Move entry i toward the root while it beats its parent. O(lg n)."""
        h = self._heap
        while i > 0:
            parent = (i - 1) // 2
            if not self._above(h[i], h[parent]):
                break
            self._swap(i, parent)
            i = parent

    def _sift_down(self, i):
        """MAX-HEAPIFY generalised to either order. O(lg n)."""
        h = self._heap
        n = len(h)
        while True:
            left = 2 * i + 1
            right = left + 1
            best = i
            if left < n and self._above(h[left], h[best]):
                best = left
            if right < n and self._above(h[right], h[best]):
                best = right
            if best == i:
                return
            self._swap(i, best)
            i = best

    def _build(self, tasks):
        """Load many tasks at once in O(n), bottom-up like BUILD-MAX-HEAP."""
        for task in tasks:
            if task.task_id in self._pos:
                raise ValueError(f"task {task.task_id} is already queued")
            self._pos[task.task_id] = len(self._heap)
            self._heap.append([getattr(task, self.key), next(self._counter), task])
        for i in range(len(self._heap) // 2 - 1, -1, -1):
            self._sift_down(i)

    # ------------------------------------------------- core operations
    def insert(self, task):
        """Add a task, keyed on its current `self.key` attribute. O(lg n).

        The new entry goes in the first free leaf (end of the list), which
        keeps the tree complete, and then sifts up past any parent it
        outranks. The path from a leaf to the root has floor(lg n) edges,
        so there are at most that many swaps.
        """
        if task.task_id in self._pos:
            raise ValueError(f"task {task.task_id} is already queued")
        self._heap.append([getattr(task, self.key), next(self._counter), task])
        self._pos[task.task_id] = len(self._heap) - 1
        self._sift_up(len(self._heap) - 1)

    def extract(self):
        """Remove and return the top task (max or min, by `order`). O(lg n).

        The root is swapped with the last leaf, the old root is popped off
        the end in O(1), and the leaf now at the root sifts down. Sift-down
        does two comparisons per level over at most floor(lg n) levels.
        """
        if not self._heap:
            raise IndexError("extract from an empty priority queue")
        self._swap(0, len(self._heap) - 1)
        _, _, task = self._heap.pop()
        del self._pos[task.task_id]
        if self._heap:
            self._sift_down(0)
        return task

    def extract_max(self):
        """HEAP-EXTRACT-MAX. Only valid on a max-heap."""
        if self.order != "max":
            raise TypeError("extract_max called on a min-heap; use extract_min")
        return self.extract()

    def extract_min(self):
        """HEAP-EXTRACT-MIN. Only valid on a min-heap."""
        if self.order != "min":
            raise TypeError("extract_min called on a max-heap; use extract_max")
        return self.extract()

    def _set_key(self, task, new_key):
        i = self._pos.get(task.task_id)
        if i is None:
            raise KeyError(f"task {task.task_id} is not in the queue")
        old_key = self._heap[i][0]
        self._heap[i][0] = new_key
        setattr(task, self.key, new_key)    # keep the Task in sync
        return i, old_key

    def increase_key(self, task, new_key):
        """Raise a queued task's key to `new_key`. O(lg n).

        Max-heap: the task can only move up (it now beats more nodes), so
        sift up -- this is CLRS's HEAP-INCREASE-KEY. Min-heap: a larger key
        means *less* urgent, so sift down. Either way: O(1) lookup through
        the position map, then at most floor(lg n) swaps.
        """
        i = self._pos.get(task.task_id)
        if i is not None and new_key < self._heap[i][0]:
            raise ValueError("new key is smaller than the current key")
        i, _ = self._set_key(task, new_key)
        if self.order == "max":
            self._sift_up(i)
        else:
            self._sift_down(i)

    def decrease_key(self, task, new_key):
        """Lower a queued task's key to `new_key`. O(lg n).

        Mirror image of increase_key: sift down in a max-heap, up in a
        min-heap (CLRS's MIN-HEAP-DECREASE-KEY).
        """
        i = self._pos.get(task.task_id)
        if i is not None and self._heap[i][0] < new_key:
            raise ValueError("new key is larger than the current key")
        i, _ = self._set_key(task, new_key)
        if self.order == "max":
            self._sift_down(i)
        else:
            self._sift_up(i)

    def change_key(self, task, new_key):
        """Set a task's key in either direction. O(lg n)."""
        if new_key < self.key_of(task):
            self.decrease_key(task, new_key)
        else:
            self.increase_key(task, new_key)

    def remove(self, task):
        """Delete an arbitrary queued task (e.g. a cancelled job). O(lg n)."""
        i = self._pos.get(task.task_id)
        if i is None:
            raise KeyError(f"task {task.task_id} is not in the queue")
        last = len(self._heap) - 1
        self._swap(i, last)
        self._heap.pop()
        del self._pos[task.task_id]
        if i < len(self._heap):
            self._sift_up(i)
            self._sift_down(i)
        return task

    def tasks(self):
        """The queued tasks in heap (array) order -- not sorted."""
        return [entry[2] for entry in self._heap]

    def check_invariants(self):
        """Assert the heap property and position map are consistent. O(n).

        Used by the tests; not called in normal operation.
        """
        h = self._heap
        assert len(self._pos) == len(h)
        for i, entry in enumerate(h):
            assert self._pos[entry[2].task_id] == i
            assert entry[0] == getattr(entry[2], self.key)
            if i > 0:
                assert not self._above(entry, h[(i - 1) // 2]), f"heap violated at {i}"


if __name__ == "__main__":
    pq = PriorityQueue()
    jobs = [Task(1, 3, name="backup"), Task(2, 9, name="page-fault"),
            Task(3, 5, name="compile"), Task(4, 1, name="log-rotate")]
    for job in jobs:
        pq.insert(job)
    print("queued   :", [t.name for t in pq.tasks()], "(heap order)")
    pq.increase_key(jobs[3], 10)
    print("log-rotate raised to 10; next up:", pq.peek().name)
    pq.decrease_key(jobs[1], 2)
    print("page-fault lowered to 2")
    order = []
    while not pq.is_empty():
        order.append(pq.extract_max().name)
    print("served   :", order)
