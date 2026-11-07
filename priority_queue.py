"""
Priority queue for Task objects, built on a binary heap stored in a list.

Why a list: a complete binary tree fits in an array with no gaps, so we
can find a parent or child with plain index math (same formulas as
heapsort.py). New items get appended and removed items get popped from
the end, and both of those are O(1) on a Python list. A linked tree would
need parent pointers and extra work to find the last node, and it
wouldn't be any faster.

Each slot in the list is a small list: [key, order, task]
  key   - the value we sort on (priority, deadline or arrival time),
          copied out of the task so we don't call getattr on every compare
  order - a counter that goes up on every insert. If two keys tie, the
          one inserted first wins, so equal-priority tasks come out FIFO.
  task  - the Task itself

We also keep `where`, a dict of task_id -> list index. It gets updated on
every swap. That way increase_key / decrease_key can find a task right
away instead of scanning the whole list, so they stay O(log n).

Running times (n = tasks in the queue):
  insert                    O(log n)
  extract_max / extract_min O(log n)
  increase_key/decrease_key O(log n)
  remove                    O(log n)
  peek, is_empty, len       O(1)
  building from a list      O(n)
"""

from dataclasses import dataclass, field
from itertools import count
from typing import Optional


@dataclass(eq=False)
class Task:
    """A job for the scheduler.

    Higher priority = more urgent (the simulation uses 1 to 10).
    deadline can be None if the task doesn't have one.
    The fields after `label` get filled in while the simulation runs.
    """

    task_id: int
    priority: int
    arrival: int = 0
    duration: int = 1
    deadline: Optional[int] = None
    label: str = ""

    original_priority: int = field(init=False)
    time_left: int = field(init=False)
    started_at: Optional[int] = field(default=None, init=False)
    finished_at: Optional[int] = field(default=None, init=False)

    def __post_init__(self):
        # remember the starting priority, since aging can bump it later
        self.original_priority = self.priority
        self.time_left = self.duration

    # these only make sense once the task is done
    @property
    def turnaround_time(self):
        return self.finished_at - self.arrival

    @property
    def wait_time(self):
        return self.turnaround_time - self.duration

    @property
    def response_time(self):
        return self.started_at - self.arrival

    @property
    def is_late(self):
        return self.deadline is not None and self.finished_at > self.deadline

    def __repr__(self):
        name = self.label or f"T{self.task_id}"
        return (f"Task({name}, prio={self.priority}, arrival={self.arrival}, "
                f"duration={self.duration}, deadline={self.deadline})")


class HeapPriorityQueue:
    """Binary heap of Tasks.

    HeapPriorityQueue()                             -> max-heap on priority
    HeapPriorityQueue(kind="min", key="deadline")   -> min-heap on deadline
    """

    def __init__(self, kind="max", key="priority", tasks=()):
        if kind not in ("max", "min"):
            raise ValueError("kind has to be 'max' or 'min'")
        self.kind = kind
        self.key = key
        self.items = []
        self.where = {}
        self._ticket = count()
        if tasks:
            self._load_all(tasks)

    def __len__(self):
        return len(self.items)

    def __contains__(self, task):
        return task.task_id in self.where

    def is_empty(self):
        return len(self.items) == 0

    def peek(self):
        """Look at the top task without taking it out."""
        if not self.items:
            raise IndexError("queue is empty")
        return self.items[0][2]

    def get_key(self, task):
        return self.items[self.where[task.task_id]][0]

    # ---- helpers that keep the heap in order ----

    def _goes_first(self, x, y):
        """Should entry x sit above entry y in the heap?"""
        if x[0] != y[0]:
            if self.kind == "max":
                return x[0] > y[0]
            return x[0] < y[0]
        return x[1] < y[1]  # tie on key -> whoever was inserted first

    def _swap(self, i, j):
        items = self.items
        items[i], items[j] = items[j], items[i]
        self.where[items[i][2].task_id] = i
        self.where[items[j][2].task_id] = j

    def _bubble_up(self, i):
        while i > 0:
            up = (i - 1) // 2
            if not self._goes_first(self.items[i], self.items[up]):
                break
            self._swap(i, up)
            i = up

    def _bubble_down(self, i):
        size = len(self.items)
        while True:
            l = 2 * i + 1
            r = l + 1
            best = i
            if l < size and self._goes_first(self.items[l], self.items[best]):
                best = l
            if r < size and self._goes_first(self.items[r], self.items[best]):
                best = r
            if best == i:
                return
            self._swap(i, best)
            i = best

    def _load_all(self, tasks):
        """Put a batch of tasks in and heapify once at the end. O(n)."""
        for t in tasks:
            if t.task_id in self.where:
                raise ValueError(f"task {t.task_id} is already in the queue")
            self.where[t.task_id] = len(self.items)
            self.items.append([getattr(t, self.key), next(self._ticket), t])
        for i in range(len(self.items) // 2 - 1, -1, -1):
            self._bubble_down(i)

    # ---- the main operations ----

    def insert(self, task):
        """Add a task. O(log n).

        Put it at the end of the list (the next open leaf) and let it bubble
        up while it beats its parent. The tree is about log n levels tall,
        so that's the most swaps it can take.
        """
        if task.task_id in self.where:
            raise ValueError(f"task {task.task_id} is already in the queue")
        self.items.append([getattr(task, self.key), next(self._ticket), task])
        self.where[task.task_id] = len(self.items) - 1
        self._bubble_up(len(self.items) - 1)

    def extract(self):
        """Take out the top task (max or min depending on kind). O(log n).

        Swap the root with the last item, pop it off the end, then bubble
        the new root down to where it belongs.
        """
        if not self.items:
            raise IndexError("queue is empty")
        self._swap(0, len(self.items) - 1)
        top = self.items.pop()[2]
        del self.where[top.task_id]
        if self.items:
            self._bubble_down(0)
        return top

    def extract_max(self):
        if self.kind != "max":
            raise TypeError("this is a min-heap, use extract_min()")
        return self.extract()

    def extract_min(self):
        if self.kind != "min":
            raise TypeError("this is a max-heap, use extract_max()")
        return self.extract()

    def _update_key(self, task, new_key):
        spot = self.where.get(task.task_id)
        if spot is None:
            raise KeyError(f"task {task.task_id} isn't in the queue")
        self.items[spot][0] = new_key
        setattr(task, self.key, new_key)  # keep the Task object in sync too
        return spot

    def increase_key(self, task, new_key):
        """Raise a task's key. O(log n).

        Finding the task is O(1) thanks to `where`. In a max-heap a bigger
        key can only move it up, so bubble up. In a min-heap bigger means
        less urgent, so it bubbles down instead.
        """
        spot = self.where.get(task.task_id)
        if spot is not None and new_key < self.items[spot][0]:
            raise ValueError("new key is smaller than the old one")
        spot = self._update_key(task, new_key)
        if self.kind == "max":
            self._bubble_up(spot)
        else:
            self._bubble_down(spot)

    def decrease_key(self, task, new_key):
        """Lower a task's key. O(log n). Opposite of increase_key."""
        spot = self.where.get(task.task_id)
        if spot is not None and self.items[spot][0] < new_key:
            raise ValueError("new key is bigger than the old one")
        spot = self._update_key(task, new_key)
        if self.kind == "max":
            self._bubble_down(spot)
        else:
            self._bubble_up(spot)

    def change_key(self, task, new_key):
        """Set the key either way, picks increase or decrease for you."""
        if new_key < self.get_key(task):
            self.decrease_key(task, new_key)
        else:
            self.increase_key(task, new_key)

    def remove(self, task):
        """Pull out any task, not just the top one (e.g. if it got cancelled)."""
        spot = self.where.get(task.task_id)
        if spot is None:
            raise KeyError(f"task {task.task_id} isn't in the queue")
        self._swap(spot, len(self.items) - 1)
        self.items.pop()
        del self.where[task.task_id]
        if spot < len(self.items):
            # the item we moved in could need to go either way
            self._bubble_up(spot)
            self._bubble_down(spot)
        return task

    def all_tasks(self):
        """Tasks in the order they sit in the list (not sorted)."""
        return [entry[2] for entry in self.items]

    def validate(self):
        """Check the heap rule and the `where` dict. Only used in tests."""
        assert len(self.where) == len(self.items)
        for i, entry in enumerate(self.items):
            assert self.where[entry[2].task_id] == i
            assert entry[0] == getattr(entry[2], self.key)
            if i > 0:
                parent = self.items[(i - 1) // 2]
                assert not self._goes_first(entry, parent), f"heap broken at {i}"


if __name__ == "__main__":
    pq = HeapPriorityQueue()
    jobs = [Task(1, 3, label="backup"), Task(2, 9, label="page-fault"),
            Task(3, 5, label="compile"), Task(4, 1, label="log-rotate")]
    for job in jobs:
        pq.insert(job)
    print("in queue:", [t.label for t in pq.all_tasks()], "(list order)")

    pq.increase_key(jobs[3], 10)
    print("bumped log-rotate to 10, top is now:", pq.peek().label)
    pq.decrease_key(jobs[1], 2)
    print("dropped page-fault to 2")

    done = []
    while not pq.is_empty():
        done.append(pq.extract_max().label)
    print("run order:", done)
