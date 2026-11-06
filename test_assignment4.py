"""Unit tests for Heapsort, the baseline sorts, the priority queue and the
scheduler.

    python3 -m unittest -v test_assignment4
"""

import random
import unittest

from heapsort import build_max_heap, heapsort, heapsorted
from other_sorts import merge_sort, quicksort
from priority_queue import PriorityQueue, Task
from scheduler import POLICIES, generate_workload, simulate


def is_max_heap(a):
    return all(not (a[(i - 1) // 2] < a[i]) for i in range(1, len(a)))


class TestSorts(unittest.TestCase):
    CASES = [[], [1], [2, 1], [1, 2], [3, 3, 3], [5, -1, 0, 5, -1],
             list(range(50)), list(range(50, 0, -1)), [0.5, -2.25, 3.0, 1e9]]

    def check(self, fn):
        rng = random.Random(1)
        cases = self.CASES + [[rng.randint(-100, 100) for _ in range(n)] for n in range(2, 300, 17)]
        cases.append([rng.randint(0, 3) for _ in range(500)])    # many duplicates
        for case in cases:
            with self.subTest(case=case[:10]):
                self.assertEqual(fn(case[:]), sorted(case))

    def test_heapsort(self):
        self.check(heapsort)

    def test_quicksort(self):
        self.check(quicksort)

    def test_merge_sort(self):
        self.check(merge_sort)

    def test_heapsort_in_place(self):
        a = [4, 1, 3]
        result = heapsort(a)
        self.assertIs(result, a)
        self.assertEqual(a, [1, 3, 4])

    def test_heapsorted_leaves_input(self):
        a = [3, 1, 2]
        self.assertEqual(heapsorted(a), [1, 2, 3])
        self.assertEqual(a, [3, 1, 2])

    def test_strings_and_tuples(self):
        words = ["pear", "apple", "fig", "banana"]
        self.assertEqual(heapsort(words[:]), sorted(words))
        pairs = [(2, "b"), (1, "z"), (2, "a")]
        self.assertEqual(heapsort(pairs[:]), sorted(pairs))

    def test_build_max_heap(self):
        rng = random.Random(2)
        for n in range(0, 100):
            a = [rng.randint(0, 50) for _ in range(n)]
            build_max_heap(a)
            self.assertTrue(is_max_heap(a))
            if a:
                self.assertEqual(a[0], max(a))


class TestPriorityQueue(unittest.TestCase):
    def test_empty(self):
        pq = PriorityQueue()
        self.assertTrue(pq.is_empty())
        self.assertEqual(len(pq), 0)
        with self.assertRaises(IndexError):
            pq.extract_max()
        with self.assertRaises(IndexError):
            pq.peek()

    def test_extract_max_order(self):
        rng = random.Random(3)
        pq = PriorityQueue()
        tasks = [Task(i, rng.randint(1, 20)) for i in range(200)]
        for t in tasks:
            pq.insert(t)
            pq.check_invariants()
        self.assertFalse(pq.is_empty())
        out = [pq.extract_max().priority for _ in range(len(tasks))]
        self.assertEqual(out, sorted(out, reverse=True))
        self.assertTrue(pq.is_empty())

    def test_min_heap(self):
        pq = PriorityQueue(order="min", key="deadline")
        for i, d in enumerate([30, 10, 20, 5]):
            pq.insert(Task(i, priority=1, deadline=d))
        self.assertEqual([pq.extract_min().deadline for _ in range(4)], [5, 10, 20, 30])

    def test_wrong_extract_raises(self):
        with self.assertRaises(TypeError):
            PriorityQueue(order="min").extract_max()
        with self.assertRaises(TypeError):
            PriorityQueue(order="max").extract_min()

    def test_fifo_tie_break(self):
        pq = PriorityQueue()
        for i in range(5):
            pq.insert(Task(i, priority=7))
        self.assertEqual([pq.extract_max().task_id for _ in range(5)], [0, 1, 2, 3, 4])

    def test_duplicate_insert_rejected(self):
        pq = PriorityQueue()
        t = Task(1, 5)
        pq.insert(t)
        with self.assertRaises(ValueError):
            pq.insert(t)

    def test_increase_key(self):
        pq = PriorityQueue()
        tasks = [Task(i, i) for i in range(10)]
        for t in tasks:
            pq.insert(t)
        pq.increase_key(tasks[0], 100)
        pq.check_invariants()
        self.assertIs(pq.peek(), tasks[0])
        self.assertEqual(tasks[0].priority, 100)     # Task object updated too
        with self.assertRaises(ValueError):
            pq.increase_key(tasks[0], 1)

    def test_decrease_key(self):
        pq = PriorityQueue()
        tasks = [Task(i, i) for i in range(10)]
        for t in tasks:
            pq.insert(t)
        pq.decrease_key(tasks[9], -1)
        pq.check_invariants()
        self.assertIs(pq.peek(), tasks[8])
        out = [pq.extract_max() for _ in range(10)]
        self.assertIs(out[-1], tasks[9])
        with self.assertRaises(ValueError):
            pq.insert(tasks[9])
            pq.decrease_key(tasks[9], 50)

    def test_key_change_on_min_heap(self):
        pq = PriorityQueue(order="min", key="deadline")
        tasks = [Task(i, 1, deadline=10 * i) for i in range(10)]
        for t in tasks:
            pq.insert(t)
        pq.decrease_key(tasks[7], -5)       # becomes most urgent
        self.assertIs(pq.peek(), tasks[7])
        pq.increase_key(tasks[7], 1000)     # becomes least urgent
        pq.check_invariants()
        self.assertIs(pq.peek(), tasks[0])

    def test_key_of_missing_task(self):
        pq = PriorityQueue()
        with self.assertRaises(KeyError):
            pq.increase_key(Task(1, 1), 5)

    def test_random_operations_against_reference(self):
        """Mixed inserts, key changes, removals and extracts vs a plain list."""
        rng = random.Random(4)
        pq = PriorityQueue()
        live = {}
        next_id = 0
        for _ in range(3000):
            op = rng.random()
            if op < 0.4 or not live:
                t = Task(next_id, rng.randint(0, 100))
                next_id += 1
                pq.insert(t)
                live[t.task_id] = t
            elif op < 0.6:
                t = rng.choice(list(live.values()))
                pq.change_key(t, rng.randint(0, 100))
            elif op < 0.7:
                t = live.pop(rng.choice(list(live)))
                pq.remove(t)
            else:
                best = max(t.priority for t in live.values())
                t = pq.extract_max()
                self.assertEqual(t.priority, best)
                del live[t.task_id]
            self.assertEqual(len(pq), len(live))
        pq.check_invariants()

    def test_bulk_build(self):
        rng = random.Random(5)
        tasks = [Task(i, rng.randint(0, 1000)) for i in range(500)]
        pq = PriorityQueue(tasks=tasks)
        pq.check_invariants()
        self.assertEqual(pq.peek().priority, max(t.priority for t in tasks))


class TestScheduler(unittest.TestCase):
    def test_all_tasks_finish_correctly(self):
        tasks = generate_workload(300, load=0.9, seed=11)
        total_burst = sum(t.burst_time for t in tasks)
        for policy in POLICIES:
            with self.subTest(policy=policy):
                finished, extra = simulate(tasks, policy)
                self.assertEqual(len(finished), len(tasks))
                busy = sum(1 for x in extra["timeline"] if x is not None)
                self.assertEqual(busy, total_burst)
                for t in finished:
                    self.assertGreaterEqual(t.start_time, t.arrival_time)
                    self.assertGreaterEqual(t.waiting, 0)
                    self.assertEqual(t.remaining, 0)

    def test_input_not_mutated(self):
        tasks = generate_workload(50, load=0.9, seed=1)
        before = [(t.priority, t.remaining, t.start_time) for t in tasks]
        simulate(tasks, "Priority + aging")
        self.assertEqual(before, [(t.priority, t.remaining, t.start_time) for t in tasks])

    def test_fcfs_order(self):
        tasks = generate_workload(100, load=0.9, seed=2)
        finished, _ = simulate(tasks, "FCFS")
        starts = [(t.arrival_time, t.task_id) for t in sorted(finished, key=lambda t: t.start_time)]
        self.assertEqual(starts, sorted(starts))

    def test_preemption(self):
        tasks = [Task(1, priority=1, arrival_time=0, burst_time=10),
                 Task(2, priority=9, arrival_time=2, burst_time=2)]
        finished, extra = simulate(tasks, "Priority (preemptive)")
        by_id = {t.task_id: t for t in finished}
        self.assertEqual(by_id[2].start_time, 2)
        self.assertEqual(by_id[2].finish_time, 4)
        self.assertEqual(by_id[1].finish_time, 12)
        self.assertEqual(extra["preemptions"], 1)

        finished, extra = simulate(tasks, "Priority")
        by_id = {t.task_id: t for t in finished}
        self.assertEqual(by_id[1].finish_time, 10)
        self.assertEqual(extra["preemptions"], 0)

    def test_aging_prevents_starvation(self):
        # A low-priority task keeps losing to a stream of high-priority ones.
        tasks = [Task(0, priority=1, arrival_time=0, burst_time=1)]
        tasks += [Task(i, priority=10, arrival_time=i - 1, burst_time=1) for i in range(1, 200)]
        plain = {t.task_id: t for t in simulate(tasks, "Priority (preemptive)")[0]}
        aged = {t.task_id: t for t in simulate(tasks, "Priority + aging")[0]}
        self.assertGreater(plain[0].waiting, 150)
        self.assertLess(aged[0].waiting, 60)

    def test_idle_gap(self):
        tasks = [Task(1, 5, arrival_time=0, burst_time=2), Task(2, 5, arrival_time=10, burst_time=1)]
        finished, extra = simulate(tasks, "Priority")
        self.assertEqual(extra["makespan"], 11)
        self.assertEqual(extra["timeline"].count(None), 8)


if __name__ == "__main__":
    unittest.main()
