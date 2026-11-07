"""
Tests for the sorts, the priority queue and the scheduler.

Run with:  python3 -m unittest -v test_heaps
"""

import random
import unittest

from heapsort import build_heap, heap_sort, heap_sorted
from priority_queue import HeapPriorityQueue, Task
from scheduler import SCHEDULERS, make_tasks, run_simulation
from sorts import merge_sort, quick_sort


def looks_like_max_heap(arr):
    return all(not (arr[(i - 1) // 2] < arr[i]) for i in range(1, len(arr)))


class SortingTests(unittest.TestCase):
    EDGE_CASES = [[], [1], [2, 1], [1, 2], [3, 3, 3], [5, -1, 0, 5, -1],
                  list(range(50)), list(range(50, 0, -1)), [0.5, -2.25, 3.0, 1e9]]

    def run_cases(self, sort_fn):
        rand = random.Random(1)
        cases = list(self.EDGE_CASES)
        cases += [[rand.randint(-100, 100) for _ in range(n)] for n in range(2, 300, 17)]
        cases.append([rand.randint(0, 3) for _ in range(500)])  # tons of duplicates
        for case in cases:
            with self.subTest(case=case[:10]):
                self.assertEqual(sort_fn(case[:]), sorted(case))

    def test_heap_sort(self):
        self.run_cases(heap_sort)

    def test_quick_sort(self):
        self.run_cases(quick_sort)

    def test_merge_sort(self):
        self.run_cases(merge_sort)

    def test_heap_sort_is_in_place(self):
        nums = [4, 1, 3]
        out = heap_sort(nums)
        self.assertIs(out, nums)
        self.assertEqual(nums, [1, 3, 4])

    def test_heap_sorted_returns_copy(self):
        nums = [3, 1, 2]
        self.assertEqual(heap_sorted(nums), [1, 2, 3])
        self.assertEqual(nums, [3, 1, 2])

    def test_other_types(self):
        words = ["pear", "apple", "fig", "banana"]
        self.assertEqual(heap_sort(words[:]), sorted(words))
        pairs = [(2, "b"), (1, "z"), (2, "a")]
        self.assertEqual(heap_sort(pairs[:]), sorted(pairs))

    def test_build_heap(self):
        rand = random.Random(2)
        for n in range(100):
            nums = [rand.randint(0, 50) for _ in range(n)]
            build_heap(nums)
            self.assertTrue(looks_like_max_heap(nums))
            if nums:
                self.assertEqual(nums[0], max(nums))


class QueueTests(unittest.TestCase):
    def test_empty_queue(self):
        pq = HeapPriorityQueue()
        self.assertTrue(pq.is_empty())
        self.assertEqual(len(pq), 0)
        with self.assertRaises(IndexError):
            pq.extract_max()
        with self.assertRaises(IndexError):
            pq.peek()

    def test_comes_out_highest_first(self):
        rand = random.Random(3)
        pq = HeapPriorityQueue()
        tasks = [Task(i, rand.randint(1, 20)) for i in range(200)]
        for t in tasks:
            pq.insert(t)
            pq.validate()
        self.assertFalse(pq.is_empty())
        prios = [pq.extract_max().priority for _ in range(len(tasks))]
        self.assertEqual(prios, sorted(prios, reverse=True))
        self.assertTrue(pq.is_empty())

    def test_min_heap_on_deadline(self):
        pq = HeapPriorityQueue(kind="min", key="deadline")
        for i, d in enumerate([30, 10, 20, 5]):
            pq.insert(Task(i, priority=1, deadline=d))
        self.assertEqual([pq.extract_min().deadline for _ in range(4)], [5, 10, 20, 30])

    def test_wrong_extract_for_kind(self):
        with self.assertRaises(TypeError):
            HeapPriorityQueue(kind="min").extract_max()
        with self.assertRaises(TypeError):
            HeapPriorityQueue(kind="max").extract_min()

    def test_ties_are_fifo(self):
        pq = HeapPriorityQueue()
        for i in range(5):
            pq.insert(Task(i, priority=7))
        self.assertEqual([pq.extract_max().task_id for _ in range(5)], [0, 1, 2, 3, 4])

    def test_cant_insert_twice(self):
        pq = HeapPriorityQueue()
        t = Task(1, 5)
        pq.insert(t)
        with self.assertRaises(ValueError):
            pq.insert(t)

    def test_increase_key(self):
        pq = HeapPriorityQueue()
        tasks = [Task(i, i) for i in range(10)]
        for t in tasks:
            pq.insert(t)
        pq.increase_key(tasks[0], 100)
        pq.validate()
        self.assertIs(pq.peek(), tasks[0])
        self.assertEqual(tasks[0].priority, 100)  # the Task itself got updated
        with self.assertRaises(ValueError):
            pq.increase_key(tasks[0], 1)

    def test_decrease_key(self):
        pq = HeapPriorityQueue()
        tasks = [Task(i, i) for i in range(10)]
        for t in tasks:
            pq.insert(t)
        pq.decrease_key(tasks[9], -1)
        pq.validate()
        self.assertIs(pq.peek(), tasks[8])
        order = [pq.extract_max() for _ in range(10)]
        self.assertIs(order[-1], tasks[9])

        pq.insert(tasks[9])
        with self.assertRaises(ValueError):
            pq.decrease_key(tasks[9], 50)

    def test_key_changes_on_min_heap(self):
        pq = HeapPriorityQueue(kind="min", key="deadline")
        tasks = [Task(i, 1, deadline=10 * i) for i in range(10)]
        for t in tasks:
            pq.insert(t)
        pq.decrease_key(tasks[7], -5)    # now the most urgent
        self.assertIs(pq.peek(), tasks[7])
        pq.increase_key(tasks[7], 1000)  # now the least urgent
        pq.validate()
        self.assertIs(pq.peek(), tasks[0])

    def test_unknown_task(self):
        pq = HeapPriorityQueue()
        with self.assertRaises(KeyError):
            pq.increase_key(Task(1, 1), 5)

    def test_random_ops_match_simple_list(self):
        """Do a bunch of random operations and compare against a plain dict."""
        rand = random.Random(4)
        pq = HeapPriorityQueue()
        alive = {}
        new_id = 0
        for _ in range(3000):
            roll = rand.random()
            if roll < 0.4 or not alive:
                t = Task(new_id, rand.randint(0, 100))
                new_id += 1
                pq.insert(t)
                alive[t.task_id] = t
            elif roll < 0.6:
                t = rand.choice(list(alive.values()))
                pq.change_key(t, rand.randint(0, 100))
            elif roll < 0.7:
                t = alive.pop(rand.choice(list(alive)))
                pq.remove(t)
            else:
                expected = max(t.priority for t in alive.values())
                t = pq.extract_max()
                self.assertEqual(t.priority, expected)
                del alive[t.task_id]
            self.assertEqual(len(pq), len(alive))
        pq.validate()

    def test_build_from_list(self):
        rand = random.Random(5)
        tasks = [Task(i, rand.randint(0, 1000)) for i in range(500)]
        pq = HeapPriorityQueue(tasks=tasks)
        pq.validate()
        self.assertEqual(pq.peek().priority, max(t.priority for t in tasks))


class SchedulerTests(unittest.TestCase):
    def test_every_task_finishes(self):
        tasks = make_tasks(300, load=0.9, seed=11)
        total_work = sum(t.duration for t in tasks)
        for policy in SCHEDULERS:
            with self.subTest(policy=policy):
                done, info = run_simulation(tasks, policy)
                self.assertEqual(len(done), len(tasks))
                busy_ticks = sum(1 for x in info["timeline"] if x is not None)
                self.assertEqual(busy_ticks, total_work)
                for t in done:
                    self.assertGreaterEqual(t.started_at, t.arrival)
                    self.assertGreaterEqual(t.wait_time, 0)
                    self.assertEqual(t.time_left, 0)

    def test_original_tasks_untouched(self):
        tasks = make_tasks(50, load=0.9, seed=1)
        before = [(t.priority, t.time_left, t.started_at) for t in tasks]
        run_simulation(tasks, "Priority + aging")
        self.assertEqual(before, [(t.priority, t.time_left, t.started_at) for t in tasks])

    def test_fcfs_goes_by_arrival(self):
        tasks = make_tasks(100, load=0.9, seed=2)
        done, _ = run_simulation(tasks, "FCFS")
        start_order = [(t.arrival, t.task_id) for t in sorted(done, key=lambda t: t.started_at)]
        self.assertEqual(start_order, sorted(start_order))

    def test_preemption(self):
        tasks = [Task(1, priority=1, arrival=0, duration=10),
                 Task(2, priority=9, arrival=2, duration=2)]
        done, info = run_simulation(tasks, "Priority (preemptive)")
        by_id = {t.task_id: t for t in done}
        self.assertEqual(by_id[2].started_at, 2)
        self.assertEqual(by_id[2].finished_at, 4)
        self.assertEqual(by_id[1].finished_at, 12)
        self.assertEqual(info["preemptions"], 1)

        done, info = run_simulation(tasks, "Priority")
        by_id = {t.task_id: t for t in done}
        self.assertEqual(by_id[1].finished_at, 10)
        self.assertEqual(info["preemptions"], 0)

    def test_aging_stops_starvation(self):
        # one low priority task vs a steady stream of high priority ones
        tasks = [Task(0, priority=1, arrival=0, duration=1)]
        tasks += [Task(i, priority=10, arrival=i - 1, duration=1) for i in range(1, 200)]
        without = {t.task_id: t for t in run_simulation(tasks, "Priority (preemptive)")[0]}
        with_aging = {t.task_id: t for t in run_simulation(tasks, "Priority + aging")[0]}
        self.assertGreater(without[0].wait_time, 150)
        self.assertLess(with_aging[0].wait_time, 60)

    def test_cpu_idles_between_tasks(self):
        tasks = [Task(1, 5, arrival=0, duration=2), Task(2, 5, arrival=10, duration=1)]
        done, info = run_simulation(tasks, "Priority")
        self.assertEqual(info["makespan"], 11)
        self.assertEqual(info["timeline"].count(None), 8)


if __name__ == "__main__":
    unittest.main()
