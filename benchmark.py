"""Empirical comparison of Heapsort, Quicksort and Merge Sort.

Part 1 - running time. Each algorithm sorts sorted, reverse-sorted, random
and few-unique (many duplicates) lists of several sizes. Time is the median
of REPEATS runs measured with time.perf_counter on a fresh copy of the
input. Python's built-in sorted() (Timsort, written in C) is timed as a
reference point only.

Part 2 - comparison counts. Wall-clock time in Python is noisy and mixes in
interpreter overhead, so each algorithm is also run on values wrapped in a
class whose comparison operators increment a counter. Dividing by n lg n
shows how close each one is to its theoretical constant, and whether the
count depends on the input order.

Part 3 - priority queue operations. Average time per insert (random and
worst-case keys), extract_max and increase_key for queues of size
2^10 .. 2^18, to check the O(lg n) bound empirically.

    python3 benchmark.py            # full run, about 1-2 minutes
    python3 benchmark.py --quick    # small sizes only
"""

import argparse
import csv
import gc
import math
import os
import random
import statistics
import sys
import time

from heapsort import heapsort
from other_sorts import merge_sort, quicksort
from priority_queue import PriorityQueue, Task

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

ALGORITHMS = {
    "Heapsort": heapsort,
    "Quicksort": quicksort,
    "Merge Sort": merge_sort,
    "sorted() (Timsort, C)": sorted,
}
DISTRIBUTIONS = ["sorted", "reverse", "random", "few-unique"]


def make_data(kind, n, rng):
    if kind == "sorted":
        return list(range(n))
    if kind == "reverse":
        return list(range(n, 0, -1))
    if kind == "random":
        return [rng.randint(0, 10 * n) for _ in range(n)]
    if kind == "few-unique":
        return [rng.randint(0, 9) for _ in range(n)]
    raise ValueError(kind)


def time_sort(fn, data, repeats):
    samples = []
    for _ in range(repeats):
        copy = data[:]
        t0 = time.perf_counter()
        result = fn(copy)
        samples.append(time.perf_counter() - t0)
    assert result == sorted(data), f"{fn.__name__} produced a wrong result"
    return statistics.median(samples)


# --------------------------------------------------------- comparisons
class Counted:
    """Wraps a value; every comparison increments Counted.count."""
    __slots__ = ("v",)
    count = 0

    def __init__(self, v):
        self.v = v

    def __lt__(self, other):
        Counted.count += 1
        return self.v < other.v

    def __eq__(self, other):
        Counted.count += 1
        return self.v == other.v

    def __le__(self, other):
        Counted.count += 1
        return self.v <= other.v


def count_comparisons(fn, data):
    wrapped = [Counted(v) for v in data]
    Counted.count = 0
    fn(wrapped)
    return Counted.count


# ------------------------------------------------------ priority queue
def time_pq_ops(n, rng, batch=2000):
    """Mean microseconds per op on a queue holding about n tasks.

    insert (random)  keys drawn from the same range as the queue
    insert (worst)   each key larger than everything queued, so it sifts up
                     all floor(lg n) levels to the root
    extract_max      root removal + full sift-down
    increase_key     random task raised by a random amount
    The garbage collector is paused while timing, as timeit does.
    """
    tasks = [Task(i, rng.randint(0, 10 * n)) for i in range(n)]
    pq = PriorityQueue(tasks=tasks)               # O(n) bulk build
    rand_new = [Task(n + i, rng.randint(0, 10 * n)) for i in range(batch)]
    worst_new = [Task(2 * n + batch + i, 10 * n + 1 + i) for i in range(batch)]
    targets = rng.sample(tasks, min(batch, n))
    bumps = [rng.randint(1, 10 * n) for _ in targets]

    def timed(fn, count):
        gc.disable()
        try:
            t0 = time.perf_counter()
            fn()
            return (time.perf_counter() - t0) / count * 1e6
        finally:
            gc.enable()

    def insert_all(items):
        for t in items:
            pq.insert(t)

    def extract_all(count):
        for _ in range(count):
            pq.extract_max()

    def increase_all():
        for t, bump in zip(targets, bumps):
            pq.increase_key(t, t.priority + bump)

    t_ins = timed(lambda: insert_all(rand_new), batch)
    for t in rand_new:
        pq.remove(t)
    t_worst = timed(lambda: insert_all(worst_new), batch)
    for t in worst_new:
        pq.remove(t)
    t_inc = timed(increase_all, len(targets))
    k = min(batch, n)
    t_ext = timed(lambda: extract_all(k), k)
    return t_ins, t_worst, t_ext, t_inc


# --------------------------------------------------------------- plots
def plot_times(rows, sizes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style
    plot_style.apply(plt)

    fig, axes = plt.subplots(1, len(DISTRIBUTIONS), figsize=(14, 3.8), sharey=True)
    for ax, dist in zip(axes, DISTRIBUTIONS):
        for i, alg in enumerate(ALGORITHMS):
            ys = [next(r["seconds"] for r in rows
                       if r["algorithm"] == alg and r["distribution"] == dist and r["n"] == n) * 1000
                  for n in sizes]
            ax.plot(sizes, ys, color=plot_style.SERIES[i], marker=plot_style.MARKERS[i], label=alg)
        ax.set_title(dist)
        ax.set_xlabel("n")
        ax.set_xscale("log")
        ax.set_yscale("log")
    axes[0].set_ylabel("Median time (ms, log scale)")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pq(pq_rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style
    plot_style.apply(plt)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    lgs = [math.log2(r["n"]) for r in pq_rows]
    for i, (key, label) in enumerate([("insert_us", "insert (random key)"),
                                      ("insert_worst_us", "insert (new maximum)"),
                                      ("extract_us", "extract_max"),
                                      ("increase_us", "increase_key")]):
        ax.plot(lgs, [r[key] for r in pq_rows], color=plot_style.SERIES[i],
                marker=plot_style.MARKERS[i], label=label)
    ax.set_xlabel("lg n (queue size n = 2^x)")
    ax.set_ylabel("Mean time per operation (µs)")
    ax.set_title("Priority queue operation cost vs lg n")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Heapsort vs Quicksort vs Merge Sort")
    parser.add_argument("--quick", action="store_true", help="small sizes only")
    args = parser.parse_args()
    rng = random.Random(42)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.quick:
        sizes, repeats, pq_sizes = [500, 1000, 2000, 4000], 3, [2 ** k for k in range(10, 15)]
    else:
        sizes, repeats, pq_sizes = ([1000, 2000, 5000, 10000, 20000, 50000, 100000], 5,
                                    [2 ** k for k in range(10, 19)])
    print(f"Python {sys.version.split()[0]}, median of {repeats} runs\n")

    # Part 1: time
    rows = []
    print("Part 1: running time (ms)")
    print(f"  {'n':>7}  {'distribution':<11}" + "".join(f"{a:>23}" for a in ALGORITHMS))
    for n in sizes:
        for dist in DISTRIBUTIONS:
            data = make_data(dist, n, rng)
            line = f"  {n:>7}  {dist:<11}"
            for alg, fn in ALGORITHMS.items():
                sec = time_sort(fn, data, repeats)
                rows.append({"algorithm": alg, "distribution": dist, "n": n, "seconds": sec})
                line += f"{sec * 1000:>23.2f}"
            print(line)
    with open(os.path.join(RESULTS_DIR, "sort_times.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["algorithm", "distribution", "n", "seconds"])
        w.writeheader()
        w.writerows(rows)

    # Part 2: comparisons
    cmp_n = 2000 if args.quick else 20000
    print(f"\nPart 2: key comparisons at n = {cmp_n}  (count / (n lg n) in brackets)")
    cmp_rows = []
    nlgn = cmp_n * math.log2(cmp_n)
    print(f"  {'algorithm':<12}" + "".join(f"{d:>22}" for d in DISTRIBUTIONS))
    for alg in ["Heapsort", "Quicksort", "Merge Sort"]:
        line = f"  {alg:<12}"
        for dist in DISTRIBUTIONS:
            c = count_comparisons(ALGORITHMS[alg], make_data(dist, cmp_n, random.Random(7)))
            cmp_rows.append({"algorithm": alg, "distribution": dist, "n": cmp_n,
                             "comparisons": c, "ratio_to_nlgn": round(c / nlgn, 3)})
            line += f"{c:>13,} [{c / nlgn:4.2f}]"
        print(line)
    with open(os.path.join(RESULTS_DIR, "comparisons.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cmp_rows[0]))
        w.writeheader()
        w.writerows(cmp_rows)

    # Part 3: priority queue
    print("\nPart 3: priority queue, mean µs per operation")
    print(f"  {'n':>8} {'lg n':>5} {'insert':>9} {'insert(max)':>12} {'extract_max':>12} {'increase_key':>13}")
    pq_rows = []
    for n in pq_sizes:
        ins, worst, ext, inc = time_pq_ops(n, rng)
        pq_rows.append({"n": n, "insert_us": round(ins, 3), "insert_worst_us": round(worst, 3),
                        "extract_us": round(ext, 3), "increase_us": round(inc, 3)})
        print(f"  {n:>8} {int(math.log2(n)):>5} {ins:>9.2f} {worst:>12.2f} {ext:>12.2f} {inc:>13.2f}")
    with open(os.path.join(RESULTS_DIR, "pq_ops.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pq_rows[0]))
        w.writeheader()
        w.writerows(pq_rows)

    try:
        plot_times(rows, sizes, os.path.join(RESULTS_DIR, "sort_times.png"))
        plot_pq(pq_rows, os.path.join(RESULTS_DIR, "pq_ops.png"))
        print(f"\nCSV files and graphs written to {RESULTS_DIR}/")
    except ImportError:
        print(f"\nCSV files written to {RESULTS_DIR}/ (install matplotlib for graphs)")


if __name__ == "__main__":
    main()
