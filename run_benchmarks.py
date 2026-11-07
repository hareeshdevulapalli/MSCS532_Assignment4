"""
Compares heap sort with quick sort and merge sort, and times the
priority queue operations.

Part 1: timing. Each sort runs on sorted, reversed, random and
"few-unique" (lots of duplicates) lists of different sizes. I take the
median of several runs, each on a fresh copy. Python's sorted() is in
there too, just as a reference point since it's written in C.

Part 2: counting comparisons. Timing in Python is noisy and a lot of it
is interpreter overhead, so I also wrap every value in a class that
counts each comparison. Dividing by n*log2(n) shows the constant factor
and whether the input order makes a difference.

Part 3: priority queue. Average time per insert (random keys and the
worst case), extract_max and increase_key on queues from 2^10 up to 2^18
tasks, to see if it really grows like log n.

    python3 run_benchmarks.py           # full run, takes a minute or two
    python3 run_benchmarks.py --quick   # small sizes only
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

from heapsort import heap_sort
from priority_queue import HeapPriorityQueue, Task
from sorts import merge_sort, quick_sort

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

SORTERS = {
    "Heapsort": heap_sort,
    "Quicksort": quick_sort,
    "Merge Sort": merge_sort,
    "sorted() (Timsort, C)": sorted,
}
INPUT_TYPES = ["sorted", "reverse", "random", "few-unique"]


def make_input(kind, n, rand):
    if kind == "sorted":
        return list(range(n))
    if kind == "reverse":
        return list(range(n, 0, -1))
    if kind == "random":
        return [rand.randint(0, 10 * n) for _ in range(n)]
    if kind == "few-unique":
        return [rand.randint(0, 9) for _ in range(n)]
    raise ValueError(f"unknown input type: {kind}")


def median_time(sort_fn, data, runs):
    times = []
    for _ in range(runs):
        fresh = data[:]
        start = time.perf_counter()
        out = sort_fn(fresh)
        times.append(time.perf_counter() - start)
    # make sure the sort actually worked
    assert out == sorted(data), f"{sort_fn.__name__} gave the wrong answer"
    return statistics.median(times)


class CountingValue:
    """Wraps a number and counts every time it gets compared."""
    __slots__ = ("val",)
    compares = 0

    def __init__(self, val):
        self.val = val

    def __lt__(self, other):
        CountingValue.compares += 1
        return self.val < other.val

    def __le__(self, other):
        CountingValue.compares += 1
        return self.val <= other.val

    def __eq__(self, other):
        CountingValue.compares += 1
        return self.val == other.val


def count_compares(sort_fn, data):
    wrapped = [CountingValue(v) for v in data]
    CountingValue.compares = 0
    sort_fn(wrapped)
    return CountingValue.compares


def time_queue_ops(n, rand, batch=2000):
    """Average microseconds per operation on a queue with about n tasks.

    "insert (worst)" uses keys bigger than everything already in the
    queue, so each one has to bubble all the way up to the root.
    Garbage collection is switched off while timing so it doesn't add
    random spikes.
    """
    tasks = [Task(i, rand.randint(0, 10 * n)) for i in range(n)]
    pq = HeapPriorityQueue(tasks=tasks)
    random_new = [Task(n + i, rand.randint(0, 10 * n)) for i in range(batch)]
    biggest_new = [Task(2 * n + batch + i, 10 * n + 1 + i) for i in range(batch)]
    to_bump = rand.sample(tasks, min(batch, n))
    bump_by = [rand.randint(1, 10 * n) for _ in to_bump]

    def timed(work, how_many):
        gc.disable()
        try:
            start = time.perf_counter()
            work()
            return (time.perf_counter() - start) / how_many * 1e6
        finally:
            gc.enable()

    def insert_each(group):
        for t in group:
            pq.insert(t)

    def bump_each():
        for t, extra in zip(to_bump, bump_by):
            pq.increase_key(t, t.priority + extra)

    def pop_many(k):
        for _ in range(k):
            pq.extract_max()

    insert_avg = timed(lambda: insert_each(random_new), batch)
    for t in random_new:
        pq.remove(t)
    insert_worst = timed(lambda: insert_each(biggest_new), batch)
    for t in biggest_new:
        pq.remove(t)
    bump_avg = timed(bump_each, len(to_bump))
    k = min(batch, n)
    extract_avg = timed(lambda: pop_many(k), k)
    return insert_avg, insert_worst, extract_avg, bump_avg


# ---------- charts ----------

def chart_sort_times(rows, sizes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import chart_style
    chart_style.setup(plt)

    fig, panels = plt.subplots(1, len(INPUT_TYPES), figsize=(14, 3.8), sharey=True)
    for ax, kind in zip(panels, INPUT_TYPES):
        for i, name in enumerate(SORTERS):
            ms = []
            for n in sizes:
                secs = next(r["seconds"] for r in rows
                            if r["algorithm"] == name and r["distribution"] == kind and r["n"] == n)
                ms.append(secs * 1000)
            ax.plot(sizes, ms, color=chart_style.COLORS[i], marker=chart_style.MARKERS[i], label=name)
        ax.set_title(kind)
        ax.set_xlabel("n")
        ax.set_xscale("log")
        ax.set_yscale("log")
    panels[0].set_ylabel("Median time (ms, log scale)")
    panels[0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_queue_ops(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import chart_style
    chart_style.setup(plt)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    log_n = [math.log2(r["n"]) for r in rows]
    lines = [("insert_us", "insert (random key)"),
             ("insert_worst_us", "insert (new maximum)"),
             ("extract_us", "extract_max"),
             ("increase_us", "increase_key")]
    for i, (col, label) in enumerate(lines):
        ax.plot(log_n, [r[col] for r in rows], color=chart_style.COLORS[i],
                marker=chart_style.MARKERS[i], label=label)
    ax.set_xlabel("lg n (queue size n = 2^x)")
    ax.set_ylabel("Mean time per operation (µs)")
    ax.set_title("Priority queue operation cost vs lg n")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Heapsort vs Quicksort vs Merge Sort")
    parser.add_argument("--quick", action="store_true", help="small sizes only")
    args = parser.parse_args()
    rand = random.Random(42)
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.quick:
        sizes = [500, 1000, 2000, 4000]
        runs = 3
        queue_sizes = [2 ** p for p in range(10, 15)]
    else:
        sizes = [1000, 2000, 5000, 10000, 20000, 50000, 100000]
        runs = 5
        queue_sizes = [2 ** p for p in range(10, 19)]
    print(f"Python {sys.version.split()[0]}, median of {runs} runs\n")

    # ---- part 1: timing ----
    time_rows = []
    print("Part 1: running time (ms)")
    print(f"  {'n':>7}  {'distribution':<11}" + "".join(f"{name:>23}" for name in SORTERS))
    for n in sizes:
        for kind in INPUT_TYPES:
            data = make_input(kind, n, rand)
            line = f"  {n:>7}  {kind:<11}"
            for name, fn in SORTERS.items():
                secs = median_time(fn, data, runs)
                time_rows.append({"algorithm": name, "distribution": kind, "n": n, "seconds": secs})
                line += f"{secs * 1000:>23.2f}"
            print(line)
    save_csv(os.path.join(OUT_DIR, "sort_times.csv"), time_rows)

    # ---- part 2: comparisons ----
    n_cmp = 2000 if args.quick else 20000
    n_log_n = n_cmp * math.log2(n_cmp)
    print(f"\nPart 2: key comparisons at n = {n_cmp}  (count / (n lg n) in brackets)")
    print(f"  {'algorithm':<12}" + "".join(f"{kind:>22}" for kind in INPUT_TYPES))
    cmp_rows = []
    for name in ["Heapsort", "Quicksort", "Merge Sort"]:
        line = f"  {name:<12}"
        for kind in INPUT_TYPES:
            c = count_compares(SORTERS[name], make_input(kind, n_cmp, random.Random(7)))
            cmp_rows.append({"algorithm": name, "distribution": kind, "n": n_cmp,
                             "comparisons": c, "ratio_to_nlgn": round(c / n_log_n, 3)})
            line += f"{c:>13,} [{c / n_log_n:4.2f}]"
        print(line)
    save_csv(os.path.join(OUT_DIR, "comparisons.csv"), cmp_rows)

    # ---- part 3: priority queue ----
    print("\nPart 3: priority queue, mean µs per operation")
    print(f"  {'n':>8} {'lg n':>5} {'insert':>9} {'insert(max)':>12} {'extract_max':>12} {'increase_key':>13}")
    queue_rows = []
    for n in queue_sizes:
        ins, worst, ext, inc = time_queue_ops(n, rand)
        queue_rows.append({"n": n, "insert_us": round(ins, 3), "insert_worst_us": round(worst, 3),
                           "extract_us": round(ext, 3), "increase_us": round(inc, 3)})
        print(f"  {n:>8} {int(math.log2(n)):>5} {ins:>9.2f} {worst:>12.2f} {ext:>12.2f} {inc:>13.2f}")
    save_csv(os.path.join(OUT_DIR, "pq_ops.csv"), queue_rows)

    try:
        chart_sort_times(time_rows, sizes, os.path.join(OUT_DIR, "sort_times.png"))
        chart_queue_ops(queue_rows, os.path.join(OUT_DIR, "pq_ops.png"))
        print(f"\nCSV files and graphs written to {OUT_DIR}/")
    except ImportError:
        print(f"\nCSV files written to {OUT_DIR}/ (install matplotlib for the graphs)")


if __name__ == "__main__":
    main()
