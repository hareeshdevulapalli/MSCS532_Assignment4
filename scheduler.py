"""
Simulates one CPU running tasks, using HeapPriorityQueue to choose
what runs next.

Time moves forward one tick at a time. Every tick:
  1. any task that has arrived gets inserted into the queue
  2. (aging only) every AGE_EVERY ticks, each waiting task gets +1 priority
  3. (preemptive only) if something in the queue beats the running task,
     the running task goes back in the queue
  4. if the CPU is free, pull the next task off the queue
  5. run that task for one tick

The policies all use the same queue class. They only differ in which
field is the key and whether it's a max-heap or min-heap:

  FCFS                  min-heap on arrival, no preemption (baseline)
  Priority              max-heap on priority, no preemption
  Priority (preemptive) max-heap on priority, can preempt
  Priority + aging      same as above, plus waiting tasks slowly gain
                        priority so low-priority ones don't starve
  EDF                   min-heap on deadline, can preempt

Usage:
    python3 scheduler.py           # example + full experiment + chart
    python3 scheduler.py --quick   # smaller version
"""

import argparse
import copy
import csv
import os
import random
import statistics

from priority_queue import HeapPriorityQueue, Task

AGE_EVERY = 5  # ticks a task has to wait to gain 1 priority
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

SCHEDULERS = {
    "FCFS": dict(kind="min", key="arrival", preempt=False, aging=False),
    "Priority": dict(kind="max", key="priority", preempt=False, aging=False),
    "Priority (preemptive)": dict(kind="max", key="priority", preempt=True, aging=False),
    "Priority + aging": dict(kind="max", key="priority", preempt=True, aging=True),
    "EDF": dict(kind="min", key="deadline", preempt=True, aging=False),
}


def should_preempt(queue, waiting, current):
    """True if the waiting task is strictly more urgent than the current one."""
    w = getattr(waiting, queue.key)
    c = getattr(current, queue.key)
    if queue.kind == "max":
        return w > c
    return w < c


def run_simulation(task_list, policy, age_every=AGE_EVERY):
    """Run one policy on a list of tasks.

    Works on copies, so the tasks passed in aren't changed.
    Returns (finished tasks, dict of extra info).
    """
    setup = SCHEDULERS[policy]
    upcoming = sorted(copy.deepcopy(task_list), key=lambda t: (t.arrival, t.task_id))
    queue = HeapPriorityQueue(kind=setup["kind"], key=setup["key"])

    total = len(upcoming)
    next_up = 0       # index of the next task that hasn't arrived yet
    clock = 0
    current = None    # task on the CPU right now
    done = []
    history = []      # which task ran on each tick (None = idle)
    preempt_count = 0
    queue_ops = 0

    while len(done) < total:
        # new arrivals
        while next_up < total and upcoming[next_up].arrival <= clock:
            queue.insert(upcoming[next_up])
            queue_ops += 1
            next_up += 1

        # aging: bump everyone who's waiting
        if setup["aging"] and clock > 0 and clock % age_every == 0:
            for t in queue.all_tasks():
                queue.increase_key(t, t.priority + 1)
                queue_ops += 1

        # kick the current task off if something better is waiting
        if (current is not None and setup["preempt"] and not queue.is_empty()
                and should_preempt(queue, queue.peek(), current)):
            queue.insert(current)
            queue_ops += 1
            current = None
            preempt_count += 1

        # CPU free? grab the next task
        if current is None and not queue.is_empty():
            current = queue.extract()
            queue_ops += 1
            if current.started_at is None:
                current.started_at = clock

        # nothing to run, skip ahead to the next arrival
        if current is None:
            idle = upcoming[next_up].arrival - clock
            history.extend([None] * idle)
            clock += idle
            continue

        current.time_left -= 1
        history.append(current.task_id)
        clock += 1
        if current.time_left == 0:
            current.finished_at = clock
            done.append(current)
            current = None

    info = {"preemptions": preempt_count, "makespan": clock,
            "pq_ops": queue_ops, "timeline": history}
    return done, info


def collect_stats(done, info):
    """Turn the finished tasks from one run into summary numbers."""
    def avg_wait_for(lo, hi):
        waits = [t.wait_time for t in done if lo <= t.original_priority <= hi]
        return statistics.mean(waits) if waits else 0.0

    waits = sorted(t.wait_time for t in done)
    has_deadline = [t for t in done if t.deadline is not None]
    late = sum(t.is_late for t in has_deadline)
    return {
        "avg_wait": statistics.mean(waits),
        "p95_wait": waits[int(0.95 * (len(waits) - 1))],
        "max_wait": waits[-1],
        "avg_turnaround": statistics.mean(t.turnaround_time for t in done),
        "avg_response": statistics.mean(t.response_time for t in done),
        "wait_low": avg_wait_for(1, 3),
        "wait_mid": avg_wait_for(4, 7),
        "wait_high": avg_wait_for(8, 10),
        "deadline_miss_pct": 100 * late / max(1, len(has_deadline)),
        "preemptions": info["preemptions"],
        "pq_ops": info["pq_ops"],
    }


def make_tasks(how_many, load, seed):
    """Random tasks for the experiment.

    Arrivals are Poisson, durations are 1-10 ticks, priorities are 1-10.
    `load` is roughly how busy the CPU is (average duration divided by
    average gap between arrivals). Each deadline gives the task between 2x
    and 6x its duration of slack.
    """
    rand = random.Random(seed)
    avg_duration = 5.5
    arrival_rate = load / avg_duration
    clock = 0.0
    tasks = []
    for i in range(how_many):
        clock += rand.expovariate(arrival_rate)
        arrives = int(clock)
        dur = rand.randint(1, 10)
        tasks.append(Task(task_id=i, priority=rand.randint(1, 10), arrival=arrives,
                          duration=dur,
                          deadline=arrives + round(dur * rand.uniform(2, 6))))
    return tasks


def small_example():
    """Six tasks by hand, printed as a timeline so you can see each policy."""
    tasks = [
        Task(1, priority=2, arrival=0, duration=6, deadline=20, label="A"),
        Task(2, priority=5, arrival=1, duration=3, deadline=8, label="B"),
        Task(3, priority=9, arrival=2, duration=2, deadline=6, label="C"),
        Task(4, priority=5, arrival=3, duration=4, deadline=18, label="D"),
        Task(5, priority=1, arrival=4, duration=2, deadline=9, label="E"),
        Task(6, priority=8, arrival=6, duration=3, deadline=12, label="F"),
    ]
    names = {t.task_id: t.label for t in tasks}

    print("Demo workload")
    print("  task  prio  arrival  burst  deadline")
    for t in tasks:
        print(f"  {t.label:>4}  {t.priority:>4}  {t.arrival:>7}  {t.duration:>5}  {t.deadline:>8}")
    print("\n  Timeline (one letter per tick)")
    print("  " + " " * 23 + "".join(str(i % 10) for i in range(20)))

    for policy in SCHEDULERS:
        done, info = run_simulation(tasks, policy, age_every=3)
        line = "".join(names[x] if x else "." for x in info["timeline"])
        stats = collect_stats(done, info)
        late = [t.label for t in sorted(done, key=lambda t: t.task_id) if t.is_late]
        print(f"  {policy:<22} {line:<20}  avg wait {stats['avg_wait']:4.1f}  "
              f"missed: {','.join(late) or '-'}")
    print("  (aging interval 3 ticks in the demo)\n")


COLUMNS = ["avg_wait", "p95_wait", "max_wait", "avg_turnaround", "avg_response",
           "wait_low", "wait_mid", "wait_high", "deadline_miss_pct", "preemptions", "pq_ops"]


def run_experiment(how_many, loads, seeds):
    rows = []
    for load in loads:
        for policy in SCHEDULERS:
            runs = []
            for seed in seeds:
                tasks = make_tasks(how_many, load, seed)
                runs.append(collect_stats(*run_simulation(tasks, policy)))
            row = {"load": load, "policy": policy}
            for col in COLUMNS:
                row[col] = statistics.mean(r[col] for r in runs)
            rows.append(row)
    return rows


def print_results(rows, load):
    print(f"Load {load:.2f}: mean over seeds")
    print(f"  {'policy':<22}{'avg wait':>9}{'p95':>7}{'max':>7}{'turn':>7}"
          f"{'low':>8}{'mid':>7}{'high':>7}{'miss %':>8}{'preempt':>9}")
    for r in rows:
        if r["load"] != load:
            continue
        print(f"  {r['policy']:<22}{r['avg_wait']:>9.1f}{r['p95_wait']:>7.0f}{r['max_wait']:>7.0f}"
              f"{r['avg_turnaround']:>7.1f}{r['wait_low']:>8.1f}{r['wait_mid']:>7.1f}"
              f"{r['wait_high']:>7.1f}{r['deadline_miss_pct']:>8.1f}{r['preemptions']:>9.0f}")
    print("  (low/mid/high = average wait for base priority 1-3 / 4-7 / 8-10)\n")


def draw_chart(rows, loads, chart_load, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import chart_style
    chart_style.setup(plt)

    names = list(SCHEDULERS)
    color_of = {name: chart_style.COLORS[i] for i, name in enumerate(names)}
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.6))

    # left: average wait split by priority group
    groups = [("wait_low", "Low (1-3)"), ("wait_mid", "Mid (4-7)"), ("wait_high", "High (8-10)")]
    bar_w = 0.16
    for i, name in enumerate(names):
        row = next(r for r in rows if r["load"] == chart_load and r["policy"] == name)
        xs = [g + (i - 2) * (bar_w + 0.01) for g in range(len(groups))]
        left.bar(xs, [row[col] for col, _ in groups], bar_w, color=color_of[name], label=name)
    left.set_xticks(range(len(groups)))
    left.set_xticklabels([label for _, label in groups])
    left.set_xlabel("Task base priority")
    left.set_ylabel("Average waiting time (ticks)")
    left.set_title(f"Waiting time by priority, load {chart_load}")
    left.grid(axis="x", visible=False)
    left.legend(loc="upper right")

    # right: % of deadlines missed as load goes up
    for i, name in enumerate(names):
        ys = [next(r for r in rows if r["load"] == L and r["policy"] == name)["deadline_miss_pct"]
              for L in loads]
        right.plot(loads, ys, color=color_of[name], marker=chart_style.MARKERS[i], label=name)
    right.set_xlabel("Offered load (CPU utilisation)")
    right.set_ylabel("Tasks missing their deadline (%)")
    right.set_title("Deadline misses vs load")
    right.set_xticks(loads)
    right.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="CPU scheduler simulation")
    parser.add_argument("--quick", action="store_true", help="smaller, faster run")
    args = parser.parse_args()

    small_example()

    if args.quick:
        how_many, seeds = 300, range(2)
    else:
        how_many, seeds = 2000, range(5)
    loads = [0.5, 0.7, 0.85, 0.95]
    chart_load = 0.85

    print(f"Experiment: {how_many} tasks per run, {len(seeds)} seeds, loads {loads}\n")
    rows = run_experiment(how_many, loads, seeds)
    for load in loads:
        print_results(rows, load)

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_file = os.path.join(OUT_DIR, "scheduler_results.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["load", "policy"] + COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})

    png_file = os.path.join(OUT_DIR, "scheduler.png")
    try:
        draw_chart(rows, loads, chart_load, png_file)
        print(f"Wrote {csv_file}\nWrote {png_file}")
    except ImportError:
        print(f"Wrote {csv_file} (install matplotlib if you want the chart)")


if __name__ == "__main__":
    main()
