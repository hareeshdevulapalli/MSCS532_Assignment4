"""Single-CPU task scheduler simulation built on the heap priority queue.

Time advances in integer ticks. At each tick the scheduler

  1. inserts every task whose arrival_time has been reached   (insert)
  2. ages waiting tasks, if the policy uses aging            (increase_key)
  3. preempts the running task if a strictly better one is
     waiting, if the policy is preemptive                    (peek, insert)
  4. if the CPU is free, dispatches the top task             (extract_max/min)
  5. runs the current task for one tick

Policies compared (all use the same PriorityQueue class, only the key and
heap order change):

  FCFS                   min-heap on arrival_time, non-preemptive (baseline)
  Priority               max-heap on priority, non-preemptive
  Priority (preemptive)  max-heap on priority, preempts on a higher arrival
  Priority + aging       as above, but every AGING_INTERVAL ticks each waiting
                         task's priority is raised by 1 with increase_key, so
                         low-priority tasks cannot starve
  EDF                    min-heap on deadline, preemptive (earliest deadline
                         first)

Run:
    python3 scheduler.py            # demo trace + full experiment + graphs
    python3 scheduler.py --quick    # fewer, smaller runs
"""

import argparse
import copy
import csv
import os
import random
import statistics

from priority_queue import PriorityQueue, Task

AGING_INTERVAL = 5          # ticks of waiting per +1 priority
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

POLICIES = {
    "FCFS": dict(order="min", key="arrival_time", preemptive=False, aging=False),
    "Priority": dict(order="max", key="priority", preemptive=False, aging=False),
    "Priority (preemptive)": dict(order="max", key="priority", preemptive=True, aging=False),
    "Priority + aging": dict(order="max", key="priority", preemptive=True, aging=True),
    "EDF": dict(order="min", key="deadline", preemptive=True, aging=False),
}


def _beats(pq, waiting, running):
    """True if `waiting` should preempt `running` under this queue's order."""
    a, b = getattr(waiting, pq.key), getattr(running, pq.key)
    return a > b if pq.order == "max" else a < b


def simulate(tasks, policy, aging_interval=AGING_INTERVAL):
    """Run one policy over a workload. Returns (finished tasks, stats dict).

    `tasks` is not modified: the simulation works on deep copies.
    """
    cfg = POLICIES[policy]
    pending = sorted(copy.deepcopy(tasks), key=lambda t: (t.arrival_time, t.task_id))
    pq = PriorityQueue(order=cfg["order"], key=cfg["key"])
    n = len(pending)
    nxt = 0                 # index of the next task yet to arrive
    time = 0
    running = None
    finished = []
    timeline = []           # task_id (or None for idle) per tick
    preemptions = 0
    pq_ops = 0

    while len(finished) < n:
        # 1. admit arrivals
        while nxt < n and pending[nxt].arrival_time <= time:
            pq.insert(pending[nxt])
            pq_ops += 1
            nxt += 1

        # 2. aging: every waiting task gains +1 priority per interval
        if cfg["aging"] and time > 0 and time % aging_interval == 0:
            for task in pq.tasks():
                pq.increase_key(task, task.priority + 1)
                pq_ops += 1

        # 3. preemption
        if (running is not None and cfg["preemptive"] and not pq.is_empty()
                and _beats(pq, pq.peek(), running)):
            pq.insert(running)
            pq_ops += 1
            running = None
            preemptions += 1

        # 4. dispatch
        if running is None and not pq.is_empty():
            running = pq.extract()
            pq_ops += 1
            if running.start_time is None:
                running.start_time = time

        # 5. execute one tick, or jump over idle time
        if running is None:
            gap = pending[nxt].arrival_time - time
            timeline.extend([None] * gap)
            time += gap
            continue
        running.remaining -= 1
        timeline.append(running.task_id)
        time += 1
        if running.remaining == 0:
            running.finish_time = time
            finished.append(running)
            running = None

    return finished, {"preemptions": preemptions, "makespan": time,
                      "pq_ops": pq_ops, "timeline": timeline}


def summarize(finished, extra):
    """Aggregate per-task metrics for one simulation run."""
    def band(lo, hi):
        waits = [t.waiting for t in finished if lo <= t.base_priority <= hi]
        return statistics.mean(waits) if waits else 0.0

    waits = sorted(t.waiting for t in finished)
    with_deadline = [t for t in finished if t.deadline is not None]
    return {
        "avg_wait": statistics.mean(waits),
        "p95_wait": waits[int(0.95 * (len(waits) - 1))],
        "max_wait": waits[-1],
        "avg_turnaround": statistics.mean(t.turnaround for t in finished),
        "avg_response": statistics.mean(t.response for t in finished),
        "wait_low": band(1, 3),
        "wait_mid": band(4, 7),
        "wait_high": band(8, 10),
        "deadline_miss_pct": 100 * sum(t.missed_deadline for t in with_deadline)
                             / max(1, len(with_deadline)),
        "preemptions": extra["preemptions"],
        "pq_ops": extra["pq_ops"],
    }


def generate_workload(n, load, seed):
    """Random tasks: Poisson arrivals, burst 1-10, priority 1-10.

    `load` is the offered CPU utilisation (mean burst / mean inter-arrival).
    Each deadline gives the task 2x to 6x its burst time of slack.
    """
    rng = random.Random(seed)
    mean_burst = 5.5
    rate = load / mean_burst
    t = 0.0
    tasks = []
    for i in range(n):
        t += rng.expovariate(rate)
        arrival = int(t)
        burst = rng.randint(1, 10)
        tasks.append(Task(task_id=i, priority=rng.randint(1, 10), arrival_time=arrival,
                          burst_time=burst,
                          deadline=arrival + round(burst * rng.uniform(2, 6))))
    return tasks


def demo():
    """Small hand-made workload, printed as a per-tick timeline."""
    tasks = [
        Task(1, priority=2, arrival_time=0, burst_time=6, deadline=20, name="A"),
        Task(2, priority=5, arrival_time=1, burst_time=3, deadline=8, name="B"),
        Task(3, priority=9, arrival_time=2, burst_time=2, deadline=6, name="C"),
        Task(4, priority=5, arrival_time=3, burst_time=4, deadline=18, name="D"),
        Task(5, priority=1, arrival_time=4, burst_time=2, deadline=9, name="E"),
        Task(6, priority=8, arrival_time=6, burst_time=3, deadline=12, name="F"),
    ]
    names = {t.task_id: t.name for t in tasks}
    print("Demo workload")
    print("  task  prio  arrival  burst  deadline")
    for t in tasks:
        print(f"  {t.name:>4}  {t.priority:>4}  {t.arrival_time:>7}  {t.burst_time:>5}  {t.deadline:>8}")
    print("\n  Timeline (one letter per tick)")
    print("  " + " " * 23 + "".join(str(i % 10) for i in range(20)))
    for policy in POLICIES:
        finished, extra = simulate(tasks, policy, aging_interval=3)
        line = "".join(names[x] if x else "." for x in extra["timeline"])
        s = summarize(finished, extra)
        missed = [t.name for t in sorted(finished, key=lambda t: t.task_id) if t.missed_deadline]
        print(f"  {policy:<22} {line:<20}  avg wait {s['avg_wait']:4.1f}  "
              f"missed: {','.join(missed) or '-'}")
    print("  (aging interval 3 ticks in the demo)\n")


METRICS = ["avg_wait", "p95_wait", "max_wait", "avg_turnaround", "avg_response",
           "wait_low", "wait_mid", "wait_high", "deadline_miss_pct", "preemptions", "pq_ops"]


def experiment(n, loads, seeds):
    rows = []
    for load in loads:
        for policy in POLICIES:
            runs = [summarize(*simulate(generate_workload(n, load, seed), policy))
                    for seed in seeds]
            row = {"load": load, "policy": policy}
            row.update({m: statistics.mean(r[m] for r in runs) for m in METRICS})
            rows.append(row)
    return rows


def print_table(rows, load):
    print(f"Load {load:.2f}: mean over seeds")
    hdr = (f"  {'policy':<22}{'avg wait':>9}{'p95':>7}{'max':>7}{'turn':>7}"
           f"{'low':>8}{'mid':>7}{'high':>7}{'miss %':>8}{'preempt':>9}")
    print(hdr)
    for r in rows:
        if r["load"] != load:
            continue
        print(f"  {r['policy']:<22}{r['avg_wait']:>9.1f}{r['p95_wait']:>7.0f}{r['max_wait']:>7.0f}"
              f"{r['avg_turnaround']:>7.1f}{r['wait_low']:>8.1f}{r['wait_mid']:>7.1f}"
              f"{r['wait_high']:>7.1f}{r['deadline_miss_pct']:>8.1f}{r['preemptions']:>9.0f}")
    print("  (low/mid/high = average wait for base priority 1-3 / 4-7 / 8-10)\n")


def plot(rows, loads, focus_load, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style
    plot_style.apply(plt)

    policies = list(POLICIES)
    colors = {p: plot_style.SERIES[i] for i, p in enumerate(policies)}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # (a) waiting time by priority band at the focus load
    bands = [("wait_low", "Low (1-3)"), ("wait_mid", "Mid (4-7)"), ("wait_high", "High (8-10)")]
    width = 0.16
    for i, p in enumerate(policies):
        r = next(r for r in rows if r["load"] == focus_load and r["policy"] == p)
        xs = [b + (i - 2) * (width + 0.01) for b in range(len(bands))]
        ax1.bar(xs, [r[k] for k, _ in bands], width, color=colors[p], label=p)
    ax1.set_xticks(range(len(bands)))
    ax1.set_xticklabels([lbl for _, lbl in bands])
    ax1.set_xlabel("Task base priority")
    ax1.set_ylabel("Average waiting time (ticks)")
    ax1.set_title(f"Waiting time by priority, load {focus_load}")
    ax1.grid(axis="x", visible=False)
    ax1.legend(loc="upper right")

    # (b) deadline misses vs load
    for i, p in enumerate(policies):
        ys = [next(r for r in rows if r["load"] == L and r["policy"] == p)["deadline_miss_pct"]
              for L in loads]
        ax2.plot(loads, ys, color=colors[p], marker=plot_style.MARKERS[i], label=p)
    ax2.set_xlabel("Offered load (CPU utilisation)")
    ax2.set_ylabel("Tasks missing their deadline (%)")
    ax2.set_title("Deadline misses vs load")
    ax2.set_xticks(loads)
    ax2.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="smaller, faster run")
    args = parser.parse_args()

    demo()
    n, seeds = (300, range(2)) if args.quick else (2000, range(5))
    loads = [0.5, 0.7, 0.85, 0.95]
    focus = 0.85
    print(f"Experiment: {n} tasks per run, {len(seeds)} seeds, loads {loads}\n")
    rows = experiment(n, loads, seeds)
    for load in loads:
        print_table(rows, load)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "scheduler_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["load", "policy"] + METRICS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    png_path = os.path.join(RESULTS_DIR, "scheduler.png")
    try:
        plot(rows, loads, focus, png_path)
        print(f"Wrote {csv_path}\nWrote {png_path}")
    except ImportError:
        print(f"Wrote {csv_path} (install matplotlib for the graph)")


if __name__ == "__main__":
    main()
