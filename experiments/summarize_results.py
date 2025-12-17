from __future__ import annotations
import os
import csv
import argparse
import glob
import statistics
from collections import defaultdict


def parse_meta(meta_path: str) -> dict:
    d = {}
    with open(meta_path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        for row in r:
            if not row:
                continue
            if len(row) < 2:
                continue
            k, v = row[0], row[1]
            d[k] = v
    # 类型转换
    if "n_intervals" in d:
        d["n_intervals"] = int(d["n_intervals"])  # type: ignore
    if "coil_amp" in d:
        try:
            d["coil_amp"] = float(d["coil_amp"])  # type: ignore
        except Exception:
            pass
    return d


def parse_summary(summary_path: str) -> dict:
    out = {}
    with open(summary_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            solver = row["solver"]
            success = row["success"].strip().lower() in ("true", "1", "yes")
            time_sec = float(row["time_sec"]) if row["time_sec"] else float("nan")
            iters = int(row["iters"]) if row["iters"] else 0
            out[solver] = {"success": success, "time_sec": time_sec, "iters": iters}
    return out


def aggregate(root: str, out_path: str):
    # 兼容两种目录：experiments/results 与 experiments/experiments/results
    candidates = []
    for base in (root, os.path.join(root, "experiments", "results")):
        if os.path.isdir(base):
            candidates.append(base)
    if not candidates:
        raise FileNotFoundError(f"No such directory: {root}")

    summary_files = []
    for base in candidates:
        summary_files.extend(glob.glob(os.path.join(base, "*_summary.csv")))

    if not summary_files:
        print("No summary files found.")
        return

    # 分组键：scenario, scale, n_intervals, coil_amp
    groups = defaultdict(lambda: {"baseline": [], "nondim": []})

    for sfile in summary_files:
        tag = os.path.basename(sfile).replace("_summary.csv", "")
        meta_path = os.path.join(os.path.dirname(sfile), f"{tag}_meta.csv")
        if not os.path.isfile(meta_path):
            # 跳过无 meta 的样本
            continue

        meta = parse_meta(meta_path)
        summary = parse_summary(sfile)

        key = (
            meta.get("scenario", ""),
            meta.get("scale", ""),
            meta.get("n_intervals", None),
            meta.get("coil_amp", None),
        )

        for solver in ("baseline", "nondim"):
            if solver in summary:
                groups[key][solver].append(summary[solver])

    # 写聚合结果
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "scenario", "scale", "n_intervals", "coil_amp", "solver",
            "n_trials", "success_rate", 
            "iters_median", "iters_q1", "iters_q3",
            "time_median", "time_q1", "time_q3",
        ])

        for key, data in sorted(groups.items()):
            scenario, scale, n_intervals, coil_amp = key
            for solver in ("baseline", "nondim"):
                lst = data[solver]
                if not lst:
                    continue
                n = len(lst)
                success_rate = sum(1 for x in lst if x["success"]) / n
                iters_vals = [x["iters"] for x in lst if isinstance(x["iters"], int)]
                time_vals = [x["time_sec"] for x in lst if isinstance(x["time_sec"], (int, float))]
                try:
                    it_median = statistics.median(iters_vals) if iters_vals else float("nan")
                    it_q1 = statistics.quantiles(iters_vals, n=4)[0] if len(iters_vals) >= 4 else float("nan")
                    it_q3 = statistics.quantiles(iters_vals, n=4)[-1] if len(iters_vals) >= 4 else float("nan")
                except Exception:
                    it_median = float("nan"); it_q1 = float("nan"); it_q3 = float("nan")
                try:
                    t_median = statistics.median(time_vals) if time_vals else float("nan")
                    t_q1 = statistics.quantiles(time_vals, n=4)[0] if len(time_vals) >= 4 else float("nan")
                    t_q3 = statistics.quantiles(time_vals, n=4)[-1] if len(time_vals) >= 4 else float("nan")
                except Exception:
                    t_median = float("nan"); t_q1 = float("nan"); t_q3 = float("nan")

                w.writerow([
                    scenario, scale, n_intervals, coil_amp, solver,
                    n, f"{success_rate:.3f}",
                    it_median, it_q1, it_q3,
                    t_median, t_q1, t_q3,
                ])

    print(f"Aggregated results saved to: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="experiments/results", help="results root")
    ap.add_argument("--out", type=str, default="experiments/summary_aggregate.csv", help="aggregated output")
    args = ap.parse_args()

    aggregate(args.root, args.out)
