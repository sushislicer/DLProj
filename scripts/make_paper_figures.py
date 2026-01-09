"""Generate paper-ready figures from pipeline + benchmark outputs.

This script is intentionally lightweight and file-based:
- Reads `logs/**/memory_history.json` (from [`utils/memory_tracker.py`](utils/memory_tracker.py:1))
- Reads `outputs/**/spinquant_layer_summary.json` (from [`scripts/spinquant.py`](scripts/spinquant.py:1))
- Reads `benchmark_results/**/final_results.json` (from [`scripts/run_benchmarks.py`](scripts/run_benchmarks.py:1))

It produces a set of PNG figures suitable for a paper/slide deck.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def _maybe_import_plots():
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    return plt, sns


def _find_latest(pattern: str) -> Optional[str]:
    matches = sorted(glob(pattern), key=lambda p: os.path.getmtime(p))
    return matches[-1] if matches else None


def load_memory_history(path: str) -> pd.DataFrame:
    with open(path, "r") as f:
        data = json.load(f)
    rows = data.get("history", [])
    if not rows:
        return pd.DataFrame()
    df = pd.json_normalize(rows)
    # timestamp is ISO string; convert to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def plot_memory_over_time(df: pd.DataFrame, out_dir: str, title: str = "GPU memory over time") -> Optional[str]:
    if df.empty:
        return None
    plt, _ = _maybe_import_plots()

    # Use allocated MB on GPU0
    if "gpu.allocated_mb" not in df.columns:
        return None
    df = df.sort_values("timestamp")
    t0 = df["timestamp"].min()
    df["t_sec"] = (df["timestamp"] - t0).dt.total_seconds()

    fig_path = os.path.join(out_dir, "fig_memory_over_time.png")
    plt.figure(figsize=(10, 4))
    plt.plot(df["t_sec"], df["gpu.allocated_mb"], linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("GPU allocated (MB)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    return fig_path


def plot_memory_by_stage(df: pd.DataFrame, out_dir: str, title: str = "Peak GPU memory by stage") -> Optional[str]:
    if df.empty:
        return None
    plt, sns = _maybe_import_plots()

    if "stage" not in df.columns or "gpu.allocated_mb" not in df.columns:
        return None

    agg = (
        df.groupby("stage", as_index=False)["gpu.allocated_mb"].max()
        .sort_values("gpu.allocated_mb", ascending=False)
    )
    fig_path = os.path.join(out_dir, "fig_memory_by_stage.png")
    plt.figure(figsize=(9, 4))
    sns.barplot(data=agg, x="stage", y="gpu.allocated_mb")
    plt.ylabel("Peak GPU allocated (MB)")
    plt.xlabel("Stage")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    return fig_path


def load_spinquant_layer_summary(path: str) -> pd.DataFrame:
    with open(path, "r") as f:
        data = json.load(f)
    rows = []
    for name, info in data.items():
        row = {"layer": name}
        if isinstance(info, dict):
            row.update(info)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_spinquant_error_hist(df: pd.DataFrame, out_dir: str) -> Optional[str]:
    if df.empty or "rotated_quant_mse" not in df.columns:
        return None
    plt, sns = _maybe_import_plots()
    fig_path = os.path.join(out_dir, "fig_spinquant_rotated_mse_hist.png")
    plt.figure(figsize=(8, 4))
    sns.histplot(df["rotated_quant_mse"], bins=30, kde=True)
    plt.xlabel("MSE(Q(W·R), W·R)")
    plt.title("SpinQuant-lite rotated-space fake-quant error")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    return fig_path


def load_final_results(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def plot_benchmark_accuracy(results: Dict[str, Any], out_dir: str) -> Optional[str]:
    """Plot accuracy for each benchmark across model variants (main + baselines)."""
    plt, sns = _maybe_import_plots()

    bm = results.get("benchmarks", {})
    rows = []
    for model_key, model_data in bm.items():
        benches = model_data.get("benchmarks", {}) if isinstance(model_data, dict) else {}
        for bench_name, bench_data in benches.items():
            metrics = bench_data.get("metrics", {}) if isinstance(bench_data, dict) else {}
            acc = metrics.get("accuracy", None)
            if isinstance(acc, (int, float)):
                rows.append({"model": model_key, "benchmark": bench_name, "accuracy": float(acc)})

    if not rows:
        return None

    df = pd.DataFrame(rows)
    fig_path = os.path.join(out_dir, "fig_benchmark_accuracy.png")
    plt.figure(figsize=(12, 5))
    sns.barplot(data=df, x="benchmark", y="accuracy", hue="model")
    plt.ylabel("Accuracy")
    plt.title("Benchmark accuracy (main vs baselines)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    return fig_path


def plot_projection_schedule_from_log(log_path: str, out_dir: str) -> Optional[str]:
    """Parse `galore.log` for projection schedule updates and plot update_gap over step."""
    if not os.path.exists(log_path):
        return None

    pat = re.compile(r"\[Projection\] schedule applied at step=(\d+): rank=(\d+), update_gap=(\d+)")
    rows: List[Dict[str, int]] = []
    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if m:
                rows.append({"step": int(m.group(1)), "rank": int(m.group(2)), "update_gap": int(m.group(3))})
    if not rows:
        return None

    df = pd.DataFrame(rows).sort_values("step")
    plt, _ = _maybe_import_plots()
    fig_path = os.path.join(out_dir, "fig_projection_update_gap.png")
    plt.figure(figsize=(10, 4))
    plt.step(df["step"], df["update_gap"], where="post")
    plt.gca().invert_yaxis()  # smaller gap = more frequent updates
    plt.xlabel("Optimizer step")
    plt.ylabel("update_gap (smaller = more frequent)")
    plt.title("GaLore-like projection update schedule")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    return fig_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate paper-ready figures from outputs")
    ap.add_argument("--log_dir", type=str, default="logs", help="Log dir (contains memory_history.json, galore.log)")
    ap.add_argument("--output_dir", type=str, default="paper_figures", help="Where to write figures")
    ap.add_argument(
        "--benchmark_results",
        type=str,
        default="benchmark_results",
        help="Benchmark results root (contains benchmark_*/final_results.json)",
    )
    ap.add_argument(
        "--pipeline_outputs",
        type=str,
        default="outputs",
        help="Pipeline outputs root (contains quantized_model/spinquant_layer_summary.json)",
    )
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Memory plots
    mem_path = _find_latest(os.path.join(args.log_dir, "**", "memory_history.json"))
    if mem_path:
        df_mem = load_memory_history(mem_path)
        plot_memory_over_time(df_mem, args.output_dir)
        plot_memory_by_stage(df_mem, args.output_dir)

    # Projection schedule plot
    galore_log = _find_latest(os.path.join(args.log_dir, "**", "galore.log"))
    if galore_log:
        plot_projection_schedule_from_log(galore_log, args.output_dir)

    # SpinQuant-lite summary plots
    sq_path = _find_latest(os.path.join(args.pipeline_outputs, "**", "spinquant_layer_summary.json"))
    if sq_path:
        df_sq = load_spinquant_layer_summary(sq_path)
        plot_spinquant_error_hist(df_sq, args.output_dir)

    # Benchmark plots
    fr_path = _find_latest(os.path.join(args.benchmark_results, "**", "final_results.json"))
    if fr_path:
        res = load_final_results(fr_path)
        plot_benchmark_accuracy(res, args.output_dir)


if __name__ == "__main__":
    main()

