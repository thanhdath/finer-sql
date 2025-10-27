#!/usr/bin/env python3
"""
Plot grouped boxplots of reasoning (think) length distributions before and after RL.

Inputs:
  - JSON produced by compare_ckpt_improvements.py that includes fields:
      improvements: [
        {
          "difficulty": str,
          "cp0_full_completions": [str, ...],
          "cpN_full_completions": [str, ...],
          ...
        }, ...
      ]

Output:
  - PNG file with grouped boxplots: per difficulty and overall, with cp0 vs cpN.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # headless backend for saving figures
import matplotlib.pyplot as plt


def extract_think_segment(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    end_tag = "</think>"
    idx = text.find(end_tag)
    if idx == -1:
        return ""
    think_part = text[:idx]
    start_tag = "<think>"
    sidx = think_part.find(start_tag)
    if sidx != -1:
        think_part = think_part[sidx + len(start_tag):]
    return think_part.strip()


def collect_lengths(improvements: List[Dict[str, Any]]) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    cp0_by_diff: Dict[str, List[int]] = {}
    cpn_by_diff: Dict[str, List[int]] = {}

    for entry in improvements:
        diff = str(entry.get("difficulty", "unknown")).strip().lower() or "unknown"
        cp0 = entry.get("cp0_full_completions") or []
        cpn = entry.get("cpN_full_completions") or []

        for arr, target in ((cp0, cp0_by_diff), (cpn, cpn_by_diff)):
            if diff not in target:
                target[diff] = []
            if isinstance(arr, list):
                for s in arr:
                    if isinstance(s, str):
                        think = extract_think_segment(s)
                        if think:
                            target[diff].append(len(think))

    return cp0_by_diff, cpn_by_diff


def plot_grouped_boxplot(cp0_by_diff: Dict[str, List[int]], cpn_by_diff: Dict[str, List[int]], *, output_path: str):
    # Determine ordered groups
    preferred_order = ["simple", "moderate", "challenging", "unknown"]
    groups = [g for g in preferred_order if (g in cp0_by_diff or g in cpn_by_diff)]
    # Add any other groups encountered
    others = sorted(set(cp0_by_diff.keys()) | set(cpn_by_diff.keys()))
    for g in others:
        if g not in groups:
            groups.append(g)

    # Build datasets for plotting and add overall group at the end
    cp0_data = [cp0_by_diff.get(g, []) for g in groups]
    cpn_data = [cpn_by_diff.get(g, []) for g in groups]

    # Overall (all difficulties)
    cp0_all = [x for xs in cp0_data for x in xs]
    cpn_all = [x for xs in cpn_data for x in xs]
    groups.append("overall")
    cp0_data.append(cp0_all)
    cpn_data.append(cpn_all)

    # Plot layout
    num_groups = len(groups)
    fig, ax = plt.subplots(figsize=(max(8, num_groups * 1.6), 6))
    width = 0.35
    positions_cp0 = [i * 2 for i in range(num_groups)]
    positions_cpn = [i * 2 + 0.9 for i in range(num_groups)]

    # Boxplots
    bp0 = ax.boxplot(
        cp0_data,
        positions=positions_cp0,
        widths=width,
        patch_artist=True,
        boxprops=dict(facecolor="#4C78A8"),
        medianprops=dict(color="black"),
        whiskerprops=dict(color="#4C78A8"),
        capprops=dict(color="#4C78A8"),
        flierprops=dict(markeredgecolor="#4C78A8", markerfacecolor="#4C78A8", markersize=3),
    )
    bpN = ax.boxplot(
        cpn_data,
        positions=positions_cpn,
        widths=width,
        patch_artist=True,
        boxprops=dict(facecolor="#F58518"),
        medianprops=dict(color="black"),
        whiskerprops=dict(color="#F58518"),
        capprops=dict(color="#F58518"),
        flierprops=dict(markeredgecolor="#F58518", markerfacecolor="#F58518", markersize=3),
    )

    ax.set_xticks([(pc0 + pcN) / 2 for pc0, pcN in zip(positions_cp0, positions_cpn)])
    ax.set_xticklabels([g.capitalize() for g in groups], rotation=20, ha="right")
    ax.set_ylabel("Reasoning length (characters in <think> ... </think>)")
    ax.set_title("Reasoning Length Distributions Before vs After RL")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    # Legend
    ax.legend([bp0["boxes"][0], bpN["boxes"][0]], ["cp0 (before)", "cpN (after)"], loc="upper right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Grouped boxplot of reasoning lengths before/after RL")
    parser.add_argument("--input-json", required=True, type=str, help="Path to improvements_from_cp0.json")
    parser.add_argument("--output-png", type=str, default=None, help="Path to save the output PNG")
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    improvements = data.get("improvements") or []

    cp0_by_diff, cpn_by_diff = collect_lengths(improvements)

    if args.output_png:
        out_path = args.output_png
    else:
        base_dir = os.path.dirname(os.path.abspath(args.input_json))
        out_path = os.path.join(base_dir, "reasoning_lengths_grouped_boxplot.png")

    plot_grouped_boxplot(cp0_by_diff, cpn_by_diff, output_path=out_path)
    print(f"Saved grouped boxplot to: {out_path}")


if __name__ == "__main__":
    main()


