"""Experiment result helpers for DEGA-SDP."""

from __future__ import annotations

from pathlib import Path
import csv

import pandas as pd


def add_result_row(rows: list[dict], dataset: str, item: str, fold: int, paper_nested: float, paper_best: float, result: dict, seconds: float) -> None:
    rows.append(
        {
            "dataset": dataset,
            "item": item,
            "fold": fold,
            "paper_nested_f1": paper_nested,
            "paper_best_existing_f1": paper_best,
            "dega_f1_weighted": result["f1_weighted"],
            "dega_auc": result["auc"],
            "dega_f1_defect": result["f1_defect"],
            "dega_precision_defect": result["precision_defect"],
            "dega_recall_defect": result["recall_defect"],
            "beats_paper_best": result["f1_weighted"] > paper_best,
            "beats_paper_nested": result["f1_weighted"] > paper_nested,
            "selected_k": result["selected_k"],
            "beta": result["beta"],
            "threshold": result["threshold"],
            "validation_f1_weighted": result["validation_f1_weighted"],
            "validation_f1_defect": result["validation_f1_defect"],
            "mean_gate": result["mean_gate"],
            "seconds": seconds,
        }
    )


def summarize(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    summary = (
        df.groupby("dataset")
        .agg(
            rows=("item", "count"),
            paper_nested_f1=("paper_nested_f1", "mean"),
            paper_best_existing_f1=("paper_best_existing_f1", "mean"),
            dega_f1=("dega_f1_weighted", "mean"),
            dega_auc=("dega_auc", "mean"),
            dega_defect_precision=("dega_precision_defect", "mean"),
            dega_defect_recall=("dega_recall_defect", "mean"),
            dega_defect_f1=("dega_f1_defect", "mean"),
            wins_vs_paper_best=("beats_paper_best", "sum"),
            wins_vs_paper_nested=("beats_paper_nested", "sum"),
        )
        .reset_index()
    )
    summary["dega_minus_paper_best_f1"] = summary["dega_f1"] - summary["paper_best_existing_f1"]
    summary["dega_minus_paper_nested_f1"] = summary["dega_f1"] - summary["paper_nested_f1"]
    return summary


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

