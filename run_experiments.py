"""Command line runner for DEGA-SDP experiments."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.dega_sdp import (
    CPDP_KAMEI_BEST_F1,
    CPDP_KAMEI_NESTED_F1,
    CPDP_PROMISE6_BEST_F1,
    CPDP_PROMISE6_NESTED_F1,
    KAMEI_SYSTEMS,
    PROMISE6_PAIRS,
    WPDP_KAMEI_BEST_F1,
    WPDP_KAMEI_NESTED_F1,
    WPDP_PROMISE_BEST_F1,
    WPDP_PROMISE_NESTED_F1,
    add_result_row,
    dega_sdp_predict,
    load_kamei,
    load_kamei_pair,
    load_promise,
    load_promise_pair,
    project_name,
    promise_files,
    resolve_data_dirs,
    set_seed,
    summarize,
    write_rows,
)


def run_wpdp_promise(rows, promise_dir, projects, total_state, selected_files=None):
    for file_name in promise_files(promise_dir):
        project = project_name(file_name)
        if project not in projects:
            continue
        if selected_files is not None and file_name not in selected_files:
            continue
        x_data, y_data = load_promise(promise_dir, file_name)
        class_counts = np.bincount(y_data.astype(int), minlength=2)
        if np.any(class_counts < 10):
            print(f"[DEGA] skip {file_name}: class_counts={class_counts.tolist()}", flush=True)
            continue
        splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(x_data, y_data), start=1):
            total_state["done"] += 1
            started = time.time()
            print(f"[DEGA {total_state['done']}/{total_state['total']}] WPDP-PROMISE {file_name} fold={fold}", flush=True)
            result = dega_sdp_predict(x_data[train_idx], y_data[train_idx], x_data[test_idx], y_data[test_idx], 42 + total_state["done"])
            add_result_row(rows, "WPDP-PROMISE", project, fold, WPDP_PROMISE_NESTED_F1[project], WPDP_PROMISE_BEST_F1[project], result, time.time() - started)


def run_wpdp_kamei(rows, kamei_dir, systems, total_state):
    for system in systems:
        x_data, y_data = load_kamei(kamei_dir, system)
        splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(x_data, y_data), start=1):
            total_state["done"] += 1
            started = time.time()
            print(f"[DEGA {total_state['done']}/{total_state['total']}] WPDP-Kamei {system} fold={fold}", flush=True)
            result = dega_sdp_predict(x_data[train_idx], y_data[train_idx], x_data[test_idx], y_data[test_idx], 42 + total_state["done"])
            add_result_row(rows, "WPDP-Kamei", system, fold, WPDP_KAMEI_NESTED_F1[system], WPDP_KAMEI_BEST_F1[system], result, time.time() - started)


def run_cpdp_kamei(rows, kamei_dir, total_state):
    for source in KAMEI_SYSTEMS:
        for target in KAMEI_SYSTEMS:
            if source == target:
                continue
            total_state["done"] += 1
            started = time.time()
            pair = f"{source}->{target}"
            print(f"[DEGA {total_state['done']}/{total_state['total']}] CPDP-Kamei {pair}", flush=True)
            train_x, train_y, test_x, test_y = load_kamei_pair(kamei_dir, source, target)
            result = dega_sdp_predict(train_x, train_y, test_x, test_y, 42 + total_state["done"])
            add_result_row(rows, "CPDP-Kamei", pair, 1, CPDP_KAMEI_NESTED_F1, CPDP_KAMEI_BEST_F1, result, time.time() - started)


def run_cpdp_promise6(rows, promise_dir, total_state):
    for source, target in PROMISE6_PAIRS:
        total_state["done"] += 1
        started = time.time()
        pair = f"{source}->{target}"
        print(f"[DEGA {total_state['done']}/{total_state['total']}] CPDP-PROMISE6 {pair}", flush=True)
        train_x, train_y, test_x, test_y = load_promise_pair(promise_dir, source, target)
        result = dega_sdp_predict(train_x, train_y, test_x, test_y, 42 + total_state["done"])
        add_result_row(rows, "CPDP-PROMISE6", pair, 1, CPDP_PROMISE6_NESTED_F1[(source, target)], CPDP_PROMISE6_BEST_F1[(source, target)], result, time.time() - started)


def main():
    parser = argparse.ArgumentParser(description="Run DEGA-SDP software defect prediction experiments.")
    parser.add_argument("--data-root", required=True, help="Dataset root containing HFS-Nested-Stacking/PROMISE and Kamei or data CSV directories.")
    parser.add_argument("--mode", choices=["smoke", "selected", "all", "wpdp-promise", "wpdp-kamei", "cpdp-kamei", "cpdp-promise6"], default="smoke")
    parser.add_argument("--output-dir", default="results", help="Directory for generated CSV files.")
    args = parser.parse_args()

    set_seed(42)
    kamei_dir, promise_dir = resolve_data_dirs(args.data_root)
    out_dir = Path(args.output_dir)
    rows = []
    started = time.time()

    if args.mode == "smoke":
        total_state = {"done": 0, "total": 20}
        run_wpdp_promise(rows, promise_dir, {"ant", "xerces"}, total_state, {"ant-1.7.csv", "xerces-1.4.csv"})
        prefix = "dega_smoke"
    elif args.mode == "selected":
        total_state = {"done": 0, "total": 80 + len(KAMEI_SYSTEMS) * 10 + 30 + 6}
        run_wpdp_promise(rows, promise_dir, {"ant", "xerces"}, total_state)
        run_wpdp_kamei(rows, kamei_dir, KAMEI_SYSTEMS, total_state)
        run_cpdp_kamei(rows, kamei_dir, total_state)
        run_cpdp_promise6(rows, promise_dir, total_state)
        prefix = "dega_selected"
    elif args.mode == "wpdp-promise":
        total_state = {"done": 0, "total": len(promise_files(promise_dir)) * 10}
        run_wpdp_promise(rows, promise_dir, set(WPDP_PROMISE_NESTED_F1), total_state)
        prefix = "dega_wpdp_promise"
    elif args.mode == "wpdp-kamei":
        total_state = {"done": 0, "total": len(KAMEI_SYSTEMS) * 10}
        run_wpdp_kamei(rows, kamei_dir, KAMEI_SYSTEMS, total_state)
        prefix = "dega_wpdp_kamei"
    elif args.mode == "cpdp-kamei":
        total_state = {"done": 0, "total": 30}
        run_cpdp_kamei(rows, kamei_dir, total_state)
        prefix = "dega_cpdp_kamei"
    elif args.mode == "cpdp-promise6":
        total_state = {"done": 0, "total": 6}
        run_cpdp_promise6(rows, promise_dir, total_state)
        prefix = "dega_cpdp_promise6"
    else:
        total_state = {"done": 0, "total": len(promise_files(promise_dir)) * 10 + len(KAMEI_SYSTEMS) * 10 + 30 + 6}
        run_wpdp_promise(rows, promise_dir, set(WPDP_PROMISE_NESTED_F1), total_state)
        run_wpdp_kamei(rows, kamei_dir, KAMEI_SYSTEMS, total_state)
        run_cpdp_kamei(rows, kamei_dir, total_state)
        run_cpdp_promise6(rows, promise_dir, total_state)
        prefix = "dega_all"

    if not rows:
        raise RuntimeError("No experiment rows were generated. Check data-root and mode.")
    result_path = out_dir / f"{prefix}_results.csv"
    summary_path = out_dir / f"{prefix}_summary.csv"
    write_rows(result_path, rows)
    summary = summarize(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"results={result_path}")
    print(f"summary={summary_path}")
    print(f"seconds={time.time() - started:.2f}")


if __name__ == "__main__":
    main()
