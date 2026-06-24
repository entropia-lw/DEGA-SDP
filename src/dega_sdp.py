"""DEGA-SDP core model and experiment utilities.

DEGA-SDP combines global discriminative evidence, class prototype evidence,
and a reliability gate for software defect prediction.
"""

from __future__ import annotations

import csv
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

KAMEI_SYSTEMS = ["bugzilla", "columba", "jdt", "mozilla", "platform", "postgres"]

PROMISE6_PAIRS = [
    ("xerces-1.3.csv", "ivy-2.0.csv"),
    ("ant-1.6.csv", "camel-1.4.csv"),
    ("jedit-4.1.csv", "camel-1.4.csv"),
    ("xalan-2.5.csv", "lucene-2.2.csv"),
    ("xalan-2.5.csv", "xerces-1.3.csv"),
    ("ivy-2.0.csv", "xerces-1.3.csv"),
]

WPDP_KAMEI_NESTED_F1 = {
    "bugzilla": 0.7481,
    "columba": 0.7754,
    "jdt": 0.8268,
    "mozilla": 0.9304,
    "platform": 0.8410,
    "postgres": 0.7986,
}
WPDP_KAMEI_BEST_F1 = {
    "bugzilla": 0.7554,
    "columba": 0.7690,
    "jdt": 0.8245,
    "mozilla": 0.9344,
    "platform": 0.8372,
    "postgres": 0.7976,
}

WPDP_PROMISE_NESTED_F1 = {
    "ant": 0.7890,
    "camel": 0.8072,
    "jedit": 0.8393,
    "log4j": 0.8245,
    "lucene": 0.6314,
    "poi": 0.8038,
    "synapse": 0.8121,
    "velocity": 0.8116,
    "xalan": 0.8414,
    "xerces": 0.8274,
}
WPDP_PROMISE_BEST_F1 = {
    "ant": 0.7662,
    "camel": 0.7943,
    "jedit": 0.8362,
    "log4j": 0.8308,
    "lucene": 0.7133,
    "poi": 0.8068,
    "synapse": 0.7987,
    "velocity": 0.8367,
    "xalan": 0.8316,
    "xerces": 0.8376,
}

CPDP_KAMEI_NESTED_F1 = 0.7073
CPDP_KAMEI_BEST_F1 = 0.6940

CPDP_PROMISE6_NESTED_F1 = {
    ("xerces-1.3.csv", "ivy-2.0.csv"): 0.8910,
    ("ant-1.6.csv", "camel-1.4.csv"): 0.7410,
    ("jedit-4.1.csv", "camel-1.4.csv"): 0.7345,
    ("xalan-2.5.csv", "lucene-2.2.csv"): 0.5256,
    ("xalan-2.5.csv", "xerces-1.3.csv"): 0.6645,
    ("ivy-2.0.csv", "xerces-1.3.csv"): 0.8565,
}
CPDP_PROMISE6_BEST_F1 = {
    ("xerces-1.3.csv", "ivy-2.0.csv"): 0.4729,
    ("ant-1.6.csv", "camel-1.4.csv"): 0.3954,
    ("jedit-4.1.csv", "camel-1.4.csv"): 0.6934,
    ("xalan-2.5.csv", "lucene-2.2.csv"): 0.7832,
    ("xalan-2.5.csv", "xerces-1.3.csv"): 0.4005,
    ("ivy-2.0.csv", "xerces-1.3.csv"): 0.4207,
}

K_VALUES = [2, 3, 4, 5, 6]
BETA_VALUES = [0.25, 0.5, 0.75, 1.0]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def resolve_data_dirs(data_root: str | Path) -> tuple[Path, Path]:
    root = Path(data_root).expanduser().resolve()
    kamei_candidates = [root / "data", root / "Kamei", root / "HFS-Nested-Stacking" / "Kamei"]
    promise_candidates = [root / "HFS-Nested-Stacking" / "PROMISE", root / "PROMISE"]
    kamei_dir = next((p for p in kamei_candidates if p.exists()), None)
    promise_dir = next((p for p in promise_candidates if p.exists()), None)
    if kamei_dir is None:
        raise FileNotFoundError(f"Could not find Kamei CSV directory under {root}")
    if promise_dir is None:
        raise FileNotFoundError(f"Could not find PROMISE CSV directory under {root}")
    return kamei_dir, promise_dir


def project_name(file_name: str) -> str:
    return file_name.split("-")[0].lower()


def promise_files(promise_dir: Path) -> list[str]:
    return sorted(path.name for path in promise_dir.glob("*.csv") if project_name(path.name) in WPDP_PROMISE_NESTED_F1)


def kamei_frame(kamei_dir: Path, system_name: str) -> pd.DataFrame:
    df = pd.read_csv(kamei_dir / f"{system_name}.csv").iloc[:, 2:17].copy()
    df["ALA"] = df["la"] / df["npt"]
    df["ALD"] = df["ld"] / df["npt"]
    df["GEXP1"] = (df["rexp"] - df["exp"]).abs()
    df["GEXP2"] = (df["exp"] - df["sexp"]).abs()
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


def load_kamei(kamei_dir: Path, system_name: str) -> tuple[np.ndarray, np.ndarray]:
    df = kamei_frame(kamei_dir, system_name)
    x_data = StandardScaler().fit_transform(df.drop("bug", axis=1).to_numpy())
    y_data = df["bug"].to_numpy().astype(int)
    return x_data, y_data


def load_kamei_pair(kamei_dir: Path, source: str, target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_df = kamei_frame(kamei_dir, source)
    target_df = kamei_frame(kamei_dir, target)
    features = [column for column in source_df.columns if column != "bug"]
    scaler = StandardScaler().fit(source_df[features].to_numpy())
    train_x = scaler.transform(source_df[features].to_numpy())
    test_x = scaler.transform(target_df[features].to_numpy())
    return train_x, source_df["bug"].to_numpy().astype(int), test_x, target_df["bug"].to_numpy().astype(int)


def promise_frame(promise_dir: Path, file_name: str) -> pd.DataFrame:
    return pd.read_csv(promise_dir / file_name).iloc[:, 3:24].copy()


def load_promise(promise_dir: Path, file_name: str) -> tuple[np.ndarray, np.ndarray]:
    df = promise_frame(promise_dir, file_name)
    x_data = StandardScaler().fit_transform(df.drop("bug", axis=1).to_numpy())
    y_data = (df["bug"].to_numpy() != 0).astype(int)
    return x_data, y_data


def load_promise_pair(promise_dir: Path, source: str, target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_df = promise_frame(promise_dir, source)
    target_df = promise_frame(promise_dir, target)
    common = [column for column in source_df.columns if column != "bug" and column in set(target_df.columns)]
    scaler = StandardScaler().fit(source_df[common].to_numpy())
    train_x = scaler.transform(source_df[common].to_numpy())
    test_x = scaler.transform(target_df[common].to_numpy())
    train_y = (source_df["bug"].to_numpy() != 0).astype(int)
    test_y = (target_df["bug"].to_numpy() != 0).astype(int)
    return train_x, train_y, test_x, test_y


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "f1_defect": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "precision_defect": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_defect": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "auc": safe_auc(y_true, y_score),
    }


def fit_global(train_x: np.ndarray, train_y: np.ndarray, seed: int) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        loss="binary_crossentropy",
        learning_rate=0.08,
        max_leaf_nodes=31,
        max_iter=80,
        l2_regularization=0.05,
        random_state=seed,
    )
    defect_weight = max(1.0, (train_y == 0).sum() / max(1, (train_y == 1).sum())) * 0.35
    weights = np.where(train_y == 1, defect_weight, 1.0)
    model.fit(train_x, train_y, sample_weight=weights)
    return model


def fit_centers(train_x: np.ndarray, train_y: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    defect_x = train_x[train_y == 1]
    clean_x = train_x[train_y == 0]
    defect_k = min(k, len(defect_x))
    clean_k = min(k, len(clean_x))
    if defect_k < 1 or clean_k < 1:
        raise ValueError("Both classes need at least one training sample.")
    defect_model = KMeans(n_clusters=defect_k, random_state=seed, n_init=10)
    clean_model = KMeans(n_clusters=clean_k, random_state=seed, n_init=10)
    defect_model.fit(defect_x)
    clean_model.fit(clean_x)
    return defect_model.cluster_centers_, clean_model.cluster_centers_


def prototype_terms(train_x: np.ndarray, x_data: np.ndarray, defect_centers: np.ndarray, clean_centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_dist = np.minimum(
        euclidean_distances(train_x, defect_centers).min(axis=1),
        euclidean_distances(train_x, clean_centers).min(axis=1),
    )
    defect_dist = euclidean_distances(x_data, defect_centers).min(axis=1)
    clean_dist = euclidean_distances(x_data, clean_centers).min(axis=1)
    margin = clean_dist - defect_dist
    lo = np.percentile(margin, 1)
    hi = np.percentile(margin, 99)
    if hi <= lo:
        proto_score = np.full_like(margin, 0.5, dtype=float)
    else:
        proto_score = (np.clip(margin, lo, hi) - lo) / (hi - lo)
    scale = max(np.median(train_dist), 1e-6)
    density = np.exp(-np.minimum(defect_dist, clean_dist) / scale)
    return proto_score, np.clip(density, 0.0, 1.0)


def gated_fusion(global_score: np.ndarray, proto_score: np.ndarray, density: np.ndarray, beta: float) -> tuple[np.ndarray, np.ndarray]:
    uncertainty = 1.0 - np.abs(global_score - 0.5) * 2.0
    agreement = 1.0 - np.abs(global_score - proto_score)
    gate = np.clip(uncertainty * agreement * density, 0.0, 1.0)
    fused = np.clip(global_score + beta * gate * (proto_score - global_score), 0.0, 1.0)
    return fused, gate


def select_gate(proto_x: np.ndarray, proto_y: np.ndarray, calib_x: np.ndarray, calib_y: np.ndarray, seed: int) -> dict:
    global_model = fit_global(proto_x, proto_y, seed)
    global_calib = global_model.predict_proba(calib_x)[:, 1]
    best = None
    for k in K_VALUES:
        defect_centers, clean_centers = fit_centers(proto_x, proto_y, k, seed)
        proto_calib, density_calib = prototype_terms(proto_x, calib_x, defect_centers, clean_centers)
        for beta in BETA_VALUES:
            fused, gate = gated_fusion(global_calib, proto_calib, density_calib, beta)
            thresholds = np.unique(np.percentile(fused, np.linspace(0, 100, 101)))
            for threshold in thresholds:
                pred = (fused >= threshold).astype(int)
                result = compute_metrics(calib_y, pred, fused)
                score = (result["f1_weighted"], result["f1_defect"], result["precision_defect"], result["recall_defect"], result["auc"])
                if best is None or score > best["score"]:
                    best = {
                        "k": int(k),
                        "beta": float(beta),
                        "threshold": float(threshold),
                        "validation": result,
                        "score": score,
                        "mean_gate": float(np.mean(gate)),
                    }
    return best


def dega_sdp_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray, seed: int = 42) -> dict[str, float]:
    proto_x, calib_x, proto_y, calib_y = train_test_split(
        train_x,
        train_y,
        test_size=0.25,
        shuffle=True,
        random_state=seed,
        stratify=train_y,
    )
    selected = select_gate(proto_x, proto_y, calib_x, calib_y, seed)
    global_model = fit_global(train_x, train_y, seed)
    global_test = global_model.predict_proba(test_x)[:, 1]
    defect_centers, clean_centers = fit_centers(train_x, train_y, selected["k"], seed)
    proto_test, density_test = prototype_terms(train_x, test_x, defect_centers, clean_centers)
    fused, gate = gated_fusion(global_test, proto_test, density_test, selected["beta"])
    pred = (fused >= selected["threshold"]).astype(int)
    result = compute_metrics(test_y, pred, fused)
    result.update(
        {
            "selected_k": selected["k"],
            "beta": selected["beta"],
            "threshold": selected["threshold"],
            "validation_f1_weighted": selected["validation"]["f1_weighted"],
            "validation_f1_defect": selected["validation"]["f1_defect"],
            "mean_gate": float(np.mean(gate)),
        }
    )
    return result


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
