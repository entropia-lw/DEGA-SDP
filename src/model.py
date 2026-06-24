"""DEGA-SDP model implementation."""

from __future__ import annotations

import random

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split

from .config import BETA_VALUES, K_VALUES


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


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
