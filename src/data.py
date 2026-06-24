"""Dataset loading utilities for DEGA-SDP."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import WPDP_PROMISE_NESTED_F1


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

