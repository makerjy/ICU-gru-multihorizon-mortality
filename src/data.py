from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing data file: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_splits(data_dir: Path, train_file: str, valid_file: str, test_file: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = data_dir / train_file
    valid_path = data_dir / valid_file
    test_path = data_dir / test_file

    missing = [str(p) for p in [train_path, valid_path, test_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "missing dataset files; check DATA_DIR or file names:\n" + "\n".join(missing)
        )

    return read_frame(train_path), read_frame(valid_path), read_frame(test_path)


def validate_columns(df: pd.DataFrame, need: list[str], tag: str) -> None:
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f"{tag} missing columns: {miss}")


def sort_df(df: pd.DataFrame, id_col: str, time_col: str) -> pd.DataFrame:
    return df.sort_values([id_col, time_col], kind="mergesort").reset_index(drop=True)


def compute_impute_stats(train_df: pd.DataFrame, feature_cols: list[str]) -> dict[str, float]:
    return {c: float(train_df[c].astype(float).mean()) for c in feature_cols}


def impute(df: pd.DataFrame, feature_cols: list[str], impute_stats: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    for c in feature_cols:
        out[c] = out[c].astype(float).fillna(impute_stats[c])
    return out


def compute_standard_stats(train_df: pd.DataFrame, feature_cols: list[str]) -> dict[str, dict[str, float]]:
    stats = {}
    for c in feature_cols:
        x = train_df[c].astype(float)
        mean = float(x.mean())
        std = float(x.std(ddof=0))
        if std == 0.0:
            std = 1.0
        stats[c] = {"mean": mean, "std": std}
    return stats


def standardize(df: pd.DataFrame, feature_cols: list[str], stats: dict[str, dict[str, float]]) -> pd.DataFrame:
    out = df.copy()
    for c in feature_cols:
        mean = stats[c]["mean"]
        std = stats[c]["std"]
        out[c] = (out[c].astype(float) - mean) / std
    return out


def add_future_label(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    event_col: str,
    horizon_hours: int,
    label_col: str = "_future_label",
) -> pd.DataFrame:
    out = df.copy()
    out[label_col] = 0

    for _, g in out.sort_values([id_col, time_col]).groupby(id_col, sort=False):
        t = g[time_col].to_numpy()
        e = g[event_col].to_numpy().astype(int)
        ev_times = t[e == 1]
        if ev_times.size == 0:
            continue

        left = np.searchsorted(ev_times, t, side="right")
        right = np.searchsorted(ev_times, t + horizon_hours, side="right")
        out.loc[g.index, label_col] = ((right - left) > 0).astype(int)

    return out


def add_label_observable_mask(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    label_col: str,
    horizon_hours: int,
    label_observable_col: str = "_label_observable",
) -> pd.DataFrame:
    out = df.copy()
    t_last = out.groupby(id_col)[time_col].max().rename("_t_last")
    out = out.merge(t_last, on=id_col, how="left")
    out[label_observable_col] = (out[label_col] == 1) | (out[time_col] + horizon_hours <= out["_t_last"])
    out = out.drop(columns=["_t_last"])
    return out


def drop_rows_after_first_event(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    event_col: str,
) -> pd.DataFrame:
    out = df.copy()
    first_t = out.loc[out[event_col].astype(int) == 1].groupby(id_col)[time_col].min()
    out = out.merge(first_t.rename("_first_event_t"), on=id_col, how="left")
    keep = out["_first_event_t"].isna() | (out[time_col] < out["_first_event_t"])
    return out.loc[keep].drop(columns=["_first_event_t"])


def prepare_labels(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    event_col: str,
    label_col: str,
    label_observable_col: str,
    horizon_hours: int,
    use_precomputed: bool,
    recompute: bool,
) -> pd.DataFrame:
    out = df.copy()
    has_label = label_col in out.columns
    has_obs = label_observable_col in out.columns

    if (not use_precomputed) or recompute or (not has_label):
        out = add_future_label(out, id_col, time_col, event_col, horizon_hours, label_col=label_col)
    if (not use_precomputed) or recompute or (not has_obs):
        out = add_label_observable_mask(
            out,
            id_col=id_col,
            time_col=time_col,
            label_col=label_col,
            horizon_hours=horizon_hours,
            label_observable_col=label_observable_col,
        )

    return out


def make_sequences(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    feature_cols: list[str],
    label_col: str,
    max_len: int | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    df_sorted = df.sort_values([id_col, time_col], kind="mergesort").reset_index(drop=True)
    sequences: list[dict] = []

    for sid, g in df_sorted.groupby(id_col, sort=False):
        X = g[feature_cols].to_numpy(dtype=np.float32)
        y = g[label_col].to_numpy(dtype=np.float32)
        t = g[time_col].to_numpy(dtype=np.int64)
        idx = g.index.to_numpy(dtype=np.int64)

        if max_len is not None and X.shape[0] > max_len:
            X = X[:max_len]
            y = y[:max_len]
            t = t[:max_len]
            idx = idx[:max_len]

        sequences.append({
            "X": X,
            "y": y,
            "t": t,
            "idx": idx,
            "sid": sid,
        })

    return df_sorted, sequences
