from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .config import RunConfig
from .data import (
    compute_impute_stats,
    compute_standard_stats,
    drop_rows_after_first_event,
    impute,
    load_splits,
    make_sequences,
    prepare_labels,
    sort_df,
    standardize,
    validate_columns,
)
from .metrics import (
    metrics_at_threshold,
    pick_threshold_by_recall_then_precision,
    safe_auc_ap,
    stay_level_from_row,
)
from .model import TimewiseGRU
from .viz import plot_metrics_table, plot_stay_metrics, plot_training_history


def _to_jsonable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


class SequenceDataset(Dataset):
    def __init__(self, sequences: list[dict], df_len: int):
        self.sequences = sequences
        self.df_len = df_len

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        return self.sequences[idx]


def collate_batch(batch: list[dict], pad_value: float, max_len: int | None):
    if not batch:
        raise ValueError("empty batch")

    lengths = [len(item["X"]) for item in batch]
    if max_len is not None:
        lengths = [min(l, max_len) for l in lengths]
    max_len_batch = max(lengths)

    n_feat = batch[0]["X"].shape[1]
    X_pad = np.full((len(batch), max_len_batch, n_feat), pad_value, dtype=np.float32)
    y_pad = np.zeros((len(batch), max_len_batch), dtype=np.float32)
    m_pad = np.zeros((len(batch), max_len_batch), dtype=np.float32)
    idx_pad = np.full((len(batch), max_len_batch), -1, dtype=np.int64)
    t_pad = np.full((len(batch), max_len_batch), -1, dtype=np.int64)

    for i, item in enumerate(batch):
        L = lengths[i]
        X_pad[i, :L, :] = item["X"][:L]
        y_pad[i, :L] = item["y"][:L]
        m_pad[i, :L] = 1.0
        idx_pad[i, :L] = item["idx"][:L]
        t_pad[i, :L] = item["t"][:L]

    return (
        torch.from_numpy(X_pad),
        torch.from_numpy(y_pad),
        torch.from_numpy(m_pad),
        torch.from_numpy(idx_pad),
        torch.from_numpy(t_pad),
        torch.tensor(lengths, dtype=torch.long),
    )


def run_epoch(
    loader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    train: bool,
    pos_weight: torch.Tensor,
    scaler: torch.cuda.amp.GradScaler | None,
    grad_clip: float,
    use_amp: bool,
) -> tuple[float, float, float]:
    model.train(train)
    total_loss, total_mask, total_elems = 0.0, 0.0, 0.0

    for Xb, yb, mb, _, _, lengths in loader:
        Xb = Xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        mb = mb.to(device, non_blocking=True)

        with torch.autocast(device_type=device, enabled=use_amp):
            logits = model(Xb, lengths)
            loss_mat = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, yb, reduction="none", pos_weight=pos_weight
            ) * mb
            loss = loss_mat.sum() / (mb.sum() + 1e-8)

        if train:
            optimizer.zero_grad(set_to_none=True)
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        total_loss += float(loss_mat.sum().detach().cpu())
        total_mask += float(mb.sum().detach().cpu())
        total_elems += float(mb.numel())

    return total_loss / (total_mask + 1e-8), total_mask, total_elems


def predict_row_scores(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    df_len: int,
) -> np.ndarray:
    row_score = np.full((df_len,), np.nan, dtype=np.float32)
    model.eval()

    with torch.inference_mode():
        for Xb, _, mb, idxb, _, lengths in loader:
            Xb = Xb.to(device, non_blocking=True)
            logits = model(Xb, lengths)
            probs = torch.sigmoid(logits).cpu().numpy()

            mask = mb.numpy().astype(bool)
            idx = idxb.numpy()

            row_score[idx[mask]] = probs[mask]

    return row_score


def train_and_evaluate(cfg: RunConfig) -> dict:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    train_df, valid_df, test_df = load_splits(
        cfg.data.data_dir, cfg.data.train_file, cfg.data.valid_file, cfg.data.test_file
    )
    print("train:", train_df.shape, "valid:", valid_df.shape, "test:", test_df.shape)

    need = [cfg.data.id_col, cfg.data.time_col, cfg.data.event_col] + cfg.data.feature_cols
    for name, df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        validate_columns(df, need, name)

    train_df = sort_df(train_df, cfg.data.id_col, cfg.data.time_col)
    valid_df = sort_df(valid_df, cfg.data.id_col, cfg.data.time_col)
    test_df = sort_df(test_df, cfg.data.id_col, cfg.data.time_col)

    train_df = prepare_labels(
        train_df,
        id_col=cfg.data.id_col,
        time_col=cfg.data.time_col,
        event_col=cfg.data.event_col,
        label_col=cfg.data.label_col,
        label_observable_col=cfg.data.label_observable_col,
        horizon_hours=cfg.data.horizon_hours,
        use_precomputed=cfg.data.use_precomputed_labels,
        recompute=cfg.data.recompute_labels,
    )
    valid_df = prepare_labels(
        valid_df,
        id_col=cfg.data.id_col,
        time_col=cfg.data.time_col,
        event_col=cfg.data.event_col,
        label_col=cfg.data.label_col,
        label_observable_col=cfg.data.label_observable_col,
        horizon_hours=cfg.data.horizon_hours,
        use_precomputed=cfg.data.use_precomputed_labels,
        recompute=cfg.data.recompute_labels,
    )
    test_df = prepare_labels(
        test_df,
        id_col=cfg.data.id_col,
        time_col=cfg.data.time_col,
        event_col=cfg.data.event_col,
        label_col=cfg.data.label_col,
        label_observable_col=cfg.data.label_observable_col,
        horizon_hours=cfg.data.horizon_hours,
        use_precomputed=cfg.data.use_precomputed_labels,
        recompute=cfg.data.recompute_labels,
    )

    for df_name, df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        if cfg.data.label_col not in df.columns:
            raise KeyError(f"{df_name} missing label_col: {cfg.data.label_col}")

    if cfg.data.label_observable_col in train_df.columns:
        train_df = train_df.loc[train_df[cfg.data.label_observable_col]].copy()
    if cfg.data.label_observable_col in valid_df.columns:
        valid_df = valid_df.loc[valid_df[cfg.data.label_observable_col]].copy()
    if cfg.data.label_observable_col in test_df.columns:
        test_df = test_df.loc[test_df[cfg.data.label_observable_col]].copy()

    if cfg.data.drop_after_event:
        train_df = drop_rows_after_first_event(train_df, cfg.data.id_col, cfg.data.time_col, cfg.data.event_col)
        valid_df = drop_rows_after_first_event(valid_df, cfg.data.id_col, cfg.data.time_col, cfg.data.event_col)
        test_df = drop_rows_after_first_event(test_df, cfg.data.id_col, cfg.data.time_col, cfg.data.event_col)

    if cfg.data.apply_cutoff_to_train:
        train_df = train_df.loc[train_df[cfg.data.time_col] <= cfg.data.cutoff_hours].copy()

    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    impute_stats = compute_impute_stats(train_df, cfg.data.feature_cols)
    train_i = impute(train_df, cfg.data.feature_cols, impute_stats)
    valid_i = impute(valid_df, cfg.data.feature_cols, impute_stats)
    test_i = impute(test_df, cfg.data.feature_cols, impute_stats)

    std_stats = compute_standard_stats(train_i, cfg.data.feature_cols)
    train_s = standardize(train_i, cfg.data.feature_cols, std_stats)
    valid_s = standardize(valid_i, cfg.data.feature_cols, std_stats)
    test_s = standardize(test_i, cfg.data.feature_cols, std_stats)

    train_s, train_seq = make_sequences(
        train_s,
        id_col=cfg.data.id_col,
        time_col=cfg.data.time_col,
        feature_cols=cfg.data.feature_cols,
        label_col=cfg.data.label_col,
        max_len=cfg.sequence.max_len,
    )
    valid_s, valid_seq = make_sequences(
        valid_s,
        id_col=cfg.data.id_col,
        time_col=cfg.data.time_col,
        feature_cols=cfg.data.feature_cols,
        label_col=cfg.data.label_col,
        max_len=cfg.sequence.max_len,
    )
    test_s, test_seq = make_sequences(
        test_s,
        id_col=cfg.data.id_col,
        time_col=cfg.data.time_col,
        feature_cols=cfg.data.feature_cols,
        label_col=cfg.data.label_col,
        max_len=cfg.sequence.max_len,
    )

    train_ds = SequenceDataset(train_seq, df_len=len(train_s))
    valid_ds = SequenceDataset(valid_seq, df_len=len(valid_s))
    test_ds = SequenceDataset(test_seq, df_len=len(test_s))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    batch_train = cfg.train.batch_train_cuda if device == "cuda" else cfg.train.batch_train_cpu
    batch_eval = cfg.train.batch_eval_cuda if device == "cuda" else cfg.train.batch_eval_cpu
    num_workers = 2 if device == "cuda" else 0
    pin_memory = device == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        collate_fn=lambda b: collate_batch(b, cfg.sequence.pad_value, cfg.sequence.max_len),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_eval,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        collate_fn=lambda b: collate_batch(b, cfg.sequence.pad_value, cfg.sequence.max_len),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_eval,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        collate_fn=lambda b: collate_batch(b, cfg.sequence.pad_value, cfg.sequence.max_len),
    )

    y_train_all = train_s[cfg.data.label_col].to_numpy(dtype=int)
    n_pos = int((y_train_all == 1).sum())
    n_neg = int((y_train_all == 0).sum())
    pos_weight_val = (n_neg / n_pos) if n_pos > 0 else 1.0
    print(f"[IMBALANCE] pos={n_pos} neg={n_neg} pos_weight={pos_weight_val:.4f}")

    model = TimewiseGRU(
        n_feat=len(cfg.data.feature_cols),
        hidden=cfg.model.hidden,
        n_layers=cfg.model.n_layers,
        dropout=cfg.model.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)

    best_val = float("inf")
    best_state = None
    best_epoch = None
    bad = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "valid_loss": [],
        "row_auc": [],
        "row_ap": [],
        "stay_auc": [],
        "stay_ap": [],
    }

    for epoch in range(1, cfg.train.max_epochs + 1):
        t0 = time.time()
        tr_loss, _, _ = run_epoch(
            train_loader,
            model,
            optimizer,
            device,
            train=True,
            pos_weight=pos_weight,
            scaler=scaler,
            grad_clip=cfg.train.grad_clip,
            use_amp=use_amp,
        )
        va_loss, _, _ = run_epoch(
            valid_loader,
            model,
            optimizer=None,
            device=device,
            train=False,
            pos_weight=pos_weight,
            scaler=None,
            grad_clip=cfg.train.grad_clip,
            use_amp=use_amp,
        )

        row_score_va = predict_row_scores(model, valid_loader, device, df_len=len(valid_s))
        valid_mask = np.isfinite(row_score_va)
        row_score_va = row_score_va[valid_mask]
        valid_eval_df = valid_s.loc[valid_mask].reset_index(drop=True)

        auc_row, ap_row = safe_auc_ap(valid_eval_df[cfg.data.label_col].to_numpy(), row_score_va)
        y_stay, s_stay = stay_level_from_row(
            valid_eval_df,
            row_score_va,
            id_col=cfg.data.id_col,
            time_col=cfg.data.time_col,
            label_col=cfg.data.label_col,
            cutoff_hours=cfg.data.cutoff_hours,
            agg=cfg.data.agg_mode,
        )
        auc_stay, ap_stay = safe_auc_ap(y_stay, s_stay)

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["valid_loss"].append(va_loss)
        history["row_auc"].append(auc_row)
        history["row_ap"].append(ap_row)
        history["stay_auc"].append(auc_stay)
        history["stay_ap"].append(ap_stay)

        improved = (best_val - va_loss) > cfg.train.min_delta
        if improved:
            best_val = va_loss
            best_state = model.state_dict()
            best_epoch = epoch
            bad = 0

            print(
                f"[BEST] epoch {epoch:02d} | train_loss={tr_loss:.6f} | valid_loss={va_loss:.6f} | "
                f"row AUC={auc_row:.4f} AP={ap_row:.4f} | "
                f"stay AUC={auc_stay:.4f} AP={ap_stay:.4f} | time={time.time()-t0:.1f}s"
            )
        else:
            bad += 1
            print(
                f"epoch {epoch:02d} | train_loss={tr_loss:.6f} | valid_loss={va_loss:.6f} | "
                f"row AUC={auc_row:.4f} AP={ap_row:.4f} | "
                f"stay AUC={auc_stay:.4f} AP={ap_stay:.4f} | "
                f"bad={bad}/{cfg.train.patience} | time={time.time()-t0:.1f}s"
            )
            if bad >= cfg.train.patience:
                print(f"[EARLY STOP] best_valid_loss={best_val:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print("[OK] loaded best model state")

    row_score_va = predict_row_scores(model, valid_loader, device, df_len=len(valid_s))
    row_score_te = predict_row_scores(model, test_loader, device, df_len=len(test_s))

    valid_mask = np.isfinite(row_score_va)
    test_mask = np.isfinite(row_score_te)

    valid_eval_df = valid_s.loc[valid_mask].reset_index(drop=True)
    test_eval_df = test_s.loc[test_mask].reset_index(drop=True)

    row_score_va = row_score_va[valid_mask]
    row_score_te = row_score_te[test_mask]

    auc_row_va, ap_row_va = safe_auc_ap(valid_eval_df[cfg.data.label_col].to_numpy(), row_score_va)
    auc_row_te, ap_row_te = safe_auc_ap(test_eval_df[cfg.data.label_col].to_numpy(), row_score_te)

    yv_stay, sv_stay = stay_level_from_row(
        valid_eval_df,
        row_score_va,
        id_col=cfg.data.id_col,
        time_col=cfg.data.time_col,
        label_col=cfg.data.label_col,
        cutoff_hours=cfg.data.cutoff_hours,
        agg=cfg.data.agg_mode,
    )
    yt_stay, st_stay = stay_level_from_row(
        test_eval_df,
        row_score_te,
        id_col=cfg.data.id_col,
        time_col=cfg.data.time_col,
        label_col=cfg.data.label_col,
        cutoff_hours=cfg.data.cutoff_hours,
        agg=cfg.data.agg_mode,
    )

    thr, thr_prec, thr_rec = pick_threshold_by_recall_then_precision(
        yv_stay, sv_stay, target_recall=cfg.data.target_recall
    )
    m_v = metrics_at_threshold(yv_stay, sv_stay, thr)
    m_t = metrics_at_threshold(yt_stay, st_stay, thr)

    auc_stay_va, ap_stay_va = safe_auc_ap(yv_stay, sv_stay)
    auc_stay_te, ap_stay_te = safe_auc_ap(yt_stay, st_stay)

    print("[VALID row]", {"auc": auc_row_va, "ap": ap_row_va})
    print(
        "[VALID stay] "
        f"AUC={auc_stay_va:.4f} AP={ap_stay_va:.4f} thr={thr:.4f} "
        f"P={m_v['precision']:.4f} R={m_v['recall']:.4f}"
    )
    print("[TEST row ]", {"auc": auc_row_te, "ap": ap_row_te})
    print(
        "[TEST stay] "
        f"AUC={auc_stay_te:.4f} AP={ap_stay_te:.4f} thr={thr:.4f} "
        f"P={m_t['precision']:.4f} R={m_t['recall']:.4f}"
    )

    stay_methods = ["max", "mean", "last"]
    stay_rows = []
    for method in stay_methods:
        yv, sv = stay_level_from_row(
            valid_eval_df,
            row_score_va,
            id_col=cfg.data.id_col,
            time_col=cfg.data.time_col,
            label_col=cfg.data.label_col,
            cutoff_hours=cfg.data.cutoff_hours,
            agg=method,
        )
        yt, st = stay_level_from_row(
            test_eval_df,
            row_score_te,
            id_col=cfg.data.id_col,
            time_col=cfg.data.time_col,
            label_col=cfg.data.label_col,
            cutoff_hours=cfg.data.cutoff_hours,
            agg=method,
        )
        auc_v, ap_v = safe_auc_ap(yv, sv)
        auc_t, ap_t = safe_auc_ap(yt, st)
        thr_m, _, _ = pick_threshold_by_recall_then_precision(yv, sv, target_recall=cfg.data.target_recall)
        mv = metrics_at_threshold(yv, sv, thr_m)
        mt = metrics_at_threshold(yt, st, thr_m)
        stay_rows.append(
            {
                "method": method,
                "valid_auc": auc_v,
                "valid_ap": ap_v,
                "valid_f1": mv["f1"],
                "test_auc": auc_t,
                "test_ap": ap_t,
                "test_f1": mt["f1"],
            }
        )

    stay_df = pd.DataFrame(stay_rows)

    artifacts_dir = cfg.output.artifacts_dir
    output_dir = cfg.output.output_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_training_history(history, output_dir)
    plot_stay_metrics(stay_df, output_dir)

    row_df = pd.DataFrame(
        [
            {"split": "valid", "auc": auc_row_va, "ap": ap_row_va},
            {"split": "test", "auc": auc_row_te, "ap": ap_row_te},
        ]
    )

    row_df.to_csv(output_dir / "row_metrics.csv", index=False)
    stay_df.to_csv(output_dir / "stay_metrics.csv", index=False)

    plot_metrics_table(row_df.round(4), "Row-level Metrics", output_dir / "row_metrics_table.png")
    plot_metrics_table(stay_df.round(4), "Stay-level Metrics", output_dir / "stay_metrics_table.png")

    torch.save(model.state_dict(), artifacts_dir / "kstep_gru_state.pt")

    meta = {
        "feature_cols": cfg.data.feature_cols,
        "impute_stats": impute_stats,
        "standardize_stats": std_stats,
        "config": _to_jsonable(asdict(cfg)),
        "best_epoch": best_epoch,
        "best_valid_loss": float(best_val),
        "pos_weight": float(pos_weight_val),
        "row_metrics_valid": {"auc": auc_row_va, "ap": ap_row_va},
        "row_metrics_test": {"auc": auc_row_te, "ap": ap_row_te},
        "stay_metrics_valid": {
            "auc": auc_stay_va,
            "ap": ap_stay_va,
            "threshold": thr,
            "precision": m_v["precision"],
            "recall": m_v["recall"],
            "f1": m_v["f1"],
        },
        "stay_metrics_test": {
            "auc": auc_stay_te,
            "ap": ap_stay_te,
            "threshold": thr,
            "precision": m_t["precision"],
            "recall": m_t["recall"],
            "f1": m_t["f1"],
        },
    }

    with open(artifacts_dir / "kstep_gru_meta.json", "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(meta), f, indent=2)

    with open(artifacts_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(history), f, indent=2)

    print("saved artifacts to:", artifacts_dir.resolve())
    print("saved outputs to:", output_dir.resolve())

    return meta
