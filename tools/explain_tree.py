#!/usr/bin/env python3
"""Explain RL portfolio allocations with tree surrogates.

This script mirrors the data loading and environment construction used by
``main.py``/``trainer/irl_trainer.py`` and fits a decision-tree surrogate to
approximate the learned policy's portfolio weights. The surrogate (and
per-stock trees) provide human-readable rules together with fidelity scores.

Example usage
-------------
python tools/explain_tree.py \
    --model-path ./checkpoints/ppo_hgat_custom_20251108_103925.zip \
    --market custom \
    --test-start-date 2024-01-02 \
    --test-end-date 2024-12-26 \
    --max-depth 5 \
    --top-k-stocks 10
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor, export_text
from torch_geometric.loader import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.data_loader import AllGraphDataSampler
from env.portfolio_env import StockPortfolioEnv
from stable_baselines3 import PPO


@dataclass
class DatasetMetadata:
    """Lightweight container for dataset descriptors."""

    data_dir: Path
    num_stocks: int
    input_dim: int
    lookback: int


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def auto_detect_metadata(args: argparse.Namespace) -> DatasetMetadata:
    """Infer dataset directory together with stock/feature/temporal dimensions."""
    base_dir = Path(args.data_root).expanduser().resolve()
    data_dir = base_dir / f"data_train_predict_{args.market}" / f"{args.horizon}_{args.relation_type}"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    sample_files = sorted(p for p in data_dir.glob("*.pkl"))
    if not sample_files:
        raise FileNotFoundError(f"No .pkl files found under {data_dir}")

    last_error: Exception | None = None
    for sample_path in sample_files:
        try:
            with sample_path.open("rb") as fh:
                sample = pickle.load(fh)

            features = sample.get("features")
            if features is None:
                raise KeyError("missing 'features'")

            if isinstance(features, torch.Tensor):
                feat_shape = tuple(features.shape)
                if features.numel() == 0:
                    raise ValueError("feature tensor is empty")
            else:
                features_np = np.asarray(features)
                if features_np.size == 0:
                    raise ValueError("feature array is empty")
                feat_shape = features_np.shape

            if len(feat_shape) < 2:
                raise ValueError(f"feature tensor has shape {feat_shape}; expected >=2 dimensions")

            num_stocks = feat_shape[-2]
            input_dim = feat_shape[-1]
            if num_stocks <= 0 or input_dim <= 0:
                raise ValueError(f"invalid feature dimensions: num_stocks={num_stocks}, input_dim={input_dim}")

            ts_features = sample.get("ts_features")
            if ts_features is None:
                raise KeyError("missing 'ts_features'")

            if isinstance(ts_features, torch.Tensor):
                ts_shape = tuple(ts_features.shape)
                if ts_features.numel() == 0:
                    raise ValueError("ts_features tensor is empty")
            else:
                ts_np = np.asarray(ts_features)
                if ts_np.size == 0:
                    raise ValueError("ts_features array is empty")
                ts_shape = ts_np.shape

            if len(ts_shape) < 3:
                raise ValueError(f"ts_features tensor has shape {ts_shape}; expected >=3 dimensions")

            ts_num_stocks = ts_shape[-3] if len(ts_shape) >= 3 else ts_shape[0]
            lookback = ts_shape[-2]
            ts_feat_dim = ts_shape[-1]

            if ts_num_stocks != num_stocks or ts_feat_dim != input_dim:
                raise ValueError(
                    "ts_features dimensions do not align with feature tensor; "
                    f"expected ({num_stocks}, *, {input_dim}) got {ts_shape}"
                )
            if lookback <= 0:
                raise ValueError("lookback inferred from ts_features must be positive")

            return DatasetMetadata(
                data_dir=data_dir,
                num_stocks=num_stocks,
                input_dim=input_dim,
                lookback=lookback,
            )
        except Exception as exc:  # noqa: PERF203
            last_error = exc
            continue

    raise RuntimeError(
        "Unable to infer metadata from dataset. Last error: "
        f"{last_error}" if last_error else "no usable sample files"
    )


def process_data(batch: Dict[str, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, ...]:
    """Align with trainer/irl_trainer.process_data."""
    def move(key: str):
        value = batch[key]
        if isinstance(value, torch.Tensor):
            return value.to(device).squeeze()
        return value

    corr = move("corr")
    ts_features = move("ts_features")
    features = move("features")
    ind = move("industry_matrix")
    pos = move("pos_matrix")
    neg = move("neg_matrix")
    returns = move("labels")
    pyg_data = batch["pyg_data"].to(device)
    mask = batch.get("mask")
    return corr, ts_features, features, ind, pos, neg, returns, pyg_data, mask


def build_feature_names(
    num_stocks: int,
    input_dim: int,
    lookback: int,
    feature_labels: Sequence[str],
    relation_labels: Sequence[str],
    tickers: Sequence[str] = None,
    include_prev_weights: bool = True,
    extra_feature_labels: Sequence[str] | None = None,
) -> List[str]:
    names: List[str] = []
    relation_labels = list(relation_labels)

    def get_ticker_name(idx):
        if tickers and idx < len(tickers):
            return str(tickers[idx])
        return f"Stock[{idx}]"

    for relation in relation_labels:
        for i in range(num_stocks):
            for j in range(num_stocks):
                names.append(f"{relation}[{get_ticker_name(i)},{get_ticker_name(j)}]")
    lookback = max(1, int(lookback))
    for stock_idx in range(num_stocks):
        stock_label = get_ticker_name(stock_idx)
        for lag in range(lookback):
            # Lag index 0 is the oldest observation, so translate to a t-N style name
            history_offset = (lookback - 1) - lag
            suffix = "t-0" if history_offset == 0 else f"t-{history_offset}"
            for feat_idx in range(input_dim):
                label = feature_labels[feat_idx] if feat_idx < len(feature_labels) else f"Feature_{feat_idx}"
                names.append(f"{stock_label}::{suffix}::{label}")
    if include_prev_weights:
        for stock_idx in range(num_stocks):
            names.append(f"{get_ticker_name(stock_idx)}::PrevWeight")
    if extra_feature_labels:
        names.extend(extra_feature_labels)
    return names


def build_augmented_feature_labels(
    num_stocks: int,
    input_dim: int,
    *,
    tickers: Sequence[str] | None,
    corr_label: str,
    include_corr: bool,
    include_ts_stats: bool,
) -> List[str]:
    extra: List[str] = []

    def get_name(idx: int) -> str:
        if tickers and idx < len(tickers):
            return str(tickers[idx])
        return f"Stock[{idx}]"

    if include_corr:
        for i in range(num_stocks):
            name_i = get_name(i)
            for j in range(num_stocks):
                name_j = get_name(j)
                extra.append(f"{corr_label}[{name_i},{name_j}]")

    if include_ts_stats:
        stat_labels = ("TSMean", "TSStd")
        for stat in stat_labels:
            for stock_idx in range(num_stocks):
                stock_name = get_name(stock_idx)
                for feat_idx in range(input_dim):
                    extra.append(f"{stock_name}::{stat}_{feat_idx}")

    return extra


def _extract_augmented_features(env: StockPortfolioEnv, args: argparse.Namespace) -> np.ndarray | None:
    extras: List[np.ndarray] = []
    step = getattr(env, "current_step", 0)

    if args.augment_corr and getattr(env, "corr_tensor", None) is not None:
        corr_slice = env.corr_tensor[step]
        extras.append(_to_numpy(corr_slice).reshape(-1))

    if args.augment_ts_stats and getattr(env, "ts_features_tensor", None) is not None:
        ts_slice = _to_numpy(env.ts_features_tensor[step])
        if ts_slice.ndim >= 3:
            # Expect [num_stocks, lookback, features]
            ts_mean = np.nan_to_num(ts_slice.mean(axis=1))
            ts_std = np.nan_to_num(ts_slice.std(axis=1))
            extras.append(ts_mean.reshape(-1))
            extras.append(ts_std.reshape(-1))

    if not extras:
        return None
    return np.concatenate(extras, axis=0)


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exp_values = np.exp(shifted)
    denom = exp_values.sum()
    return exp_values / denom if denom != 0 else np.zeros_like(exp_values)


def collect_trajectories(
    loader: DataLoader,
    model: PPO,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    observations: List[np.ndarray] = []
    weights: List[np.ndarray] = []

    for batch_idx, batch in enumerate(loader):
        print(f"Processing batch {batch_idx + 1}/{len(loader)}")
        corr, ts_features, features, ind, pos, neg, returns, pyg_data, mask = process_data(batch, device)

        env = StockPortfolioEnv(
            args=args,
            corr=corr,
            ts_features=ts_features,
            features=features,
            ind=ind,
            pos=pos,
            neg=neg,
            returns=returns,
            pyg_data=pyg_data,
            mode="test",
            ind_yn=args.ind_yn,
            pos_yn=args.pos_yn,
            neg_yn=args.neg_yn,
            reward_net=None,
            device=str(device),
        )
        env.seed(args.seed)
        vec_env, obs = env.get_sb_env()
        vec_env.reset()

        max_steps = returns.shape[0] if hasattr(returns, "shape") else len(returns)
        # Allow an explicit cap on the number of timesteps processed. This protects
        # against accidentally loading the entire dataset when date slicing fails
        # or when a smaller window is desired for quick explainability runs.
        if getattr(args, "max_steps", None):
            try:
                cap = int(args.max_steps)
                if cap > 0:
                    max_steps = min(int(max_steps), cap)
            except Exception:
                # ignore invalid values and use detected max_steps
                pass
        print(f"Collecting trajectories: limiting to {int(max_steps)} steps (detected full={returns.shape[0] if hasattr(returns,'shape') else len(returns)})")
        for step in range(int(max_steps)):
            action, _ = model.predict(obs, deterministic=args.deterministic)
            if isinstance(obs, np.ndarray) and obs.ndim == 2 and obs.shape[0] == 1:
                obs_sample = obs[0].copy()
            else:
                obs_sample = np.asarray(obs).reshape(-1).copy()

            extra_features = _extract_augmented_features(env, args)
            if extra_features is not None:
                obs_sample = np.concatenate([obs_sample, extra_features], axis=0)

            observations.append(obs_sample)
            weights.append(softmax(np.asarray(action).reshape(-1)))

            obs, rewards, dones, _info = vec_env.step(action)
            if dones[0]:
                break

        vec_env.close()

    X = np.vstack(observations)
    Y = np.vstack(weights)
    return X, Y


def train_multi_output_tree(X: np.ndarray, Y: np.ndarray, depth: int, random_state: int) -> DecisionTreeRegressor:
    tree = DecisionTreeRegressor(max_depth=depth, random_state=random_state)
    tree.fit(X, Y)
    return tree


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain RL portfolio allocations with tree surrogates")
    parser.add_argument("--model-path", required=True, help="Path to trained PPO .zip file")
    parser.add_argument("--market", default="hs300", help="Market code used when generating the dataset")
    parser.add_argument("--horizon", default="1", help="Prediction horizon subdirectory")
    parser.add_argument("--relation-type", default="hy", help="Relation type subdirectory (e.g. hy)")
    parser.add_argument("--test-start-date", required=True, help="Test range start (YYYY-MM-DD)")
    parser.add_argument("--test-end-date", required=True, help="Test range end (YYYY-MM-DD)")
    parser.add_argument("--data-root", default="dataset_default", help="Root directory that stores prepared datasets")
    parser.add_argument("--device", default="cpu", help="Torch device for loading tensors")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for environment seeding")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy predictions")
    parser.add_argument("--max-depth", type=int, default=5, help="Maximum depth for decision tree models")
    parser.add_argument("--top-k-stocks", type=int, default=5, help="Number of top-weight stocks to explain in detail")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for surrogate models")
    parser.add_argument("--output-dir", default="./explainability_results", help="Directory to store reports and models")
    parser.add_argument("--feature-names", default=None, help="Comma-separated feature labels (optional)")
    parser.add_argument("--ind-label", default="Industry", help="Label for industry relation matrix")
    parser.add_argument("--pos-label", default="Momentum", help="Label for positive relation matrix")
    parser.add_argument("--neg-label", default="Reversal", help="Label for negative relation matrix")
    parser.add_argument("--corr-label", default="Corr", help="Label for flattened correlation features")
    parser.add_argument("--ind-yn", action="store_true", help="Enable industry relation")
    parser.add_argument("--no-ind-yn", dest="ind_yn", action="store_false")
    parser.set_defaults(ind_yn=True)
    parser.add_argument("--pos-yn", action="store_true", help="Enable positive relation")
    parser.add_argument("--no-pos-yn", dest="pos_yn", action="store_false")
    parser.set_defaults(pos_yn=True)
    parser.add_argument("--neg-yn", action="store_true", help="Enable negative relation")
    parser.add_argument("--no-neg-yn", dest="neg_yn", action="store_false")
    parser.set_defaults(neg_yn=True)
    parser.add_argument("--augment-corr", dest="augment_corr", action="store_true", help="Append corr matrix to surrogate inputs")
    parser.add_argument("--no-augment-corr", dest="augment_corr", action="store_false")
    parser.set_defaults(augment_corr=True)
    parser.add_argument("--augment-ts-stats", dest="augment_ts_stats", action="store_true", help="Append per-stock time-series stats")
    parser.add_argument("--no-augment-ts-stats", dest="augment_ts_stats", action="store_false")
    parser.set_defaults(augment_ts_stats=True)
    parser.add_argument("--save-joblib", action="store_true", help="Persist fitted trees via joblib (enabled by default)")
    parser.add_argument("--no-save-joblib", dest="save_joblib", action="store_false", help="Disable joblib persistence")
    parser.set_defaults(save_joblib=True)

    parser.add_argument("--save-summary", action="store_true", help="Persist JSON summary report (enabled by default)")
    parser.add_argument("--no-save-summary", dest="save_summary", action="store_false", help="Disable JSON summary")
    parser.set_defaults(save_summary=True)
    parser.add_argument(
        "--tickers-csv",
        default="tickers.csv",
        help="CSV containing a 'ticker' column to map stock indices to symbols (defaults to repo tickers.csv)",
    )
    parser.add_argument(
        "--focus-log-csv",
        default=None,
        help="Optional CSV (same format as monthly log) to specify snapshot holdings",
    )
    parser.add_argument(
        "--focus-date",
        default=None,
        help="Snapshot date (YYYY-MM-DD) whose holdings should be prioritized",
    )
    parser.add_argument(
        "--focus-run-id",
        default=None,
        help="Optional run_id filter for focus log",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on the number of timesteps to collect from the test dataset (useful for quick runs)",
    )
    return parser.parse_args(argv)


def _load_tickers(csv_path: str, expected_count: int | None = None) -> Dict[str, object]:
    """Load ticker symbols from the provided CSV file."""

    candidates = [Path(csv_path).expanduser()]
    if not candidates[0].is_absolute():
        candidates.append((REPO_ROOT / csv_path).expanduser())

    resolved: Path | None = None
    for candidate in candidates:
        if candidate.exists():
            resolved = candidate
            break

    if resolved is None:
        print(f"[WARN] Ticker CSV not found for path '{csv_path}'. Stock names will use fallback labels.")
        return {"tickers": [], "source": csv_path}

    tickers: List[str] = []
    with resolved.open("r", encoding="utf-8") as handle:
        try:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            if "ticker" in fieldnames:
                for row in reader:
                    value = (row.get("ticker") or "").strip()
                    if value:
                        tickers.append(value)
            else:
                handle.seek(0)
                raw_reader = csv.reader(handle)
                for idx, row in enumerate(raw_reader):
                    if not row:
                        continue
                    value = (row[0] or "").strip()
                    if idx == 0 and value.lower() == "ticker":
                        continue
                    if value:
                        tickers.append(value)
        except csv.Error as exc:
            print(f"[WARN] Failed to parse tickers CSV '{resolved}': {exc}. Falling back to generic labels.")
            tickers = []

    if expected_count is not None and tickers and len(tickers) != expected_count:
        print(
            f"[WARN] Ticker count ({len(tickers)}) does not match num_stocks ({expected_count}). "
            "Indices beyond the provided list will use generic labels."
        )

    return {"tickers": tickers, "source": str(resolved)}


def _ticker_for_index(index: int, tickers_info: Dict[str, object]) -> str:
    tickers_raw = tickers_info.get("tickers")
    tickers_list = tickers_raw if isinstance(tickers_raw, list) else []
    if 0 <= index < len(tickers_list):
        return str(tickers_list[index])
    return f"Stock_{index}"


def _resolve_path(path_str: str) -> Path:
    candidate = Path(path_str).expanduser()
    if candidate.exists():
        return candidate
    alt = (REPO_ROOT / path_str).expanduser()
    return alt if alt.exists() else candidate


def _focus_tickers_from_log(
    csv_path: str | None,
    focus_date: str | None,
    *,
    top_k: int,
    run_id: str | None = None,
) -> List[str]:
    if not csv_path or not focus_date:
        return []
    resolved = _resolve_path(csv_path)
    if not resolved.exists():
        print(f"[WARN] focus log CSV not found at {resolved}. Falling back to avg weights.")
        return []
    try:
        df = pd.read_csv(resolved)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to read focus log CSV {resolved}: {exc}. Falling back to avg weights.")
        return []

    df.columns = [str(col).strip().lower() for col in df.columns]
    if run_id and "run_id" in df.columns:
        df = df[df["run_id"] == run_id]
    if df.empty:
        print("[WARN] focus log CSV is empty after run_id filter.")
        return []
    date_cols = [col for col in ("as_of", "date") if col in df.columns]
    if not date_cols:
        print("[WARN] focus log CSV missing 'as_of' or 'date' column.")
        return []
    target = pd.to_datetime(focus_date).date()
    filtered = None
    for col in date_cols:
        series = pd.to_datetime(df[col], errors="coerce")
        mask = series.dt.date == target
        if mask.any():
            filtered = df[mask].copy()
            filtered["as_of"] = series[mask].dt.strftime("%Y-%m-%d")
            break
    if filtered is None or filtered.empty:
        print(f"[WARN] No rows found in focus log for date {focus_date}.")
        return []
    if "weight" not in filtered.columns:
        print("[WARN] focus log CSV missing 'weight' column.")
        return []
    filtered = filtered.sort_values("weight", ascending=False).head(top_k)
    focus_symbols = [str(val).strip().upper() for val in filtered.get("ticker", []) if str(val).strip()]
    return focus_symbols


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    metadata = auto_detect_metadata(args)
    args.num_stocks = metadata.num_stocks
    args.input_dim = metadata.input_dim
    args.lookback = int(getattr(metadata, "lookback", 0) or 0)
    if args.lookback <= 0:
        raise ValueError("Failed to infer a positive lookback window from dataset metadata")

    if args.feature_names:
        feature_labels = [name.strip() for name in args.feature_names.split(",") if name.strip()]
        if feature_labels and len(feature_labels) != args.input_dim:
            raise ValueError(
                f"feature-names count {len(feature_labels)} does not match inferred input_dim {args.input_dim}"
            )
    else:
        # Defaults based on build_dataset_yf.py
        if args.input_dim == 6:
            feature_labels = ["Close", "Open", "High", "Low", "Prev_Close", "Volume"]
        else:
            feature_labels = [f"Feature_{idx}" for idx in range(args.input_dim)]

    relation_labels = [args.ind_label, args.pos_label, args.neg_label]

    tickers = _load_tickers(args.tickers_csv, expected_count=metadata.num_stocks)

    print("Configuration:")
    print(f"  Model path     : {args.model_path}")
    print(f"  Dataset dir    : {metadata.data_dir}")
    print(f"  Market         : {args.market}")
    print(f"  Num stocks     : {args.num_stocks}")
    print(f"  Feature dim    : {args.input_dim}")
    print(f"  Lookback       : {args.lookback}")
    if tickers:
        print(f"  Tickers CSV    : {tickers['source']}")
    print(f"  Test range     : {args.test_start_date} to {args.test_end_date}")

    test_dataset = AllGraphDataSampler(
        base_dir=str(metadata.data_dir),
        date=True,
        test_start_date=args.test_start_date,
        test_end_date=args.test_end_date,
        mode="test",
    )
    if len(test_dataset) == 0:
        raise RuntimeError("Test dataset is empty for the specified date range")
    resolved_span = getattr(test_dataset, "gnames_all", [])
    if resolved_span:
        print(
            f"Resolved test slice: {resolved_span[0]} → {resolved_span[-1]} "
            f"({len(resolved_span)} files)"
        )

    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), pin_memory=True)

    print("Loading PPO model …")
    model = PPO.load(args.model_path, env=None, device=args.device)

    X, Y = collect_trajectories(test_loader, model, args, torch.device(args.device))
    print(f"Collected {X.shape[0]} timesteps with observation dimension {X.shape[1]}")

    surrogate = train_multi_output_tree(X, Y, depth=args.max_depth, random_state=args.random_state)
    Y_hat = surrogate.predict(X)
    fidelity = r2_score(Y, Y_hat, multioutput="uniform_average")
    print(f"Global surrogate R²: {fidelity:.4f}")

    # Build feature names with ticker mapping
    tickers_list = tickers.get("tickers") if tickers else None
    extra_feature_labels = build_augmented_feature_labels(
        args.num_stocks,
        args.input_dim,
        tickers=tickers_list,
        corr_label=args.corr_label,
        include_corr=args.augment_corr,
        include_ts_stats=args.augment_ts_stats,
    )

    feature_names = build_feature_names(
        args.num_stocks,
        args.input_dim,
        args.lookback,
        feature_labels,
        relation_labels,
        tickers=tickers_list,
        extra_feature_labels=extra_feature_labels,
    )

    ticker_map = {}
    if tickers_list:
        ticker_map = {str(sym).strip().upper(): idx for idx, sym in enumerate(tickers_list) if str(sym).strip()}
    focus_symbols = _focus_tickers_from_log(
        args.focus_log_csv,
        args.focus_date,
        top_k=args.top_k_stocks,
        run_id=args.focus_run_id,
    )
    focus_indices: List[int] = []
    if focus_symbols and not ticker_map:
        print("[WARN] Focus tickers supplied but ticker CSV is missing; unable to map symbols to indices.")
    for symbol in focus_symbols:
        idx = ticker_map.get(symbol)
        if idx is None:
            print(f"[WARN] Focus ticker {symbol} not found in dataset mapping; skipping.")
            continue
        if idx not in focus_indices:
            focus_indices.append(idx)

    avg_weights = Y.mean(axis=0)
    default_indices = np.argsort(avg_weights)[::-1][: args.top_k_stocks].tolist()
    target_indices = focus_indices if focus_indices else default_indices

    per_stock_rules: Dict[int, Dict[str, object]] = {}
    for rank, stock_idx in enumerate(target_indices, start=1):
        stock_label = _ticker_for_index(stock_idx, tickers)
        tree = DecisionTreeRegressor(max_depth=args.max_depth, random_state=args.random_state)
        tree.fit(X, Y[:, stock_idx])
        stock_r2 = r2_score(Y[:, stock_idx], tree.predict(X))
        importances = tree.feature_importances_
        top_features = np.argsort(importances)[::-1][:10]
        
        # Export text uses the updated feature_names containing tickers
        readable_rules = export_text(tree, feature_names=feature_names, max_depth=args.max_depth)

        print("\n" + "=" * 80)
        print(f"Rank {rank}: Stock {stock_idx} ({stock_label})")
        # print(f"Average weight: {avg_weights[stock_idx]:.4%}")  # Removed as requested
        print("Top feature contributions:")
        for idx in top_features:
            if importances[idx] <= 0:
                continue
            print(f"  - {feature_names[idx]}: {importances[idx]:.4f}")
        print("Decision rules:\n" + readable_rules)

        feature_payload = [
            {
                "name": feature_names[idx],
                "importance": float(importances[idx]),
            }
            for idx in top_features
            if importances[idx] > 0
        ]
        per_stock_rules[stock_idx] = {
            "ticker": stock_label,
            "top_features": feature_payload,
            "rules": readable_rules,
        }

    selected_tickers = [_ticker_for_index(idx, tickers) for idx in target_indices]

    output_dir = _ensure_dir(Path(args.output_dir).expanduser().resolve())

    if args.save_joblib:
        artifact_path = output_dir / f"explain_tree_{args.market}.joblib"
        joblib.dump(
            {
                "surrogate": surrogate,
                "per_stock": per_stock_rules,
                "feature_names": feature_names,
                "avg_weights": avg_weights,
                "top_indices": target_indices,
                "selected_tickers": selected_tickers,
                "X_shape": X.shape,
                "Y_shape": Y.shape,
                "global_r2": fidelity,
            },
            artifact_path,
        )
        print(f"Saved joblib artifact to {artifact_path}")

    if args.save_summary:
        summary_path = output_dir / f"explain_tree_{args.market}.json"
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "model_path": args.model_path,
                    "dataset_dir": str(metadata.data_dir),
                    "market": args.market,
                    "num_stocks": args.num_stocks,
                    "input_dim": args.input_dim,
                    "global_r2": fidelity,
                    "top_indices": target_indices,
                    "avg_weights": [float(avg_weights[idx]) for idx in target_indices],
                    "top_tickers": selected_tickers,
                    "focus_source": {
                        "csv": args.focus_log_csv,
                        "date": args.focus_date,
                        "run_id": args.focus_run_id,
                    },
                },
                fh,
                indent=2,
            )
        print(f"Saved summary JSON to {summary_path}")


if __name__ == "__main__":
    main()
