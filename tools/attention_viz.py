#!/usr/bin/env python3
"""Collect and summarise HGAT attention patterns.

This utility replays a trained PPO policy on the custom ``StockPortfolioEnv``
and records every attention tensor emitted by the HGAT backbone. The resulting
statistics can be used to build heatmaps or temporal plots that explain how the
model routes information between stocks.

Example
-------
python tools/attention_viz.py \
    --model-path ./checkpoints/ppo_hgat_custom_20251108_103925.zip \
    --market custom \
    --test-start-date 2024-01-02 \
    --test-end-date 2024-12-26 \
    --output-dir ./explainability_results \
    --save-raw --plot
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.loader import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.portfolio_env import StockPortfolioEnv
from dataloader.data_loader import AllGraphDataSampler
from stable_baselines3 import PPO


@dataclass
class DatasetMetadata:
    """Container for dataset descriptors detected at runtime."""

    data_dir: Path
    num_stocks: int
    input_dim: int


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def auto_detect_metadata(args: argparse.Namespace) -> DatasetMetadata:
    """Infer dataset directory, number of stocks, and feature dimension."""
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

            return DatasetMetadata(data_dir=data_dir, num_stocks=num_stocks, input_dim=input_dim)
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


def top_edges(matrix: np.ndarray, k: int = 5) -> List[Tuple[int, int, float]]:
    """Return the top-K directed edges by average attention weight."""
    flat = matrix.reshape(-1)
    if flat.size == 0:
        return []
    k = min(k, flat.size)
    indices = np.argpartition(flat, -k)[-k:]
    sorted_idx = indices[np.argsort(flat[indices])[::-1]]
    size = matrix.shape[0]
    results: List[Tuple[int, int, float]] = []
    for idx in sorted_idx:
        row = int(idx // size)
        col = int(idx % size)
        results.append((row, col, float(matrix[row, col])))
    return results


def collect_attention(
    loader: DataLoader,
    model: PPO,
    args: argparse.Namespace,
    device: torch.device,
    component_labels: Sequence[str],
) -> Dict[str, np.ndarray]:
    relation_history: Dict[str, List[np.ndarray]] = {"industry": [], "positive": [], "negative": []}
    semantic_history: List[np.ndarray] = []
    allocation_history: List[np.ndarray] = []

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
        step_limit = args.max_steps if args.max_steps is not None else max_steps

        for step in range(int(step_limit)):
            obs_tensor, _ = model.policy.obs_to_tensor(obs)
            with torch.no_grad():
                features_tensor = model.policy.extract_features(obs_tensor.to(device))
                logits, attn = model.policy.mlp_extractor.policy_net(features_tensor, require_weights=True)
            allocation_history.append(logits.detach().cpu().numpy()[0])

            for key in ("industry", "positive", "negative"):
                weights = attn.get(key)
                if weights is None:
                    continue
                relation_history[key].append(weights.detach().cpu().numpy()[0])

            semantic = attn.get("semantic")
            if semantic is not None:
                semantic_history.append(semantic.detach().cpu().numpy()[0])

            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, rewards, dones, _info = vec_env.step(action)
            if dones[0]:
                break

        vec_env.close()

    def stack_or_empty(values: List[np.ndarray], expected_dims: int) -> np.ndarray:
        if not values:
            return np.zeros((0,) * expected_dims, dtype=np.float32)
        return np.stack(values, axis=0).astype(np.float32)

    return {
        "industry": stack_or_empty(relation_history["industry"], 4),
        "positive": stack_or_empty(relation_history["positive"], 4),
        "negative": stack_or_empty(relation_history["negative"], 4),
        "semantic": stack_or_empty(semantic_history, 3),
        "allocations": stack_or_empty(allocation_history, 2),
        "component_labels": np.asarray(component_labels),
    }


def summarise_attention(attention: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    summary: Dict[str, np.ndarray] = {}
    for key in ("industry", "positive", "negative"):
        tensor = attention[key]
        if tensor.size == 0:
            continue
        # Average over time, then heads
        mean_matrix = tensor.mean(axis=0).mean(axis=0)
        summary[f"{key}_mean"] = mean_matrix
        summary[f"{key}_per_head"] = tensor.mean(axis=0)
    semantic = attention.get("semantic")
    if semantic is not None and semantic.size > 0:
        semantic_mean = semantic.mean(axis=0)
        if semantic_mean.ndim >= 3:
            semantic_mean = semantic_mean.mean(axis=-1)
        if semantic_mean.ndim >= 2:
            semantic_mean = semantic_mean.mean(axis=-1)
        summary["semantic_mean"] = semantic_mean
    allocations = attention.get("allocations")
    if allocations is not None and allocations.size > 0:
        summary["avg_allocations"] = allocations.mean(axis=0)
    return summary


def try_plot(attention: Dict[str, np.ndarray], summary: Dict[str, np.ndarray], output_dir: Path, labels: Sequence[str]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: PERF203
        print(f"Plotting skipped (matplotlib unavailable: {exc})")
        return

    cmap = "viridis"
    for key in ("industry", "positive", "negative"):
        mean_matrix = summary.get(f"{key}_mean")
        if mean_matrix is None or mean_matrix.size == 0:
            continue
        plt.figure(figsize=(6, 5))
        plt.imshow(mean_matrix, cmap=cmap)
        plt.colorbar(label="Average attention")
        plt.title(f"HGAT {key} attention (avg over time & heads)")
        plt.xlabel("Destination stock")
        plt.ylabel("Source stock")
        plt.tight_layout()
        plot_path = output_dir / f"attention_{key}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved heatmap to {plot_path}")

    semantic = summary.get("semantic_mean")
    if semantic is not None and semantic.size > 0:
        plt.figure(figsize=(6, 3))
        plt.bar(np.arange(len(semantic)), semantic)
        plt.xticks(np.arange(len(semantic)), labels[: len(semantic)], rotation=45, ha="right")
        plt.ylabel("Average semantic weight")
        plt.title("HGAT semantic fusion weights")
        plt.tight_layout()
        plot_path = output_dir / "attention_semantic.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved semantic weights plot to {plot_path}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect HGAT attention tensors for interpretability")
    parser.add_argument("--model-path", required=True, help="Path to trained PPO .zip file")
    parser.add_argument("--market", default="hs300", help="Market code used when generating the dataset")
    parser.add_argument("--horizon", default="1", help="Prediction horizon subdirectory")
    parser.add_argument("--relation-type", default="hy", help="Relation type subdirectory (e.g. hy)")
    parser.add_argument("--test-start-date", required=True, help="Test range start (YYYY-MM-DD)")
    parser.add_argument("--test-end-date", required=True, help="Test range end (YYYY-MM-DD)")
    parser.add_argument("--data-root", default="dataset_default", help="Root directory storing prepared datasets")
    parser.add_argument("--device", default="cpu", help="Torch device for loading tensors")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for environment seeding")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy predictions")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap on rollout steps per batch")
    parser.add_argument("--ind-yn", action="store_true", help="Enable industry relation")
    parser.add_argument("--no-ind-yn", dest="ind_yn", action="store_false")
    parser.set_defaults(ind_yn=True)
    parser.add_argument("--pos-yn", action="store_true", help="Enable positive relation")
    parser.add_argument("--no-pos-yn", dest="pos_yn", action="store_false")
    parser.set_defaults(pos_yn=True)
    parser.add_argument("--neg-yn", action="store_true", help="Enable negative relation")
    parser.add_argument("--no-neg-yn", dest="neg_yn", action="store_false")
    parser.set_defaults(neg_yn=True)
    parser.add_argument("--output-dir", default="./explainability_results", help="Directory to store artefacts")
    parser.add_argument("--save-raw", action="store_true", help="Persist raw attention tensors (.npz)")
    parser.add_argument("--save-summary", action="store_true", help="Persist JSON summary")
    parser.add_argument("--plot", action="store_true", help="Generate heatmaps (requires matplotlib)")
    parser.add_argument("--top-k-edges", type=int, default=5, help="Number of strongest edges to report")
    parser.add_argument(
        "--tickers-csv",
        default="tickers.csv",
        help="CSV containing a 'ticker' column to map stock indices to symbols (defaults to repo tickers.csv)",
    )
    return parser.parse_args(argv)


def _load_tickers(csv_path: str, expected_count: int | None = None) -> Dict[str, object]:
    candidates = [Path(csv_path).expanduser()]
    if not candidates[0].is_absolute():
        candidates.append((REPO_ROOT / csv_path).expanduser())

    resolved: Path | None = None
    for candidate in candidates:
        if candidate.exists():
            resolved = candidate
            break

    if resolved is None:
        print(f"[WARN] Ticker CSV not found for path '{csv_path}'. Edge summaries will use generic names.")
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
            print(f"[WARN] Failed to parse tickers CSV '{resolved}': {exc}. Using generic labels instead.")
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


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    metadata = auto_detect_metadata(args)
    args.num_stocks = metadata.num_stocks
    args.input_dim = metadata.input_dim

    tickers = _load_tickers(args.tickers_csv, expected_count=metadata.num_stocks)

    print("Configuration:")
    print(f"  Model path     : {args.model_path}")
    print(f"  Dataset dir    : {metadata.data_dir}")
    print(f"  Market         : {args.market}")
    print(f"  Num stocks     : {args.num_stocks}")
    print(f"  Feature dim    : {args.input_dim}")
    if tickers.get("source"):
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

    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), pin_memory=True)

    model_path = Path(args.model_path).expanduser()
    if model_path.is_dir():
        raise ValueError(f"Model path {model_path} is a directory; provide a specific checkpoint file")
    if not model_path.exists():
        alternate = model_path.with_suffix("") if model_path.suffix == ".zip" else model_path.with_suffix(".zip")
        hint = alternate if alternate.exists() else None
        message = f"Checkpoint not found: {model_path}"
        if hint:
            message += f". Did you mean {hint}?"
        raise FileNotFoundError(message)

    print("Loading PPO model …")
    model = PPO.load(str(model_path), env=None, device=args.device)

    hgat_policy = model.policy.mlp_extractor.policy_net
    component_labels = ["Self", "Industry", "Positive", "Negative"]
    if getattr(hgat_policy, "no_ind", False):
        component_labels = ["Self", "Negative"]
    elif getattr(hgat_policy, "no_neg", False):
        component_labels = ["Self", "Industry"]

    attention = collect_attention(test_loader, model, args, torch.device(args.device), component_labels)
    summary = summarise_attention(attention)

    output_dir = _ensure_dir(Path(args.output_dir).expanduser().resolve())

    print("\nAverage relation attention (time & head averaged):")
    edge_summaries: Dict[str, List[Tuple[int, int, float]]] = {}
    for key in ("industry", "positive", "negative"):
        matrix = summary.get(f"{key}_mean")
        if matrix is None or matrix.size == 0:
            continue
        edges = top_edges(matrix, k=args.top_k_edges)
        edge_summaries[key] = edges
        print(f"- {key.title()} strongest edges:")
        for src, dst, value in edges:
            src_label = _ticker_for_index(src, tickers)
            dst_label = _ticker_for_index(dst, tickers)
            print(f"    {src:>3} ({src_label}) -> {dst:>3} ({dst_label}): {value:.4f}")

    semantic = summary.get("semantic_mean")
    if semantic is not None and semantic.size > 0:
        print("\nAverage semantic fusion weights:")
        for label, value in zip(component_labels, semantic):
            print(f"- {label:>8}: {value:.4f}")

    if args.save_summary:
        summary_path = output_dir / f"attention_summary_{args.market}.json"
        top_edges_payload: Dict[str, List[Dict[str, object]]] = {}
        for key, edges in edge_summaries.items():
            formatted: List[Dict[str, object]] = []
            for src, dst, val in edges:
                formatted.append(
                    {
                        "source_index": int(src),
                        "source_ticker": _ticker_for_index(src, tickers),
                        "target_index": int(dst),
                        "target_ticker": _ticker_for_index(dst, tickers),
                        "mean_attention": float(val),
                    }
                )
            top_edges_payload[key] = formatted
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "model_path": args.model_path,
                    "dataset_dir": str(metadata.data_dir),
                    "market": args.market,
                    "num_stocks": args.num_stocks,
                    "tickers": tickers.get("tickers", []),
                    "tickers_source": tickers.get("source"),
                    "semantic_labels": component_labels,
                    "semantic_mean": summary.get("semantic_mean", []).tolist() if "semantic_mean" in summary else [],
                    "avg_allocations": summary.get("avg_allocations", []).tolist() if "avg_allocations" in summary else [],
                    "top_edges": top_edges_payload,
                },
                fh,
                indent=2,
            )
        print(f"Saved summary JSON to {summary_path}")

    if args.save_raw:
        raw_path = output_dir / f"attention_tensors_{args.market}.npz"
        np.savez_compressed(
            raw_path,
            industry=attention["industry"],
            positive=attention["positive"],
            negative=attention["negative"],
            semantic=attention["semantic"],
            allocations=attention["allocations"],
            semantic_labels=attention["component_labels"],
        )
        print(f"Saved raw attention tensors to {raw_path}")

    if args.plot:
        try_plot(attention, summary, output_dir, component_labels)


if __name__ == "__main__":
    main()
