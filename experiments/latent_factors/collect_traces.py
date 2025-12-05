#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import pickle

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from stable_baselines3 import PPO

# Adjust path to find repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.portfolio_env import StockPortfolioEnv
from dataloader.data_loader import AllGraphDataSampler


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _call_policy_with_attention(policy_net, features_tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor] | None]:
    output = policy_net(features_tensor, require_weights=True)
    if isinstance(output, tuple):
        if len(output) >= 2:
            return output[0], output[1]
        if len(output) == 1:
            return output[0], None
        raise RuntimeError("policy_net returned an empty tuple; cannot proceed")
    return output, None


def process_data(batch: Dict[str, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, ...]:
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


def auto_detect_metadata(args: argparse.Namespace) -> Dict[str, object]:
    """Infer dataset directory, number of stocks, and feature dimension."""
    base_dir = Path(args.data_root).expanduser().resolve()
    # UPDATED: Use args.market to construct path
    data_dir = base_dir / f"data_train_predict_{args.market}" / f"{args.horizon}_{args.relation_type}"
    if not data_dir.is_dir():
        # Fallback for legacy folder names if needed
        alt_dir = base_dir / "data_train_predict_custom" / f"{args.horizon}_{args.relation_type}"
        if alt_dir.is_dir():
            data_dir = alt_dir
        else:
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
                features_np = features.cpu().numpy()
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

            return {"data_dir": data_dir, "num_stocks": num_stocks, "input_dim": input_dim}
        except Exception as exc:
            last_error = exc
            continue

    raise RuntimeError(
        "Unable to infer metadata from dataset. Last error: "
        f"{last_error}" if last_error else "no usable sample files"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect PPO traces for latent factor experiments")
    parser.add_argument("--model-path", required=True, help="Path to trained PPO .zip file")
    parser.add_argument("--market", default="custom", help="Market name (e.g. hs300, custom)")
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
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "experiments" / "latent_factors" / "data"),
        help="Directory to store trace artefacts",
    )
    parser.add_argument("--output-name", default="traces", help="Base filename (without extension) for outputs")
    parser.add_argument("--save-attention", action="store_true", help="Persist attention weights per step (can be large)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    meta = auto_detect_metadata(args)

    args.num_stocks = meta["num_stocks"]
    args.input_dim = meta["input_dim"]

    output_dir = _ensure_dir(Path(args.output_dir).expanduser().resolve())
    data_path = output_dir / f"{args.output_name}.npz"
    meta_path = output_dir / f"{args.output_name}.json"

    print("Configuration:")
    print(f"  Model path  : {args.model_path}")
    print(f"  Market      : {args.market}")
    print(f"  Dataset dir : {meta['data_dir']}")
    print(f"  Num stocks  : {args.num_stocks}")
    print(f"  Feature dim : {args.input_dim}")
    print(f"  Test range  : {args.test_start_date} to {args.test_end_date}")

    test_dataset = AllGraphDataSampler(
        base_dir=str(meta["data_dir"]),
        date=True,
        test_start_date=args.test_start_date,
        test_end_date=args.test_end_date,
        mode="test",
    )
    if len(test_dataset) == 0:
        raise RuntimeError("Test dataset is empty for the specified date range")

    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), pin_memory=True)

    model_path = Path(args.model_path).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    print("Loading PPO model …")
    model = PPO.load(str(model_path), env=None, device=args.device)
    device = torch.device(args.device)
    captured_activations = {}
    def get_activation(name):
        def hook(model, input, output):
            captured_activations[name] = output.detach()
        return hook

    policy_net = model.policy.mlp_extractor.policy_net
    if hasattr(policy_net, "pn"):
        hook_handle = policy_net.pn.register_forward_hook(get_activation("embedding"))
        print(f"Hook registered on {policy_net.pn}")
    else:
        print("Warning: Could not find 'pn' layer in policy_net. Activations will not be collected.")
        hook_handle = None

    obs_buffer = []
    activations_buffer = []
    logits_buffer = []
    actions_buffer = []
    rewards_buffer = []
    dones_buffer = []
    attn_buffer = {"industry": [], "positive": [], "negative": [], "semantic": []}

    lookback_seen = None

    for batch_idx, batch in enumerate(test_loader):
        print(f"Processing batch {batch_idx + 1}/{len(test_loader)}")
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
        lookback_seen = getattr(env, "lookback", lookback_seen)

        max_steps = returns.shape[0] if hasattr(returns, "shape") else len(returns)
        step_limit = args.max_steps if args.max_steps is not None else max_steps

        for step in range(int(step_limit)):
            obs_tensor, _ = model.policy.obs_to_tensor(obs)
            
            with torch.no_grad():
                features_tensor = model.policy.extract_features(obs_tensor.to(device))
                logits, attn = _call_policy_with_attention(model.policy.mlp_extractor.policy_net, features_tensor)

            if "embedding" in captured_activations:
                activations_buffer.append(captured_activations["embedding"].cpu().numpy()[0])
            
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, rewards, dones, _info = vec_env.step(action)

            obs_buffer.append(np.asarray(obs)[0].copy())
            logits_buffer.append(logits.detach().cpu().numpy()[0])
            actions_buffer.append(np.asarray(action)[0].copy())
            rewards_buffer.append(float(rewards[0]))
            dones_buffer.append(bool(dones[0]))

            if args.save_attention and isinstance(attn, dict):
                for key in ("industry", "positive", "negative", "semantic"):
                    tensor = attn.get(key)
                    if tensor is None:
                        continue
                    attn_buffer[key].append(tensor.detach().cpu().numpy()[0])

            if dones[0]:
                break
        vec_env.close()
    
    if hook_handle:
        hook_handle.remove()

    obs_arr = np.stack(obs_buffer, axis=0).astype(np.float32) if obs_buffer else np.zeros((0,), dtype=np.float32)
    activations_arr = np.stack(activations_buffer, axis=0).astype(np.float32) if activations_buffer else np.zeros((0,), dtype=np.float32)
    logits_arr = np.stack(logits_buffer, axis=0).astype(np.float32) if logits_buffer else np.zeros((0,), dtype=np.float32)
    actions_arr = np.stack(actions_buffer, axis=0).astype(np.float32) if actions_buffer else np.zeros((0,), dtype=np.float32)
    rewards_arr = np.asarray(rewards_buffer, dtype=np.float32)
    dones_arr = np.asarray(dones_buffer, dtype=bool)

    payload = {
        "obs": obs_arr,
        "activations": activations_arr,
        "logits": logits_arr,
        "actions": actions_arr,
        "rewards": rewards_arr,
        "dones": dones_arr,
    }

    if args.save_attention:
        for key, values in attn_buffer.items():
            if values:
                payload[f"attn_{key}"] = np.stack(values, axis=0).astype(np.float32)

    np.savez_compressed(data_path, **payload)
    print(f"Saved traces to {data_path}")

    meta_payload = {
        "horizon": args.horizon,
        "relation_type": args.relation_type,
        "num_stocks": args.num_stocks,
        "input_dim": args.input_dim,
        "lookback": lookback_seen,
        "total_steps": int(obs_arr.shape[0]) if obs_arr.ndim >= 1 else 0,
        "model_path": args.model_path,
        "dataset_dir": str(meta["data_dir"]),
        "save_attention": args.save_attention,
    }
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta_payload, fh, indent=2)
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()