from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import TraceDataset
from .model import SparseAutoencoder, TopKSparseAutoencoder, JumpReLUSparseAutoencoder


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align latent factors to PPO logits with a linear head")
    parser.add_argument("--data-path", required=True, help="Path to traces NPZ")
    parser.add_argument("--checkpoint", required=True, help="Path to sparse_ae.pt")
    parser.add_argument("--output-dir", default="experiments/latent_factors/analysis", help="Where to store outputs")
    parser.add_argument("--ridge-lambda", type=float, default=1e-3, help="Ridge regularization strength")
    parser.add_argument("--device", default="cpu", help="Torch device")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for encoding")
    return parser.parse_args(argv)


def load_autoencoder(ckpt_path: Path, device: torch.device) -> Tuple[torch.nn.Module, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    input_dim = ckpt.get("input_dim")
    latent_dim = config.get("latent_dim")
    hidden_dim = config.get("hidden_dim", 512)
    num_hidden = config.get("num_hidden", 2)
    model_type = config.get("model_type", "l1")
    
    if input_dim is None or latent_dim is None:
        raise ValueError("Checkpoint missing input_dim or latent_dim")
    
    if model_type == "topk":
        k = config.get("k", 32)
        model = TopKSparseAutoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_hidden=num_hidden,
            k=k,
            tied_weights=config.get("tied_weights", False),
            normalize_decoder=config.get("normalize_decoder", True),
            activation=config.get("activation", "relu"),
        )
    elif model_type == "jumprelu":
        model = JumpReLUSparseAutoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_hidden=num_hidden,
        )
    else:
        model = SparseAutoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_hidden=num_hidden,
        )
    
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, {"config": config, "model_type": model_type}


def fit_ridge(Z: np.ndarray, Y: np.ndarray, lam: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form ridge regression: Predicts Logits (Y) from Latents (Z)."""
    Z = Z.astype(np.float64)
    Y = Y.astype(np.float64)
    
    z_std = Z.std(axis=0)
    active_mask = z_std > 1e-6
    n_active = active_mask.sum()
    print(f"Active latents: {n_active}/{Z.shape[1]}")
    
    if n_active == 0:
        return np.zeros_like(Y), np.zeros((Z.shape[1] + 1, Y.shape[1])), active_mask
    
    Z_active = Z[:, active_mask]
    
    # Standardize
    z_mean = Z_active.mean(axis=0, keepdims=True)
    z_std_active = Z_active.std(axis=0, keepdims=True) + 1e-8
    Z_norm = (Z_active - z_mean) / z_std_active
    
    ones = np.ones((Z_norm.shape[0], 1), dtype=np.float64)
    Zb = np.concatenate([Z_norm, ones], axis=1)
    
    dim = Zb.shape[1]
    A = Zb.T @ Zb + lam * np.eye(dim, dtype=np.float64)
    B = Zb.T @ Y
    
    try:
        W_active = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        W_active = np.linalg.lstsq(A, B, rcond=None)[0]
    
    preds = Zb @ W_active
    
    W_full = np.zeros((Z.shape[1] + 1, Y.shape[1]), dtype=np.float64)
    active_indices = np.where(active_mask)[0]
    W_full[active_indices, :] = W_active[:-1]
    W_full[-1, :] = W_active[-1]
    
    return preds, W_full, active_mask


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = TraceDataset(args.data_path)
    if dataset.logits is None:
        raise ValueError("Dataset does not contain 'logits' for alignment.")

    device = torch.device(args.device)
    model, meta = load_autoencoder(Path(args.checkpoint).expanduser(), device)
    model_type = meta.get("model_type", "l1")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    latents = []
    targets = []
    
    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device)
            target = batch["logits"] 
            if model_type in ("topk", "jumprelu"):
                output = model(x)
                z = output[1]
            else:
                output = model(x)
                z = output[1] 
            
            latents.append(z.cpu().numpy())
            targets.append(target.numpy().squeeze(1)) # Flatten to [batch]

    Z = np.concatenate(latents, axis=0)
    Y = np.concatenate(targets, axis=0)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    print(f"Fitting Ridge Regression: Z{Z.shape} -> Y{Y.shape}")
    preds, weights, active_mask = fit_ridge(Z, Y, args.ridge_lambda)
    
    r2 = r2_score(Y, preds)
    mse = np.mean((Y - preds) ** 2, axis=0)

    metrics = {
        "ridge_lambda": args.ridge_lambda,
        "r2_mean": float(np.mean(r2)),
        "mse_mean": float(np.mean(mse)),
        "active_latents": int(active_mask.sum()),
    }
    
    print(f"Alignment R2: {metrics['r2_mean']:.4f}")
    
    with open(output_dir / "alignment_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    np.savez_compressed(
        output_dir / "alignment_outputs.npz",
        latents=Z,
        logits=Y,
        preds=preds,
        weights=weights,
        active_mask=active_mask,
    )
    print(f"Saved alignment outputs to {output_dir / 'alignment_outputs.npz'}")


if __name__ == "__main__":
    main()