from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import TraceDataset
from .align_factors import load_autoencoder


def parse_args(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(description="Extract and interpret features from a trained Sparse Autoencoder")
    p.add_argument("--checkpoint", required=True, help="Path to SAE checkpoint (.pt)")
    p.add_argument("--data-path", required=True, help="Path to traces NPZ (same used for training)")
    p.add_argument("--output-dir", default="experiments/latent_factors/analysis/feature_extraction", help="Output directory")
    p.add_argument("--device", default="cpu", help="Torch device")
    p.add_argument("--batch-size", type=int, default=256, help="Batch size for encoding")
    p.add_argument("--top-k", type=int, default=10, help="Top-k input features to report per latent")
    p.add_argument("--num-samples", type=int, default=256, help="Number of input samples to use for linearization (keeps runtime manageable)")
    p.add_argument("--eps", type=float, default=1e-3, help="Finite-difference epsilon for empirical Jacobian")
    p.add_argument("--tickers-path", default="tickers.csv", help="Path to tickers CSV for stock names")
    return p.parse_args(argv)


def build_feature_names(obs_dim: int, num_stocks: int, lookback: int, feat_dim: int) -> list:
    """Build human-readable feature names for observation vector.
    
    Observation structure:
    - ind_matrix: [num_stocks * num_stocks] - industry adjacency
    - pos_matrix: [num_stocks * num_stocks] - positive correlation
    - neg_matrix: [num_stocks * num_stocks] - negative correlation  
    - ts_features: [num_stocks * lookback * feat_dim] - time-series
    - prev_weights: [num_stocks] - previous portfolio weights
    """
    names = []
    
    # Adjacency matrices (3 types)
    for i in range(num_stocks):
        for j in range(num_stocks):
            names.append(f"ind_adj[{i},{j}]")
    for i in range(num_stocks):
        for j in range(num_stocks):
            names.append(f"pos_adj[{i},{j}]")
    for i in range(num_stocks):
        for j in range(num_stocks):
            names.append(f"neg_adj[{i},{j}]")
    
    # Time-series features
    ts_feature_names = ["open", "high", "low", "close", "volume", "returns"]  # Common features
    for stock_idx in range(num_stocks):
        for t in range(lookback):
            for f in range(feat_dim):
                feat_name = ts_feature_names[f] if f < len(ts_feature_names) else f"feat_{f}"
                names.append(f"stock_{stock_idx}_t-{lookback-t}_{feat_name}")
    
    # Previous weights
    for i in range(num_stocks):
        names.append(f"prev_weight[{i}]")
    
    return names


def load_tickers(path: str) -> dict:
    """Load ticker names from CSV."""
    import csv
    ticker_map = {}
    try:
        # Handle relative path resolution
        resolved_path = Path(path).expanduser()
        if not resolved_path.exists():
             # Fallback to looking in current or parent dirs if not found
             resolved_path = Path(__file__).parents[2] / path
        
        if resolved_path.exists():
            with open(resolved_path, 'r') as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader):
                    if 'ticker' in row:
                        ticker_map[idx] = row['ticker']
                    elif 'symbol' in row:
                        ticker_map[idx] = row['symbol']
    except Exception as e:
        print(f"Warning: Could not load tickers from {path}: {e}")
    return ticker_map


def compute_empirical_jacobian(model, zs: torch.Tensor, eps: float, device: torch.device) -> np.ndarray:
    """Compute empirical Jacobian of decoder: recon (B, D) vs latent z (B, L).
    Returns mean absolute derivative per latent: shape (D, L) averaged over batch.
    """
    model.eval()
    zs = zs.to(device)
    with torch.no_grad():
        base_recon = model.decode(zs).cpu().numpy()  # (B, D)

    B, L = zs.shape
    D = base_recon.shape[1]
    jac = np.zeros((D, L), dtype=np.float64)

    # We'll perturb one latent at a time to approximate partial derivatives
    for j in range(L):
        zs_p = zs.clone()
        zs_p[:, j] = zs_p[:, j] + eps
        with torch.no_grad():
            recon_p = model.decode(zs_p).cpu().numpy()
        # derivative approx (recon_p - base_recon) / eps, average magnitude across batch
        delta = (recon_p - base_recon) / eps  # (B, D)
        jac[:, j] = np.mean(np.abs(delta), axis=0)
    return jac


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    outdir = Path(args.output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    # Load model using existing loader to respect checkpoint config
    model, meta = load_autoencoder(Path(args.checkpoint), device)
    model_type = meta.get("model_type", "l1")

    # Load dataset and metadata
    dataset = TraceDataset(args.data_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Load metadata from JSON if available
    meta_path = Path(args.data_path).with_suffix('.json')
    num_stocks = 97  # default
    lookback = 20    # default
    feat_dim = 6     # default
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            num_stocks = meta.get('num_stocks', num_stocks)
            lookback = meta.get('lookback', lookback)
            feat_dim = meta.get('input_dim', feat_dim)
    
    # Load ticker names
    ticker_map = load_tickers(args.tickers_path)

    # Collect a limited number of samples for analysis
    samples = []
    count = 0
    max_samples = args.num_samples
    for batch in loader:
        x = batch["input"]
        bs = x.shape[0]
        take = min(bs, max_samples - count)
        if take <= 0:
            break
        samples.append(x[:take])
        count += take
        if count >= max_samples:
            break
    if len(samples) == 0:
        raise RuntimeError("No samples found in dataset")
    sample_batch = torch.cat(samples, dim=0).to(device)  # (N, input_dim)
    print(f"Using {sample_batch.shape[0]} samples for feature extraction")

    # Encode to latents
    model.eval()
    with torch.no_grad():
        if model_type in ("topk", "jumprelu"):
            # model(x) returns (recon, latent, aux_info)
            recon, latent, aux = model(sample_batch)
        else:
            recon, latent = model(sample_batch)

    z = latent  # tensor (N, L)

    # Save latents and stats
    z_np = z.cpu().numpy()
    np.savez_compressed(outdir / "latents_sample.npz", latents=z_np)

    latent_stats = {
        "mean": z_np.mean(axis=0).tolist(),
        "std": z_np.std(axis=0).tolist(),
        "median": np.median(z_np, axis=0).tolist(),
        "num_samples": int(z_np.shape[0]),
    }

    # Determine mapping from latents -> input features
    # Prefer direct linear decoder weights if available
    input_dim = sample_batch.shape[1]
    L = z_np.shape[1]
    decoder_matrix = None

    # Case 1: model has explicit linear mapping latent -> input (decoder_out with identity hidden)
    if hasattr(model, "decoder_out") and getattr(model, "decoder_hidden", None) is not None:
        # If decoder_hidden is Identity and decoder_out maps latent -> input, we can read weight
        try:
            hidden = getattr(model, "decoder_hidden")
            is_identity = isinstance(hidden, torch.nn.Identity)
        except Exception:
            is_identity = False
        if is_identity and model.decoder_out is not None and model.decoder_out.weight is not None:
            # decoder_out.weight shape: (input_dim, latent_dim)
            decoder_matrix = model.decoder_out.weight.detach().cpu().numpy()
            print("Using decoder_out weight as linear mapping (input_dim x latent_dim)")

    # Case 2: tied weights (single-layer encoder)
    if decoder_matrix is None:
        if getattr(model, "decoder_out", None) is None and hasattr(model, "encoder_out"):
            # decode uses encoder_out.weight.t()
            try:
                W_enc = model.encoder_out.weight.detach().cpu().numpy()  # (latent_dim, prev_dim)
                # For single-layer tied weights, encoder_out maps input->latent, so transpose gives latent->input
                decoder_matrix = W_enc.T
                print("Using encoder_out.weight.T for tied-weights mapping")
            except Exception:
                decoder_matrix = None

    # Case 3: fallback to empirical Jacobian linearization
    if decoder_matrix is None:
        print("Falling back to empirical Jacobian approximation (may be slower)")
        jac = compute_empirical_jacobian(model, z, eps=args.eps, device=device)  # (D, L)
        decoder_matrix = jac
    else:
        # Ensure shape is (input_dim, latent_dim)
        decoder_matrix = np.asarray(decoder_matrix)
        if decoder_matrix.shape[0] != input_dim:
            # In some architectures decoder_out maps from some hidden dim; try to infer by passing unit latents
            print("Decoder weight shape does not match input dim; performing empirical linearization for safety")
            jac = compute_empirical_jacobian(model, z, eps=args.eps, device=device)
            decoder_matrix = jac

    # Build feature names
    obs_dim = decoder_matrix.shape[0]
    feature_names = build_feature_names(obs_dim, num_stocks, lookback, feat_dim)
    
    # Enrich feature names with ticker symbols if available
    def get_readable_name(idx: int) -> str:
        if idx >= len(feature_names):
            return f"feature_{idx}"
        name = feature_names[idx]
        # Try to replace stock_N with actual ticker
        for stock_idx, ticker in ticker_map.items():
            if f"stock_{stock_idx}_" in name or f"[{stock_idx}," in name or f",{stock_idx}]" in name:
                name = name.replace(f"stock_{stock_idx}_", f"{ticker}_")
                name = name.replace(f"[{stock_idx},", f"[{ticker},")
                name = name.replace(f",{stock_idx}]", f",{ticker}]")
        return name
    
    # Now, for each latent, compute top-k input features by absolute magnitude
    abs_map = np.abs(decoder_matrix)
    topk = args.top_k
    top_features: Dict[str, Dict] = {}
    for j in range(decoder_matrix.shape[1]):
        col = abs_map[:, j]
        idx = np.argsort(-col)[:topk]
        values = decoder_matrix[idx, j].tolist()
        readable_names = [get_readable_name(i) for i in idx]
        top_features[f"latent_{j}"] = {
            "top_indices": idx.tolist(), 
            "weights": values,
            "feature_names": readable_names
        }

    # Save artifacts
    np.savez_compressed(outdir / "decoder_matrix.npz", decoder_matrix=decoder_matrix)
    with open(outdir / "top_features.json", "w") as f:
        json.dump({"top_k": topk, "top_features": top_features, "latent_stats": latent_stats}, f, indent=2)

    print(f"Saved outputs to {outdir}")

if __name__ == "__main__":
    main()