from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.utils.data import DataLoader, random_split

from .dataset import TraceDataset
from .model import (
    SparseAutoencoder,
    TopKSparseAutoencoder,
    JumpReLUSparseAutoencoder,
    sparse_ae_loss,
    topk_sae_loss,
    variance_explained,
)

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sparse autoencoder on PPO activations")
    parser.add_argument("--data-path", required=True, help="Path to traces NPZ produced by updated collect_traces.py")
    parser.add_argument("--output-dir", default="experiments/latent_factors/checkpoints", help="Where to store checkpoints")
    
    parser.add_argument("--model-type", type=str, default="topk", choices=["topk", "l1", "jumprelu"],
                        help="Type of sparse autoencoder: topk (recommended), l1 (original), jumprelu")
    parser.add_argument("--latent-dim", type=int, default=64, help="Latent dimensionality (increase for complex data)")
    parser.add_argument("--hidden-dim", type=int, default=1024, help="Hidden layer width")
    parser.add_argument("--num-hidden", type=int, default=1, help="Number of hidden layers (1 is often best for SAE)")
    parser.add_argument("--k", type=int, default=32, help="TopK: number of active latents per sample")
    parser.add_argument("--tied-weights", action="store_true", help="Use tied encoder-decoder weights")
    parser.add_argument("--normalize-decoder", action="store_true", default=True, help="Normalize decoder columns")
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "gelu", "silu"],
                        help="Activation function in hidden layers")
    
    # Loss and regularization
    parser.add_argument("--sparsity-weight", type=float, default=1e-3, help="L1 penalty weight (for l1 model)")
    parser.add_argument("--aux-weight", type=float, default=1e-3, help="Auxiliary loss weight for dead latent prevention")
    parser.add_argument("--use-auxiliary", action="store_true", default=True, help="Use auxiliary loss for TopK")
    
    # Training
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size") # Increased default for dense vectors
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine", "onecycle"],
                        help="Learning rate scheduler")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Warmup epochs for scheduler")
    
    # Data
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraction of data for validation")
    parser.add_argument("--device", default="cpu", help="Torch device")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    
    # Logging
    parser.add_argument("--log-interval", type=int, default=10, help="Epochs between detailed logs")
    parser.add_argument("--save-best", action="store_true", default=True, help="Save best model by val R²")
    
    return parser.parse_args(argv)


def compute_r2_batch(model, loader, device, model_type: str) -> dict:
    """Compute R² and other metrics over a data loader."""
    model.eval()
    all_targets = []
    all_recons = []
    all_latents = []
    total_mse = 0.0
    total_sparsity = 0.0
    count = 0
    
    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device)
            
            if model_type == "topk" or model_type == "jumprelu":
                recon, latent, aux_info = model(x)
                sparsity = aux_info.get("sparsity", 0.0)
            else:
                recon, latent = model(x)
                sparsity = (latent != 0).float().mean()
            
            all_targets.append(x.cpu())
            all_recons.append(recon.cpu())
            all_latents.append(latent.cpu())
            total_mse += float(torch.nn.functional.mse_loss(recon, x).item()) * x.size(0)
            total_sparsity += float(sparsity if isinstance(sparsity, float) else sparsity.item()) * x.size(0)
            count += x.size(0)
    
    targets = torch.cat(all_targets, dim=0)
    recons = torch.cat(all_recons, dim=0)
    latents = torch.cat(all_latents, dim=0)
    
    # R² computation
    r2 = variance_explained(recons, targets).item()
    
    latent_active = (latents.abs() > 1e-6).float().mean().item()
    latent_var = latents.var(dim=0).mean().item()
    
    return {
        "r2": r2,
        "mse": total_mse / count,
        "sparsity": total_sparsity / count,
        "latent_active_ratio": latent_active,
        "latent_var": latent_var,
    }

def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = TraceDataset(args.data_path)
    input_dim = dataset.activations_flat.shape[1]
    
    print(f"Loaded dataset: {len(dataset)} samples")
    print(f"Activation dimension (SAE Input): {input_dim}")

    total_len = len(dataset)
    val_size = max(1, int(total_len * args.val_split))
    train_size = total_len - val_size
    
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device(args.device)
    
    if args.model_type == "topk":
        model = TopKSparseAutoencoder(
            input_dim=input_dim,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            num_hidden=args.num_hidden,
            k=args.k,
            tied_weights=args.tied_weights,
            normalize_decoder=args.normalize_decoder,
            activation=args.activation,
            use_pre_bias=True, 
        ).to(device)
    elif args.model_type == "jumprelu":
        model = JumpReLUSparseAutoencoder(
            input_dim=input_dim,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            num_hidden=args.num_hidden,
        ).to(device)
    else:
        model = SparseAutoencoder(
            input_dim=input_dim,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            num_hidden=args.num_hidden,
        ).to(device)
    
    print(f"Model: {args.model_type}, Latents: {args.latent_dim}, K: {args.k if args.model_type=='topk' else 'N/A'}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize pre-bias from data for better centering (TopK models only)
    if args.model_type == "topk" and hasattr(model, "set_pre_bias_from_data"):
        print("Initializing pre-bias from training data...")
        sample_data = next(iter(train_loader))["input"].to(device)
        model.set_pre_bias_from_data(sample_data)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=max(1, args.epochs // 3), T_mult=2)
    elif args.scheduler == "onecycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=args.warmup_epochs / args.epochs,
        )

    history = {
        "train_loss": [], "train_mse": [], "train_r2": [],
        "val_loss": [], "val_mse": [], "val_r2": [], "val_sparsity": [],
        "dead_latent_ratio": [], "lr": [],
    }
    
    best_val_r2 = -float("inf")
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        if hasattr(model, "reset_activation_stats"):
            model.reset_activation_stats()
        
        train_loss = 0.0
        train_mse = 0.0
        
        for batch in train_loader:
            x = batch["input"].to(device)
            optimizer.zero_grad()
            
            if args.model_type == "topk" or args.model_type == "jumprelu":
                recon, latent, aux_info = model(x)
                losses = topk_sae_loss(
                    recon, x, latent, aux_info,
                    aux_weight=args.aux_weight,
                    use_auxiliary=args.use_auxiliary,
                )
            else:
                recon, latent = model(x)
                losses = sparse_ae_loss(recon, x, latent, args.sparsity_weight)
            
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Normalize decoder weights for TopK models (improves interpretability)
            if args.model_type == "topk" and hasattr(model, "_normalize_decoder_weights"):
                model._normalize_decoder_weights()
            
            if args.scheduler == "onecycle" and scheduler is not None:
                scheduler.step()
            
            train_loss += float(losses["loss"].item()) * x.size(0)
            train_mse += float(losses["mse"].item()) * x.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_mse /= len(train_loader.dataset)
        
        if args.scheduler == "cosine" and scheduler is not None:
            scheduler.step()
        
        # Metrics
        val_metrics = compute_r2_batch(model, val_loader, device, args.model_type)
        train_metrics = compute_r2_batch(model, train_loader, device, args.model_type)
        
        dead_ratio = 0.0
        if hasattr(model, "get_dead_latent_ratio"):
            dead_ratio = model.get_dead_latent_ratio()
        
        current_lr = optimizer.param_groups[0]["lr"]
        
        history["train_loss"].append(train_loss)
        history["train_mse"].append(train_mse)
        history["train_r2"].append(train_metrics["r2"])
        history["val_loss"].append(val_metrics["mse"])
        history["val_mse"].append(val_metrics["mse"])
        history["val_r2"].append(val_metrics["r2"])
        history["val_sparsity"].append(val_metrics["sparsity"])
        history["dead_latent_ratio"].append(dead_ratio)
        history["lr"].append(current_lr)
        
        # Save best
        if val_metrics["r2"] > best_val_r2:
            best_val_r2 = val_metrics["r2"]
            best_epoch = epoch
            if args.save_best:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": vars(args),
                        "input_dim": input_dim,
                        "epoch": epoch,
                        "val_r2": best_val_r2,
                    },
                    output_dir / "sparse_ae_best.pt",
                )
        
        if epoch == 1 or epoch % args.log_interval == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:03d} | "
                f"loss={train_loss:.6f} tr_R²={train_metrics['r2']:.4f} | "
                f"val_R²={val_metrics['r2']:.4f} sparsity={val_metrics['sparsity']:.3f} dead={dead_ratio:.3f}"
            )

    print(f"\nBest val R²: {best_val_r2:.4f} at epoch {best_epoch}")
    
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": vars(args),
            "input_dim": input_dim,
            "epoch": args.epochs,
            "best_val_r2": best_val_r2,
        },
        output_dir / "sparse_ae.pt",
    )
    
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    main()
