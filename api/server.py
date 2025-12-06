#!/usr/bin/env python3
"""
FastAPI endpoints for SmartFolio.

Endpoints:
- POST /inference: run model inference over a date range, return metrics and weight CSV path.
- POST /finetune: trigger monthly fine-tuning via fine_tune_month.
- GET  /stream_snapshot: return latest streaming display snapshot (written by gen_data/yf_stream_consumer.py).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from torch_geometric.loader import DataLoader
from stable_baselines3 import PPO

from dataloader.data_loader import AllGraphDataSampler
from env.portfolio_env import StockPortfolioEnv
from trainer.irl_trainer import process_data
from main import fine_tune_month, get_risk_score_dir
from utils.ticker_mapping import get_ticker_mapping_for_period

import subprocess
import sys


class InferenceRequest(BaseModel):
    model_path: str = Field(..., description="Path to PPO checkpoint (.zip)")
    market: str = "custom"
    save_dir: str = "./checkpoints"
    horizon: str = "1"
    relation_type: str = "hy"
    test_start_date: str = Field(..., description="YYYY-MM-DD")
    test_end_date: str = Field(..., description="YYYY-MM-DD")
    deterministic: bool = True
    ind_yn: bool = True
    pos_yn: bool = True
    neg_yn: bool = True
    lookback: int = 30
    input_dim: Optional[int] = None
    risk_score: float = 0.5
    output_dir: str = "./logs/api"


class PortfolioValuePoint(BaseModel):
    """Single portfolio value data point."""
    step: int
    portfolio_value: Optional[float]
    daily_return: Optional[float]

class FinetuneRequest(BaseModel):
    save_dir: str = Field("./checkpoints", description="Base checkpoint directory (risk suffix is added automatically)")
    device: str = Field("cpu", description="Matches -device/--device")
    run_monthly_fine_tune: bool = Field(True, description="Set when using --run_monthly_fine_tune")
    market: str = Field("custom", description="Dataset market identifier passed via --market")
    horizon: str = Field("1", description="Matches --horizon")
    relation_type: str = Field("hy", description="Matches --relation_type")
    fine_tune_steps: int = Field(1, description="Matches --fine_tune_steps")
    baseline_checkpoint: Optional[str] = Field(
        None,
        description="Optional override for --baseline_checkpoint; defaults to <save_dir_risk>/baseline.zip",
    )
    promotion_min_sharpe: float = Field(0.5, description="Matches --promotion_min_sharpe")
    promotion_max_drawdown: float = Field(0.2, description="Matches --promotion_max_drawdown")
    resume_model_path: Optional[str] = Field(
        None,
        description="Optional override for --resume_model_path; defaults to baseline in risk-scored directory",
    )
    batch_size: int = Field(16, description="Matches --batch_size")
    n_steps: int = Field(2048, description="Matches --n_steps")
    num_stocks: Optional[int] = Field(
        None, description="Optional manual override if auto-detect fails"
    )
    ptr_mode: bool = Field(True, description="Set true to mirror --ptr_mode flag")
    use_ptr: bool = Field(True, description="Deprecated; use ptr_mode instead")
    ptr_coef: float = Field(0.3, description="Matches --ptr_coef")
    ptr_memory_size: int = Field(1000, description="Matches --ptr_memory_size")
    ptr_priority_type: str = Field(
        "max",
        description="Matches --ptr_priority_type (when PTR is enabled)",
    )
    risk_score: float = Field(0.5, description="Risk score used for directory routing and reward shaping")
    stream: Optional[str] = Field(
        None,
        description="Pathway streaming flag (matches positional 'stream' arg in main.py)",
    )
    finetune_month: Optional[str] = Field(
        None,
        description="Optional month label (YYYY-MM) to finetune on; if not provided, uses latest available month",
    )


def _dataset_dir(req: InferenceRequest) -> Path:
    return Path("dataset_default") / f"data_train_predict_{req.market}" / f"{req.horizon}_{req.relation_type}"


def _load_test_loader(req: InferenceRequest) -> DataLoader:
    data_dir = _dataset_dir(req)
    test_dataset = AllGraphDataSampler(
        base_dir=str(data_dir),
        date=True,
        test_start_date=req.test_start_date,
        test_end_date=req.test_end_date,
        mode="test",
    )
    if len(test_dataset) == 0:
        raise ValueError(f"Empty test dataset for range {req.test_start_date} to {req.test_end_date}")
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), pin_memory=True)
    return test_loader


def _run_inference(req: InferenceRequest) -> Dict[str, Any]:
    import torch
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SERVER] Using device: {device}", flush=True)
    
    test_loader = _load_test_loader(req)
    print(f"[SERVER] Loaded test loader with {len(test_loader)} batches", flush=True)

    ticker_map = get_ticker_mapping_for_period(
        req.market,
        req.test_start_date,
        req.test_end_date,
        base_dir="dataset_default"
    )
    date_keys = list(ticker_map.keys()) if ticker_map else []
    req.model_path = get_risk_score_dir(req.save_dir, req.risk_score) + "/" + "baseline.zip"
    model_path = Path(req.model_path).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    print(f"[SERVER] Loading model from {model_path}", flush=True)
    model = PPO.load(str(model_path), env=None, device=device)
    print(f"[SERVER] Model loaded successfully", flush=True)

    records: List[Dict[str, Any]] = []
    all_weights_data: List[Dict[str, Any]] = []
    portfolio_value_data: List[Dict[str, Any]] = []
    
    final_net_value: Optional[float] = None
    peak_value: Optional[float] = None
    current_value: Optional[float] = None

    print(f"[SERVER] Starting inference loop", flush=True)
    
    final_weights_map: Dict[str, float] = {}

    for batch_idx, data in enumerate(test_loader):
        print(f"[SERVER] Processing batch {batch_idx}", flush=True)
        sys.stdout.flush()
        
        corr, ts_features, features, ind, pos, neg, labels, pyg_data, mask = process_data(data, device=device)
        
        print(f"[SERVER] Data processed. ts_features shape: {ts_features.shape if ts_features is not None else 'None'}", flush=True)
        
        # Extract actual input_dim from ts_features shape
        if ts_features is not None:
            if len(ts_features.shape) == 4:
                actual_input_dim = ts_features.shape[-1]
            elif len(ts_features.shape) == 3:
                actual_input_dim = ts_features.shape[-1]
            else:
                actual_input_dim = features.shape[-1]
        else:
            actual_input_dim = features.shape[-1]
        
        print(f"[SERVER] actual_input_dim: {actual_input_dim}", flush=True)
        
        args_stub = argparse.Namespace(
            risk_score=req.risk_score,
            ind_yn=req.ind_yn,
            pos_yn=req.pos_yn,
            neg_yn=req.neg_yn,
            lookback=req.lookback,
            input_dim=actual_input_dim,
        )
        
        print(f"[SERVER] Creating StockPortfolioEnv...", flush=True)
        sys.stdout.flush()
        df_benchmark = pd.read_csv(f"./dataset_default/index_data/{req.market}_index.csv")
        df_benchmark = df_benchmark[(df_benchmark['datetime'] >= req.test_start_date) &
                                    (df_benchmark['datetime'] <= req.test_end_date)]
        benchmark_return = df_benchmark['daily_return']
        print(f"[SERVER] Benchmark data loaded with {len(benchmark_return)} entries", flush=True)
        env_test = StockPortfolioEnv(
            args=args_stub,
            corr=corr,
            ts_features=ts_features,
            features=features,
            ind=ind,
            pos=pos,
            neg=neg,
            returns=labels,
            pyg_data=pyg_data,
            mode="test",
            ind_yn=req.ind_yn,
            pos_yn=req.pos_yn,
            neg_yn=req.neg_yn,
            risk_profile={"risk_score": req.risk_score},
            benchmark_return=benchmark_return,
        )

        print(f"[SERVER] Environment created successfully", flush=True)
        
        obs_test = env_test.reset()
        print(f"[SERVER] Environment reset. obs shape: {obs_test.shape}", flush=True)
        
        max_step = len(labels)
        print(f"[SERVER] Running {max_step} inference steps", flush=True)

        for step in range(max_step):
            action, _states = model.predict(obs_test, deterministic=req.deterministic)
            obs_test, reward, done, info = env_test.step(action)
            if done:
                break

        print(f"[SERVER] Inference complete for batch {batch_idx}", flush=True)
        
        metrics, benchmark_metrics = env_test.evaluate()
        metrics_record = {"batch": batch_idx}
        metrics_record.update(metrics)
        if benchmark_metrics:
            metrics_record.update({f"benchmark_{k}": v for k, v in benchmark_metrics.items()})
        records.append(metrics_record)

        net_value_history = env_test.get_net_value_history()
        daily_returns_history = env_test.get_daily_returns_history()
        
        final_net_value = float(net_value_history[-1]) if len(net_value_history) > 0 else None
        peak_value = float(env_test.peak_value)
        current_value = float(env_test.net_value)

        weights_array = env_test.get_weights_history()
        if weights_array.size > 0:
            num_steps, num_stocks = weights_array.shape
            step_labels = getattr(env_test, "dates", list(range(num_steps)))
            # Build a ticker list for each step if available
            def _tickers_for_step(step_idx: int):
                if step_idx < len(date_keys):
                    candidate = ticker_map.get(date_keys[step_idx])
                    if candidate and len(candidate) == num_stocks:
                        return candidate
                return [f"stock_{i}" for i in range(num_stocks)]

            for step_idx in range(num_steps):
                weights = weights_array[step_idx]
                step_value = step_labels[step_idx] if step_idx < len(step_labels) else step_idx
                
                portfolio_value = float(net_value_history[step_idx]) if step_idx < len(net_value_history) else None
                daily_return = float(daily_returns_history[step_idx]) if step_idx < len(daily_returns_history) else None
                tickers = _tickers_for_step(step_idx)
                
                for idx, weight in enumerate(weights):
                    if weight > 0.0001:
                        all_weights_data.append({
                            "run_id": f"api_run_{batch_idx}",
                            "batch": batch_idx,
                            "date": date_keys[step_idx] if step_idx < len(date_keys) else None,
                            "step": step_value,
                            "ticker": tickers[idx] if idx < len(tickers) else f"stock_{idx}",
                            "weight": float(weight),
                            "weight_pct": float(weight * 100),
                            "portfolio_value": portfolio_value,
                            "daily_return": daily_return,
                        })
                
                portfolio_value_data.append({
                    "run_id": f"api_run_{batch_idx}",
                    "batch": batch_idx,
                    "step": step_value,
                    "portfolio_value": portfolio_value,
                    "daily_return": daily_return,
                })

            # Save final step weights mapping for quick access in response
            final_step_idx = num_steps - 1
            final_tickers = _tickers_for_step(final_step_idx)
            final_weights_map = {
                (final_tickers[i] if i < len(final_tickers) else f"stock_{i}"): float(w)
                for i, w in enumerate(weights_array[final_step_idx])
            }

    out_dir = Path(req.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    weights_csv = out_dir / f"weights_{req.market}_{req.test_start_date}_{req.test_end_date}.csv"
    if all_weights_data:
        pd.DataFrame(all_weights_data).to_csv(weights_csv, index=False)
    
    portfolio_csv = out_dir / f"portfolio_values_{req.market}_{req.test_start_date}_{req.test_end_date}.csv"
    if portfolio_value_data:
        pd.DataFrame(portfolio_value_data).to_csv(portfolio_csv, index=False)

    portfolio_summary = [
        PortfolioValuePoint(
            step=int(pv["step"]),
            portfolio_value=pv["portfolio_value"],
            daily_return=pv["daily_return"]
        )
        for pv in portfolio_value_data
    ]

    return {
        "metrics": records,
        # "weights_csv": str(weights_csv) if all_weights_data else None,
        # "portfolio_values_csv": str(portfolio_csv) if portfolio_value_data else None,
        # "portfolio_values_summary": portfolio_summary,
        "final_portfolio_value": final_net_value,
        "peak_value": peak_value,
        "current_value": current_value,
        "final_weights": final_weights_map if final_weights_map else None,
    }


app = FastAPI(title="SmartFolio API", version="0.1.0")


@app.post("/inference")
def inference(req: InferenceRequest):
    try:
        result = _run_inference(req)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/finetune")
def finetune(req: FinetuneRequest):
    try:
        # Auto-detect num_stocks from a sample pickle file (same as main.py)
        import pickle
        data_dir = f'dataset_default/data_train_predict_{req.market}/{req.horizon}_{req.relation_type}/'
        sample_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
        num_stocks = req.num_stocks
        if sample_files:
            sample_path = os.path.join(data_dir, sample_files[0])
            with open(sample_path, 'rb') as f:
                sample_data = pickle.load(f)
            num_stocks = sample_data['features'].shape[0]
            print(f"Auto-detected num_stocks: {num_stocks}")
        else:
            print(
                "No pickle files available for auto-detect; falling back to request num_stocks="
                f"{num_stocks}"
            )
            if not num_stocks:
                raise ValueError(f"Unable to determine num_stocks for {data_dir} (provide num_stocks in request)")

        # Load replay buffer if available (same as main.py)
        replay_buffer = []
        # Treat empty save_dir as default "./checkpoints"
        save_dir_base = req.save_dir if req.save_dir else "./checkpoints"
        save_dir_risk = get_risk_score_dir(save_dir_base, req.risk_score)
        buffer_path = os.path.join(save_dir_risk, f"replay_buffer_{req.market}.pkl")
        if os.path.exists(buffer_path):
            with open(buffer_path, "rb") as f:
                replay_buffer = pickle.load(f)
            print(f"Loaded replay buffer with {len(replay_buffer)} samples from {buffer_path}.")

        # Resolve checkpoint paths with risk-scored directory fallback
        baseline_ckpt = req.baseline_checkpoint or os.path.join(save_dir_risk, "baseline.zip")
        resume_ckpt = baseline_ckpt
        os.makedirs(save_dir_risk, exist_ok=True)

        args = argparse.Namespace(
            device=req.device,
            model_name="SmartFolio",
            horizon=req.horizon,
            relation_type=req.relation_type,
            ind_yn=True,
            pos_yn=True,
            neg_yn=True,
            use_ptr=req.use_ptr,
            ptr_coef=req.ptr_coef,
            ptr_memory_size=req.ptr_memory_size,
            ptr_priority_type=req.ptr_priority_type,
            batch_size=req.batch_size,
            n_steps=req.n_steps,
            tickers_file="tickers.csv",
            multi_reward_yn=True,
            resume_model_path=resume_ckpt,
            fine_tune_steps=req.fine_tune_steps,
            save_dir=save_dir_risk,
            baseline_checkpoint=baseline_ckpt,
            promotion_min_sharpe=req.promotion_min_sharpe,
            promotion_max_drawdown=req.promotion_max_drawdown,
            run_monthly_fine_tune=req.run_monthly_fine_tune,
            discover_months_with_pathway=False,
            month_cutoff_days=None,
            min_month_days=10,
            expert_cache_path=None,
            irl_epochs=5,
            rl_timesteps=1,
            dd_base_weight=1.0,
            dd_risk_factor=1.0,
            market=req.market,
            seed=123,
            input_dim=6,
            ind=True,
            pos=True,
            neg=True,
            relation="hy",
            num_stocks=num_stocks,
            lookback=30,
            finrag_prior=None,
            finrag_weights_path=None,
            ptr_mode=req.ptr_mode,
            risk_score=req.risk_score,
            stream=req.stream if req.stream else None,  # Treat empty string as None
        )
        
        # Call fine_tune_month (returns checkpoint path AND new replay samples)
        # Treat empty string finetune_month as None
        finetune_month = req.finetune_month if req.finetune_month else None
        checkpoint, new_samples = fine_tune_month(args, replay_buffer=replay_buffer, finetune_month=finetune_month)
        
        # Update replay buffer with new samples (same as main.py)
        if new_samples:
            replay_buffer.extend(new_samples)
            max_buffer = req.ptr_memory_size
            if len(replay_buffer) > max_buffer:
                # Keep the most recent ones
                replay_buffer = replay_buffer[-max_buffer:]
            print(f"Replay buffer updated. Current size: {len(replay_buffer)}")
            os.makedirs(save_dir_risk, exist_ok=True)
            with open(buffer_path, "wb") as f:
                pickle.dump(replay_buffer, f)
            print(f"Persisted replay buffer to {buffer_path}")
        
        return {
            "checkpoint": checkpoint,
            "replay_buffer_size": len(replay_buffer),
            "new_samples_added": len(new_samples) if new_samples else 0,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/stream_snapshot")
def stream_snapshot(path: str = "display_data/stream_snapshot.parquet", limit: int = 200):
    """
    Return the latest streaming display snapshot produced by gen_data/yf_stream_consumer.py.
    """
    snapshot_path = Path(path).expanduser()
    if not snapshot_path.exists():
        raise HTTPException(status_code=404, detail=f"Snapshot not found: {snapshot_path}")
    try:
        df = pd.read_parquet(snapshot_path)
        if "dt" in df.columns:
            df = df.sort_values("dt")
        if limit and limit > 0:
            df = df.tail(limit)
        return {
            "path": str(snapshot_path),
            "rows": len(df),
            "data": df.to_dict(orient="records"),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Run SmartFolio FastAPI server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("api.server:app", host=args.host, port=args.port, reload=False)
