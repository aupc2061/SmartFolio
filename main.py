import os
import time
import json
import argparse
import warnings
from datetime import datetime
import calendar
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore", category=UserWarning)
import pandas as pd
import torch
print(torch.cuda.is_available())
from dataloader.data_loader import *
from policy.policy import *
from stable_baselines3 import PPO
from trainer.irl_trainer import *
from torch_geometric.loader import DataLoader
from utils.risk_profile import build_risk_profile
from tools.pathway_temporal import discover_monthly_shards_with_pathway
from tools.pathway_monthly_builder import build_monthly_shards_with_pathway
from gen_data.update_monthly_dataset import fetch_latest_month_data
from trainer.evaluation_utils import apply_promotion_gate, aggregate_metric_records, persist_metrics, create_metric_record

import shutil
import pickle
from stable_baselines3.common.save_util import load_from_zip_file
from trainer.ptr_ppo import PTR_PPO

PATH_DATA = f'./dataset_default/'


def get_risk_score_dir(base_dir: str, risk_score: float) -> str:
    """Get the checkpoint directory for a specific risk score.
    
    Examples:
        base_dir='checkpoints', risk_score=0.5 -> 'checkpoints_risk05'
        base_dir='./checkpoints', risk_score=0.1 -> './checkpoints_risk01'
    """
    # Convert risk score to tag (0.5 -> '05', 0.1 -> '01')
    risk_tag = str(risk_score).replace('.', '')
    # Append risk tag to directory name
    return f"{base_dir.rstrip('/')}_risk{risk_tag}"


def _copy_compatible_policy_weights(policy_module, loaded_state, checkpoint_path):
    """Load only the tensors that match by key and shape to avoid shape-mismatch crashes."""
    if not loaded_state:
        print(f"Warning: {checkpoint_path} did not contain policy weights; skipping weight transfer.")
        return 0

    current_state = policy_module.state_dict()
    matched = 0
    skipped = []

    for key, tensor in loaded_state.items():
        target_tensor = current_state.get(key)
        if target_tensor is None:
            continue
        if target_tensor.shape != tensor.shape:
            skipped.append((key, tuple(tensor.shape), tuple(target_tensor.shape)))
            continue
        if isinstance(tensor, torch.Tensor):
            current_state[key] = tensor.to(target_tensor.device)
        else:
            current_state[key] = torch.as_tensor(tensor, device=target_tensor.device)
        matched += 1

    policy_module.load_state_dict(current_state)

    if matched == 0:
        print(f"Warning: No compatible policy weights found in {checkpoint_path}; starting from scratch.")
    else:
        print(f"Loaded {matched} compatible policy tensors from {checkpoint_path}.")

    if skipped:
        preview = ", ".join(f"{name}: {src}->{dst}" for name, src, dst in skipped[:5])
        if len(skipped) > 5:
            preview += ", ..."
        print(
            f"Skipped {len(skipped)} tensors from {checkpoint_path} due to shape mismatches."
            f" Examples: {preview}"
        )
    if not skipped:
        print("All loaded policy tensors matched successfully.")
    return matched

def load_weights_into_new_model(
    path,
    env,
    device,
    policy_kwargs,
    ptr_mode=False,
    ptr_coef=0.1,
    prior_policy=None,
):
    def _extract_policy_state_dict():
        try:
            _, params, _ = load_from_zip_file(path, device=device)
        except ValueError as exc:
            print(f"Warning: load_from_zip_file failed ({exc}). Falling back to PPO.load...")
            temp_model = PPO.load(path, env=None, device=device)
            return temp_model.policy.state_dict()

        if params is None:
            return None
        if isinstance(params, dict) and "policy" in params:
            return params["policy"]
        return params

    policy_state = _extract_policy_state_dict()

    model_cls = PTR_PPO if ptr_mode else PPO
    init_kwargs = dict(
        policy=HGATActorCriticPolicy,
        env=env,
        policy_kwargs=policy_kwargs,
        device=device,
        **PPO_PARAMS,
    )
    if ptr_mode:
        init_kwargs["ptr_coef"] = ptr_coef
        init_kwargs["prior_policy"] = prior_policy

    model = model_cls(**init_kwargs)
    _copy_compatible_policy_weights(model.policy, policy_state, path)
    return model

def load_finrag_prior(weights_path, num_stocks, tickers_csv="tickers.csv"):
    """
    Load FinRAG weights from a JSON file and normalize them to a simplex vector.
    Supports payloads shaped as:
      - [w1, w2, ...]
      - {"weights": [...]} or {"scores": [...]}
      - {"TICKER": weight, ...} (ordered by tickers_csv when available)
    """
    if not weights_path:
        print("FinRAG weights path not provided; skipping prior initialization.")
        return None
    if not os.path.exists(weights_path):
        print(f"FinRAG weights path not found: {weights_path}; skipping prior initialization.")
        return None

    with open(weights_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    if isinstance(payload, dict) and ("weights" in payload or "scores" in payload):
        payload = payload.get("weights") or payload.get("scores")

    # Resolve to a list of weights
    weights = None
    if isinstance(payload, dict):
        # Map tickers to weights using the CSV order when available
        if os.path.exists(tickers_csv):
            tickers = pd.read_csv(tickers_csv)["ticker"].tolist()
        else:
            tickers = list(payload.keys())
        weights = [float(payload.get(ticker, 0.0)) for ticker in tickers]
    else:
        weights = list(payload)

    weights_arr = np.asarray(weights, dtype=np.float32)
    if weights_arr.shape[0] != num_stocks:
        print(
            f"FinRAG weights length {weights_arr.shape[0]} does not match num_stocks {num_stocks}; "
            "skipping prior initialization."
        )
        return None

    weights_arr = np.clip(weights_arr, 0.0, None)
    total = float(weights_arr.sum())
    if total <= 0:
        print("FinRAG weights sum to zero; skipping prior initialization.")
        return None

    prior = weights_arr / total
    print(f"Loaded FinRAG prior from {weights_path} (len={len(prior)})")
    return prior


def init_policy_bias_from_prior(model, prior_weights):
    """
    Initialize the policy action head bias so the mean action roughly matches the prior.
    Works for SB3 ActorCriticPolicy subclasses where action_net is a Linear layer.
    """

    if prior_weights is None:
        return
    policy = getattr(model, "policy", None)
    action_net = getattr(policy, "action_net", None)
    if action_net is None or not hasattr(action_net, "bias"):
        print("Policy action_net missing; cannot apply FinRAG prior bias.")
        return
    if action_net.bias.shape[0] != len(prior_weights):
        print(
            f"Action bias shape {action_net.bias.shape[0]} does not match prior length {len(prior_weights)}; "
            "skipping prior bias init."
        )
        return

    prior_logits = torch.log(torch.from_numpy(prior_weights + 1e-8)).to(action_net.bias.device)
    with torch.no_grad():
        action_net.bias.copy_(prior_logits)
    print("Initialized policy action bias from FinRAG prior.")

def select_replay_samples(model, env, dataset, k_percent=0.3):
    """    
    Select top k% samples based on absolute reward magnitude (proxy for importance).
    """
    print("Selecting replay samples...")
    obs = env.reset()
    rewards = []

    env.reset()

    real_env = env.envs[0]
    max_steps = real_env.max_step
    
    step_rewards = []
    
    for i in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        step_rewards.append((i, abs(reward[0]))) # Store index and abs reward
        if done:
            break
            
    step_rewards.sort(key=lambda x: x[1], reverse=True)
    
    # Select top k
    num_to_select = int(len(step_rewards) * k_percent)
    selected_indices = [x[0] for x in step_rewards[:num_to_select]]
    
    selected_samples = [dataset.data_all[i] for i in selected_indices if i < len(dataset.data_all)]
    
    print(f"Selected {len(selected_samples)} replay samples from {len(dataset)} total.")
    return selected_samples

def fine_tune_month(args, bookkeeping_path=None, replay_buffer=None, fetch_new_data=False, stream=None, finetune_month=None):
    """
    Fine-tune the PPO model on the latest month derived from pkl files in the data directory.
    
    Simple logic:
    1. Optionally fetch new data from yfinance (if fetch_new_data=True)
    2. Scan pkl files in data_dir (named like 2024-11-29.pkl)
    3. Parse dates from filenames
    4. Group by month (YYYY-MM)
    5. Find the latest month with data (or use finetune_month if provided)
    6. Fine-tune on that month's data
    
    Args:
        args: Namespace with market, horizon, relation_type, etc.
        bookkeeping_path: Deprecated (ignored)
        replay_buffer: Optional list of replay samples from previous training
        fetch_new_data: If True, fetch latest month from yfinance before fine-tuning
        stream: Flag for using pathway streaming
        finetune_month: Optional month label (YYYY-MM) to use for finetuning instead of auto-detecting latest
    """
    # Data directory containing daily pkl files
    base_dir = f'dataset_default/data_train_predict_{args.market}/{args.horizon}_{args.relation_type}/'
    
    # Use stream parameter if provided, otherwise check args.stream
    stream_lock = stream if stream is not None else getattr(args, "stream", None)

    # Optionally fetch new data from yfinance
    if fetch_new_data:
        try:
            print("Fetching latest month data from yfinance...")
            tickers_file = getattr(args, "tickers_file", "tickers.csv")
            fetched_month = fetch_latest_month_data(
                market=args.market,
                horizon=int(args.horizon),
                relation_type=args.relation_type,
                tickers_file=tickers_file,
                lookback=getattr(args, "lookback", 30),
                stream=stream_lock
            )
            print(f"Successfully fetched data for month: {fetched_month}")
        except Exception as e:
            print(f"Warning: Could not fetch new data: {e}")
            print("Continuing with existing data...")
    
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Data directory not found: {base_dir}")
    
    # Scan pkl files and parse dates from filenames
    pkl_files = [f for f in os.listdir(base_dir) if f.endswith('.pkl')]
    if not pkl_files:
        raise ValueError(f"No pkl files found in {base_dir}")
    
    # Parse dates from filenames (e.g., "2024-11-29.pkl" -> datetime(2024, 11, 29))
    dates = []
    for f in pkl_files:
        try:
            date_str = f.replace('.pkl', '')
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            dates.append(dt)
        except ValueError:
            continue  # Skip files that don't match date format

    if not dates:
        raise ValueError(f"No valid date-named pkl files found in {base_dir}")

    # Unique month labels sorted
    dates.sort()
    month_labels = sorted({d.strftime("%Y-%m") for d in dates})
    if not month_labels:
        raise ValueError("No months could be derived from pkl filenames.")

    # Use finetune_month if provided, otherwise use latest month
    # Also check args.finetune_month as fallback
    target_month = finetune_month or getattr(args, "finetune_month", None)
    if target_month:
        if target_month not in month_labels:
            raise ValueError(f"Specified finetune_month '{target_month}' not found in available months: {month_labels}")
        latest_month = target_month
        print(f"Using specified finetune_month: {latest_month}")
    else:
        latest_month = month_labels[-1]
        print(f"Auto-detected latest month: {latest_month}")

    # Training month = t-7 (if available), else earliest
    train_month_idx = max(0, len(month_labels) - 8)
    train_month = month_labels[train_month_idx]

    # Evaluation window = months t-6 .. t (or as many as available)
    eval_months = month_labels[-7:] if len(month_labels) >= 7 else month_labels

    def _month_dates(label):
        return [d for d in dates if d.strftime("%Y-%m") == label]

    train_month_dates = _month_dates(train_month)
    if not train_month_dates:
        raise ValueError(f"No data available for training month {train_month}")
    train_start = min(train_month_dates).strftime("%Y-%m-%d")
    train_end = max(train_month_dates).strftime("%Y-%m-%d")

    eval_dates = [d for d in dates if d.strftime("%Y-%m") in eval_months]
    if not eval_dates:
        raise ValueError(f"No data available for evaluation window months={eval_months}")
    eval_start = min(eval_dates).strftime("%Y-%m-%d")
    eval_end = max(eval_dates).strftime("%Y-%m-%d")

    print(f"Training month: {train_month} ({train_start} to {train_end})")
    print(f"Evaluation window: {eval_months[0]} to {eval_months[-1]} ({eval_start} to {eval_end})")

    # Load datasets
    train_dataset = AllGraphDataSampler(
        base_dir=base_dir,
        date=True,
        train_start_date=train_start,
        train_end_date=train_end,
        mode="train",
    )
    eval_dataset = AllGraphDataSampler(
        base_dir=base_dir,
        date=True,
        test_start_date=eval_start,
        test_end_date=eval_end,
        mode="test",
    )

    if len(train_dataset) == 0:
        raise ValueError(f"Training dataset for {train_month} is empty (start={train_start}, end={train_end})")
    if len(eval_dataset) == 0:
        raise ValueError(f"Evaluation dataset for window {eval_months} is empty (start={eval_start}, end={eval_end})")

    # Pad short lookback windows to the configured lookback for training (eval left untouched)
    target_lookback = getattr(args, "lookback", 30)

    def _pad_sample(sample):
        ts_feats = sample.get("ts_features")
        if ts_feats is None:
            return sample
        try:
            length = ts_feats.shape[1]
        except Exception:
            return sample
        if length < target_lookback:
            pad_len = target_lookback - length
            if isinstance(ts_feats, torch.Tensor):
                pad_slice = ts_feats[:, :1, :].repeat(1, pad_len, 1)
                ts_padded = torch.cat([pad_slice, ts_feats], dim=1)
            else:
                pad_slice = np.repeat(ts_feats[:, :1, :], pad_len, axis=1)
                ts_padded = np.concatenate([pad_slice, ts_feats], axis=1)
            sample = dict(sample)
            sample["ts_features"] = ts_padded
        return sample

    for i, sample in enumerate(train_dataset.data_all):
        train_dataset.data_all[i] = _pad_sample(sample)

    # We inject replay buffer into train dataset if provided
    if replay_buffer:
        print("Injecting replay buffer samples into training dataset...")
        padded_replay = [_pad_sample(s) for s in replay_buffer]
        train_dataset.data_all.extend(padded_replay)

    train_loader = DataLoader(
        train_dataset,
        batch_size=len(train_dataset),
        pin_memory=True,
        collate_fn=lambda x: x,
        drop_last=True,
    )

    env_init = create_env_init(args, data_loader=train_loader)
    env_ref = env_init.envs[0] if hasattr(env_init, "envs") else env_init
    args.lookback = getattr(env_ref, "lookback", getattr(args, "lookback", 30))
    args.input_dim = getattr(env_ref, "feat_dim", getattr(args, "input_dim", 6))

    # Find checkpoint to fine-tune from
    checkpoint_candidates = [
        getattr(args, "resume_model_path", None),
        getattr(args, "baseline_checkpoint", None),
    ]
    checkpoint_candidates = [p for p in checkpoint_candidates if p]
    checkpoint_path = next((p for p in checkpoint_candidates if os.path.exists(p)), None)

    if checkpoint_path is None:
        raise FileNotFoundError("No valid base checkpoint found for fine-tuning. Provide --resume_model_path or --baseline_checkpoint")

    print(f"Fine-tuning {checkpoint_path} using training month {train_month} ({train_start} to {train_end}) for {args.fine_tune_steps} timesteps")

    # Determine lookback
    lookback = getattr(env_init, 'lookback', getattr(args, 'lookback', 30))
    if hasattr(env_init, 'envs'):
         lookback = getattr(env_init.envs[0], 'lookback', lookback)
    
    policy_kwargs = dict(
        last_layer_dim_pi=args.num_stocks,
        last_layer_dim_vf=args.num_stocks,
        n_head=8,
        hidden_dim=128,
        no_ind=(not args.ind_yn),
        no_neg=(not args.neg_yn),
        lookback=lookback,
    )

    if getattr(args, "ptr_mode", True):
        print(f"Using PTR (Policy Transfer via Regularization) with coef={args.ptr_coef}")
        # Load the "prior" (frozen old policy)
        prior_model = load_weights_into_new_model(
            checkpoint_path,
            env_init,
            args.device,
            policy_kwargs,
            ptr_mode=False,
        )
        prior_policy = prior_model.policy
        # Load the "current" (trainable new policy)
        model = load_weights_into_new_model(
            checkpoint_path,
            env_init,
            args.device,
            policy_kwargs,
            ptr_mode=True,
            ptr_coef=args.ptr_coef,
            prior_policy=prior_policy,
        )
    else:
        # Standard PPO loading
        model = load_weights_into_new_model(
            checkpoint_path,
            env_init,
            args.device,
            policy_kwargs,
            ptr_mode=False,
        )

    model.set_env(env_init)
    model.learn(total_timesteps=getattr(args, "fine_tune_steps", 1))

    new_replay_samples = []
    if getattr(args, "ptr_mode", False):
        # Select high-value samples from the current month to carry forward
        new_replay_samples = select_replay_samples(model, env_init, train_dataset, k_percent=0.1)

    # save_dir already includes risk score (e.g., checkpoints_risk05/)
    os.makedirs(args.save_dir, exist_ok=True)
    month_slug = latest_month.replace("/", "-")
    out_path = os.path.join(args.save_dir, f"{args.model_name}_{month_slug}.zip")
    model.save(out_path)
    print(f"Saved fine-tuned checkpoint to {out_path}")

    # Evaluate the fine-tuned model on the monthly data for promotion decision
    print(f"Evaluating fine-tuned model for promotion gate...")
    
    # Temporarily set test dates to the month's date range for model_predict
    original_test_start = getattr(args, 'test_start_date', None)
    original_test_end = getattr(args, 'test_end_date', None)
    args.test_start_date = eval_start
    args.test_end_date = eval_end
    
    try:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=len(eval_dataset),
            pin_memory=True,
            collate_fn=lambda x: x,
            drop_last=True,
        )
        final_eval = model_predict(args, model, eval_loader, split="finetune_eval")
    finally:
        # Restore original test dates
        args.test_start_date = original_test_start
        args.test_end_date = original_test_end
    
    promotion_decision = apply_promotion_gate(
        args,
        out_path,
        final_eval.get("summary"),
        final_eval.get("log")
    )
    
    if promotion_decision.promoted:
        print(f"Model promoted to baseline: {getattr(args, 'baseline_checkpoint', 'N/A')}")
    else:
        print(f"Model NOT promoted. Reasons: {promotion_decision.reasons}")

    return out_path, new_replay_samples


def train_predict(args, predict_dt):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    data_dir = f'dataset_default/data_train_predict_{args.market}/{args.horizon}_{args.relation_type}/'
    train_dataset = AllGraphDataSampler(base_dir=data_dir, date=True,
                                        train_start_date=args.train_start_date, train_end_date=args.train_end_date,
                                        mode="train")
    val_dataset = AllGraphDataSampler(base_dir=data_dir, date=True,
                                      val_start_date=args.val_start_date, val_end_date=args.val_end_date,
                                      mode="val")
    test_dataset = AllGraphDataSampler(base_dir=data_dir, date=True,
                                       test_start_date=args.test_start_date, test_end_date=args.test_end_date,
                                       mode="test")
    train_loader_all = DataLoader(train_dataset, batch_size=len(train_dataset), pin_memory=True, collate_fn=lambda x: x,
                                  drop_last=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, collate_fn=lambda x: x,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), pin_memory=True)
    env_init = create_env_init(args, dataset=train_dataset)
    env_ref = env_init.envs[0] if hasattr(env_init, "envs") else env_init
    args.lookback = getattr(env_ref, "lookback", getattr(args, "lookback", 30))
    args.input_dim = getattr(env_ref, "feat_dim", getattr(args, "input_dim", 6))
    if args.policy == 'MLP':
        if getattr(args, 'resume_model_path', None) and os.path.exists(args.resume_model_path):
            print(f"Loading PPO model from {args.resume_model_path}")
            model = PPO.load(args.resume_model_path, env=env_init, device=args.device)
        else:
            model = PPO(policy='MlpPolicy',
                        env=env_init,
                        **PPO_PARAMS,
                        seed=args.seed,
                        device=args.device)
    elif args.policy == 'HGAT':
        lookback = getattr(env_init, 'lookback', getattr(args, 'lookback', 30))
        if hasattr(env_init, 'envs'):
             lookback = getattr(env_init.envs[0], 'lookback', lookback)

        policy_kwargs = dict(
            last_layer_dim_pi=args.num_stocks,
            last_layer_dim_vf=args.num_stocks,
            n_head=8,
            hidden_dim=128,
            no_ind=(not args.ind_yn),
            no_neg=(not args.neg_yn),
        )
        if getattr(args, 'resume_model_path', None) and os.path.exists(args.resume_model_path):
            print(f"Loading PPO model from {args.resume_model_path}")
            model = PPO.load(args.resume_model_path, env=env_init, device=args.device)
        else:
            model = PPO(policy=HGATActorCriticPolicy,
                        env=env_init,
                        policy_kwargs=policy_kwargs,
                        **PPO_PARAMS,
                        seed=args.seed,
                        device=args.device)
    # Initialize policy bias with FinRAG prior if available
    init_policy_bias_from_prior(model, getattr(args, "finrag_prior", None))
    train_model_and_predict(model, args, train_loader, val_loader, test_loader)

    # save_dir already includes risk score (e.g., checkpoints_risk05/)
    if getattr(args, "ptr_mode", False):
        print("Selecting initial replay samples from pre-training data...")
        env_selection = create_env_init(args, dataset=train_dataset)
        model.set_env(env_selection)
        
        initial_buffer = select_replay_samples(model, env_selection, train_dataset, k_percent=0.1)
        
        os.makedirs(args.save_dir, exist_ok=True)
        buffer_path = os.path.join(args.save_dir, f"replay_buffer_{args.market}.pkl")
        with open(buffer_path, "wb") as f:
            pickle.dump(initial_buffer, f)
        print(f"Saved initial replay buffer ({len(initial_buffer)} samples) to {buffer_path}")

    checkpoint_path = None
    try:
        os.makedirs(args.save_dir, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        checkpoint_path = os.path.join(
            args.save_dir,
            f"ppo_{args.policy.lower()}_{args.market}_{ts}.zip",
        )
        model.save(checkpoint_path)
        print(f"Saved pre-training checkpoint to {checkpoint_path}")

        # Copy to baseline.zip in the same risk-score directory
        baseline_path = os.path.join(args.save_dir, "baseline.zip")
        shutil.copy2(checkpoint_path, baseline_path)
        print(f"Updated baseline checkpoint at {baseline_path}")
    except Exception as exc:
        print(f"Failed to save pre-training checkpoint: {exc}")

    return model, checkpoint_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transaction ..")
    parser.add_argument("-device", "-d", default="cuda:0", help="gpu")
    parser.add_argument("-model_name", "-nm", default="Inter", help="Model name used in checkpoints and logs")
    parser.add_argument("-horizon", "-hrz", default="1", help="Return prediction horizon in trading days")
    parser.add_argument("-relation_type", "-rt", default="hy", help="Correlation relation type label (default: hy)")
    parser.add_argument("-ind_yn", "-ind", default="y", help="Enable industry relation graph")
    parser.add_argument("-pos_yn", "-pos", default="y", help="Enable momentum relation graph")
    parser.add_argument("-neg_yn", "-neg", default="y", help="Enable reversal relation graph")
    parser.add_argument("-multi_reward_yn", "-mr", default="y", help="Enable multi-reward IRL head")
    parser.add_argument("-policy", "-p", default="HGAT", help="Policy architecture identifier")
    # continual learning / resume
    parser.add_argument("--resume_model_path", default=None, help="Path to previously saved PPO model to resume from")
    parser.add_argument("--reward_net_path", default=None, help="Path to saved IRL reward network state_dict to resume from")
    parser.add_argument("--fine_tune_steps", type=int, default=5000, help="Timesteps for monthly fine-tuning when resuming")
    parser.add_argument("--save_dir", default="./checkpoints", help="Directory to save trained models")
    parser.add_argument("--baseline_checkpoint", default="./checkpoints/baseline.zip",
                        help="Destination checkpoint promoted after passing gating criteria")
    parser.add_argument("--promotion_min_sharpe", type=float, default=0.5,
                        help="Minimum Sharpe ratio required to promote a fine-tuned checkpoint")
    parser.add_argument("--promotion_max_drawdown", type=float, default=0.2,
                        help="Maximum acceptable drawdown (absolute fraction, e.g. 0.2 for 20%) for promotion")
    
    parser.add_argument("--run_monthly_fine_tune", action="store_true",
                        help="Run monthly fine-tuning")
    parser.add_argument("--finetune_month", default=None,
                        help="Optional month label (YYYY-MM) to finetune on; if not provided, uses latest available month")
    parser.add_argument("--discover_months_with_pathway", action="store_true",
                        help="group daily pickle files into monthly windows using Pathway")
    parser.add_argument("--month_cutoff_days", type=int, default=None,
                        help="Optional cutoff (days) to drop late daily files when building monthly shards via Pathway")
    parser.add_argument("--min_month_days", type=int, default=10,
                        help="Minimum number of daily files required to keep a discovered month window")
    parser.add_argument("--fine_tune_start_month", default=None,
                        help="Skip shards earlier than this month label (YYYY-MM) when selecting the next month")
    
    parser.add_argument("--expert_cache_path", default=None,
                        help="Optional path to cache expert trajectories for reuse")
    parser.add_argument("--num_expert_trajectories", type=int, default=700,
                        help="Number of expert trajectories to generate for IRL pretraining")
    parser.add_argument("--max_epochs", type=int, default=10, help="Number of IRL+RL epochs to run")
    parser.add_argument("--batch_size", type=int, default=512, help="Training batch size for loaders and IRL")
    parser.add_argument("--finrag_weights_path", default=None,
                        help="Path to FinRAG weights JSON used to initialize the policy prior")
    # Training hyperparameters
    parser.add_argument("--irl_epochs", type=int, default=30, help="Number of IRL training epochs")
    parser.add_argument("--rl_timesteps", type=int, default=10000, help="Number of RL timesteps for training")
    parser.add_argument(
        "--disable-tensorboard",
        action="store_true",
        help="Skip configuring TensorBoard logging to avoid importing the optional dependency.",
    )
    parser.add_argument("--n_steps", type=int, default=2048, help="Rollout horizon (environment steps) per PPO update cycle")

    # Risk-adaptive reward parameters
    parser.add_argument("--risk_score", type=float, default=0.1, help="User risk score: 0=conservative, 1=aggressive")
    parser.add_argument("--dd_base_weight", type=float, default=1.0, help="Base weight for drawdown penalty")
    parser.add_argument("--dd_risk_factor", type=float, default=1.0, help="Risk factor k in β_dd(ρ) = β_base*(1+k*(1-ρ))")

    # PTR (Policy Transfer Regularization) parameters
    parser.add_argument("--ptr_mode", action="store_true", help="Enable Policy Transfer via Regularization (PTR) for continual learning")
    parser.set_defaults(ptr_mode=True)
    parser.add_argument("--ptr_coef", type=float, default=0.3, help="Coefficient for PTR loss (KL divergence penalty)")
    parser.add_argument("--use_ptr", action="store_true", default=True, help="Backward-compatible alias for --ptr_mode")
    parser.add_argument("--ptr_memory_size", type=int, default=1000, help="Maximum number of samples retained in the PTR replay buffer")
    parser.add_argument("--ptr_priority_type", type=str, default="max", help="Replay buffer priority aggregation strategy")
    parser.add_argument("--stream", default=None, help="Pathway streaming flag/lock for reading streaming CSV")

    # Date ranges
    parser.add_argument("--train_start_date", default="2016-01-02", help="Start date for training")
    parser.add_argument("--train_end_date", default="2023-12-31", help="End date for training")
    parser.add_argument("--val_start_date", default="2024-01-02", help="Start date for validation")
    parser.add_argument("--val_end_date", default="2024-01-31", help="End date for validation")
    parser.add_argument("--test_start_date", default="2024-01-02", help="Start date for testing")
    parser.add_argument("--test_end_date", default="2024-12-31", help="End date for testing")
    parser.add_argument("--tickers_file", default="tickers.csv", help="Path to CSV file containing list of tickers")

    args = parser.parse_args()
    args.market = 'custom'
    if getattr(args, "use_ptr", False):
        args.ptr_mode = True

    PPO_PARAMS["batch_size"] = args.batch_size
    PPO_PARAMS["n_steps"] = args.n_steps

    if getattr(args, "disable_tensorboard", False):
        PPO_PARAMS["tensorboard_log"] = None
        print("TensorBoard logging disabled (--disable-tensorboard); PPO will not attempt to import tensorboard.")

    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.model_name = 'Falcon'
    args.relation_type = getattr(args, "relation_type", "hy") or "hy"
    args.seed = 123

    try:
        data_dir_detect = f'dataset_default/data_train_predict_{args.market}/{args.horizon}_{args.relation_type}/'
        sample_files_detect = [f for f in os.listdir(data_dir_detect) if f.endswith('.pkl')]
        if sample_files_detect:
            import pickle
            sample_path_detect = os.path.join(data_dir_detect, sample_files_detect[0])
            with open(sample_path_detect, 'rb') as f:
                sample_data_detect = pickle.load(f)
            # Expect features shaped [T, num_stocks, input_dim]
            feats = sample_data_detect.get('features')
            if feats is not None:
                try:
                    shape = feats.shape
                except Exception:
                    try:
                        shape = feats.size()
                    except Exception:
                        shape = None
                if shape and len(shape) >= 2:
                    args.input_dim = shape[-1]
                    print(f"Auto-detected input_dim: {args.input_dim}")
                else:
                    print("Warning: could not determine input_dim from sample; falling back to 6")
                    args.input_dim = 6
            else:
                print("Warning: 'features' not found in sample; falling back to input_dim=6")
                args.input_dim = 6
        else:
            print(f"Warning: No sample files found in {data_dir_detect}; falling back to input_dim=6")
            args.input_dim = 6
    except Exception as e:
        print(f"Warning: input_dim auto-detection failed ({e}); falling back to 6")
        args.input_dim = 6
    args.input_dim = 6
    args.ind_yn = True
    args.pos_yn = True
    args.neg_yn = True
    args.multi_reward = True
    args.irl_epochs = getattr(args, 'irl_epochs', 50)
    args.rl_timesteps = getattr(args, 'rl_timesteps', 10000)
    args.risk_score = getattr(args, 'risk_score', 0.5)
    args.dd_base_weight = getattr(args, 'dd_base_weight', 1.0)
    args.dd_risk_factor = getattr(args, 'dd_risk_factor', 1.0)
    args.risk_profile = build_risk_profile(args.risk_score)
    if not getattr(args, "expert_cache_path", None):
        args.expert_cache_path = os.path.join(
            "dataset_default",
            "expert_cache"
        )

    data_dir = f'dataset_default/data_train_predict_{args.market}/{args.horizon}_{args.relation_type}/'
    sample_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
    if sample_files:
        import pickle
        sample_path = os.path.join(data_dir, sample_files[0])
        with open(sample_path, 'rb') as f:
            sample_data = pickle.load(f)
        # features shape is [num_stocks, feature_dim], so use shape[0]
        args.num_stocks = sample_data['features'].shape[0]
        print(f"Auto-detected num_stocks for custom market: {args.num_stocks}")
    else:
        raise ValueError(f"No pickle files found in {data_dir} to determine num_stocks")

    args.finrag_prior = load_finrag_prior(args.finrag_weights_path, args.num_stocks)

    # Modify save_dir to include risk score (e.g., checkpoints -> checkpoints_risk05)
    risk_score = getattr(args, "risk_score", 0.5)
    args.save_dir = get_risk_score_dir(args.save_dir, risk_score)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set baseline checkpoint path (now just baseline.zip in the risk-score directory)
    args.baseline_checkpoint = os.path.join(args.save_dir, "baseline.zip")
    print(f"Using risk score: {risk_score} -> save_dir: {args.save_dir}")

    print("market:", args.market, "num_stocks:", args.num_stocks)
    if args.run_monthly_fine_tune:
        replay_buffer = []
        
        # Load initial buffer if available (in the risk-score directory)
        buffer_path = os.path.join(args.save_dir, f"replay_buffer_{args.market}.pkl")
        if os.path.exists(buffer_path):
            with open(buffer_path, "rb") as f:
                replay_buffer = pickle.load(f)
            print(f"Loaded initial replay buffer with {len(replay_buffer)} samples from {buffer_path}")
        
        # Use risk-score-based baseline as resume path if no explicit resume path provided
        if not getattr(args, "resume_model_path", None) or not os.path.exists(args.resume_model_path):
            if os.path.exists(args.baseline_checkpoint):
                args.resume_model_path = args.baseline_checkpoint
                print(f"Using baseline checkpoint as resume path: {args.resume_model_path}")
        
        # Call fine_tune_month once (fetches latest month and fine-tunes)
        # Pass finetune_month if specified via CLI
        checkpoint, new_samples = fine_tune_month(
            args, 
            replay_buffer=replay_buffer,
            finetune_month=getattr(args, "finetune_month", None)
        )
        print(f"Monthly fine-tuning complete. Checkpoint: {checkpoint}")
        
        # Update replay buffer with new samples
        if new_samples:
            replay_buffer.extend(new_samples)
            max_buffer = getattr(args, "ptr_memory_size", 500)
            if len(replay_buffer) > max_buffer:
                # Keep the most recent ones
                replay_buffer = replay_buffer[-max_buffer:]
            print(f"Replay buffer updated. Current size: {len(replay_buffer)}")
            with open(buffer_path, "wb") as f:
                pickle.dump(replay_buffer, f)
            print(f"Persisted replay buffer to {buffer_path}")
        # Update resume model path so that next run uses this checkpoint
        args.resume_model_path = checkpoint
    else:
        trained_model = train_predict(args, predict_dt='2024-12-30')
        # save PPO model checkpoint
        try:
            ts = time.strftime('%Y%m%d_%H%M%S')
            out_path = os.path.join(args.save_dir, f"ppo_{args.policy.lower()}_{args.market}_{ts}")
            print(f"Training run complete. To save PPO model, call model.save('{out_path}') where model is your PPO instance.")
        except Exception as e:
            print(f"Skip saving PPO model here: {e}")

        print(1)
