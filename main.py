import os
import time
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pandas as pd
import torch
print(torch.cuda.is_available())
from dataloader.data_loader import *
from policy.policy import *
from policy.trajectory_transformer import TrajectoryTransformerActorCriticPolicy, TrajectoryTransformerAdapter
# from trainer.trainer import *
from stable_baselines3 import PPO
from trainer.irl_trainer import *
from torch_geometric.loader import DataLoader
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3 import TD3

PATH_DATA = f'./dataset/'

class TT_TD3Policy(TD3Policy):
    """
    TD3-compatible policy that can use a TrajectoryTransformer adapter.
    Fully compatible with SB3 (no squash_output / device assignment issues).
    """

    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        # Drop unsupported kwargs from PPO-style configs
        for key in ["last_layer_dim_pi", "last_layer_dim_vf", "n_head", "hidden_dim", "no_ind", "no_neg"]:
            kwargs.pop(key, None)

        # Initialize these BEFORE calling super().__init__
        self.adapter = None
        self.adapter_device = None
        self.ortho_init = False
        
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def set_adapter(self, adapter):
        """Attach a TrajectoryTransformerAdapter."""
        self.adapter = adapter
        self.adapter_device = adapter.device
        print(f"[TT_TD3Policy] Attached transformer adapter on {self.adapter_device}")

        emb_dim = getattr(adapter.model.config, "n_embd", 128)
        
        # Use net_arch to get features_dim, or fallback to a default
        features_dim = getattr(self, 'features_dim', 256)
        
        self.extractor = nn.Sequential(
            nn.Linear(emb_dim, features_dim),
            nn.ReLU(),
        ).to(self.adapter_device)

    def extract_features(self, obs: torch.Tensor) -> torch.Tensor:
        """Use transformer embeddings as features if adapter is attached."""
        if self.adapter is None:
            return super().extract_features(obs)

        device = self.adapter_device or next(self.parameters()).device
        self.adapter.model.eval()
        with torch.no_grad():
            obs_np = obs.detach().cpu().numpy() if isinstance(obs, torch.Tensor) else obs
            seqs = [self.adapter._make_token_sequence(x, [0] * self.adapter.action_dim) for x in obs_np]
            tokens = torch.tensor(seqs, dtype=torch.long, device=device)
            outputs = self.adapter.model(tokens, return_dict=True)
            emb = outputs.last_hidden_state[:, -1, :]
        
        # Move embeddings to the correct device for the extractor
        return self.extractor(emb.to(next(self.extractor.parameters()).device))

    def forward(self, obs, deterministic: bool = False):
        """TD3 forward pass using transformer extractor if available."""
        features = self.extract_features(obs)
        mean_actions = self.actor(features)
        if deterministic:
            return mean_actions
        noise = torch.normal(0, self.actor.noise_std, size=mean_actions.shape).to(mean_actions.device)
        return torch.tanh(mean_actions + noise)


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
    print(len(train_loader), len(val_loader), len(test_loader))

    env_init = create_env_init(args, dataset=train_dataset)
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
    elif args.policy == 'TT' or args.policy == 'TRAJ':
        # Use TD3 but attach transformer adapter
        TD3_PARAMS = dict(buffer_size=100000, learning_rate=1e-3, batch_size=256,
                        tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1)
        policy_kwargs = dict(
            last_layer_dim_pi=args.num_stocks,
            last_layer_dim_vf=args.num_stocks,
        )
        if getattr(args, 'resume_model_path', None) and os.path.exists(args.resume_model_path):
            print(f"Loading TD3 model from {args.resume_model_path}")
            model = TD3.load(args.resume_model_path, env=env_init, device=args.device)
        else:
            model = TD3(policy=TT_TD3Policy,
                        env=env_init,
                        policy_kwargs=policy_kwargs,
                        seed=args.seed,
                        verbose=1,
                        device=args.device,
                        **TD3_PARAMS)

        # Attach trajectory transformer adapter
        try:
            adapter = TrajectoryTransformerAdapter(
                vocab_size=128,
                action_dim=args.num_stocks,
                observation_dim=args.input_dim * args.num_stocks,  # Flattened observation
                block_size=256,
                n_layer=4,
                n_head=4,
                n_embd=128,
                device=args.device if isinstance(args.device, str)
                    else ("cuda" if torch.cuda.is_available() else "cpu"),
            )
            print("Computing normalization bounds from training dataset...")
            adapter.set_normalization_from_dataset(train_dataset)
            
            model.policy.set_adapter(adapter)
            print("Attached TrajectoryTransformerAdapter to TD3 policy")
        except Exception as e:
            print(f"Warning: failed to attach transformer adapter to TD3 policy: {e}")

    train_model_and_predict(model, args, train_loader, val_loader, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transaction ..")
    parser.add_argument("-device", "-d", default="cuda:0", help="gpu")
    parser.add_argument("-model_name", "-nm", default="SmartFolio", help="模型名称")
    parser.add_argument("-market", "-mkt", default="hs300", help="股票市场")
    parser.add_argument("-horizon", "-hrz", default="1", help="预测距离")
    parser.add_argument("-relation_type", "-rt", default="hy", help="股票关系类型")
    parser.add_argument("-ind_yn", "-ind", default="y", help="是否加入行业关系图")
    parser.add_argument("-pos_yn", "-pos", default="y", help="是否加入动量关系图")
    parser.add_argument("-neg_yn", "-neg", default="y", help="是否加入反转关系图")
    parser.add_argument("-multi_reward_yn", "-mr", default="y", help="是否加入多奖励学习")
    parser.add_argument("-policy", "-p", default="TT", help="策略网络")
    parser.add_argument("--resume_model_path", default=None)
    parser.add_argument("--reward_net_path", default=None)
    parser.add_argument("--fine_tune_steps", type=int, default=5000)
    parser.add_argument("--save_dir", default="./checkpoints")
    parser.add_argument("--irl_epochs", type=int, default=50)
    parser.add_argument("--rl_timesteps", type=int, default=10000)
    parser.add_argument("--ga_generations", type=int, default=30)
    args = parser.parse_args()

    # Default run setup
    args.model_name = 'SmartFolio'
    args.market = 'hs300'
    args.relation_type = 'hy'
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.train_start_date = '2019-01-02'
    args.train_end_date = '2022-12-30'
    args.val_start_date = '2023-01-03'
    args.val_end_date = '2023-12-29'
    args.test_start_date = '2024-01-02'
    args.test_end_date = '2024-12-30'
    args.batch_size = 32
    args.max_epochs = 5
    args.seed = 123
    args.input_dim = 6
    args.ind_yn = True
    args.pos_yn = True
    args.neg_yn = True
    args.multi_reward = True
    args.use_ga_expert = True
    os.makedirs(args.save_dir, exist_ok=True)

    if args.market == 'hs300':
        args.num_stocks = 102
    elif args.market == 'zz500':
        args.num_stocks = 80
    elif args.market == 'nd100':
        args.num_stocks = 84
    elif args.market == 'sp500':
        args.num_stocks = 472
    elif args.market == 'custom':
        data_dir = f'dataset_default/data_train_predict_{args.market}/{args.horizon}_{args.relation_type}/'
        sample_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
        if sample_files:
            import pickle
            sample_path = os.path.join(data_dir, sample_files[0])
            with open(sample_path, 'rb') as f:
                sample_data = pickle.load(f)
            args.num_stocks = sample_data['features'].shape[1]
            print(f"Auto-detected num_stocks: {args.num_stocks}")
        else:
            raise ValueError(f"No pickle files found in {data_dir}")

    trained_model = train_predict(args, predict_dt='2025-02-05')
    print("Training complete.")
