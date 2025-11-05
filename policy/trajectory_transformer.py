
import math
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional

try:
    from transformers import TrajectoryTransformerModel
    from transformers import TrajectoryTransformerConfig
except Exception:
    TrajectoryTransformerModel = None
    TrajectoryTransformerConfig = None

from stable_baselines3.common.policies import ActorCriticPolicy
import gym
import torch as th
import torch.nn as nn


class TrajectoryTransformerActorCriticPolicy(ActorCriticPolicy):
    """
    ActorCriticPolicy compatible with Stable-Baselines3 that can use a
    TrajectoryTransformerAdapter as its feature extractor.
    """

    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 lr_schedule,
                 net_arch=None,
                 activation_fn=nn.Tanh,
                 *args,
                 **kwargs):
        self.last_layer_dim_pi = kwargs.pop('last_layer_dim_pi', 64)
        self.last_layer_dim_vf = kwargs.pop('last_layer_dim_vf', 64)
        super().__init__(observation_space, action_space, lr_schedule, net_arch, activation_fn, *args, **kwargs)
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        feature_dim = int(self.observation_space.shape[0]) if hasattr(self.observation_space, 'shape') else 0

        class SimpleMLP(nn.Module):
            def __init__(self, in_dim, pi_dim, vf_dim):
                super().__init__()
                hidden = max(64, pi_dim)
                self.fc = nn.Sequential(
                    nn.Linear(in_dim, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                )
                self.latent_pi = nn.Linear(hidden, pi_dim)
                self.latent_vf = nn.Linear(hidden, vf_dim)
                self.latent_dim_pi = pi_dim
                self.latent_dim_vf = vf_dim

            def forward(self, x):
                h = self.fc(x)
                return self.latent_pi(h), self.latent_vf(h)

            def forward_actor(self, x):
                return self.latent_pi(self.fc(x))

            def forward_critic(self, x):
                return self.latent_vf(self.fc(x))

        self.mlp_extractor = SimpleMLP(feature_dim, self.last_layer_dim_pi, self.last_layer_dim_vf)

    def set_adapter(self, adapter) -> None:
        class TTExtractor(nn.Module):
            def __init__(self, adapter, pi_dim, vf_dim):
                super().__init__()
                self.adapter = adapter
                emb_dim = getattr(getattr(adapter, 'model', None), 'config', None)
                if emb_dim is not None and hasattr(adapter.model.config, 'n_embd'):
                    self.emb_dim = adapter.model.config.n_embd
                else:
                    self.emb_dim = getattr(adapter, 'n_embd', 128)
                self.proj_pi = nn.Linear(self.emb_dim, pi_dim)
                self.proj_vf = nn.Linear(self.emb_dim, vf_dim)
                self.latent_dim_pi = pi_dim
                self.latent_dim_vf = vf_dim

            def _obs_to_tokens(self, x: torch.Tensor) -> torch.LongTensor:
                arr = x.detach().cpu().numpy()
                seqs = []
                act_pad = [0] * getattr(self.adapter, 'action_dim', 1)
                for i in range(arr.shape[0]):
                    seq = self.adapter._make_token_sequence(arr[i], act_pad)
                    seqs.append(seq)
                return torch.from_numpy(np.stack(seqs, axis=0)).long().to(self.adapter.device)

            def forward(self, x):
                tokens = self._obs_to_tokens(x)
                self.adapter.model.eval()
                with torch.no_grad():
                    outputs = self.adapter.model(tokens, return_dict=True)
                if hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                    emb = outputs.last_hidden_state[:, -1, :]
                elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    emb = outputs.hidden_states[-1][:, -1, :]
                else:
                    emb = outputs.logits.mean(dim=1)
                return self.proj_pi(emb), self.proj_vf(emb)

            def forward_actor(self, x):
                pi, _ = self.forward(x)
                return pi

            def forward_critic(self, x):
                _, vf = self.forward(x)
                return vf

        self.mlp_extractor = TTExtractor(adapter, self.last_layer_dim_pi, self.last_layer_dim_vf)

    def forward(self, obs: th.Tensor, deterministic: bool = False):
        extractor = getattr(self, 'mlp_extractor', None)
        if extractor is not None and hasattr(extractor, 'adapter'):
            adapter = extractor.adapter
            device = adapter.device
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)

            B = obs_t.shape[0]
            action_dim = getattr(adapter, 'action_dim', 1)
            obs_dim = getattr(adapter, 'observation_dim', obs_t.shape[1])
            seqs = []
            for i in range(B):
                seq = adapter._make_token_sequence(obs_t[i].cpu().numpy(), [0] * action_dim)
                seqs.append(seq)
            tokens = torch.from_numpy(np.stack(seqs, axis=0)).long().to(device)

            total_logprob = torch.zeros(B, device=device)
            selected_tokens = []

            for k in range(action_dim):
                adapter.model.eval()
                with torch.no_grad():
                    outputs = adapter.model(tokens, return_dict=True)
                logits_k = outputs.logits[:, obs_dim + k, :]
                probs_k = torch.softmax(logits_k, dim=-1)
                cat = torch.distributions.Categorical(probs_k)
                sampled = cat.sample()
                total_logprob += cat.log_prob(sampled)
                tokens[:, obs_dim + k] = sampled
                selected_tokens.append(sampled.unsqueeze(1))

            actions = torch.cat(selected_tokens, dim=1)
            with torch.no_grad():
                outputs = adapter.model(tokens, return_dict=True)
            emb = outputs.last_hidden_state[:, -1, :] if hasattr(outputs, 'last_hidden_state') else outputs.logits.mean(dim=1)
            value = extractor.proj_vf(emb).mean(dim=1, keepdim=True)

            return actions, value, total_logprob

        return super().forward(obs, deterministic)

    def _predict(self, observation, deterministic: bool = False):
        out = self.forward(observation, deterministic)
        actions = out[0] if isinstance(out, tuple) else out
        return actions


class TrajectoryTransformerAdapter:
    """Adapter around Hugging Face TrajectoryTransformerModel."""

    def __init__(self, model_name=None, device=None, vocab_size=100, action_dim=6,
                 observation_dim=17, block_size=249, n_layer=4, n_head=4, n_embd=128,
                 resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1):
        if TrajectoryTransformerModel is None or TrajectoryTransformerConfig is None:
            raise ImportError("transformers>=4.40 required with TrajectoryTransformerModel")

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab_size = vocab_size
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.block_size = max(block_size, observation_dim + action_dim + 1)
        self.seq_length = self.block_size

        if model_name:
            self.model = TrajectoryTransformerModel.from_pretrained(model_name)
        else:
            config = TrajectoryTransformerConfig(
                vocab_size=self.vocab_size,
                action_dim=self.action_dim,
                observation_dim=self.observation_dim,
                block_size=self.block_size,
                n_layer=n_layer,
                n_head=n_head,
                n_embd=n_embd,
                resid_pdrop=resid_pdrop,
                embd_pdrop=embd_pdrop,
                attn_pdrop=attn_pdrop,
            )
            self.model = TrajectoryTransformerModel(config)

        self.model.to(self.device)
        self.model.train()

    def _discretize_dataset(self, expert_trajectories):
        obs_list, act_list = [], []
        for state, action in expert_trajectories:
            state = np.asarray(state).flatten()
            obs = np.zeros(self.observation_dim)
            obs[:min(len(state), self.observation_dim)] = state[:self.observation_dim]
            obs_list.append(obs)
            action = np.asarray(action).astype(int)
            idxs = np.where(action == 1)[0].tolist()
            idxs = idxs[:self.action_dim] + [0] * (self.action_dim - len(idxs))
            act_list.append(idxs)
        obs_arr = np.stack(obs_list)
        self.obs_min = obs_arr.min(axis=0)
        self.obs_max = obs_arr.max(axis=0)
        ranges = self.obs_max - self.obs_min
        ranges[ranges == 0] = 1.0
        self.obs_ranges = ranges
        return obs_arr, np.array(act_list, dtype=int)

    def _make_token_sequence(self, obs: np.ndarray, act_idxs: List[int]) -> np.ndarray:
        scaled = (obs - self.obs_min) / self.obs_ranges
        tokens_obs = np.floor(scaled * (self.vocab_size - 1)).astype(int)
        tokens_act = np.array([int(x) % self.vocab_size for x in act_idxs], dtype=int)
        seq = np.concatenate([tokens_obs, tokens_act, np.array([1], dtype=int)])
        if len(seq) < self.seq_length:
            seq = np.concatenate([seq, np.zeros(self.seq_length - len(seq), dtype=int)])
        else:
            seq = seq[:self.seq_length]
        return seq

    def prepare_dataset(self, expert_trajectories):
        obs_arr, act_arr = self._discretize_dataset(expert_trajectories)
        sequences, targets = [], []
        for i in range(len(obs_arr)):
            seq = self._make_token_sequence(obs_arr[i], act_arr[i])
            target = seq.copy()
            target[self.observation_dim + self.action_dim + 1:] = -100
            sequences.append(seq)
            targets.append(target)
        X = torch.tensor(np.stack(sequences), dtype=torch.long)
        Y = torch.tensor(np.stack(targets), dtype=torch.long)
        return X, Y

    def fine_tune(self, expert_trajectories, epochs=3, batch_size=32, lr=1e-4):
        X, Y = self.prepare_dataset(expert_trajectories)
        dataset = torch.utils.data.TensorDataset(X, Y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_f = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                outputs = self.model(xb, targets=yb, return_dict=True)
                logits = outputs.logits[:, :self.observation_dim + self.action_dim + 1, :]
                loss = loss_f(logits.reshape(-1, logits.size(-1)), yb[:, :self.observation_dim + self.action_dim + 1].reshape(-1))
                optim.zero_grad()
                loss.backward()
                optim.step()
                total_loss += loss.item()
            print(f"TT fine-tune epoch {epoch+1}/{epochs}, loss={total_loss/len(loader):.4f}")

    def predict(self, sequences: torch.Tensor):
        self.model.eval()
        with torch.no_grad():
            seq = sequences.to(self.device)
            outputs = self.model(seq, use_cache=True, return_dict=True)
        preds = outputs.logits.argmax(dim=-1)
        return outputs, preds
