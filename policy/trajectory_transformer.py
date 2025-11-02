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
    A light-weight ActorCriticPolicy compatible with Stable-Baselines3 that
    can either wrap the TrajectoryTransformerAdapter (if available) or fall
    back to a small MLP extractor. This provides a drop-in policy to select
    with `--policy TT` from `main.py`.

    NOTE: This is intentionally minimal: it implements `_build_mlp_extractor`
    to set `self.mlp_extractor` and keeps `ortho_init=False` similar to the
    HGAT policy in `policy/policy.py`.
    """

    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 lr_schedule,
                 net_arch=None,
                 activation_fn=nn.Tanh,
                 *args,
                 **kwargs):
        # allow passing last layer dims (for compatibility with main.py usage)
        self.last_layer_dim_pi = kwargs.pop('last_layer_dim_pi', 64)
        self.last_layer_dim_vf = kwargs.pop('last_layer_dim_vf', 64)

        super(TrajectoryTransformerActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            **kwargs,
        )

        # Keep same behavior as HGAT policy
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        # Build a simple MLP extractor that produces latent features for actor and critic
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

            # Stable Baselines3 may call `forward_actor` / `forward_critic`
            # on the extractor. Provide these for compatibility.
            def forward_actor(self, x):
                h = self.fc(x)
                return self.latent_pi(h)

            def forward_critic(self, x):
                h = self.fc(x)
                return self.latent_vf(h)

        self.mlp_extractor = SimpleMLP(feature_dim, self.last_layer_dim_pi, self.last_layer_dim_vf)
        try:
            self.mlp_extractor.latent_dim_pi = self.last_layer_dim_pi
            self.mlp_extractor.latent_dim_vf = self.last_layer_dim_vf
        except Exception:
            pass

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> tuple:
        return super().forward(obs, deterministic)

    def _predict(self, observation, deterministic: bool = False) -> th.Tensor:
        # Delegate to parent implementation which will use the policy network
        actions, values, log_prob = self.forward(observation, deterministic)
        return actions


class TrajectoryTransformerAdapter:
    """
    Adapter to train / run a Hugging Face TrajectoryTransformerModel on the
    expert trajectories produced by `gen_data.generate_expert_trajectories_ga`.

    Assumptions / tokenization strategy (simple and deterministic):
      - Each expert trajectory is a tuple (state, action_multi_hot).
      - We take the first `observation_dim` elements of the flattened state
        as observation tokens. If fewer, we pad with zeros.
      - The multi-hot action is converted into a fixed-length list of
        `action_dim` indices: the indices of selected stocks (padded with 0).
      - Continuous values are discretized into integer tokens in [0, vocab_size)
        by linear scaling using per-feature min/max computed from the dataset.

    This performs a light fine-tuning (cross-entropy on logits vs targets)
    and provides `predict` to obtain token predictions.
    """

    def __init__(self,
                 model_name: Optional[str] = None,
                 device: Optional[str] = None,
                 vocab_size: int = 100,
                 action_dim: int = 6,
                 observation_dim: int = 17,
                 block_size: int = 249,
                 n_layer: int = 4,
                 n_head: int = 4,
                 n_embd: int = 128,
                 resid_pdrop: float = 0.1,
                 embd_pdrop: float = 0.1,
                 attn_pdrop: float = 0.1,
                 **kwargs):
        if TrajectoryTransformerModel is None or TrajectoryTransformerConfig is None:
            raise ImportError("transformers with TrajectoryTransformer is required. Install a recent `transformers` package.")

        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab_size = vocab_size
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.block_size = block_size

        # Sequence token length used by example: obs + action + 1
        self.seq_length = self.observation_dim + self.action_dim + 1

        if model_name is not None:
            # Load pretrained model when requested
            self.model = TrajectoryTransformerModel.from_pretrained(model_name)
        else:
            # Build config from provided hyperparameters
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

    def _discretize_dataset(self, expert_trajectories: List[Tuple[np.ndarray, np.ndarray]]):
        """Compute min/max and discretize observations to integer tokens."""
        # Build array of shape (N, observation_dim)
        obs_list = []
        act_list = []
        for state, action in expert_trajectories:
            state = np.asarray(state).flatten()
            # take first observation_dim values
            if state.shape[0] >= self.observation_dim:
                obs = state[:self.observation_dim]
            else:
                obs = np.zeros(self.observation_dim, dtype=float)
                obs[: state.shape[0]] = state
            obs_list.append(obs.astype(float))

            # action: multi-hot vector -> list of indices (length action_dim)
            action = np.asarray(action).astype(int)
            idxs = np.where(action == 1)[0].tolist()
            if len(idxs) >= self.action_dim:
                idxs = idxs[: self.action_dim]
            else:
                # pad with zeros
                idxs = idxs + [0] * (self.action_dim - len(idxs))
            act_list.append(idxs)

        obs_arr = np.stack(obs_list, axis=0) if len(obs_list) > 0 else np.zeros((0, self.observation_dim))

        # Compute per-feature min/max
        self.obs_min = obs_arr.min(axis=0) if obs_arr.size else np.zeros(self.observation_dim)
        self.obs_max = obs_arr.max(axis=0) if obs_arr.size else np.ones(self.observation_dim)
        # Avoid zero range
        ranges = self.obs_max - self.obs_min
        ranges[ranges == 0] = 1.0
        self.obs_ranges = ranges

        return obs_arr, np.array(act_list, dtype=int)

    def _make_token_sequence(self, obs: np.ndarray, act_idxs: List[int]) -> np.ndarray:
        """Convert one observation vector and action indices into integer token sequence."""
        # obs: (observation_dim,)
        # Scale obs to 0..vocab_size-1
        scaled = (obs - self.obs_min) / self.obs_ranges
        tokens_obs = np.floor(scaled * (self.vocab_size - 1)).astype(int)
        tokens_obs = np.clip(tokens_obs, 0, self.vocab_size - 1)

        # actions indices are already ints, map them into token space by modulo
        tokens_act = np.array([int(x) % self.vocab_size for x in act_idxs], dtype=int)

        seq = np.concatenate([tokens_obs, tokens_act, np.array([0], dtype=int)])
        # Ensure length equals seq_length
        if seq.shape[0] < self.seq_length:
            pad = np.zeros(self.seq_length - seq.shape[0], dtype=int)
            seq = np.concatenate([seq, pad])
        elif seq.shape[0] > self.seq_length:
            seq = seq[: self.seq_length]
        return seq

    def prepare_dataset(self, expert_trajectories: List[Tuple[np.ndarray, np.ndarray]]):
        obs_arr, act_arr = self._discretize_dataset(expert_trajectories)
        sequences = []
        targets = []
        for i in range(len(obs_arr)):
            seq = self._make_token_sequence(obs_arr[i], act_arr[i].tolist())
            # For language modeling-style training we can set targets = seq (teacher forcing)
            sequences.append(seq)
            targets.append(seq.copy())

        X = np.stack(sequences, axis=0).astype(np.int64)
        Y = np.stack(targets, axis=0).astype(np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y)

    def fine_tune(self, expert_trajectories: List[Tuple[np.ndarray, np.ndarray]],
                  epochs: int = 3, batch_size: int = 32, lr: float = 1e-4):
        X, Y = self.prepare_dataset(expert_trajectories)
        dataset = torch.utils.data.TensorDataset(X, Y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_f = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                outputs = self.model(xb, targets=yb, return_dict=True)
                # outputs.logits: (B, seq_len, vocab)
                if hasattr(outputs, 'logits') and outputs.logits is not None:
                    logits = outputs.logits.view(-1, outputs.logits.size(-1))
                    targets = yb.view(-1)
                    loss = loss_f(logits, targets)
                else:
                    # fallback: try outputs.loss
                    loss = getattr(outputs, 'loss', None)
                    if loss is None:
                        raise RuntimeError('Model did not return logits or loss')

                optim.zero_grad()
                loss.backward()
                optim.step()
                epoch_loss += float(loss.detach().cpu().numpy())
            # print progress
            print(f"TT fine-tune epoch {epoch+1}/{epochs}, loss={epoch_loss/len(loader):.4f}")

    def predict(self, sequences: torch.Tensor):
        """Run the model in eval mode on integer sequences and return logits and preds."""
        self.model.eval()
        with torch.no_grad():
            seq = sequences.to(self.device)
            outputs = self.model(seq, use_cache=True, return_dict=True)
        preds = None
        if hasattr(outputs, 'logits') and outputs.logits is not None:
            preds = outputs.logits.argmax(dim=-1)
        return outputs, preds
