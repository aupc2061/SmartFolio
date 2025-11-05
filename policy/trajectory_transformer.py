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

    def set_adapter(self, adapter) -> None:
        """Attach a TrajectoryTransformerAdapter (pretrained/fine-tuned) to the policy.

        After attaching, the policy will use the transformer as its feature extractor
        instead of the default SimpleMLP. The adapter is expected to have a
        `.model` attribute (a HuggingFace TrajectoryTransformerModel) and a
        `_make_token_sequence(obs, act_idxs)` helper to convert observations to
        token sequences.
        """
        # Define an extractor that wraps the transformer adapter
        class TTExtractor(nn.Module):
            def __init__(self, adapter, pi_dim, vf_dim):
                super().__init__()
                self.adapter = adapter
                # Try to infer embedding dim from model config
                emb_dim = getattr(getattr(adapter, 'model', None), 'config', None)
                if emb_dim is not None and hasattr(adapter.model.config, 'n_embd'):
                    self.emb_dim = adapter.model.config.n_embd
                else:
                    # fallback to a reasonable default
                    self.emb_dim = getattr(adapter, 'n_embd', 128)
                self.proj_pi = nn.Linear(self.emb_dim, pi_dim)
                self.proj_vf = nn.Linear(self.emb_dim, vf_dim)
                self.latent_dim_pi = pi_dim
                self.latent_dim_vf = vf_dim

            def _obs_to_tokens(self, x: torch.Tensor) -> torch.LongTensor:
                # x: [B, obs_len] float tensor
                arr = x.detach().cpu().numpy()
                seqs = []
                # produce dummy action indices (zeros) for tokenization
                act_pad = [0] * getattr(self.adapter, 'action_dim', 1)
                for i in range(arr.shape[0]):
                    seq = self.adapter._make_token_sequence(arr[i], act_pad)
                    seqs.append(seq)
                seqs = np.stack(seqs, axis=0).astype(np.int64)
                return torch.from_numpy(seqs).long().to(self.adapter.device)

            def forward(self, x: torch.Tensor):
                # x: [B, obs_len]
                tokens = self._obs_to_tokens(x)
                self.adapter.model.eval()
                with torch.no_grad():
                    outputs = self.adapter.model(tokens, return_dict=True)
                # Try to get a pooled embedding: prefer last_hidden_state if available
                emb = None
                if hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                    # take last token embedding
                    emb = outputs.last_hidden_state[:, -1, :]
                elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    emb = outputs.hidden_states[-1][:, -1, :]
                elif hasattr(outputs, 'logits') and outputs.logits is not None:
                    # fallback: average logits across sequence
                    emb = outputs.logits.mean(dim=1)
                else:
                    raise RuntimeError('Transformer model did not return usable embeddings')

                return self.proj_pi(emb), self.proj_vf(emb)

            def forward_actor(self, x: torch.Tensor):
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
                return self.proj_pi(emb)

            def forward_critic(self, x: torch.Tensor):
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
                return self.proj_vf(emb)

        # attach extractor
        tt_extractor = TTExtractor(adapter, self.last_layer_dim_pi, self.last_layer_dim_vf)
        self.mlp_extractor = tt_extractor

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> tuple:
        # If a transformer adapter has been attached (mlp_extractor is TTExtractor),
        # perform autoregressive decoding using the transformer model directly.
        try:
            extractor = self.mlp_extractor
        except Exception:
            extractor = None

        # If extractor has adapter attribute, it's our TTExtractor
        if extractor is not None and hasattr(extractor, 'adapter'):
            adapter = extractor.adapter
            device = adapter.device

            # obs: torch tensor [B, obs_len]
            if isinstance(obs, np.ndarray):
                obs_t = torch.from_numpy(obs).float().to(device)
            else:
                obs_t = obs.float().to(device)

            B = obs_t.shape[0]
            action_dim = getattr(adapter, 'action_dim', 1)
            obs_dim = getattr(adapter, 'observation_dim', obs_t.shape[1])
            vocab_size = getattr(adapter, 'vocab_size', None)
            if vocab_size is None:
                vocab_size = getattr(adapter, 'vocab_size', 1)

            # create initial tokens with zeros in action positions
            # use adapter._make_token_sequence to obtain tokenized sequence for each sample
            seqs = []
            for i in range(B):
                arr = obs_t[i].detach().cpu().numpy()
                # pad with zeros for action indices
                act_pad = [0] * action_dim
                seq = adapter._make_token_sequence(arr, act_pad)
                seqs.append(seq)
            tokens = torch.from_numpy(np.stack(seqs, axis=0)).long().to(device)  # [B, seq_len]

            total_logprob = torch.zeros(B, device=device)
            selected_tokens = []

            # determine number of stocks from action_space when possible
            try:
                num_stocks = int(self.action_space.nvec[0])
            except Exception:
                num_stocks = getattr(adapter, 'num_stocks', None) or getattr(adapter, 'action_dim', 1)

            for k in range(action_dim):
                # run transformer to get logits
                adapter.model.eval()
                with torch.no_grad():
                    outputs = adapter.model(tokens, return_dict=True)

                # get logits for position corresponding to k-th action
                pos = obs_dim + k
                if hasattr(outputs, 'logits') and outputs.logits is not None:
                    logits_k = outputs.logits[:, pos, :]  # [B, vocab]
                else:
                    raise RuntimeError('Transformer did not return logits for autoregressive decoding')

                probs_k = torch.softmax(logits_k, dim=-1)
                from torch.distributions import Categorical
                cat = Categorical(probs_k)
                if deterministic:
                    sampled = torch.argmax(probs_k, dim=-1)
                else:
                    sampled = cat.sample()

                # compute log probability for selected tokens (works for both deterministic and stochastic)
                logp = cat.log_prob(sampled)
                total_logprob += logp

                # map sampled token ids back to stock indices (0..num_stocks-1)
                if num_stocks is not None:
                    sampled_stock_idx = (sampled % num_stocks).long()
                else:
                    sampled_stock_idx = sampled.long()

                # set sampled token into tokens for next iteration
                tokens[:, pos] = sampled
                selected_tokens.append(sampled_stock_idx.unsqueeze(1))

            # stack selected tokens -> actions shape [B, action_dim]
            actions = torch.cat(selected_tokens, dim=1)

            # Compute value using the transformer embedding from final tokens
            adapter.model.eval()
            with torch.no_grad():
                outputs = adapter.model(tokens, return_dict=True)
            # pool embedding
            if hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                emb = outputs.last_hidden_state[:, -1, :]
            elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                emb = outputs.hidden_states[-1][:, -1, :]
            else:
                emb = outputs.logits.mean(dim=1)

            # project to value using extractor's proj_vf if available
            if hasattr(extractor, 'proj_vf'):
                value_latent = extractor.proj_vf(emb)
                # reduce to scalar value
                value = value_latent.mean(dim=1, keepdim=True)
            else:
                # fallback: mean of embedding
                value = emb.mean(dim=1, keepdim=True)

            # SB3 expects actions as numpy array with shape (n_envs, action_dim)
            actions_np = actions.detach().cpu().numpy()
            values_np = value.detach().cpu().numpy()
            logp_np = total_logprob.detach().cpu().numpy()

            # Return tensors in the format (actions, values, log_prob)
            return torch.from_numpy(actions_np).to(device), torch.from_numpy(values_np).to(device), torch.from_numpy(logp_np).to(device)

        # fallback to default behavior
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

        # Calculate sequence length to match block_size for full context
        # This ensures we use the transformer's full attention window
        if self.block_size < self.observation_dim + self.action_dim + 1:
            print(f"\nWARNING: block_size ({self.block_size}) is smaller than minimum required sequence length "
                  f"({self.observation_dim + self.action_dim + 1}). Adjusting block_size.")
            self.block_size = self.observation_dim + self.action_dim + 1
        
        # Use full block_size as sequence length to maintain transformer's context window
        self.seq_length = self.block_size

        if model_name is not None:
            # Load pretrained model when requested
            self.model = TrajectoryTransformerModel.from_pretrained(model_name)
        else:
            # Build config with block_size matching our sequence length
            cfg_block_size = self.block_size
            config = TrajectoryTransformerConfig(
                vocab_size=self.vocab_size,
                action_dim=self.action_dim,
                observation_dim=self.observation_dim,
                block_size=cfg_block_size,
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
        """Convert one observation vector and action indices into integer token sequence.
        
        The sequence structure is:
        [observation_tokens | action_tokens | EOS_token | padding_tokens]
        where:
        - observation_tokens: scaled and discretized observation values
        - action_tokens: action indices mapped to vocab space
        - EOS_token: end of sequence marker (1)
        - padding_tokens: zeros to fill up to block_size
        """
        # obs: (observation_dim,)
        # Scale obs to 0..vocab_size-1
        scaled = (obs - self.obs_min) / self.obs_ranges
        tokens_obs = np.floor(scaled * (self.vocab_size - 1)).astype(int)
        tokens_obs = np.clip(tokens_obs, 0, self.vocab_size - 1)

        # actions indices are already ints, map them into token space by modulo
        tokens_act = np.array([int(x) % self.vocab_size for x in act_idxs], dtype=int)

        # Create the core sequence with meaningful tokens
        # Use token 1 as EOS marker (0 is reserved for padding)
        core_seq = np.concatenate([tokens_obs, tokens_act, np.array([1], dtype=int)])
        
        # Pad with zeros up to block_size for transformer's attention window
        if core_seq.shape[0] < self.seq_length:
            pad = np.zeros(self.seq_length - core_seq.shape[0], dtype=int)
            seq = np.concatenate([core_seq, pad])
        else:
            # This shouldn't happen with our size checks in __init__
            print(f"\nWARNING: Sequence length {core_seq.shape[0]} exceeds block_size {self.seq_length}")
            seq = core_seq[: self.seq_length]
        
        return seq

    def prepare_dataset(self, expert_trajectories: List[Tuple[np.ndarray, np.ndarray]]):
        print(f"\nDEBUG: TrajectoryTransformerAdapter preparing dataset:")
        print(f"- observation_dim: {self.observation_dim}")
        print(f"- action_dim: {self.action_dim}")
        print(f"- seq_length: {self.seq_length}")
        print(f"- block_size: {self.block_size}")
        print(f"- vocab_size: {self.vocab_size}")
        print(f"- num trajectories: {len(expert_trajectories)}")
        if expert_trajectories:
            print(f"- example state shape: {expert_trajectories[0][0].shape}")
            print(f"- example action shape: {expert_trajectories[0][1].shape}")

        # Calculate meaningful sequence length (no padding)
        min_seq_len = self.observation_dim + self.action_dim + 1  # +1 for EOS token
        print(f"- meaningful sequence length: {min_seq_len}")
        
        obs_arr, act_arr = self._discretize_dataset(expert_trajectories)
        print(f"- obs_arr shape after discretize: {obs_arr.shape}")
        print(f"- act_arr shape after discretize: {act_arr.shape}")
        
        sequences = []
        targets = []
        for i in range(len(obs_arr)):
            seq = self._make_token_sequence(obs_arr[i], act_arr[i].tolist())
            # For targets, we use -100 for padding positions so they're ignored by the loss
            target = seq.copy()
            target[min_seq_len:] = -100  # Mark padding positions to be ignored by loss
            sequences.append(seq)
            targets.append(target)

        X = np.stack(sequences, axis=0).astype(np.int64)
        Y = np.stack(targets, axis=0).astype(np.int64)
        print(f"- sequences tensor shape (X): {X.shape}")
        print(f"- targets tensor shape (Y): {Y.shape}")
        return torch.from_numpy(X), torch.from_numpy(Y)

    def fine_tune(self, expert_trajectories: List[Tuple[np.ndarray, np.ndarray]],
                  epochs: int = 3, batch_size: int = 32, lr: float = 1e-4):
        print("\nDEBUG: Starting fine-tuning:")
        print(f"- epochs: {epochs}")
        print(f"- batch_size: {batch_size}")
        print(f"- learning rate: {lr}")
        
        X, Y = self.prepare_dataset(expert_trajectories)
        dataset = torch.utils.data.TensorDataset(X, Y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print(f"- dataset size: {len(dataset)}")
        print(f"- num batches: {len(loader)}")

        optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_f = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_idx, (xb, yb) in enumerate(loader):
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                print(f"\nBatch {batch_idx+1}/{len(loader)}:")
                print(f"- input batch (xb): {xb.shape}")
                print(f"- target batch (yb): {yb.shape}")
                outputs = self.model(xb, targets=yb, return_dict=True)
                # outputs.logits: (B, seq_len, vocab)
                if hasattr(outputs, 'logits') and outputs.logits is not None:
                    print(f"\nDEBUG Tensor shapes:")
                    print(f"- model output logits: {outputs.logits.shape}")
                    print(f"- model config block_size: {self.model.config.block_size}")
                    print(f"- adapter block_size: {self.block_size}")
                    print(f"- adapter seq_length: {self.seq_length}")
                    print(f"- vocab_size: {self.vocab_size}")
                    
                    # The model outputs (batch, seq, vocab) but we only want to compute loss
                    # on the actual sequence positions, not the padding
                    min_seq_len = self.observation_dim + self.action_dim + 1
                    logits = outputs.logits[:, :min_seq_len, :]  # Only take predictions up to actual sequence
                    print(f"- trimmed logits shape: {logits.shape}")
                    
                    # Reshape both tensors to 2D for cross entropy: (batch*seq, vocab) and (batch*seq)
                    logits = logits.view(-1, logits.size(-1))
                    targets = yb[:, :min_seq_len].contiguous().view(-1)  # Match the logits trimming
                    print(f"- reshaped logits: {logits.shape}")
                    print(f"- reshaped targets: {targets.shape}")
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
