"""Actor-Critic Networks with Replay Buffer for HRLAD Anomaly Detection.

DDPG-style actor-critic architecture with target networks, replay buffer,
and epsilon-greedy exploration for anomaly detection in time series data.

Reference: Paper Table III - Network Architecture Specification
"""

import random
from collections import deque
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Action constants
# ---------------------------------------------------------------------------
ACTION_NORMAL = 0
ACTION_WARNING = 1
ACTION_ALARM = 2
N_ACTIONS = 3


# ---------------------------------------------------------------------------
# Actor Network
# ---------------------------------------------------------------------------
class ActorNetwork(nn.Module):
    """Policy network that maps states to action probabilities.

    Architecture (Paper Table III):
        Input(state_dim) -> 256 -> 128 -> 64 -> Output(n_actions)
        With ReLU, Dropout(0.1), BatchNorm1d, final Softmax.

    Args:
        state_dim: Dimension of the state vector (default 8).
        n_actions: Number of discrete actions (default 3).
    """

    def __init__(self, state_dim: int = 8, n_actions: int = N_ACTIONS):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions

        self.fc1 = nn.Linear(state_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.1)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.1)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.drop3 = nn.Dropout(0.1)

        self.fc_out = nn.Linear(64, n_actions)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass producing action probabilities.

        Args:
            state: Shape (batch, state_dim) or (state_dim,).

        Returns:
            Action probabilities of shape (batch, n_actions).
        """
        x = state
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # BatchNorm1d requires batch_size > 1 during training;
        # temporarily use running stats for single-sample batches.
        was_training = self.training
        if x.size(0) == 1:
            self.eval()

        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.drop3(F.relu(self.bn3(self.fc3(x))))
        x = self.fc_out(x)

        if was_training:
            self.train()

        return F.softmax(x, dim=-1)


# ---------------------------------------------------------------------------
# Critic Network
# ---------------------------------------------------------------------------
class CriticNetwork(nn.Module):
    """Q-value network that maps (state, action) pairs to a scalar value.

    Architecture (Paper Table III):
        Input(state_dim + n_actions) -> 256 -> 128 -> 64 -> Output(1)
        Actions are one-hot encoded before concatenation.
        With ReLU, Dropout(0.1), BatchNorm1d.

    Args:
        state_dim: Dimension of the state vector (default 8).
        n_actions: Number of discrete actions (default 3).
    """

    def __init__(self, state_dim: int = 8, n_actions: int = N_ACTIONS):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions

        input_dim = state_dim + n_actions

        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.1)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.1)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.drop3 = nn.Dropout(0.1)

        self.fc_out = nn.Linear(64, 1)

    def forward(self, state: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        """Forward pass producing a Q-value.

        Args:
            state: Shape (batch, state_dim) or (state_dim,).
            action_onehot: One-hot encoded action, shape (batch, n_actions)
                or (n_actions,).

        Returns:
            Q-value of shape (batch, 1).
        """
        s = state
        a = action_onehot
        if s.dim() == 1:
            s = s.unsqueeze(0)
        if a.dim() == 1:
            a = a.unsqueeze(0)

        x = torch.cat([s, a], dim=-1)

        # BatchNorm1d requires batch_size > 1 during training;
        # temporarily use running stats for single-sample batches.
        was_training = self.training
        if x.size(0) == 1:
            self.eval()

        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.drop3(F.relu(self.bn3(self.fc3(x))))
        x = self.fc_out(x)

        if was_training:
            self.train()

        return x


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """Fixed-capacity experience replay buffer.

    Stores transitions (state, action, reward, next_state, done) and
    provides uniform random sampling for off-policy learning.

    Args:
        capacity: Maximum number of transitions to store (default 50 000).
    """

    def __init__(self, capacity: int = 50000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        """Store a transition."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a random batch of transitions.

        Returns:
            Tuple of tensors (states, actions, rewards, next_states, dones).
        """
        batch = random.sample(self.buffer, batch_size)

        states = torch.stack([t[0] for t in batch])
        actions = torch.tensor([t[1] for t in batch], dtype=torch.long)
        rewards = torch.tensor([t[2] for t in batch], dtype=torch.float32)
        next_states = torch.stack([t[3] for t in batch])
        dones = torch.tensor([t[4] for t in batch], dtype=torch.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# DDPG Agent
# ---------------------------------------------------------------------------
class DDPGAgent:
    """DDPG-style agent with actor-critic networks and target copies.

    Implements:
    - Epsilon-greedy exploration with linear decay
    - Soft (Polyak) target network updates
    - Huber loss for critic training
    - Gradient clipping at 1.0

    Args:
        state_dim: Dimension of the state vector.
        n_actions: Number of discrete actions.
        actor_lr: Actor learning rate (default 3e-4).
        critic_lr: Critic learning rate (default 1e-3).
        tau: Polyak averaging coefficient (default 0.005).
        gamma: Discount factor (default 0.99).
        buffer_capacity: Replay buffer capacity (default 50 000).
        epsilon_start: Initial exploration rate (default 1.0).
        epsilon_end: Final exploration rate (default 0.01).
        epsilon_decay_episodes: Episodes over which to decay (default 200).
        target_update_freq: Steps between target updates (default 5).
        device: Torch device; auto-detected if None.
    """

    def __init__(
        self,
        state_dim: int = 8,
        n_actions: int = N_ACTIONS,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        tau: float = 0.005,
        gamma: float = 0.99,
        buffer_capacity: int = 50000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_episodes: int = 200,
        target_update_freq: int = 5,
        device: torch.device | None = None,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.tau = tau
        self.gamma = gamma
        self.target_update_freq = target_update_freq

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Networks
        self.actor = ActorNetwork(state_dim, n_actions).to(self.device)
        self.critic = CriticNetwork(state_dim, n_actions).to(self.device)
        self.target_actor = ActorNetwork(state_dim, n_actions).to(self.device)
        self.target_critic = CriticNetwork(state_dim, n_actions).to(self.device)

        # Copy weights to targets
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # Internal counters
        self._train_steps = 0

    # ----- epsilon schedule ------------------------------------------------
    def _epsilon(self, episode: int) -> float:
        """Linear decay from epsilon_start to epsilon_end."""
        if episode >= self.epsilon_decay_episodes:
            return self.epsilon_end
        ratio = episode / self.epsilon_decay_episodes
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * ratio

    # ----- action selection ------------------------------------------------
    def select_action(self, state: torch.Tensor, epsilon: float) -> int:
        """Select action with epsilon-greedy exploration.

        With probability *epsilon* a uniform random action is chosen;
        otherwise the actor's highest-probability action is used.

        Args:
            state: State tensor of shape (state_dim,).
            epsilon: Current exploration rate.

        Returns:
            Integer action index.
        """
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)

        with torch.no_grad():
            state = state.to(self.device)
            probs = self.actor(state)
            return probs.argmax(dim=-1).item()

    def get_policy_probs(self, state: torch.Tensor) -> np.ndarray:
        """Return action probabilities for a single state.

        Args:
            state: State tensor of shape (state_dim,).

        Returns:
            NumPy array of shape (n_actions,).
        """
        with torch.no_grad():
            state = state.to(self.device)
            probs = self.actor(state)
            return probs.squeeze(0).cpu().numpy()

    # ----- network update --------------------------------------------------
    def soft_update_targets(self) -> None:
        """Polyak-averaging target update: θ' ← τ·θ + (1-τ)·θ'."""
        with torch.no_grad():
            for param, target_param in zip(
                self.actor.parameters(), self.target_actor.parameters()
            ):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * param.data)

            for param, target_param in zip(
                self.critic.parameters(), self.target_critic.parameters()
            ):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * param.data)

    def update(self, batch_size: int) -> dict:
        """Perform one DDPG update step.

        1. Sample mini-batch from replay buffer.
        2. Compute target Q via target networks.
        3. Update critic with Huber loss.
        4. Update actor via policy gradient through critic.
        5. Soft-update target networks.
        6. Clip gradients at 1.0.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Dict with 'actor_loss' and 'critic_loss' scalars.
        """
        if len(self.replay_buffer) < batch_size:
            return {"actor_loss": 0.0, "critic_loss": 0.0}

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            batch_size
        )

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # One-hot encode actions
        action_onehot = F.one_hot(actions, num_classes=self.n_actions).float()

        # --- Critic update --------------------------------------------------
        with torch.no_grad():
            next_probs = self.target_actor(next_states)
            # Differentiable sampling: use next_probs directly as soft actions
            next_q = self.target_critic(next_states, next_probs).squeeze(-1)
            target_q = rewards + self.gamma * (1.0 - dones) * next_q

        current_q = self.critic(states, action_onehot).squeeze(-1)
        critic_loss = F.huber_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # --- Actor update ---------------------------------------------------
        # Use Gumbel-softmax for differentiable action sampling during training
        actor_logits = self.actor(states)
        soft_actions = F.gumbel_softmax(
            actor_logits, tau=1.0, hard=False
        )
        actor_q = self.critic(states, soft_actions).squeeze(-1)
        actor_loss = -actor_q.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # --- Target update --------------------------------------------------
        self._train_steps += 1
        if self._train_steps % self.target_update_freq == 0:
            self.soft_update_targets()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
        }

    # ----- episode training ------------------------------------------------
    def train_episode(
        self,
        env_data: list,
        episode: int,
        batch_size: int = 64,
    ) -> dict:
        """Train for one episode using provided environment data.

        Args:
            env_data: List of (state, reward, next_state, done) tuples.
                Each state is a torch.Tensor of shape (state_dim,).
            episode: Current episode number (used for epsilon schedule).
            batch_size: Mini-batch size for updates.

        Returns:
            Dict with episode metrics.
        """
        epsilon = self._epsilon(episode)
        total_reward = 0.0
        total_loss = {"actor_loss": 0.0, "critic_loss": 0.0}
        n_steps = 0

        for state, reward, next_state, done in env_data:
            state = state.to(self.device)
            next_state = next_state.to(self.device)

            action = self.select_action(state, epsilon)
            self.replay_buffer.push(state, action, reward, next_state, done)

            loss_dict = self.update(batch_size)
            total_loss["actor_loss"] += loss_dict["actor_loss"]
            total_loss["critic_loss"] += loss_dict["critic_loss"]
            total_reward += reward
            n_steps += 1

        if n_steps > 0:
            total_loss["actor_loss"] /= n_steps
            total_loss["critic_loss"] /= n_steps

        return {
            "episode": episode,
            "epsilon": epsilon,
            "total_reward": total_reward,
            "n_steps": n_steps,
            **total_loss,
        }
