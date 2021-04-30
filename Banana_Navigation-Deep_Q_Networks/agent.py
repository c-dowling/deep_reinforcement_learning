import numpy as np
import random
from collections import deque, namedtuple, defaultdict

from model import QNetwork, DuellingQNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, fc1_units, fc2_units, buffer_size, batch_size, alpha, gamma, tau,
                 local_update_every, target_update_every, seed, a=1, b=1, dbl_dqn=False, priority_rpl=False, duel_dqn=False):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            _____________
            _____________
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Hyperparameters
        self.alpha = alpha                              # Learning rate
        self.gamma = gamma                              # Discount parameter
        self.tau = tau                                  # Interpolation parameter
        self.local_update_every = local_update_every    # Number of actions to take before updating local net weights
        self.target_update_every = target_update_every  # Number of actions to take before updating target net weights
        self.batch_size = batch_size                    # Number of experiences to sample during learning
        self.buffer_size = buffer_size                  # Size of memory buffer
        self.a = a                                      # Sampling probability (0=random | 1=priority)
        self.b = b                                      # Influence of importance sampling weights over learning

        # Agent modifications
        self.dbl_dqn = dbl_dqn                          # Double Q Learning
        self.priority_rpl = priority_rpl                # Prioritised Experience Replay
        self.duel_dqn = duel_dqn                        # Duelling Q Networks

        # Q-Network
        if self.duel_dqn:
            self.qnetwork_local = DuellingQNetwork(state_size, action_size, fc1_units, fc2_units, seed).to(device)
            self.qnetwork_target = DuellingQNetwork(state_size, action_size, fc1_units, fc2_units, seed).to(device)
        else:
            self.qnetwork_local = QNetwork(state_size, action_size, fc1_units, fc2_units, seed).to(device)
            self.qnetwork_target = QNetwork(state_size, action_size, fc1_units, fc2_units, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.alpha)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed, priority_rpl)
        # Initialize time step (for updating every local_update_every/target_update_every steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every local_update_every time steps.
        self.t_step = (self.t_step + 1)
        if self.t_step % self.local_update_every == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)
        if self.t_step % self.target_update_every == 0:
            self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

        #Update eps and b
        #self.eps = max(self.eps_end, self.eps_decay*self.eps)
        #self.b = 1-self.eps

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        (states, actions, rewards, next_states, dones), indices = experiences

        # Get max predicted Q values (for next states) from target model
        if self.dbl_dqn:
            local_best_actions = self.qnetwork_local(next_states).detach().argmax(1)
            Q_next_states = self.qnetwork_target(next_states)
            Q_targets_next = Q_next_states.gather(1, local_best_actions.unsqueeze(1))
        else:
            Q_targets_next = self.qnetwork_target(next_states).detach().argmax(1).unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        if self.priority_rpl:
            errors = abs(Q_expected - Q_targets)
            self.memory.update_priorities(indices, errors)
            importance = [self.memory.importance[idx] for idx in indices]
            importance = np.array(importance)**self.b
            loss = torch.mean(torch.mul(errors.float(), torch.from_numpy(importance).float().to(device)))
        else:
            loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, is_priority):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.is_priority = is_priority
        self.action_size = action_size
        self.memory = deque(maxlen=int(buffer_size))
        self.priorities = deque(maxlen=int(buffer_size))
        self.importance = deque(maxlen=int(buffer_size))
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        self.priorities.append(max(self.priorities, default=1))   # Save all new experiences as maximum priority

    def get_probs(self, a):
        """Gets the sampling probabilities associated with each experience."""
        with torch.no_grad():
            probabilities = (np.array(self.priorities) ** a) / sum(np.array(self.priorities) ** a)
        return probabilities

    def sample(self, a=1):
        """Randomly sample a batch of experiences from memory."""
        if self.is_priority:
            probabilities = self.get_probs(a)
            experiences_idx = random.choices(range(len(self.memory)), k=self.batch_size, weights=probabilities)
            experiences = np.array(self.memory)[experiences_idx]
            importance = 1/len(self.memory) * 1/probabilities               # Calculate our importance sampling weight
            importance_norm = importance / max(importance)
            for idx, imp in zip(experiences_idx, importance_norm):
                self.importance[idx] = imp
        else:
            experiences_idx = random.sample(range(len(self.memory)), k=self.batch_size)
            experiences = [self.memory[idx] for idx in experiences_idx]

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones), experiences_idx

    def update_priorities(self, indices, errors):
        for idx, err in zip(indices, errors):
            self.priorities[idx] = abs(err) + 0.001

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)