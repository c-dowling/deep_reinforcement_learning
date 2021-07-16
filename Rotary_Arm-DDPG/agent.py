import random
import numpy as np
import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque

from model import ActorNet, CriticNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, fc1_units, fc2_units, seed, gamma, lr_actor, lr_critic, tau,
                 buffer_size, batch_size, weight_decay):
        """Initialize an Agent object.

        Params
        ======
            state_size (int):                           dimension of each state
            action_size (int):                          dimension of each action
            fc1_units (int):                            number of nodes in layer 1 of neural network
            fc2_units (int):                            number of nodes in layer 2 of neural network
            seed (int):                                 seed
            gamma (float):                              discount parameter
            lr_actor (float):                           learning rate for Actor
            lr_critic (float):                          leanring rate for Critic
            tau (float):                                interpolation parameter
            buffer_size (int):                          size of memory buffer
            batch_size (int):                           number of experiences to sample during learning
            weight_decay (int):                         weight decay parameter
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.tau = tau
        self.batch_size = batch_size

        # Neural Netowrk Params
        self.actor_target = ActorNet(state_size, action_size, fc1_units, fc2_units, seed).to(device)
        self.actor_local = ActorNet(state_size, action_size, fc1_units, fc2_units, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor)
        self.critic_target = CriticNet(state_size, action_size, fc1_units, fc2_units, seed).to(device)
        self.critic_local = CriticNet(state_size, action_size, fc1_units, fc2_units, seed).to(device)
        self.critic_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_critic)

        # Noise process
        self.noise = OUNoise(action_size, seed)

        # Memory buffer
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn if there are enough samples for a batch
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, self.gamma)

    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state)
        self.actor_local.train()

        if add_noise:
            action += torch.as_tensor(self.noise.sample()).float().to(device)
        return torch.clamp(action, -1, 1).cpu().data.numpy().tolist()

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        ### UPDATE CRITIC ###
        # Get predicted Q values (for next states) from target model
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute loss (critic)
        Q_expected = self.critic_local(states, actions.float())
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimise the loss (critic)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Use grad clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        ### UPDATE ACTOR ###
        # Compute loss (actor)
        actions_pred = self.actor_local(states)
        actor_loss = self.critic_local(states, actions_pred).mean()

        # Minimise the loss (actor)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # Use grad clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        self.actor_optimizer.step()

        self.soft_update(self.actor_local, self.actor_target, self.tau)
        self.soft_update(self.critic_local, self.critic_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            is_priority (bool): if "True" will sample episodes using prioritised experience replay
        """
        self.action_size = action_size
        self.memory = deque(maxlen=int(buffer_size))
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, a=1):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)