import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from network import PolicyNetwork, ActorCriticNetwork
import torch.nn.functional as F

class PolicyGradientAgent:
    def __init__(self, input_dim: int, output_dim: int, learning_rate: float = 0.01):
        self.policy_network = PolicyNetwork(input_dim, output_dim)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.rewards = []
        self.log_probs = []

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_probs = self.policy_network(state)
        action = torch.multinomial(action_probs, num_samples=1)
        self.log_probs.append(torch.log(action_probs[0, action]))
        return action.item()

    def update_policy(self):
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        policy_loss = []
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        self.rewards = []
        self.log_probs = []

    def store_reward(self, reward: float):
        self.rewards.append(reward)

    def save(self, filepath: str):
        torch.save(self.policy_network.state_dict(), filepath)

    def load(self, filepath: str):
        self.policy_network.load_state_dict(torch.load(filepath))

class PPOAgent:
    def __init__(self, input_dim: int, output_dim: int, learning_rate: float = 3e-4, gamma: float = 0.99, epsilon: float = 0.2, value_coef: float = 0.5, entropy_coef: float = 0.01):
        self.actor_critic = ActorCriticNetwork(input_dim, output_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_probs, _ = self.actor_critic(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), action_probs[0, action]

    def update(self, states, actions, old_log_probs, old_action_probs, rewards, next_states, dones):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        old_action_probs = torch.FloatTensor(old_action_probs)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        _, next_values = self.actor_critic(next_states)
        returns = self.compute_returns(rewards, next_values, dones)
        advantages = returns - self.actor_critic(states)[1].detach()

        for _ in range(10):  
            action_probs, state_values = self.actor_critic(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(state_values.squeeze(), returns)
            entropy = dist.entropy().mean()
            
            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def compute_returns(self, rewards, next_values, dones):
        returns = []
        R = next_values[-1] * (1 - dones[-1])
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * (1 - dones[step])
            returns.insert(0, R)
        return torch.tensor(returns)

    def save(self, filepath: str):
        torch.save(self.actor_critic.state_dict(), filepath)

    def load(self, filepath: str):
        self.actor_critic.load_state_dict(torch.load(filepath))