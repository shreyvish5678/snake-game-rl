import gym
import torch
import torch.nn as nn
import gym_game
from policy_gradient_agent import PPOAgent, PolicyGradientAgent
env = gym.make('Snake-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent_vanilla = PolicyGradientAgent(state_dim, action_dim)
agent_ppo = PPOAgent(state_dim, action_dim)
agent_vanilla.policy_network.load_state_dict(torch.load('policy_snake_best.pth'))
agent_ppo.actor_critic.load_state_dict(torch.load('ppo_snake_best.pth'))
state, _ = env.reset()

from tqdm import tqdm
total_reward_vanilla = 0
total_reward_ppo = 0
total_score_vanilla = 0 
total_score_ppo = 0
for n in tqdm(range(100)):
    state, _ = env.reset()
    for _ in range(500):
        action = agent_vanilla.select_action(state)
        next_state, reward, done, score, _ = env.step(action)
        total_reward_vanilla += reward
        state = next_state
        if done:
            break
    total_score_vanilla += score
    state, _ = env.reset()
    for _ in range(500):
        action, _, _ = agent_ppo.select_action(state)
        next_state, reward, done, score, _ = env.step(action)
        total_reward_ppo += reward
        state = next_state
        if done:
            break
    total_score_ppo += score
print(f"Average Reward Vanilla: {total_reward_vanilla / 100}")
print(f"Average Reward PPO: {total_reward_ppo / 100}")
print(f"Average Score Vanilla: {total_score_vanilla / 100}")
print(f"Average Score PPO: {total_score_ppo / 100}")