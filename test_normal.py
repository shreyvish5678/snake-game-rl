import gym
import torch
import torch.nn as nn
import gym_game
from policy_gradient_agent import PolicyGradientAgent
env = gym.make('Snake-v1', render_mode='human')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = PolicyGradientAgent(state_dim, action_dim)
model_path = 'policy_snake_best.pth'
agent.policy_network.load_state_dict(torch.load(model_path))
state, _ = env.reset()
fps = 100
import time 
for n in range(500):
    action = agent.select_action(state)
    next_state, reward, done, _, _ = env.step(action)
    state = next_state
    time.sleep(1/fps)
    if done:
        break