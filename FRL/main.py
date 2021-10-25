import gym
from torch import nn
from torch.nn.utils import clip_grad_norm_
import torch
import torch.nn.functional as F
import numpy as np



def train():




env = gym.make('CartPole-v0')
state_dim = env.observation_space.shape[0]  # 4
action_dim = env.action_space.n  # 2
epsilon = 0.01
batch_size = 64
gamma = 0.98
lr = 0.002
episode_num = 500
capacity = 10000
agent = DQN(state_dim, action_dim, epsilon, batch_size, capacity, gamma, lr)
return_list = []
C_iter = 5

for i_ep in range(episode_num):
    state = env.reset()
    done = False
    ep_reward = 0
    #     print(state)
    while True:
        action = agent.choose_action(state)
        n_state, reward, done, _ = env.step(action)
        ep_reward += reward
        agent.memory.add(state, action, reward, n_state, done)
        agent.UpdateQ()
        state = n_state
        #         print(n_state, action)
        if done == True:
            break
    if i_ep % C_iter == 0:
        agent.UpdateTarget()

    if i_ep % 10 == 0:
        print(ep_reward)
    return_list.append(ep_reward)