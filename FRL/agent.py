# -*- coding: utf-8 -*-
# @Time    : 2021/10/25 14:42
# @Author  : Weiming Mai
# @FileName: agent.py
# @Software: PyCharm
import numpy as np
from torch.nn.utils import clip_grad_norm_
import torch
from torch import nn
from Utils.Memory import replay_buffer
from Network import MLP
import torch.optim as optim

def try_gpu(): #single gpu
    i = 0
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

class DQN():
    def __init__(self, state_dim, action_dim, epsilon, batch_size, capacity, gamma, lr):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon  # could be enhance
        self.device = try_gpu()
        self.memory = replay_buffer(capacity)
        self.batch_size = batch_size
        self.policy_net = MLP(state_dim, action_dim).to(self.device)
        self.target_net = MLP(state_dim, action_dim).to(self.device)
        self.UpdateTarget()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        #         self.optimizer = optim.SGD(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def choose_action(self, state):  # state.type: np array
        if np.random.random() > self.epsilon:
            with torch.no_grad():
                # state = torch.from_numpy(state).to(torch.float32)
                state = torch.tensor(state, device=self.device, dtype=torch.float32)  # or
                q_val = self.policy_net(state)
                action = q_val.argmax().item()  # argmax->tensor
        else:
            action = np.random.randint(self.action_dim)
        return action

    def predict(self, state):  # same as above
        with torch.no_grad():
            #             state = torch.from_numpy(state).to(torch.float32)
            state = torch.tensor(state, device=self.device, dtype=torch.float32)  # or
            q_val = self.policy_net(state)
            action = q_val.argmax().item()  # argmax->tensor
        return action

    def UpdateTarget(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def UpdateQ(self):
        if len(self.memory) < self.batch_size:
            return
        '''
        action: interger, .argmax().item()
        '''
        state_batch, action_batch, reward_batch, n_state_batch, done_batch = self.memory.sample(
            self.batch_size)
        state_batch = torch.tensor(
            state_batch, device=self.device, dtype=torch.float)
        action_batch = torch.tensor(
            action_batch, device=self.device).view(-1, 1)  # dtype has to be int64, for 'gather'
        reward_batch = torch.tensor(
            reward_batch, device=self.device, dtype=torch.float).view(-1, 1)
        n_state_batch = torch.tensor(
            n_state_batch, device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device, dtype=torch.float).view(-1, 1)
        current_q_val = self.policy_net(state_batch).gather(dim=1, index=action_batch)  # Q(s,a)
        max_target_q_val = self.target_net(n_state_batch).max(1)[0].detach().view(-1, 1)         # detach the gradient
        y_hat = reward_batch + self.gamma * max_target_q_val * (1 - done_batch)
        loss = self.loss_fn(current_q_val, y_hat)
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.policy_net.parameters(), max_norm=20, norm_type=2)
        self.optimizer.step()
