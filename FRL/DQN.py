import gym
from torch import nn
from torch.nn.utils import clip_grad_norm_
import torch
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from collections import deque
import random

def try_gpu(): #single gpu
    i = 0
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


class MLP(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        l1 = F.relu(self.fc1(x))
        l2 = F.relu(self.fc2(l1))
        output = self.fc3(l2)

        return output


class replay_buffer():
    def __init__(self, capacity):
        self.buffer = deque()
        self.capacity = capacity
        self.count = 0

    def add(self, state, action, reward, n_state, done):  # done: whether the final state, TD error would be different.
        experience = (state, action, reward, n_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def sample(self, batch_size):  # return a tuple
        batch = random.sample(self.buffer, batch_size)  # a list [(s,a,r,s), ...]
        return zip(*batch)

    #         return batch

    def clear(self):
        self.buffer.clear()
        self.count = 0

    def __len__(self):
        return len(self.buffer)

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
        max_target_q_val = self.target_net(state_batch).detach().view(-1, 1)
        y_hat = reward_batch + self.gamma * max_target_q_val * (1 - done_batch)
        loss = self.loss_fn(current_q_val, y_hat)
        self.optimizer.zero_grad()
        loss.backward()
        #         clip_grad_norm_(self.policy_net.parameters(), max_norm=20, norm_type=2)
        self.optimizer.step()


env = gym.make('CartPole-v0')
state_dim = env.observation_space.shape[0]  # 4
action_dim = env.action_space.n  # 2
epsilon = 0.01
batch_size = 64
gamma = 0.99
lr = 0.01
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