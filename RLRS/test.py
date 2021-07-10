import time
start = time.time()
from processing import train_data, embed
end = time.time()
print(end - start)

import numpy as np
from random import randint
import pandas as pd
from functools import partial
import torch
from torch import nn
from torch.nn import functional as act
from torch.distributions import Categorical
from torch.autograd import Variable

class itemEnv():
    def __init__(self, alpha=0.4, sigma=0.9):  # albha parameter in consine similarity
        self.observation_space = train_data
        self.item_info = embed
        self.current_state = None
        self.init_state = self.reset()
        self.sigma = sigma
        self.alpha = alpha
        self.embedding_dim = 7
        self.reward_val = 1

    #         self.rewards, self.avg_state, self.avg_action, self.nx_size = self.avg_group()

    def reset(self):
        init_state = self.observation_space['embedded_state'].sample(1).values[0]
        #         init_state = self.observation_space.loc[7,'embedded_state']
        self.current_state = init_state
        return init_state

    #     def feedback(self, pair):  # pair: state-action pair
    #         prob = list()
    #         denominator = 0.
    #         max_prob = 0.
    #         result = 0.
    #         feed_back = ""
    #         for r, s, a in zip(self.rewards, self.avg_state, self.avg_action):
    #             numerator = np.dot(pair[0], s.T) / np.linalg.norm(pair[0], 2) + np.dot(pair[1], a.T) / np.linalg.norm(pair[1], 2)
    #             denominator += numerator
    #             prob.append(numerator)
    #             if numerator > max_prob:
    #                 max_prob = numerator
    #                 feed_back = r
    #         prob = prob/denominator
    # #         for p, r in zip(prob, self.rewards):
    # #             for k in range(1):
    # #                 result += p * self.reward_val * np.power(self.sigma, k) * int(r.split(',')[k])

    #         result = self.reward_val * int(feed_back.split(',')[0])
    #         print(prob)
    #         return feed_back, result

    @staticmethod
    def cos_sim(group, pair):
        max_prob = 0.
        # nx_size = len(group[1])
        s_i = np.array(group[1]['embedded_state'].values.tolist())  # 2 dim array:0-->sample, 1-->feature_num (N,(7*12))
        s_t = pair[0]  # 1 dim array:0-->sample, (7*12,)
        a_i = np.array(group[1]['embedded_action'].values.tolist())  # (N,9,7)
        a_t = pair[1]  # (7,)
        norm_si = np.linalg.norm(s_i, 2, axis=1)  # 1 dim (N,)
        norm_st = np.linalg.norm(s_t, 2, axis=0)  # 1 dim (1,)
        norm_ai = np.linalg.norm(a_i, 2, axis=2)  # 2 dim (N, 9)
        norm_at = np.linalg.norm(a_t, 2, axis=0)  # 1 dim (1,)

        # first term: (N,1), second term: (N,9)
        cos = env.alpha * (np.dot(s_i, s_t) / (norm_si * norm_st + 1e-10))[:, np.newaxis] + (1 - env.alpha) * np.dot(
            a_i, a_t) / (norm_ai * norm_at + 1e-10)  # 1 dim
        # cos = cos / np.sum(cos)
        # cos: (N,9)
        max_id = np.argmax(cos)
        id_x = int(max_id / cos.shape[1])
        id_y = max_id % cos.shape[1]
        # print('i am here!')
        return (np.max(cos), id_x, id_y)

    def feedback_2(self, pair):
        data = train_data.groupby(['reward'])
        #         denominator = 0.
        reward_ind = -1
        max_prob = 0.
        result = 0.
        feed_back = ""
        fnc = partial(self.cos_sim, pair=pair)
        prob = map(fnc, data)
        for tup, group in zip(prob, data):
            if tup[0] > max_prob:
                max_prob = tup[0]
                reward_ind = tup[2]  # id_y
                feed_back = group[0]  # index: reward str
                #                 feed_back = group[1].iloc[tup[1], 2]
                #                 idx = tup[1]

        r = 1 * int(feed_back.split(',')[reward_ind])  # reward_val
        result = r if r > 0 else -1

        return feed_back, result

    def step(self, action):  # action: 1 dim array
        feed_back, result = self.feedback_2((self.current_state, action))
        #         for i,r in enumerate([feed_back.split(',')[0]]): #0---> one action
        if result == 1:  # reward_val
            tmp = np.append(self.current_state, action)
            tmp = tmp[self.embedding_dim:]
            self.current_state = tmp  # state: 1 dim array
        # else:
        #             self.current_state = self.observation_space['embedded_state'].sample(1).values[0]

        return result, self.current_state, feed_back


def gen_action(item_id):
    return embed[item_id]

def try_gpu():
    i = 0
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


class LSTM(nn.Module):
    def __init__(self, inpt_sz, hidden_sz, hidden_l1, oupt_sz):  # embedding dim, , , 381
        super().__init__()
        # input_gate
        device = try_gpu()
        init_weights = partial(self.init_, device=device)
        #         self.init_state = None
        self.reward = []
        self.inpt_sz = inpt_sz
        self.hidden_sz = hidden_sz
        self.hidden_l1 = hidden_l1
        self.oupt_sz = oupt_sz
        self.W_xi, self.W_hi, self.bias_i = (init_weights((inpt_sz, hidden_sz)), init_weights(
            (hidden_sz, hidden_sz)), init_weights((1, hidden_sz))
                                             )
        # forget_gate
        self.W_xf, self.W_hf, self.bias_f = (init_weights((inpt_sz, hidden_sz)), init_weights(
            (hidden_sz, hidden_sz)), init_weights((1, hidden_sz))
                                             )
        # output_gate
        self.W_xo, self.W_ho, self.bias_o = (init_weights((inpt_sz, hidden_sz)), init_weights(
            (hidden_sz, hidden_sz)), init_weights((1, hidden_sz))
                                             )
        # candidate memory cell
        self.W_xc, self.W_hc, self.bias_c = (init_weights((inpt_sz, hidden_sz)), init_weights(
            (hidden_sz, hidden_sz)), init_weights((1, hidden_sz))
                                             )
        # output_layer
        self.W_o1, self.W_o2, self.bias_o1, self.bias_o2 = (init_weights((hidden_sz, hidden_l1)), init_weights(
            (hidden_l1, oupt_sz)), init_weights((1, hidden_l1)),
                                                            init_weights((1, oupt_sz))
                                                            )

    @staticmethod
    def init_(shape, device):
        #         param = torch.tensor(shape)
        def xvaier(param):
            return nn.init.xavier_uniform_(param)

        param = xvaier(torch.rand(shape, device=device))
        return nn.Parameter(param)

    def forward(self, X, init_state=None):  # X: batch size, seq size, input size
        batch_size, seq_size, _ = X.shape
        hidden_sz = self.hidden_sz
        #         oupts = []
        if init_state == None:  # H, C actually are constant, not trainable
            H, C = (torch.zeros(batch_size, hidden_sz, device=X.device),
                    torch.zeros(batch_size, hidden_sz, device=X.device)
                    )
        else:
            H, C = init_state  # in some circumstance
        # softmax = act.softmax(dim = 1)
        for seq in range(seq_size):
            x_t = X[:, seq, :]
            I = torch.sigmoid(x_t @ self.W_xi + H @ self.W_hi + self.bias_i)
            F = torch.sigmoid(x_t @ self.W_xf + H @ self.W_hf + self.bias_f)
            O = torch.sigmoid(x_t @ self.W_xo + H @ self.W_ho + self.bias_o)
            C_tilda = torch.tanh(x_t @ self.W_xc + H @ self.W_hc + self.bias_c)

            C = F * C + I * C_tilda
            H = torch.tanh(C) + O
            hidden_layer1 = torch.relu(H @ self.W_o1 + self.bias_o1)
            output = act.softmax(hidden_layer1 @ self.W_o2 + self.bias_o2, dim=1)
        # output = torch.relu(hidden_layer1 @ self.W_o2 + self.bias_o2)
        #             oupts.append(output)

        #         self.init_state = (H, C)
        #         return torch.cat(oupts, dim = 0)
        return output, (H, C)


env = itemEnv()
actor = LSTM(7, 64, 128, 381)
actor.load_state_dict(torch.load('modelpara.pth'))
# train
device = try_gpu()
# loss_fn = nn.NLLLoss(reduction='sum')
max_episodes = 50
max_episodes_len = 4
gamma = 0.9
optimizer = torch.optim.SGD(actor.parameters(), lr=0.003)
for episode in range(max_episodes):
    env.reset()
    state = env.current_state
    for j in range(2):
        #     env.current_state = train_data.loc[7,'embedded_state']
        env.current_state = state

        policy_loss_l = []
        reward_l = []
        optimizer.zero_grad()
        for i in range(max_episodes_len):
            n_state = np.reshape(env.current_state, [-1, 12, 7])
            #             n_state = torch.from_numpy(n_state).cuda().to(torch.float32)
            n_state = torch.from_numpy(n_state).to(device).to(torch.float32)
            #             bat_x.append(n_state)
            prob, _ = actor(n_state)

            #             prob = act.softmax(prob, dim = 1)
            m = Categorical(prob)
            #             action = m.sample() if np.random.random() > 0.2 else torch.randint(0, 381, [1]).cuda()  #0~380 [low, high)
            action = m.sample() if np.random.random() > 0.2 else torch.randint(0, 381, [1]).to(
                device)  # 0~380 [low, high)
            #         action = m.sample()
            #         target.append(action)
            with torch.no_grad():
                embed_action = gen_action(str(action.item() + 1))  # action : char
                r, _, _ = env.step(embed_action)
            # G_t = gamma*G_t + r
            reward_l.append(r)
            policy_loss_l.append(-m.log_prob(action))
            print(f'{action.item()}:{r}', end=' ')
        # print(f'\nproba:{prob[0][action.item()]}')

        for k in range(max_episodes_len - 1):
            reward_l[max_episodes_len - k - 2] = reward_l[max_episodes_len - k - 2] + gamma * reward_l[
                max_episodes_len - k - 1]
        policy_loss_l = [a * b for a, b in zip(policy_loss_l, reward_l)]

        loss = torch.cat(policy_loss_l).sum()
        loss.backward()

        optimizer.step()


        #         print(f'after:{pr[0][action.item()]}')
        print(f'total reward:{sum(reward_l)}, loss:{loss.data}')
