import numpy as np
from processing import train_data   

class itemEnv():
    def __init__(self, alpha = 0.7, sigma = 0.9): # albha parameter in consine similarity
        self.observation_space = train_data
        self.init_state = self.reset()
        self.current_state = None
        self.sigma = sigma
        self.alpha = alpha
        self.embedding_dim = 7
        self.reward_val = 5
        self.rewards, self.avg_state, self.avg_action, self.nx_size = self.avg_group()
    
    def reset(self):
        init_state = self.observation_space['embedded_state'].sample(1).values[0]
        self.current_state = init_state
        return init_state

    def feedback(self, pair):  # pair: state-action pair
        prob = list()
        denominator = 0.
        max_prob = 0.
        result = 0.
        feed_back = ""
        for r, s, a in zip(self.rewards, self.avg_state, self.avg_action):
            numerator = self.alpha * np.dot(pair[0], s.T) / np.linalg.norm(pair[0], 2) + (1 - self.alpha) * np.dot(
                pair[1], a.T) / np.linalg.norm(pair[1], 2)
            denominator += numerator
            prob.append(numerator)
            if numerator > max_prob:
                max_prob = numerator
                feed_back = r
        prob /= denominator
        for p, r in zip(prob, self.rewards):
            for k in range(1):
                result += p * self.reward_val * np.power(self.sigma, k) * int(r.split(',')[k])

        return feed_back, result

    def step(self, action):   #action: 1 dim array
        feed_back, result = self.feedback((self.current_state, action))
        for i,r in enumerate([feed_back.split(',')[0]]): #0---> one action
            if r == '1':
                tmp = np.append(self.current_state, action)
                tmp = tmp[self.embedding_dim:]
                self.current_state = tmp  # state: 1 dim array

        return result, self.current_state
        
    def avg_group(self):
        nx_size = list()
        avg_state = list()
        avg_action = list()
        rewards = list()
        for reward, group in self.observation_space.groupby(['reward']):
            nx_size.append(group.shape[0])
#             state = np.mean(data['embedded_action'], axis = 0)
            norm_s = np.linalg.norm(np.array(group['embedded_state'].values.tolist()), 2, axis = 1)
            norm_s = np.where(norm_s == 0, 0.001, norm_s)
            state = np.sum(group['embedded_state'] / norm_s) / group.shape[0]
            norm_a = np.linalg.norm(np.array(group['embedded_action'].values.tolist()), 2, axis = 1)
            norm_a = np.where(norm_a == 0, 0.001, norm_a)
            action = np.sum(group['embedded_action'] / norm_a) / group.shape[0]
            avg_state.append(state)
            avg_action.append(action)
            rewards.append(reward)
        return rewards, avg_state, avg_action, nx_size
        

env = itemEnv()
env.reset()
r, s = env.step(train_data.loc[0,'embedded_action'])