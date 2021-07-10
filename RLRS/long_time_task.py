import numpy as np
import pandas as pd
import os, time, random
# from multiprocessing import Pool
# from concurrent.futures import ProcessPoolExecutor
from functools import partial
# from processing import train_data


class itemEnv():
    def __init__(self, alpha=0.5, sigma=0.9):  # albha parameter in consine similarity
        self.observation_space = train_data
        self.current_state = None
        self.init_state = self.reset()
        self.sigma = sigma
        self.alpha = alpha
        self.embedding_dim = 7
        self.reward_val = 5

    #         self.rewards, self.avg_state, self.avg_action, self.nx_size = self.avg_group()

    def reset(self):
        init_state = self.observation_space['embedded_state'].sample(1).values[0]
        #         init_state = self.observation_space.loc[0,'embedded_state']
        self.current_state = init_state
        return init_state


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
        cos = 0.4 * (np.dot(s_i, s_t) / (norm_si * norm_st + 1e-10))[:, np.newaxis] + (1 - 0.4) * np.dot(a_i, a_t) / (
        norm_ai * norm_at + 1e-10)  # 1 dim
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
                # feed_back = group[0]          #index: reward str
                feed_back = group[1].iloc[tup[1], 2]

        result = 5 * int(feed_back.split(',')[reward_ind])

        return feed_back, result

    def step(self, action):  # action: 1 dim array
        feed_back, result = self.feedback_2((self.current_state, action))
        for i, r in enumerate([feed_back.split(',')[0]]):  # 0---> one action
            if r == '1':
                tmp = np.append(self.current_state, action)
                tmp = tmp[self.embedding_dim:]
                self.current_state = tmp  # state: 1 dim array
            else:
                self.current_state = self.observation_space['embedded_state'].sample(1).values[0]

        return result, self.current_state, feed_back



def long_time_task(name):
    aa = 1
    print('Run task %s (%s)...' % (name, os.getpid()))
    start = time.time()
    time.sleep(random.random() * 3)
    end = time.time()
    print('Task %s runs %0.2f seconds.' % (name, (end - start)))
    return aa


def compare(group, x):
    max_prob = 0.
    nx_size = len(group[1])

    for i in range(nx_size):
        tmp = group[1].iloc[i, 1] + x
        if tmp > max_prob:
            max_prob = tmp
    print(max_prob)
    return (max_prob,x)

def A():
    print('Parent process %s.' % os.getpid())
    p = Pool(4)
    for i in range(5):
        p.apply_async(long_time_task, args=(i,))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join() #表明要执行子进程，等待子进程结束后主进程再继续往下运行，通常用于进程间的同步。
    print('All subprocesses done.')


def step(action):  # action: 1 dim array
    current_state = train_data['embedded_state'].sample(1).values[0]
    feed_back, result = feedback_2((current_state, action))
    for i, r in enumerate([feed_back.split(',')[0]]):  # 0---> one action
        if r == '1':
            tmp = np.append(current_state, action)
            tmp = tmp[7:]
            current_state = tmp  # state: 1 dim array

    return result, current_state



def feedback_2(pair):
    data = train_data.groupby(['reward'])
    num_workers = 1
    denominator = 0.
    max_prob = 0.
    result = 0.
    feed_back = ""
    # vec = 0
    # act = 0
    # pool = ProcessPoolExecutor(max_workers = 8)
    # pool = Pool(2)
    fnc = partial(cos_sim, pair = pair)
    prob = map(fnc, data)
    for tup, group in zip(prob, data):
        if tup[0] > max_prob:
            max_prob = tup[0]
            # feed_back = group[0]          #index: reward str
            feed_back = group[1].iloc[tup[1], 2]
            # vec = group[1].iloc[tup[1], 0]
            # act = group[1].iloc[tup[1], 1]

    result = 5 * int(feed_back.split(',')[0])
    prob = list(prob)
    return feed_back, result, prob


def cos_sim(group, pair):
    max_prob = 0.
    # nx_size = len(group[1])
    s_i = np.array(group[1]['embedded_state'].values.tolist()) # 2 dim array:0-->group, 1-->sample
    s_t = pair[0]                                              # 1 dim array:0-->sample
    a_i = np.array(group[1]['embedded_action'].values.tolist())
    a_t = pair[1]
    norm_si = np.linalg.norm(s_i, 2, axis = 1)                 # 1 dim
    norm_st = np.linalg.norm(s_t, 2, axis = 0)                 # 1 dim
    norm_ai = np.linalg.norm(a_i, 2, axis = 1)                 # 1 dim
    norm_at = np.linalg.norm(a_t, 2, axis = 0)                 # 1 dim
    
    cos = 0.5 * np.dot(s_i, s_t)/(norm_si * norm_st + 1e-10) + (1 - 0.5) * np.dot(a_i, a_t)/(norm_ai * norm_at + 1e-10)  #1 dim
    # cos = cos / np.sum(cos)
    

    # print('i am here!')
    return (max(cos), np.argmax(cos))



if __name__=='__main__':
    # aa = {'one': ['1', '2', '1', '2'], 'two': [2, 3, 4, 6], 'three': [3, 4, 5, 8]}
    # aa = pd.DataFrame(aa)
    # pp = partial(compare, x = 1)
    # res = []
    # pool = Pool(8)
    # yy = pool.map(pp, aa.groupby('one'))
    # print(yy[0][0])
    # env = itemEnv()
    from processing import train_data
    # current_state = train_data.loc[0, 'embedded_state']
    # # current_state = train_data['embedded_state'].sample(1).values[0]
    # start = time.time()
    # feed_back, result, prob= feedback_2((current_state, train_data.loc[0,'embedded_action']))

    env = itemEnv()
    env.current_state = train_data.loc[0, 'embedded_state']
    action = train_data.loc[0, 'embedded_action'][0]
    r, s, f = env.step(action)
    # pair = (current_state, train_data.loc[0,'embedded_action'])
    # data = train_data.groupby(['reward'])
    # num_workers = 4
    # denominator = 0.
    # max_prob = 0.
    # result = 0.
    # feed_back = ""
    # # pool = ProcessPoolExecutor(max_workers = 8)
    # pool = Pool(num_workers)
    # fnc = partial(cos_sim, pair=pair)
    # prob = pool.map(fnc, data)
    # for tup, group in zip(prob, data):
    #     if tup[0] > max_prob:
    #         max_prob = tup[0]
    #         feed_back = group[0]  # index: reward str
    #
    # result = 5 * int(feed_back.split(',')[0])
    # end = time.time()
    # print(prob)
    # print(feed_back)
    # print(result)
    # print('time:%.2f sec' % (end - start))
    # cc = 0.5 * np.dot(vec, train_data.loc[0, 'embedded_state'])/(np.linalg.norm(vec, 2) * np.linalg.norm(train_data.loc[0, 'embedded_state'], 2)) + 0.5* np.dot(act, train_data.loc[0, 'embedded_action'])/(np.linalg.norm(act, 2) * np.linalg.norm(train_data.loc[0, 'embedded_action'], 2))
    # print(cc)
