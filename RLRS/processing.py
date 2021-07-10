import numpy as np 
import pandas as pd 
from sklearn.utils import shuffle

def read_data(embedding_dim = 20):
	trainset = pd.read_csv('./bigdata2021-rl-recsys/trainset.csv', sep = ' ')
	trainset['history_num'] = trainset["user_click_history"].apply(
		lambda row: list(map(lambda t: int(t.split(":")[0]), row.split(",")))
		)
	# trainset['time_line'] = trainset["user_click_history"].apply(
	# 	lambda row: list(map(lambda t: int(t.split(":")[1]), row.split(",")))
	# 	)
	trainset['reward'] = trainset["labels"]
	# trainset['reward'] = trainset["labels"].apply(
	# 	lambda row: list(map(lambda t: reward * int(t), row.split(',')))
	# 	)
	trainset['action'] = trainset["exposed_items"].apply(
		lambda row: list(map(lambda t: int(t), row.split(',')))
		)
	trainset['state'] = trainset['history_num'].apply(
		lambda row: row[-embedding_dim:] if len(row) >= embedding_dim else [0]*(embedding_dim - len(row)) + row
		)
	#create train
	dic = {'state':trainset['state'], 'action':trainset['action'], 'reward':trainset['reward']}
	train = pd.DataFrame(dic)
	#get embedding dictionary
	embed = embedding_dict()
	#transfer the data into embedding vector
	data = pd.DataFrame()
	data['embedded_state'] = train['state'].apply(
		lambda row: np.array(list(map(lambda key: embed[str(key)], row))).reshape(-1,)
		)
	# data['state'] = train['state']
	data['embedded_action'] = train['action'].apply(
	# lambda row: np.array(list(map(lambda key: embed[str(key)], [row[0]]))).reshape(-1,)
	lambda row: np.array(list(map(lambda key: embed[str(key)], row)))
	)  # 0 for the first action
	# data['action'] = train['action']
	data['reward'] = train['reward']
	#delete zero state
	data['del_zero'] = train['state'].apply(lambda row: np.sum(row))
	data = data[data['del_zero'] != 0].drop(columns = ['del_zero'])
	#print information of the traning data
	print(data.info(memory_usage = 'deep'))  
	#split dataset
	train_data, val_data = train_test_split(data, random_seed = 1)

	return train_data, val_data, embed


def embedding_dict():
	item_info = pd.read_csv('./bigdata2021-rl-recsys/item_info.csv', sep = ' ')
	info_size = len(item_info)
	for i in range(info_size):
	    item_info.loc[i, 'item_vec_num'] = item_info.loc[i, 'item_vec'] + ','+ str(item_info.loc[i,'price']) + ',' + str(item_info.loc[i,'location'])

	item_info['item_vec_num'] = item_info['item_vec_num'].apply(
		lambda row: np.array(list(map(lambda t: float(t), row.split(','))))
		)
	embed_vec = np.array(item_info['item_vec_num'].values.tolist(), dtype = float)
	keys = [str(item) for item in item_info['item_id']]
	# from sklearn.preprocessing import MinMaxScaler
	from sklearn.preprocessing import StandardScaler
	# embed_vec = MinMaxScaler().fit_transform(embed_vec)
	embed_vec = StandardScaler().fit_transform(embed_vec)
	embed = dict(zip(keys, embed_vec))
	embed['0'] = np.zeros([embed_vec.shape[1]])  #新增0元素

	return embed

def train_test_split(data, train_size = 0.7, shuffle_data = True, random_seed = None):
    if shuffle_data == True:
        data = shuffle(data, random_state = random_seed)

    ind = int(len(data)*train_size)
    train_data = data[0:ind].reset_index(drop = True)
    val_data = data[ind:].reset_index(drop = True)

    return train_data, val_data

# if __name__ == '__main__':
train_data, val_data, embed = read_data()
