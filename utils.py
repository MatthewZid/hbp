import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OBJ_SIZE_POS = 4

def read_expert(dataset_name='preprocessed_2D'):
    # data path
    basepath = '/home/matthew/Documents/AI/thesis/hbp/dataset'
    datapath = os.path.join(basepath, dataset_name)

    # init dataset dict
    dataset = {}

    # read csv's
    print('Reading expert dataset...')
    datalist = [os.path.join(datapath, dirname) for dirname in os.listdir(datapath) if os.path.isdir(os.path.join(datapath, dirname))]
    for dl in datalist:
        for datafile in os.listdir(dl):
            fl = os.path.join(dl, datafile)
            dataset[datafile] = pd.read_csv(fl, float_precision='round_trip')
    
    return dataset

def get_coords(df, keep=2, start=3):
    sk = []
    for i in range(start, len(df.columns)):
        if keep != 0:
            sk.append(df.iloc[:,i])
            keep -= 1
        else: keep = 2
    
    return pd.concat(sk, axis=1)

def plot_feature(feature, type='actions'):
    plt.figure()
    plt.title('Expert '+type)
    for i in range(0, feature.shape[1]-1, 2):
        plt.scatter(feature[:,i], feature[:,i+1], s=4, alpha=0.4)
    plt.savefig('./plots/'+type, dpi=100)
    plt.close()

def extract_features(dataset):
    one_hot = {'S': [1,0,0], 'M': [0,1,0], 'L': [0,0,1]}
    features = {}
    features['states'] = []
    features['actions'] = []
    features['codes'] = []

    for key in dataset.keys():
        states = get_coords(dataset[key])
        actions = []

        for i in range(len(states)-1):
            actions.append(states.iloc[i+1] - states.iloc[i])

        actions = pd.concat(actions, ignore_index=True, axis=1).T
        states = states.drop(len(states.index)-1)

        features['states'].append(states)
        features['actions'].append(actions)
        features['codes'].append(np.array([one_hot[key[OBJ_SIZE_POS]] for _ in range(len(states))]))
    
    features['states'] = pd.concat(features['states'], ignore_index=True).to_numpy()
    features['actions'] = pd.concat(features['actions'], ignore_index=True).to_numpy()
    features['codes'] = np.concatenate(features['codes'], axis=0)

    return features