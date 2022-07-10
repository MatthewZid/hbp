import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math
from scipy import signal
from sklearn.preprocessing import MinMaxScaler, StandardScaler

OBJ_SIZE_POS = 4
PROB_THRESHOLD = 0.6
DIST_THRESHOLD = 10
WINDOW = 9
MOVEMENTS = 715

improved = 0
save_loss = True
save_models = True
resume_training = False
use_ppo = False
LOGSTD = tf.math.log(0.008)
SPEED = 0.02
BUFFER_RATIO = 0.67
TRAIN_BATCH_SIZE = 2000

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

def confidence_prob_plot(dataset):
    aggr_set = {}
    aggr_set['thumb'] = []
    aggr_set['index'] = []
    aggr_set['wrist'] = []
    for key in dataset.keys():
        aggr_set['thumb'].append(dataset[key]['RThumb4FingerTip.prob'].to_numpy())
        aggr_set['index'].append(dataset[key]['RIndex4FingerTip.prob'].to_numpy())
        aggr_set['wrist'].append(dataset[key]['RWrist.prob'].to_numpy())
    
    aggr_set['thumb'] = np.concatenate(aggr_set['thumb'], axis=0)
    aggr_set['index'] = np.concatenate(aggr_set['index'], axis=0)
    aggr_set['wrist'] = np.concatenate(aggr_set['wrist'], axis=0)
    
    plt.figure()
    plt.boxplot([aggr_set['wrist'], aggr_set['thumb'], aggr_set['index']])
    plt.xticks([1,2,3],['Wrist','Thumb','Index'])
    plt.savefig('conf_prob', dpi=100)
    plt.close()

def neighbour_dist(coords):
    dist = np.abs(coords[1:] - coords[:-1])
    prev_dist = np.r_[dist[0], dist]
    next_dist = np.r_[dist, dist[-1]]
    neighbour_dist = np.min([prev_dist, next_dist], axis=0)

    return neighbour_dist

def extract_apertures(movement, joint1, joint2):
    x1 = movement[joint1+'.x'].iloc[WINDOW:].to_numpy()
    y1 = movement[joint1+'.y'].iloc[WINDOW:].to_numpy()
    prob1 = movement[joint1+'.prob'].iloc[WINDOW:].to_numpy()

    x2 = movement[joint2+'.x'].iloc[WINDOW:].to_numpy()
    y2 = movement[joint2+'.y'].iloc[WINDOW:].to_numpy()
    prob2 = movement[joint2+'.prob'].iloc[WINDOW:].to_numpy()

    apertures = np.sqrt(np.power(x1 - x2, 2) + np.power(y1 - y2, 2))

    invalid_frames = np.logical_not((prob1 > PROB_THRESHOLD) & (prob2 > PROB_THRESHOLD) & \
                                    (neighbour_dist(x1) < DIST_THRESHOLD) & (neighbour_dist(x2) < DIST_THRESHOLD) & \
                                    (neighbour_dist(y1) < DIST_THRESHOLD) & (neighbour_dist(y2) < DIST_THRESHOLD))

    apertures[invalid_frames] = np.nan
    return apertures

def plot_time_dist(dataset, name='time_dist'):
    timedist = []
    for key in dataset.keys():
        timediff = dataset[key]['time'].iloc[1:].to_numpy() - dataset[key]['time'].iloc[:-1].to_numpy()
        timedist.append(timediff)
    
    timedist = np.concatenate(timedist, axis=0)
    
    plt.figure()
    plt.title('Sampling rate for every frame')
    plt.xlabel('Frame No.')
    plt.ylabel('Sampling rate (sec)')
    plt.scatter(np.arange(timedist.shape[0]), timedist, s=4, alpha=0.4)
    plt.savefig(name, dpi=100)
    plt.close()

def extract_features(dataset):
    for key in dataset.keys():
        features = {}
        features['time'] = dataset[key]['Time'].iloc[WINDOW:].reset_index(drop=True)
        abs_time = features['time'].to_numpy()
        norm_time = 100.0 * (abs_time - abs_time[0]) / (abs_time[-1] - abs_time[0])
        features['norm_time'] = norm_time
        features['apertures'] = extract_apertures(dataset[key], "RThumb4FingerTip", "RIndex4FingerTip")
        features['wrist_x'] = dataset[key]['RWrist.x'].iloc[WINDOW:].reset_index(drop=True)
        features['wrist_y'] = dataset[key]['RWrist.y'].iloc[WINDOW:].reset_index(drop=True)
        
        features_df = pd.DataFrame(features)
        dataset[key] = features_df
    
    return dataset

def count_nan_apertures(dataset):
    ignored = 0
    group_by_mode = {}
    group_by_mode['S'] = np.zeros((5,), dtype=np.float64)
    group_by_mode['M'] = np.zeros((5,), dtype=np.float64)
    group_by_mode['L'] = np.zeros((5,), dtype=np.float64)
    compl_per = [20.0, 40.0, 60.0, 80.0, 100.0]

    for key in dataset.keys():
        nans = np.where(np.isnan(dataset[key]['apertures'].to_numpy()))[0]
        if nans.shape[0] > 0:
            if nans[-1] == (len(dataset[key]['apertures'])-1) or nans[0] == 0:
                ignored += 1
                continue
        
        missing = []
        for per in compl_per:
            compl_data = dataset[key][dataset[key]['norm_time'] <= per]
            nans = np.where(np.isnan(compl_data['apertures'].to_numpy()))[0].shape[0] # count nans
            missing.append(nans)

        group_by_mode[key[OBJ_SIZE_POS]] = np.add(group_by_mode[key[OBJ_SIZE_POS]], missing)
    
    for i in range(len(compl_per)):
        plt.figure()
        plt.title(str(int(compl_per[i]))+'% interval of movement (total '+str(MOVEMENTS - ignored)+')')
        plt.xlabel('Object size')
        plt.ylabel('Rejected aperture (NaN) count')
        plt.bar(['S','M','L'], [group_by_mode['S'][i], group_by_mode['M'][i], group_by_mode['L'][i]])
        plt.savefig(str(int(compl_per[i]))+'_per', dpi=100)
        plt.close()

def count_nan_apertures_per(dataset):
    group_by_mode = {}
    group_by_mode['S'] = []
    group_by_mode['M'] = []
    group_by_mode['L'] = []
    ignored = 0

    for key in dataset.keys():
        nans = np.where(np.isnan(dataset[key]['apertures'].to_numpy()))[0]
        if nans.shape[0] > 0:
            if nans[-1] == (len(dataset[key]['apertures'])-1) or nans[0] == 0:
                ignored += 1
                continue

        group_by_mode[key[OBJ_SIZE_POS]].append((nans.shape[0] / len(dataset[key])) * 100.0)
    
    plt.figure()
    plt.title('Aperture NaN percentage for {:d} movements'.format(MOVEMENTS - ignored))
    plt.xlabel('Movement No.')
    plt.ylabel('NaN count (%)')
    for sz in ['S','M','L']:
        plt.bar(np.arange(len(group_by_mode[sz])), group_by_mode[sz])
    plt.legend(['Small', 'Medium', 'Large'], loc='upper right')
    plt.savefig('count_nans_per', dpi=100)
    plt.close()

def count_consecutive_nans(dataset):
    ignored = 0
    group_by_mode = {}
    group_by_mode['S'] = []
    group_by_mode['M'] = []
    group_by_mode['L'] = []

    for key in dataset.keys():
        nans = np.where(np.isnan(dataset[key]['apertures'].to_numpy()))[0]
        if nans.shape[0] > 0:
            if nans[-1] == (len(dataset[key]['apertures'])-1) or nans[0] == 0:
                ignored += 1
                continue

        count_consecutive_nans = 0
        for i in range(-2, -len(dataset[key]), -1):
            if np.isnan(dataset[key]['apertures'].iloc[i]): count_consecutive_nans += 1
            else: break
        
        group_by_mode[key[OBJ_SIZE_POS]].append((count_consecutive_nans / len(dataset[key])) * 100.0)
    
    plt.figure()
    plt.title('Last consecutive aperture NaN percentage for {:d} movements'.format(MOVEMENTS - ignored))
    plt.xlabel('Movement No.')
    plt.ylabel('Consecutive NaN count (%)')
    for sz in ['S','M','L']:
        plt.bar(np.arange(len(group_by_mode[sz])), group_by_mode[sz])
    plt.legend(['Small', 'Medium', 'Large'], loc='upper right')
    plt.savefig('last_consecutive_nans', dpi=100)
    plt.close()

def max_consecutive_nans(dataset):
    ignored = 0
    group_by_mode = {}
    group_by_mode['S'] = []
    group_by_mode['M'] = []
    group_by_mode['L'] = []

    for key in dataset.keys():
        nans = np.where(np.isnan(dataset[key]['apertures'].to_numpy()))[0]
        if nans.shape[0] > 0:
            if nans[-1] == (len(dataset[key]['apertures'])-1) or nans[0] == 0:
                ignored += 1
                continue

        count_consecutive_nans = 0
        max_nan = np.NINF
        for i in range(-2, -len(dataset[key]), -1):
            if np.isnan(dataset[key]['apertures'].iloc[i]): count_consecutive_nans += 1
            else:
                if count_consecutive_nans > max_nan: max_nan = count_consecutive_nans
                count_consecutive_nans = 0
        
        group_by_mode[key[OBJ_SIZE_POS]].append((max_nan / len(dataset[key])) * 100.0)
    
    plt.figure()
    plt.title('Max consecutive aperture NaN percentage for {:d} movements'.format(MOVEMENTS - ignored))
    plt.xlabel('Movement No.')
    plt.ylabel('Max consecutive NaN count (%)')
    for sz in ['S','M','L']:
        plt.bar(np.arange(len(group_by_mode[sz])), group_by_mode[sz])
    plt.legend(['Small', 'Medium', 'Large'], loc='upper right')
    plt.savefig('max_consecutive_nans', dpi=100)
    plt.close()

def interpolated_boxplot(dataset):
    group_by_mode = {}
    group_by_mode['S'] = [[],[],[],[],[]]
    group_by_mode['M'] = [[],[],[],[],[]]
    group_by_mode['L'] = [[],[],[],[],[]]
    compl_per = [20.0, 40.0, 60.0, 80.0, 100.0]

    for key in dataset.keys():
        nans = np.where(np.isnan(dataset[key]['apertures'].to_numpy()))[0]
        if nans.shape[0] > 0:
            if nans[-1] == (len(dataset[key]['apertures'])-1) or nans[0] == 0: continue
            dataset[key] = dataset[key].interpolate(method='linear')
        
        for i in range(len(compl_per)):
            compl_data = dataset[key][dataset[key]['norm_time'] <= compl_per[i]]['apertures'].to_numpy()
            group_by_mode[key[OBJ_SIZE_POS]][i].append(compl_data)
    
    for i in range(len(compl_per)):
        group_by_mode['S'][i] = np.concatenate(group_by_mode['S'][i], axis=0)
        group_by_mode['M'][i] = np.concatenate(group_by_mode['M'][i], axis=0)
        group_by_mode['L'][i] = np.concatenate(group_by_mode['L'][i], axis=0)

        plt.figure()
        plt.title('Interpolated aperture boxplot of '+str(int(compl_per[i]))+'% movement interval')
        plt.boxplot([group_by_mode['S'][i], group_by_mode['M'][i], group_by_mode['L'][i]])
        plt.xticks([1,2,3],['S','M','L'])
        plt.xlabel('Object size')
        plt.savefig('boxplot_'+str(int(compl_per[i]))+'_per', dpi=100)
        plt.close()

# def extract_apertures_wrist_mdp(dataset):
#     one_hot = {'S': [1,0,0], 'M': [0,1,0], 'L': [0,0,1]}
#     features = {}
#     features['states'] = []
#     features['actions'] = []
#     features['codes'] = []
#     feature_size = []

#     for key in dataset.keys():
#         nans = np.where(np.isnan(dataset[key]['apertures'].to_numpy()))[0]
#         if nans.shape[0] > 0:
#             if nans[-1] == (len(dataset[key]['apertures'])-1) or nans[0] == 0: continue
#             dataset[key] = dataset[key].interpolate(method='linear')

#         points = pd.concat([dataset[key]['apertures'], dataset[key]['wrist_x'], dataset[key]['wrist_y']], axis=1).to_numpy()

#         window = np.zeros((5,points.shape[1]), dtype=np.float64)
#         for i in range(5):
#             window[i,0] = points[0,0]
#             window[i,1] = points[0,1]
#             window[i,2] = points[0,2]
#         states = [np.copy(window.flatten())]
#         actions = []

#         for i in range(points.shape[0]-1):
#             actions.append(points[i+1,:] - points[i,:])
#             for j in range(4): window[j,:] = window[j+1,:]
#             window[4,:] = points[i+1,:]
#             states.append(np.copy(window.flatten()))
        
#         actions = np.array(actions, dtype=np.float64)
#         states = np.array(states, dtype=np.float64)
#         states = states[:-1,:]

#         feature_size.append(states.shape[0])

#         features['states'].append(states)
#         features['actions'].append(actions)
#         features['codes'].append(np.array([one_hot[key[OBJ_SIZE_POS]] for _ in range(len(states))]))
    
#     features['states'] = np.concatenate(features['states'], axis=0)
#     features['actions'] = np.concatenate(features['actions'], axis=0)
#     features['codes'] = np.concatenate(features['codes'], axis=0)
#     feature_size = np.array(feature_size, dtype=int)

#     return features, feature_size, dataset, 3

# def extract_norm_apertures_wrist_mdp(dataset):
#     one_hot = {'S': [1,0,0], 'M': [0,1,0], 'L': [0,0,1]}
#     features = {}
#     features['states'] = []
#     features['actions'] = []
#     features['codes'] = []
#     feature_size = []
#     data = []
#     new_feature_size = []
    

#     for key in dataset.keys():
#         nans = np.where(np.isnan(dataset[key]['apertures'].to_numpy()))[0]
#         if nans.shape[0] > 0:
#             if nans[-1] == (len(dataset[key]['apertures'])-1) or nans[0] == 0: continue
#             dataset[key] = dataset[key].interpolate(method='linear')
        
#         points = pd.concat([dataset[key]['apertures'], dataset[key]['wrist_x'], dataset[key]['wrist_y']], axis=1).to_numpy()
#         # features['actions'].append(points[1:] - points[:-1])
#         # points = points[:-1, :]
#         features['codes'].append(np.array([one_hot[key[OBJ_SIZE_POS]] for _ in range(points.shape[0])]))
#         data.append(points)
#         feature_size.append(points.shape[0])
    
#     data = np.concatenate(data, axis=0)
#     # features['actions'] = np.concatenate(features['actions'], axis=0)
#     feature_size = np.array(feature_size, dtype=int)

#     state_scaler = MinMaxScaler(feature_range=(-1,1))
#     # action_scaler = MinMaxScaler(feature_range=(-1,1))
#     data = state_scaler.fit_transform(data)
#     # features['actions'] = action_scaler.fit_transform(features['actions'])

#     pos = 0
#     for sz in feature_size:
#         window = np.zeros((5,data.shape[1]), dtype=np.float64)
#         for i in range(5):
#             window[i,0] = data[pos,0]
#             window[i,1] = data[pos,1]
#             window[i,2] = data[pos,2]
#         states = [np.copy(window.flatten())]
#         features['actions'].append(data[pos+1:pos+sz, :] - data[pos:pos+sz-1, :])

#         for i in range(pos+1, pos+sz-1):
#             for j in range(4): window[j,:] = window[j+1,:]
#             window[4,:] = data[i,:]
#             states.append(np.copy(window.flatten()))
        
#         states = np.array(states, dtype=np.float64)
#         new_feature_size.append(states.shape[0])
#         features['states'].append(states)
#         pos += sz
    
#     features['states'] = np.concatenate(features['states'], axis=0)
#     features['actions'] = np.concatenate(features['actions'], axis=0)
#     features['codes'] = np.concatenate(features['codes'], axis=0)
#     new_feature_size = np.array(new_feature_size, dtype=int)

#     return features, new_feature_size, dataset, 3

def extract_norm_apertures_wrist_mdp(dataset):
    one_hot = {'S': [1,0,0], 'M': [0,1,0], 'L': [0,0,1]}
    features = {}
    features['train'] = {}
    features['test'] = {}
    feature_size = {}
    new_feature_size = {}
    data = {}

    for tp in ['train', 'test']:
        features[tp]['states'] = []
        features[tp]['actions'] = []
        features[tp]['codes'] = []
        features[tp]['time'] = []
        features[tp]['norm_time'] = []

        feature_size[tp] = []
        new_feature_size[tp] = []
        data[tp] = []

    new_dataset = {}

    for key in dataset.keys():
        nans = np.where(np.isnan(dataset[key]['apertures'].to_numpy()))[0]
        if nans.shape[0] > 0:
            if nans[-1] == (len(dataset[key]['apertures'])-1) or nans[0] == 0: continue
            dataset[key] = dataset[key].interpolate(method='linear')
        new_dataset[key] = dataset[key].copy()
    
    train_ratio = int((80 * len(new_dataset.keys())) / 100.0)
    np_keys = np.array(list(new_dataset.keys()))
    np.random.shuffle(np_keys)
    keys = {}
    keys['train'] = np_keys[:train_ratio]
    keys['test'] = np_keys[train_ratio:]
    
    for tp in ['train', 'test']:
        for key in keys[tp]:
            points = pd.concat([new_dataset[key]['apertures'], new_dataset[key]['wrist_x'], new_dataset[key]['wrist_y']], axis=1).to_numpy()
            features[tp]['codes'].append(np.array([one_hot[key[OBJ_SIZE_POS]] for _ in range(points.shape[0] - 1)]))
            features[tp]['time'].append(new_dataset[key]['time'].iloc[:-1].to_numpy())
            features[tp]['norm_time'].append(new_dataset[key]['norm_time'].iloc[:-1].to_numpy())
            data[tp].append(points)
            feature_size[tp].append(points.shape[0])
    
        data[tp] = np.concatenate(data[tp], axis=0)
        feature_size[tp] = np.array(feature_size[tp], dtype=int)

    # state_scaler = MinMaxScaler(feature_range=(-1,1))
    # data['train'] = state_scaler.fit_transform(data['train'])
    # data['test'] = state_scaler.transform(data['test'])
    # ...OR...
    wrist_scaler = MinMaxScaler(feature_range=(-1,1))
    aperture_scaler = StandardScaler()
    train_apertures = aperture_scaler.fit_transform(data['train'][:, 0].reshape((-1, 1)))
    test_apertures = aperture_scaler.transform(data['test'][:, 0].reshape((-1, 1)))
    train_wrist = wrist_scaler.fit_transform(data['train'][:, 1:])
    test_wrist = wrist_scaler.transform(data['test'][:, 1:])
    data['train'] = np.concatenate([train_apertures, train_wrist], axis=1)
    data['test'] = np.concatenate([test_apertures, test_wrist], axis=1)

    for tp in ['train', 'test']:
        pos = 0
        for sz in feature_size[tp]:
            window = np.zeros((5,data[tp].shape[1]), dtype=np.float64)
            for i in range(5):
                window[i,0] = data[tp][pos,0]
                window[i,1] = data[tp][pos,1]
                window[i,2] = data[tp][pos,2]
            states = [np.copy(window.flatten())]
            features[tp]['actions'].append(data[tp][pos+1:pos+sz, :] - data[tp][pos:pos+sz-1, :])

            for i in range(pos+1, pos+sz-1):
                for j in range(4): window[j,:] = window[j+1,:]
                window[4,:] = data[tp][i,:]
                states.append(np.copy(window.flatten()))
            
            states = np.array(states, dtype=np.float64)
            new_feature_size[tp].append(states.shape[0])
            features[tp]['states'].append(states)
            pos += sz
        
        features[tp]['states'] = np.concatenate(features[tp]['states'], axis=0, dtype=np.float64)
        features[tp]['actions'] = np.concatenate(features[tp]['actions'], axis=0, dtype=np.float64)
        features[tp]['codes'] = np.concatenate(features[tp]['codes'], axis=0, dtype=np.float64)
        features[tp]['time'] = np.concatenate(features[tp]['time'], axis=0, dtype=np.float64)
        features[tp]['norm_time'] = np.concatenate(features[tp]['norm_time'], axis=0, dtype=np.float64)
        new_feature_size[tp] = np.array(new_feature_size[tp], dtype=int)

    return features, new_feature_size, dataset, 3

def extract_start_pos(features, feat_size, feat_col_len):
    start_pos = []
    codes = []
    pos = 0

    for sz in feat_size:
        start_pos.append(features['states'][pos, -feat_col_len:])
        codes.append(features['codes'][pos])
        pos += sz
    
    return np.array(start_pos, dtype=np.float64), np.array(codes, dtype=np.float64)

def movement_end(dataset):
    ignored = 0
    final_pos = {}
    final_pos['S'] = [[], []]
    final_pos['M'] = [[], []]
    final_pos['L'] = [[], []]

    for key in dataset.keys():
        nans = np.where(np.isnan(dataset[key]['apertures'].to_numpy()))[0]
        if nans.shape[0] > 0:
            if nans[-1] == (len(dataset[key]['apertures'])-1) or nans[0] == 0:
                ignored += 1
                continue

            dataset[key] = dataset[key].interpolate(method='linear')
        
        final_pos[key[OBJ_SIZE_POS]][0].append(dataset[key]['apertures'].iloc[-1])
        final_pos[key[OBJ_SIZE_POS]][1].append(dataset[key]['wrist_y'].iloc[-1])
    
    plt.figure()
    plt.title('Aperture endpoint for {:d} movements'.format(MOVEMENTS - ignored))
    plt.boxplot([final_pos['S'][0], final_pos['M'][0], final_pos['L'][0]])
    plt.xticks([1,2,3],['S','M','L'])
    plt.xlabel('Object size')
    plt.ylabel('aperture endpoint')
    plt.savefig('aperture_movement_ends', dpi=100)
    plt.close()

    plt.figure()
    plt.title('y-wrist endpoint for {:d} movements'.format(MOVEMENTS - ignored))
    plt.boxplot([final_pos['S'][1], final_pos['M'][1], final_pos['L'][1]])
    plt.xticks([1,2,3],['S','M','L'])
    plt.xlabel('Object size')
    plt.ylabel('y-wrist coordinate endpoint')
    plt.savefig('ywrist_movement_ends', dpi=100)
    plt.close()

def norm_movement_ends(features, feat_size):
    size_map = {'0': 'S', '1': 'M', '2': 'L'}
    final_pos = {}
    final_pos['S'] = [[], []]
    final_pos['M'] = [[], []]
    final_pos['L'] = [[], []]
    pos = 0

    for sz in feat_size:
        key = str(np.where(features['codes'][pos] == 1)[0][0])
        final_pos[size_map[key]][0].append(features['states'][pos+sz-1, 0] + features['actions'][pos+sz-1, 0]) # aperture
        final_pos[size_map[key]][1].append(features['states'][pos+sz-1, -1] + features['actions'][pos+sz-1, -1]) # y-wrist
        pos += sz
    
    plt.figure()
    plt.title('Normalized aperture endpoint for {:d} movements'.format(feat_size.shape[0]))
    plt.boxplot([final_pos['S'][0], final_pos['M'][0], final_pos['L'][0]])
    plt.xticks([1,2,3],['S','M','L'])
    plt.xlabel('Object size')
    plt.ylabel('aperture endpoint')
    plt.savefig('aperture_movement_ends', dpi=100)
    plt.close()

    plt.figure()
    plt.title('Normalized y-wrist endpoint for {:d} movements'.format(feat_size.shape[0]))
    plt.boxplot([final_pos['S'][1], final_pos['M'][1], final_pos['L'][1]])
    plt.xticks([1,2,3],['S','M','L'])
    plt.xlabel('Object size')
    plt.ylabel('y-wrist coordinate endpoint')
    plt.savefig('ywrist_movement_ends', dpi=100)
    plt.close()

def std_10(dataset):
    plt.figure()
    plt.xlabel('time (sec)')
    plt.ylabel('y-wrist std (10 pixels)')
    xmax = np.NINF
    xmin = np.inf
    final_stds = []

    for key in dataset.keys():
        if len(dataset[key]) < 10:
            continue
        std10 = []
        secs = []
        for i in range(len(dataset[key])-9):
            stdy = dataset[key].iloc[i:i+10, 4].to_numpy().std()
            if stdy < xmin: xmin = stdy
            std10.append(stdy)
            secs.append(dataset[key].iloc[i+9,1] - dataset[key].iloc[0,1])
        
        if max(secs) > xmax: xmax = max(secs)
        final_stds.append(std10[-1])
        plt.scatter(secs, std10, s=4, alpha=0.4, color='steelblue')
    
    print(xmin)
    print(np.array(final_stds).min())
    print(np.array(final_stds).max())
    print(np.array(final_stds).mean())
    print(np.array(final_stds).std())

    plt.xlim(0,xmax)
    plt.grid(color='lightgray', alpha=0.6)
    plt.savefig('std10', dpi=100)
    plt.close()

    plt.figure()
    plt.scatter(np.arange(len(final_stds)), final_stds, s=4, alpha=0.4, color='steelblue')
    plt.grid(color='lightgray', alpha=0.6)
    plt.savefig('std10_movement_ends', dpi=100)
    plt.close()

##################################################################
############################# TRPO ###############################
##################################################################

def discount(x, gamma):
    assert x.ndim >= 1
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def gauss_log_prob(mu, logstd, x):
    var = tf.math.exp(2*logstd)
    gp = -tf.math.square(x - mu)/(2 * var) - .5*tf.math.log(tf.constant(2*np.pi, dtype=tf.float32)) - logstd
    return tf.math.reduce_sum(gp, [1])

def gauss_ent(logstd):
    h = tf.reduce_sum(logstd + tf.constant(0.5*tf.math.log(2*math.pi*math.e), dtype=tf.float32))
    return h

def var_shape(x):
    out = [k.value for k in x.get_shape()]
    assert all(isinstance(a, int) for a in out), \
            "shape function assumes that shape is fully known"
    return out

def numel(x):
    return np.prod(x.shape)

def get_flat(model):
    var_list = model.trainable_weights
    return tf.concat([tf.reshape(v, [numel(v)]) for v in var_list], 0)

def set_from_flat(model, theta):
    var_list = model.trainable_weights
    shapes = [v.shape for v in var_list]
    start = 0

    for (shape, v) in zip(shapes, var_list):
        size = np.prod(shape)
        v.assign(tf.reshape(theta[start:start + size], shape))
        start += size

def flatgrad(model, loss, tape):
    var_list = model.trainable_weights
    grads = tape.gradient(loss, var_list)
    return tf.concat([tf.reshape(grad, [numel(v)]) for (v, grad) in zip(var_list, grads)], 0)

def gauss_selfKL_firstfixed(mu, logstd):
    mu1, logstd1 = map(tf.stop_gradient, [mu, logstd])
    mu2, logstd2 = mu, logstd
    return gauss_KL(mu1, logstd1, mu2, logstd2)

def gauss_KL(mu1, logstd1, mu2, logstd2):
    var1 = tf.exp(2*logstd1)
    var2 = tf.exp(2*logstd2)
    kl = tf.reduce_sum(logstd2 - logstd1 + (var1 + tf.square(mu1 - mu2))/(2*var2) - 0.5)
    return kl

def linesearch(f, x, feed, fullstep, expected_improve_rate):
    accept_ratio = .1
    max_backtracks = 10
    fval = f(x, feed)
    for (_, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        newfval = f(xnew, feed)
        actual_improve = fval - newfval
        # actual_improve = newfval - fval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        if ratio > accept_ratio and actual_improve > 0:
            global improved
            improved += 1
            return xnew
    return x

def conjugate_gradient(f_Ax, feed, b, cg_iters=10, residual_tol=1e-10):
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)
    for i in range(cg_iters):
        z = f_Ax(p, feed)
        z = z.numpy()
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p
        assert z.shape == p.shape and p.shape == x.shape \
            and x.shape == r.shape, "Conjugate shape difference"
        rdotr = newrdotr
        if rdotr < residual_tol:
            break
    return x