import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math
from scipy import signal

OBJ_SIZE_POS = 4

improved = 0
save_loss = True
save_models = True
resume_training = False
use_ppo = False
LOGSTD = tf.math.log(0.005)
SPEED = 0.02
BUFFER_RATIO = 0.67

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

# def extract_features(dataset):
#     one_hot = {'S': [1,0,0], 'M': [0,1,0], 'L': [0,0,1]}
#     features = {}
#     features['states'] = []
#     features['actions'] = []
#     features['codes'] = []

#     for key in dataset.keys():
#         states = get_coords(dataset[key])
#         actions = []

#         for i in range(len(states)-1):
#             actions.append(states.iloc[i+1] - states.iloc[i])

#         actions = pd.concat(actions, ignore_index=True, axis=1).T
#         states = states.drop(len(states.index)-1)

#         features['states'].append(states)
#         features['actions'].append(actions)
#         features['codes'].append(np.array([one_hot[key[OBJ_SIZE_POS]] for _ in range(len(states))]))
    
#     features['states'] = pd.concat(features['states'], ignore_index=True).to_numpy()
#     features['actions'] = pd.concat(features['actions'], ignore_index=True).to_numpy()
#     features['codes'] = np.concatenate(features['codes'], axis=0)

#     return features

def extract_features(dataset):
    one_hot = {'S': [1,0,0], 'M': [0,1,0], 'L': [0,0,1]}
    features = {}
    features['states'] = []
    features['actions'] = []
    features['codes'] = []

    for key in dataset.keys():
        wrist = pd.concat([dataset[key].iloc[:,3], dataset[key].iloc[:,4]], axis=1)
        thumb_tip = pd.concat([dataset[key].iloc[:,18], dataset[key].iloc[:,19]], axis=1)
        index_tip = pd.concat([dataset[key].iloc[:,30], dataset[key].iloc[:,31]], axis=1)
        apertures = pd.DataFrame(np.sqrt(np.power(thumb_tip.iloc[:,0] - index_tip.iloc[:,0], 2) + np.power(thumb_tip.iloc[:,1] - index_tip.iloc[:,1], 2)), columns=['Aperture'])
        points = pd.concat([apertures, wrist], axis=1)
        
        # resample expert trajectory with linear interpolation for equal timestamp intervals (20ms)
        points['td'] = pd.to_timedelta(dataset[key]['Time'].round(2), 's')
        if points['td'].duplicated().sum() > 0:
            points.at[points[points['td'].duplicated() == True].index.astype(int)[0], 'td'] = pd.to_timedelta(dataset[key][points['td'].duplicated() == True]['Time'].round(3), 's').iloc[0]
        points = points.set_index('td')
        interpolated = points.resample('20ms').interpolate(method='linear').to_numpy()
    
        window = np.zeros((5,interpolated.shape[1]), dtype=np.float32)
        for i in range(5):
            window[i,0] = interpolated[0,0]
            window[i,1] = interpolated[0,1]
            window[i,2] = interpolated[0,2]
        states = [np.copy(window.flatten())]
        actions = []

        for i in range(interpolated.shape[0]-1):
            actions.append(interpolated[i+1,:] - interpolated[i,:])
            for j in range(4): window[j,:] = window[j+1,:]
            window[4,:] = interpolated[i+1,:]
            states.append(np.copy(window.flatten()))
        
        actions = np.array(actions, dtype=np.float32)
        states = np.array(states, dtype=np.float32)
        states = states[:-1,:]

        features['states'].append(states)
        features['actions'].append(actions)
        features['codes'].append(np.array([one_hot[key[OBJ_SIZE_POS]] for _ in range(len(states))]))
    
    features['states'] = np.concatenate(features['states'], axis=0)
    features['actions'] = np.concatenate(features['actions'], axis=0)
    features['codes'] = np.concatenate(features['codes'], axis=0)

    return features

def extract_wrist_feature(dataset):
    one_hot = {'S': [1,0,0], 'M': [0,1,0], 'L': [0,0,1]}
    features = {}
    features['states'] = []
    features['actions'] = []
    features['codes'] = []

    for key in dataset.keys():
        points = pd.concat([dataset[key].iloc[:,3], dataset[key].iloc[:,4]], axis=1).to_numpy()
        window = np.zeros((5,2), dtype=np.float32)
        for i in range(5):
            window[i,0] = points[0,0]
            window[i,1] = points[0,1]
        states = [np.copy(window.flatten())]
        actions = []

        for i in range(points.shape[0]-1):
            actions.append(points[i+1,:] - points[i,:])
            for j in range(4): window[j,:] = window[j+1,:]
            window[4,:] = points[i+1,:]
            states.append(np.copy(window.flatten()))

        actions = np.array(actions, dtype=np.float32)
        states = np.array(states, dtype=np.float32)
        states = states[:-1,:]

        features['states'].append(states)
        features['actions'].append(actions)
        features['codes'].append(np.array([one_hot[key[OBJ_SIZE_POS]] for _ in range(len(states))]))
    
    features['states'] = np.concatenate(features['states'], axis=0)
    features['actions'] = np.concatenate(features['actions'], axis=0)
    features['codes'] = np.concatenate(features['codes'], axis=0)

    return features

def extract_aperture_feature(dataset):
    one_hot = {'S': [1,0,0], 'M': [0,1,0], 'L': [0,0,1]}
    features = {}
    features['states'] = []
    features['actions'] = []
    features['codes'] = []

    for key in dataset.keys():
        thumb_tip = pd.concat([dataset[key].iloc[:,18], dataset[key].iloc[:,19]], axis=1)
        index_tip = pd.concat([dataset[key].iloc[:,30], dataset[key].iloc[:,31]], axis=1)
        total_coords = pd.concat([thumb_tip, index_tip], axis=1)

        window = np.zeros((5,), dtype=np.float32)
        apertures = np.sqrt(np.power(thumb_tip.iloc[:,0] - index_tip.iloc[:,0], 2) + np.power(thumb_tip.iloc[:,1] - index_tip.iloc[:,1], 2)).values
        for i in range(5): window[i] = apertures[0]
        states = [np.copy(window)]
        actions = []

        for i in range(len(total_coords)-1):
            actions.append(total_coords.iloc[i+1] - total_coords.iloc[i])
            for j in range(4): window[j] = window[j+1]
            window[4] = apertures[i+1]
            states.append(np.copy(window))

        actions = pd.concat(actions, ignore_index=True, axis=1).T
        states = np.array(states, dtype=np.float32)
        states = states[:-1]

        features['states'].append(states)
        features['actions'].append(actions)
        features['codes'].append(np.array([one_hot[key[OBJ_SIZE_POS]] for _ in range(len(states))]))
    
    features['states'] = np.concatenate(features['states'], axis=0)
    features['actions'] = pd.concat(features['actions'], ignore_index=True).to_numpy()
    features['codes'] = np.concatenate(features['codes'], axis=0)

    return features

def extract_start_pos(dataset):
    start_pos = []

    for key in dataset.keys():
        wrist = [dataset[key].iloc[0,3], dataset[key].iloc[0,4]]
        thumb_tip = [dataset[key].iloc[0,18], dataset[key].iloc[0,19]]
        index_tip = [dataset[key].iloc[0,30], dataset[key].iloc[0,31]]
        aperture = math.sqrt(((thumb_tip[0] - index_tip[0])**2) + ((thumb_tip[1] - index_tip[1])**2))
        points = [aperture]
        points.extend(wrist)

        start_pos.append(points)
    
    return np.array(start_pos)

def find_end(dataset):
    endpoints = []

    for key in dataset.keys():
        norm = np.sqrt(np.power(dataset[key].iloc[-1,3] - dataset[key].iloc[-2,3], 2) + np.power(dataset[key].iloc[-1,4] - dataset[key].iloc[-2,4], 2))
        endpoints.append(norm)
    
    endpoints = np.array(endpoints, dtype=np.float32)
    xspace = np.arange(endpoints.shape[0])

    plt.figure()
    plt.title('dx before grasp')
    plt.hist(endpoints, color='steelblue', edgecolor='black', alpha=0.8)
    plt.savefig('ends', dpi=100)
    plt.close()

def discount(x, gamma):
    assert x.ndim >= 1
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def gauss_log_prob(mu, logstd, x):
    var = tf.exp(2*logstd)
    gp = -tf.square(x - mu)/(2 * var) - .5*tf.math.log(tf.constant(2*np.pi, dtype=tf.float32)) - logstd
    return tf.reduce_sum(gp, [1])

def gauss_ent(logstd):
    h = tf.reduce_sum(logstd + tf.constant(0.5*tf.math.log(2*math.pi*math.e), tf.float32))
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

def flatgrad(model, loss, tape, type='n'):
    var_list = model.trainable_weights
    if type == 'n': grads = tape.gradient(loss, var_list)
    else: grads = tape.jacobian(loss, var_list, unconnected_gradients=tf.UnconnectedGradients.ZERO)
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
    fval = f(x, feed)[0]
    for (_, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        newfval = f(xnew, feed)[0]
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