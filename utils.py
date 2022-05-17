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

PROB_THRESHOLD = 0.6
DIST_THRESHOLD = 10

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

def filter_out(df, col):
    df = df.loc[df[col+".prob"] > PROB_THRESHOLD]
    coords = df[col+".y"].to_numpy()
    dist = np.abs(coords[1:] - coords[:-1])
    prev_dist = np.r_[dist[0], dist]
    next_dist = np.r_[dist, dist[-1]]
    neighbour_dist = np.min([prev_dist, next_dist], axis=0)
    df = df[neighbour_dist < DIST_THRESHOLD]

    return df

def extract_features(dataset):
    one_hot = {'S': [1,0,0], 'M': [0,1,0], 'L': [0,0,1]}
    features = {}
    features['states'] = []
    features['actions'] = []
    features['codes'] = []
    feature_size = []
    discarded = []

    for key in dataset.keys():
        dataset[key] = filter_out(dataset[key], "RThumb4FingerTip")
        dataset[key] = filter_out(dataset[key], "RIndex4FingerTip")
        if len(dataset[key]) == 0:
            discarded.append(key)
            continue
        
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
    
        window = np.zeros((5,interpolated.shape[1]), dtype=np.float64)
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
        
        actions = np.array(actions, dtype=np.float64)
        states = np.array(states, dtype=np.float64)
        states = states[:-1,:]

        feature_size.append(states.shape[0])

        features['states'].append(states)
        features['actions'].append(actions)
        features['codes'].append(np.array([one_hot[key[OBJ_SIZE_POS]] for _ in range(len(states))]))
    
    features['states'] = np.concatenate(features['states'], axis=0)
    features['actions'] = np.concatenate(features['actions'], axis=0)
    features['codes'] = np.concatenate(features['codes'], axis=0)
    feature_size = np.array(feature_size, dtype=int)

    for key in discarded: dataset.pop(key, None)

    return features, feature_size

def extract_start_pos(features, feat_size):
    start_pos = []
    codes = []
    pos = 0

    for sz in feat_size:
        start_pos.append(features['states'][pos, -3:])
        codes.append(features['codes'][pos])
        pos += sz
    
    return np.array(start_pos, dtype=np.float64), np.array(codes, dtype=np.float64)

def std_10(dataset):
    plt.figure()
    plt.xlabel('time (sec)')
    plt.ylabel('y-wrist std (10 pixels)')
    xmax = np.NINF
    final_stds = []
    for key in dataset.keys():
        std10 = []
        secs = []
        for i in range(len(dataset[key])-10):
            stdy = dataset[key].iloc[i:i+10, 4].to_numpy().std()
            std10.append(stdy)
            secs.append(dataset[key].iloc[i,1] - dataset[key].iloc[0,1])
        
        if max(secs) > xmax: xmax = max(secs)
        final_stds.append(std10[-1])
        plt.scatter(secs, std10, s=4, alpha=0.4, color='steelblue')
    
    plt.xlim(0,xmax)
    plt.grid(color='lightgray', alpha=0.6)
    plt.savefig('std10', dpi=100)
    plt.close()

##################################################################
############################# TRPO ###############################
##################################################################

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