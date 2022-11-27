import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf
from tensorflow.python.keras.layers import Input, Dense, ReLU, Concatenate, LeakyReLU, Add
from tensorflow.python.keras.models import Model
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import trange
import yaml

BATCH_SIZE = 512
EPOCHS = 100
K = 10
show_fig = True

def create_generator(state_dims, action_dims, code_dims):
    # initializer = tf.keras.initializers.GlorotNormal()
    states = Input(shape=state_dims)
    # default: x = Dense(260, kernel_initializer=initializer, activation='tanh')(states)
    x = Dense(128, activation='tanh')(states)
    # x = LeakyReLU()(x)
    codes = Input(shape=code_dims)
    c = Dense(32, activation='tanh')(codes)
    # c = LeakyReLU()(c)
    h = Concatenate(axis=1)([x,c])
    actions = Dense(action_dims)(h)

    model = Model(inputs=[states,codes], outputs=actions)
    return model

# load data
expert_data = read_expert()
expert_data = extract_features(expert_data)
expert_data = sample_replace_start_end(expert_data)
features, feature_size, expert_data = extract_norm_apertures_wrist_mdp(expert_data)

yaml_conf = {
    'train_states': features['train']['states'].tolist(),
    'train_actions': features['train']['actions'].tolist(),
    'train_codes': features['train']['codes'].tolist(),
    'train_time': features['train']['time'].tolist(),
    'train_norm_time': features['train']['norm_time'].tolist(),
    'test_states': features['test']['states'].tolist(),
    'test_actions': features['test']['actions'].tolist(),
    'test_codes': features['test']['codes'].tolist(),
    'test_time': features['test']['time'].tolist(),
    'test_norm_time': features['test']['norm_time'].tolist(),
    'train_feat_size': feature_size['train'].tolist(),
    'test_feat_size': feature_size['test'].tolist()
}

with open("./saved_models/trpo/dataset.yml", 'w') as f:
    yaml.dump(yaml_conf, f, sort_keys=False, default_flow_style=False)

def train(train_states, train_actions, train_codes, val_states, val_actions, val_codes):
    train_states = tf.convert_to_tensor(train_states, dtype=tf.float32)
    train_actions = tf.convert_to_tensor(train_actions, dtype=tf.float32)
    train_codes = tf.convert_to_tensor(train_codes, dtype=tf.float32)
    val_states = tf.convert_to_tensor(val_states, dtype=tf.float32)
    val_actions = tf.convert_to_tensor(val_actions, dtype=tf.float32)
    val_codes = tf.convert_to_tensor(val_codes, dtype=tf.float32)

    train_data = tf.data.Dataset.from_tensor_slices((train_states, train_actions, train_codes))
    train_data = train_data.batch(batch_size=BATCH_SIZE)

    val_data = tf.data.Dataset.from_tensor_slices((val_states, val_actions, val_codes))
    val_data = val_data.batch(batch_size=BATCH_SIZE)

    # train
    generator = create_generator(features['train']['states'].shape[1], features['train']['actions'].shape[1], features['train']['codes'].shape[1])

    # gen_optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4)
    gen_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    mse = tf.keras.losses.MeanSquaredError()

    epochs = EPOCHS
    # total_train_size = sum([el[0].shape[0] for el in list(train_data.as_numpy_iterator())])
    total_val_size = sum([el[0].shape[0] for el in list(val_data.as_numpy_iterator())])
    result_train = []
    result_train_std = []
    result_val = []
    for epoch in trange(epochs, desc='Epoch'):
        # loss = 0.0
        pure_train_losses = []
        for _, (states_batch, actions_batch, codes_batch) in enumerate(train_data):
            with tf.GradientTape() as gen_tape:
                actions_mu = generator([states_batch, codes_batch], training=True)
                gen_loss = mse(actions_batch, actions_mu)
            
            policy_gradients = gen_tape.gradient(gen_loss, generator.trainable_weights)
            gen_optimizer.apply_gradients(zip(policy_gradients, generator.trainable_weights))

            # loss += tf.get_static_value(gen_loss) * states_batch.shape[0]
            pure_train_losses.append(tf.get_static_value(gen_loss))
        
        # epoch_loss = loss / total_train_size
        pure_train_losses = np.array(pure_train_losses, dtype=np.float32)
        result_train_std.append(pure_train_losses.std())
        result_train.append(pure_train_losses.mean())

        loss = 0.0
        for _, (states_batch, actions_batch, codes_batch) in enumerate(val_data):
            actions_mu = generator([states_batch, codes_batch], training=False)
            gen_loss = mse(actions_batch, actions_mu)
            loss += tf.get_static_value(gen_loss) * states_batch.shape[0]
        
        epoch_loss = loss / total_val_size
        result_val.append(epoch_loss)
    
    return generator, result_train, result_val, np.array(result_train_std, dtype=np.float32)

def train_kfold():
    # initialize k-fold cross validation
    kf = KFold(n_splits=K, shuffle=False, random_state=None)
    avg_train_loss = np.zeros((EPOCHS,), dtype=np.float32)
    avg_val_loss = np.zeros((EPOCHS,), dtype=np.float32)
    avg_train_std = np.zeros((EPOCHS,), dtype=np.float32)
    best_avg_loss = np.inf
    best_gen = None
    mse = tf.keras.losses.MeanSquaredError()

    for train_index, test_index in kf.split(features['train']['states']):
        states_train, states_test = features['train']['states'][train_index], features['train']['states'][test_index]
        actions_train, actions_test = features['train']['actions'][train_index], features['train']['actions'][test_index]
        codes_train, codes_test = features['train']['codes'][train_index], features['train']['codes'][test_index]

        generator, result_train, result_val, result_train_std = train(states_train, actions_train, codes_train, states_test, actions_test, codes_test)
        avg_train_loss += np.array(result_train) / float(K)
        avg_val_loss += np.array(result_val) / float(K)
        avg_train_std += np.array(result_train_std) / float(K)

        if (sum(result_val)/float(len(result_val))) < best_avg_loss:
            best_avg_loss = sum(result_val)/float(len(result_val))
            best_gen = generator

    test_actions_mu = best_gen([features['test']['states'], features['test']['codes']], training=False)
    test_gen_loss = mse(features['test']['actions'], test_actions_mu)

    return best_gen, avg_train_loss, avg_val_loss, avg_train_std, test_gen_loss

def simple_train():
    return train(features['train']['states'], features['train']['actions'], features['train']['codes'], features['test']['states'], features['test']['actions'], features['test']['codes'])

generator = None
result_test = None
result_train = None
result_val = None
result_train_std = None

test_flag = True
if test_flag:
    generator, result_train, result_val, result_train_std, result_test = train_kfold()
    print('\nTest loss: {:f}'.format(result_test))
else: generator, result_train, result_val, result_train_std = simple_train()

if show_fig:
    epoch_space = np.arange(1, len(result_train)+1, dtype=int)
    plt.figure()
    plt.title('Behaviour Cloning')
    plt.plot(epoch_space, result_train, label='train loss')
    plt.fill_between(epoch_space, result_train-result_train_std, result_train+result_train_std, alpha=0.2)
    plt.plot(epoch_space, result_val, label='validation loss')
    plt.legend(loc="upper right")
    plt.savefig('./plots/behaviour_cloning', dpi=100)
    plt.close()

generator.save_weights('./saved_models/bc/generator.h5')
print('\nGenerator saved!') 