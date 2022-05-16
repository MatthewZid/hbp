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

BATCH_SIZE = 2048
EPOCHS = 200
K = 10
show_fig = True

def create_generator(state_dims, action_dims, code_dims):
    initializer = tf.keras.initializers.GlorotNormal()
    states = Input(shape=state_dims)
    # default: x = Dense(260, kernel_initializer=initializer, activation='tanh')(states)
    x = Dense(260, kernel_initializer=initializer, activation='tanh')(states)
    # x = LeakyReLU()(x)
    codes = Input(shape=code_dims)
    c = Dense(64, kernel_initializer=initializer, activation='tanh')(codes)
    # c = LeakyReLU()(c)
    h = Concatenate(axis=1)([x,c])
    actions = Dense(action_dims)(h)

    model = Model(inputs=[states,codes], outputs=actions)
    return model

# load data
expert_data = read_expert()
features, _ = extract_features(expert_data)

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
    generator = create_generator(features['states'].shape[1], features['actions'].shape[1], features['codes'].shape[1])

    # gen_optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4)
    gen_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    mse = tf.keras.losses.MeanSquaredError()

    epochs = EPOCHS
    total_train_size = sum([el[0].shape[0] for el in list(train_data.as_numpy_iterator())])
    total_val_size = sum([el[0].shape[0] for el in list(val_data.as_numpy_iterator())])
    result_train = []
    result_val = []
    for epoch in trange(epochs, desc='Epoch'):
        loss = 0.0
        for _, (states_batch, actions_batch, codes_batch) in enumerate(train_data):
            with tf.GradientTape() as gen_tape:
                actions_mu = generator([states_batch, codes_batch], training=True)
                gen_loss = mse(actions_batch, actions_mu)
            
            policy_gradients = gen_tape.gradient(gen_loss, generator.trainable_weights)
            gen_optimizer.apply_gradients(zip(policy_gradients, generator.trainable_weights))

            loss += tf.get_static_value(gen_loss) * states_batch.shape[0]
        
        epoch_loss = loss / total_train_size
        result_train.append(epoch_loss)

        loss = 0.0
        for _, (states_batch, actions_batch, codes_batch) in enumerate(val_data):
            actions_mu = generator([states_batch, codes_batch], training=False)
            gen_loss = mse(actions_batch, actions_mu)
            loss += tf.get_static_value(gen_loss) * states_batch.shape[0]
        
        epoch_loss = loss / total_val_size
        result_val.append(epoch_loss)
    
    return generator, result_train, result_val

def train_kfold():
    # initialize k-fold cross validation
    kf = KFold(n_splits=K, shuffle=False, random_state=None)
    avg_train_loss = np.zeros((EPOCHS,), dtype=np.float32)
    avg_val_loss = np.zeros((EPOCHS,), dtype=np.float32)
    best_avg_loss = np.inf
    best_gen = None
    mse = tf.keras.losses.MeanSquaredError()

    idx = np.arange(features['states'].shape[0])
    np.random.shuffle(idx)
    shuffled_expert_states = features['states'][idx]
    shuffled_expert_actions = features['actions'][idx]
    shuffled_expert_codes = features['codes'][idx]

    train_ratio = int((80*features['states'].shape[0])/100.0)
    train_states = shuffled_expert_states[0:train_ratio, :]
    test_states = shuffled_expert_states[train_ratio:, :]
    train_actions = shuffled_expert_actions[0:train_ratio, :]
    test_actions = shuffled_expert_actions[train_ratio:, :]
    train_codes = shuffled_expert_codes[0:train_ratio, :]
    test_codes = shuffled_expert_codes[train_ratio:, :]

    for train_index, test_index in kf.split(train_states):
        states_train, states_test = train_states[train_index], train_states[test_index]
        actions_train, actions_test = train_actions[train_index], train_actions[test_index]
        codes_train, codes_test = train_codes[train_index], train_codes[test_index]

        generator, result_train, result_val = train(states_train, actions_train, codes_train, states_test, actions_test, codes_test)
        avg_train_loss += np.array(result_train) / float(K)
        avg_val_loss += np.array(result_val) / float(K)

        if (sum(result_val)/float(len(result_val))) < best_avg_loss:
            best_avg_loss = sum(result_val)/float(len(result_val))
            best_gen = generator
    
    test_actions_mu = best_gen([test_states, test_codes], training=False)
    test_gen_loss = mse(test_actions, test_actions_mu)

    return best_gen, avg_train_loss, avg_val_loss, test_gen_loss

def simple_train():
    # pretrain with behavioural cloning
    idx = np.arange(features['states'].shape[0])
    np.random.shuffle(idx)
    shuffled_expert_states = features['states'][idx]
    shuffled_expert_actions = features['actions'][idx]
    shuffled_expert_codes = features['codes'][idx]

    train_ratio = int((80*features['states'].shape[0])/100.0)
    train_states = shuffled_expert_states[0:train_ratio, :]
    val_states = shuffled_expert_states[train_ratio:, :]
    train_actions = shuffled_expert_actions[0:train_ratio, :]
    val_actions = shuffled_expert_actions[train_ratio:, :]
    train_codes = shuffled_expert_codes[0:train_ratio, :]
    val_codes = shuffled_expert_codes[train_ratio:, :]

    # normalize states/actions
    state_normalizer = MinMaxScaler(feature_range=(-1,1))
    action_normalizer = MinMaxScaler(feature_range=(-1,1))

    train_states = state_normalizer.fit_transform(train_states)
    train_actions = action_normalizer.fit_transform(train_actions)
    val_states = state_normalizer.transform(val_states)
    val_actions = action_normalizer.transform(val_actions)

    return train(train_states, train_actions, train_codes, val_states, val_actions, val_codes)

generator = None
result_test = None
result_train = None
result_val = None

test_flag = False
if test_flag:
    generator, result_train, result_val, result_test = train_kfold()
    print('\nTest loss: {:f}'.format(result_test))
else: generator, result_train, result_val = simple_train()

if show_fig:
    epoch_space = np.arange(1, len(result_train)+1, dtype=int)
    plt.figure()
    plt.title('Behaviour Cloning')
    plt.plot(epoch_space, result_train)
    plt.plot(epoch_space, result_val)
    plt.legend(['train loss', 'validation loss'], loc="upper right")
    plt.savefig('./plots/behaviour_cloning', dpi=100)
    plt.close()

generator.save_weights('./saved_models/bc/generator.h5')
print('\nGenerator saved!') 