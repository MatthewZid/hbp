import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import yaml
import time

from tqdm import trange
from utils import *
import utils
from scipy.ndimage import shift
import multiprocessing as mp
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
from models import *
from env import Env
import seaborn as sn
from copy import deepcopy

class Agent():
    def __generate_trajectory(self, code, start, feat_size):
        s_traj = []
        a_traj = []
        c_traj = []
        env = Env()

        # generate actions for every current state
        state_obsrv = env.reset(start, feat_size) # reset environment state
        code_tf = tf.constant(code)
        code_tf = tf.expand_dims(code_tf, axis=0)

        while True:
            # 1. generate actions with generator
            state_tf = tf.constant(state_obsrv)
            state_tf = tf.expand_dims(state_tf, axis=0)
            action_mu = models.generator.model([state_tf, code_tf], training=False)#.numpy()[0] # when action_dims > 1, don't use .numpy()[0]
            action_mu = tf.squeeze(action_mu).numpy()

            s_traj.append(state_obsrv)
            a_traj.append(action_mu)
            c_traj.append(code)

            # 2. environment step
            state_obsrv, done = env.step(action_mu)

            if done:
                s_traj = np.array(s_traj, dtype=np.float64)
                a_traj = np.array(a_traj, dtype=np.float64)
                c_traj = np.array(c_traj, dtype=np.float64)
                break
        
        return (s_traj, a_traj, c_traj)
    
    def run(self, code, start, feat_size):
        try:
            trajectory_dict = {}
            trajectory = self.__generate_trajectory(code, start, feat_size)
            trajectory_dict['states'] = np.copy(trajectory[0])
            trajectory_dict['actions'] = np.copy(trajectory[1])
            trajectory_dict['codes'] = np.copy(trajectory[2])
            return trajectory_dict
        except KeyboardInterrupt:
            time.sleep(1)
        
    def run_test(self, code, start, feat_size):
        try:
            current_code = np.argmax(code)
            trajectory_total = {
                'code': current_code,
                'trajectories': []
            }

            latent_codes = []
            for i in range(3):
                c = np.zeros((3,))
                c[i] = 1
                latent_codes.append(np.copy(c))
            latent_codes = np.array(latent_codes)

            for lc in latent_codes:
                trajectory_dict = {}
                trajectory = self.__generate_trajectory(lc, start, feat_size)
                trajectory_dict['states'] = np.copy(trajectory[0])
                trajectory_dict['actions'] = np.copy(trajectory[1])
                trajectory_dict['codes'] = np.copy(trajectory[2])
                trajectory_total['trajectories'].append(trajectory_dict)

            return trajectory_total
        except KeyboardInterrupt:
            time.sleep(1)

class InfoGAIL():
    def __init__(self, batch_size=256, code_batch=400, episodes=10000, gamma=0.997, lam=0.97):
        self.batch = batch_size
        self.code_batch = code_batch
        self.episodes = episodes
        self.gamma = gamma
        self.lam = lam
        self.starting_episode = 0
        self.gen_result = []
        self.disc_result = []
        self.post_result = []
        self.post_result_val = []
        self.value_result = []
        self.total_rewards = []
    
    def __saveplot(self, x, y, episode, element='element', mode='plot'):
        plt.figure()
        if mode == 'plot': plt.plot(x, y)
        else: plt.scatter(x, y, alpha=0.4)
        plt.savefig('./plots/'+element+'_'+str(episode), dpi=100)
        plt.close()
    
    def show_loss(self):
        epoch_space = np.arange(1, len(self.gen_result)+1, dtype=int)
        plt.figure()
        # plt.ylim(-100, 100)
        plt.title('Surrogate loss')
        plt.plot(epoch_space, self.gen_result)
        plt.savefig('./plots/trpo_loss', dpi=100)
        plt.close()

        plt.figure()
        plt.title('Disc/Post losses')
        plt.plot(epoch_space, self.disc_result)
        plt.plot(epoch_space, self.post_result)
        plt.plot(epoch_space, self.post_result_val)
        plt.legend(['disc', 'post train', 'post val'], loc="upper right")
        plt.savefig('./plots/disc_post', dpi=100)
        plt.close()

        plt.figure()
        plt.title('Value loss')
        plt.plot(epoch_space, self.value_result)
        plt.savefig('./plots/value_loss', dpi=100)
        plt.close()
    
    def train(self, agent):
        features = {}
        feature_size = {}
        generator_weight_path = ''

        if resume_training:
            with open("./saved_models/trpo/model.yml", 'r') as f:
                data = yaml.safe_load(f)
                self.starting_episode = data['episode']
                print('\nRestarting from episode {:d}'.format(self.starting_episode))
                self.gen_result = data['gen_loss']
                self.disc_result = data['disc_loss']
                self.post_result = data['post_loss']
                self.post_result_val = data['post_loss_val']
                self.value_result = data['value_loss']
            generator_weight_path = './saved_models/trpo/generator.h5'
            models.discriminator.model.load_weights('./saved_models/trpo/discriminator.h5')
            models.posterior.model.load_weights('./saved_models/trpo/posterior.h5')
            models.value_net.model.load_weights('./saved_models/trpo/value_net.h5')

            with open("./saved_models/trpo/dataset.yml", 'r') as f:
                data = yaml.safe_load(f)
                features['train'] = {
                    'states': np.array(data['train_states'], dtype=np.float64),
                    'actions': np.array(data['train_actions'], dtype=np.float64),
                    'codes': np.array(data['train_codes'], dtype=np.float64),
                    'norm_time': np.array(data['train_norm_time'], dtype=np.float64)
                }
                feature_size['train'] = np.array(data['train_feat_size'], dtype=int)
        else:
            generator_weight_path = './saved_models/bc/generator.h5'

            # load data
            # expert_data = read_expert()
            # expert_data = extract_features(expert_data)
            # features, feature_size, expert_data, feat_width = extract_norm_apertures_wrist_mdp(expert_data)

            with open("./saved_models/trpo/dataset.yml", 'r') as f:
                data = yaml.safe_load(f)
                features = {
                    'states': np.array(data['train_states'], dtype=np.float64),
                    'actions': np.array(data['train_actions'], dtype=np.float64),
                    'codes': np.array(data['train_codes'], dtype=np.float64),
                    'norm_time': np.array(data['train_norm_time'], dtype=np.float64)
                }
                feature_size = np.array(data['train_feat_size'], dtype=int)
        
        models.generator.model.load_weights(generator_weight_path)
        models.generator.old_model.load_weights(generator_weight_path)
        # train_start_pos, train_start_codes = extract_start_pos(features, feature_size, feat_width)
        partial_start_pos, partial_start_codes, partial_feat_size, partial_intervals = self.partial_starting_points(features, feature_size)
        print('\nTraining setup ready!')

        for episode in trange(self.starting_episode, self.episodes, desc="Episode"):
            # Sample a batch of latent codes: ci ∼ p(c)
            # pick_whole = np.random.choice(train_start_codes.shape[0], self.code_batch, replace=False)
            pick_partial = np.random.choice(partial_start_codes.shape[0], self.code_batch, replace=False)
            # pick = np.random.choice(train_start_codes.shape[0], self.code_batch, replace=False)
            # sampled_codes = train_start_codes[pick]
            # sampled_pos = train_start_pos[pick]
            # sampled_feat_size = total_feat_size[pick]
            # sampled_whole_codes = train_start_codes[pick_whole]
            sampled_partial_codes = partial_start_codes[pick_partial]
            # sampled_whole_pos = train_start_pos[pick_whole]
            sampled_partial_pos = partial_start_pos[pick_partial]
            # sampled_whole_feat_size = feature_size[pick_whole]
            sampled_partial_feat_size = partial_feat_size[pick_partial]
            # sampled_codes = np.concatenate([sampled_whole_codes, sampled_partial_codes], axis=0)
            # sampled_pos = np.concatenate([sampled_whole_pos, sampled_partial_pos], axis=0)
            # sampled_feat_size = np.concatenate([sampled_whole_feat_size, sampled_partial_feat_size], axis=0)
            # starting_pos_code_pairs = list(zip(sampled_codes, sampled_pos, sampled_feat_size))
            starting_pos_code_pairs = list(zip(sampled_partial_codes, sampled_partial_pos, sampled_partial_feat_size))
            # starting_pos_code_pairs = list(zip(train_start_codes, train_start_pos, feature_size))

            # Sample trajectories: τi ∼ πθi(ci), with the latent code fixed during each rollout
            trajectories = []
            with mp.Pool(mp.cpu_count()) as pool:
                trajectories = pool.starmap(agent.run, starting_pos_code_pairs)
            
            # Sample from buffer
            # for traj in trajectories:
            #     self.buffer.add(traj)
            # trajectories = self.buffer.sample()
            
            # for traj in trajectories:
            #     traj['old_action_mus'] = models.generator.model([traj['states'], traj['codes']], training=False)
            
            generated_states = np.concatenate([traj['states'] for traj in trajectories], dtype=np.float32)
            generated_actions = np.concatenate([traj['actions'] for traj in trajectories], dtype=np.float32)
            generated_codes = np.concatenate([traj['codes'] for traj in trajectories], dtype=np.float32)
            # generated_oldactions = np.concatenate([traj['old_action_mus'] for traj in trajectories], dtype=np.float32)

            # train discriminator
            # Sample state-action pairs χi ~ τi and χΕ ~ τΕ with the same batch size
            expert_idx = np.random.choice(features['states'].shape[0], TRAIN_BATCH_SIZE, replace=False)
            sample_population = int((TRAIN_BATCH_SIZE * 100) / 70.0)
            generated_idx_also_for_post = np.random.choice(generated_states.shape[0], sample_population, replace=False)
            generated_idx = generated_idx_also_for_post[:TRAIN_BATCH_SIZE]
            
            shuffled_expert_states = features['states'][expert_idx, :]
            shuffled_expert_actions = features['actions'][expert_idx, :]
            shuffled_generated_states = generated_states[generated_idx, :]
            shuffled_generated_actions = generated_actions[generated_idx, :]
            shuffled_generated_codes = generated_codes[generated_idx, :]

            # if features['states'].shape[0] < generated_states.shape[0]:
            #     expert_idx = np.arange(features['states'].shape[0])
            #     np.random.shuffle(expert_idx)
            #     shuffled_expert_states = features['states'][expert_idx, :]
            #     shuffled_expert_actions = features['actions'][expert_idx, :]

            #     generated_idx = np.random.choice(generated_states.shape[0], features['states'].shape[0], replace=False)
            #     shuffled_generated_states = generated_states[generated_idx, :]
            #     shuffled_generated_actions = generated_actions[generated_idx, :]
            # elif features['states'].shape[0] > generated_states.shape[0]:
            #     generated_idx = np.arange(generated_states.shape[0])
            #     np.random.shuffle(generated_idx)
            #     shuffled_generated_states = generated_states[generated_idx, :]
            #     shuffled_generated_actions = generated_actions[generated_idx, :]

            #     expert_idx = np.random.choice(features['states'].shape[0], generated_states.shape[0], replace=False)
            #     shuffled_expert_states = features['states'][expert_idx, :]
            #     shuffled_expert_actions = features['actions'][expert_idx, :]
            # else:
            #     expert_idx = np.arange(features['states'].shape[0])
            #     np.random.shuffle(expert_idx)
            #     shuffled_expert_states = features['states'][expert_idx, :]
            #     shuffled_expert_actions = features['actions'][expert_idx, :]

            #     generated_idx = np.arange(generated_states.shape[0])
            #     np.random.shuffle(generated_idx)
            #     shuffled_generated_states = generated_states[generated_idx, :]
            #     shuffled_generated_actions = generated_actions[generated_idx, :]

            shuffled_generated_states = tf.convert_to_tensor(shuffled_generated_states, dtype=tf.float64)
            shuffled_generated_actions = tf.convert_to_tensor(shuffled_generated_actions, dtype=tf.float64)
            shuffled_generated_codes = tf.convert_to_tensor(shuffled_generated_codes, dtype=tf.float64)
            shuffled_expert_states = tf.convert_to_tensor(shuffled_expert_states, dtype=tf.float64)
            shuffled_expert_actions = tf.convert_to_tensor(shuffled_expert_actions, dtype=tf.float64)

            dataset = tf.data.Dataset.from_tensor_slices((shuffled_generated_states, shuffled_generated_actions, shuffled_expert_states, shuffled_expert_actions))
            dataset = dataset.batch(batch_size=self.batch)

            loss = models.discriminator.train(dataset)
            if save_loss: self.disc_result.append(loss)

            # train posterior
            # if features['states'].shape[0] < generated_states.shape[0]:
            #     generated_idx = np.random.choice(generated_states.shape[0], features['states'].shape[0], replace=False)
            # else:
            #     generated_idx = np.arange(generated_states.shape[0])
            #     np.random.shuffle(generated_idx)

            # shuffled_generated_states = generated_states[generated_idx, :]
            # shuffled_generated_actions = generated_actions[generated_idx, :]
            # shuffled_generated_codes = generated_codes[generated_idx, :]
            # shuffled_generated_states = tf.convert_to_tensor(shuffled_generated_states, dtype=tf.float64)
            # shuffled_generated_actions = tf.convert_to_tensor(shuffled_generated_actions, dtype=tf.float64)
            # shuffled_generated_codes = tf.convert_to_tensor(shuffled_generated_codes, dtype=tf.float64)

            # create validation dataset for posterior
            val_generated_idx = generated_idx_also_for_post[TRAIN_BATCH_SIZE:]
            val_generated_states = generated_states[val_generated_idx, :]
            val_generated_actions = generated_actions[val_generated_idx, :]
            val_generated_codes = generated_codes[val_generated_idx, :]
            train_dataset = tf.data.Dataset.from_tensor_slices((shuffled_generated_states, shuffled_generated_actions, shuffled_generated_codes))
            train_dataset = train_dataset.batch(batch_size=self.batch)
            val_dataset = tf.data.Dataset.from_tensor_slices((val_generated_states, val_generated_actions, val_generated_codes))
            val_dataset = val_dataset.batch(batch_size=self.batch)

            train_loss = models.posterior.train(train_dataset)
            val_loss = models.posterior.train(val_dataset)
            if save_loss:
                self.post_result.append(train_loss)
                self.post_result_val.append(val_loss)

            # TRPO/PPO
            # calculate rewards from discriminator and posterior
            episode_rewards = []
            for traj in trajectories:
                reward_d = (-tf.math.log(tf.keras.activations.sigmoid(models.discriminator.model([traj['states'], traj['actions']], training=False)))).numpy()
                reward_p = models.posterior.target_model([traj['states'], traj['actions']], training=False).numpy()
                reward_p = np.sum(np.ma.log(reward_p).filled(0) * traj['codes'], axis=1).flatten() # np.ma.log over tf.math.log, fixes log of zero

                traj['rewards'] = reward_d.flatten() + reward_p
                episode_rewards.append(traj['rewards'].sum())

                # calculate values, advants and returns
                values = models.value_net.model([traj['states'], traj['codes']], training=False).numpy().flatten() # Value function
                values_next = shift(values, -1, cval=0)
                deltas = traj['rewards'] + self.gamma * values_next - values # Advantage(st,at) = rt+1 + γ*V(st+1) - V(st)
                traj['advants'] = discount(deltas, self.gamma * self.lam)
                traj['returns'] = discount(traj['rewards'], self.gamma)
            
            advants = np.concatenate([traj['advants'] for traj in trajectories], dtype=np.float32)
            # advants /= advants.std()

            # train value net for next iter
            returns = np.expand_dims(np.concatenate([traj['returns'] for traj in trajectories]), axis=1)

            # if features['states'].shape[0] < generated_states.shape[0]:
            #     generated_idx = np.random.choice(generated_states.shape[0], features['states'].shape[0], replace=False)
            # else:
            #     generated_idx = np.arange(generated_states.shape[0])
            #     np.random.shuffle(generated_idx)

            generated_idx = np.random.choice(generated_states.shape[0], TRAIN_BATCH_SIZE, replace=False)
                
            shuffled_generated_states = generated_states[generated_idx, :]
            shuffled_generated_codes = generated_codes[generated_idx, :]
            shuffled_returns = returns[generated_idx, :]
            shuffled_generated_states = tf.convert_to_tensor(shuffled_generated_states, dtype=tf.float64)
            shuffled_generated_codes = tf.convert_to_tensor(shuffled_generated_codes, dtype=tf.float64)
            shuffled_returns = tf.convert_to_tensor(shuffled_returns, dtype=tf.float64)
            dataset = tf.data.Dataset.from_tensor_slices((shuffled_generated_states, shuffled_generated_codes, shuffled_returns))
            dataset = dataset.batch(batch_size=self.batch)

            loss = models.value_net.train(dataset)
            if save_loss: self.value_result.append(loss)

            # generator training
            feed = {
                'states': generated_states,
                'actions': generated_actions,
                'codes': generated_codes,
                'advants': advants
                # 'old_mus': generated_oldactions
            }

            loss = models.generator.train(feed)
            if save_loss: self.gen_result.append(loss)

            if save_loss:
                # plot rewards and losses
                episode_reward = np.array(episode_rewards, dtype=np.float64).mean()
                self.total_rewards.append(episode_reward)
                if ((episode+1) % 100) == 0:
                    self.show_loss()
                    epoch_space = np.arange(1, len(self.total_rewards)+1, dtype=int)
                    self.__saveplot(epoch_space, self.total_rewards, 0, 'rewards')
            
            if ((episode+1) % 100 == 0): print('Theta updates so far: {:d}'.format(utils.improved))

            if save_models and ((episode+1) % 100 == 0):
                models.generator.model.save_weights('./saved_models/trpo/generator.h5')
                models.discriminator.model.save_weights('./saved_models/trpo/discriminator.h5')
                models.posterior.model.save_weights('./saved_models/trpo/posterior.h5')
                models.posterior.target_model.save_weights('./saved_models/trpo/posterior_target.h5')
                models.value_net.model.save_weights('./saved_models/trpo/value_net.h5')
                yaml_conf = {
                    'episode': episode+1,
                    'gen_loss': self.gen_result,
                    'disc_loss': self.disc_result,
                    'post_loss': self.post_result,
                    'post_loss_val': self.post_result_val,
                    'value_loss': self.value_result
                }
                
                with open("./saved_models/trpo/model.yml", 'w') as f:
                    yaml.dump(yaml_conf, f, sort_keys=False, default_flow_style=False)

    def partial_starting_points(self, features, feature_size):
        intervals = [0.0, 20.0, 40.0, 60.0, 80.0]
        part_start_pos = []
        part_start_codes = []
        part_feat_size = []
        partial_interval = []

        pos = 0
        for sz in feature_size:
            expert_states = features['states'][pos:pos+sz]
            expert_codes = features['codes'][pos:pos+sz]
            expert_norm_time = features['norm_time'][pos:pos+sz]

            # get starting positions
            for ci in intervals:
                partial_expert_idx = np.where(expert_norm_time >= ci)[0]
                part_start_pos.append(expert_states[partial_expert_idx[0]])
                part_start_codes.append(expert_codes[partial_expert_idx[0]])
                part_feat_size.append(partial_expert_idx.shape[0])
                partial_interval.append(ci)
            pos += sz

        return np.array(part_start_pos, dtype=np.float64), np.array(part_start_codes, dtype=np.float64), np.array(part_feat_size, dtype=int), np.array(partial_interval, dtype=np.float64)

    def test(self, agent):
        features = {}
        feature_size = None

        models.generator.model.load_weights('./saved_models/trpo/generator.h5')
        models.discriminator.model.load_weights('./saved_models/trpo/discriminator.h5')
        models.posterior.model.load_weights('./saved_models/trpo/posterior.h5')
        models.posterior.target_model.load_weights('./saved_models/trpo/posterior_target.h5')

        with open("./saved_models/trpo/dataset.yml", 'r') as f:
            data = yaml.safe_load(f)
            features = {
                'states': np.array(data['test_states'], dtype=np.float64),
                'actions': np.array(data['test_actions'], dtype=np.float64),
                'codes': np.array(data['test_codes'], dtype=np.float64),
                'norm_time': np.array(data['test_norm_time'], dtype=np.float64)
            }

            training_features = {
                'states': np.array(data['train_states'], dtype=np.float64),
                'actions': np.array(data['train_actions'], dtype=np.float64),
                'codes': np.array(data['train_codes'], dtype=np.float64)
            }
            feature_size = np.array(data['test_feat_size'], dtype=int)
        
        # InfoGAIL accuracy method
        sampled_expert_idx = np.random.choice(training_features['states'].shape[0], 1000, replace=False)
        sampled_expert_states = training_features['states'][sampled_expert_idx, :]
        sampled_expert_actions = training_features['actions'][sampled_expert_idx, :]
        sampled_expert_codes = np.argmax(training_features['codes'][sampled_expert_idx, :], axis=1)
        probs = models.posterior.target_model([sampled_expert_states, sampled_expert_actions], training=False).numpy()
        codes_pred = np.argmax(probs, axis=1)

        print('Posterior accuracy over expert state-action pairs')
        print(classification_report(sampled_expert_codes, codes_pred))
        print('\n')

        ######################################################################################################

        intervals = [0.0, 20.0, 40.0, 60.0, 80.0]
        part_prob_percs_per_object = []

        part_total_codes_true = []
        part_total_codes_pred = []

        total_disc_expert = []
        total_disc_gen = []
        total_eval_post = []

        obj_trajectories = {
            '0': [],
            '1': [],
            '2': []
        }

        obj_size = {'Small': 0, 'Medium': 1, 'Large': 2}

        for i in range(len(intervals)):
            part_prob_percs_per_object.append([])
            part_total_codes_true.append([])
            part_total_codes_pred.append([])
            for j in range(features['codes'].shape[1]):
                part_prob_percs_per_object[i].append([])
                for _ in range(features['codes'].shape[1]):
                    part_prob_percs_per_object[i][j].append([])

        # partial movements
        print('Starting from completion percentage (e.g. 40%) until movement completion:')
        count = 0
        for ci in intervals:
            # get starting positions
            pos = 0
            part_start_pos = []
            part_start_codes = []
            part_feat_size = []
            partial_features = {}
            partial_features['states'] = []
            partial_features['actions'] = []
            partial_features['codes'] = []
            for sz in feature_size:
                expert_states = features['states'][pos:pos+sz]
                expert_actions = features['actions'][pos:pos+sz]
                expert_codes = features['codes'][pos:pos+sz]
                expert_norm_time = features['norm_time'][pos:pos+sz]
                partial_expert_idx = np.where(expert_norm_time >= ci)[0]
                partial_features['states'].append(expert_states[partial_expert_idx, :])
                partial_features['actions'].append(expert_actions[partial_expert_idx, :])
                partial_features['codes'].append(expert_codes[partial_expert_idx, :])
                part_start_pos.append(expert_states[partial_expert_idx[0]])
                part_start_codes.append(expert_codes[partial_expert_idx[0]])
                part_feat_size.append(partial_expert_idx.shape[0])
                pos += sz

            partial_features['states'] = np.concatenate(partial_features['states'], axis=0, dtype=np.float64)
            partial_features['actions'] = np.concatenate(partial_features['actions'], axis=0, dtype=np.float64)
            partial_features['codes'] = np.concatenate(partial_features['codes'], axis=0, dtype=np.float64)

            part_start_pos = np.array(part_start_pos, dtype=np.float64)
            part_feat_size = np.array(part_feat_size, dtype=int)

            # trajectory generation
            starting_pos_code_pairs = list(zip(part_start_codes, part_start_pos, part_feat_size))

            trajectories = []
            with mp.Pool(mp.cpu_count()) as pool:
                trajectories = pool.starmap(agent.run_test, starting_pos_code_pairs)

            pos = 0
            expert_i = 0
            for traj in trajectories:
                expert_states = partial_features['states'][pos:pos+part_feat_size[expert_i]]
                expert_actions = partial_features['actions'][pos:pos+part_feat_size[expert_i]]

                if count == 0:
                    # discriminate
                    if expert_states.shape[0] < traj['trajectories'][traj['code']]['states'].shape[0]:
                        expert_idx = np.arange(expert_states.shape[0])
                        np.random.shuffle(expert_idx)
                        shuffled_expert_states = expert_states[expert_idx, :]
                        shuffled_expert_actions = expert_actions[expert_idx, :]

                        generated_idx = np.random.choice(traj['trajectories'][traj['code']]['states'].shape[0], expert_states.shape[0], replace=False)
                        shuffled_generated_states = traj['trajectories'][traj['code']]['states'][generated_idx, :]
                        shuffled_generated_actions = traj['trajectories'][traj['code']]['actions'][generated_idx, :]
                    elif expert_states.shape[0] > traj['trajectories'][traj['code']]['states'].shape[0]:
                        generated_idx = np.arange(traj['trajectories'][traj['code']]['states'].shape[0])
                        np.random.shuffle(generated_idx)
                        shuffled_generated_states = traj['trajectories'][traj['code']]['states'][generated_idx, :]
                        shuffled_generated_actions = traj['trajectories'][traj['code']]['actions'][generated_idx, :]

                        expert_idx = np.random.choice(expert_states.shape[0], traj['trajectories'][traj['code']]['states'].shape[0], replace=False)
                        shuffled_expert_states = expert_states[expert_idx, :]
                        shuffled_expert_actions = expert_actions[expert_idx, :]
                    else:
                        expert_idx = np.arange(expert_states.shape[0])
                        np.random.shuffle(expert_idx)
                        shuffled_expert_states = expert_states[expert_idx, :]
                        shuffled_expert_actions = expert_actions[expert_idx, :]

                        generated_idx = np.arange(traj['trajectories'][traj['code']]['states'].shape[0])
                        np.random.shuffle(generated_idx)
                        shuffled_generated_states = traj['trajectories'][traj['code']]['states'][generated_idx, :]
                        shuffled_generated_actions = traj['trajectories'][traj['code']]['actions'][generated_idx, :]

                    score1 = models.discriminator.model([shuffled_generated_states, shuffled_generated_actions], training=False)
                    score2 = models.discriminator.model([shuffled_expert_states, shuffled_expert_actions], training=False)
                    binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
                    score1 = tf.squeeze(score1).numpy()
                    score2 = tf.squeeze(score2).numpy()
                    y_ones = np.ones_like(score1)
                    y_zeros = np.zeros_like(score2)
                    total_disc_expert.append(binary_cross_entropy(y_zeros, score2))
                    total_disc_gen.append(binary_cross_entropy(y_ones, score1))

                code_true = traj['code']
                best_traj = {}
                min_rmse = np.inf

                # generated trajectories for every latent code
                for i in range(partial_features['codes'].shape[1]):
                    generated_states = traj['trajectories'][i]['states']
                    generated_actions = traj['trajectories'][i]['actions']
                    generated_codes = traj['trajectories'][i]['codes']
                
                    # rmse to find the closest generated trajectory to expert
                    rmse_traj = np.sqrt(mean_squared_error(expert_states, generated_states))# / expert_states.shape[0]
                    if rmse_traj < min_rmse:
                        min_rmse = rmse_traj
                        best_traj['states'] = np.copy(generated_states)
                        best_traj['actions'] = np.copy(generated_actions)
                        best_traj['codes'] = np.copy(generated_codes)
                
                if expert_i == 60 and count == 0:
                    # plot best and expert trajectories
                    # plt.figure(1)
                    # plt.scatter(best_traj['states'][:, 0], best_traj['states'][:, 1], alpha=0.6)
                    # plt.scatter(expert_states[:, 0], expert_states[:, 1], alpha=0.6)
                    # plt.legend(['generated', 'expert'], loc='lower right')
                    # plt.savefig('./plots/trajectories', dpi=100)
                    # plt.close(1)
                    plt.figure(1)
                    plt.scatter(np.arange(best_traj['states'].shape[0]), best_traj['states'][:, 2])
                    plt.scatter(np.arange(expert_states.shape[0]), expert_states[:, 2], alpha=0.6)
                    plt.legend(['generated', 'expert'], loc='lower right')
                    plt.savefig('./plots/trajectories', dpi=100)
                    plt.close(1)
                
                # percentages for the partial trajectory
                part_prob = models.posterior.model([best_traj['states'], best_traj['actions']], training=False)
                if count == 0:
                    cross_entropy = tf.keras.losses.CategoricalCrossentropy()
                    loss = cross_entropy(best_traj['codes'], part_prob)
                    loss = tf.reduce_mean(loss).numpy()
                    total_eval_post.append(loss)
                part_prob_mean = tf.reduce_mean(part_prob, axis=0).numpy()
                part_prob_pred = np.argmax(part_prob_mean)
                part_prob_percs_per_object[count][code_true][0].append(part_prob_mean[0])
                part_prob_percs_per_object[count][code_true][1].append(part_prob_mean[1])
                part_prob_percs_per_object[count][code_true][2].append(part_prob_mean[2])
                part_total_codes_true[count].append(code_true)
                part_total_codes_pred[count].append(part_prob_pred)

                if count == 0:
                    obj_trajectories[str(code_true)].append((best_traj['states'], best_traj['actions']))

                pos += part_feat_size[expert_i]
                expert_i += 1
            
            count += 1

        # plot partial trajectory results
        count = 0
        for ci in intervals:
            cm = confusion_matrix(part_total_codes_true[count], part_total_codes_pred[count])
            group_counts = ['{}'.format(v) for v in cm.flatten()]
            group_perc = ['{:.2%}'.format(v) for v in cm.flatten()/np.sum(cm)]
            annot_labels = [f'{v1}\n({v2})' for v1,v2 in zip(group_perc, group_counts)]
            annot_labels = np.asarray(annot_labels).reshape(cm.shape)
            hm = sn.heatmap(cm, annot=annot_labels, fmt='', cmap='Blues')
            hm.set_title('Mode confusion matrix: '+str(int(ci))+'% completion')
            hm.set_xlabel('Predicted modes')
            hm.set_ylabel('Actual modes')
            hm.set_xticklabels(['Small', 'Medium', 'Large'])
            hm.set_yticklabels(['Small', 'Medium', 'Large'])
            plt.savefig('./plots/cf_matrix_'+str(int(ci)), dpi=100)
            plt.close()

            print('{:d}%'.format(int(ci)))
            print(classification_report(part_total_codes_true[count], part_total_codes_pred[count], zero_division=1))
            print('-----------------------------------------------')

            for sz in ['Small', 'Medium', 'Large']:
                plt.figure()
                plt.title(sz+': '+str(int(ci))+'% completion')
                plt.xlabel('Trajectory No.')
                plt.ylabel('Probability (%)')
                plt.scatter(np.arange(len(part_prob_percs_per_object[count][obj_size[sz]][0])), part_prob_percs_per_object[count][obj_size[sz]][0], alpha=0.6)
                plt.scatter(np.arange(len(part_prob_percs_per_object[count][obj_size[sz]][1])), part_prob_percs_per_object[count][obj_size[sz]][1], alpha=0.6)
                plt.scatter(np.arange(len(part_prob_percs_per_object[count][obj_size[sz]][2])), part_prob_percs_per_object[count][obj_size[sz]][2], alpha=0.6)
                plt.legend(['Small', 'Medium', 'Large'], loc='lower right')
                plt.savefig('./plots/conf_percs_'+sz+'_'+str(int(ci)), dpi=100)
                plt.close()

            count += 1
        
        # franken-trajectories
        franken_codes_true = []
        franken_codes_pred = []
        franken_prob_perc_per_object = []

        for i in range(features['codes'].shape[1]):
            franken_prob_perc_per_object.append([])
            for _ in range(features['codes'].shape[1]): franken_prob_perc_per_object[i].append([])

        for _ in range(30):
            random_code_idx = np.random.choice(features['codes'].shape[1], 2, replace=False)
            random_upper_half_idx = np.random.choice(len(obj_trajectories[str(random_code_idx[0])]), 1)[0]
            random_lower_half_idx = np.random.choice(len(obj_trajectories[str(random_code_idx[1])]), 1)[0]
            upper_splitter = int((50 * obj_trajectories[str(random_code_idx[0])][random_upper_half_idx][0].shape[0]) / 100.0)
            lower_splitter = int((50 * obj_trajectories[str(random_code_idx[1])][random_lower_half_idx][0].shape[0]) / 100.0)
            random_upper_half = (obj_trajectories[str(random_code_idx[0])][random_upper_half_idx][0][:upper_splitter], \
                                obj_trajectories[str(random_code_idx[0])][random_upper_half_idx][1][:upper_splitter])
            random_lower_half = (obj_trajectories[str(random_code_idx[1])][random_lower_half_idx][0][lower_splitter:], \
                                obj_trajectories[str(random_code_idx[1])][random_lower_half_idx][1][lower_splitter:])
            
            franken_traj = {}
            franken_traj['states'] = np.concatenate([random_upper_half[0], random_lower_half[0]], axis=0)
            franken_traj['actions'] = np.concatenate([random_upper_half[1], random_lower_half[1]], axis=0)
            franken_true_code = random_code_idx[1]

            franken_prob = models.posterior.model([franken_traj['states'], franken_traj['actions']], training=False)
            franken_prob_mean = tf.reduce_mean(franken_prob, axis=0).numpy()
            franken_prob_pred = np.argmax(franken_prob_mean)
            franken_prob_perc_per_object[franken_true_code][0].append(franken_prob_mean[0])
            franken_prob_perc_per_object[franken_true_code][1].append(franken_prob_mean[1])
            franken_prob_perc_per_object[franken_true_code][2].append(franken_prob_mean[2])
            franken_codes_true.append(franken_true_code)
            franken_codes_pred.append(franken_prob_pred)
        
        cm = confusion_matrix(franken_codes_true, franken_codes_pred)
        group_counts = ['{}'.format(v) for v in cm.flatten()]
        group_perc = ['{:.2%}'.format(v) for v in cm.flatten()/np.sum(cm)]
        annot_labels = [f'{v1}\n({v2})' for v1,v2 in zip(group_perc, group_counts)]
        annot_labels = np.asarray(annot_labels).reshape(cm.shape)
        hm = sn.heatmap(cm, annot=annot_labels, fmt='', cmap='Blues')
        hm.set_title('Mixed Mode confusion matrix')
        hm.set_xlabel('Predicted modes')
        hm.set_ylabel('Actual modes')
        hm.set_xticklabels(['Small', 'Medium', 'Large'])
        hm.set_yticklabels(['Small', 'Medium', 'Large'])
        plt.savefig('./plots/franken_cf_matrix', dpi=100)
        plt.close()

        for sz in ['Small', 'Medium', 'Large']:
            plt.figure()
            plt.title('Mixed probabilities: '+sz)
            plt.xlabel('Trajectory No.')
            plt.ylabel('Probability (%)')
            plt.scatter(np.arange(len(franken_prob_perc_per_object[obj_size[sz]][0])), franken_prob_perc_per_object[obj_size[sz]][0], alpha=0.6)
            plt.scatter(np.arange(len(franken_prob_perc_per_object[obj_size[sz]][1])), franken_prob_perc_per_object[obj_size[sz]][1], alpha=0.6)
            plt.scatter(np.arange(len(franken_prob_perc_per_object[obj_size[sz]][2])), franken_prob_perc_per_object[obj_size[sz]][2], alpha=0.6)
            plt.legend(['Small', 'Medium', 'Large'], loc='lower right')
            plt.savefig('./plots/franken_conf_percs_'+sz, dpi=100)
            plt.close()

        # multiple random starting points
        # p = 0
        # for traj in trajectories:
        #     pos = feature_size[:pick[p]].sum()

        #     expert_states = features['states'][pos:pos+feature_size[pick[p]]]
        #     expert_actions = features['actions'][pos:pos+feature_size[pick[p]]]
        #     expert_norm_time = features['norm_time'][pos:pos+feature_size[pick[p]]]

        #     # posterior
        #     few_examples = False
        #     code_true = traj['code']
        #     best_traj = {}
        #     min_rmse = np.inf

        #     for i in range(features['codes'].shape[1]):
        #         shuffled_generated_states = traj['trajectories'][i]['states']
        #         shuffled_generated_actions = traj['trajectories'][i]['actions']
        #         shuffled_generated_codes = traj['trajectories'][i]['codes']

        #         # rmse to find the closest generated trajectory to expert
        #         rmse_traj = np.sqrt(mean_squared_error(expert_states, shuffled_generated_states))# / shuffled_expert_states.shape[0]
        #         if rmse_traj < min_rmse:
        #             min_rmse = rmse_traj
        #             best_traj['states'] = np.copy(shuffled_generated_states)
        #             best_traj['actions'] = np.copy(shuffled_generated_actions)
        #             best_traj['codes'] = np.copy(shuffled_generated_codes)
            
        #     # percentages for the whole trajectory
        #     prob = models.posterior.model([best_traj['states'], best_traj['actions']], training=False)
        #     cross_entropy = tf.keras.losses.CategoricalCrossentropy()
        #     loss = cross_entropy(best_traj['codes'], prob)
        #     loss = tf.reduce_mean(loss).numpy()
        #     total_eval_post.append(loss)
        #     prob_mean = tf.reduce_mean(prob, axis=0).numpy()
        #     prob_pred = np.argmax(prob_mean)
        #     prob_percs_per_object[code_true][0].append(prob_mean[0])
        #     prob_percs_per_object[code_true][1].append(prob_mean[1])
        #     prob_percs_per_object[code_true][2].append(prob_mean[2])
        #     # total_codes_true.append(np.argmax(best_traj['codes'][0]))
        #     total_codes_true.append(code_true)
        #     total_codes_pred.append(prob_pred)
        #     total_rmse_true.append(code_true)
        #     total_rmse_pred.append(np.argmax(best_traj['codes'][0]))

        #     p += 1

        plt.figure()
        plt.title('Net losses')
        plt.xlabel('Trajectory No.')
        plt.ylabel('Loss')
        plt.plot(np.arange(len(total_disc_expert)), total_disc_expert)
        plt.plot(np.arange(len(total_disc_gen)), total_disc_gen)
        plt.plot(np.arange(len(total_eval_post)), total_eval_post)
        plt.legend(['disc expert ce', 'disc gen ce', 'post cross ent'], loc='upper left')
        plt.savefig('./plots/test_net_losses', dpi=100)
        plt.close()

models = Models(state_dims=3, action_dims=3, code_dims=3)
# models = Models(state_dims=1, action_dims=1, code_dims=3)

# main
def main():
    agent = Agent()
    infogail = InfoGAIL()
    # infogail.train(agent)
    infogail.test(agent)

if __name__ == '__main__':
    main()