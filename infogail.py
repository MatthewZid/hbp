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
    def __generate_trajectory(self, code, start):
        s_traj = []
        a_traj = []
        c_traj = []
        env = Env()

        # generate actions for every current state
        state_obsrv = env.reset(code, start) # reset environment state
        code_tf = tf.constant(code)
        code_tf = tf.expand_dims(code_tf, axis=0)

        while True:
            # 1. generate actions with generator
            state_tf = tf.constant(state_obsrv)
            state_tf = tf.expand_dims(state_tf, axis=0)
            action_mu = models.generator.model([state_tf, code_tf], training=False)
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
    
    def run(self, code, start):
        try:
            trajectory_dict = {}
            trajectory = self.__generate_trajectory(code, start)
            trajectory_dict['states'] = np.copy(trajectory[0])
            trajectory_dict['actions'] = np.copy(trajectory[1])
            trajectory_dict['codes'] = np.copy(trajectory[2])
            return trajectory_dict
        except KeyboardInterrupt:
            time.sleep(1)

class InfoGAIL():
    def __init__(self, batch_size=2000, code_batch=500, episodes=10000, gamma=0.997, lam=0.97):
        self.batch = batch_size
        self.code_batch = code_batch
        self.episodes = episodes
        self.gamma = gamma
        self.lam = lam
        self.starting_episode = 0
        self.gen_result = []
        self.disc_result = []
        self.post_result = []
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
        plt.legend(['disc', 'post'], loc="lower left")
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
        feat_width = None
        generator_weight_path = ''

        if resume_training:
            with open("./saved_models/trpo/model.yml", 'r') as f:
                data = yaml.safe_load(f)
                self.starting_episode = data['episode']
                print('\nRestarting from episode {:d}'.format(self.starting_episode))
                self.gen_result = data['gen_loss']
                self.disc_result = data['disc_loss']
                self.post_result = data['post_loss']
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
                    'codes': np.array(data['train_codes'], dtype=np.float64)
                }
                feature_size['train'] = np.array(data['train_feat_size'], dtype=int)
                feat_width = data['feat_width']
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
                    'codes': np.array(data['train_codes'], dtype=np.float64)
                }
                feature_size = np.array(data['train_feat_size'], dtype=int)
                feat_width = data['feat_width']
        
        models.generator.model.load_weights(generator_weight_path)
        train_start_pos, train_start_codes = extract_start_pos(features, feature_size, feat_width)
        print('\nTraining setup ready!')

        for episode in trange(self.starting_episode, self.episodes, desc="Episode"):
            # Sample a batch of latent codes: ci ∼ p(c)
            pick = np.random.choice(train_start_codes.shape[0], self.code_batch)
            sampled_codes = train_start_codes[pick]
            sampled_pos = train_start_pos[pick]
            starting_pos_code_pairs = list(zip(sampled_codes, sampled_pos))

            # Sample trajectories: τi ∼ πθi(ci), with the latent code fixed during each rollout
            trajectories = []
            with mp.Pool(mp.cpu_count()) as pool:
                trajectories = pool.starmap(agent.run, starting_pos_code_pairs)
            
            # Sample from buffer
            # for traj in trajectories:
            #     self.buffer.add(traj)
            # trajectories = self.buffer.sample()
            
            for traj in trajectories:
                traj['old_action_mus'] = models.generator.model([traj['states'], traj['codes']], training=False)
            
            generated_states = np.concatenate([traj['states'] for traj in trajectories], dtype=np.float64)
            generated_actions = np.concatenate([traj['actions'] for traj in trajectories], dtype=np.float64)
            generated_codes = np.concatenate([traj['codes'] for traj in trajectories], dtype=np.float64)
            generated_oldactions = np.concatenate([traj['old_action_mus'] for traj in trajectories], dtype=np.float64)

            # train discriminator
            # Sample state-action pairs χi ~ τi and χΕ ~ τΕ with the same batch size
            if features['states'].shape[0] < generated_states.shape[0]:
                expert_idx = np.arange(features['states'].shape[0])
                np.random.shuffle(expert_idx)
                shuffled_expert_states = features['states'][expert_idx, :]
                shuffled_expert_actions = features['actions'][expert_idx, :]

                generated_idx = np.random.choice(generated_states.shape[0], features['states'].shape[0], replace=False)
                shuffled_generated_states = generated_states[generated_idx, :]
                shuffled_generated_actions = generated_actions[generated_idx, :]
            elif features['states'].shape[0] > generated_states.shape[0]:
                generated_idx = np.arange(generated_states.shape[0])
                np.random.shuffle(generated_idx)
                shuffled_generated_states = generated_states[generated_idx, :]
                shuffled_generated_actions = generated_actions[generated_idx, :]

                expert_idx = np.random.choice(features['states'].shape[0], generated_states.shape[0], replace=False)
                shuffled_expert_states = features['states'][expert_idx, :]
                shuffled_expert_actions = features['actions'][expert_idx, :]
            else:
                expert_idx = np.arange(features['states'].shape[0])
                np.random.shuffle(expert_idx)
                shuffled_expert_states = features['states'][expert_idx, :]
                shuffled_expert_actions = features['actions'][expert_idx, :]

                generated_idx = np.arange(generated_states.shape[0])
                np.random.shuffle(generated_idx)
                shuffled_generated_states = generated_states[generated_idx, :]
                shuffled_generated_actions = generated_actions[generated_idx, :]

            shuffled_generated_states = tf.convert_to_tensor(shuffled_generated_states, dtype=tf.float64)
            shuffled_generated_actions = tf.convert_to_tensor(shuffled_generated_actions, dtype=tf.float64)
            shuffled_expert_states = tf.convert_to_tensor(shuffled_expert_states, dtype=tf.float64)
            shuffled_expert_actions = tf.convert_to_tensor(shuffled_expert_actions, dtype=tf.float64)

            dataset = tf.data.Dataset.from_tensor_slices((shuffled_generated_states, shuffled_generated_actions, shuffled_expert_states, shuffled_expert_actions))
            dataset = dataset.batch(batch_size=self.batch)

            loss = models.discriminator.train(dataset)
            if save_loss: self.disc_result.append(loss)

            # train posterior
            if features['states'].shape[0] < generated_states.shape[0]:
                generated_idx = np.random.choice(generated_states.shape[0], features['states'].shape[0], replace=False)
            else:
                generated_idx = np.arange(generated_states.shape[0])
                np.random.shuffle(generated_idx)

            shuffled_generated_states = generated_states[generated_idx, :]
            shuffled_generated_actions = generated_actions[generated_idx, :]
            shuffled_generated_codes = generated_codes[generated_idx, :]
            shuffled_generated_states = tf.convert_to_tensor(shuffled_generated_states, dtype=tf.float64)
            shuffled_generated_actions = tf.convert_to_tensor(shuffled_generated_actions, dtype=tf.float64)
            shuffled_generated_codes = tf.convert_to_tensor(shuffled_generated_codes, dtype=tf.float64)
            dataset = tf.data.Dataset.from_tensor_slices((shuffled_generated_states, shuffled_generated_actions, shuffled_generated_codes))
            dataset = dataset.batch(batch_size=self.batch)

            loss = models.posterior.train(dataset)
            if save_loss: self.post_result.append(loss)

            # TRPO/PPO
            # calculate rewards from discriminator and posterior
            episode_rewards = []
            for traj in trajectories:
                reward_d = (-tf.math.log(tf.keras.activations.sigmoid(models.discriminator.model([traj['states'], traj['actions']], training=False)))).numpy()
                reward_p = models.posterior.model([traj['states'], traj['actions']], training=False).numpy()
                reward_p = np.sum(np.ma.log(reward_p).filled(0) * traj['codes'], axis=1).flatten() # np.ma.log over tf.math.log, fixes log of zero

                traj['rewards'] = 0.6 * reward_d.flatten() + 0.4 * reward_p
                episode_rewards.append(traj['rewards'].sum())

                # calculate values, advants and returns
                values = models.value_net.model([traj['states'], traj['codes']], training=False).numpy().flatten() # Value function
                values_next = shift(values, -1, cval=0)
                deltas = traj['rewards'] + self.gamma * values_next - values # Advantage(st,at) = rt+1 + γ*V(st+1) - V(st)
                traj['advants'] = discount(deltas, self.gamma * self.lam)
                traj['returns'] = discount(traj['rewards'], self.gamma)
            
            advants = np.concatenate([traj['advants'] for traj in trajectories], dtype=np.float64)
            # advants /= advants.std()

            # train value net for next iter
            returns = np.expand_dims(np.concatenate([traj['returns'] for traj in trajectories]), axis=1)

            if features['states'].shape[0] < generated_states.shape[0]:
                generated_idx = np.random.choice(generated_states.shape[0], features['states'].shape[0], replace=False)
            else:
                generated_idx = np.arange(generated_states.shape[0])
                np.random.shuffle(generated_idx)
                
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
                'advants': advants,
                'old_mus': generated_oldactions
            } 

            loss = models.generator.train(feed)
            if save_loss: self.gen_result.append(loss)

            if save_loss:
                # plot rewards and losses
                episode_reward = np.array(episode_rewards, dtype=np.float64).mean()
                self.total_rewards.append(episode_reward)
                if episode != 0:
                    self.show_loss()
                    epoch_space = np.arange(1, len(self.total_rewards)+1, dtype=int)
                    self.__saveplot(epoch_space, self.total_rewards, 0, 'rewards')
            
            if episode != 0 and (episode % 100 == 0): print('Theta updates so far: {:d}'.format(utils.improved))

            if save_models:
                models.generator.model.save_weights('./saved_models/trpo/generator.h5')
                models.discriminator.model.save_weights('./saved_models/trpo/discriminator.h5')
                models.posterior.model.save_weights('./saved_models/trpo/posterior.h5')
                models.value_net.model.save_weights('./saved_models/trpo/value_net.h5')
                yaml_conf = {
                    'episode': episode+1,
                    'gen_loss': self.gen_result,
                    'disc_loss': self.disc_result,
                    'post_loss': self.post_result,
                    'value_loss': self.value_result
                }
                
                with open("./saved_models/trpo/model.yml", 'w') as f:
                    yaml.dump(yaml_conf, f, sort_keys=False, default_flow_style=False)

    def test(self, agent):
        features = {}
        feature_size = None

        models.generator.model.load_weights('./saved_models/trpo/generator.h5')
        models.discriminator.model.load_weights('./saved_models/trpo/discriminator.h5')
        models.posterior.model.load_weights('./saved_models/trpo/posterior.h5')

        with open("./saved_models/trpo/dataset.yml", 'r') as f:
            data = yaml.safe_load(f)
            features = {
                'states': np.array(data['test_states'], dtype=np.float64),
                'actions': np.array(data['test_actions'], dtype=np.float64),
                'codes': np.array(data['test_codes'], dtype=np.float64),
                'norm_time': np.array(data['test_norm_time'], dtype=np.float64)
            }
            feature_size = np.array(data['test_feat_size'], dtype=int)
            feat_width = data['feat_width']
        
        start_pos, start_codes = extract_start_pos(features, feature_size, feat_width)
        print('\nTest setup ready!')

        # trajectory generation
        starting_pos_code_pairs = list(zip(start_codes, start_pos))
        trajectories = []
        with mp.Pool(mp.cpu_count()) as pool:
            trajectories = pool.starmap(agent.run, starting_pos_code_pairs)
        
        total_disc_expert = []
        total_disc_gen = []
        total_eval_post = []
        total_codes_true = []
        total_codes_pred = []
        prob_percs_per_object = []
        intervals = [20.0, 40.0, 60.0, 80.0, 100.0]
        inverse_intervals = [20.0, 40.0, 60.0, 80.0]
        part_prob_percs_per_object = []
        end_part_prob_percs_per_object = []
        part_total_codes_true = []
        part_total_codes_pred = []
        end_part_total_codes_true = []
        end_part_total_codes_pred = []
        pos = 0
        expert_i = 0

        obj_size = {'Small': 0, 'Medium': 1, 'Large': 2}

        for i in range(features['codes'].shape[1]):
            prob_percs_per_object.append([])
            for _ in range(features['codes'].shape[1]): prob_percs_per_object[i].append([])
        
        for i in range(len(intervals)):
            part_prob_percs_per_object.append([])
            end_part_prob_percs_per_object.append([])
            part_total_codes_true.append([])
            part_total_codes_pred.append([])
            end_part_total_codes_true.append([])
            end_part_total_codes_pred.append([])
            for j in range(features['codes'].shape[1]):
                part_prob_percs_per_object[i].append([])
                end_part_prob_percs_per_object[i].append([])
                for _ in range(features['codes'].shape[1]):
                    part_prob_percs_per_object[i][j].append([])
                    end_part_prob_percs_per_object[i][j].append([])
        
        obj_trajectories = {
            '0': [],
            '1': [],
            '2': []
        }
        
        for traj in trajectories:
            expert_states = features['states'][pos:pos+feature_size[expert_i]]
            expert_actions = features['actions'][pos:pos+feature_size[expert_i]]
            expert_norm_time = features['norm_time'][pos:pos+feature_size[expert_i]]

            # discriminate
            if expert_states.shape[0] < traj['states'].shape[0]:
                expert_idx = np.arange(expert_states.shape[0])
                np.random.shuffle(expert_idx)
                shuffled_expert_states = expert_states[expert_idx, :]
                shuffled_expert_actions = expert_actions[expert_idx, :]

                generated_idx = np.random.choice(traj['states'].shape[0], expert_states.shape[0], replace=False)
                shuffled_generated_states = traj['states'][generated_idx, :]
                shuffled_generated_actions = traj['actions'][generated_idx, :]
            elif expert_states.shape[0] > traj['states'].shape[0]:
                generated_idx = np.arange(traj['states'].shape[0])
                np.random.shuffle(generated_idx)
                shuffled_generated_states = traj['states'][generated_idx, :]
                shuffled_generated_actions = traj['actions'][generated_idx, :]

                expert_idx = np.random.choice(expert_states.shape[0], traj['states'].shape[0], replace=False)
                shuffled_expert_states = expert_states[expert_idx, :]
                shuffled_expert_actions = expert_actions[expert_idx, :]
            else:
                expert_idx = np.arange(expert_states.shape[0])
                np.random.shuffle(expert_idx)
                shuffled_expert_states = expert_states[expert_idx, :]
                shuffled_expert_actions = expert_actions[expert_idx, :]

                generated_idx = np.arange(traj['states'].shape[0])
                np.random.shuffle(generated_idx)
                shuffled_generated_states = traj['states'][generated_idx, :]
                shuffled_generated_actions = traj['actions'][generated_idx, :]

            score1 = models.discriminator.model([shuffled_generated_states, shuffled_generated_actions], training=False)
            score2 = models.discriminator.model([shuffled_expert_states, shuffled_expert_actions], training=False)
            binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            score1 = tf.squeeze(score1).numpy()
            score2 = tf.squeeze(score2).numpy()
            y_ones = np.ones_like(score1)
            y_zeros = np.zeros_like(score2)
            total_disc_expert.append(binary_cross_entropy(y_zeros, score2))
            total_disc_gen.append(binary_cross_entropy(y_ones, score1))

            # posterior
            if expert_states.shape[0] < traj['states'].shape[0]:
                shuffled_generated_states = traj['states'][:expert_states.shape[0]]
                shuffled_generated_actions = traj['actions'][:expert_states.shape[0]]
                shuffled_generated_codes = traj['codes'][:expert_states.shape[0]]
            elif expert_states.shape[0] > traj['states'].shape[0]:
                print('Generated trajectory smaller than expert!')
                pos += feature_size[expert_i]
                expert_i += 1
                continue
            else:
                shuffled_generated_states = traj['states']
                shuffled_generated_actions = traj['actions']
                shuffled_generated_codes = traj['codes']

            # percentages for the whole trajectory
            prob = models.posterior.model([shuffled_generated_states, shuffled_generated_actions], training=False)
            cross_entropy = tf.keras.losses.CategoricalCrossentropy()
            loss = cross_entropy(shuffled_generated_codes, prob)
            loss = tf.reduce_mean(loss).numpy()
            total_eval_post.append(loss)
            prob_mean = tf.reduce_mean(prob, axis=0)
            prob_pred = tf.math.argmax(prob_mean).numpy()
            code_true = tf.math.argmax(shuffled_generated_codes, axis=1).numpy()[0]
            prob_percs_per_object[code_true][0].append(prob_mean.numpy()[0])
            prob_percs_per_object[code_true][1].append(prob_mean.numpy()[1])
            prob_percs_per_object[code_true][2].append(prob_mean.numpy()[2])
            total_codes_true.append(code_true)
            total_codes_pred.append(prob_pred)

            # group trajectories
            obj_trajectories[str(code_true)].append((shuffled_generated_states, shuffled_generated_actions))

            # franken-trajectories
            random_code_idx = np.random.choice(features['codes'].shape[1], 1)[0]
            random_upper_half_idx = np.random.choice(len(obj_trajectories[str(random_code_idx)]), 1)[0]

            # intervals
            count = 0
            for ci in intervals:
                partial_expert_idx = np.where(expert_norm_time <= ci)[0]
                partial_gen_states = shuffled_generated_states[partial_expert_idx, :]
                partial_gen_actions = shuffled_generated_actions[partial_expert_idx, :]
                part_prob = models.posterior.model([partial_gen_states, partial_gen_actions], training=False)
                part_prob_mean = tf.reduce_mean(part_prob, axis=0)
                part_prob_pred = tf.math.argmax(part_prob_mean).numpy()
                part_code_true = tf.math.argmax(shuffled_generated_codes, axis=1).numpy()[0]
                part_prob_percs_per_object[count][part_code_true][0].append(deepcopy(part_prob_mean.numpy()[0]))
                part_prob_percs_per_object[count][part_code_true][1].append(deepcopy(part_prob_mean.numpy()[1]))
                part_prob_percs_per_object[count][part_code_true][2].append(deepcopy(part_prob_mean.numpy()[2]))
                part_total_codes_true[count].append(deepcopy(part_code_true))
                part_total_codes_pred[count].append(deepcopy(part_prob_pred))
                count += 1
            
            # inverse intervals
            count = 0
            for ci in inverse_intervals:
                partial_expert_idx = np.where(expert_norm_time >= ci)[0]
                partial_gen_states = shuffled_generated_states[partial_expert_idx, :]
                partial_gen_actions = shuffled_generated_actions[partial_expert_idx, :]
                part_prob = models.posterior.model([partial_gen_states, partial_gen_actions], training=False)
                part_prob_mean = tf.reduce_mean(part_prob, axis=0)
                part_prob_pred = tf.math.argmax(part_prob_mean).numpy()
                part_code_true = tf.math.argmax(shuffled_generated_codes, axis=1).numpy()[0]
                end_part_prob_percs_per_object[count][part_code_true][0].append(deepcopy(part_prob_mean.numpy()[0]))
                end_part_prob_percs_per_object[count][part_code_true][1].append(deepcopy(part_prob_mean.numpy()[1]))
                end_part_prob_percs_per_object[count][part_code_true][2].append(deepcopy(part_prob_mean.numpy()[2]))
                end_part_total_codes_true[count].append(deepcopy(part_code_true))
                end_part_total_codes_pred[count].append(deepcopy(part_prob_pred))
                count += 1
            
            pos += feature_size[expert_i]
            expert_i += 1

        print(classification_report(total_codes_true, total_codes_pred))
        cm = confusion_matrix(total_codes_true, total_codes_pred)
        group_counts = ['{}'.format(v) for v in cm.flatten()]
        group_perc = ['{:.2%}'.format(v) for v in cm.flatten()/np.sum(cm)]
        annot_labels = [f'{v1}\n({v2})' for v1,v2 in zip(group_perc, group_counts)]
        annot_labels = np.asarray(annot_labels).reshape(cm.shape)
        hm = sn.heatmap(cm, annot=annot_labels, fmt='', cmap='Blues')
        hm.set_title('Mode confusion matrix')
        hm.set_xlabel('Predicted modes')
        hm.set_ylabel('Actual modes')
        hm.set_xticklabels(['Small', 'Medium', 'Large'])
        hm.set_yticklabels(['Small', 'Medium', 'Large'])
        plt.savefig('./plots/cf_matrix', dpi=100)
        plt.close()

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
        
        count = 0
        for ci in inverse_intervals:
            cm = confusion_matrix(end_part_total_codes_true[count], end_part_total_codes_pred[count])
            group_counts = ['{}'.format(v) for v in cm.flatten()]
            group_perc = ['{:.2%}'.format(v) for v in cm.flatten()/np.sum(cm)]
            annot_labels = [f'{v1}\n({v2})' for v1,v2 in zip(group_perc, group_counts)]
            annot_labels = np.asarray(annot_labels).reshape(cm.shape)
            hm = sn.heatmap(cm, annot=annot_labels, fmt='', cmap='Blues')
            hm.set_title('Mode confusion matrix: From '+str(int(ci))+'% of movement until end')
            hm.set_xlabel('Predicted modes')
            hm.set_ylabel('Actual modes')
            hm.set_xticklabels(['Small', 'Medium', 'Large'])
            hm.set_yticklabels(['Small', 'Medium', 'Large'])
            plt.savefig('./plots/cf_matrix_inverse_'+str(int(ci)), dpi=100)
            plt.close()

            for sz in ['Small', 'Medium', 'Large']:
                plt.figure()
                plt.title(sz+': From '+str(int(ci))+'% completion until end')
                plt.xlabel('Trajectory No.')
                plt.ylabel('Probability (%)')
                plt.scatter(np.arange(len(end_part_prob_percs_per_object[count][obj_size[sz]][0])), end_part_prob_percs_per_object[count][obj_size[sz]][0], alpha=0.6)
                plt.scatter(np.arange(len(end_part_prob_percs_per_object[count][obj_size[sz]][1])), end_part_prob_percs_per_object[count][obj_size[sz]][1], alpha=0.6)
                plt.scatter(np.arange(len(end_part_prob_percs_per_object[count][obj_size[sz]][2])), end_part_prob_percs_per_object[count][obj_size[sz]][2], alpha=0.6)
                plt.legend(['Small', 'Medium', 'Large'], loc='lower right')
                plt.savefig('./plots/inverse_conf_percs_'+sz+'_'+str(int(ci)), dpi=100)
                plt.close()

            count += 1

        plt.figure()
        plt.title('Net losses')
        plt.xlabel('Trajectory No.')
        plt.ylabel('Loss')
        plt.plot(np.arange(len(total_disc_expert)), total_disc_expert)
        plt.plot(np.arange(len(total_disc_expert)), total_disc_gen)
        plt.plot(np.arange(len(total_disc_expert)), total_eval_post)
        plt.legend(['disc expert ce', 'disc gen ce', 'post cross ent'], loc='upper left')
        plt.savefig('./plots/test_net_losses', dpi=100)
        plt.close()

        for sz in ['Small', 'Medium', 'Large']:
            plt.figure()
            plt.title(sz)
            plt.xlabel('Trajectory No.')
            plt.ylabel('Probability (%)')
            plt.scatter(np.arange(len(prob_percs_per_object[obj_size[sz]][0])), prob_percs_per_object[obj_size[sz]][0], alpha=0.6)
            plt.scatter(np.arange(len(prob_percs_per_object[obj_size[sz]][1])), prob_percs_per_object[obj_size[sz]][1], alpha=0.6)
            plt.scatter(np.arange(len(prob_percs_per_object[obj_size[sz]][2])), prob_percs_per_object[obj_size[sz]][2], alpha=0.6)
            plt.legend(['Small', 'Medium', 'Large'], loc='lower right')
            plt.savefig('./plots/conf_percs_'+sz, dpi=100)
            plt.close()

models = Models(state_dims=15, action_dims=3, code_dims=3)

# main
def main():
    agent = Agent()
    infogail = InfoGAIL()
    # infogail.train(agent)
    infogail.test(agent)

if __name__ == '__main__':
    main()