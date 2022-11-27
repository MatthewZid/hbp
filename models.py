import os
from statistics import mean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.keras.layers import Input, Dense, LeakyReLU, Concatenate, ReLU, Add
from tensorflow.python.keras.models import Model, clone_model
import numpy as np
import matplotlib.pyplot as plt
from utils import *

tfd = tfp.distributions

class Generator():
    def __init__(self, state_dims, action_dims, code_dims, epsilon=0.2, max_kl=0.01):
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.code_dims = code_dims
        self.epsilon = epsilon
        self.max_kl = max_kl
        self.model = self.create_generator()
        self.old_model = self.create_generator()
        # self.old_model = clone_model(self.model)
    
    def create_generator(self):
        # initializer = tf.keras.initializers.GlorotNormal()
        states = Input(shape=self.state_dims)
        # default: x = Dense(260, kernel_initializer=initializer, activation='tanh')(states)
        x = Dense(128, activation='tanh')(states)
        # x = LeakyReLU()(x)
        codes = Input(shape=self.code_dims)
        c = Dense(32, activation='tanh')(codes)
        # c = LeakyReLU()(c)
        h = Concatenate(axis=1)([x,c])
        actions = Dense(self.action_dims)(h)

        model = Model(inputs=[states,codes], outputs=actions)
        return model
    
    @tf.function
    def __generator_loss(self, feed, model_type='current', ret='g'):
        # calculate ratio between old and new policy (surrogate loss)
        model = None
        if model_type == 'current': model = self.model
        else: model = self.old_model

        with tf.GradientTape() as grad_tape:
            actions_mu = model([feed['states'], feed['codes']], training=True)
            old_actions_mu = self.model([feed['states'], feed['codes']], training=True)

            log_p_n = gauss_log_prob(actions_mu, LOGSTD, feed['actions'])
            log_oldp_n = gauss_log_prob(old_actions_mu, LOGSTD, feed['actions'])
            # ...OR...
            # dist = tfd.MultivariateNormalDiag(loc=actions_mu, scale_diag=[tf.exp(LOGSTD) for _ in range(actions_mu.shape[1])])
            # dist_old = tfd.MultivariateNormalDiag(loc=old_actions_mu, scale_diag=[tf.exp(LOGSTD) for _ in range(old_actions_mu.shape[1])])
            # log_p_n = dist.log_prob(feed['actions'])
            # log_oldp_n = dist_old.log_prob(feed['actions'])

            ratio_n = log_p_n / log_oldp_n
            surrogate_loss = None
            if use_ppo:
                surrogate1 = ratio_n * feed['advants']
                surrogate2 = tf.clip_by_value(ratio_n, 1 - self.epsilon, 1 + self.epsilon) * feed['advants']
                surrogate_loss = tf.math.reduce_mean(tf.math.minimum(surrogate1, surrogate2))
            else: surrogate_loss = tf.math.reduce_mean(ratio_n * feed['advants'])
        
        if ret == 'g':
            var_list = self.model.trainable_weights
            grads = grad_tape.gradient(surrogate_loss, var_list, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            return tf.concat([tf.reshape(grad, [numel(v)]) for (v, grad) in zip(var_list, grads)], 0)
        else: return surrogate_loss
    
    def get_loss(self, theta, feed):
        # set_from_flat(self.generator, theta)
        var_list = self.old_model.trainable_weights
        shapes = [v.shape for v in var_list]
        start = 0

        weight_idx = 0
        for shape in shapes:
            size = np.prod(shape)
            self.old_model.trainable_weights[weight_idx].assign(tf.reshape(theta[start:(start + size)], shape))
            weight_idx += 1
            start += size
        return self.__generator_loss(feed, 'update', 'l')
    
    def plot_gradients(self, g):
        space = np.arange(1, g.numpy().size+1, dtype=int)
        plt.figure()
        plt.plot(space, g.numpy())
        plt.savefig('./plots/gradients', dpi=100)
        plt.close()
    
    def fisher_vector_product(self, p, feed, cg_damping=0.1):
        N = feed['states'].shape[0]
        Nf = tf.cast(N, tf.float32)
        var_list = self.model.trainable_weights

        with tf.GradientTape() as tape_gvp:
            tape_gvp.watch(var_list)
            with tf.GradientTape() as grad_tape:
                actions_mu = self.model([feed['states'], feed['codes']], training=True)
                kl_firstfixed = gauss_selfKL_firstfixed(actions_mu, LOGSTD)

            grads = flatgrad(self.model, kl_firstfixed, grad_tape)
            gvp = tf.reduce_sum(grads * p)

        fvp = flatgrad(self.model, gvp, tape_gvp)
        nans = tf.math.is_nan(fvp)
        if(tf.where(nans).numpy().flatten().shape[0] != 0): print('Fisher vector: NAN!!!!!!!!!!!')

        return fvp + p * cg_damping
    
    def train(self, feed):
        if use_ppo:
            gen_optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)

            # for _, (states_batch, actions_batch, oldactions_batch, codes_batch, advants_batch) in enumerate(dataset):
            for _ in range(3):
                (surrogate_loss, grad_tape) = self.__generator_loss(feed, use_ppo)
                gen_grads = grad_tape.gradient(surrogate_loss, self.model.trainable_weights)
                gen_optimizer.apply_gradients(zip(gen_grads, self.model.trainable_weights))
        
        else:
            # calculate previous theta (θold)
            thprev = get_flat(self.model)

            # policy gradient
            policy_gradient = self.__generator_loss(feed)
            # self.plot_gradients(policy_gradient)
            nans = tf.math.is_nan(policy_gradient)
            if(tf.where(nans).numpy().flatten().shape[0] != 0): print('NAN!!!!!!!!!!!')
            stepdir = conjugate_gradient(self.fisher_vector_product, feed, policy_gradient.numpy())
            shs = stepdir.dot(self.fisher_vector_product(stepdir, feed))
            assert shs > 0

            lm = np.sqrt(2 * self.max_kl / shs)
            fullstep = lm * stepdir
            neggdotstepdir = policy_gradient.numpy().dot(stepdir)

            # find new theta with linesearch and update the model
            theta = linesearch(self.get_loss, thprev, feed, fullstep, neggdotstepdir)
            surrogate_loss = self.__generator_loss(feed, 'update', 'l')
            # set_from_flat(self.generator, theta)
            var_list = self.model.trainable_weights
            shapes = [v.shape for v in var_list]
            start = 0

            weight_idx = 0
            for shape in shapes:
                size = np.prod(shape)
                self.model.trainable_weights[weight_idx].assign(tf.reshape(theta[start:(start + size)], shape))
                weight_idx += 1
                start += size
        
        episode_loss = tf.get_static_value(surrogate_loss)
        episode_loss = episode_loss.item()

        return episode_loss

class Discriminator():
    def __init__(self, state_dims, action_dims):
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.model = self.create_discriminator()
        self.disc_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    
    def create_discriminator(self):
        # initializer = tf.keras.initializers.HeNormal()
        states = Input(shape=self.state_dims)
        actions = Input(shape=self.action_dims)
        # merged = tf.concat([states,actions], 1)
        merged = Concatenate(axis=1)([states,actions])
        x = Dense(128, activation='tanh')(merged)
        # x = LeakyReLU()(x)
        # x = Dense(128, activation='tanh')(x)
        # x = LeakyReLU()(x)
        score = Dense(1)(x)

        model = Model(inputs=[states, actions], outputs=score)
        return model
    
    def __disc_loss(self, score1, score2):
        # cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        # genloss = cross_entropy(tf.ones_like(score1), score1)
        # expertloss = cross_entropy(tf.zeros_like(score2), score2)
        genloss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(score1), logits=score1)
        expertloss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(score2), logits=score2)
        loss = tf.reduce_mean(genloss) + tf.reduce_mean(expertloss)

        return loss
    
    def train(self, dataset):
        # loss = 0.0
        pure_losses = []
        # total_train_size = sum([el[0].shape[0] for el in list(dataset.as_numpy_iterator())])
        for _, (generated_states_batch, generated_actions_batch, expert_states_batch, expert_actions_batch) in enumerate(dataset):
            with tf.GradientTape() as disc_tape:
                score1 = self.model([generated_states_batch, generated_actions_batch], training=True)
                score2 = self.model([expert_states_batch, expert_actions_batch], training=True)

                # wasserstein loss: D(x) - D(G(z))
                # score1 = tf.reduce_mean(score1)
                # score2 = -score2
                # score2 = tf.reduce_mean(score2)
                # disc_loss = tf.math.add(score1, score2)

                # cross entropy loss: D(G(z)) + (1 - D(x))
                disc_loss = self.__disc_loss(score1, score2)
            
            if save_loss:
                # loss += tf.get_static_value(disc_loss) * generated_states_batch.shape[0]
                pure_losses.append(tf.get_static_value(disc_loss).item())
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.model.trainable_weights)
            self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.model.trainable_weights))

            # for l in self.discriminator.layers:
            #     weights = l.get_weights()
            #     weights = [np.clip(w, -0.01, 0.01) for w in weights]
            #     l.set_weights(weights)
        
        # episode_loss = loss / total_train_size
        # episode_loss = episode_loss.item()
        pure_losses = np.array(pure_losses, dtype=np.float32)

        return pure_losses.mean(), pure_losses.std()

class Posterior():
    def __init__(self, state_dims, action_dims, code_dims):
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.code_dims = code_dims
        self.model = self.create_posterior()
        self.target_model = self.create_posterior()
        self.posterior_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    
    def create_posterior(self):
        # initializer = tf.keras.initializers.HeNormal()
        states = Input(shape=self.state_dims)
        actions = Input(shape=self.action_dims)
        # merged = tf.concat([states,actions], 1)
        merged = Concatenate(axis=1)([states,actions])
        x = Dense(128, activation='tanh')(merged)
        # x = LeakyReLU()(x)
        # x = Dense(128, activation='tanh')(x)
        # x = LeakyReLU()(x)
        x = Dense(self.code_dims)(x)
        output = tf.keras.activations.softmax(x)

        model = Model(inputs=[states, actions], outputs=output)
        return model
    
    def __post_loss(self, prob, codes_batch):
        cross_entropy = tf.keras.losses.CategoricalCrossentropy()

        loss = cross_entropy(codes_batch, prob)
        loss = tf.reduce_mean(loss)

        return loss
    
    def train(self, dataset):
        # loss = 0.0
        pure_losses = []
        # total_train_size = sum([el[0].shape[0] for el in list(dataset.as_numpy_iterator())])
        for _, (states_batch, actions_batch, codes_batch) in enumerate(dataset):
            with tf.GradientTape() as post_tape:
                prob = self.model([states_batch, actions_batch], training=True)

                post_loss = self.__post_loss(prob, codes_batch)
            
            if save_loss:
                # loss += tf.get_static_value(post_loss) * states_batch.shape[0]
                pure_losses.append(tf.get_static_value(post_loss).item())
            gradients_of_posterior = post_tape.gradient(post_loss, self.model.trainable_weights)
            self.posterior_optimizer.apply_gradients(zip(gradients_of_posterior, self.model.trainable_weights))
        
        # episode_loss = loss / total_train_size
        # episode_loss = episode_loss.item()
        pure_losses = np.array(pure_losses, dtype=np.float32)

        # set posterior target weights
        posterior_weights = self.model.get_weights()
        posterior_target_weights = self.target_model.get_weights()
        for i in range(len(posterior_weights)):
            posterior_target_weights[i] = 0.5 * posterior_weights[i] + 0.5 * posterior_target_weights[i]
            self.target_model.set_weights(posterior_target_weights)

        return pure_losses.mean(), pure_losses.std()
    
    def validate(self, dataset):
        loss = 0.0
        total_val_size = sum([el[0].shape[0] for el in list(dataset.as_numpy_iterator())])
        for _, (states_batch, actions_batch, codes_batch) in enumerate(dataset):
            prob = self.target_model([states_batch, actions_batch], training=False)
            post_loss = self.__post_loss(prob, codes_batch)
            loss += tf.get_static_value(post_loss) * states_batch.shape[0]
        
        episode_loss = loss / total_val_size

        return episode_loss

class ValueNet():
    def __init__(self, state_dims, code_dims):
        self.state_dims = state_dims
        self.code_dims = code_dims
        self.model = self.create_valuenet()
        self.value_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    def create_valuenet(self):
        # initializer = tf.keras.initializers.HeNormal()
        states = Input(shape=self.state_dims)
        codes = Input(shape=self.code_dims)
        # merged = tf.concat([states,codes], 1)
        merged = Concatenate(axis=1)([states, codes])
        x = Dense(128, activation='tanh')(merged)
        # x = LeakyReLU()(x)
        # x = Dense(128, activation='tanh')(x)
        # x = LeakyReLU()(x)
        output = Dense(1)(x)
        # output = tf.keras.activations.sigmoid(output)

        model = Model(inputs=[states, codes], outputs=output)
        return model
    
    def __value_loss(self, pred, returns):
        mse = tf.keras.losses.MeanSquaredError()

        loss = mse(returns, pred)

        return loss
    
    def train(self, dataset):
        loss = 0.0
        total_train_size = sum([el[0].shape[0] for el in list(dataset.as_numpy_iterator())])
        for _, (states_batch, codes_batch, returns_batch) in enumerate(dataset):
            with tf.GradientTape() as value_tape:
                value_pred = self.model([states_batch, codes_batch], training=True)

                value_loss = self.__value_loss(value_pred, returns_batch)
            
            if save_loss: loss += tf.get_static_value(value_loss) * states_batch.shape[0]
            value_grads = value_tape.gradient(value_loss, self.model.trainable_weights)
            self.value_optimizer.apply_gradients(zip(value_grads, self.model.trainable_weights))
        
        episode_loss = loss / total_train_size
        episode_loss = episode_loss.item()

        return episode_loss

class Models():
    def __init__(self, state_dims=10, action_dims=2, code_dims=3):
        self.generator = Generator(state_dims, action_dims, code_dims)
        self.discriminator = Discriminator(state_dims, action_dims)
        self.posterior = Posterior(state_dims, action_dims, code_dims)
        self.value_net = ValueNet(state_dims, code_dims)