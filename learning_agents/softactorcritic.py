import copy

from keras import layers
from tensorflow import keras
import tensorflow as tf

from genericfunctions import max_key
from learning_agents.learningagent import LearningAgent
import numpy as np
import random


class SoftActorCritic(LearningAgent):

    def __init__(self, actions, state_shape, lr=3e-4, gamma=0.99, buffer_size=10e6, model_layers=[256, 256],
                 batch_size=16,
                 tau=0.005,
                 discrete_actions=True):
        self.actions = actions
        self.num_actions = len(self.actions)
        self.num_steps_til_update = 0
        self.state_shape = state_shape
        self.lr = lr
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.model_layers = model_layers
        self.num_model_layers = len(self.model_layers)
        self.batch_size = batch_size
        self.tau = tau

        self.experience = []
        self.current_experience_size = 0

        self.discrete_actions = discrete_actions

        self.value_net = None
        self.value_net_copy = None
        self.q_net_1 = None
        self.q_net_2 = None
        self.mu_network = None
        self.sigma_network = None
        self.build_networks()

        self.noise = 1e-3
        return

    def build_networks(self):
        def add_layers(input_layer_size, final_layer_size, final_layer_activation):
            network = keras.Sequential()
            network.add(layers.Dense(units=self.model_layers[0],
                                     activation='relu',
                                     input_dim=input_layer_size))
            for i in range(1, self.num_model_layers):
                network.add(layers.Dense(units=self.model_layers[i],
                                         activation='relu'))

            network.add(layers.Dense(units=final_layer_size, activation=final_layer_activation))
            network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                            loss='mse', metrics=None)
            return network

        self.value_net = add_layers(self.state_shape, 1, 'linear')
        self.value_net_copy = add_layers(self.state_shape, 1, 'linear')

        self.q_net_1 = add_layers(self.state_shape + self.num_actions, 1, 'linear')
        self.q_net_2 = add_layers(self.state_shape + self.num_actions, 1, 'linear')

        self.mu_network = add_layers(self.state_shape, 1, 'linear')
        self.sigma_network = add_layers(self.state_shape, 1, 'relu')
        return

    def choose_action(self, state, optimal_choice=False,
                      possible_actions=None):
        if possible_actions is None:
            possible_actions = self.actions

        mu = self.mu_network(state.reshape(1, self.state_shape))
        sigma = self.sigma_network(state.reshape(1, self.state_shape))

        action = None
        while action not in possible_actions:
            action_sample = np.random.normal(mu, sigma)
            action = (1 / (1 + np.exp(-action_sample))) * (self.num_actions - 1)

            if self.discrete_actions:
                action = int(action)
        return action

    def clear_experience(self):
        self.experience = []
        self.current_experience_size = 0
        return

    def copy_agent(self, agent_to_copy):
        self = copy.copy(agent_to_copy)
        return

    def learn(self, state, action, reward, next_state, terminal=None,
              next_state_possible_actions=None, no_learning=False):
        self.save_transition(state, action, reward, next_state, terminal)

        if (self.current_experience_size < self.batch_size * 32) or no_learning:
            return

        experience_sample = random.sample(self.experience, self.batch_size * 32)
        states = np.array([trajectory['state'] for trajectory in experience_sample])
        next_states = np.array([trajectory['next_state'] for trajectory in experience_sample])
        states_actions = []
        for trajectory in experience_sample:
            state_action = np.zeros(self.state_shape + self.num_actions)
            state_action[0:self.state_shape] = trajectory['state']
            state_action[self.state_shape + trajectory['action']] = 1
            states_actions.append(state_action)
        states_actions = np.array(states_actions)
        rewards = np.array([trajectory['reward'] for trajectory in experience_sample])

        value_copy_next_state_prediction = self.value_net_copy(np.reshape(next_states, (self.batch_size * 32,
                                                                                        self.state_shape))).numpy()
        q_net_1_prediction = self.q_net_1(states_actions.reshape(self.batch_size * 32,
                                                                 self.state_shape + self.num_actions)).numpy()
        q_net_2_prediction = self.q_net_2(states_actions.reshape(self.batch_size * 32,
                                                                 self.state_shape + self.num_actions)).numpy()
        min_q_net_prediction = np.minimum(q_net_1_prediction, q_net_2_prediction)
        mu = self.mu_network(states.reshape(self.batch_size * 32, self.state_shape)).numpy()
        sigma = self.sigma_network(states.reshape(self.batch_size * 32, self.state_shape)).numpy()
        action_sample = np.random.normal(mu, sigma)
        log_policy = np.random.lognormal(mu, sigma) - np.log(self.noise + 1 - (1 / (1 + np.exp(-action_sample)) ** 2))

        value_target = min_q_net_prediction - log_policy
        q_net_1_target = rewards + (self.gamma * value_copy_next_state_prediction)
        q_net_2_target = rewards + (self.gamma * value_copy_next_state_prediction)
        policy_target = min_q_net_prediction

        print('Value Net Fitting')
        self.value_net.fit(states.reshape(self.batch_size * 32, self.state_shape),
                           value_target, batch_size=self.batch_size, steps_per_epoch=32)
        print('Q_net 1 Fitting')
        self.q_net_1.fit(states_actions.reshape(self.batch_size * 32, self.state_shape + self.num_actions),
                         q_net_1_target, batch_size=self.batch_size, steps_per_epoch=32)
        print('Q_net 2 Fitting')
        self.q_net_2.fit(states_actions.reshape(self.batch_size * 32, self.state_shape + self.num_actions),
                         q_net_2_target, batch_size=self.batch_size, steps_per_epoch=32)
        print('Policy Net Fitting')
        self.mu_network.fit(states.reshape(self.batch_size * 32, self.state_shape),
                            policy_target, batch_size=self.batch_size, steps_per_epoch=32)
        self.sigma_network.fit(states.reshape(self.batch_size * 32, self.state_shape),
                               policy_target, batch_size=self.batch_size, steps_per_epoch=32)

        weights = []
        copy_weights = self.value_net_copy.weights
        for i, weight in enumerate(self.value_net.weights):
            weights.append((self.tau * weight) + ((1 - self.tau) * copy_weights[i]))
        self.value_net_copy.set_weights(weights)
        return

    def save_transition(self, state, action, reward, next_state, terminal):
        if len(self.experience) > self.buffer_size:
            to_remove = random.randint(0, self.current_experience_size - 1)
            del (self.experience[to_remove])
            self.current_experience_size -= 1

        to_save = {'state': state,
                   'action': action,
                   'reward': reward,
                   'next_state': next_state,
                   'terminal': terminal}
        self.experience.append(to_save)
        self.current_experience_size += 1
        return
