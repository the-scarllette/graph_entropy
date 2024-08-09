from environments.environment import Environment
from learning_agents.metacontrolleragent import MetaControllerAgent
from learning_agents.softactorcritic import SoftActorCritic

from keras import layers
import scipy
from tensorflow import keras
import tensorflow as tf
import numpy as np
import random


class DADSAgent(MetaControllerAgent):

    def __init__(self, actions, state_shape=None,
                 num_skills=None, skill_legnth=10, model_layers=[128, 128],
                 lr=3e-4, policy_tau=0.005,
                 state_predictor_batch_size=32):
        self.actions = actions
        self.num_actions = len(self.actions)
        self.state_shape = state_shape
        self.model_layers = model_layers
        self.num_model_layers = len(self.model_layers)
        self.current_skill = None
        self.num_skills = num_skills
        self.state_skill_shape = self.num_skills + self.state_shape
        self.action_skill_shape = self.num_skills + self.num_actions
        self.state_predictor_batch_size = state_predictor_batch_size
        self.skill_length = skill_legnth
        self.current_skill_length = 0

        # Creating Policy Network: pi(a | s, z)
        self.policy = SoftActorCritic(self.actions, self.state_skill_shape, lr,
                                      batch_size=self.state_predictor_batch_size,
                                      model_layers=self.model_layers, tau=policy_tau)

        # Creating the meta-controller:
        self.meta_controller = SoftActorCritic([i for i in range(self.num_skills)], self.state_shape,
                                               batch_size=self.state_predictor_batch_size,
                                               model_layers=self.model_layers, tau=policy_tau)

        # Creating state predictor network: q(s' | s, z)
        self.state_predictor = keras.Sequential()
        self.state_predictor.add(layers.Dense(units=self.model_layers[0],
                                              activation='relu',
                                              input_dim=self.state_skill_shape))
        for i in range(1, self.num_model_layers):
            self.state_predictor.add(layers.Dense(units=self.model_layers[i],
                                                  activation='relu'))
        self.state_predictor.add(layers.Dense(units=self.state_shape, activation='linear'))
        self.state_predictor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                                     loss='mse', metrics=None)
        return

    def get_policy_reward(self, state, next_state, skill):
        state_skill_probabilities = np.zeros(self.num_skills)
        for z in range(self.num_skills):
            skill_state = self.create_skill_state_vector(z, state)
            next_state_mean = self.state_predictor(skill_state.reshape(1, self.state_skill_shape))
            needed_next_state = next_state - state
            next_state_mean = next_state_mean.numpy()[0]
            q_prob = scipy.stats.multivariate_normal.pdf(needed_next_state, next_state_mean)
            state_skill_probabilities[z] = q_prob

        reward = (state_skill_probabilities[skill] / state_skill_probabilities.sum()) + np.log(self.num_skills)
        return reward

    def learn_skills(self, environment: Environment, num_cycles, num_steps,
                     all_actions_valid=False):
        if all_actions_valid:
            possible_actions = environment.possible_actions
        state_skill_buffer = []
        num_state_skills = 0
        minimum_steps = self.state_predictor_batch_size * 32
        steps_to_run = num_steps
        if num_steps < minimum_steps:
            steps_to_run = minimum_steps

        for _ in range(num_cycles):
            terminal = True

            if num_state_skills >= minimum_steps:
                steps_to_run = num_steps

            for _ in range(steps_to_run):
                if terminal:
                    state = environment.reset()
                    self.sample_skill()
                    if not all_actions_valid:
                        possible_actions = environment.get_possible_actions()

                skill_state_vector = self.create_skill_state_vector(self.current_skill, state)
                action = self.policy.choose_action(skill_state_vector, False, possible_actions)
                next_state, _, terminal, _ = environment.step(action)

                state_skill = self.create_skill_state_vector(self.current_skill, state)
                next_state_skill = self.create_skill_state_vector(self.current_skill, next_state)
                state_skill_buffer.append({'state': state,
                                           'skill': self.current_skill,
                                           'state_skill': state_skill,
                                           'action': action,
                                           'next_state': next_state,
                                           'next_state_skill': next_state_skill,
                                           'terminal': terminal})
                reward = self.get_policy_reward(state, next_state, self.current_skill)

                self.policy.save_transition(state_skill, action, reward, next_state_skill, terminal)
                num_state_skills += 1

                state = next_state
                if not all_actions_valid and not terminal:
                    possible_actions = environment.get_possible_actions()

            # Training State Predictor Network
            samples = random.sample(state_skill_buffer, minimum_steps)
            state_skills = np.array([trajectory['state_skill'] for trajectory in samples])
            state_differences = np.array([trajectory['next_state'] - trajectory['state'] for trajectory in samples])
            print("Fitting State Predictor")
            self.state_predictor.fit(state_skills.reshape(self.state_predictor_batch_size * 32, self.state_skill_shape),
                                     state_differences, batch_size=self.state_predictor_batch_size,
                                     steps_per_epoch=32, epochs=1)

            # Training Policy Network
            sample = samples[0]
            state = sample['state']
            state_skill = sample['state_skill']
            action = sample['action']
            next_state = sample['next_state']
            next_state_skill = sample['next_state_skill']
            skill = sample['skill']
            sample_terminal = sample['terminal']
            reward = self.get_policy_reward(state, next_state, skill)
            self.policy.learn(state_skill, action, reward, next_state_skill, sample_terminal)
        return
