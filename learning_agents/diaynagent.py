from environments.environment import Environment
from learning_agents.metacontrolleragent import MetaControllerAgent
from learning_agents.softactorcritic import SoftActorCritic

from keras import layers
import scipy
from tensorflow import keras
import tensorflow as tf
import numpy as np
import random

# DIAYN Process
# While not converged: num_cycles
#   pick a skill from p(z)
#   for t to steps_per_ep: num_steps
#       take action a_t ~ policy(a_t | s_t, z)
#       get next step in environment
#       compute q(z|s_{t + 1})
#       r_t = log(q(z|s_{t + 1})) - log(p(z))
#       update policy with reward r_t
#       update q with SGD


class DIAYNAgent(MetaControllerAgent):

    def __init__(self, actions, state_shape=None,
                 num_skills=None, skill_length=10, model_layers=[128, 128],
                 lr=3e-4, policy_tau=0.005,
                 batch_size=16,
                 skill_predictor_noise=1e-4):
        self.actions = actions
        self.num_actions = len(self.actions)
        self.state_shape = state_shape
        self.model_layers = model_layers
        self.num_model_layers = len(self.model_layers)
        self.current_skill = None
        self.num_skills = num_skills
        self.state_skill_shape = self.num_skills + self.state_shape
        self.action_skill_shape = self.num_skills + self.num_actions
        self.batch_size = batch_size
        self.skill_length = skill_length
        self.current_skill_length = 0
        self.skill_predictor_noise = skill_predictor_noise
        self.policy_reward_constant = np.log(self.num_skills)
        self.experience = []
        self.experience_size = 0
        self.memory_size = 10e6

        # Creating Policy network (p(a | s, z))
        self.policy = SoftActorCritic(self.actions, self.state_skill_shape, lr,
                                      batch_size=self.batch_size,
                                      tau=policy_tau, model_layers=self.model_layers)

        # Creating the meta-controller
        self.meta_controller = SoftActorCritic([i for i in range(self.num_skills)], self.state_shape,
                                               batch_size=self.batch_size,
                                               tau=policy_tau, model_layers=self.model_layers)

        # Creating skill predictor network q(z|s_{t + 1})
        self.skill_predictor = keras.Sequential()
        self.skill_predictor.add(layers.Dense(units=self.model_layers[0],
                                              activation='relu',
                                              input_dim=self.state_shape))
        for i in range(1, self.num_model_layers):
            self.skill_predictor.add(layers.Dense(units=self.model_layers[i],
                                                  activation='relu'))
        self.skill_predictor.add(layers.Dense(units=self.num_skills, activation='sigmoid'))
        self.skill_predictor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                                     loss='mse', metrics=None)
        return

    def get_policy_reward(self, next_state, skill):
        skill_probabilities = self.skill_predictor(next_state.reshape(1, self.state_shape)).numpy()[0]
        skill_probability = skill_probabilities[skill] + self.skill_predictor_noise
        reward = np.log(skill_probability) + self.policy_reward_constant
        return reward

    def learn_skills(self, environment: Environment, num_episodes, max_steps_per_episode,
                     all_actions_valid=False):
        if all_actions_valid:
            possible_actions = environment.possible_actions

        for _ in range(num_episodes):
            self.sample_skill()
            terminal = False
            state = environment.reset()

            possible_actions = environment.possible_actions
            if not all_actions_valid:
                possible_actions = environment.get_possible_actions()
            for _ in range(max_steps_per_episode):
                if terminal:
                    break

                skill_state_vector = self.create_skill_state_vector(self.current_skill, state)
                action = self.policy.choose_action(skill_state_vector, False, possible_actions)

                next_state, _, terminal, _ = environment.step(action)
                next_skill_state = self.create_skill_state_vector(self.current_skill, next_state)
                if not all_actions_valid:
                    possible_actions = environment.get_possible_actions()

                reward = self.get_policy_reward(next_state, self.current_skill)
                self.policy.learn(skill_state_vector, action, reward, next_skill_state,
                                  terminal, possible_actions)

                self.train_skill_predictor(next_state, self.current_skill)

        return

    def train_skill_predictor(self, state, skill):
        if self.experience_size >= self.memory_size:
            index_to_remove = random.randint(0, self.memory_size - 1)
            del self.experience[index_to_remove]
            self.experience_size -= 1

        skill_vector = np.zeros(self.num_skills)
        skill_vector[skill] = 1

        self.experience.append({'state': state,
                                'skill_vector': skill_vector})
        self.experience_size += 1

        if self.experience_size < self.batch_size * 32:
            return

        samples = random.sample(self.experience, self.batch_size * 32)
        states = np.array([sample['state'] for sample in samples])
        skill_vectors = np.array([sample['skill_vector'] for sample in samples])
        print("Fitting Skill Predictor")
        self.skill_predictor.fit(states.reshape(self.batch_size * 32, self.state_shape),
                                 skill_vectors, batch_size=self.batch_size,
                                 steps_per_epoch=32, epochs=1)
        return
