import json
import numpy as np

from environments.environment import Environment
import random as rand
from genericfunctions import max_index, max_key


class QLearningAgent:

    default_intrinsic_reward_lambda = 0.5

    def __init__(self, actions, alpha, epsilon, gamma, intrinsic_reward=None,
                 intrinsic_reward_lambda=None):
        self.actions = actions

        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        self.q_values = {}
        self.intrinsic_reward = intrinsic_reward
        self.intrinsic_reward_lambda = intrinsic_reward_lambda
        if self.intrinsic_reward is not None and intrinsic_reward_lambda is None:
            self.intrinsic_reward_lambda = self.default_intrinsic_reward_lambda
        return

    def copy(self, copy_from):
        self.q_values = copy_from.q_values.copy()
        return

    def choose_action(self, state, optimal_choice=False, possible_actions=None):
        if possible_actions is None:
            possible_actions = self.actions
        if not optimal_choice and rand.uniform(0, 1) < self.epsilon:
            return rand.choice(possible_actions)

        action_values = self.get_action_values(state)

        if possible_actions is None:
            return max_key(action_values)

        valid_action_values = {}
        for action in possible_actions:
            valid_action_values[action] = action_values[action]
        return max_key(valid_action_values)

    def get_action_values(self, state):
        state = np.array2string(state)
        try:
            action_values_str = self.q_values[state]
        except KeyError:
            self.initialise_action_values(state)
            action_values_str = self.q_values[state]
        action_values_int = {int(key): action_values_str[key] for key in action_values_str}
        return action_values_int

    def initialise_action_values(self, state):
        self.q_values[state] = {action: 0.0 for action in self.actions}
        return

    def learn(self, state, action, reward, next_state, terminal=None, next_state_possible_actions=None):
        action_values = self.get_action_values(state)
        next_action_values_dict = self.get_action_values(next_state)

        if next_state_possible_actions is None:
            next_state_possible_actions = self.actions

        next_action_values = [next_action_values_dict[a] for a in next_state_possible_actions]
        if not next_action_values:
            next_action_values = [0.0]

        if self.intrinsic_reward is not None:
            intrinsic_reward = self.intrinsic_reward(next_state)
            reward = reward + (self.intrinsic_reward_lambda * intrinsic_reward)

        action_value = action_values[action]
        self.q_values[np.array2string(state)][action] = action_value + (self.alpha *
                                                        (reward + (self.gamma * max(next_action_values)
                                                                   - action_value)))
        return

    def load(self, load_path):
        try:
            with open(load_path, 'r') as f:
                self.q_values = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError()

        for state in self.q_values:
            old_dict = self.q_values[state]
            new_dict = {int(key): old_dict[key] for key in old_dict}
            self.q_values[state] = new_dict
        return

    def save(self, save_path):
        try:
            with open(save_path, 'w') as f:
                json.dump(self.q_values, f)
        except FileNotFoundError:
            f = open(save_path, 'x')
            f.close()
            self.save(save_path)
        return
