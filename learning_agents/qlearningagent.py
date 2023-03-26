import json
import numpy as np

from environments.environment import Environment
import random as rand
from genericfunctions import max_index, max_key


class QLearningAgent:

    def __init__(self, actions, alpha, epsilon, gamma):
        self.actions = actions

        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        self.q_values = {}
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
            action_values = self.q_values[state]
        except KeyError:
            self.initialise_action_values(state)
            action_values = self.q_values[state]
        return action_values

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

        action_value = action_values[action]
        action_values[action] = action_value + (self.alpha *
                                                (reward + (self.gamma * max(next_action_values) - action_value)))
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
