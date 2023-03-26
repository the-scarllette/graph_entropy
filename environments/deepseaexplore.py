from environments.environment import Environment

import numpy as np
from typing import Any


class DeepSeaExplore(Environment):

    default_n = 10
    default_start = (0, 0)
    success_reward = 1.0
    failure_reward = -1.0

    def __init__(self, N=default_n, start=default_start):
        if N < 2:
            N = DeepSeaExplore.default_n
        self.N = N
        self.start = start

        self.terminal = True
        self.x = self.y = None

        self.step_reward = 0.001 / self.N
        return

    def get_adjacency_matrix(self, directed=True):
        connected_states = {}

        all_states = [{'x': self.start[0], 'y': self.start[1]}]
        to_add = [{'x': self.start[0], 'y': self.start[1]}]

        def dict_to_str(d):
            return "(" + str(d['x']) + ', ' + str(d['y']) + ")"

        while len(to_add) > 0:
            state = to_add.pop()
            successors = self.get_successor_states(state)

            connected_states[dict_to_str(state)] = successors.copy()

            for successor in successors:
                if successor not in all_states:
                    all_states.append(successor)
                    to_add.append(successor)

        num_states = len(all_states)
        adj_matrix = np.zeros((num_states, num_states))
        for i in range(num_states):
            state = all_states[i]
            connected = connected_states[dict_to_str(state)]

            for connected_state in connected:
                j = all_states.index(connected_state)
                adj_matrix.itemset((i, j), 1.0)
                if not directed:
                    adj_matrix.itemset((j, i), 1.0)
        return adj_matrix, all_states

    def get_successor_states(self, state):
        successors = []

        new_y = state['y'] + 1
        if new_y >= self.N:
            return []

        left_x = state['x'] - 1
        right_x = state['x'] + 1

        if 0 <= left_x:
            successors.append({'x': left_x, 'y': new_y})
        if right_x < self.N:
            successors.append({'x': right_x, 'y': new_y})
        return successors

    def get_current_state(self, true_state=False):
        if self.terminal:
            raise AttributeError("Environment is terminal")

        if true_state:
            return {'x': self.x, 'y': self.y}
        return str(self.x) + '/' + str(self.y)

    def reset(self, true_state=False) -> Any:
        self.terminal = False
        self.x = self.y = 0
        return self.get_current_state()
