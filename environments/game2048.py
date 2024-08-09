import numpy as np
import random
from typing import Any

from environments.environment import Environment

def double_adjacent(array, array_len=None, zero_stop=False):
    if array_len is None:
        array_len = len(array)

    new_array = np.zeros(array_len)
    i = 0
    j = 0
    while j < array_len - 1:
        elm = array[j]

        if zero_stop and elm == 0:
            break

        if elm == array[j + 1]:
            new_array[i] = elm * 2
            array[j + 1] = 0
            j += 1
        else:
            new_array[i] = elm

        i += 1
        j += 1

    if array[array_len - 1] != 0:
        new_array[i] = array[array_len - 1]

    return new_array


# Takes a numpy array and moves the values such that all the zeroes in the array are grouped at the end:
# [4, 0, 2, 0, 4] -> [4, 2, 4, 0, 0]
def zeros_at_end(array, array_len=None):
    if array_len is None:
        array_len = len(array)

    new_array = np.zeros(array_len)
    i = 0
    for elm in array:
        if elm != 0:
            new_array[i] = elm
            i += 1
    return new_array


class Game2048(Environment):

    push_up_action = 0
    push_down_action = 1
    push_right_action = 2
    push_left_action = 3
    possible_actions = [push_up_action,
                        push_down_action,
                        push_right_action,
                        push_left_action]
    num_possible_actions = len(possible_actions)
    action_prob = 1/num_possible_actions

    step_reward = 0.0

    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.current_state = None
        self.environment_name = 'game_2048_' + str(self.width) + 'x' + str(self.height)
        self.terminal = True
        return

    # The state immediately action the action is taken but not before ful transition dynamics apply,
    # i.e: push the blocks before generating a new block
    def get_action_state(self, state, action):
        action_state = np.zeros((self.width, self.height))

        if action == self.push_up_action:
            for col_index in range(self.height):
                col = state[:, col_index]
                shifted_col = double_adjacent(zeros_at_end(col, self.width), self.width, True)
                action_state[:, col_index] = shifted_col
        elif action == self.push_down_action:
            for col_index in range(self.height):
                col = state[:, col_index]
                shifted_col = np.flip(col)
                shifted_col = double_adjacent(zeros_at_end(shifted_col, self.width), self.width, True)
                action_state[:, col_index] = np.flip(shifted_col)
        elif action == self.push_left_action:
            for row_index in range(self.height):
                row = state[row_index, :]
                shifted_row = double_adjacent(zeros_at_end(row, self.width), self.width, True)
                action_state[row_index, :] = shifted_row
        elif action == self.push_right_action:
            for row_index in range(self.height):
                row = state[row_index, :]
                shifted_row = np.flip(row)
                shifted_row = double_adjacent(zeros_at_end(shifted_row, self.width), self.width, True)
                action_state[row_index, :] = np.flip(shifted_row)

        return action_state

    def get_start_states(self):
        start_state = np.zeros((self.width, self.height))
        start_state[(0, 0)] = 2
        return [start_state]

    def get_successor_states(self, state, probability_weights=False):
        if self.is_terminal(state):
            return [], []

        num_successor_states = 0
        successors = []
        weights = []
        stationary_actions = 0
        for action in self.possible_actions:
            action_state = self.get_action_state(state, action)

            empty_tiles = []
            num_empty_tiles = 0
            for x in range(self.width):
                for y in range(self.height):
                    if action_state[x, y] == 0:
                        num_empty_tiles += 1
                        empty_tiles.append((x, y))

            if num_empty_tiles <= 0:
                if np.array_equal(state, action_state):
                    stationary_actions += 1
                    continue

                successors.append(action_state)
                weights.append(1)
                num_successor_states += 1
                continue

            base_prob = 1 / num_empty_tiles
            for empty_tile in empty_tiles:
                successor = action_state.copy()
                successor[empty_tile[0], empty_tile[1]] = 2
                successors.append(successor)
                weights.append(base_prob)
                num_successor_states += 1

        # adding stationary state
        if stationary_actions > 0:
            successors.append(state.copy())
            weights.append(stationary_actions / self.num_possible_actions)
            num_successor_states += 1

        if not probability_weights:
            weights = [1] * num_successor_states
            return successors, weights

        # Modifying probabilities for action choices
        for i in range(num_successor_states - 1):
            weights[i] = weights[i] * self.action_prob

        return successors, weights

    def is_terminal(self, state=None):
        # Terminal if there is no adjacent pair of values and no empty tiles
        if state is None:
            state = self.current_state

        if state is None:
            raise AttributeError("State must not be none or environment not terminal")

        for x in range(self.width):
            next_x = x + 1
            for y in range(self.height):
                last_value = state[x, y]
                if last_value == 0:
                    return False

                next_y = y + 1

                if next_x < self.width:
                    if state[next_x, y] == last_value:
                        return False
                if next_y < self.height:
                    if state[x, next_y] == last_value:
                        return False
                last_value = state[x, y]
        return True

    def reset(self) -> Any:
        self.current_state = np.zeros((self.width, self.height))
        self.current_state[(0, 0)] = 2
        self.terminal = False
        return self.current_state.copy()

    def step(self, action) -> (Any, float, bool, Any):
        reward = self.step_reward

        self.current_state = self.get_action_state(self.current_state, action)

        empty_tiles = []
        has_empty_tiles = False
        for x in range(self.width):
            for y in range(self.height):
                if self.current_state[x, y] == 0:
                    has_empty_tiles = True
                    empty_tiles.append((x, y))

        if has_empty_tiles:
            new_tile = random.choice(empty_tiles)
            self.current_state[(new_tile[0], new_tile[1])] = 2

        if self.is_terminal():
            self.terminal = True
            reward += np.max(self.current_state)

        return self.current_state.copy(), reward, self.terminal, None
