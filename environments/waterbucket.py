from environments.environment import Environment

import numpy as np
from typing import Any


class WaterBucket(Environment):

    default_buckets = [8, 5, 3]
    default_start = np.array([[8, 8],
                              [5, 0],
                              [3, 0]])
    default_goal = np.array([[8, 4],
                             [5, 4],
                             [3, 0]])

    step_reward = -0.1
    success_reward = 1.0

    def __init__(self, buckets=default_buckets, start=default_start, goal=default_goal):
        self.buckets = buckets
        self.num_buckets = len(self.buckets)
        self.start = start
        self.goal = goal
        self.terminal = True

        self.action_lookup = [np.array([i, j]) for i in range(self.num_buckets)
                              for j in range(self.num_buckets) if i != j]
        self.num_actions = len(self.action_lookup)
        self.possible_actions = list(range(self.num_actions))

        self.current_state = None
        return

    def pour(self, action, state):
        state = state.copy()

        chosen_action = self.action_lookup[action]

        pour_from = chosen_action[0]
        pour_to = chosen_action[1]

        if state[pour_from][1] <= 0:
            return state

        state[pour_to][1] += state[pour_from][1]
        excess_water = max(state[pour_to][1] - state[pour_to][0], 0)
        state[pour_from][1] = excess_water
        if excess_water > 0:
            state[pour_to][1] = state[pour_to][0]
        return state

    def get_start_states(self):
        return [self.start]

    def get_successor_states(self, state):
        if np.array_equal(state, self.goal):
            return []
        successors = [successor for action in range(self.num_actions)
                      if not np.array_equal(successor := self.pour(action, state), state)]
        return successors

    def step(self, action, true_state=False) -> (Any, float, bool, Any):
        if self.terminal:
            raise AttributeError("Environment is terminal")

        if not (0 <= action <= self.num_actions):
            raise AttributeError("Invalid action " + str(action))

        next_state = self.pour(action, self.current_state)

        reward = self.step_reward

        self.current_state = next_state
        if np.array_equal(self.current_state, self.goal):
            reward += self.success_reward
            self.terminal = True

        if true_state:
            return self.current_state.copy(), reward, self.terminal, None

        return np.array2string(self.current_state), reward, self.terminal, None

    def reset(self, true_state=False) -> Any:
        self.current_state = self.start.copy()
        self.terminal = False

        if true_state:
            return self.current_state
        return np.array2string(self.current_state)
