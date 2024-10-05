import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random as rand
from typing import Any, Tuple

from environments.environment import Environment


class SimpleWindGridWorld(Environment):
    possible_actions = [0, 1, 2]

    num_possible_actions = len(possible_actions)

    action_lookup = {0: np.array([0, 0, 0]),
                     1: np.array([0, 0, -1]),
                     2: np.array([0, 0, 1])}

    height = 5

    wind_layers = np.array([
        [0, 0],  # Stationary
        [1, 1],  # NE
        [1, -1],  # SE
        [-1, -1],  # SW
        [-1, 1]  # NW
    ])

    wind_vector_display: np.ndarray = np.array([
        [0., 0., 0., 0., 0.],  # Stationary
        [0.25, 0.25, 0.25, 0.25, 1.],  # NE
        [0.25, 0.75, 0.2475, -0.2475, 1.],  # SE
        [0.75, 0.75, -0.25, -0.25, 1.],  # SW
        [0.75, 0.25, -0.353, 0.353, 1.],  # NW
    ])

    step_reward = -0.01
    success_reward = 1.0

    def __init__(self, size: Tuple[int, int], agent_camera_radius: float = 1,
                 land: bool = False,
                 random_wind: bool = False):
        self.state_dtype = int
        self.current_state = None
        self.size = (size[0], size[1], self.height)
        self.goal = np.array([self.size[1] // 2, self.size[0] // 2, 0])
        self.terminal = True

        self.agent_camera_radius = agent_camera_radius
        self.environment_name = ('simple_wind_gridworld_' + str(self.agent_camera_radius) +
                                 'x' + str(self.size[0]) + 'x' + str(self.size[1]))

        self.state_len = 3 + (size[0] * size[1])
        self.state_shape = (self.state_len,)

        self.start_states = [np.array([x, y, 0] + ([0] * (self.state_len - 3)))
                             for x in range(self.size[0]) for y in range(self.size[1])]

        self.land = land
        if self.land:
            self.environment_name += '_land'
        self.random_wind = random_wind

        if self.land:
            self.environment_name += '_land'
        return

    def agent_visual_update(self, state):
        state = state.copy()

        for x in range(self.size[0]):
            for y in range(self.size[1]):
                if np.linalg.norm(state[:2] - np.array([y, x])) <= self.agent_camera_radius:
                    state[(x * self.size[0]) + y + 3] = 1
        return state

    def get_start_states(self):
        return self.start_states

    def get_successor_states(self, state, probability_weights=False):
        if self.is_terminal(state):
            return [], []

        successor_states = []
        probability_weights = []
        default_weight = 1/self.num_possible_actions
        if not probability_weights:
            default_weight = 1.0

        def get_successor_state(altitude):
            successor = np.zeros(self.state_len, dtype=int)
            successor[2] = altitude
            successor[0:2] = np.clip(state[0:2] + self.wind_layers[altitude], 0,
                                     np.array([self.size[1], self.size[0]]) - 1)
            successor[3:] = state[3:]

            successor = self.agent_visual_update(successor)
            return successor

        available_altitudes = np.clip(np.array([state[2], state[2] - 1, state[2] + 1]), 0, self.height - 1)

        for available_altitude in available_altitudes:
            successor_states.append(get_successor_state(available_altitude))

        num_successors = 3
        trimmed_successors = []

        for i in range(num_successors):
            successor = successor_states[i]
            if successor is None:
                continue

            weight = default_weight

            for j in range(num_successors):
                other_successor = successor_states[j]
                if other_successor is None:
                    continue

                if np.array_equal(successor, other_successor):
                    successor_states[j] = None
                    if probability_weights:
                        weight += default_weight

            trimmed_successors.append(successor)
            probability_weights.append(weight)

        return trimmed_successors, probability_weights

    def is_scanning_complete(self, state=None):
        if state is None:
            state = self.current_state
        if state is None:
            raise ValueError("Either provide a state or ensure environment is not terminal.")

        for i in range(3, self.state_len):
            if state[i] == 0:
                return False
        return True

    def is_terminal(self, state=None):
        if state is None:
            state = self.current_state
        if state is None:
            raise ValueError("Either provide a state or ensure environment is not terminal.")

        # If areas still left to scan: not terminal
        if not self.is_scanning_complete(state):
            return False

        # No areas left to scan and do not need to land: terminal
        if not self.land:
            return True
        # No areas left to scan and need to land: terminal if on the ground
        return state[2] == 0

    def print_state(self, state=None):
        if state is None:
            if self.terminal:
                raise AttributeError("Either provide a state or print state while environment is not terminal.")
            state = self.current_state

        state_to_print = ["" for _ in range(self.size[1])]
        print("Ground seen: ")
        for y in range(self.size[1]):
            for x in range(self.size[0]):
                to_place = "#"
                if state[(x * self.size[0]) + y + 3] == 1:
                    to_place = "-"
                state_to_print[y] += to_place
            print(state_to_print[y])

        print("Height: " + str(state[2]))
        print("x: " + str(state[0]))
        print("y: " + str(state[1]))
        return

    def step(self, action) -> (Any, float, bool, Any):
        # Finding action
        try:
            move_vector = self.action_lookup[action]
        except KeyError:
            raise ValueError(str(action) + " is not a valid action.")

        move_vector = np.append(move_vector, [0] * (self.state_len - 3))

        # Moving agent
        move_altitude = np.clip((self.current_state + move_vector)[2], 0, self.height - 1)
        self.current_state[2] = move_altitude
        wind_vector = self.wind_layers[move_altitude]
        self.current_state[0:2] = np.clip(self.current_state[0:2] + wind_vector, 0,
                                          np.array([self.size[1], self.size[0]]) - 1)

        # Scanning ground
        self.current_state = self.agent_visual_update(self.current_state)

        # Providing reward and checking if terminal
        reward = self.step_reward
        if self.is_terminal():
            self.terminal = True
            reward += self.success_reward

        return self.current_state.copy(), reward, self.terminal, None

    def reset(self, start_state=None) -> Any:
        self.terminal = False

        if start_state is not None:
            self.current_state = start_state
            return self.current_state

        self.current_state = np.array([0] * self.state_len)
        self.current_state[0] = rand.randint(0, self.size[1] - 1)
        self.current_state[1] = rand.randint(0, self.size[0] - 1)

        return self.current_state

    def visualise_subgoals(self, subgoal_key, title=""):

        try:
            with open(self.environment_name + '_stg_values.json', 'r') as f:
                stg_data = json.load(f)
        except FileNotFoundError:
            raise AttributeError("No STG of environment exists. Created one before visualising subgoals.")

        for altitude in range(self.height):
            visualised_grid = np.zeros(self.size[:-1], dtype=float)
            fig, ax = plt.subplots()

            for x in range(self.size[1]):
                ax.axvline(x=x, color='black', linestyle='-')
            for y in range(self.size[0]):
                ax.axhline(y=y, color='black', linestyle='-')

            # Set x and y ticks to match grid size and align with the center of each cell
            ax.set_xticks(np.arange(self.size[1]) + 0.5)
            ax.set_yticks(np.arange(self.size[0]) + 0.5)
            # Set tick labels to integer values
            ax.set_xticklabels(range(self.size[1]))
            ax.set_yticklabels(range(self.size[0]))

            for node in stg_data:
                state = stg_data[node]['state']
                state = state[1:-1]
                state = np.fromstring(state, dtype=int, sep=' ')
                if state[2] == altitude:
                    value = 1
                    if stg_data[node][subgoal_key] == 'True':
                        value = 0.5
                    visualised_grid[state[1], state[0]] = value

            for x in range(visualised_grid.shape[0]):
                for y in range(visualised_grid.shape[1]):
                    color = 'black'
                    if visualised_grid[y, x] <= 0:
                        continue

                    if self.wind_vector_display[altitude][4]:
                        ax.arrow(self.wind_vector_display[altitude][0] + x,
                                 self.wind_vector_display[altitude][1] + y,
                                 *tuple(self.wind_vector_display[altitude][2:4]), head_width=0.1,
                                 head_length=0.15, fc=color, ec=color)

            ax.imshow(visualised_grid, cmap='hot', vmin=0, vmax=1, extent=[0, self.size[1], 0, self.size[0]],
                      origin='lower')
            plt.title(title + " Altitude: " + str(altitude))
            plt.plot()
            plt.show()
        return
