import numpy as np
import random as rand

from environments.environment import Environment

from typing import Any


class GridworldTron(Environment):

    north_action = 0
    south_action = 1
    east_action = 2
    west_action = 3

    possible_actions = [north_action,
                        south_action,
                        east_action,
                        west_action]

    agent = 1
    coin = 3
    empty_square = 0
    opponent = 4
    wall = 2

    coin_reward = 1.0
    defeat_opponent_reward = 1.0
    failure_reward = -1.0
    step_reward = -0.01

    def __init__(self, width, height, use_opponent=False):

        self.width = width
        self.height = height
        if self.width <= 0:
            raise ValueError("Width must be larger than 0")
        if self.height <= 0:
            raise ValueError("Height must be larger than 0")

        self.using_opponent = use_opponent

        self.x = self.y = self.coin_x = self.coin_y = self.opp_x = self.opp_y = None
        self.current_state = None

        self.terminal = True

        self.environment_name = 'gridworldtron_' + str(self.width) + 'x' + str(self.height)
        if self.using_opponent:
            self.environment_name += '_opponent'
        return

    def get_start_states(self):
        start_states = []
        start_state_template = np.full((self.width, self.height), self.empty_square)

        start_state_template[0, 0] = self.agent

        if self.using_opponent:
            start_state_template[self.width - 1, self.height - 1] = self.opponent
            return [start_state_template]

        elm = self.coin

        for elm_x in range(self.width):
            for elm_y in range(self.height):
                if (elm_x == 0) and (elm_y == 0):
                    continue
                start_state = start_state_template.copy()
                start_state[elm_x, elm_y] = elm
                start_states.append(start_state)

        return start_states

    def get_successor_states(self, state, probability_weights=False):
        successor_states = []
        num_successor_states = 0
        weights = []
        coin_x = coin_y = agent_x = agent_y = opp_x= opp_y = None
        for x in range(self.width):
            for y in range(self.height):
                if state[x, y] == self.agent:
                    agent_x = x
                    agent_y = y
                elif state[x, y] == self.coin:
                    coin_x = x
                    coin_y = y
                elif state[x, y] == self.opponent:
                    opp_x = x
                    opp_y = y

        if (agent_x is None) or ((opp_x is None) and self.using_opponent):
            return successor_states, weights

        # Terminal and Move Successors
        next_agent_locations = [(agent_x - 1, agent_y), (agent_x + 1, agent_y),
                                (agent_x, agent_y - 1), (agent_x, agent_y + 1)]
        successor_template = state.copy()
        successor_template[agent_x, agent_y] = self.wall
        if self.using_opponent:
            successor_template[opp_x, opp_y] = self.wall
        terminal_successors = 0
        coin_reachable = False
        for location in next_agent_locations:
            next_x = location[0]
            next_y = location[1]

            if (not 0 <= next_x < self.width) or (not 0 <= next_y < self.height):
                terminal_successors += 1
                continue
            if not (state[next_x, next_y] in [self.empty_square, self.coin]):
                terminal_successors += 1
                continue

            if state[next_x, next_y] == self.coin:
                coin_reachable = True
                continue

            successor = successor_template.copy()
            successor[next_x, next_y] = self.agent
            successor_states.append(successor)
            weights.append(1/4)
            num_successor_states += 1

        # Opponent Successors
        if self.using_opponent:
            possible_opponent_next_spaces = [(opp_x - 1, opp_y), (opp_x + 1, opp_y),
                                             (opp_x, opp_y - 1), (opp_x, opp_y + 1)]
            opponent_successor_states = []
            opponent_successor_weights = []
            current_num_successor_states = num_successor_states
            for i in range(current_num_successor_states):
                successor = successor_states[i].copy()

                opponent_next_spaces = []
                num_opponent_next_spaces = 0
                for space in possible_opponent_next_spaces:
                    if (not 0 <= space[0] < self.width) or (not 0 <= space[1] < self.height):
                        continue
                    if successor[space[0], space[1]] == self.empty_square:
                        opponent_next_spaces.append(space)
                        num_opponent_next_spaces += 1

                if num_opponent_next_spaces <= 0:
                    opponent_successor_states.append(successor)
                    opponent_successor_weights.append(weights[i])
                    continue

                for opponent_next_space in opponent_next_spaces:
                    successor = successor_states[i].copy()
                    successor[opponent_next_space[0], opponent_next_space[1]] = self.opponent
                    opponent_successor_states.append(successor)
                    opponent_successor_weights.append(weights[i] * (1/num_opponent_next_spaces))
                    num_successor_states += 1

            successor_states = opponent_successor_states
            weights = opponent_successor_weights

        if terminal_successors > 0:
            successor_states.append(successor_template)
            num_successor_states += 1
            weights.append(terminal_successors/4)

        if coin_reachable:
            successor_coin_template = successor_template.copy()
            successor_coin_template[coin_x, coin_y] = self.agent
            coin_successors = 0
            for x in range(self.width):
                for y in range(self.height):
                    if successor_coin_template[x, y] != self.empty_square:
                        continue

                    successor = successor_coin_template.copy()
                    successor[x, y] = self.coin
                    coin_successors += 1
                    successor_states.append(successor)

            if coin_successors <= 0:
                successor_coin_template[coin_x, coin_y] == self.wall
                successor_states.append(successor_coin_template)
                coin_successors = 1

            weights += ([1/(4 * coin_successors)] * coin_successors)

        if not probability_weights:
            weights = [1.0] * len(successor_states)

        return successor_states, weights

    def print_state(self, state=None):
        if state is None:
            if self.terminal:
                raise AttributeError("To print a state either environment must not be terminal or provide a state")
            state = self.current_state

        symbol_lookup = {self.agent: 'A',
                         self.wall: '#',
                         self.coin: 'C',
                         self.empty_square: '_',
                         self.opponent: 'O'}

        for x in range(self.width - 1, -1, -1):
            row = ''
            for y in range(self.height):
                row += symbol_lookup[state[y, x]]
            print(row)
        return

    def reset(self) -> Any:
        self.current_state = np.full((self.width, self.height), self.empty_square)
        self.terminal = False

        self.x = 0
        self.y = 0
        self.current_state[self.x, self.y] = self.agent

        if self.using_opponent:
            self.opp_x = self.width - 1
            self.opp_y = self.height - 1
            self.current_state[self.opp_x, self.opp_y] = self.opponent
            return self.current_state.copy()

        self.coin_x = self.x
        self.coin_y = self.y
        while (self.coin_y == self.y) and (self.coin_x == self.x):
            self.coin_x = rand.randint(0, self.width - 1)
            self.coin_y = rand.randint(0, self.height - 1)

        self.current_state[self.coin_x, self.coin_y] = self.coin

        return self.current_state.copy()

    def step(self, action) -> (Any, float, bool, Any):
        if self.terminal:
            raise AttributeError("Cannot step in environment when it is terminal")

        next_x = self.x
        next_y = self.y

        if action == self.north_action:
            next_y += 1
        elif action == self.south_action:
            next_y -= 1
        elif action == self.east_action:
            next_x += 1
        elif action == self.west_action:
            next_x -= 1
        else:
            raise ValueError(str(action) + " is not a valid action")

        self.current_state[self.x, self.y] = self.wall

        reward = self.step_reward

        if (not 0 <= next_x < self.width) or (not 0 <= next_y < self.height) or (
                self.current_state[next_x, next_y] in [self.wall, self.opponent]):
            reward = self.failure_reward
            self.terminal = True
            return self.current_state.copy(), reward, self.terminal, None

        self.x = next_x
        self.y = next_y
        self.current_state[next_x, next_y] = self.agent

        if self.using_opponent:
            possible_opponent_spaces = [(self.opp_x - 1, self.opp_y), (self.opp_x + 1, self.opp_y),
                                        (self.opp_x, self.opp_y - 1), (self.opp_x, self.opp_y + 1)]
            self.current_state[self.opp_x, self.opp_y] = self.wall

            opponent_spaces = []
            no_spaces = True
            for space in possible_opponent_spaces:
                if (not 0 <= space[0] < self.width) or (not 0 <= space[1] < self.height):
                    continue
                if self.current_state[space[0], space[1]] == self.empty_square:
                    opponent_spaces.append(space)
                    no_spaces = False

            if no_spaces:
                reward = self.defeat_opponent_reward
                self.terminal = True
                return self.current_state.copy(), reward, self.terminal, None

            opponent_space = rand.choice(opponent_spaces)
            self.opp_x = opponent_space[0]
            self.opp_y = opponent_space[1]
            self.current_state[self.opp_x, self.opp_y] = self.opponent

        if (next_x == self.coin_x) and (next_y == self.coin_y):
            reward += self.coin_reward

            empty_squares = [(x, y) for x in range(self.width) for y in range(self.height)
                             if self.current_state[x, y] == self.empty_square]

            if len(empty_squares) == 0:
                self.current_state[next_x, next_y] = self.wall
                self.terminal = True
                return self.current_state.copy(), reward, self.terminal, None

            next_coin = rand.choice(empty_squares)
            self.coin_x = next_coin[0]
            self.coin_y = next_coin[1]
            self.current_state[next_coin[0], next_coin[1]] = self.coin

        return self.current_state.copy(), reward, self.terminal, None
