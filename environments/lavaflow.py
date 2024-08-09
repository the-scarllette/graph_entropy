import numpy as np
import random
from typing import Any

from environments.environment import Environment


class LavaFlow(Environment):
    north_action = 0
    south_action = 1
    east_action = 2
    west_action = 3
    north_block_action = 4
    south_block_action = 5
    east_block_action = 6
    west_block_action = 7

    move_actions = [north_action, south_action, east_action, west_action]
    possible_actions = [north_action, south_action, east_action, west_action,
                        north_block_action, south_block_action, east_block_action, west_block_action]

    empty_tile = 0
    agent_tile = 1
    lava_tile = 2
    block_tile = 3
    potential_goal_tile = 4
    goal_tile = 5
    goal_and_agent_tile = 6
    possible_goal_and_agent_tile = 7

    blocked_tiles = [lava_tile, block_tile]

    default_board = np.array([[4, 0, 1, 2],
                              [4, 0, 0, 2]])
    default_board_name = 'corridor'

    step_reward = 0.0
    success_reward = 1.0
    failure_reward = -1.0

    def __init__(self, board=None, prob_goal_appearing=0.1, board_name=None):
        self.prob_goal_appearing = prob_goal_appearing

        self.board = board
        if self.board is None:
            self.board = self.default_board
        self.width = self.board.shape[1]
        self.height = self.board.shape[0]
        self.potential_goal_locations = []
        self.agent_start = None
        for x in range(self.width):
            for y in range(self.height):
                tile = self.board[y, x]
                if tile == self.agent_tile:
                    if self.agent_start is not None:
                        raise ValueError("Invalid board, boards can only contain 1 agent start location")
                    self.agent_start = (x, y)
                elif tile == self.potential_goal_tile:
                    self.potential_goal_locations.append((x, y))

        self.goal_location = None
        self.x = self.y = None
        self.current_state = None
        self.terminal = True

        self.environment_name = 'lavaflow'
        if board_name is None:
            board_name = self.default_board_name
        self.environment_name += '_' + board_name
        return

    def get_start_states(self):
        return [self.board.copy()]

    def get_successor_states(self, state, probability_weights=False):
        stationary_actions = 0
        num_successors = 0
        successors = []
        weights = []

        if state[0, 0] == 7:
            w = 2

        if self.is_terminal(state):
            return successors, weights

        agent_x = agent_y = goal_x = goal_y = None
        for x in range(self.width):
            for y in range(self.height):
                if state[y, x] in [self.agent_tile, self.possible_goal_and_agent_tile]:
                    agent_x = x
                    agent_y = y
                elif state[y, x] == self.goal_tile:
                    goal_x = x
                    goal_y = y

        removing_agent_tile = {self.agent_tile: self.empty_tile,
                               self.goal_and_agent_tile: self.goal_tile,
                               self.possible_goal_and_agent_tile: self.potential_goal_tile}
        placing_agent_tile = {self.empty_tile: self.agent_tile,
                              self.goal_tile: self.goal_and_agent_tile,
                              self.potential_goal_tile: self.possible_goal_and_agent_tile,
                              self.lava_tile: self.lava_tile}
        agent_neighbours = [(agent_x + 1, agent_y), (agent_x - 1, agent_y),
                            (agent_x, agent_y + 1), (agent_x, agent_y - 1)]
        for neighbour in agent_neighbours:
            if self.is_node_blocked(neighbour, state, [self.block_tile]):
                stationary_actions += 2
                continue
            # Moving Agent
            if state[neighbour[1], neighbour[0]] == self.lava_tile:
                stationary_actions += 1
            else:
                successor = state.copy()
                successor[agent_y, agent_x] = removing_agent_tile[successor[agent_y, agent_x]]
                successor[neighbour[1], neighbour[0]] = placing_agent_tile[successor[neighbour[1], neighbour[0]]]
                successors.append(successor)
                num_successors += 1
            # Placing Block
            successor = state.copy()
            successor[neighbour[1], neighbour[0]] = self.block_tile
            successors.append(successor)
            num_successors += 1

        # Adding Weights
        weights = [1 / len(self.possible_actions)] * num_successors

        # Adding Stationary State
        if stationary_actions > 0:
            num_successors += 1
            successors.append(state.copy())
            weights += [stationary_actions / len(self.possible_actions)]

        # Spreading Lava
        successors = [self.spread_lava(successors[i]) for i in range(num_successors)]

        # Potentially Placing goal
        if (goal_x is None) or (goal_y is None):
            placing_goal_lookup = {self.potential_goal_tile: self.goal_tile,
                                   self.possible_goal_and_agent_tile: self.goal_and_agent_tile}

            for i in range(num_successors):
                successor = successors[i].copy()
                valid_goals = [potential_goal for potential_goal in self.potential_goal_locations
                               if successor[potential_goal[1], potential_goal[0]]
                               in [self.potential_goal_tile, self.possible_goal_and_agent_tile]]
                num_valid_goals = len(valid_goals)
                if num_valid_goals > 0:
                    weights[i] = weights[i] * (1 - self.prob_goal_appearing)
                for potential_goal in valid_goals:
                    goal_successor = successor.copy()
                    goal_successor[potential_goal[1], potential_goal[0]] = placing_goal_lookup[
                        goal_successor[potential_goal[1], potential_goal[0]]]
                    successors.append(goal_successor)
                    num_successors += 1
                    weights.append(weights[i] * self.prob_goal_appearing * (1 / num_valid_goals))

        # Merging Matching Successors
        merged_successors = []
        merged_weights = []
        found_successors = []
        num_merged_successors = 0
        for i in range(num_successors):
            successor = successors[i]
            weight = weights[i]
            successor_bytes = successor.tobytes()

            if successor_bytes in found_successors:
                index = found_successors.index(successor_bytes)
                merged_weights[index] += weight
                continue

            merged_successors.append(successor)
            found_successors.append(successor_bytes)
            merged_weights.append(weight)
            num_merged_successors += 1

        # Returning Successors
        if not probability_weights:
            merged_weights = [1.0] * num_merged_successors
        return merged_successors, merged_weights

    def is_node_blocked(self, node, state=None, tiles_to_block=None):
        if state is None:
            if self.terminal:
                raise AttributeError("To find if a node is blocked, provide a state or make the environment"
                                     "non-terminal")
            state = self.current_state
        if tiles_to_block is None:
            tiles_to_block = self.blocked_tiles

        node_x = node[0]
        node_y = node[1]
        if (not (0 <= node_x < self.width)) or (not (0 <= node_y < self.height)):
            return True
        return state[node_y, node_x] in tiles_to_block

    def is_terminal(self, state=None):
        # Terminal if
        # Agent is on a goal location
        # No Agent on the board
        # Agent cannot reach a goal location or a possible goal location
        # A block or lava is on a goal location

        if state is None:
            if self.terminal:
                return True
            state = self.current_state

        agent_x = agent_y = goal_x = goal_y = None
        for x in range(self.width):
            for y in range(self.height):
                if state[y, x] in [self.agent_tile, self.possible_goal_and_agent_tile]:
                    agent_x = x
                    agent_y = y
                elif state[y, x] == self.goal_tile:
                    goal_x = x
                    goal_y = y
                elif state[y, x] == self.goal_and_agent_tile:
                    return True

        if goal_y is None:
            for potential_goal in self.potential_goal_locations:
                if state[potential_goal[1], potential_goal[0]] not in [self.potential_goal_tile,
                                                                       self.possible_goal_and_agent_tile]:
                    return True

        if (agent_x is None) or (agent_y is None):
            return True

        if (goal_x is None) and (goal_y is None):
            return False

        return not self.path_exists(agent_x, agent_y, goal_x, goal_y, state)

    def path_exists(self, start_x, start_y, end_x, end_y, state=None):
        if state is None:
            if self.terminal:
                raise AttributeError("To find a path between nodes, provide a state or make the environment"
                                     "non-terminal")
            state = self.current_state

        start_node = (start_x, start_y)

        def is_end_node(node):
            return (node[0] == end_x) and (node[1] == end_y)

        if is_end_node(start_node):
            return not self.is_node_blocked(start_node, state)

        nodes_to_search = [start_node]
        searched = []
        num_to_search = 1
        while num_to_search > 0:
            current_node = nodes_to_search.pop()
            num_to_search -= 1
            if is_end_node(current_node):
                return True

            searched.append(current_node)

            x = current_node[0]
            y = current_node[1]
            neighbours = [(x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)]
            for neighbour in neighbours:
                if self.is_node_blocked(neighbour, state) or (neighbour in searched):
                    continue
                nodes_to_search.append(neighbour)
                num_to_search += 1

        return False

    def reset(self) -> Any:
        self.current_state = self.board.copy()
        self.x = self.agent_start[0]
        self.y = self.agent_start[1]
        self.goal_location = None

        self.terminal = False
        return self.current_state.copy()

    def spread_lava(self, state):
        new_state = state.copy()
        for x in range(self.width):
            for y in range(self.height):
                if state[y, x] == self.lava_tile:
                    next_lava = [(x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)]
                    for node in next_lava:
                        if self.is_node_blocked(node, state):
                            continue
                        new_state[node[1], node[0]] = self.lava_tile
        return new_state

    def step(self, action) -> (Any, float, bool, Any):
        if (self.goal_location == (0, 1)) and (self.current_state[1, 0] == self.block_tile):
            self.terminal = True
            reward = self.failure_reward
            return self.current_state.copy(),reward, True, None

        current_state_temp = self.current_state.copy()
        x = self.x
        y = self.y
        reward = self.step_reward

        # Agent takes action
        if action in [self.north_action, self.north_block_action]:
            y -= 1
        elif action in [self.south_action, self.south_block_action]:
            y += 1
        elif action in [self.east_action, self.east_block_action]:
            x += 1
        else:
            x -= 1
        if action in self.move_actions:
            if not self.is_node_blocked((x, y), self.current_state, [self.block_tile]):
                new_tile_lookup = {self.agent_tile: self.empty_tile,
                                   self.goal_and_agent_tile: self.goal_tile,
                                   self.possible_goal_and_agent_tile: self.potential_goal_tile,
                                   self.empty_tile: self.agent_tile,
                                   self.goal_tile: self.goal_and_agent_tile,
                                   self.potential_goal_tile: self.possible_goal_and_agent_tile,
                                   self.lava_tile: self.lava_tile}
                current_state_temp[self.y, self.x] = new_tile_lookup[current_state_temp[self.y, self.x]]
                self.x = x
                self.y = y
                current_state_temp[(self.y, self.x)] = new_tile_lookup[current_state_temp[(self.y, self.x)]]
        elif (0 <= x < self.width) and (0 <= y < self.height):
            if (current_state_temp[y, x] == self.goal_tile) or  (self.goal_location is None and
                                                                 current_state_temp[y, x] == self.potential_goal_tile):
                self.terminal = True
                reward += self.failure_reward
                self.current_state = current_state_temp.copy()
                return self.current_state.copy(), reward, self.terminal, None
            current_state_temp[y, x] = self.block_tile

        # Lava spreads
        current_state_temp = self.spread_lava(current_state_temp)

        # Assign Goal
        if self.goal_location is None:
            if random.uniform(0, 1) <= self.prob_goal_appearing:
                valid_goal_locations = [loc for loc in self.potential_goal_locations
                                        if current_state_temp[loc[1], loc[0]] not in
                                               [self.lava_tile, self.block_tile]]

                if len(valid_goal_locations) <= 0:
                    reward += self.failure_reward
                    self.terminal = True
                    return self.current_state.copy(), reward, self.terminal, None

                self.goal_location = random.choice(valid_goal_locations)
                new_tile_lookup = {self.potential_goal_tile: self.goal_tile,
                                   self.possible_goal_and_agent_tile: self.goal_and_agent_tile}
                current_state_temp[self.goal_location[1], self.goal_location[0]] = new_tile_lookup[
                    current_state_temp[self.goal_location[1], self.goal_location[0]]]

        # Check if goal reached
        self.current_state = current_state_temp.copy()

        if self.current_state[self.y, self.x] == self.goal_and_agent_tile:
            reward += self.success_reward
            self.terminal = True
        elif self.is_terminal(self.current_state):
            reward += self.failure_reward
            self.terminal = True

        return self.current_state.copy(), reward, self.terminal, None
