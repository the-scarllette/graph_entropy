import networkx as nx
import numpy as np
import random
from typing import Any

from environments.environment import Environment

# Agent appears in a maze with 1 or more squares of lava
# agent can move (N, S, E, W), place a block (N, S, E, W) or terminate the environment
# each timestep all lava spreads to adjacent squares, not through walls or blocks
# terminal if:
#   take the terminate action
#   lava and agent occupy the same tile
# agent cannot take the place blocks action if there is no path such that lava could reach the agent
# reward
#   0 each timestep
#   -0.1 for an illegal action (placing a block on an occupied square, moving into a wall)
#   -1.0 for entering lava or terminating when lava can reach the agent
#   when terminating and lava can no loner reach agent, +1.0 for each empty square the agent could reach


class LavaFlow(Environment):
    north_action = 0
    south_action = 1
    east_action = 2
    west_action = 3
    north_block_action = 4
    south_block_action = 5
    east_block_action = 6
    west_block_action = 7
    terminate_action = 8
    possible_actions = [north_action, south_action, east_action, west_action,
                        north_block_action, south_block_action, east_block_action, west_block_action]
    move_actions = [north_action, south_action, east_action, west_action]
    block_actions = [north_block_action, south_block_action, east_block_action, west_block_action]

    empty_tile = 0
    agent_tile = 1
    lava_tile = 2
    block_tile = 3

    blocked_tiles = [lava_tile, block_tile]

    default_board = np.array([[empty_tile, empty_tile, empty_tile, lava_tile],
                              [empty_tile, agent_tile, empty_tile, empty_tile]])
    default_board_name = 'corridor'

    failure_reward = -1.0
    invalid_action_reward = -0.1
    step_reward = 0.0

    def __init__(self, board: None | np.ndarray=None, board_name: None | str =None):
        self.state_dtype = int

        self.board = board
        self.board_name = board_name
        if self.board is None:
            self.board = self.default_board
            self.board_name = self.default_board_name

        self.board_graph = None
        self.state_shape = self.board.shape

        self.agent_start_i = None
        self.agent_start_j = None
        self.agent_i = None
        self.agent_j = None
        for i in range(self.state_shape[0]):
            for j in range(self.state_shape[1]):
                if self.board[i, j] == self.agent_tile:
                    if self.agent_start_i is not None:
                        raise ValueError("Board must include exactly 1 agent tile")
                    self.agent_start_i, self.agent_start_j = i, j
                    break
            if self.agent_start_i is not None:
                break
        if self.agent_start_j is None:
            raise ValueError("Board must include exactly 1 agent tile")
        self.safe_from_lava = False
        self.lava_nodes = []

        self.current_state = None
        self.terminal = True
        self.environment_name += 'lavaflow_' + board_name
        return

    def build_state_graph(self, state: np.ndarray | None) -> nx.Graph:
        if state is None:
            state = self.board
        state_graph = nx.Graph()
        for i in range(self.state_shape[0]):
            for j in range(self.state_shape[1]):
                if state[i, j] == self.block_tile:
                    continue
                node = self.cord_node_key(i, j)
                state_graph.add_node(node)

                for next_i in [max(i - 1, 0), min(i + 1, self.state_shape[0] - 1)]:
                    for next_j in [max(j - 1, 0), min(j + 1, self.state_shape[1] - 1)]:
                        if state[next_i, next_j] == self.block_tile:
                            continue
                        connected_node = self.cord_node_key(next_i, next_j)
                        state_graph.add_node(connected_node)
                        state_graph.add_edge(connected_node, node)
        return state_graph

    def cord_node_key(self, i: int, j: int) -> int:
        return (self.state_shape[0] * j) + i

    def get_start_states(self):
        return [self.board.copy()]

    # TODO: Successor states
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

    def has_path_to_lava(self, state: np.ndarray | None=None) -> bool:
        if state is None:
            if self.terminal:
                raise AttributeError("Must provide a state or environment must not be terminal")
            state_graph = self.board_graph
        else:
            state_graph = self.build_state_graph(state)

        agent_node = self.cord_node_key(self.agent_i, self.agent_j)

        for lava_node in self.lava_nodes:
            if nx.has_path(state_graph, lava_node, agent_node):
                return True
        return False

    def is_terminal(self, state: None | np.ndarray) -> bool:
        if state is None:
            state = self.current_state

        for i in range(self.state_shape[0]):
            for j in range(self.state_shape[1]):
                if state[i, j] == self.agent_tile:
                    return True
        return False

    # TODO: add variable state input
    def reset(self, state: np.ndarray | None) -> np.ndarray:

        self.current_state = self.board.copy()
        self.build_board_graph()
        self.agent_i, self.agent_j = self.agent_start_i, self.agent_start_j
        self.safe_from_lava = False
        self.terminal = False
        self.reset = []
        return self.current_state.copy()

    def step(self, action: int) -> (np.ndarray, float, bool, None):
        reward = self.step_reward
        i, j = self.agent_i, self.agent_j

        # Finding position of action
        if action in [self.north_action, self.north_block_action]:
            i -= 1
        elif action in [self.south_action, self.south_block_action]:
            i += 1
        elif action in [self.east_action, self.east_block_action]:
            j += 1
        else:
            j -= 1
        # Finding out if action possible
        action_possible = True
        if i < 0 or i >= self.state_shape[0]:
            reward = self.invalid_action_reward
            action_possible = False
        elif j < 0 or j >= self.state_shape[1]:
            reward = self.invalid_action_reward
            action_possible = False
        else:
            next_tile = self.current_state[i, j]
            if next_tile == self.block_tile:
                reward = self.invalid_action_reward
                action_possible = False
        # Moving Agent
        if (action in self.move_actions) and action_possible:
            self.current_state[self.agent_i, self.agent_j] = self.empty_tile
            self.agent_i, self.agent_j = i, j
            if next_tile == self.lava_tile:
                reward = self.failure_reward
                self.terminal = True
            else:
                self.current_state[self.agent_i, self.agent_j] = self.agent_tile
        #Placing Block
        elif (action in self.place_block_actions) and action_possible:
            if self.safe_from_lava:
                reward = self.invalid_action_reward
                action_possible = False
            else:
                self.current_state[i, j] = self.block_tile
                self.board_graph.remove_node(self.cord_node_key(i, j)) # updating graph
        elif action_possible and (action == self.terminate_action):
            self.terminal = True

        # Spread Lava;
        for i in range(self.state_shape[0]):
            for j in range(self.state_shape[1]):
                if self.current_state[i, j] == self.lava_tile:
                    for next_i in [max(i - 1, 0), min(i + 1, self.state_shape[0] - 1)]:
                        for next_j in [max(j - 1, 0), min(j + 1, self.state_shape[1] - 1)]:
                            if self.current_state[next_i, next_j] == self.agent_tile:
                                self.terminal = True
                            self.current_state[next_i, next_j] = self.lava_tile
                            lava_node = self.cord_node_key(next_i, next_j)
                            if lava_node not in self.lava_nodes:
                                self.lava_nodes.append(lava_node)

        # Check if path from agent to lava exists
        if not self.safe_from_lava or self.terminal:
            self.safe_from_lava = not self.has_path_to_lava()

        # Check if terminal
        return self.current_state.copy(), reward, self.terminal, None
