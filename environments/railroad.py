from environments.environment import Environment

import numpy as np
import random as rand
from typing import Any


# TODO: Add implimentation for being given a tile to place, tile generation
# TODO: Change so can only place truly valid tiles
class RailRoad(Environment):
    failure_reward = 0.0
    step_reward = -0.001
    success_reward = 1.0

    default_width = 3
    default_height = 3

    default_stations = [(0, 0), (2, 2)]

    blank_tile = 0
    vert_line_tile = 1
    hor_line_tile = 2
    plus_tile = 3
    right_turn_tile = 4
    left_turn_tile = 5
    inv_right_turn_tile = 6
    inv_left_turn_tile = 7
    possible_tiles = [vert_line_tile, hor_line_tile,
                      plus_tile]

    tile_placement_rewards = {vert_line_tile: -0.001,
                              hor_line_tile: -0.001,
                              plus_tile: -0.005,
                              right_turn_tile: -0.001,
                              left_turn_tile: -0.001,
                              inv_right_turn_tile: -0.001,
                              inv_left_turn_tile: -0.001}

    tile_adjacency_key = {vert_line_tile: [[blank_tile, vert_line_tile, plus_tile, right_turn_tile, left_turn_tile],
                                           [blank_tile, vert_line_tile, plus_tile, inv_right_turn_tile,
                                            inv_left_turn_tile],
                                           [blank_tile, vert_line_tile, left_turn_tile, inv_left_turn_tile],
                                           [blank_tile, vert_line_tile, right_turn_tile, inv_right_turn_tile]],
                          hor_line_tile: [[blank_tile, hor_line_tile, inv_left_turn_tile, inv_right_turn_tile],
                                          [blank_tile, hor_line_tile, right_turn_tile, left_turn_tile],
                                          [blank_tile, hor_line_tile, plus_tile, right_turn_tile, inv_right_turn_tile],
                                          [blank_tile, hor_line_tile, plus_tile, left_turn_tile, inv_left_turn_tile]],
                          plus_tile: [[blank_tile, vert_line_tile, plus_tile, right_turn_tile, left_turn_tile],
                                      [blank_tile, vert_line_tile, plus_tile, inv_right_turn_tile, inv_left_turn_tile],
                                      [blank_tile, hor_line_tile, plus_tile, right_turn_tile, inv_right_turn_tile],
                                      [blank_tile, hor_line_tile, plus_tile, left_turn_tile, inv_left_turn_tile]]}

    tile_connection_key = {blank_tile: [],
                           vert_line_tile: [(0, -1), (0, 1)],
                           hor_line_tile: [(-1, 0), (1, 0)],
                           plus_tile: [(0, 1), (0, -1), (1, 0), (-1, 0)],
                           right_turn_tile: [(0, 1), (1, 0)],
                           left_turn_tile: [(0, 1), (-1, 0)],
                           inv_right_turn_tile: [(0, -1), (1, 0)],
                           inv_left_turn_tile: [(0, -1), (-1, 0)]}

    for tile in tile_connection_key:
        new_connections = []
        for connection in tile_connection_key[tile]:
            new_connections.append(np.array(connection))
        tile_connection_key[tile] = new_connections.copy()

    def __init__(self, width=default_width, height=default_height, stations=default_stations,
                 tile_generation='choice'):
        self.width = width
        self.height = height
        self.stations = stations
        self.num_stations = len(self.stations)
        self.tile_generation = tile_generation

        self.num_possible_tiles = len(self.possible_tiles)

        self.symmetry_functions = [lambda s: self.fliplr_state(s),
                                   lambda s: self.rot90_state(s, 1),
                                   lambda s: self.rot90_state(s, 2),
                                   lambda s: self.rot90_state(s, 3),
                                   lambda s: self.rot90_state(self.fliplr_state(s), 1),
                                   lambda s: self.rot90_state(self.fliplr_state(s), 2),
                                   lambda s: self.rot90_state(self.fliplr_state(s), 3)]

        self.terminal = True
        self.board = None

        self.action_lookup = {}
        self.num_possible_actions = 0
        for y in range(self.height):
            for x in range(self.width):
                for t in self.possible_tiles:
                    self.action_lookup[self.num_possible_actions] = {'x': x, 'y': y, 'tile': t}
                    self.num_possible_actions += 1
        self.possible_actions = list(range(self.num_possible_actions))

        self.environment_name = "railroad_" + self.tile_generation + "_" + str(self.width) + "x" + str(self.height)
        return

    def all_stations_connected(self):
        for i in range(self.num_stations):
            start_station = self.stations[i]
            for j in range(i + 1, self.num_stations):
                if not self.is_path(start_station, self.stations[j]):
                    return False

        return True

    def can_place_tile(self, x, y, tile, state=None):
        if state is None:
            state = self.board

        if (not (0 <= x < self.width)) or (not (0 <= y < self.height)):
            return False

        if not state[y, x] == self.blank_tile:
            return False

        adjacent_tiles = self.get_adjacent_tiles(x, y, state)

        for k in range(4):
            adj_tile = adjacent_tiles[k]
            needed_adj = self.tile_adjacency_key[tile][k]
            if adj_tile not in needed_adj:
                return False

        if (x, y) in self.stations:
            return True

        has_directions = []
        for out_direction in self.tile_connection_key[tile]:
            in_direction = -1 * out_direction
            adj_x = x + out_direction[0]
            adj_y = y + out_direction[1]
            adj_tile = self.blank_tile
            if 0 <= adj_x < self.width and 0 <= adj_y < self.height:
                adj_tile = state[adj_y, adj_x]
            has_in_direction = False
            for d in self.tile_connection_key[adj_tile]:
                if np.array_equal(d, in_direction):
                    has_in_direction = True
                    break
            if not self.tile_connection_key[adj_tile]:
                has_in_direction = 'BLANK'
            has_directions.append(has_in_direction)

        num_blank = 0
        for elm in has_directions:
            if elm == False:
                return False
            if elm == 'BLANK':
                num_blank += 1
        if num_blank == len(has_directions):
            return False
        return True

    def fliplr_state(self, state):
        if self.tile_generation == 'choice':
            return np.fliplr(state)

        flipped_state = state.copy()
        flipped_state[0:self.height, 0:self.width] = np.fliplr(flipped_state[0:self.height, 0:self.width])

    def get_adjacent_tiles(self, x, y, state=None):
        if state is None:
            state = self.board

        adjacent_tiles = []
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        for m in directions:
            new_x = x + m[0]
            new_y = y + m[1]
            if not (0 <= new_x < self.width and 0 <= new_y < self.height):
                adj_tile = self.blank_tile
            else:
                adj_tile = state[new_y, new_x]
            adjacent_tiles.append(adj_tile)
        return adjacent_tiles

    def get_current_state(self):
        return self.board.copy()

    def get_start_states(self):
        if self.tile_generation == 'choice':
            start_state = np.full((self.height, self.width), self.blank_tile)
            return [start_state]

        start_states = []
        start_state_template = np.full((self.width + 1, self.height + 1), self.blank_tile)
        for tile in self.possible_tiles:
            start_state = start_state_template.copy()
            start_state.itemset((self.height, self.width), tile)
            start_states.append(start_state)
        return start_states

    def get_possible_actions(self, state=None):
        if state is None:
            state = self.board

        possible_tiles = self.possible_tiles
        if self.tile_generation == 'random':
            possible_tiles = [state[self.height, self.width]]

        possible_actions = [action for action in self.possible_actions
                            if self.action_lookup[action]['tile'] in possible_tiles and
                            self.can_place_tile(self.action_lookup[action]['x'],
                                                self.action_lookup[action]['y'],
                                                self.action_lookup[action]['tile'],
                                                state)]

        return possible_actions

    def get_successor_states(self, state, probability_weights=False):
        if self.is_terminal(state):
            return [], []

        successor_states = []
        num_successor_states = 0
        weights = []

        tiles_to_place = self.possible_tiles
        if self.tile_generation == 'random':
            tiles_to_place = [state[self.height, self.width]]

        for tile in tiles_to_place:
            for j in range(self.height):
                for i in range(self.width):
                    if not self.can_place_tile(i, j, tile, state):
                        continue

                    successor = state.copy()
                    successor.itemset((j, i), tile)
                    successor_states.append(successor)
                    num_successor_states += 1

        weight = 1.0
        if probability_weights:
            weight = 1.0 / num_successor_states

        if self.tile_generation == 'choice':
            weights = [weight] * num_successor_states
            return successor_states, weights

        random_successor_states = []
        num_templates = num_successor_states
        for template in successor_states:
            tiles_to_terminal = 0
            num_successors_of_template = 0
            for tile in self.possible_tiles:
                successor = template.copy()
                successor.itemset((self.height, self.width),
                                  tile)
                if self.is_terminal(successor):
                    tiles_to_terminal += 1
                    continue
                random_successor_states.append(successor)
                num_successors_of_template += 1
            if tiles_to_terminal > 0:
                successor = template.copy()
                successor.itemset((self.height, self.width),
                                  self.blank_tile)
                random_successor_states.append(successor)
                num_successors_of_template += 1

            if not probability_weights:
                weights += ([1.0] * num_successors_of_template)
                continue

            non_terminal_weight = 1.0 / (num_templates * self.num_possible_tiles)
            if tiles_to_terminal <= 0:
                weights += ([non_terminal_weight] * num_successors_of_template)
                continue
            terminal_weight = tiles_to_terminal / (num_templates * self.num_possible_tiles)
            weights += (([non_terminal_weight] * (num_successors_of_template - 1)) +
                        [terminal_weight])

        return random_successor_states, weights

    def is_path(self, state, start, end):
        current_x = start[0]
        current_y = start[1]

        squares_checked = [(current_x, current_y)]

        def next_tiles_to_search(x, y):
            tile = state[y, x]
            next_tiles = []
            for direction in self.tile_connection_key[tile]:
                new_square = np.array([x, y]) + np.array(direction)
                if (not (0 <= new_square[0] < self.width)) or (not (0 <= new_square[1] < self.height)):
                    continue
                if (new_square[0], new_square[1]) in squares_checked:
                    continue
                if state[new_square[1], new_square[0]] == self.blank_tile:
                    continue
                next_tiles.append((new_square[0], new_square[1]))
            return next_tiles

        next_to_search = next_tiles_to_search(current_x, current_y)

        while not next_to_search == []:
            square = next_to_search.pop()
            current_x = square[0]
            current_y = square[1]

            if current_x == end[0] and current_y == end[1]:
                return True

            squares_checked.append((current_x, current_y))
            adjacent_tiles = next_tiles_to_search(current_x, current_y)
            next_to_search += adjacent_tiles

        return False

    def is_board_full(self, state):
        for j in range(self.height):
            for i in range(self.width):
                tile = state[j, i]
                if tile == self.blank_tile:
                    return False
        return

    def is_space_to_place(self, state, given_tiles=None):
        tiles_to_try = self.possible_tiles
        if given_tiles is not None:
            tiles_to_try = given_tiles
        can_place_tile = False
        for j in range(self.height):
            for i in range(self.width):
                if not state[j, i] == self.blank_tile:
                    continue

                for tile in tiles_to_try:
                    if self.can_place_tile(i, j, tile, state):
                        can_place_tile = True
                        break

                if can_place_tile:
                    return True
        return False

    def is_successful(self, state):
        # Check if all stations are connected
        stations_connected = True
        start_station = self.stations[0]
        for i in range(1, self.num_stations):
            stations_connected = self.is_path(state, start_station, self.stations[i])
            if not stations_connected:
                break
        if stations_connected:
            return True
        return

    def is_terminal(self, state):
        if self.is_successful(state):
            return True

        # If random and no available next tile
        if self.tile_generation == 'random' and state[self.height, self.width] == self.blank_tile:
            return True

        # Check if board is full of tiles
        if not self.is_space_to_place(state):
            return True

        try:
            next_tile = state[self.height, self.width]
            return not self.is_space_to_place(state, [next_tile])
        except IndexError:
            return False
        return False

    def reset(self) -> Any:
        if self.tile_generation == 'choice':
            self.board = np.full((self.width, self.height), self.blank_tile)
        elif self.tile_generation == 'random':
            self.board = np.full((self.width + 1, self.height + 1), self.blank_tile)
            self.board.itemset((self.height, self.width), rand.choice(self.possible_tiles))
        else:
            raise AttributeError(self.tile_generation + " is invalid tile generation method")

        self.terminal = False

        return self.board.copy()

    def rot90_state(self, state, num_rotations):
        if self.tile_generation == 'choice':
            return np.rot90(state, num_rotations)

        rotated_state = state.copy()
        rotated_state[0:self.height, 0:self.width] = np.rot90(rotated_state[0:self.height, 0:self.width],
                                                              num_rotations)
        return rotated_state

    def step(self, action) -> (Any, float, bool, Any):
        if action not in self.get_possible_actions():
            raise AttributeError("Invalid action for current state")

        action_dict = self.action_lookup[action]
        x = action_dict['x']
        y = action_dict['y']

        tile = action_dict['tile']
        self.board.itemset((y, x), tile)
        reward = self.tile_placement_rewards[tile]

        # set next tile to be blank
        if self.tile_generation == 'random':
            self.board.itemset((self.height, self.width), self.blank_tile)
            next_tile = rand.choice(self.possible_tiles)

        if self.is_successful(self.board):
            reward += self.success_reward
            self.terminal = True
        elif not self.is_space_to_place(self.board):
            reward += self.failure_reward
            self.terminal = True
        elif self.tile_generation == 'random':
            self.board.itemset((self.height, self.width), next_tile)
            if not self.is_space_to_place(self.board, [next_tile]):
                reward += self.failure_reward
                self.terminal = True
                self.board.itemset((self.height, self.width), self.blank_tile)

        next_state = self.board.copy()
        return next_state, reward, self.terminal, None
