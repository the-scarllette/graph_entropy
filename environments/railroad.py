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

        if (not(0 <= x < self.width)) or (not (0 <= y < self.height)):
            return False

        if not state[y, x] == self.blank_tile:
            return False

        adjacent_tiles = self.get_adjacent_tiles(x, y, state)

        can_place = True
        for k in range(4):
            adj_tile = adjacent_tiles[k]
            needed_adj = self.tile_adjacency_key[tile][k]
            if adj_tile not in needed_adj:
                return False

        if (x, y) in self.stations:
            return True

        tile_in = True
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

    def get_adjacency_matrix(self, directed=True):
        all_states = self.get_start_states()

        connected_states = {}

        to_add = [s.copy() for s in all_states]

        def get_equivalent_states(to_get):
            equiv = [to_get]
            return equiv

        while len(to_add) > 0:
            current_state = to_add.pop()
            successor_states = self.get_successor_states(current_state)

            connected_states[np.array2string(current_state)] = successor_states

            for state in successor_states:
                in_all_states = False

                to_check = get_equivalent_states(state)
                for s in all_states:
                    for check in to_check:
                        if np.array_equal(check, s):
                            in_all_states = True
                            break
                    if in_all_states:
                        break
                if not in_all_states:
                    all_states.append(state.copy())
                    to_add.append(state.copy())

        # my name's Scarllette and i love commenting my code ;)
        num_states = len(all_states)
        adj_matrix = np.zeros((num_states, num_states))
        for i in range(num_states):
            state = all_states[i]
            connected = connected_states[np.array2string(state)]
            for connected_state in connected:
                j = 0
                to_check = get_equivalent_states(connected_state)
                connection_found = False
                while j < num_states and not connection_found:
                    for s in to_check:
                        if np.array_equal(s, all_states[j]):
                            connection_found = True
                            j -= 1
                            break
                    j += 1
                if i == j:
                    continue
                adj_matrix.itemset((i, j), 1.0)
                if not directed:
                    adj_matrix.itemset((j, i), 1.0)
        return adj_matrix, all_states

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

    def get_current_state(self, true_state=False):
        if true_state:
            return self.board.copy()
        return np.array2string(self.board)

    def get_start_states(self):
        if self.tile_generation == 'choice':
            start_state = np.full((self.width, self.height), self.blank_tile)
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

    def get_successor_states(self, state):
        successor_states = []

        if (self.tile_generation == 'random' and state[self.height, self.width] == self.blank_tile) or\
                self.is_terminal(state):
            return successor_states

        for i in range(self.width):
            for j in range(self.height):
                current_tile = state[j, i]
                if not current_tile == self.blank_tile:
                    continue

                adjacent_tiles = self.get_adjacent_tiles(i, j, state)
                is_station = (i, j) in self.stations
                if all([self.blank_tile == t for t in adjacent_tiles]) and not is_station:
                    continue

                to_place = []
                if self.tile_generation == 'choice':
                    poss_tiles = self.possible_tiles
                elif self.tile_generation == 'random':
                    poss_tiles = [state[self.height, self.width]]

                to_place = [tile for tile in poss_tiles
                            if self.can_place_tile(i, j, tile, state)]

                if self.tile_generation == 'choice':
                    for tile in to_place:
                        successor = state.copy()
                        successor.itemset((j, i), tile)
                        successor_states.append(successor)
                elif self.tile_generation == 'random':
                    for tile in to_place:
                        for next_tile in self.possible_tiles:
                            successor = state.copy()
                            successor.itemset((j, i), tile)
                            successor.itemset((self.height, self.width), next_tile)
                            if self.is_terminal(successor):
                                successor.itemset((self.height, self.width), self.blank_tile)
                            successor_states.append(successor)

                '''
                connecting_moves = self.tile_connection_key[tile]
                for move in connecting_moves:
                    new_i = i + move[0]
                    new_j = j + move[1]
                    if not (0 <= new_i < self.width and 0 <= new_j < self.height):
                        continue
                    if not (state[new_i, new_j] == self.blank_tile):
                        continue

                    reverse_move = move * -1
                    possible_tiles = []
                    for key in self.possible_tiles:
                        for m in self.tile_connection_key[key]:
                            if np.array_equal(m, reverse_move):
                                possible_tiles.append(key)
                                break
                    '''

        return successor_states

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

        # Check if board is full of tiles
        if not self.is_space_to_place(state):
            return True

        try:
            next_tile = state[self.height, self.width]
            return not self.is_space_to_place(state, [next_tile])
        except IndexError:
            return False
        return False

    def reset(self, true_state=False) -> Any:
        if self.tile_generation == 'choice':
            self.board = np.full((self.width, self.height), self.blank_tile)
        elif self.tile_generation == 'random':
            self.board = np.full((self.width + 1, self.height + 1), self.blank_tile)
            self.board.itemset((self.height, self.width), rand.choice(self.possible_tiles))
        else:
            raise AttributeError(self.tile_generation + " is invalid tile generation method")

        self.terminal = False

        if true_state:
            return self.board.copy()
        return np.array2string(self.board)

    def step(self, action, true_state=True) -> (Any, float, bool, Any):
        if action not in self.get_possible_actions():
            raise AttributeError("Invalid action for current state")

        action_dict = self.action_lookup[action]
        x = action_dict['x']
        y = action_dict['y']

        tile = action_dict['tile']
        self.board.itemset((y, x), tile)
        reward = self.step_reward

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
        if not true_state:
            next_state = np.array2string(next_state)
        return next_state, reward, self.terminal, None
