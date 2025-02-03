import numpy as np
import random as rand
from typing import List, Tuple

from environments.environment import Environment

'''
actions:
0: North
1: South
2: East
3: West
4: pickup
5: putdown
6: fillup
'''


def code_to_state(code):
    values = []
    value = ''
    for c in code:
        if c == '/':
            values.append(int(value))
            value = ''
        else:
            value += c

    keys = ['taxi_x', 'taxi_y', 'fuel_level', 'passenger_location', 'passenger_destination']
    state = {keys[i]: values[i] for i in range(5)}
    return state


def is_terminal(state):
    return state['passenger_location'] == 5 or state['fuel_level'] < 0

class TaxiCab(Environment):

    possible_actions = [i for i in range(7)]

    no_passenger_index = 6

    fuel_station = (2, 1)
    stops = {0: (0, 0),
             1: (0, 4),
             2: (4, 4),
             3: (3, 0),
             no_passenger_index: (None, None)}
    num_stops = 4

    no_right = [(0, 0), (0, 1), (1, 3), (1, 4), (2, 0), (2, 1)]
    no_left = [(1, 0), (1, 1), (2, 3), (2, 4), (3, 0), (3, 1)]

    max_time = 500

    step_reward = 0.0
    success_reward = 2
    failure_reward = -2
    illegal_action_reward = -1

    def __init__(self, use_time=True, use_fuel=True,
                 arrival_probabilities=None,
                 one_hot_encoding=False,
                 hashable_states=False,
                 continuous=False):
        self.use_time = use_time
        self.use_fuel = use_fuel

        self.terminal = True

        self.time = None

        self.taxi_x = None
        self.taxi_y = None
        self.fuel_level = None

        self.passenger_loc = None  # 0 - 3 for each stop, 4 in taxi, 5 at destination, 6 no passenger
        self.passenger_destination = None # 0 - 3 for each stop, 6 no passenger
        self.possible_passenger_locations = [0, 1, 2, 3]
        self.possible_passenger_destinations = [0, 1, 2, 3]

        self.arrival_probabilities = arrival_probabilities
        self.using_arrival_probabilities = self.arrival_probabilities is not None
        if self.using_arrival_probabilities:
            if abs(sum(self.arrival_probabilities) - 1.0) > 0.001:
                raise ValueError("Arrival Probabilities must sum to 1")
            if len(self.arrival_probabilities) != self.num_stops + 1:
                raise ValueError("Must be an arrival probability for each stop and 1 for no passenger")
            self.arrival_probabilities = {location: arrival_probabilities[location]
                                          for location in self.possible_passenger_locations}
            self.arrival_probabilities[self.no_passenger_index] = arrival_probabilities[4]

            self.possible_passenger_locations = []
            self.arrival_probabilities_list = []
            for possible_location in self.arrival_probabilities:
                if self.arrival_probabilities[possible_location] > 0:
                    self.possible_passenger_locations.append(possible_location)
                    self.arrival_probabilities_list.append(self.arrival_probabilities[possible_location])

        if not self.use_fuel:
            self.possible_actions = [0, 1, 2, 3, 4, 5]

        self.no_right = TaxiCab.no_right
        self.no_left = TaxiCab.no_left

        self.current_state = None

        self.output_true_state = False

        self.environment_name = 'taxicab'
        if not self.use_fuel:
            self.environment_name += "_no_fuel"
        if self.using_arrival_probabilities:
            self.environment_name += '_arrival_probabilities'
            for prob in self.arrival_probabilities_list:
                self.environment_name += '_' + str(round(prob, 3))

        self.one_hot_encoding = one_hot_encoding
        self.state_shape = 4
        if self.one_hot_encoding:
            self.state_shape = 20
            if self.using_arrival_probabilities:
                self.state_shape += 2
            if self.use_fuel:
                self.state_shape += 12
        elif self.use_fuel:
            self.state_shape = 5

        self.state_dtype = int

        self.hashable_states = hashable_states

        self.continuous = continuous

        self.options = []
        return

    def generate_random_state(self):
        if self.use_fuel:
            return self.reset()
        x = rand.randint(0, 4)
        y = rand.randint(0, 4)
        passenger_location = rand.choice([0, 1, 2, 3, 4, 6])
        passenger_destination = 6

        if passenger_location != 6:
            passenger_destination = rand.randint(0, 3)

        state = np.array([x, y, passenger_location, passenger_destination])
        return state

    def get_current_state(self):
        if not self.hashable_states:
            return self.current_state.copy()
        return self.current_state

    def get_start_states(self):
        # location x: 0-4
        # location y: 0-4
        # passenger location: 0-3, 6
        # passenger destination: 0-3, 6
        # fuel level: 5-12

        start_states = []
        start_state_len = 4
        if self.use_fuel:
            start_state_len = 5
        start_state_template = np.full(start_state_len, 0)

        for x in range(5):
            start_state_template[0] = x
            for y in range(5):
                start_state_template[1] = y
                for passenger_location in self.possible_passenger_locations:
                    start_state_template[2] = passenger_location

                    if passenger_location == self.no_passenger_index:
                        start_state_template[3] = self.no_passenger_index
                        start_states.append(start_state_template.copy())
                        continue

                    for passenger_destination in self.possible_passenger_destinations:
                        start_state_template[3] = passenger_destination
                        start_states.append(start_state_template.copy())

        if not self.use_fuel:
            return start_states

        fuel_start_states = []
        for state in start_states:
            for fuel_level in range(5, 13):
                start_state = state.copy()
                start_state[4] = fuel_level
                fuel_start_states.append(start_state)
        return fuel_start_states

    def get_state_space(self):
        state_array = np.load(self.environment_name + '_all_states.npy')
        all_states = list(state_array)
        if not self.hashable_states:
            return all_states

        return [state.tostring() for state in all_states]

    def get_successor_states(self, state: np.ndarray, probability_weights: bool=False) ->(
            Tuple)[List[np.ndarray], List[float]]:
        successor_states = []
        weights = []

        stationary_actions = 0
        total_actions = 6

        taxi_x = state[0]
        taxi_y = state[1]
        passenger_location = state[2]
        passenger_destination = state[3]
        fuel_level = None
        state = state.copy()

        no_passenger_index = self.no_passenger_index
        possible_passenger_destinations = self.possible_passenger_destinations

        if self.use_fuel:
            total_actions = 7
            state[4] -= 1
            fuel_level = state[4]

        if passenger_location == 5:
            return successor_states, weights
        if self.use_fuel and fuel_level <= 0:
            return successor_states, weights
        if (not self.continuous) and self.is_terminal(state):
            return successor_states, weights

        def add_successor_state(index: int, new_value: int, weight: float) -> None:
            successor_state = state.copy()

            if (index is not None) and (new_value is not None):
                successor_state[index] = new_value

            successor_states.append(successor_state)

            if not probability_weights:
                weight = 1.0
            weights.append(weight)
            return
        if self.using_arrival_probabilities and (passenger_location == self.no_passenger_index):
            def add_successor_state(index, new_value, weight):
                successor_state_template = state.copy()
                if (index is not None) and (new_value is not None):
                    successor_state_template[index] = new_value
                for loc in self.possible_passenger_locations:
                    successor_state = successor_state_template.copy()
                    successor_state[2] = loc

                    if loc == no_passenger_index:
                        successor_state[3] = no_passenger_index
                        successor_weight = 1.0
                        if probability_weights:
                            successor_weight = weight * self.arrival_probabilities[no_passenger_index]
                        successor_states.append(successor_state)
                        weights.append(successor_weight)
                        continue

                    for des in possible_passenger_destinations:
                        successor_state = successor_state.copy()
                        successor_state[3] = des
                        successor_weight = 1.0
                        if probability_weights:
                            successor_weight = weight * self.arrival_probabilities[loc] * (1 / self.num_stops)
                        successor_states.append(successor_state)
                        weights.append(successor_weight)
                return

        # Move successor states
        base_cord = (state[0], state[1])
        if base_cord[1] < 4:
            add_successor_state(1, base_cord[1] + 1, 1/total_actions)
        else:
            stationary_actions += 1
        if 0 < base_cord[1]:
            add_successor_state(1, base_cord[1] - 1, 1/total_actions)
        else:
            stationary_actions += 1
        if (base_cord not in self.no_right) and (base_cord[0] < 4):
            add_successor_state(0, base_cord[0] + 1, 1/total_actions)
        else:
            stationary_actions += 1
        if (base_cord not in self.no_left) and (0 < base_cord[0]):
            add_successor_state(0, base_cord[0] - 1, 1/total_actions)
        else:
            stationary_actions += 1

        # Pickup successor state
        if passenger_location <= 3 and (taxi_x, taxi_y) == self.stops[passenger_location]:
            add_successor_state(2, 4, 1/total_actions)
        else:
            stationary_actions += 1

        # Putdown successor state
        if passenger_location == 4 and (taxi_x, taxi_y) == self.stops[passenger_destination]:
            if not self.continuous:
                successor_state = state.copy()
                successor_state[2] = 5
                successor_state[3] = 5
                successor_states.append(successor_state)
                weight = 1.0
                if probability_weights:
                    weight = 1 / total_actions
                weights.append(weight)
            elif self.arrival_probabilities:
                successor_state = state.copy()
                successor_state[2] = self.no_passenger_index
                successor_state[3] = self.no_passenger_index
                successor_states.append(successor_state)
                weight = 1.0
                if probability_weights:
                    weight = 1/total_actions
                weights.append(weight)
            else:
                for loc in self.possible_passenger_locations:
                    for des in possible_passenger_destinations:
                        successor_state = successor_state.copy()
                        successor_state[2] = loc
                        successor_state[3] = des
                        successor_states.append(successor_state)
                        weight = 1.0
                        if probability_weights:
                            weight = 1 / (total_actions * self.num_stops * self.num_stops)
                        weights.append(weight)
        else:
            stationary_actions += 1

        # Fillup successor state
        if self.use_fuel:
            if (taxi_x, taxi_y) == self.fuel_station:
                add_successor_state(4, 12, 1/total_actions)
            else:
                stationary_actions += 1

        # Finding probability weights - if using fuel not stationary, but fuel just decreases by 1
        add_successor_state(None, None, stationary_actions / total_actions)

        return successor_states, weights

    def get_state_str(self, state):
        taxi_x = state[0]
        taxi_y = state[1]
        passenger_location = state[2]

        state_array = [['1', '_', '_', '_', '2'],
                       ['_', '_', '_', '_', '_'],
                       ['_', '_', '_', '_', '_'],
                       ['_', '_', 'F', '_', '_'],
                       ['0', '_', '_', '3', '_']]

        state_array[4 - taxi_y][taxi_x] = 'T'
        if passenger_location < 4:
            passenger_cords = self.stops[passenger_location]
            state_array[4 - passenger_cords[1]][passenger_cords[0]] = 'P'

        print("Current State:")
        y = 4
        while y >= 0:
            to_print = ""
            for x in range(5):
                to_print += state_array[4 - y][x]
                wall = '.'
                if (x, y) in self.no_right:
                    wall = '|'
                to_print += wall
            print(to_print + '\n')
            y -= 1

        return np.array2string(np.array(state_array))

    def get_transition_probability(self, state, action, next_state):
        if self.is_state_terminal(state):
            return 0.0

        after_state = state.copy()
        if action <= 3:
            taxi_cord = (state[0], state[1])
            next_x = state[0]
            next_y = state[1]
            if action == 0:
                next_y += 1
            elif action == 1:
                next_y -= 1
            elif action == 2 and taxi_cord not in self.no_right:
                next_x += 1
            elif action == 3 and taxi_cord not in self.no_left:
                next_x -= 1

            if not (next_x < 0 or next_y < 0 or next_x > 4 or next_y > 4):
                after_state[0] = next_x
                after_state[1] = next_y
        elif action == 4:
            passenger_loc = state[2]
            if passenger_loc <= 3:
                passenger_cords = self.stops[passenger_loc]
                if state[0] == passenger_cords[0] and state[1] == passenger_cords[1]:
                    after_state[2] = 4
        elif action == 5:
            if state[2] == 4:
                passenger_destination_cords = self.stops[state[3]]
                if state[0] == passenger_destination_cords[0] and state[1] == passenger_destination_cords[1]:
                    after_state[2] = 5

        if (after_state[2] != self.no_passenger_index) or not self.using_arrival_probabilities:
            if np.array_equal(next_state, after_state):
                return 1.0
            return 0

        # after_state has no passenger appeared, so now checking arrival probabilities
        if next_state[2] in [4, 5] or next_state[0] != after_state[0] or next_state[1] != after_state[1]:
            return 0.0

        return self.arrival_probabilities[next_state[2]]

    def is_terminal(self, state):
        if self.hashable_states:
            try:
                state = np.frombuffer(state, dtype=self.state_dtype)
            except TypeError:
                state = self.get_state_space()[int(state)]
                return self.is_state_terminal(state)

        try:
            passenger_at_destination = state[2] == 5
        except IndexError:
            self.hashable_states = False
            state = self.get_state_space()[int(state)]
            self.hashable_states
            return self.is_state_terminal(state)

        if not self.use_fuel:
            return passenger_at_destination

        return (state[4] <= 0) or passenger_at_destination

    def reset(self, start_state=None):

        state_len = 4
        if self.use_fuel:
            state_len = 5
        if start_state is None:
            start_state = [None] * state_len
        elif self.hashable_states:
            start_state = np.frombuffer(start_state, dtype=self.state_dtype)

        def draw_from_start_state(currrent_value, index):
            if start_state[index] is None:
                return currrent_value
            return start_state[index]

        self.terminal = False

        self.time = 0

        self.taxi_x = rand.randint(0, 4)
        self.taxi_x = draw_from_start_state(self.taxi_x, 0)
        self.taxi_y = rand.randint(0, 4)
        self.taxi_y = draw_from_start_state(self.taxi_y, 1)

        if self.using_arrival_probabilities:
            self.passenger_loc = rand.choices(self.possible_passenger_locations,
                                              self.arrival_probabilities_list)[0]
            self.passenger_loc = draw_from_start_state(self.passenger_loc, 2)

            if self.passenger_loc == self.no_passenger_index:
                self.passenger_destination = self.no_passenger_index
            else:
                self.passenger_destination = rand.choice(self.possible_passenger_destinations)
                self.passenger_destination = draw_from_start_state(self.passenger_destination, 3)
                if self.passenger_destination == self.no_passenger_index:
                    raise ValueError("Cannot assign passenger destination to no location when they are"
                                     "at a location")
        else:
            self.passenger_loc = rand.randint(0, 3)
            self.passenger_loc = draw_from_start_state(self.passenger_loc, 2)
            self.passenger_destination = rand.randint(0, 3)
            self.passenger_destination = draw_from_start_state(self.passenger_destination, 3)

        if self.use_fuel:
            self.fuel_level = rand.randint(5, 12)
            self.fuel_level = draw_from_start_state(self.fuel_level, 4)

        self.update_state()
        return self.get_current_state()

    def step(self, action: int) -> np.ndarray:
        if self.terminal:
            raise AttributeError("Environment must be reset before calling step")

        if action not in self.possible_actions:
            raise ValueError("No valid action " + str(action))

        reward = self.step_reward

        next_passenger_location = None
        next_passenger_destination = None
        if self.using_arrival_probabilities and (self.passenger_loc == self.no_passenger_index):
            next_passenger_location = rand.choices(self.possible_passenger_locations,
                                                   self.arrival_probabilities_list)[0]
            if next_passenger_location != self.no_passenger_index:
                next_passenger_destination = rand.choice(self.possible_passenger_destinations)

        if self.use_fuel:
            self.fuel_level -= 1
        self.time += 1
        out_of_fuel = False
        if self.use_fuel:
            out_of_fuel = self.fuel_level <= 0
        if out_of_fuel or (self.time > self.max_time and self.use_time):
            reward += TaxiCab.failure_reward
            self.terminal = True
            self.update_state()
            return self.current_state, reward, True, {'success': False}
        if action <= 3:  # Move Action
            taxi_cord = (self.taxi_x, self.taxi_y)
            if (action == 2 and taxi_cord in self.no_right) or (
               (action == 3 and taxi_cord in self.no_left)):
                if next_passenger_destination is not None:
                    self.passenger_loc = next_passenger_location
                    self.passenger_destination = next_passenger_destination
                self.update_state()
                reward += self.illegal_action_reward
                return self.current_state, reward, False, None

            next_x = self.taxi_x
            next_y = self.taxi_y
            if action == 0:
                next_y += 1
            elif action == 1:
                next_y -= 1
            elif action == 2:
                next_x += 1
            else:
                next_x -= 1

            if not (next_x < 0 or next_y < 0 or next_x > 4 or next_y > 4):
                self.taxi_x = next_x
                self.taxi_y = next_y

            if next_passenger_destination is not None:
                self.passenger_loc = next_passenger_location
                self.passenger_destination = next_passenger_destination
            self.update_state()
            return self.current_state, reward, False, None

        if action == 4:  # Pickup Action
            can_pickup = False
            if self.passenger_loc < 4:
                passenger_cords = self.stops[int(self.passenger_loc)]
                can_pickup = passenger_cords[0] == self.taxi_x and passenger_cords[1] == self.taxi_y

            if can_pickup:
                self.passenger_loc = 4
            else:
                reward += self.illegal_action_reward

            if next_passenger_destination is not None:
                self.passenger_loc = next_passenger_location
                self.passenger_destination = next_passenger_destination
            self.update_state()
            return self.current_state, reward, False, None

        if action == 5:  # Putdown
            can_putdown = False
            if self.passenger_loc == 4:
                destination_cords = self.stops[int(self.passenger_destination)]
                can_putdown = destination_cords[0] == self.taxi_x and destination_cords[1] == self.taxi_y

            info = None
            if can_putdown:
                putdown_place = self.no_passenger_index
                if not self.continuous:
                    self.terminal = True
                    putdown_place = 5
                self.passenger_loc = putdown_place
                self.passenger_destination = putdown_place
                reward += self.success_reward
                info = {'success': True}
            else:
                reward += self.illegal_action_reward

            if next_passenger_destination is not None:
                self.passenger_loc = next_passenger_location
                self.passenger_destination = next_passenger_destination
            self.update_state()
            return self.current_state, reward, self.terminal, info

        if self.use_fuel and action == 6:  # Fillup
            can_fillup = self.taxi_at_cords(self.fuel_station)
            if can_fillup:
                self.fuel_level = 12
            else:
                reward += self.illegal_action_reward

            if next_passenger_destination is not None:
                self.passenger_loc = next_passenger_location
                self.passenger_destination = next_passenger_destination
            self.update_state()
            return self.current_state, reward, False, None

        return

    def taxi_at_cords(self, cord):
        return self.taxi_x == cord[0] and self.taxi_y == cord[1]

    def to_one_hot_encoding(self, state):
        one_hot_encoded_state = np.zeros(self.state_shape)
        feature_size_count = 0

        # Taxi X
        one_hot_encoded_state[int(state[0])] = 1.0
        feature_size_count += 5

        # Taxi Y
        one_hot_encoded_state[int(state[1] + feature_size_count)] = 1.0
        feature_size_count += 5

        # Passenger Location
        one_hot_encoded_state[int(state[2] + feature_size_count)] = 1.0
        feature_size_count += 6
        if self.using_arrival_probabilities:
            feature_size_count += 1

        # Passenger Destination
        passenger_destination = state[3]
        if self.passenger_destination == 6:
            passenger_destination = 4
        one_hot_encoded_state[int(passenger_destination + feature_size_count)] = 1.0

        if not self.use_fuel:
            return one_hot_encoded_state

        feature_size_count += 4
        if self.arrival_probabilities:
            feature_size_count += 1

        # Fuel Leve;
        one_hot_encoded_state[int(state[4] + feature_size_count)] = 1.0

        return one_hot_encoded_state

    def update_state(self):
        if self.use_fuel:
            self.current_state = np.array([self.taxi_x, self.taxi_y,
                                           self.passenger_loc, self.passenger_destination, self.fuel_level],
                                           dtype=float)
            return

        self.current_state = np.array([self.taxi_x, self.taxi_y,
                                       self.passenger_loc, self.passenger_destination],
                                       dtype=float)

        if self.one_hot_encoding:
            self.current_state = self.to_one_hot_encoding(self.current_state)

        if self.hashable_states:
            self.current_state = self.current_state.tostring()
        return
