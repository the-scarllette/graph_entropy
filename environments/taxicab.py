import numpy as np
import random as rand

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

    fuel_station = (2, 1)
    stops = [(0, 0), (0, 4), (4, 4), (3, 0)]

    no_right = [(0, 0), (0, 1), (1, 3), (1, 4), (2, 0), (2, 1)]
    no_left = [(1, 0), (1, 1), (2, 3), (2, 4), (3, 0), (3, 1)]

    max_time = 100

    step_reward = 0.0
    success_reward = 2
    failure_reward = -2
    illegal_action_reward = -1

    def __init__(self, use_time=True):
        self.use_time = use_time

        self.terminal = True

        self.time = None

        self.taxi_x = None
        self.taxi_y = None
        self.fuel_level = None

        self.passenger_loc = None  # 0 - 3 for each stop, 4 in taxi
        self.passenger_destination = None

        self.no_right = TaxiCab.no_right
        self.no_left = TaxiCab.no_left

        self.current_state = None

        self.output_true_state = False
        return

    def get_current_state(self):
        return self.current_state.copy()

    def get_start_states(self):
        # location x: 0-4
        # location y: 0-4
        # fuel level: 5-12
        # passenger location: 0-3
        # passenger destination: 0-3

        start_states = []
        start_state_template = np.full(5, 0)

        for x in range(5):
            start_state_template[0] = x
            for y in range(5):
                start_state_template[1] = y
                for fuel_level in range(5, 13):
                    start_state_template[2] = fuel_level
                    for passenger_location in range(4):
                        start_state_template[3] = passenger_location
                        for passenger_destination in range(4):
                            start_state_template[4] = passenger_destination
                            start_states.append(start_state_template.copy())

        return start_states

    def get_successor_states(self, state):
        successor_states = []

        taxi_x = state[0]
        taxi_y = state[1]
        fuel_level = state[2]
        passenger_location = state[3]
        passenger_destination = state[4]

        fuel_level -= 1

        if (fuel_level < 0) or (passenger_location == 5):
            return successor_states

        def add_successor_state(index, new_value):
            successor_state = state.copy()
            successor_state[index] = new_value
            successor_states.append(successor_state)
            return

        # Move successor states
        for i in range(2):
            base_coordinate = state[i]
            for new_cord in range(base_coordinate - 1, base_coordinate + 2):
                if (new_cord == base_coordinate) or (not (0 <= new_cord <= 4)):
                    continue
                add_successor_state(i, new_cord)

        # Pickup successor state
        if passenger_location <= 3 and (taxi_x, taxi_y) == self.stops[passenger_location]:
            add_successor_state(3, 4)

        # Putdown successor state
        if passenger_location == 4 and (taxi_x, taxi_y) == self.stops[passenger_destination]:
            add_successor_state(3, 5)

        # Fillup successor state
        if (taxi_x, taxi_y) == self.fuel_station:
            add_successor_state(2, 12)

        return successor_states

    def get_state_str(self, state):
        taxi_x = state[0]
        taxi_y = state[1]
        passenger_location = state[3]

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

    def reset(self, true_state=False):
        self.terminal = False

        self.time = 0

        self.taxi_x = rand.randint(0, 4)
        self.taxi_y = rand.randint(0, 4)
        self.fuel_level = rand.randint(5, 12)

        self.passenger_loc = rand.randint(0, 3)
        self.passenger_destination = rand.randint(0, 3)

        self.update_state()
        if true_state:
            return self.current_state.copy()
        return np.array2string(self.current_state)

    def step(self, action, true_state=False):
        if self.terminal:
            raise AttributeError("Environment must be reset before calling step")

        if action not in TaxiCab.possible_actions:
            raise ValueError("No valid action " + str(action))

        reward = TaxiCab.step_reward
        self.fuel_level -= 1
        self.time += 1
        if self.fuel_level < 0 or (self.time > self.max_time and self.use_time):
            reward += TaxiCab.failure_reward
            self.terminal = True
            self.update_state()
            if true_state:
                to_output = self.current_state.copy()
            else:
                to_output = np.array2string(self.current_state)
            return to_output, reward, True, {'success': False}
        if action <= 3:
            taxi_cord = (self.taxi_x, self.taxi_y)
            if (action == 2 and taxi_cord in self.no_right) or (
               (action == 3 and taxi_cord in self.no_left)):
                self.update_state()
                if true_state:
                    to_output = self.current_state.copy()
                else:
                    to_output = np.array2string(self.current_state)
                return to_output, reward, False, None

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
            self.update_state()
            if true_state:
                to_output = self.current_state.copy()
            else:
                to_output = np.array2string(self.current_state)
            return to_output, reward, False, None

        if action == 4:  # Pickup Action
            can_pickup = False
            if self.passenger_loc < 4:
                passenger_cords = self.stops[self.passenger_loc]
                can_pickup = passenger_cords[0] == self.taxi_x and passenger_cords[1] == self.taxi_y

            if can_pickup:
                self.passenger_loc = 4
            else:
                reward += TaxiCab.illegal_action_reward
            self.update_state()
            if true_state:
                to_output = self.current_state.copy()
            else:
                to_output = np.array2string(self.current_state)
            return to_output, reward, False, None

        if action == 5:  # Putdown
            can_putdown = False
            if self.passenger_loc == 4:
                destination_cords = self.stops[self.passenger_destination]
                can_putdown = destination_cords[0] == self.taxi_x and destination_cords[1] == self.taxi_y

            info = None
            if can_putdown:
                self.passenger_loc = 5
                reward += TaxiCab.success_reward
                self.terminal = True
                info = {'success': True}
            else:
                reward += TaxiCab.illegal_action_reward
            self.update_state()
            if true_state:
                to_output = self.current_state.copy()
            else:
                to_output = np.array2string(self.current_state)
            return to_output, reward, self.terminal, info

        if action == 6:  # Fillup
            can_fillup = self.taxi_at_cords(self.fuel_station)
            if can_fillup:
                self.fuel_level = 12
            else:
                reward += TaxiCab.illegal_action_reward
            self.update_state()
            if true_state:
                to_output = self.current_state.copy()
            else:
                to_output = np.array2string(self.current_state)
            return to_output, reward, False, None

        return

    def taxi_at_cords(self, cord):
        return self.taxi_x == cord[0] and self.taxi_y == cord[1]

    def update_state(self):
        self.current_state = np.array([self.taxi_x, self.taxi_y,
                                       self.fuel_level, self.passenger_loc, self.passenger_destination])
        return
