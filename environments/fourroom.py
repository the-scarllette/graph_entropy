import numpy as np
import random as rand
from typing import Any

from environments.environment import Environment


class FourRoom(Environment):

    goals = [(2, 2), (2, 8), (8, 8), (8, 2)]

    possible_actions = [0, 1, 2, 3]

    rooms = [{'corner': (0, 0),
              'width': 5,
              'height':5,
              'corridors': [(5, 2), (2, 5)]},
             {'corner': (0, 6),
              'width': 5,
              'height': 5,
              'corridors': [(2, 5), (5, 8)]},
             {'corner': (6, 5),
              'width': 5,
              'height': 6,
              'corridors': [(5, 8), (8, 4)]},
             {'corner': (6, 0),
              'width': 5,
              'height': 4,
              'corridors': [(8, 4), (5, 2)]}]

    height = width = 11

    step_reward = -0.1
    goal_reward = 1.0

    encoded_state_len = height + width

    def __init__(self):
        self.goal = None
        self.x = None
        self.y = None
        self.terminal = True
        return

    def code_to_state(self, state_code):
        x = ''
        y = ''
        state_code_len = len(state_code)
        i = 0
        while True:
            if state_code[i] == '/':
                i += 1
                break
            x += state_code[i]
            i += 1
        y = state_code[i]
        return {'x': int(x), 'y': int(y)}

    def get_successor_states(self, state, probability_weights=False):
        x = state[0]
        y = state[1]

        self_transition_actions = 0
        successors = []
        num_successors = 0

        to_check = [[x - 1, y], [x + 1, y], [x, y - 1], [x, y + 1]]

        for cord in to_check:
            rooms = self.get_rooms(cord[0], cord[1])

            if not rooms:
                self_transition_actions += 1
                continue
            successors.append(np.array(cord))
            num_successors += 1

        transition_probability = 1.0
        self_transition_probability = 1.0
        if probability_weights:
            num_actions = len(self.possible_actions)
            transition_probability = (num_actions - self_transition_actions) / (num_actions * num_successors)
            self_transition_probability = self_transition_actions / num_actions

        probability_weights = [transition_probability] * len(successors)

        if self_transition_actions > 0:
            successors.append(state)
            probability_weights.append(self_transition_probability)

        return successors, probability_weights

    def at_location(self, location: (int, int)) -> bool:
        if self.terminal:
            return False
        return location[0] == self.x and location[1] == self.y

    def get_start_states(self):
        return [np.array([0, 0])]

    def get_current_state(self):
        if self.terminal:
            return None
        return {'x': self.x, 'y': self.y}

    def get_rooms(self, x, y):
        return [room for room in self.rooms if self.in_room(room, x, y)]

    def get_str_state(self):
        if self.terminal:
            return None
        return str(self.x) + '/' + str(self.y)

    def index_to_state(self, index):
        state_y = index // self.width
        state_x = index - state_y * self.width
        return {'x': state_x, 'y': state_y}

    def in_room(self, room, x, y):
        return (x, y) in room['corridors'] or \
               ((room['corner'][0] <= x < room['corner'][0] + room['width']) and
                (room['corner'][1] <= y < room['corner'][1] + room['height']))

    def state_encoder(self, *states):
        encoded_states = []
        for state in states:
            to_add_x = [0.0] * self.width
            to_add_x[state['x']] = 1.0

            to_add_y = [0.0] * self.height
            to_add_y[state['y']] = 1.0

            encoded_states += to_add_x + to_add_y

        return np.array(encoded_states)

    def step(self, action, true_state=False) -> (Any, float, bool, Any):
        new_x = self.x
        new_y = self.y

        if action == 0: # N
            new_y += 1
        elif action == 1: # S
            new_y -= 1
        elif action == 2: # E
            new_x += 1
        elif action == 3: # W
            new_x -= 1
        else:
            raise AttributeError('Invalid Action')

        reward = self.step_reward
        rooms = self.get_rooms(new_x, new_y)
        if not rooms:
            return self.return_current_state(true_state=true_state), reward, False, None

        self.x = new_x
        self.y = new_y
        done = False
        info = None
        if self.at_location(self.goals[self.goal]):
            done = True
            reward += self.goal_reward
            info = {'success': True}

        return self.return_current_state(true_state=true_state), reward, done, info

    def reset(self, true_state=False) -> Any:
        self.terminal = False

        self.goal = rand.randint(0, len(self.rooms) - 1)

        start_room = rand.choice(self.rooms)
        room_corner = start_room['corner']
        start_location = rand.choice([(room_corner[0] + i, room_corner[1] + j)
                                      for i in range(start_room['width']) for j in range(start_room['height'])]
                                     + start_room['corridors'])
        self.x = start_location[0]
        self.y = start_location[1]
        return self.return_current_state(true_state)

    def return_current_state(self, true_state=False):
        if true_state:
            return self.get_current_state()
        return self.get_str_state()

    def valid_state(self, x, y):
        for room in self.rooms:
            if self.in_room(room, x, y):
                return True
        return False

