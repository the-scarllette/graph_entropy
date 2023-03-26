from environments.environment import Environment

import numpy as np


class Room:

    def __init__(self, width, height, anchor, doors, joining_rooms=[], directions=[], key_cards=None, master_key='Master'):
        self.width = width
        self.height = height
        self.anchor = anchor
        self.doors = doors
        self.num_doors = len(doors)
        self.joining_rooms = joining_rooms
        self.directions = directions
        self.key_cards = key_cards
        self.master_key = master_key
        return

    def in_room(self, x, y):
        return (0 <= x - self.anchor[0] < self.width) and (0 <= y - self.anchor[1] < self.height)

    def move(self, x_start, y_start, key_cards, x_move, y_move, one_use_cards):
        x_new = x_start
        y_new = y_start

        can_move = self.in_room(x_start + x_move, y_start + y_move)
        new_room = self

        if not can_move:
            new_room = None
            key_cards.append(None)
            for i in range(self.num_doors):
                door = self.doors[i]
                key_card_needed = None
                if self.key_cards is not None:
                    key_card_needed = self.key_cards[i]
                if ((x_start, y_start) == door) and\
                        (self.master_key in key_cards or key_card_needed in key_cards) and \
                        ((x_move, y_move) == self.directions[i]):
                    can_move = True
                    new_room = self.joining_rooms[i]
                    if one_use_cards:
                        try:
                            key_cards.remove(key_card_needed)
                        except ValueError:
                            ()
                    break
            key_cards.remove(None)

        if can_move:
            x_new += x_move
            y_new += y_move

        return can_move, new_room, x_new, y_new, key_cards

    def set_joining_rooms(self, to_set):
        self.joining_rooms = to_set
        return


class KeyCard(Environment):
    possible_actions = {0: (0, 1), 1: (0, -1), 2: (1, 0), 3: (-1, 0)}

    center_room = Room(4, 4, (4, 0), [(4, 1), (5, 3), (7, 1)], key_cards=['B', 'R', 'G'],
                       directions=[(-1, 0), (0, 1), (1, 0)])
    blue_room = Room(4, 4, (0, 0), [(3, 1)], joining_rooms=[center_room], key_cards=['B'],
                     directions=[(1, 0)])
    red_room = Room(4, 4, (4, 4), [(5, 4)], joining_rooms=[center_room], key_cards=['R'],
                    directions=[(0, -1)])
    green_room = Room(4, 4, (8, 0), [(8, 1)], joining_rooms=[center_room], key_cards=['G'],
                      directions=[(-1, 0)])
    center_room.set_joining_rooms([blue_room, red_room, green_room])

    default_rooms = [center_room, blue_room, red_room, green_room]

    default_key_cards = {'B': [(4, 0)], 'R': [(4, 3)], 'G': [(7, 3)], 'Master': [(7, 0)]}

    default_start = (5, 1)
    default_goal = (6, 6)

    '''

    mid_room = Room(2, 3, (0, 3), [(0, 3), (1, 3), (0, 5), (1, 5)], directions=[(0, -1), (0, -1), (0, 1), (0, 1)],
                    key_cards=['B', 'B', 'R', 'R'])
    start_room = Room(2, 3, (0, 0), [(0, 2), (1, 2)], directions=[(0, 1), (0, 1)],
                      key_cards=['B', 'B'], joining_rooms=[mid_room, mid_room])
    end_room = Room(2, 3, (0, 6), [(0, 6), (1, 6)], directions=[(0, -1), (0, -1)],
                    key_cards=['R', 'R'], joining_rooms=[mid_room, mid_room])
    mid_room.set_joining_rooms([start_room, start_room, end_room, end_room])

    default_rooms = [start_room, mid_room, end_room]
    default_key_cards = {'B': [(0, 1)], 'R': [(0, 4)], 'Master': [(1, 1)]}

    default_start = (0, 0)
    default_goal = (1, 8)
    '''

    step_reward = -0.1
    goal_reward = 1.0

    def __init__(self, rooms=default_rooms, key_cards=default_key_cards, start=default_start, goal=default_goal,
                 one_use_cards=True, max_key_hold=1):
        self.rooms = rooms
        self.key_cards = key_cards
        self.start = start
        self.goal = goal
        self.x = self.y = self.current_room = None
        self.one_use_cards = one_use_cards
        self.max_key_hold = 1
        self.key_cards_held = []

        self.terminal = True
        return

    def get_adjacency_matrix(self, start=default_start, directed=True):
        connected_states = {}

        all_states = [{'x': start[0],
                       'y': start[1],
                       'key_cards': []}]
        to_add = [{'x': start[0],
                   'y': start[1],
                   'key_cards': []}]

        def dict_to_str(d):
            str_state = str(d['x']) + '/' + str(d['y']) + '/'
            d['key_cards'].sort()
            for elm in d['key_cards']:
                str_state += elm
            return str_state

        while len(to_add) > 0:
            current_state = to_add.pop()
            successor_states = self.get_successor_states(current_state['x'], current_state['y'],
                                                         current_state['key_cards'].copy())

            connected_states[dict_to_str(current_state)] = successor_states

            for state in successor_states:
                if state not in all_states:
                    all_states.append(state)
                    to_add.append(state)

        num_states = len(all_states)
        adj_matrix = np.zeros((num_states, num_states))
        for i in range(num_states):
            state = all_states[i]
            connected = connected_states[dict_to_str(state)]
            for connected_state in connected:
                j = all_states.index(connected_state)
                adj_matrix.itemset((i, j), 1.0)
                if not directed:
                    adj_matrix.itemset((j, i), 1.0)

        return adj_matrix, all_states

    def get_current_state(self, str_state=True):
        if self.terminal:
            raise AttributeError("Environment is terminal")

        self.key_cards_held.sort()
        current_state = {'x': self.x,
                         'y': self.y,
                         'key_cards': self.key_cards_held.copy()}

        if not str_state:
            return current_state

        str_current_state = ""
        for value in list(current_state.values()):
            str_current_state += str(value) + '/'
        return str_current_state

    def get_keycards_at_cords(self, x, y):
        key_cards_found = []
        for key_card in self.key_cards:
            if (x, y) in self.key_cards[key_card]:
                key_cards_found.append(key_card)
        return key_cards_found

    def get_room_by_cords(self, x, y):
        for room in self.rooms:
            if room.in_room(x, y):
                return room
        return None

    def get_successor_states(self, x, y, key_cards):
        if x == 7 and y == 1 and key_cards == []:
            print('here')
        moves = list(KeyCard.possible_actions.values())
        room = self.get_room_by_cords(x, y)

        successor_states = []
        for move in moves:
            can_move, new_room, x_new, y_new, key_cards_remaining = room.move(x, y, key_cards.copy(), move[0], move[1],
                                                                              one_use_cards=self.one_use_cards)
            if can_move:
                key_cards_found = self.get_keycards_at_cords(x_new, y_new)
                if len(key_cards_remaining) < self.max_key_hold:
                    for key_card in key_cards_remaining:
                        if key_card not in key_cards_found and key_card is not None:
                            key_cards_found.append(key_card)
                else:
                    key_cards_found = key_cards_remaining.copy()
                key_cards_found.sort()
                successor_states.append({'x': x_new,
                                         'y': y_new,
                                         'key_cards': key_cards_found})

        return successor_states

    def reset(self):
        self.x = self.start[0]
        self.y = self.start[1]

        self.current_room = self.get_room_by_cords(self.x, self.y)

        self.key_cards_held = []
        self.terminal = False
        return self.get_current_state()

    def step(self, action):
        if self.terminal:
            raise AttributeError("Environment is terminal")

        try:
            move = KeyCard.possible_actions[action]
        except KeyError:
            raise KeyError("Invalid action")

        reward = self.step_reward

        x_move = move[0]
        y_move = move[1]

        move_done, self.current_room, self.x, self.y, remaining_cards = self.current_room.move(self.x, self.y,
                                                                                               self.key_cards_held,
                                                                                               x_move, y_move,
                                                                                               self.one_use_cards)
        self.key_cards_held = remaining_cards.copy()

        if move_done:
            pos = (self.x, self.y)
            if len(self.key_cards_held) < self.max_key_hold:
                for key_card in self.key_cards:
                    if pos in self.key_cards[key_card] and pos not in self.key_cards_held:
                        self.key_cards_held.append(key_card)

            if self.goal == pos:
                self.terminal = True
                reward += self.goal_reward

        return self.get_current_state(), reward, self.terminal, None
