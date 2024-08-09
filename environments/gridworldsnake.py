import numpy as np
import random as rand
from typing import Any

from environments.environment import Environment

#TODO: ALTER SO WE COUNT THE BODY, body has head 1 then 2, 3, 4... etc.

class GridworldSnake(Environment):
    north_action = 0
    south_action = 1
    east_action = 2
    west_action = 3

    possible_actions = [north_action, south_action, east_action, west_action]

    empty_square = 0
    body = 1
    head = 2
    end_tail = 4
    food = 3

    feed_reward = 1.0
    failure_reward = -1.0
    step_reward = -0.001

    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.current_state = None

        self.x = self.y = self.food_x = self.food_y = None
        self.tail = []
        self.tail_length = 0
        self.terminal = False
        return

    def get_start_states(self):
        start_states = []
        start_state_template = np.full((self.width, self.height), self.empty_square)

        for x in range(self.width):
            for y in range(self.height):
                for food_x in range(self.width):
                    for food_y in range(self.height):
                        if (x == food_x) and (y == food_y):
                            continue
                    start_state = start_state_template.copy()
                    start_state[x, y] = self.head
                    start_state[food_x, food_y] = self.food
                    start_states.append(start_state)

        return start_states

    def get_successor_states(self, state, probability_weights=False):
        if self.is_terminal(state):
            return [], []

        head_found = False
        for head_x in range(self.width):
            for head_y in range(self.height):
                if state[head_x, head_y] == self.head:
                    head_x = True
                    break
            if head_found:
                break

        tail = [(head_x, head_y)]
        tail_complete = False
        while not tail_complete:
            tail_end = tail[-1]
            tail_end_x = tail_end[0]
            tail_end_y = tail_end[1]
            for cord in [(tail_end_x + 1, tail_end_y), (tail_end_x - 1, tail_end_y),
                         (tail_end_x, tail_end_y + 1), (tail_end_x, tail_end_y - 1)]:
                if (not 0 <= cord[0] < self.width) or (not 0 <= cord[1] < self.height):
                    continue
                if state[c]

        return [], []

    def is_terminal(self, state=None):
        if state is None:
            state = self.current_state

        for x in range(self.width):
            for y in range(self.height):
                if state[x, y] == self.head:
                    for cord in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                        if (not 0 <= cord[0] < self.width) or (not 0 <= cord[1] < self.height):
                            continue
                        if state[cord[0], cord[1]] == self.empty_square:
                            return False
                    return False
        return True

    def reset(self) -> Any:
        self.x = rand.randint(0, self.width - 1)
        self.y = rand.randint(0, self.height - 1)
        self.tail = [(self.x, self.y)]
        self.tail_length = 1

        self.food_x = self.x
        self.food_y = self.y
        while (self.food_x == self.x) and (self.food_y == self.y):
            self.food_x = rand.randint(0, self.width - 1)
            self.food_y = rand.randint(0, self.height - 1)

        self.current_state = np.full((self.width, self.height), self.empty_square)

        self.current_state[self.x, self.y] = self.head

        return self.current_state

    def step(self, action) -> (Any, float, bool, Any):
        new_x = self.x
        new_y = self.y

        if action == self.north_action:
            new_y += 1
        elif action == self.south_action:
            new_y -= 1
        elif action == self.east_action:
            new_x += 1
        elif action == self.west_action:
            new_x -= 1
        else:
            raise ValueError(str(action) + " is not a valid action")

        reward = self.step_reward

        if new_x == self.food_x and new_y == self.food_y:
            reward = self.feed_reward
            self.tail.insert(0, (new_x, new_y))
            self.current_state[self.x, self.y] = self.body
            self.current_state[new_x, new_y] = self.head
            self.tail_length += 1

            while (self.food_x, self.food_y) in self.tail:
                self.food_x = rand.randint(0, self.width - 1)
                self.food_y = rand.randint(0, self.height - 1)
            self.current_state[self.food_x, self.food_y] = self.food

        else:
            tail_end = self.tail[-1]
            for i in range(1, self.tail_length):
                self.tail[i] = self.tail[i + 1]
            self.tail[0] = (new_x, new_y)
            self.current_state[tail_end[0], tail_end[1]] = self.empty_square

        self.current_state[self.tail[-1][0], self.tail[-1][0]] = self.end_tail

        if (not 0 <= new_x < self.width) or (not 0 <= new_y < self.height) or (
                (self.current_state[new_x, new_y] == self.tail)):
            self.terminal = True
            reward = self.failure_reward
        else:
            self.current_state[new_x, new_y] = self.head



        return self.current_state, reward, self.terminal, _
