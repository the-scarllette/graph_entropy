import numpy as np
from typing import Any

from environments.environment import Environment


class FlashPoint(Environment):

    start_x = 0
    start_y = 0

    north_action = 0
    south_aciton = 1
    east_action = 2
    west_action = 3
    north_putout_action = 4
    south_putout_aciton = 5
    east_putout_action = 6
    west_putout_action = 7

    possible_actions = [north_action, south_aciton, east_action, west_action,
                        north_putout_action, south_putout_aciton, east_putout_action, west_putout_action]

    failure_reward = -1.0
    step_reward = -0.01
    success_reward = 1.0

    empty_tile = 0
    fire_tile = 1
    agent_tile = 2
    person_tile = 3

    no_person_appeared = 0
    person_appeared = 1
    person_found = 2

    # start in a state where there is fire
    # cant move into fire
    # can put out adjacent fire
    # each turn person has chance of appearing
    # success if get to person and get them out of building
    # failure if fire spreads to person
    def __init__(self, height, width):
        self.height = height
        self.width = width

        self.start_fire_x = int(self.width) / 2
        self.start_fire_y = int(self.height) / 2

        self.current_state = None
        self.terminal = True
        return

    def get_start_states(self):
        start_state = np.full((self.height + 1, self.width + 1), self.empty_tile)
        start_state[self.start_y, self.start_x] = self.agent_tile
        start_state[self.start_fire_y, self.start_fire_x] = self.fire_tile
        start_state[self.height, self.width] = self.no_person_appeared
        return [start_state]

    def get_successor_states(self, state, probability_weights=False):

        return

    def is_terminal(self, state=None):
        if state is None:
            if self.terminal:
                raise AttributeError("Environment must not be terminal or provide a state")
            state = self.current_state

        if state[self.height, self.width] == self.person_found:
            return True

        agent_on_state = False
        person_on_state = False
        for x in range(self.width):
            for y in range(self.height):
                tile = state[y, x]
                if tile == self.agent_tile:
                    agent_on_state = True
                elif tile == self.person_tile:
                    person_on_state = True
                if agent_on_state and person_on_state:
                    break

        if not agent_on_state:
            return True

        if (state[self.height, self.width] == self.person_appeared) and (not person_on_state):
            return True

        return False

    def reset(self) -> Any:
        return None

    def step(self, action) -> (Any, float, bool, Any):
        return None, 0, False, None
