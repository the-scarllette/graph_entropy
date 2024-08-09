import numpy as np
from typing import Any

from environments.environment import Environment


class KeyCard(Environment):

    default_width = 3
    default_height = 3
    default_key_cards = [(default_width - 1, 0), (0, default_height - 1)]

    goal_locations = [(0, 0)]
    no_key_needed = -1

    north_action = 0
    south_action = 1
    east_action = 2
    west_action = 3
    pickup_action = 4
    possible_actions = [north_action, south_action, east_action, west_action,
                        pickup_action]
    num_possible_actions = len(possible_actions)

    invalid_action_reward = -0.5
    step_reward = -0.1
    step_reward_no_key = -0.01
    success_reward = 1.0

    def __init__(self, width=None, height=None, key_cards=None, goal_reveal_prob=0.1):
        self.width = width
        if self.width is None:
            self.width = self.default_width
        self.height = height
        if self.height is None:
            self.height = self.default_height
        self.key_cards = key_cards
        if self.key_cards is None:
            self.key_cards = self.default_key_cards
        self.goal_reveal_prob = goal_reveal_prob

        self.num_keys = len(self.key_cards)
        self.current_state = None
        self.environment_name = 'keycard_' + str(self.width) + '_' + str(self.height)
        self.key_needed_index = self.num_keys + 2
        self.state_len = self.num_keys + 3
        self.terminal = True
        self.x_index = self.num_keys
        self.y_index = self.num_keys + 1
        return

    def get_start_states(self):
        start_state = np.full(self.state_len, 0)
        start_state[self.key_needed_index] = self.no_key_needed
        return [start_state]

    def get_successor_states(self, state, probability_weights=False):
        successor_states = []
        weights = []

        if self.is_terminal(state):
            return successor_states, weights

        # Move successors
        # Pickup successors
        # For every state, if key is not locked in yet potential for key needed being decided
        def add_successor_state(value, index, prob):
            if not probability_weights:
                prob = 1
            successor = state.copy()

            if not ((value is None) or (index is None)):
                successor[index] = value

            successor_states.append(successor)
            weights.append(prob)
            return

        stationary_actions = 0
        default_prob = 1/self.num_possible_actions

        # Move Successor States
        x = state[self.x_index]
        y = state[self.y_index]
        if 0 <= x - 1:
            add_successor_state(x - 1, self.x_index, default_prob)
        else:
            stationary_actions += 1
        if x + 1 < self.width:
            add_successor_state(x + 1, self.x_index, default_prob)
        else:
            stationary_actions += 1
        if 0 <= y - 1:
            add_successor_state(y - 1, self.y_index, default_prob)
        else:
            stationary_actions += 1
        if y + 1 < self.height:
            add_successor_state(y + 1, self.y_index, default_prob)
        else:
            stationary_actions += 1

        # Pickup Successor States
        can_pickup = False
        for key in range(self.num_keys):
            if state[key] == 1:
                continue
            key_location = self.key_cards[key]
            if x == key_location[0] and y == key_location[1]:
                can_pickup = True
                add_successor_state(1, key, default_prob)
                break

        if not can_pickup:
            stationary_actions += 1

        # Stationary Successor
        if stationary_actions > 0:
            add_successor_state(None, None, stationary_actions/self.num_possible_actions)

        # Key Card Needed being decided
        if state[self.key_needed_index] != self.no_key_needed:
            return successor_states, weights

        num_successor_states = len(successor_states)
        for i in range(num_successor_states):
            for key in range(self.num_keys):
                successor = successor_states[i].copy()
                successor[self.key_needed_index] = key
                successor_states.append(successor)

                successor_weight = 1
                if probability_weights:
                    successor_weight = weights[i] * self.goal_reveal_prob * (1/self.num_keys)
                    weights[i] = weights[i] * (1 - self.goal_reveal_prob)
                weights.append(successor_weight)

        return successor_states, weights

    def is_terminal(self, state=None):
        if state is None:
            state = self.current_state

        key_needed = state[self.key_needed_index]
        if key_needed == self.no_key_needed:
            return False

        agent_has_key = state[key_needed] == 1
        if not agent_has_key:
            return False

        agent_at_goal = (state[self.x_index], state[self.y_index]) in self.goal_locations
        if agent_at_goal:
            return True

        return False

    def reset(self) -> Any:

        return

    def step(self, action) -> (Any, float, bool, Any):
        return
