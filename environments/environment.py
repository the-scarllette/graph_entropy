import numpy as np
from typing import Any


class Environment:

    encoded_state_len = 0

    possible_actions = None

    def __init__(self):
        self.start = None
        return

    def get_current_state(self):
        return

    def get_adjacency_matrix(self, directed=True):
        connected_states = {}

        start_states = self.get_start_states()
        all_states = []
        to_add = []
        for state in start_states:
            all_states.append(state.copy())
            to_add.append(state.copy())

        while len(to_add) > 0:
            current_state = to_add.pop()
            successor_states = self.get_successor_states(current_state)

            connected_states[np.array2string(current_state)] = [np.array2string(s) for s in successor_states]

            for state in successor_states:
                in_all_states = False
                for s in all_states:
                    if np.array_equal(state, s):
                        in_all_states = True
                        break
                if not in_all_states:
                    all_states.append(state.copy())
                    to_add.append(state.copy())

        # my name's Scarllette and i love commenting my code ;)
        all_states_str = [np.array2string(s) for s in all_states]
        num_states = len(all_states)
        adj_matrix = np.zeros((num_states, num_states))
        for i in range(num_states):
            state = all_states[i]
            connected = connected_states[np.array2string(state)]
            for connected_state in connected:
                j = 0
                j = all_states_str.index(connected_state)
                if i == j:
                    continue
                adj_matrix.itemset((i, j), 1.0)
                if not directed:
                    adj_matrix.itemset((j, i), 1.0)
        return adj_matrix, all_states

    def state_encoder(self, *states):
        return None

    def get_start_states(self):
        return []

    def get_successor_states(self, state):
        return []

    def step(self, action, true_state=False) -> (Any, float, bool, Any):
        return None, 0.0, False, None

    def reset(self, true_state=False) -> Any:
        return None
