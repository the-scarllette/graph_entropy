import copy
import numpy as np

from progressbar import print_progress_bar

from scipy import sparse
from typing import Any


class Environment:

    encoded_state_len = 0

    possible_actions = None

    def __init__(self):
        self.start = None
        self.environment_name = ""
        self.options = []
        self.state_dtype = int
        return

    def generate_random_state(self):
        return self.reset()

    def get_adjacency_matrix(self, directed=True, probability_weights=False,
                             compressed_matrix=False,
                             symmetry_functions=[],
                             progress_bar=False):
        connected_states = {}
        state_indexes = {}
        all_states = []
        state_indexer = 0

        symmetry_functions = [lambda x: x] + symmetry_functions
        using_symmetry_functions = len(symmetry_functions) > 1

        states_to_check = self.get_start_states()
        num_states_to_check = len(states_to_check)
        iteration = 0
        total_states = num_states_to_check

        while num_states_to_check > 0:
            iteration += 1
            print_progress_bar(iteration, total_states, "Finding STG for " + self.environment_name + ":")

            state = states_to_check.pop().copy()
            num_states_to_check -= 1
            state_bytes = state.tobytes()

            try:
                # If already found successor states, skip state
                successor_states = connected_states[state_bytes]['states']
                continue
            except KeyError:
                # If successor states not found, find them
                successor_states, weights = self.get_successor_states(state,
                                                                      probability_weights=probability_weights)

            num_unfiltered_successor_states = len(successor_states)
            if using_symmetry_functions:
                filtered_successor_states = []
                filtered_weights = []
                for i in range(num_unfiltered_successor_states):
                    successor = successor_states[i]
                    if successor is None:
                        continue
                    filtered_successor_states.append(successor)
                    successor_weight = weights[i]
                    for j in range(i + 1, num_unfiltered_successor_states):
                        successor_to_check = successor_states[j]
                        if successor_to_check is None:
                            continue

                        for symmetry_function in symmetry_functions:
                            successor_after_func = symmetry_function(successor_to_check)
                            if np.array_equal(successor, successor_after_func):
                                successor_states[j] = None
                                if probability_weights:
                                    successor_weight += weights[j]
                                break
                    filtered_weights.append(successor_weight)
                successor_states = filtered_successor_states
                weights = filtered_weights

            all_states.append(state)

            state_indexes[state_bytes] = state_indexer
            state_indexer += 1

            successor_states_after_symmetry = []
            for state in successor_states:
                symmetry_found = False
                state_after_symmetry = state
                for symmetry_function in symmetry_functions:
                    symmetric_state = symmetry_function(state)

                    try:
                        _ = connected_states[symmetric_state.tobytes()]['states']
                        symmetry_found = True
                        state_after_symmetry = symmetric_state
                        break
                    except KeyError:
                        continue

                successor_states_after_symmetry.append(state_after_symmetry)
                if not symmetry_found:
                    states_to_check.append(state)
                    num_states_to_check += 1
                    total_states += 1

            connected_states[state_bytes] = {'states': successor_states_after_symmetry,
                                             'weights': weights}

        data_type = int
        if probability_weights:
            data_type = float
        if compressed_matrix:
            adj_matrix = sparse.lil_matrix((state_indexer, state_indexer), dtype=data_type)
        else:
            adj_matrix = np.zeros((state_indexer, state_indexer), dtype=data_type)
        state_num = 1
        for state in all_states:
            state_bytes = state.tobytes()
            i = state_indexes[state_bytes]

            successor_states = connected_states[state_bytes]['states']
            weights = connected_states[state_bytes]['weights']
            num_successor_states = len(successor_states)

            for j in range(num_successor_states):
                successor = successor_states[j]
                weight = 1.0
                if probability_weights:
                    weight = weights[j]
                successor_index = state_indexes[successor.tobytes()]

                adj_matrix[i, successor_index] = weight
                if not directed:
                    adj_matrix[successor_index, i] = weight
            state_num += 1

        if compressed_matrix:
            adj_matrix = adj_matrix.tocsr()

        return adj_matrix, all_states

    def get_current_state(self):
        return None

    def get_available_actions(self, state):
        return self.get_possible_actions(state)

    def get_possible_actions(self, state):
        return self.possible_actions

    def get_transition_probability(self, state, action, next_state):
        return 1.0

    def state_encoder(self, *states):
        return None

    def get_action_space(self):
        return self.possible_actions

    def get_state_space(self):
        state_array = np.load(self.environment_name + '_all_states.npy')
        state_space = list(state_array)
        return state_space

    def get_start_states(self):
        return []

    def get_successor_states(self, state, probability_weights=False):
        return [], []

    def is_terminal(self, state=None):
        return True

    def print_state(self, state=None):
        if state is None:
            if self.terminal:
                raise AttributeError("Either provide a state or print state while environment is not terminal.")
            state = self.current_state

        print(np.array2string(state))
        return

    def set_options(self, new_options, append=False):
        """
        Sets the set of options available in this environment.
        By default, replaces the current list of available options. If you wish to extend the
        list of currently avaialble options, set the `append` parameter to `True`.

        Args:
            new_options (List[BaseOption]): The list of options to make avaialble.
            append (bool, optional): Whether to append the new options to the current set of options. Defaults to False.
        """
        if not append:
            self.options = set(copy.copy(new_options))
        else:
            self.options.update(copy.copy(new_options))

    def step(self, action) -> (Any, float, bool, Any):
        return None, 0.0, False, None

    def reset(self, start_state=None) -> Any:
        return None
