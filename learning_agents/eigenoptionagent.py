import json
import numpy as np
import networkx as nx
import os
import random
from scipy import sparse
from typing import Callable, List, Tuple, Type

from environments.environment import Environment
from learning_agents.optionsagent import Option, OptionsAgent
from learning_agents.qlearningagent import QLearningAgent
from progressbar import print_progress_bar

# EigenOptions Method
# Get adjacency matrix A
# Get D: diagonal row-sum
# Get Laplacian matrix: L = D^(-1/2)(D - A)D^(-1/2)
# Get eigenvalues and corresponding eigenvectors
# Lowest k (=64) eigenvalues and their eigenvectors are the options
# For each option with eigenvector e:
#   Goal state is s where e[s] is max
#   Initiation function is states which can reach s
#   Termination is environment is terminal
#   Option actions are primitive actions + termination action
#   Option is trained using reward r(s`, s) = e[s`] - e[s]
# Meta controller has access to options and primitive actions


class EigenOption(Option):

    def __init__(self, actions: List[int], eigenvector: np.ndarray, goal_index: int,
                 terminate_action: int, initiation_func: Callable[[np.ndarray], bool],
                 alpha: float=0.9, epsilon: float=0.1, gamma: float=0.9):
        self.actions = None
        self.eigenvector = eigenvector
        self.terminate_action = terminate_action
        self.possible_actions = actions + [self.terminate_action]

        self.initiation_func = initiation_func
        self.terminating_func = None

        self.goal_index = goal_index
        self.policy = QLearningAgent(self.possible_actions, alpha, epsilon, gamma)
        return


class EigenOptionAgent(OptionsAgent):

    def __init__(self, adjacency_matrix: sparse.csr_matrix,
                 state_transition_graph: nx.MultiDiGraph,
                 alpha: float, epsilon: float, gamma: float, actions: List[int], state_dtype: Type, state_shape: Tuple[int, int],
                 num_options: int=64):
        self.adjacency_matrix = adjacency_matrix
        self.adjacency_matrix[adjacency_matrix.nonzero()] = 1.0

        self.num_states = self.adjacency_matrix.shape[0]
        self.state_transition_graph = state_transition_graph

        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        self.intra_option = False
        self.state_dtype = state_dtype
        self.state_shape = state_shape

        self.current_option = None
        self.current_option_index = None
        self.option_start_state = None
        self.total_option_reward = 0
        self.current_option_step = 0

        self.state_dtype = state_dtype
        self.stg = None

        self.state_option_values = {}
        self.intra_state_option_values = {}

        self.last_possible_actions = None

        self.actions = actions
        self.terminate_action = self.actions[-1] + 1

        self.options = []
        self.num_options = num_options

        self.state_to_index_lookup = {}

        self.initiation_lookup = {}
        return

    def choose_action(self, state: np.ndarray, optimal_choice: bool=False,
                      possible_actions: None | List[int]=None) -> int:
        if self.options is None:
            raise AttributeError("Options have not been found yet, run the 'find options' method first")

        self.last_possible_actions = possible_actions

        if self.current_option is None:
            self.current_option = self.choose_option(state, optimal_choice, possible_actions)
            if self.current_option is None:
                return None

        chosen_action = self.current_option.choose_action(state, possible_actions)
        self.current_option_step += 1

        if chosen_action == self.terminate_action:
            self.learn(state, chosen_action, 0, state, next_state_possible_actions=possible_actions)
            self.current_option = None
            self.current_option_index = None
            return self.choose_action(state, optimal_choice, possible_actions)

        return chosen_action

    def copy_agent(self, copy_from: 'EigenOptionAgent'):
        self.state_option_values = copy_from.state_option_values.copy()
        self.current_option = None
        self.current_option_index = None
        return

    def find_options(self, progress_bar: bool=False):
        laplacian = sparse.csgraph.laplacian(self.adjacency_matrix, True)
        _, eigenvectors = sparse.linalg.eigs(laplacian, self.num_options, which='SR')
        goal_indexes = np.argmax(eigenvectors, axis=0)

        for state_index in range(self.num_states):
            if progress_bar:
                print_progress_bar(state_index, self.num_states, "Finding options: ")
            distances = sparse.csgraph.dijkstra(self.adjacency_matrix, indices=state_index,
                                                unweighted=True, min_only=True)
            self.initiation_lookup[str(state_index)] = {str(goal_index): str(distances[goal_index] < np.inf)
                                                        for goal_index in goal_indexes}

        for i in range(self.num_options):
            eigenvector = eigenvectors[:, i]
            goal_state_index = goal_indexes[i]
            option = EigenOption(self.actions, eigenvector, goal_state_index, self.terminate_action,
                                 lambda s: self.option_initiation_function(s, goal_state_index))
            self.options.append(option)
        for action in self.actions:
            option = Option([action])
            self.options.append(option)

        return

    def get_available_options(self, state: np.ndarray, possible_actions: None | List[int]=None) -> List[int]:
        available_options = [str(i) for i in range(self.num_options)
                             if self.options[i].initiated(state)]

        if possible_actions is None:
            available_options += [str(i) for i in range(self.num_options, len(self.options))]
            return available_options

        available_options += [str(i) for i in range(self.num_options, len(self.options))
                              if self.options[i].actions[0] in possible_actions]
        
        return available_options

    def get_state_index(self, state: np.ndarray) -> int:
        state_str = self.state_to_state_str(state)
        try:
            index = self.state_to_index_lookup[state_str]
        except KeyError:
            for index, values in self.state_transition_graph.nodes(data=True):
                if values['state'] == state_str:
                    break
            index = int(index)
            self.state_to_index_lookup[state_str] = index
        return index

    def learn(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray,
              terminal: None | bool=None, next_state_possible_actions: None | List[int]=None):
        self.total_option_reward += reward
        if not (terminal or self.current_option.terminated(next_state)):
            return

        try:
            self.current_option.policy.current_option = None
        except AttributeError:
            ()

        option_value = self.get_state_option_values(self.option_start_state)[int(self.current_option_index)]
        all_next_options = []
        if not terminal:
            all_next_options = self.get_state_option_values(next_state)

        next_options = all_next_options
        if next_state_possible_actions is not None:
            next_options = self.get_available_options(next_state, next_state_possible_actions)

        if (not next_options) or (not all_next_options):
            next_option_values = [0.0]
        else:
            next_option_values = [all_next_options[int(option)] for option in next_options]
        max_next_option = max(next_option_values)

        state_str = np.array2string(np.ndarray.astype(self.option_start_state, dtype=self.state_dtype))

        self.state_option_values[state_str][int(self.current_option_index)] += self.alpha * \
                                                                    (self.total_option_reward +
                                                                     (self.gamma ** self.current_option_step) *
                                                                     max_next_option
                                                                     - option_value)
        self.current_option = None
        self.current_option_index = None
        self.option_start_state = None
        self.total_option_reward = 0
        return

    def load(self, save_path: str):
        with open(save_path, 'r') as f:
            data = json.load(f)

        self.initiation_lookup = data['initiation lookup'].copy()

        for i in range(self.num_options):
            option_data = data['options'][str(i)]
            eigenvector = np.frombuffer(eval(option_data['eigenvector']), dtype=self.state_dtype)
            goal_state_index = option_data['goal_index']
            option = EigenOption(self.actions, eigenvector, goal_state_index, self.terminate_action,
                                 lambda s: self.option_initiation_function(s, goal_state_index))
            option.policy.q_values = option_data['policy'].copy()
            self.options.append(option)
        for action in self.actions:
            option = Option([action])
            self.options.append(option)

        self.state_option_values = data['option values'].copy()
        return

    def option_initiation_function(self, state: np.ndarray, goal_state_index: int) -> bool:
        state_index = self.get_state_index(state)
        return self.initiation_lookup[str(state_index)][str(goal_state_index)] == 'True'

    def train_option(self, environment: Environment, option: EigenOption, training_steps: int,
                     possible_start_states: None | List[np.ndarray]=None,
                     all_actions_valid: bool=False, progress_bar: bool=False):
        terminal = True
        possible_actions = environment.possible_actions
        start_states = []

        if possible_start_states is None:
            possible_start_states = environment.get_start_states()
        for state in possible_start_states:
            if self.option_initiation_function(state, option.goal_index):
                start_states.append(state)

        for total_steps in range(training_steps):
            if progress_bar:
                print_progress_bar(total_steps, training_steps,
                                   prefix='Eigenoption Training: ', suffix='Complete')

            if terminal:
                state = environment.reset(random.choice(start_states))
                state_index = self.get_state_index(state)
                if not all_actions_valid:
                    possible_actions = environment.get_possible_actions(state)

            action = option.policy.choose_action(state, possible_actions=possible_actions)

            if action == self.terminate_action:
                next_state = state
                terminal = True
            else:
                next_state, _, terminal, _ = environment.step(action)

            next_state_index = self.get_state_index(next_state)
            reward = option.eigenvector[next_state_index] - option.eigenvector[state_index]

            if not all_actions_valid:
                possible_actions = environment.get_possible_actions(next_state)
            option.policy.learn(state, action, reward, next_state, terminal, possible_actions)
            state = next_state

        return

    def train_options(self, environment: Environment, training_steps: int,
                      all_actions_valid: bool=True, progress_bar: bool=False):
        possible_start_states = environment.get_start_states()
        for i in range(self.num_options):
            if progress_bar:
                print("Training EigenOption " + str(i + 1) + "/" + str(self.num_options))
            option = self.options[i]
            self.train_option(environment, option, training_steps, possible_start_states,
                              all_actions_valid, progress_bar)

        return

    def save(self, save_path: str):
        try:
            f = open(save_path, 'x')
            f.close()
        except FileExistsError:
            ()

        data = {'options': {}, 'option values': {}, 'initiation lookup': self.initiation_lookup}
        for i in range(self.num_options):
            option = self.options[i]
            data['options'][i] = {'policy': option.policy.q_values,
                                  'goal_index': int(option.goal_index),
                                  'eigenvector': str(option.eigenvector.tobytes())}

        data['option values'] = self.state_option_values.copy()

        with open(save_path, 'w') as f:
            json.dump(data, f)
        return
