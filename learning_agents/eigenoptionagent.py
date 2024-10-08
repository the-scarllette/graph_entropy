from learning_agents.optionsagent import Option, OptionsAgent
from learning_agents.qlearningagent import QLearningAgent
from progressbar import print_progress_bar

import json
import networkx as nx
import numpy as np
import os
import random as rand

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

    def __init__(self, actions, eigenvector, goal_index, terminate_action, initiation_func, alpha=0.9, epsilon=0.1, gamma=0.9):
        self.actions = None
        self.eigenvector = eigenvector
        self.terminate_action = terminate_action
        self.initiation_func = initiation_func
        self.terminating_func = None
        self.possible_actions = actions + [self.terminate_action]
        self.goal_index = goal_index
        self.policy = QLearningAgent(self.possible_actions, alpha, epsilon, gamma)
        return


class EigenOptionAgent(OptionsAgent):

    def __init__(self, alpha, epsilon, gamma, actions, state_dtype=int,
                 num_options=64):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        self.intra_option = False
        self.state_dtype = state_dtype

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
        self.index_to_state_lookup = {}
        return

    def choose_action(self, state, optimal_choice=False, possible_actions=None):
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

    def choose_option(self, state, no_random, possible_actions=None):
        self.current_option_index = super().choose_option(state, no_random, possible_actions)
        option = self.options[self.current_option_index]
        return option

    def copy_agent(self, copy_from):
        self.state_option_values = copy_from.state_option_values.copy()
        self.current_option = None
        self.current_option_index = None
        return

    def create_initiation_func(self, goal_index):
        def initiation_func(state):
            state_index = self.get_state_index(state)
            return nx.has_path(self.stg, str(state_index), str(goal_index))

        return initiation_func

    def find_options(self, environment=None, all_states=None, adjacency_matrix=None, stg=None, remove_weights=False,
                     print_goals=False):
        if environment is None and (adjacency_matrix is None or all_states is None):
            raise AttributeError("Either provide an environment or an environments adjacency matrix")
#
        if adjacency_matrix is None or all_states is None:
            adjacency_matrix, all_states = environment.get_adjacency_matrix(True, False, False)
        elif remove_weights:
            num_states = adjacency_matrix.shape[0]
            for i in range(num_states):
                for j in range(num_states):
                    if adjacency_matrix[i, j] > 0:
                        adjacency_matrix[i, j] = 1.0

        # Creating stg
        if stg is None:
            stg = nx.from_numpy_array(adjacency_matrix)
        self.stg = stg

        # Creating states lookup
        num_states = adjacency_matrix.shape[0]
        self.state_to_index_lookup = {np.array(all_states[i], dtype=self.state_dtype).tobytes():
                                          i for i in range(num_states)}
        self.index_to_state_lookup = {i: all_states[i] for i in range(num_states)}

        # Constructing D
        row_sums = np.sum(adjacency_matrix, axis=1)
        row_sums[np.where(row_sums == 0)] = 1
        root_reciprocal_row_sums = np.sqrt(row_sums)
        root_reciprocal_row_sums = np.reciprocal(root_reciprocal_row_sums)
        diagonal_row_sum = np.zeros(adjacency_matrix.shape)
        np.fill_diagonal(diagonal_row_sum, row_sums)
        root_reciprocal_diagonal = np.zeros(adjacency_matrix.shape)
        np.fill_diagonal(root_reciprocal_diagonal, root_reciprocal_row_sums)

        # Finding the laplacian matrix
        laplacian = root_reciprocal_diagonal @ (diagonal_row_sum - adjacency_matrix) @ root_reciprocal_diagonal

        # Getting the eigenvectors and eigenvalues of the laplacian matrix
        eigenvalues, eigenvectors = np.linalg.eig(laplacian)

        # Choosing the k (usually 64) smallest eigenvalues as options
        sorted_eigenvalue_indexes = np.argsort(eigenvalues, kind='mergesort')
        eigenoptions = eigenvectors[:, sorted_eigenvalue_indexes[:self.num_options]].real

        # Creating EigenOptions
        if print_goals:
            print("Eigenoption subgoals: ")
        for i in range(self.num_options):
            eigenvector = eigenoptions[:, i]
            goal_state_index = np.argmax(eigenvector)
            option = EigenOption(self.actions, eigenvector, goal_state_index, self.terminate_action,
                                 self.create_initiation_func(goal_state_index))
            self.options.append(option)

            if print_goals:
                print(str(self.index_to_state_lookup[goal_state_index]))

        # Adding Primitive Options
        for action in self.actions:
            option = Option(actions=[action])
            self.options.append(option)

        return

    def get_available_options(self, state, possible_actions=None):
        available_options = [i for i in range(self.num_options)
                             if self.options[i].initiated(state)]

        if possible_actions is None:
            available_options += [i for i in range(self.num_options, len(self.options))]
            return available_options

        available_options += [i for i in range(self.num_options, len(self.options))
                              if self.options[i].actions[0] in possible_actions]
        return available_options

    def get_state_index(self, state):
        return self.state_to_index_lookup[state.tobytes()]

    def learn(self, state, action, reward, next_state,
              terminal=None, next_state_possible_actions=None):
        self.total_option_reward += reward
        if not (terminal or self.current_option.terminated(next_state)):
            return

        try:
            self.current_option.policy.current_option = None
        except AttributeError:
            ()

        option_value = self.get_state_option_values(self.option_start_state)[self.current_option_index]
        all_next_options = []
        if not terminal:
            all_next_options = self.get_state_option_values(next_state)

        next_options = all_next_options
        if next_state_possible_actions is not None:
            next_options = self.get_available_options(next_state, next_state_possible_actions)

        if (not next_options) or (not all_next_options):
            next_option_values = [0.0]
        else:
            next_option_values = [all_next_options[option] for option in next_options]
        max_next_option = max(next_option_values)

        state_str = np.array2string(np.ndarray.astype(self.option_start_state, dtype=self.state_dtype))

        self.state_option_values[state_str][self.current_option_index] += self.alpha * \
                                                                    (self.total_option_reward +
                                                                     (self.gamma ** self.current_option_step) *
                                                                     max_next_option
                                                                     - option_value)
        self.current_option = None
        self.current_option_index = None
        self.option_start_state = None
        self.total_option_reward = 0
        return

    def load(self, save_path, all_states, adjacency_matrix, stg):
        self.stg = stg

        with open(save_path, 'r') as f:
            data = json.load(f)

        for i in range(self.num_options):
            option_data = data['options'][str(i)]
            eigenvector = np.frombuffer(eval(option_data['eigenvector']), dtype=self.state_dtype)
            goal_state_index = option_data['goal_index']
            option = EigenOption(self.actions, eigenvector, goal_state_index, self.terminate_action,
                                 self.create_initiation_func(goal_state_index))
            self.options.append(option)
        for action in self.actions:
            option = Option([action])
            self.options.append(option)

        self.state_option_values = data['option values'].copy()

        num_states = adjacency_matrix.shape[0]
        self.state_to_index_lookup = {np.array(all_states[i], dtype=self.state_dtype).tobytes():
                                      i for i in range(num_states)}
        self.index_to_state_lookup = {i: all_states[i] for i in range(num_states)}
        return

    def train_options(self, environment, training_steps, all_actions_possible=True, progress_bar=False):
        for i in range(self.num_options):
            option = self.options[i]
            if progress_bar:
                print("Training EigenOption " + str(i + 1) + "/" + str(self.num_options))

            terminal = True
            possible_actions = environment.possible_actions
            for total_steps in range(training_steps):
                if progress_bar:
                    print_progress_bar(total_steps, training_steps,
                                       prefix='Eigenoption Training: ', suffix='Complete')

                if terminal:
                    option_can_run = False
                    while not option_can_run:
                        state = environment.reset()
                        option_can_run = option.initiated(state)
                        state_index = self.get_state_index(state)

                    if not all_actions_possible:
                        possible_actions = environment.get_possible_actions()

                action = option.policy.choose_action(state, possible_actions=possible_actions)

                if action == self.terminate_action:
                    next_state = state
                    terminal = True
                else:
                    next_state, _, terminal, _ = environment.step(action)

                next_state_index = self.get_state_index(next_state)
                reward = option.eigenvector[next_state_index] - option.eigenvector[state_index]

                if not all_actions_possible:
                    possible_actions = environment.get_possible_actions()
                option.policy.learn(state, action, reward, next_state, terminal, possible_actions)

        return

    def save(self, save_path):
        try:
            f = open(save_path, 'x')
            f.close()
        except FileExistsError:
            ()

        data = {'options': {}, 'option values': {}}
        for i in range(self.num_options):
            option = self.options[i]
            data['options'][i] = {'policy': option.policy.q_values,
                                  'goal_index': int(option.goal_index),
                                  'eigenvector': str(option.eigenvector.tobytes())}

        data['option values'] = self.state_option_values.copy()

        with open(save_path, 'w') as f:
            json.dump(data, f)
        return

    def save_options(self, save_path):
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        for i in range(self.num_options):
            option = self.options[i]
            option.save(save_path)
        return
