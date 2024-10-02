from environments.environment import Environment
from learning_agents.optionsagent import Option, OptionsAgent
from learning_agents.qlearningagent import QLearningAgent
from progressbar import print_progress_bar

import json
import networkx as nx
import numpy as np
import random as rand


class BetweennessOption(Option):

    def __init__(self, actions, goal_index,
                 initiation_func, termination_func,
                 alpha=0.9, epsilon=0.1, gamma=0.9):
        self.actions = actions
        self.goal_index = goal_index

        self.initiation_func = initiation_func
        self.terminating_func = termination_func

        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        self.policy = QLearningAgent(self.actions, self.alpha, self.epsilon, self.gamma)
        return


class BetweennessAgent(OptionsAgent):
    option_training_failure_reward = -1.0
    option_training_step_reward = -0.001
    option_training_success_reward = 1.0

    def __init__(self, actions, alpha, epsilon, gamma, state_transition_graph,
                 state_shape, state_dtype=int):
        self.actions = actions
        self.num_actions = len(self.actions)

        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        self.state_transition_graph = state_transition_graph

        self.state_shape = state_shape
        self.state_dtype = state_dtype

        self.current_option = None
        self.option_start_state = None
        self.total_option_reward = 0
        self.current_option_step = 0
        self.state_option_values = {}

        self.state_index_lookup = {state: node
                                   for node, state in nx.get_node_attributes(self.state_transition_graph,
                                                                             'state').items()}

        self.num_options = 0
        self.options = []
        self.option_initiation_lookup = {state: {} for state in self.state_index_lookup}
        return

    def copy_agent(self, copy_from):
        self.options = copy_from.options.copy()
        self.state_option_values = copy_from.state_option_values.copy()
        self.option_initiation_lookup = copy_from.option_initiation_lookup.copy()
        return

    def find_options(self, existing_stg_values=None,
                     stg_save_path=None):
        betweenness_values = nx.betweenness_centrality(self.state_transition_graph)

        if existing_stg_values is None:
            existing_stg_values = {node: {} for node in betweenness_values}

        for node in existing_stg_values:
            existing_stg_values[node]['betweenness'] = betweenness_values[node]

        for node in existing_stg_values:
            local_maxima_str = 'True'
            local_maxima = True
            for neighbour in self.state_transition_graph.neighbors(node):
                if neighbour == node:
                    continue
                betweenness_value = existing_stg_values[node]['betweenness']
                if existing_stg_values[node]['betweenness'] > betweenness_value:
                    local_maxima_str = 'False'
                    local_maxima = False
                    break

            existing_stg_values[node]['betweenness_local_maxima'] = local_maxima_str

            if local_maxima:
                option = BetweennessOption(self.actions, node,
                                           lambda s: self.option_initiation_function(s, node),
                                           lambda s: self.option_termination_function(s, node),
                                           self.alpha, self.epsilon, self.gamma)
                self.num_options += 1
                self.options.append(option)

        nx.set_node_attributes(self.state_transition_graph, existing_stg_values)

        # Adding primitive Options
        for action in self.actions:
            self.options.append(Option([action]))

        if stg_save_path is None:
            return self.state_transition_graph, existing_stg_values
        nx.write_gexf(self.state_transition_graph, stg_save_path)
        return self.state_transition_graph, existing_stg_values

    def get_available_options(self, state, possible_actions=None):
        available_options = []
        for i in range(self.num_options):
            option = self.options[i]
            if option.initiated(state):
                available_options.append(option)

        if possible_actions is None:
            available_options += self.options[self.num_options:]
            return available_options

        i = 0
        num_possible_actions = len(possible_actions)
        action_index = 0
        action = possible_actions[action_index]
        while i < self.num_actions:
            option = self.options[self.num_options + i]
            if option.actions[0] == action:
                available_options.append(option)
                action_index += 1
                if action_index >= num_possible_actions:
                    break
                action = possible_actions[action_index]
            i += 1

        return available_options

    def load(self, save_path):
        with open(save_path, 'r') as f:
            data = json.load(f)

        self.option_initiation_lookup = data['option initiation lookup']

        for index in data['options']:
            option_data = data['options'][index]
            node = option_data['goal_index']
            option = BetweennessOption(self.actions, node,
                                       lambda s: self.option_initiation_function(s, node),
                                       lambda s: self.option_termination_function(s, node),
                                       self.alpha, self.epsilon, self.gamma)
            option.policy.q_values = option_data['policy']
            self.options.append(option)
            self.num_options += 1
        for action in self.actions:
            option = Option(actions=[action])
            self.options.append(option)


        return

    def option_initiation_function(self, state, goal_index):
        state_str = np.array2string(np.ndarray.astype(state, dtype=self.state_dtype))

        try:
            can_initiate_str = self.option_initiation_lookup[state_str][goal_index]
            can_initiate = can_initiate_str == 'True'
        except KeyError:
            state_node = self.state_index_lookup[state_str]

            if state_node == goal_index:
                can_initiate = False
            else:
                can_initiate = nx.has_path(self.state_transition_graph, state_node, goal_index)

            can_initiate_str = 'False'
            if can_initiate:
                can_initiate_str = 'True'
            self.option_initiation_lookup[state_str][goal_index] = can_initiate_str

        return can_initiate

    def option_termination_function(self, state, goal_index):
        state_str = np.array2string(np.ndarray.astype(state, dtype=self.state_dtype))
        state_index = self.state_index_lookup[state_str]

        if state_index == goal_index:
            return True

        return not self.option_initiation_function(state, goal_index)

    def save(self, save_path):
        try:
            f = open(save_path, 'x')
            f.close()
        except FileExistsError:
            ()

        data = {'options': {}, 'option values': {}, 'option initiation lookup': {}}
        for i in range(self.num_options):
            option = self.options[i]
            data['options'][i] = {'policy': option.policy.q_values,
                                  'goal_index': option.goal_index}
        data['option values'] = self.state_option_values.copy()
        data['option initiation lookup'] = self.option_initiation_lookup.copy()

        with open(save_path, 'w') as f:
            json.dump(data, f)
        return

    def train_option(self, option: BetweennessOption,
                     training_timesteps: int,
                     environment: Environment,
                     all_actions_valid: bool=False,
                     progress_bar: bool=False):
        start_states = []
        for state_str in self.option_initiation_lookup:
            state = self.state_str_to_state(state_str)
            if self.option_initiation_function(state, option.goal_index) and (not environment.is_terminal(state)):
                start_states.append(state)

        done = True
        current_iterations = 0
        possible_actions = environment.possible_actions
        next_possible_actions = environment.possible_actions

        while current_iterations < training_timesteps:
            if progress_bar:
                print_progress_bar(current_iterations, training_timesteps,
                                   'Training Option to Subgoal ' + str(option.goal_index),
                                   'Complete')

            if done:
                state = rand.choice(start_states)
                state = environment.reset(state)
                if not all_actions_valid:
                    possible_actions = environment.get_possible_actions(state)

            action = option.choose_action(state, possible_actions)
            next_state, _, done, _ = environment.step(action)

            reward = self.option_training_step_reward
            next_state_index = self.state_index_lookup[np.array2string(next_state)]
            if next_state_index == option.goal_index:
                reward = self.option_training_success_reward
                done = True
            elif option.terminated(state):
                reward = self.option_training_failure_reward
                done = True

            if not all_actions_valid:
                next_possible_actions = environment.get_possible_actions(next_state)

            option.policy.learn(state, action, reward, next_state, done, next_possible_actions)

            state = next_state
            possible_actions = next_possible_actions
            current_iterations += 1

        return

    def train_option_value_iteration(self, option: BetweennessOption,
                                     theta: float, max_iterations: int,
                                     environment: Environment,
                                     all_actions_valid: bool=False,
                                     progress_bar: bool=False):
        def v(s):
            state_values = option.policy.get_action_values(s)
            return max(state_values.values())

        states = [state for state in self.option_initiation_lookup
                  if option.initiated(self.state_str_to_state(state))]
        node_state_lookup = nx.get_node_attributes(self.state_transition_graph, 'state')

        delta = np.inf
        possible_actions = environment.possible_actions
        current_iteration = 0
        while (theta < delta) and (current_iteration < max_iterations):
            if progress_bar:
                print_progress_bar(current_iteration, max_iterations,
                                   prefix='Training Option to subgoal ' + str(option.goal_index),
                                   suffix='Complete')

            delta = 0
            for state_str in states:
                state = self.state_str_to_state(state_str)
                state_node = self.state_index_lookup[state_str]

                if not all_actions_valid:
                    possible_actions = environment.get_possible_actions(state)

                v_value = v(state)

                for action in possible_actions:
                    state_action_value = 0
                    for successor_node in self.state_transition_graph[state_node]:
                        successor_state_str = node_state_lookup[successor_node]
                        successor_state = self.state_str_to_state(successor_state_str)

                        transition_prob = environment.get_transition_probability(state, action, successor_state)
                        if transition_prob <= 0:
                            continue

                        reward = self.option_training_step_reward
                        if successor_node == option.goal_index:
                            reward = self.option_training_success_reward
                        elif option.terminated(successor_state):
                            reward = self.option_training_failure_reward

                        successor_state_value = v(successor_state)

                        state_action_value += transition_prob * (reward + successor_state_value)

                    option.policy.q_values[state_str][action] = state_action_value

                delta = max(delta, abs(v_value - v(state)))
            current_iteration += 1

        return

    def train_options(self, environment: Environment,
                      maximum_iterations: int,
                      value_iteration: bool, theta: float=np.inf,
                      all_actions_valid: bool=False,
                      progress_bar: bool=False):
        if progress_bar:
            print("Training Betweenness Options")

        for i in range(self.num_options):
            if progress_bar:
                print()
                print("Option " + str(i + 1) + "/" + str(self.num_options))
            option = self.options[i]
            if value_iteration:
                self.train_option_value_iteration(option, theta, maximum_iterations, environment, all_actions_valid, progress_bar)
            else:
                self.train_option(option, maximum_iterations, environment, all_actions_valid, progress_bar)
        return
