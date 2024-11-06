import networkx as nx
import numpy as np
import random as rand
from scipy import sparse
from typing import Callable, Dict, List, Type

from environments.environment import Environment
from optionsagent import Option, OptionsAgent
from learningagent import LearningAgent
from qlearningagent import QLearningAgent
from progressbar import print_progress_bar


class PreparednessOption(Option):

    def __init__(self, actions: List[Option] | List[int], start_node: None | str, end_node: None | str,
                 start_state_str: None | str, end_state_str: str,
                 hierarchy_level: int,
                 initiation_func: Callable[[np.ndarray], bool],
                 primitive_actions: bool,
                 alpha: float, epsilon: float, gamma: float):
        self.actions = actions
        self.start_node = start_node
        self.start_state_str = start_state_str
        self.end_node = end_node
        self.end_state_str = end_state_str
        self.hierarchy_level = hierarchy_level
        self.initiation_func = initiation_func

        if primitive_actions:
            self.policy = QLearningAgent(actions, alpha, epsilon, gamma)
        else:
            self.policy = OptionsAgent(alpha, epsilon, gamma, actions)

        self.has_policy = True
        return

    def initiated(self, state: np.ndarray) -> bool:
        if self.start_node is not None:
            return self.start_state_str == np.array2string(state)
        return self.initiation_func(state)

    def terminated(self, state: np.ndarray) -> bool:
        if self.end_state_str == np.array2string(state):
            return True
        return not self.initiation_func(state)


class PreparednessAgent(OptionsAgent):

    option_failure_reward = -1.0
    option_step_reward = -0.001
    option_success_reward = 1.0

    preparedness_subgoal_key = 'preparedness subgoal level'

    def __init__(self, actions: List[int], alpha: float, epsilon: float, gamma: float, state_dtype: Type,
                 state_transition_graph: nx.MultiDiGraph,
                 aggregate_graph: nx.MultiDiGraph,
                 option_onboarding: str):
        assert actions is not None
        assert option_onboarding == 'none' or option_onboarding == 'specific' or option_onboarding == 'generic'

        self.actions = actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.state_dtype = state_dtype
        self.state_transition_graph = state_transition_graph
        self.aggregate_graph = aggregate_graph
        self.option_onboarding = option_onboarding

        self.min_subgoal_level = np.inf
        self.max_subgoal_level = -np.inf
        self.subgoals = {}
        self.subgoals_list = []
        for node in self.aggregate_graph.nodes(data=True):
            subgoal_level = self.aggregate_graph[node[0]][self.preparedness_subgoal_key]
            try:
                self.subgoals[subgoal_level].append(node)
            except KeyError:
                self.subgoals[subgoal_level] = [node]
            self.subgoals_list.append(node)
            subgoal_level_int = int(subgoal_level)
            if subgoal_level_int < self.min_subgoal_level:
                self.min_subgoal_level = subgoal_level_int
            elif subgoal_level_int > self.max_subgoal_level:
                self.max_subgoal_level = subgoal_level_int

        self.specific_onboarding_possible = None
        self.options = []
        self.primitive_options = [Option([action]) for action in self.actions]
        self.options_between_subgoals = {}
        self.generic_onboarding_option = None
        self.generic_onboarding_index = None
        self.specific_onboarding_options = {}
        self.generic_onboarding_subgoal_options = []
        self.specific_onboarding_subgoal_options = []
        self.state_node_lookup = {}
        self.path_lookup = {node: {} for node in self.state_transition_graph.nodes()}

        self.environment_start_states_str = None

        self.current_step = 0
        self.current_option = None
        self.current_option_index = None
        self.option_start_state = None
        self.total_option_reward = 0
        self.current_option_step = 0
        self.state_option_values = {'none': {}, 'generic': {}, 'specific': {}}
        self.state_option_values_no_onboarding = {}
        self.state_option_values_generic_onboarding = {}
        self.state_option_values_specific_onboarding = {}
        return

    def choose_action(self, state, optimal_choice=False, possible_actions=None):
        if self.current_option is None:
            self.current_option = self.choose_option(state, optimal_choice, possible_actions)
            if self.current_option is None:
                return None

        try:
            if self.current_option.policy.current_option.terminated(state):
                self.current_option.policy.current_option = None
        except AttributeError:
            ()

        if self.current_option.has_policy():
            chosen_action = self.current_option.choose_action(state, possible_actions)
        else:
            chosen_action = self.current_option.actions[self.current_option_step]

        self.current_option_step += 1
        return chosen_action

    def choose_option(self, state, no_random, possible_actions=None):
        self.current_option_step = 0
        self.option_start_state = state

        available_options = self.get_available_options(state, possible_actions)

        if (not no_random) and (rand.uniform(0, 1) < self.epsilon):
            self.current_option_index = rand.choice(available_options)
            return self.option_index_lookup(self.current_option_index)

        option_values = self.get_state_option_values(state, available_options)

        ops = [available_options[0]]
        max_value = option_values[available_options[0]]
        for i in range(1, len(available_options)):
            op = available_options[i]
            value = option_values[op]
            if value > max_value:
                max_value = value
                ops = [op]
            elif value == max_value:
                ops.append(op)

        self.current_option_index = rand.choice(ops)
        return self.option_index_lookup(self.current_option_index)

    def copy_agent(self, copy_from):
        return

    def create_option(self, start_node: None | str, end_node: str, start_state_str: None | str, end_state_str: str,
                      hierarchy_level: int, options: None | List[PreparednessOption]=None) -> PreparednessOption:
        primitive_actions = hierarchy_level <= 1
        if primitive_actions:
            options = self.actions

        option = PreparednessOption(options, start_node, end_node,
                                    start_state_str, end_state_str, hierarchy_level,
                                    lambda s: self.has_path_to_node(s, end_node),
                                    primitive_actions,
                                    self.alpha, self.epsilon, self.gamma)
        return option

    def create_options(self, environment: Environment) -> None:
        # An option from subgoals i -> j is in level k where k is the length of shortest path from i -> j in the
        # aggregate graph. If there is no such path, then there is no such option.

        aggregate_graph_distances = nx.floyd_warshall_numpy(self.aggregate_graph)
        max_option_level = np.max(aggregate_graph_distances)
        self.options_between_subgoals = {str(i): [] for i in range(1, max_option_level + 1)}
        options_for_option = []

        # Options Between Subgoals
        for k in range(1, max_option_level + 1):
            for start_node in self.aggregate_graph.nodes(data=True):
                start_node_str = self.aggregate_graph[start_node]['state']
                for end_node in self.aggregate_graph.nodes(data=True):
                    if k != aggregate_graph_distances[int(start_node)][int(end_node)]:
                        continue
                    end_node_str = self.aggregate_graph[end_node]['state']
                    option = self.create_option(start_node, end_node, start_node_str, end_node_str, k,
                                                options_for_option)
                    self.options_between_subgoals[str(k)].append(option)
            options_for_option += self.options_between_subgoals[str(k)]

        # Onboarding Options
        # can vary how options are constructed:
        # no_onboarding: Only options are between subgoals
        # generic onboarding: A single option that navigates from any start state to any subgoal
        # specific onboarding: An option for each node in the aggregate graph that has no in-edges, each option
        # navigates to one of these nodes (only available in some cases)

        # Generic Onboarding
        self.environment_start_states_str = [np.array2string(state) for state in environment.get_start_states()]
        self.generic_onboarding_option = Option(policy=QLearningAgent(self.actions,
                                                                      self.alpha, self.epsilon, self.gamma),
                                                initiation_func=lambda s: np.array2string(s) in
                                                                          self.environment_start_states_str,
                                                terminating_func=lambda s: self.get_state_node(s) in self.subgoal_list)
        # Specific Onboarding
        self.specific_onboarding_possible = False
        for node in self.aggregate_graph.nodes(data=True):
            if len(self.aggregate_graph.in_egdes(node)) <= 0:
                self.specific_onboarding_possible = True
                node_str = self.aggregate_graph[node]['state']
                option = self.create_option(None, node, None, node_str, 1)
                self.specific_onboarding_options.append(option)

        # Options to Subgoals
        # Creates options from any state to each subgoal, for every subgoal.
        # Only possible if a form of onboarding is used
        # Creates two sets of options, one that uses generic_onboarding and one that uses specific_onboarding

        # Generic Onboarding Subgoal Options
        options_for_generic_onboarding_subgoal_option = options_for_option + [self.generic_onboarding_option]
        for node in self.aggregate_graph.nodes(data=True):
            node_str = self.aggregate_graph[node]['state']
            option = self.create_option(None, node, None, node_str, max_option_level + 1,
                                        options_for_generic_onboarding_subgoal_option)
            self.generic_onboarding_subgoal_options.append(option)

        # Specific Onboarding Subgoal Options
        if not self.specific_onboarding_possible:
            return
        options_for_specific_onboarding_subgoal_option = options_for_option + self.specific_onboarding_options
        for node in self.aggregate_graph.nodes(data=True):
            if node in self.specific_onboarding_subgoal_options.keys():
                continue
            node_str = self.aggregate_graph[node]['state']
            option = self.create_option(None, node, None, node_str, max_option_level + 1,
                                        options_for_specific_onboarding_subgoal_option)
            self.specific_oboarding_subgoal_options.append(option)
        return

    def get_state_node(self, state: np.ndarray) -> str:
        state_str = np.array2string(state)
        try:
            node = self.state_node_lookup[state_str]
        except KeyError:
            for node in self.state_transition_graph.nodes(data=True):
                if state_str == self.state_node_lookup[node]['state']:
                    break
            self.state_node_lookup[state_str] = node
        return node

    def get_available_options(self, state: np.ndarry, possible_actions: None | List[int]=None) -> List[int]:
        state_str = np.array2string(state.astype(self.state_dtype))
        available_options = []
        option_index = 0

        # Primitive Options
        for primitive_option in self.primitive_options:
            if (possible_actions is None) or (primitive_option.actions[0] in possible_actions):
                available_options.append(option_index)
            option_index += 1

        # Options Between Subgoals
        for option_level in self.options_between_subgoals:
            for option in self.options_between_subgoals[option_level]:
                if option.start_state_str == state_str:
                    available_options.append(option_index)
                option_index += 1

        # Onboarding Options
        if self.option_onboarding == 'none':
            return available_options
        if self.option_onboarding == 'generic':
            if self.generic_onboarding_option.initiated(state):
                available_options.append(option_index)
            self.generic_onboarding_index = option_index
            option_index += 1
            subgoal_options = self.generic_onboarding_subgoal_options
        elif self.option_onboarding == 'specific':
            for option in self.specific_onboarding_options:
                if option.initiated(state):
                    available_options.append(option_index)
                option_index += 1
            subgoal_options = self.specific_onboarding_subgoal_options

        # Subgoal Options
        for option in subgoal_options:
            if option.initiated(state):
                available_options.append(option_index)
            option_index += 1


        return available_options

    def get_state_option_values(self, state: np.ndarray, available_options: None | List[Option]) -> Dict[int, float]:
        state_str = np.array2string(state.astype(self.state_dtype))

        try:
            option_values = self.state_option_values[self.option_onboarding][state_str]
        except KeyError:
            if available_options is None:
                available_options = self.get_available_options(state)
            option_values = {option: 0.0 for option in available_options}
            self.state_option_values[self.option_onboarding][state_str] = option_values
        return option_values

    def has_path_to_node(self, state: np.ndarray, goal_node: str):
        state_str = np.array2string(state)

        try:
            has_path_str = self.path_lookup[goal_node][state_str]
            has_path = has_path_str == 'True'
        except KeyError:
            state_node = self.get_state_node(state)
            has_path = nx.has_path(self.state_transition_graph, state_node, goal_node)
            self.path_lookup[goal_node][state_str] = str(has_path)

        return has_path

    # TODO: Make Learn method
    def learn(self, state, action, reward, next_state,
              terminal=None, next_state_possible_actions=None):
        return

    # TODO: Make Load method
    def load(self, save_path):
        return

    def option_index_lookup(self, option_index: int) -> Option:
        # Generic Onboarding Option
        if (self.option_onboarding == 'generic') and (option_index == self.generic_onboarding_index):
            return self.generic_onboarding_option

        # Primitive Options
        try:
            option = self.primitive_options[option_index]
            return option
        except IndexError:
            option_index -= len(self.primitive_options)

        # Options Between Subgoals
        for option_level in self.options_between_subgoals:
            try:
                option = self.options_between_subgoals[option_level][option_index]
                return option
            except IndexError:
                option_index -= len(self.options_between_subgoals[option_level])

        if self.option_onboarding == 'none':
            raise AttributeError("Invalid option in for option onboarding " + self.option_onboarding)

        # Subgoal Options
        if self.option_onboarding == 'generic':
            option_index -= 1
            subgoal_options = self.generic_onboarding_subgoal_options
        elif self.option_onboarding == 'specific':
            try:
                option = self.specific_onboarding_options[option_index]
                return option
            except IndexError:
                option_index -= len(self.specific_onboarding_options)
                subgoal_options = self.specific_onboarding_subgoal_options

        option = subgoal_options[option_index]
        return option

    # TODO: Make Save method
    def save(self, save_path):
        return

    def set_onboarding(self, option_onboarding: str) -> None:
        assert option_onboarding == 'none' or option_onboarding == 'specific' or option_onboarding == 'generic'
        self.option_onboarding = option_onboarding

        self.options = self.options_between_subgoals
        if self.option_onboarding == 'none':
            return
        if self.option_onboarding == 'generic':
            self.options += [self.generic_onboarding_option] + self.generic_onboarding_subgoal_options
            return
        if not self.specific_onboarding_possible:
            raise AttributeError("Specific Onboarding not possible in this domain, use generic or no onboarding")
        self.options += list(self.specific_onboarding_options.values()) + self.specific_onboarding_subgoal_options
        return

    def train_option(self, option: Option, environment: Environment,
                     training_timesteps: int,
                     option_start_states: List[np.ndarray],
                     option_success_states: List[str],
                     all_actions_possible: bool=False,
                     progress_bar: bool=False) -> None:
        # Getting Start States
        terminated = True
        possible_actions = self.actions

        for current_timesteps in range(training_timesteps):
            if progress_bar:
                print_progress_bar(current_timesteps, training_timesteps,
                                   '            >')

            if terminated:
                option_initiated = False
                while not option_initiated:
                    state = rand.choice(option_start_states)
                    state = environment.reset(state)
                    option_initiated = option.initiated(state)
                terminated = False
                if not all_actions_possible:
                    possible_actions = environment.get_possible_actions(state)

            action = option.choose_action(state, possible_actions)

            next_state, _, terminated, _ = environment.step(action)

            next_state_str = np.array2string(next_state.astype(self.state_dtype))

            reward = self.option_step_reward
            if next_state_str in option_success_states:
                terminated = True
                reward = self.option_success_reward
            elif terminated or option.terminated(next_state):
                terminated = True
                reward = self.option_failure_reward

            if not all_actions_possible:
                possible_actions = environment.get_possible_actions(next_state)

            option.policy.learn(state, action, reward, next_state, terminated, possible_actions)

            state = next_state

        return

    def train_options(self, environment: Environment,
                      training_timesteps: int,
                      all_actions_possible: bool=False,
                      progress_bar: bool=False) -> None:

        # Options between subgoals
        if progress_bar:
            print("Training Options Between Subgoals")
        for level in self.options_between_subgoals:
            if progress_bar:
                print("     Training Options at level: " + level)
            for option in self.options_between_subgoals[level]:
                if progress_bar:
                    print("         Option: " + option.start_node + " -> " + option.end_node)
                start_states = [self.state_str_to_state(option.start_state_str)]
                success_states = [self.state_str_to_state(option.end_state_str)]
                self.train_option(option, environment, training_timesteps,
                                  start_states, success_states,
                                  all_actions_possible, progress_bar)

        # Onboarding Options
        if progress_bar:
            print("Training Generic Onboarding options")
        start_states = [self.state_str_to_state(state) for state in self.environment_start_states_str]
        success_states = [self.state_str_to_state(self.aggregate_graph[subgoal_node]['state'])
                          for subgoal_node in self.subgoal_list]
        self.train_option(self.generic_onboarding_option, environment, training_timesteps,
                          start_states, success_states,
                          all_actions_possible, progress_bar)
        if progress_bar:
            print("Training Specific Onboarding Options")
        for option in self.specific_onboarding_options:
            if progress_bar:
                print("     Option towards state: " + option.end_node)
            self.train_option(option, environment, training_timesteps,
                              start_states, [self.state_str_to_state(option.end_state_str)],
                              all_actions_possible, progress_bar)

        # Subgoal Options
        if progress_bar:
            print("Training Generic Subgoal Options")
        for option in self.generic_onboarding_subgoal_options:
            if progress_bar:
                print("     Options towards state: " + option.end_node)
            self.train_option(option, environment, training_timesteps,
                              start_states, [self.state_str_to_state(option.end_state_str)],
                              all_actions_possible, progress_bar)
        if progress_bar:
            print("Training Specific Subgoal Options")
        for option in self.specific_onboarding_subgoal_options:
            if progress_bar:
                print("     Option towards state: " + option.end_node)
            self.train_option(option, environment, training_timesteps,
                              start_states, [self.state_str_to_state(option.end_state_str)],
                              all_actions_possible, progress_bar)

        return
