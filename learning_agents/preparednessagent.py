import copy
import json
import networkx as nx
import numpy as np
import random as rand
import sys
from typing import Callable, Dict, List, Tuple, Type

from environments.environment import Environment
from learning_agents.optionsagent import Option, OptionsAgent
from learning_agents.qlearningagent import QLearningAgent
from progressbar import print_progress_bar


class PreparednessOption(Option):

    def __init__(self, actions: List[Option] | List[int], start_node: None | List[str], end_node: None | str,
                 start_state_str: None | List[str], end_state_str: str,
                 hierarchy_level: int,
                 initiation_func: Callable[[np.ndarray], bool],
                 continuation_func: Callable[[np.ndarray], bool],
                 primitive_actions: bool,
                 alpha: float, epsilon: float, gamma: float,
                 state_dtype: Type,
                 subgoal_graph: None | nx.MultiGraph=None):
        self.actions = actions
        self.start_node = start_node
        self.start_state_str = start_state_str
        self.end_node = end_node
        self.end_state_str = end_state_str
        self.hierarchy_level = hierarchy_level
        self.initiation_func = initiation_func
        self.continuation_func = continuation_func
        self.state_dtype = state_dtype
        self.primitive_actions = primitive_actions

        if self.primitive_actions:
            self.policy = QLearningAgent(actions, alpha, epsilon, gamma)
        else:
            self.policy = PreparednessOptionPolicy(alpha, epsilon, gamma, self.actions,
                                                   self.state_dtype, subgoal_graph, self.end_node)
        return

    def get_option_lookup(self) -> None | Dict[str, List[int]]:
        if self.primitive_actions:
            return None
        return self.policy.option_lookup.copy()

    def get_state_values(self) -> Dict[str, Dict[str, float]]:
        if self.hierarchy_level <= 1:
            return self.policy.q_values
        return self.policy.state_option_values

    def initiated(self, state: np.ndarray) -> bool:
        if self.start_node is not None:
            return np.array2string(state.astype(self.state_dtype)) in self.start_state_str
        return self.initiation_func(state)

    def set_option_lookup(self, option_lookup: Dict[str, List[int]]) -> None:
        if self.primitive_actions:
           return
        self.policy.option_lookup = option_lookup.copy()
        return

    def set_state_values(self, state_values: Dict[str, Dict[str, float]]) -> None:
        if self.hierarchy_level <= 1:
            self.policy.q_values = state_values
            return
        self.policy.state_option_values = state_values
        return

    def terminated(self, state: np.ndarray) -> bool:
        if self.end_state_str == np.array2string(state.astype(self.state_dtype)):
            return True
        return not self.continuation_func(state)


class PreparednessOptionPolicy(OptionsAgent):

    def __init__(self, alpha: float, epsilon: float, gamma: float, options: List[Option],
                 state_dtype: Type,
                 subgoal_graph: nx.MultiDiGraph, end_node: str):
        super(PreparednessOptionPolicy, self).__init__(alpha, epsilon, gamma, options, state_dtype=state_dtype)
        self.subgoal_graph = subgoal_graph
        self.end_node = end_node
        self.option_lookup = {}
        return

    def get_available_options(self, state: np.ndarray, possible_actions: None|List[int]=None) -> List[int]:
        available_options = []
        option_index = 0
        state_str = self.state_to_state_str(state)

        try:
            available_options = self.option_lookup[state_str]
            return available_options
        except KeyError:
            pass

        for option in self.options:
            if (possible_actions is not None) and (not option.has_policy()):
                if option.actions[0] in possible_actions:
                    available_options.append(option_index)
                    option_index += 1
                    continue
            elif option.initiated(state):
                try:
                    option_end_node = option.end_node
                    if (option_end_node == self.end_node) or nx.has_path(self.subgoal_graph,
                                                                         option_end_node, self.end_node):
                        available_options.append(option_index)
                except AttributeError:
                    available_options.append(option_index)
            option_index += 1

        self.option_lookup[state_str] = available_options.copy()

        return available_options


class PreparednessAgent(OptionsAgent):

    option_failure_reward = -1.0
    option_step_reward = -0.0001
    option_success_reward = 1.0

    preparedness_subgoal_key = 'preparedness subgoal level'

    def __init__(self, actions: List[int], alpha: float, epsilon: float, gamma: float, state_dtype: Type,
                 state_shape: Tuple[int, int],
                 state_transition_graph: nx.MultiDiGraph,
                 aggregate_graph: nx.MultiDiGraph,
                 option_onboarding: str,
                 max_option_length: int=np.inf):
        assert actions is not None
        assert option_onboarding == 'none' or option_onboarding == 'specific' or option_onboarding == 'generic'

        self.actions = actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.state_dtype = state_dtype
        self.state_shape = state_shape
        self.state_transition_graph = state_transition_graph
        self.aggregate_graph = aggregate_graph
        self.option_onboarding = option_onboarding
        self.max_option_length = max_option_length

        self.min_subgoal_level = np.inf
        self.max_subgoal_level = -np.inf
        self.subgoals = {}
        self.subgoals_list = []
        for node, values in self.aggregate_graph.nodes(data=True):
            subgoal_level = values[self.preparedness_subgoal_key]
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
        self.specific_onboarding_options = []
        self.generic_onboarding_subgoal_options = []
        self.specific_onboarding_subgoal_options = []
        self.state_node_lookup = {}
        self.path_lookup = {node: {} for node in self.state_transition_graph.nodes()}

        self.environment_start_states_str = None
        self.environment_start_nodes = None

        self.current_step = 0
        self.current_option = None
        self.current_option_index = None
        self.option_start_state = None
        self.last_possible_actions = None
        self.total_option_reward = 0
        self.current_option_step = 0
        self.state_option_values = {'none': {}, 'generic': {}, 'specific': {}}
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
            chosen_action = self.current_option.actions[0]

        self.current_option_step += 1
        return chosen_action

    def choose_option(self, state, no_random, possible_actions=None):
        self.current_option_step = 0
        self.option_start_state = state

        available_options = self.get_available_options(state, possible_actions)

        if (not no_random) and (rand.uniform(0, 1) < self.epsilon):
            self.current_option_index = int(rand.choice(available_options))
            return self.option_index_lookup(self.current_option_index)

        option_values = self.get_state_option_values(state, available_options)

        ops = [available_options[0]]
        str_options = False
        try:
            max_value = option_values[available_options[0]]
        except KeyError:
            max_value = option_values[str(available_options[0])]
            str_options = True
        for i in range(1, len(available_options)):
            op = available_options[i]
            if str_options:
                op = str(op)
            value = option_values[op]
            if value > max_value:
                max_value = value
                ops = [op]
            elif value == max_value:
                ops.append(op)

        self.current_option_index = int(rand.choice(ops))
        return self.option_index_lookup(self.current_option_index)

    def copy_agent(self, copy_from: 'PreparednessAgent') -> None:
        self.specific_onboarding_possible = copy_from.specific_onboarding_possible
        self.options = copy_from.options.copy()
        self.primitive_options = copy_from.primitive_options.copy()
        self.options_between_subgoals = copy_from.options_between_subgoals.copy()
        self.generic_onboarding_option = copy_from.generic_onboarding_option
        self.generic_onboarding_index = copy_from.generic_onboarding_index
        self.specific_onboarding_options = copy_from.specific_onboarding_options.copy()
        self.generic_onboarding_subgoal_options = copy_from.generic_onboarding_subgoal_options.copy()
        self.specific_onboarding_subgoal_options = copy_from.specific_onboarding_subgoal_options.copy()
        self.state_node_lookup = copy_from.state_node_lookup
        self.path_lookup = copy_from.path_lookup
        self.environment_start_states_str = copy_from.environment_start_states_str
        self.environment_start_nodes = copy_from.environment_start_nodes
        self.state_option_values = copy_from.state_option_values.copy()

        self.current_step = 0
        self.current_option = None
        self.current_option_index = None
        self.option_start_state = None
        self.total_option_reward = 0
        self.current_option_step = 0
        return

    def create_option(self, start_node: None | str, end_node: str, start_state_str: None | str, end_state_str: str,
                      hierarchy_level: int, options: None | List[PreparednessOption]=None,
                      initiation_func: None | Callable[[np.ndarray], bool]=None) -> PreparednessOption:
        primitive_actions = hierarchy_level <= 1
        if primitive_actions:
            options = self.actions

        continuation_func = lambda s: self.get_state_node(s) != end_node and self.has_path_to_node(s, end_node)
        if self.max_option_length != np.inf:
            def continuation_func(s: np.ndarray) -> bool:
                s_node = self.get_state_node(s)
                if s_node == end_node:
                    return False
                if not self.has_path_to_node(s, end_node):
                    return False
                start_state = self.state_str_to_state(start_state_str[0])
                return self.has_path_to_node(start_state, s_node)

        if initiation_func is None:
            initiation_func = continuation_func
        option = PreparednessOption(options.copy(), start_node, end_node,
                                    start_state_str, end_state_str, hierarchy_level,
                                    initiation_func, continuation_func,
                                    primitive_actions,
                                    self.alpha, self.epsilon, self.gamma,
                                    self.state_dtype, self.aggregate_graph)
        return option

    def create_options(self, environment: Environment) -> None:
        # An option from subgoals i -> j is in level k where k is the length of shortest path from i -> j in the
        # aggregate graph. If there is no such path, then there is no such option.

        aggregate_graph_distances = nx.floyd_warshall(self.aggregate_graph)
        max_option_level = -np.inf
        for start_node in self.aggregate_graph.nodes(data=False):
            for end_node in self.aggregate_graph.nodes(data=False):
                distance = aggregate_graph_distances[start_node][end_node]
                if distance >= np.inf:
                    continue
                if distance > max_option_level:
                    max_option_level = distance
        max_option_level = int(max_option_level)

        self.options_between_subgoals = {str(i): [] for i in range(1, max_option_level + 1)}
        options_for_option = []

        # Options Between Subgoals
        for k in range(1, max_option_level + 1):
            for start_node, start_values in self.aggregate_graph.nodes(data=True):
                start_node_str = start_values['state']
                for end_node, end_values in self.aggregate_graph.nodes(data=True):
                    if k != aggregate_graph_distances[start_node][end_node]:
                        continue
                    if self.max_option_length != np.inf:
                        if not self.has_path_to_node(self.state_str_to_state(start_node_str), end_node):
                            continue
                    end_node_str = end_values['state']
                    option = self.create_option([start_node], end_node,
                                                [start_node_str], end_node_str, k,
                                                options_for_option)
                    self.options_between_subgoals[str(k)].append(option)
            options_for_option += self.options_between_subgoals[str(k)]

        # Onboarding Options
        # can vary how options are constructed:
        # no_onboarding: Only options are between subgoals
        # generic onboarding: A single option that navigates from any state with a path to a subgoal, to a subgoal
        # specific onboarding: An option for each node in the aggregate graph that has no in-edges, each option
        # navigates to one of these nodes (only available in some cases)

        # Generic Onboarding
        self.environment_start_states_str = []
        self.environment_start_nodes = []
        for state in environment.get_start_states():
            self.environment_start_states_str.append(np.array2string(state.astype(self.state_dtype)))
            self.environment_start_nodes.append(self.get_state_node(state))
        self.generic_onboarding_option = Option(policy=QLearningAgent(self.actions,
                                                                      self.alpha, self.epsilon, self.gamma),
                                                initiation_func=self.generic_onboarding_initiation_function,
                                                terminating_func=lambda s: (self.get_state_node(s) in
                                                                            self.subgoals_list) or (
                                                    not self.generic_onboarding_initiation_function(s)))
        # Specific Onboarding
        self.specific_onboarding_possible = False
        specific_onboarding_nodes = []
        for node, values in self.aggregate_graph.nodes(data=True):
            if len(self.aggregate_graph.in_edges(node)) <= 0:
                self.specific_onboarding_possible = True
                specific_onboarding_nodes.append(node)
                option = self.create_option(None, node, None, values['state'],
                                            1, None)
                self.specific_onboarding_options.append(option)

        # Options to Subgoals
        # Creates options from any state to each subgoal, for every subgoal.
        # Only possible if a form of onboarding is used
        # Creates two sets of options, one that uses generic_onboarding and one that uses specific_onboarding

        # Generic Onboarding Subgoal Options
        # Can initiate from any state where there is a path to their subgoal
        options_for_generic_onboarding_subgoal_option = options_for_option + [self.generic_onboarding_option]
        for node, values in self.aggregate_graph.nodes(data=True):
            node_str = values['state']
            option = self.create_option(None, node, None, node_str,
                                        max_option_level + 1,
                                        options_for_generic_onboarding_subgoal_option)
            self.generic_onboarding_subgoal_options.append(option)

        # Specific Onboarding Subgoal Options
        # Initiation states:
        #   Start states
        #   Subgoal states that have a path to the corresponding subgoal
        #   States that have a path to an onboarded subgoal and a path to the corresponding subgoal
        if not self.specific_onboarding_possible:
            return
        options_for_specific_onboarding_subgoal_option = options_for_option + self.specific_onboarding_options
        for node, values in self.aggregate_graph.nodes(data=True):
            if node in specific_onboarding_nodes:
                continue
            node_str = values['state']
            initiation_func = self.create_specific_subgoal_option_initiation_func(node)
            option = self.create_option(None, node, None, node_str,
                                        max_option_level + 1,
                                        options_for_specific_onboarding_subgoal_option,
                                        initiation_func)
            self.specific_onboarding_subgoal_options.append(option)
        return

    def create_specific_subgoal_option_initiation_func(self, subgoal: str) -> Callable[[np.ndarray], bool]:
        def initiation_function(state: np.ndarray) -> bool:
            state_node = self.get_state_node(state)
            if state_node == subgoal:
                return False

            for node in self.aggregate_graph.nodes(data=False):
                if state_node != node:
                    continue
                if nx.has_path(self.aggregate_graph, node, subgoal):
                    return True

            for option in self.specific_onboarding_options:
                onboarding_subgoal = option.end_node
                if not nx.has_path(self.aggregate_graph, onboarding_subgoal, subgoal):
                    continue
                if self.has_path_to_node(state, onboarding_subgoal):
                    return True

            return False

        return initiation_function

    def generic_onboarding_initiation_function(self, state: np.ndarray) -> bool:
        for subgoal in self.subgoals_list:
            if self.has_path_to_node(state, subgoal):
                return True
        return False

    def get_state_node(self, state: np.ndarray) -> str:
        state_str = np.array2string(state.astype(self.state_dtype))
        try:
            node = self.state_node_lookup[state_str]
        except KeyError:
            for node, values in self.state_transition_graph.nodes(data=True):
                if state_str == values['state']:
                    break
            self.state_node_lookup[state_str] = node
        return node

    def get_available_options(self, state: np.ndarray, possible_actions: None | List[int]=None) -> List[str]:
        state_str = np.array2string(state.astype(self.state_dtype))
        available_options = []
        option_index = 0

        # Primitive Options
        for primitive_option in self.primitive_options:
            if (possible_actions is None) or (primitive_option.actions[0] in possible_actions):
                available_options.append(str(option_index))
            option_index += 1

        # Options Between Subgoals
        for option_level in self.options_between_subgoals:
            for option in self.options_between_subgoals[option_level]:
                if option.start_state_str[0] == state_str:
                    available_options.append(str(option_index))
                option_index += 1

        # Onboarding Options
        if self.option_onboarding == 'none':
            return available_options
        if self.option_onboarding == 'generic':
            if self.generic_onboarding_option.initiated(state):
                available_options.append(str(option_index))
            self.generic_onboarding_index = option_index
            option_index += 1
            subgoal_options = self.generic_onboarding_subgoal_options
        elif self.option_onboarding == 'specific':
            for option in self.specific_onboarding_options:
                if option.initiated(state):
                    available_options.append(str(option_index))
                option_index += 1
            subgoal_options = self.specific_onboarding_subgoal_options

        # Subgoal Options
        for option in subgoal_options:
            if option.initiated(state):
                available_options.append(str(option_index))
            option_index += 1


        return available_options

    def get_state_option_values(self, state: np.ndarray, available_options: None | List[Option]=None) -> Dict[str, float]:
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
        state_str = np.array2string(state.astype(self.state_dtype))

        try:
            has_path_str = self.path_lookup[goal_node][state_str]
            has_path = has_path_str == 'True'
        except KeyError:
            state_node = self.get_state_node(state)
            if state_node == goal_node:
                has_path = False
            else:
                if self.max_option_length >= np.inf:
                    has_path = nx.has_path(self.state_transition_graph, state_node, goal_node)
                else:
                    try:
                        path_length = nx.shortest_path_length(self.state_transition_graph, state_node, goal_node)
                        has_path = path_length <= self.max_option_length
                    except nx.NetworkXError:
                        has_path = False
            self.path_lookup[goal_node][state_str] = str(has_path)

        return has_path

    def learn(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray,
              terminal: None | bool=None, next_state_possible_actions: None | List[int]=None) -> None:
        # Q(s, o) = Q(s, o) + \alpha(r - Q(s, o) + \gamma((1 - \beta)Q(s_prime, o) + \beta(MAXQ(s_prime, o_prime)))))

        # if terminal in next_state
        # Q(s, o) = Q(s, o) + \alpha(r - Q(s, o) + \gamma*MAXQ(next_state, o_prime))

        # if not terminal in next_state
        # Q(s, o) = Q(s, o) + \alpha*(r - Q(s, o) + \gamma*Q(s_prime, o))
        self.total_option_reward += reward

        state_str = self.state_to_state_str(state)
        available_options = self.get_available_options(state, self.last_possible_actions)
        state_option_values = self.get_state_option_values(state, available_options)

        next_available_options = []
        if not terminal:
            next_available_options = self.get_available_options(next_state, next_state_possible_actions)

        next_state_option_values_list = [0.0]
        next_state_option_values = self.get_state_option_values(next_state, next_available_options)
        if next_available_options:
            next_state_option_values_list = [next_state_option_values[option] for option in next_available_options]
        max_next_state_option_value = max(next_state_option_values_list)

        for option_index in available_options:
            option = self.option_index_lookup(int(option_index))
            if option.has_policy():
                train_option = option.choose_action(state, self.last_possible_actions) == action
                try:
                    reset_inner_option_policy = option.policy.current_option is None
                except AttributeError:
                    reset_inner_option_policy = False
            else:
                train_option = option.actions[0] == action
                reset_inner_option_policy = False

            if train_option:
                if reset_inner_option_policy:
                    option.policy.current_option = None

                gamma_product = max_next_state_option_value
                if not option.terminated(next_state):
                    try:
                        gamma_product = next_state_option_values[option_index]
                    except KeyError:
                        gamma_product = max_next_state_option_value

                self.state_option_values[self.option_onboarding][state_str][option_index] += self.alpha * (reward -
                                                                                   state_option_values[option_index] +
                                                                                   self.gamma * gamma_product)

        if (not (terminal or self.current_option.terminated(next_state))) and (
                self.current_option_step < self.max_option_length):
            return

        option_value = self.get_state_option_values(self.option_start_state)[str(self.current_option_index)]
        option_start_state_str = self.state_to_state_str(self.option_start_state)
        self.state_option_values[self.option_onboarding][option_start_state_str][str(self.current_option_index)] \
            += self.alpha * (self.total_option_reward + (self.gamma ** self.current_option_step) *
                             max_next_state_option_value
                             - option_value)
        self.current_option = None
        self.option_start_state = None
        self.current_option_index = None
        self.total_option_reward = 0
        self.current_option_step = 0
        return

    def load(self, save_path: str) -> None:
        with open(save_path, 'r') as f:
            agent_save_file = json.load(f)

        self.options_between_subgoals = {}
        options_for_option = []
        for level in agent_save_file['options between subgoals']:
            option_list = agent_save_file['options between subgoals'][level]
            self.options_between_subgoals[level] = []
            for option_dict in option_list:
                hierarchy_level = int(option_dict['hierarchy level'])
                option = self.create_option([option_dict['start node']], option_dict['end node'],
                                            [option_dict['start state str']], option_dict['end state str'],
                                            hierarchy_level, options_for_option)
                option.set_state_values(option_dict['policy'])
                if hierarchy_level > 1:
                    try:
                        option.set_option_lookup(option_dict['option lookup'])
                    except KeyError:
                        pass
                self.options_between_subgoals[level].append(option)
            options_for_option += self.options_between_subgoals[level]

        self.environment_start_states_str = agent_save_file['environment start states str']
        self.environment_start_nodes = agent_save_file['environment start nodes']
        self.generic_onboarding_option = Option(policy=QLearningAgent(self.actions,
                                                                      self.alpha, self.epsilon, self.gamma),
                                                initiation_func=self.generic_onboarding_initiation_function,
                                                terminating_func=lambda s: (self.get_state_node(s) in
                                                                            self.subgoals_list) or (
                                                    not self.generic_onboarding_initiation_function(s)))
        self.generic_onboarding_option.q_values = agent_save_file['generic onboarding option']['policy'].copy()

        self.specific_onboarding_options = []
        self.specific_onboarding_possible = False
        for option_dict in agent_save_file['specific onboarding options']:
            self.specific_onboarding_possible = True
            node = option_dict['end node']
            option = self.create_option(None, node, None, option_dict['end state str'],
                                        1, None)
            option.set_state_values(option_dict['policy'])
            self.specific_onboarding_options.append(option)

        self.generic_onboarding_subgoal_options = []
        options_for_generic_subgoal_options = options_for_option + [self.generic_onboarding_option]
        max_option_level = int(level) + 1
        for option_dict in agent_save_file['generic onboarding subgoal options']:
            option = self.create_option(None, option_dict['end node'], None, option_dict['end state str'],
                                        max_option_level,
                                        options_for_generic_subgoal_options)
            option.set_state_values(option_dict['policy'])
            try:
                option.set_option_lookup(option_dict['option lookup'])
            except KeyError:
                pass
            self.generic_onboarding_subgoal_options.append(option)

        self.specific_onboarding_subgoal_options = []
        options_for_specific_subgoal_options = options_for_option + self.specific_onboarding_options
        for option_dict in agent_save_file['specific onboarding subgoal options']:
            node = option_dict['end node']
            initiation_func = self.create_specific_subgoal_option_initiation_func(node)
            option = self.create_option(None, node, None, option_dict['end state str'],
                                        max_option_level,
                                        options_for_specific_subgoal_options,
                                        initiation_func)
            option.set_state_values(option_dict['policy'])
            try:
                option.set_option_lookup(option_dict['option lookup'])
            except KeyError:
                pass
            self.specific_onboarding_subgoal_options.append(option)

        self.generic_onboarding_index = agent_save_file['generic onboarding index']
        if self.generic_onboarding_index is not None:
            self.generic_onboarding_index = int(self.generic_onboarding_index)
        self.state_node_lookup = agent_save_file['state node lookup']
        self.path_lookup = agent_save_file['path lookup']
        self.state_option_values = agent_save_file['state option values']
        self.max_option_length = agent_save_file['max option length']
        return

    def node_to_state(self, node: str) -> np.ndarray:
        state_str = self.state_transition_graph.nodes(data=True)[node]['state']
        return self.state_str_to_state(state_str)

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

    def save(self, save_path: str) -> None:
        agent_save_file = {'options between subgoals': {level: [{'start node': option.start_node[0],
                                                         'end node': option.end_node,
                                                         'start state str': option.start_state_str[0],
                                                         'end state str': option.end_state_str,
                                                         'hierarchy level': option.hierarchy_level,
                                                         'policy': option.get_state_values(),
                                                         'option lookup': option.get_option_lookup()
                                                         } for option in self.options_between_subgoals[level]]
                                                        for level in self.options_between_subgoals},
                           'generic onboarding option': {'policy':
                                                             self.generic_onboarding_option.policy.q_values},
                           'generic onboarding index': self.generic_onboarding_index,
                           'specific onboarding options': [{'end node': option.end_node,
                                                            'end state str': option.end_state_str,
                                                            'policy': option.get_state_values(),
                                                            'option lookup': option.get_option_lookup()}
                                                           for option in self.specific_onboarding_options],
                           'generic onboarding subgoal options': [{'end node': option.end_node,
                                                                   'end state str': option.end_state_str,
                                                                   'policy': option.get_state_values(),
                                                                   'option lookup': option.get_option_lookup()}
                                                                  for option in self.generic_onboarding_subgoal_options],
                           'specific onboarding subgoal options': [{'end node': option.end_node,
                                                                    'end state str': option.end_state_str,
                                                                    'policy': option.get_state_values(),
                                                                    'option lookup': option.get_option_lookup()}
                                                                   for option in self.specific_onboarding_subgoal_options],
                           'state node lookup': self.state_node_lookup,
                           'path lookup': self.path_lookup,
                           'environment start states str': self.environment_start_states_str,
                           'environment start nodes': self.environment_start_nodes,
                           'state option values': self.state_option_values,
                           'max option length': self.max_option_length}

        with open(save_path, 'w') as f:
            json.dump(agent_save_file, f)
        return

    def set_onboarding(self, option_onboarding: str) -> None:
        assert option_onboarding == 'none' or option_onboarding == 'specific' or option_onboarding == 'generic'
        self.option_onboarding = option_onboarding

        self.options = self.primitive_options.copy()
        for level in self.options_between_subgoals:
            self.options += self.options_between_subgoals[level]

        if self.option_onboarding == 'none':
            return
        if self.option_onboarding == 'generic':
            self.options += [self.generic_onboarding_option] + self.generic_onboarding_subgoal_options
            return
        if not self.specific_onboarding_possible:
            raise AttributeError("Specific Onboarding not possible in this domain, use generic or no onboarding")
        self.options += self.specific_onboarding_options + self.specific_onboarding_subgoal_options
        return

    def set_option_by_pathing(self, option: PreparednessOption) -> None:
        for node, values in self.aggregate_graph.nodes(data=True):
            start_state = self.state_str_to_state(values['state'])
            if option.terminated(start_state):
                continue
            if (node != option.start_node[0]) and (not nx.has_path(self.aggregate_graph, node, option.start_node[0])):
                continue

            path = nx.dijkstra_path(self.aggregate_graph, node, option.end_node)

            for i in range(len(path) - 1):
                first_node = path[i]
                next_node = path[i + 1]
                current_state = self.node_to_state(first_node)

                possible_options = option.policy.get_available_options(current_state)

                values = {int(option_index): 0.0 for option_index in possible_options}
                for option_index in possible_options:
                    current_option = option.policy.options[int(option_index)]
                    if current_option.end_node == next_node and current_option.start_nodes[0] == first_node:
                        values[int(option_index)] = 1.0
                        break

                option.policy.set_state_option_values(values, current_state)

        return

    def set_options_by_pathing(self, levels_to_set: None | List[int]=None,
                               options_to_set: None | List[Tuple[str, str]]=None) -> None:
        if levels_to_set is None:
            levels_to_set = [level for level in range(self.min_subgoal_level, self.max_subgoal_level + 1)]
        levels_to_set = [str(level) for level in levels_to_set]

        for level in levels_to_set:
            for option in self.options_between_subgoals[level]:
                if (options_to_set is None) or ((option.start_node[0], option.end_node) in options_to_set):
                    self.set_option_by_pathing(option)

        return

    def train_option(self, option: Option, environment: Environment,
                     training_timesteps: int,
                     option_success_states: List[str],
                     option_start_states: None | List[np.ndarray] = None,
                     all_actions_possible: bool=False,
                     progress_bar: bool=False) -> Tuple[int, int]:

        # Getting Start States
        terminated = True
        possible_actions = self.actions
        total_end_states = 0
        total_successes = 0

        for current_timesteps in range(training_timesteps):
            if progress_bar:
                print_progress_bar(current_timesteps, training_timesteps,
                                   '            >')

            if terminated:
                option_initiated = False
                while not option_initiated:
                    if option_start_states is not None:
                        state = rand.choice(option_start_states)
                    else:
                        state_node = rand.choice(list(self.path_lookup.keys()))
                        state = self.node_to_state(state_node)
                    if environment.is_terminal(state):
                        continue
                    state = environment.reset(state)
                    option_initiated = option.initiated(state)
                if not all_actions_possible:
                    possible_actions = environment.get_possible_actions(state)

            action = option.choose_action(state, possible_actions)

            # Occurs if sub-option is not fully trained and takes itself to a state where it terminates
            # but the parent option does not terminate, but has no other options to initiate
            if action is None:
                terminated = True
                continue

            next_state, _, terminated, _ = environment.step(action)

            next_state_str = np.array2string(next_state.astype(self.state_dtype))

            reward = self.option_step_reward
            if next_state_str in option_success_states:
                terminated = True
                reward = self.option_success_reward
                total_successes += 1
            elif terminated or option.terminated(next_state):
                terminated = True
                reward = self.option_failure_reward

            if terminated:
                total_end_states += 1

            if not all_actions_possible:
                possible_actions = environment.get_possible_actions(next_state)

            option.policy.learn(state, action, reward, next_state, terminated, possible_actions)

            state = next_state

        return total_end_states, total_successes

    # TODO: Finish value iteration training
    def train_option_value_iteration(self, training_option: PreparednessOption, environment: Environment,
                                     min_delta: float, option_runs: int) -> None:
        def run_option(start_state: np.ndarray, option: Option) -> Tuple[np.ndarray, float]:
            terminal = False
            current_state = environment.reset(start_state)
            total_reward = 0
            while not terminal:
                possible_actions = environment.get_possible_actions(current_state)
                action = option.choose_action(current_state, possible_actions)
                current_state, _, terminal, _ = environment.step(action)
                total_reward += self.option_step_reward
                if not terminal:
                    terminal = option.terminated(current_state)

            if self.get_state_node(current_state) == training_option.end_node:
                total_reward += self.option_success_reward
            elif training_option.terminated(current_state):
                total_reward += self.option_failure_reward
            return current_state, total_reward

        def v(s: np.ndarray) -> float:
            state_option_values = training_option.policy.get_state_option_values(s)
            return max(state_option_values.values())

        delta = np.inf
        while delta > min_delta:
            delta = 0
            for node, values in self.aggregate_graph.nodes(data=True):
                state_str = values['state']
                state = self.state_str_to_state(state_str)
                if training_option.terminated(state):
                    continue

                for i in range(option_runs):
                    possible_actions = environment.get_possible_actions(state)
                    possible_options = training_option.policy.get_available_options(state, possible_actions)
                    for possible_option in possible_options:
                        state_after_option, reward = run_option(state, training_option)

        return

    def train_options(self, environment: Environment,
                      training_timesteps: int,
                      min_level: None | int=None, max_level: None | int=None,
                      train_between_options: bool=True,
                      train_onboarding_options: bool=True, train_subgoal_options: bool=True,
                      options_to_train: None | List[List[str]]=None,
                      all_actions_possible: bool=False,
                      progress_bar: bool=False,
                      trained_benchmark: None | float=None) -> None | List[Tuple[str, str]]:

        def percentage(x: int, y: int) -> float:
            if y <= 0:
                return -1.0
            return round((x/y) * 100, 3)

        if min_level is None:
            min_level = -np.inf
        if max_level is None:
            max_level = np.inf
        untrained_options = []

        # Options between subgoals
        if train_between_options:
            if progress_bar:
                print("Training Options Between Subgoals")
            for level in self.options_between_subgoals:
                if not (min_level <= int(level) <= max_level):
                    continue
                if progress_bar:
                    print("     Training Options at level: " + level)
                    num_options = str(len(self.options_between_subgoals[level]))
                    option_count = 0
                for option in self.options_between_subgoals[level]:
                    if options_to_train is not None:
                        if [option.start_node[0], option.end_node] not in options_to_train:
                            continue

                    if progress_bar:
                        option_count += 1
                        print("         Option: " + option.start_node[0] + " -> " + option.end_node +
                              " - " + str(option_count) + "/" + num_options)
                    start_states = [self.state_str_to_state(option.start_state_str[0])]
                    success_states = [option.end_state_str]
                    total_end_states, total_successes = self.train_option(option, environment, training_timesteps,
                                                                          success_states, start_states,
                                                                          all_actions_possible, progress_bar)

                    percentage_hits = percentage(total_successes, total_end_states)
                    if trained_benchmark is not None:
                        if percentage_hits < (trained_benchmark * 100):
                            untrained_options.append((option.start_node[0], option.end_node))

                    if progress_bar:
                        sys.stdout.flush()
                        print("\r         Option: " + option.start_node[0] + " -> " + option.end_node + " "
                              + str(percentage_hits) + "% hits, " + str(total_successes) + " total hits")

        # Onboarding Options
        # Generic Onboarding Options
        if train_onboarding_options:
            if progress_bar:
                print("Training Generic Onboarding option")
            success_states = [values['state']
                              for _, values in self.aggregate_graph.nodes(data=True)]
            total_end_states, total_successes = self.train_option(self.generic_onboarding_option, environment,
                                                                  training_timesteps,
                                                                  success_states, None,
                                                                  all_actions_possible, progress_bar)
            if progress_bar:
                percentage_hits = percentage(total_successes, total_end_states)
                print(" Onboarding Option " + str(percentage_hits) + "% hits, " + str(total_successes) + " total hits")
            if progress_bar:
                print("Training Specific Onboarding Options")
            # Specific onboarding options
            for option in self.specific_onboarding_options:
                if progress_bar:
                    print("     Option towards state: " + option.end_node)
                total_end_states, total_successes = self.train_option(option, environment, training_timesteps,
                                                                      [option.end_state_str],
                                                                      None,
                                                                      all_actions_possible, progress_bar)

                if progress_bar:
                    sys.stdout.flush()
                    percentage_hits = percentage(total_successes, total_end_states)
                    print("\r     Option towards state: " + option.end_node + " " + str(percentage_hits) +
                          "% hits, " + str(total_successes) + " total hits")

        # Subgoal Options
        # Generic Subgoal Options
        if train_subgoal_options:
            if progress_bar:
                print("Training Generic Subgoal Options")
            for option in self.generic_onboarding_subgoal_options:
                if progress_bar:
                    print("     Options towards state: " + option.end_node)
                total_end_states, total_successes = self.train_option(option, environment, training_timesteps,
                                                                      [option.end_state_str],
                                                                      None,
                                                                      all_actions_possible, progress_bar)
                if progress_bar:
                    sys.stdout.flush()
                    percentage_hits = percentage(total_successes, total_end_states)
                    print("\r     Option towards state: " + option.end_node + " " + str(percentage_hits) +
                          "% hits, " + str(total_successes) + " total hits")
            # Specific Subgoal Options
            if progress_bar:
                print("Training Specific Subgoal Options")
            for option in self.specific_onboarding_subgoal_options:
                if progress_bar:
                    print("     Option towards state: " + option.end_node)
                total_end_states, total_successes = self.train_option(option, environment, training_timesteps,
                                                                      [option.end_state_str],
                                                                      None,
                                                                      all_actions_possible, progress_bar)
                if progress_bar:
                    sys.stdout.flush()
                    percentage_hits = percentage(total_successes, total_end_states)
                    print("\r     Option towards state: " + option.end_node + " " + str(percentage_hits) +
                          "% hits, " + str(total_successes) + " total hits")

        if trained_benchmark is None:
            return

        if not progress_bar:
            return untrained_options

        print("Untrained Options: ")
        for untrained_option in untrained_options:
            print("     " + untrained_option[0] + ' -> ' + untrained_option[1])
        return untrained_options
