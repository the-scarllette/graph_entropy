import networkx as nx
import numpy as np
import random as rand
from scipy import sparse
from typing import Callable, Dict, List, Type

from environments.environment import Environment
from optionsagent import Option, OptionsAgent
from learningagent import LearningAgent
from qlearningagent import QLearningAgent


class PreparednessOption(Option):

    def __init__(self, actions: List[Option] | List[int], start_node: None | str, end_node: None | str,
                 start_state_str: str, end_state_str: str,
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


class PreparednessAgent(LearningAgent):

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
        self.specific_onboarding_options = {}
        self.generic_onboarding_subgoal_options = []
        self.specific_onboarding_subgoal_options = []
        self.state_node_lookup = {}
        self.path_lookup = {node: {} for node in self.state_transition_graph.nodes()}

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

    def copy_agent(self, copy_from):
        return

    def create_option(self, start_node: None | str, end_node: str, start_state_str: None | str, end_state_str: str,
                      hierarchy_level: int, options: None | List[PreparednessOption]=None) -> PreparednessOption:
        primitive_actions = hierarchy_level <= 1
        if primitive_actions:
            options = self.actions#

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
        start_states = [np.array2string(state) for state in environment.get_start_states()]
        self.generic_onboarding_option = Option(policy=QLearningAgent(self.actions,
                                                                      self.alpha, self.epsilon, self.gamma),
                                                initiation_func=lambda s: np.array2string(s) in start_states,
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

    def learn(self, state, action, reward, next_state,
              terminal=None, next_state_possible_actions=None):
        return

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

    def save(self, save_path):
        pass

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

    def train_options(self, ):
