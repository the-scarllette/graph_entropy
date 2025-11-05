import copy
import json
import networkx as nx
import numpy as np
import random as rand
from scipy import sparse, stats
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
    option_step_reward = -0.1
    option_success_reward = 1.0

    preparedness_subgoal_key = 'preparedness-subgoal-height'

    def __init__(self, actions: List[int], alpha: float, epsilon: float, gamma: float, state_dtype: Type,
                 state_shape: Tuple[int, int],
                 state_transition_graph: nx.MultiDiGraph,
                 adjacency_matrix: sparse.csr_matrix,
                 stg_values: Dict[str, Dict[str, int|str|float]],
                 subgoal_graph: None|nx.MultiDiGraph,
                 option_onboarding: str,
                 max_option_length: int=np.inf,
                 max_hierarchy_height : int=10):
        assert actions is not None
        assert option_onboarding == 'none' or option_onboarding == 'specific' or option_onboarding == 'generic'

        self.actions = actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.state_dtype = state_dtype
        self.state_shape = state_shape
        self.state_transition_graph = state_transition_graph
        self.adjacency_matrix = adjacency_matrix
        self.distance_matrix = sparse.csgraph.dijkstra(
            adjacency_matrix,
            True,
            unweighted=True,
            limit=max_hierarchy_height
        )
        self.subgoal_graph = subgoal_graph
        self.stg_values = stg_values
        self.option_onboarding = option_onboarding
        self.max_option_length = max_option_length
        self.max_hierarchy_height = max_hierarchy_height

        self.min_subgoal_level = np.inf
        self.max_subgoal_level = -np.inf
        self.subgoals = {}
        self.subgoals_list = []
        if self.subgoal_graph is not None:
            for node, values in self.subgoal_graph.nodes(data=True):
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

        # {state, next_state, n} -> P(S_{t + n} = next_state | S_{t} = state) actions taken randomly
        self.random_transition_prob: Dict[Tuple[int, int, int], float] = {}
        return

    def assign_subgoals(
            self,
            unassigned_subgoals: Dict[int, List[int]],
            max_subgoal_height: int,
            key: str
    ) -> Dict[int, List[str]]:
        subgoal_height_found: bool
        subgoal_height_value: str
        height: int
        subgoals = {i: [] for i in range(1, max_subgoal_height + 1)}
        for node in self.state_transition_graph.nodes():
            subgoal_height_value = "None"

            subgoal_height_found = False
            height = max_subgoal_height
            while height > 0:
                if int(node) in unassigned_subgoals[height]:
                    subgoal_height_found = True
                    break
                height -= 1

            if subgoal_height_found:
                subgoals[height].append(node)
                subgoal_height_value = str(height)

            self.stg_values[node][key] = subgoal_height_value
        return subgoals

    def choose_option(self, state, no_random, possible_actions=None):
        self.current_option_step = 0
        self.option_start_state = np.copy(state)

        available_options = self.get_available_options(state, possible_actions)
        if len(available_options) <= 0:
            pass

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

    def compute_preparedness(
            self,
            hops: int,
            use_existing_values: bool,
            verbose: bool=False
    ):
        frequency_entropy: None| float
        neighbourhood_entropy: None| float
        preparedness: None| float

        for node in self.stg_values:
            if verbose:
                print_progress_bar(
                    int(node),
                    self.adjacency_matrix.shape[0],
                    "Computing Preparedness node " + str(node),
                    " Complete"
                )

            frequency_entropy = None
            neighbourhood_entropy = None
            preparedness = None

            if use_existing_values:
                try:
                    frequency_entropy = self.stg_values[node][self.frequency_entropy_key(hops)]
                except KeyError:
                    ()
                try:
                    neighbourhood_entropy = self.stg_values[node][self.neighbourhood_entropy_key(hops)]
                except KeyError:
                    ()
                try:
                    preparedness = self.stg_values[node][self.preparedness_key(hops)]
                except KeyError:
                    ()

            if frequency_entropy is None:
                frequency_entropy = self.frequency_entropy(int(node), hops)
                self.stg_values[str(node)][PreparednessAgent.frequency_entropy_key(hops)] = frequency_entropy
            if neighbourhood_entropy is None:
                neighbourhood_entropy = self.neighbourhood_entropy(int(node), hops)
                self.stg_values[str(node)][PreparednessAgent.neighbourhood_entropy_key(hops)] = neighbourhood_entropy
            if preparedness is None:
                preparedness = frequency_entropy + neighbourhood_entropy
                self.stg_values[str(node)][PreparednessAgent.preparedness_key(hops)] = preparedness

        return

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

        self.max_option_length = copy_from.max_option_length
        self.max_hierarchy_height = copy_from.max_hierarchy_height

        self.current_step = 0
        self.current_option = None
        self.current_option_index = None
        self.option_start_state = None
        self.total_option_reward = 0
        self.current_option_step = 0
        return

    def count_available_skills(self, state: np.ndarray, possible_actions: None|List[int]=None) -> int:
        num_available_skills = 0

        available_options = self.get_available_options(state, possible_actions)

        for option_index in available_options:
            option = self.option_index_lookup(int(option_index))
            if option.has_policy():
                num_available_skills += 1

        return num_available_skills

    def count_skills(self) -> Dict[int, int]:
        skills_count = {}

        # Counting skills between subgoals
        for level in self.options_between_subgoals:
            skills_count[int(level)] = len(self.options_between_subgoals[level])

        if self.option_onboarding == 'none':
            return skills_count

        # Counting Onboarding options
        level = int(level) + 1
        if self.option_onboarding == 'generic': # generic onboarding
            skills_count[1] += 1 # generic onboarding option
            skills_count[level] = len(self.generic_onboarding_subgoal_options)
            return skills_count

        # specific onboarding
        skills_count[1] += len(self.specific_onboarding_options)
        skills_count[level] = len(self.generic_onboarding_subgoal_options)

        return skills_count

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
                if start_state_str is not None:
                    start_state = self.state_str_to_state(start_state_str[0])
                    return self.has_path_to_node(start_state, s_node)
                return True

        if initiation_func is None:
            initiation_func = continuation_func
        option = PreparednessOption(options.copy(), start_node, end_node,
                                    start_state_str, end_state_str, hierarchy_level,
                                    initiation_func, continuation_func,
                                    primitive_actions,
                                    self.alpha, self.epsilon, self.gamma,
                                    self.state_dtype, self.subgoal_graph)
        return option

    def create_options(self, environment: Environment) -> None:
        # An option from subgoals i -> j is in level k where k is the length of shortest path from i -> j in the
        # aggregate graph. If there is no such path, then there is no such option.

        aggregate_graph_distances = nx.floyd_warshall(self.subgoal_graph)
        max_option_level = -np.inf
        for start_node in self.subgoal_graph.nodes(data=False):
            for end_node in self.subgoal_graph.nodes(data=False):
                distance = aggregate_graph_distances[start_node][end_node]
                if distance >= np.inf:
                    continue
                if distance > max_option_level:
                    max_option_level = distance
        max_option_level = int(max_option_level)
        if self.max_hierarchy_height is None:
            self.max_hierarchy_height = max_option_level
        else:
            max_option_level = min(self.max_hierarchy_height, max_option_level)

        self.options_between_subgoals = {str(i): [] for i in range(1, max_option_level + 1)}
        options_for_option = []

        # Options Between Subgoals
        for k in range(1, max_option_level + 1):
            for start_node, start_values in self.subgoal_graph.nodes(data=True):
                start_node_str = start_values['state']
                for end_node, end_values in self.subgoal_graph.nodes(data=True):
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
        for node, values in self.subgoal_graph.nodes(data=True):
            if len(self.subgoal_graph.in_edges(node)) <= 0:
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
        for node, values in self.subgoal_graph.nodes(data=True):
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
        for node, values in self.subgoal_graph.nodes(data=True):
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

            for node in self.subgoal_graph.nodes(data=False):
                if state_node != node:
                    continue
                if nx.has_path(self.subgoal_graph, node, subgoal):
                    return True

            for option in self.specific_onboarding_options:
                onboarding_subgoal = option.end_node
                if not nx.has_path(self.subgoal_graph, onboarding_subgoal, subgoal):
                    continue
                if self.has_path_to_node(state, onboarding_subgoal):
                    return True

            return False

        return initiation_function

    def create_subgoal_graph(
            self
    ) -> nx.MultiDiGraph:

        h_low: int
        h_high: int
        connection_found: bool
        connecting_subgoals: List[str]

        max_height: int = max(self.subgoals.keys())
        self.subgoal_graph = nx.MultiDiGraph()

        for height in self.subgoals:
            for start_subgoal in self.subgoals[height]:
                connecting_subgoals = []
                if not self.subgoal_graph.has_node(start_subgoal):
                    self.subgoal_graph.add_node(start_subgoal)

                # Connections to subgoals of lower heights
                h_low = height - 1
                connection_found = False
                while (h_low >= 1) and (not connection_found):
                    for end_subgoal in self.subgoals[h_low]:
                        if nx.has_path(
                                self.state_transition_graph,
                                start_subgoal,
                                end_subgoal
                        ):
                            connection_found = True
                            connecting_subgoals.append(end_subgoal)
                    h_low -= 1

                # Connections to subgoals of higher heights
                h_high = height + 1
                connection_found = False
                while (h_high <= max_height) and (not connection_found):
                    for end_subgoal in self.subgoals[h_high]:
                        if nx.has_path(
                                self.state_transition_graph,
                                start_subgoal,
                                end_subgoal
                        ):
                            connection_found = True
                            connecting_subgoals.append(end_subgoal)
                    h_high += 1

                # Adding connections to subgoal graph
                for end_subgoal in connecting_subgoals:
                    if not self.subgoal_graph.has_node(end_subgoal):
                        self.subgoal_graph.add_node(end_subgoal)
                    self.subgoal_graph.add_edge(
                        start_subgoal,
                        end_subgoal
                    )

        nx.set_node_attributes(self.subgoal_graph, self.stg_values)
        return self.subgoal_graph

    def find_local_maxima(
            self,
            key: str,
            hops: int
    ) -> List[int]:
        local_maxima: List[int] = []
        local_maxima_key: str = key + "-local-maxima"
        is_subgoal: str
        node_value: float

        for node in self.state_transition_graph.nodes():
            is_subgoal = 'True'

            out_neighbourhood = self.get_out_neighbourhood(int(node), hops)
            in_neighbourhood = self.get_in_neighbourhood(int(node), hops)

            if (len(out_neighbourhood) <= 0) or (len(in_neighbourhood) <= 0):
                self.stg_values[node][local_maxima_key] = 'False'
                continue

            node_value = self.stg_values[node][key]

            for neighbour in out_neighbourhood + in_neighbourhood:
                if int(node) == int(neighbour):
                    continue
                neighbour_value = self.stg_values[str(neighbour)][key]
                if node_value <= neighbour_value:
                    is_subgoal = 'False'
                    break

            self.stg_values[node][local_maxima_key] = is_subgoal

            if is_subgoal == 'True':
                local_maxima.append(int(node))

        return local_maxima

    def find_preparedness_subgoals(
            self,
            find_frequency_entropy_subgoals: bool=False,
            find_neighbourhood_entropy_subgoals: bool=False,
            use_existing_values: bool=False,
            verbose: bool=False
    ):
        subgoals: Dict[int, List[int]] = {}
        frequency_entropy_subgoals: Dict[int, List[int]] = {}
        neighbourhood_entropy_subgoals: Dict[int, List[int]] = {}
        preparedness_subgoals_found: bool = False
        frequency_entropy_subgoals_found: bool = not find_frequency_entropy_subgoals
        neighbourhood_entropy_subgoals_found: bool = not find_neighbourhood_entropy_subgoals
        max_subgoal_height: int = self.max_hierarchy_height
        max_frequency_entropy_height: int = self.max_hierarchy_height
        max_neighbourhood_entropy_height: int = self.max_hierarchy_height

        for hops in range(1, self.max_hierarchy_height + 1):
            if verbose:
                print("Finding Subgoals for height " + str(hops))
                print("     Computing Preparedness at height " + str(hops))

            self.compute_preparedness(hops, use_existing_values, verbose)

            if not preparedness_subgoals_found:
                if verbose:
                    print("     Finding preparedness local maxima at height " + str(hops))
                subgoals[hops] = copy.copy(self.find_local_maxima(self.preparedness_key(hops), hops))

            if not frequency_entropy_subgoals_found:
                if verbose:
                    print("     Finding frequency entropy local maxima at height " + str(hops))
                frequency_entropy_subgoals[hops] = copy.copy(
                    self.find_local_maxima(self.frequency_entropy_key(hops), hops)
                )
            if not neighbourhood_entropy_subgoals_found:
                if verbose:
                    print("     Finding neighbourhood entropy local maxima at height " + str(hops))
                neighbourhood_entropy_subgoals[hops] = copy.copy(
                    self.find_local_maxima(self.neighbourhood_entropy_key(hops), hops)
                )

            if hops > 1:
                if not preparedness_subgoals_found:
                    if subgoals[hops] == subgoals[hops - 1]:
                        preparedness_subgoals_found = True
                        max_subgoal_height = hops
                    elif subgoals[hops] == []:
                        preparedness_subgoals_found = True
                        max_subgoal_height = hops - 1
                    if preparedness_subgoals_found:
                        if verbose:
                            print("Preparedness subgoals found")
                if not frequency_entropy_subgoals_found:
                    if frequency_entropy_subgoals[hops] == frequency_entropy_subgoals[hops - 1]:
                        frequency_entropy_subgoals_found = True
                        max_frequency_entropy_height = hops
                    elif frequency_entropy_subgoals[hops] == []:
                        frequency_entropy_subgoals_found = True
                        max_frequency_entropy_height = hops - 1
                    if frequency_entropy_subgoals_found:
                        if verbose:
                            print("Frequency entropy subgoals found")
                if not neighbourhood_entropy_subgoals_found:
                    if neighbourhood_entropy_subgoals[hops] == neighbourhood_entropy_subgoals[hops - 1]:
                        neighbourhood_entropy_subgoals_found = True
                        max_neighbourhood_entropy_height = hops
                    elif neighbourhood_entropy_subgoals[hops] == []:
                        neighbourhood_entropy_subgoals_found = True
                        max_neighbourhood_entropy_height = hops - 1
                    if neighbourhood_entropy_subgoals_found:
                        if verbose:
                            print("Neighbourhood entropy subgoals found")


            if (
                    preparedness_subgoals_found and
                    frequency_entropy_subgoals_found and
                    neighbourhood_entropy_subgoals_found
            ):
                break

        if not preparedness_subgoals_found and verbose:
            print("Preparedness Subgoal Height maxed-out")
        if not frequency_entropy_subgoals_found and verbose:
            print("Frequency Entropy Subgoal Height maxed-out")
        if not neighbourhood_entropy_subgoals_found and verbose:
            print("Neighbourhood Entropy Subgoal Height maxed-out")

        self.subgoals = self.assign_subgoals(
            subgoals,
            max_subgoal_height,
            'preparedness-subgoal-height'
        )
        self.subgoals_list = []
        for key in self.subgoals:
            self.subgoals_list += self.subgoals[key]

        if find_frequency_entropy_subgoals:
            _ = self.assign_subgoals(
                frequency_entropy_subgoals,
                max_frequency_entropy_height,
                'frequency-entropy-subgoal-height'
            )
        if find_neighbourhood_entropy_subgoals:
            _ = self.assign_subgoals(
                neighbourhood_entropy_subgoals,
                max_neighbourhood_entropy_height,
                'neighbourhood-entropy-subgoal-height'
            )

        nx.set_node_attributes(self.state_transition_graph, self.stg_values)
        return

    def frequency_entropy(
            self,
            node: int,
            hops: int
    ) -> float:
        prob_values: List[float] = []
        out_neighbourhood = self.get_out_neighbourhood(node, hops)
        for neighbour in out_neighbourhood:
            prob_values.append(self.get_random_transition_prob(node, neighbour, hops))

        if sum(prob_values) <= 0:
            return 0.0
        frequency_entropy = stats.entropy(prob_values, base=2)
        return frequency_entropy

    @staticmethod
    def frequency_entropy_key(
            hops: int
    ) -> str:
        return str(hops) + "-frequency-entropy"

    def generic_onboarding_initiation_function(self, state: np.ndarray) -> bool:
        for subgoal in self.subgoals_list:
            if self.has_path_to_node(state, subgoal):
                return True
        return False

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

    def get_in_neighbourhood(
            self,
            node: int,
            hops: int
    ) -> List[int]:
        in_neighbourhood: List[int] = [
            j for j in range(self.distance_matrix.shape[0]) if 0 < self.distance_matrix[j, node] <= hops
        ]
        return in_neighbourhood

    def get_out_neighbourhood(
            self,
            node: int,
            hops: int
    ) -> List[int]:
        out_neighbourhood: List[int] = [
            j for j in range(self.distance_matrix.shape[0]) if 0 < self.distance_matrix[node, j] <= hops
        ]
        return out_neighbourhood

    def get_random_transition_prob(
            self,
            start_node: int,
            end_node: int,
            hops: int
    ) -> float:
        if hops <= 0:
            if start_node == end_node:
                return 1.0
            else:
                return 0.0
        elif hops == 1:
            return self.adjacency_matrix[start_node, end_node]

        try:
            trans_prob = self.random_transition_prob[(start_node, end_node, hops)]
            return trans_prob
        except KeyError:
            ()

        out_neighbourhood = self.get_out_neighbourhood(start_node, 1)
        trans_prob = 0.0
        for out_neighbour in out_neighbourhood:
            trans_prob += (
                    self.adjacency_matrix[start_node, out_neighbour] *
                    self.get_random_transition_prob(out_neighbour, end_node, hops - 1)
            )

        self.random_transition_prob[(start_node, end_node, hops)] = trans_prob
        return trans_prob

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
                self.current_option_step < self.max_option_length and self.max_option_length > np.inf):
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
        level = '0'
        for level in agent_save_file['options between subgoals']:
            if self.max_hierarchy_height is not None:
                if int(level) > self.max_hierarchy_height:
                    continue

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
        self.random_transition_prob = agent_save_file['random transition prob']
        return

    def neighbourhood_entropy(
            self,
            node: int,
            hops: int
    ) -> float:
        out_neighbourhood = self.get_out_neighbourhood(node, hops)
        if node not in out_neighbourhood:
            out_neighbourhood.append(node)
        probs: List[float] = []

        for end_node in out_neighbourhood:
            transition_prob: float = 0.0
            for start_node in out_neighbourhood:
                transition_prob += self.get_random_transition_prob(start_node, end_node, hops)
            probs.append(transition_prob)

        prob_sum: float = sum(probs)
        if prob_sum <= 0.0:
            return 0.0

        probs = [prob / prob_sum for prob in probs]

        neighbourhood_entropy = stats.entropy(probs, base=2)
        return neighbourhood_entropy

    @staticmethod
    def neighbourhood_entropy_key(
            hops: int
    ) -> str:
        return str(hops) + "-neighbourhood-entropy"

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

    @staticmethod
    def preparedness_key(
            hops: int
    ) -> str:
        return str(hops) + "-preparedness"

    def save(self, save_path: str) -> None:
        agent_save_file = {
            'options between subgoals': {level: [{'start node': option.start_node[0],
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
           'max option length': self.max_option_length,
           'random transition prob': self.random_transition_prob
        }

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
        for node, values in self.subgoal_graph.nodes(data=True):
            start_state = self.state_str_to_state(values['state'])
            if option.terminated(start_state):
                continue
            if (node != option.start_node[0]) and (not nx.has_path(self.subgoal_graph, node, option.start_node[0])):
                continue

            path = nx.dijkstra_path(self.subgoal_graph, node, option.end_node)

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
                        state = np.copy(rand.choice(option_start_states))
                    else:
                        state_node = rand.choice(list(self.path_lookup.keys()))
                        state = self.node_to_state(state_node)
                    if environment.is_terminal(state):
                        continue
                    state = np.copy(environment.reset(state))
                    option_initiated = option.initiated(state)
                if not all_actions_possible:
                    possible_actions = environment.get_possible_actions(state)

            action = option.choose_action(state, False, possible_actions)

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
                              for _, values in self.subgoal_graph.nodes(data=True)]
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
