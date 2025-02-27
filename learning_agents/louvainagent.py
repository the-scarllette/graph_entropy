import copy

import igraph as ig
import json
import leidenalg as la
import networkx as nx
import numpy as np
import random as rand
from typing import Dict

from environments.environment import Environment
from learning_agents.optionsagent import OptionsAgent, Option
from learning_agents.multilevelgoalagent import MultiLevelGoalAgent
from learning_agents.qlearningagent import QLearningAgent
from progressbar import print_progress_bar


class LouvainOption(Option):

    def __init__(self, hierarchy_level, source_cluster, target_cluster, policy,
                 initiation_function, termination_function):
        self.hierarchy_level = hierarchy_level
        self.source_cluster = source_cluster
        self.target_cluster = target_cluster
        self.policy = policy
        self.initiation_func = initiation_function
        self.terminating_func = termination_function

        self.can_initiate = {}
        return

    def get_action(self, state):
        if self.hierarchy_level <= 1:
            return self.policy.choose_action(state, True)

        option_values = self.policy.get_state_option_values(state)
        option_chosen = max(option_values, option_values.get)
        return option_chosen.get_action(state)


class LouvainAgent(MultiLevelGoalAgent):
    option_training_failure_reward = -1.0
    option_training_step_reward = -0.001
    option_training_success_reward = 1.0

    def __init__(self, primitive_actions, stg, state_dtype, state_shape, alpha=0.9, epsilon=0.1, gamma=0.9,
                 min_hierarchy_level=0):
        self.primitive_actions = primitive_actions
        self.stg_nx = stg
        self.stg = ig.Graph.from_networkx(self.stg_nx)
        self.state_dtype = state_dtype
        self.aggregate_graphs = None

        self.state_indexer = {}
        self.option_state_initiation_lookup = {}
        self.primitive_options = [Option([action]) for action in primitive_actions]
        self.options = []

        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        self.min_hierarchy_level = min_hierarchy_level
        self.hierarchy_level = None
        self.num_clusters = None

        self.state_shape = state_shape

        self.last_possible_actions = None
        self.current_option = None
        self.option_start_state = None
        self.total_option_reward = 0
        self.current_option_step = 0

        self.state_option_values = {}

        self.action_transition_probs = {}
        self.option_transition_probs = {}

        self.nodes_in_cluster = {}
        self.available_options = {}

        self.num_primitive_options = len(self.primitive_actions)
        return

    def apply_louvain(self,
                      resolution: float = 0.1,
                      partition_type: la.VertexPartition.LinearResolutionParameterVertexPartition = None,
                      return_aggregate_graphs: bool = True,
                      first_levels_to_skip=0, weights=None,
                      state_transition_graph_values=None,
                      graph_save_path=None):
        # Set optimisation metric, define initial partition, initialise optimiser.
        if partition_type is None:
            partition = la.RBConfigurationVertexPartition(self.stg, resolution_parameter=resolution, weights=weights)
        else:
            partition = partition_type(self.stg, resolution_parameter=resolution, weights=weights)
        optimiser = la.Optimiser()

        # Initialise hierarchy level and initial aggregate graph.
        hierarchy_level = 0
        levels = 0
        partition_agg = partition.aggregate_partition()

        if return_aggregate_graphs:
            self.aggregate_graphs = [partition_agg.cluster_graph()]

        while optimiser.move_nodes(
                partition_agg) > 0:  # Move nodes between neighbouring clusters to improve modularity.

            # Derive individual the cluster membership of individual nodes from old aggregate graph.
            partition.from_coarse_partition(partition_agg)

            # Derive new aggregate graph from new cluster memberships.
            partition_agg = partition_agg.aggregate_partition()

            # Store current aggregate graph.
            if levels >= first_levels_to_skip:
                self.stg.vs[f"cluster-{hierarchy_level}"] = partition.membership
                hierarchy_level += 1

                if return_aggregate_graphs:
                    self.aggregate_graphs.append(partition_agg.cluster_graph())

            levels += 1

        self.hierarchy_level = hierarchy_level
        self.num_clusters = [(max(self.stg.vs[f"cluster-{level}"]) + 1) for level in range(self.hierarchy_level)]
        self.option_state_initiation_lookup = {i: {source_cluster: {target_cluster: {}
                                                                    for target_cluster in range(self.num_clusters[i])}
                                                   for source_cluster in range(self.num_clusters[i])}
                                               for i in range(self.hierarchy_level)}
        self.nodes_in_cluster = {level: {} for level in range(self.hierarchy_level)}

        if graph_save_path is not None:
            nx_graph = self.stg.to_networkx()
            nx.write_gexf(nx_graph, graph_save_path)

        if state_transition_graph_values is None:
            return

        for level in range(hierarchy_level):
            key = "cluster-" + str(level)
            for node in self.stg.vs.indices:
                state_transition_graph_values[str(node)][key] = self.stg.vs[key][node]

        return state_transition_graph_values

    def can_initiate_option(self, hierarchy_level, source_cluster, target_cluster, node):
        try:
            can_initiate = self.option_state_initiation_lookup[hierarchy_level][source_cluster][target_cluster][node]
        except KeyError:
            nodes_in_source_cluster = self.get_nodes_in_cluster(hierarchy_level, source_cluster)
            if node not in nodes_in_source_cluster:
                try:
                    self.option_state_initiation_lookup[hierarchy_level][source_cluster][target_cluster][node] = False
                except KeyError:
                    self.option_state_initiation_lookup[str(hierarchy_level)][str(source_cluster)][str(target_cluster)][
                        node] = False
                return False

            nodes_in_target_cluster = self.get_nodes_in_cluster(hierarchy_level, target_cluster)
            can_initiate = False
            for target_node in nodes_in_target_cluster:
                if nx.has_path(self.stg_nx, str(node), str(target_node)):
                    can_initiate = True
                    break
            try:
                self.option_state_initiation_lookup[hierarchy_level][source_cluster][target_cluster][
                    node] = can_initiate
            except KeyError:
                self.option_state_initiation_lookup[str(hierarchy_level)][str(source_cluster)][str(target_cluster)][
                    node] = can_initiate

        return can_initiate

    def choose_option(self, state, no_random, possible_actions=None):
        self.current_option_step = 0
        self.option_start_state = state

        available_options = self.get_available_options(state, possible_actions=possible_actions)
        if len(available_options) == 0:
            return None

        if not no_random and rand.uniform(0, 1) < self.epsilon:
            self.current_option_index = rand.choice(available_options)
            return self.get_option_from_index(self.current_option_index)

        option_values = self.get_state_option_values(state, available_options)

        ops = [available_options[0]]
        try:
            max_value = option_values[available_options[0]]
        except KeyError:
            option_values = {str(option_index): option_values[option_index] for option_index in option_values}
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
        return self.get_option_from_index(self.current_option_index)

    def create_option(self, hierarchy_level, source_cluster, target_cluster):
        def create_initiation_function():
            def initiation_func(state):
                state_index = self.get_state_index(state)
                return self.can_initiate_option(hierarchy_level, source_cluster, target_cluster, state_index)

            return initiation_func

        def create_termination_function():
            def termination_func(state):
                state_index = self.get_state_index(state)
                if state_index is None:
                    return True
                if not self.stg.vs[f"cluster-{hierarchy_level}"][state_index] == source_cluster:
                    return True
                can_initiate = self.can_initiate_option(hierarchy_level, source_cluster, target_cluster, state_index)
                return not can_initiate

            return termination_func

        if hierarchy_level == self.min_hierarchy_level:
            policy = QLearningAgent(self.primitive_actions, self.alpha, self.epsilon, self.gamma)
        else:
            available_sub_options = [option for option in self.options
                                     if option.hierarchy_level == hierarchy_level - 1]
            policy = OptionsAgent(self.alpha, self.epsilon, self.gamma, available_sub_options)
        option = LouvainOption(hierarchy_level, source_cluster, target_cluster, policy,
                               create_initiation_function(), create_termination_function())
        return option

    def create_options(self):
        if self.hierarchy_level is None:
            raise AttributeError("Louvain partition must be found before creating options")

        for level in range(self.min_hierarchy_level, self.hierarchy_level):
            aggregate_graph = self.aggregate_graphs[level + 1]
            for source_cluster in range(self.num_clusters[level]):
                potential_target_clusters = aggregate_graph.neighbors(source_cluster, 'out')
                for target_cluster in potential_target_clusters:
                    if source_cluster == target_cluster:
                        continue
                    option_possible = False
                    nodes_in_source_cluster = self.get_nodes_in_cluster(level, source_cluster)
                    for source_node in nodes_in_source_cluster:
                        option_possible = self.can_initiate_option(level, source_cluster, target_cluster, source_node)
                        if option_possible:
                            break

                    if not option_possible:
                        continue
                    option = self.create_option(level, source_cluster, target_cluster)
                    self.options.append(option)

        return

    def get_available_options(self, state, possible_actions=None):
        state_str = np.array2string(state)

        try:
            available_options = self.available_options[state_str]
        except KeyError:
            available_options = []

            if possible_actions is None:
                available_options = [str(option_index) for option_index in range(self.num_primitive_options)]
                option_index = self.num_primitive_options
            else:
                option_index = 0
                for primitive_option in self.primitive_options:
                    action = primitive_option.actions[0]
                    if action in possible_actions:
                        available_options.append(str(option_index))
                    option_index += 1

            for option in self.options:
                if option.initiation_func(state):
                    available_options.append(str(option_index))
                option_index += 1

            self.available_options[state_str] = available_options

        return available_options

    def get_nodes_in_cluster(self, hierarchy_level, cluster):
        try:
            nodes_in_cluster = self.nodes_in_cluster[hierarchy_level][cluster]
        except KeyError:
            try:
                nodes_in_cluster = self.nodes_in_cluster[str(hierarchy_level)][str(cluster)]
            except KeyError:
                nodes_in_cluster = [node.index for node in self.stg.vs
                                    if self.stg.vs[f"cluster-{hierarchy_level}"][node.index] == cluster]
            try:
                self.nodes_in_cluster[hierarchy_level][cluster] = nodes_in_cluster
            except KeyError:
                self.nodes_in_cluster[str(hierarchy_level)][str(cluster)] = nodes_in_cluster
        return nodes_in_cluster

    def get_state_index(self, state):
        state_str = self.state_to_state_str(state)
        index = None
        try:
            index = self.state_indexer[state_str]
        except KeyError:
            for node in self.stg.vs:
                node_state = self.state_str_to_state(self.stg.vs['state'][node.index])
                if np.array_equal(node_state, state):
                    index = node.index
                    self.state_indexer[state_str] = index
                    break
        return index

    def load(self, load_path):
        with open(load_path, 'r') as f:
            data = json.load(f)

        # Getting Agent Data
        self.state_option_values = data['agent']['policy']
        self.state_indexer = data['agent']['state_indexer']
        self.option_state_initiation_lookup = data['agent']['option_initiation_lookup']
        self.nodes_in_cluster = data['agent']['nodes_in_cluster']
        self.available_options = data['agent']['available_options']

        self.hierarchy_level = -np.inf
        self.num_clusters = []

        # Getting Option Data
        self.options = []
        for option_data in data['options']:
            hierarchy_level = option_data['hierarchy_level']
            option = self.create_option(hierarchy_level,
                                        option_data['source_cluster'], option_data['target_cluster'])
            if hierarchy_level == 0:
                option.policy.q_values = option_data['policy']
            else:
                option.policy.state_option_values = option_data['policy']

            self.options.append(option)

            if hierarchy_level > self.hierarchy_level:
                self.hierarchy_level = hierarchy_level
                self.num_clusters.append(0)

            self.num_clusters[hierarchy_level] += 1

        return

    def print_options(self):
        level = -1
        source_cluster = -1
        for option in self.options:
            if level != option.hierarchy_level:
                level = option.hierarchy_level
                print("Options for Level " + str(level) + ":")

            print_text = "            "
            if source_cluster != option.source_cluster:
                source_cluster = option.source_cluster
                print_text = "   cluster " + str(source_cluster)
            else:
                if source_cluster >= 10:
                    print_text += " "

            print_text += " -> " + str(option.target_cluster)

            print(print_text)
        return

    def state_index_to_state(self, state_index):
        state = self.stg.vs['state'][state_index]
        state = self.state_str_to_state(state)
        return state

    def save(self, save_path):
        data = {"options": []}

        # Saving Option Policies
        for option in self.options:
            option_data = {'hierarchy_level': option.hierarchy_level,
                           'source_cluster': option.source_cluster,
                           'target_cluster': option.target_cluster}
            if option.hierarchy_level == self.min_hierarchy_level:
                option_data['policy'] = option.policy.q_values
            else:
                option_data['policy'] = option.policy.state_option_values
            data['options'].append(copy.copy(option_data))

        # Saving Agent Policy
        data['agent'] = {'policy': self.state_option_values,
                         'state_indexer': self.state_indexer,
                         'option_initiation_lookup': self.option_state_initiation_lookup,
                         'nodes_in_cluster': self.nodes_in_cluster,
                         'available_options': self.available_options}

        # Saving Data
        with open(save_path, 'w') as f:
            json.dump(data, f)
        return

    def train_option(self, option: LouvainOption, num_training_steps: int, environment: Environment,
                     all_actions_valid=True,
                     progress_bar=False):
        nodes_in_source_cluster = self.get_nodes_in_cluster(option.hierarchy_level,
                                                            option.source_cluster)
        start_states = []
        for node in nodes_in_source_cluster:
            start_state = self.stg.vs['state'][node]
            start_state = self.state_str_to_state(start_state)
            if environment.is_terminal(start_state):
                continue
            if option.initiation_func(start_state):
                start_states.append(copy.deepcopy(start_state))

        done = True
        current_step = 0
        while current_step < num_training_steps:
            if progress_bar:
                print_progress_bar(current_step, num_training_steps,
                                   prefix='Training Option from cluster '
                                          + str(option.source_cluster) + ' to cluster ' + str(option.target_cluster),
                                   suffix='Complete')

            if done:
                start_state = rand.choice(start_states)
                state = environment.reset(start_state)
                current_possible_actions = environment.get_possible_actions(state)
                done = False

            action = option.choose_action(state, current_possible_actions)
            next_state, _, done, _ = environment.step(action)
            current_step += 1
            reward = self.option_training_step_reward

            if option.terminated(next_state) or done:
                done = True
                reward = self.option_training_failure_reward
                if self.stg.vs[f"cluster-{option.hierarchy_level}"][self.get_state_index(next_state)] == \
                        option.target_cluster:
                    reward = self.option_training_success_reward

            if all_actions_valid:
                option.policy.learn(state, action, reward, next_state, terminal=done)
            else:
                current_possible_actions = environment.get_possible_actions(next_state)
                option.policy.learn(state, action, reward, next_state, terminal=done,
                                    next_state_possible_actions=current_possible_actions)
            state = next_state

        return

    def train_option_value_iteration(self, option: LouvainOption, environment: Environment, final_delta: float,
                                     option_rollouts: int=50, all_actions_valid: bool=False):
        primitive_option = option.hierarchy_level <= 0

        def t(start_state: np.ndarray, o: LouvainOption | int) -> Dict[str, float]:
            start_state_str = self.state_to_state_str(start_state)
            if primitive_option:
                try:
                    probabilities = self.action_transition_probs[(start_state_str, o)]
                    return probabilities
                except KeyError:
                    self.action_transition_probs[(start_state_str, o)] = {}
                    for _ in range(option_rollouts):
                        _ = environment.reset(start_state)
                        next_state, _, done, _ = environment.step(o)
                        next_state_str = self.state_to_state_str(next_state)
                        try:
                            self.action_transition_probs[(start_state_str, o)][next_state_str] += (1 / option_rollouts)
                        except KeyError:
                            self.action_transition_probs[(start_state_str, o)][next_state_str] = (1 / option_rollouts)
                    probabilities = self.action_transition_probs[(start_state_str, o)]
                    return probabilities

            try:
                probabilities = self.option_transition_probs[(start_state_str,
                                                            o.hierarchy_level,
                                                            o.source_cluster,
                                                            o.target_cluster)]
                return probabilities
            except KeyError:
                self.option_transition_probs[(start_state_str,
                                              o.hierarchy_level,
                                              o.source_cluster,
                                              o.target_cluster)] = {}
                for _ in range(option_rollouts):
                    next_state = environment.reset(start_state)
                    option_terminal =  False
                    while not option_terminal:
                        action = o.choose_action(next_state, environment.get_possible_actions(next_state))
                        next_state, _, done, _ = environment.step(action)
                        option_terminal = done or o.terminated(next_state)
                    next_state_str = self.state_to_state_str(next_state)
                    try:
                        self.option_transition_probs[(start_state_str,
                                                      o.hierarchy_level,
                                                      o.source_cluster,
                                                      o.target_cluster)][next_state_str] += (1 / option_rollouts)
                    except KeyError:
                        self.option_transition_probs[(start_state_str,
                                                      o.hierarchy_level,
                                                      o.source_cluster,
                                                      o.target_cluster)][next_state_str] = (1 / option_rollouts)
                probabilities = self.option_transition_probs[(start_state_str,
                                                        o.hierarchy_level,
                                                        o.source_cluster,
                                                        o.target_cluster)]
                return probabilities

        def v(s: np.ndarray) -> float:
            if primitive_option:
                state_values = option.policy.get_action_values(s)
            else:
                state_values = option.policy.get_state_option_values(s)
            if not state_values.values():
                return 0.0
            return max(state_values.values())

        delta = np.inf
        possible_actions = environment.possible_actions

        nodes = []
        clusters = (list(self.aggregate_graphs[option.hierarchy_level].neighbors(option.source_cluster)) +
                    [option.source_cluster])
        for cluster in clusters:
            nodes += self.get_nodes_in_cluster(option.hierarchy_level, cluster)

        while delta >= final_delta:
            delta = 0
            for node in nodes:
                state = self.state_index_to_state(node)
                if option.terminated(state):
                    continue

                temp = v(state)

                if not all_actions_valid:
                    possible_actions = environment.get_possible_actions(state)
                if primitive_option:
                    possible_options = possible_actions
                else:
                    possible_options = option.policy.get_available_options(state, possible_actions)

                option_values = {possible_option: 0.0 for possible_option in possible_options}
                for possible_option in possible_options:
                    # if node == 822 and possible_option == 4:
                    #     pass

                    if primitive_option:
                        transition_probabilities = t(state, possible_option)
                    else:
                        transition_probabilities = t(state, option.policy.options[possible_option])
                    option_value = 0.0
                    for tilde_state_str in list(transition_probabilities.keys()):
                        tilde_state = self.state_str_to_state(tilde_state_str)

                        transition_prob = transition_probabilities[tilde_state_str]
                        if transition_prob <= 0.0:
                            continue

                        tilde_node = self.get_state_index(tilde_state)

                        reward = self.option_training_failure_reward
                        if self.stg.vs[f"cluster-{option.hierarchy_level}"][tilde_node] == \
                                option.source_cluster:
                            reward = self.option_training_step_reward
                        elif self.stg.vs[f"cluster-{option.hierarchy_level}"][tilde_node] == \
                                option.target_cluster:
                            reward = self.option_training_success_reward

                        option_value += transition_prob * (reward + (self.gamma * v(tilde_state)))

                    option_values[possible_option] = option_value
                if primitive_option:
                    option.policy.q_values[self.state_to_state_str(state)] = option_values
                else:
                    option.policy.set_state_option_values(option_values, state)

                delta = max(delta, abs(temp - v(state)))

        return

    def train_options(self, num_training_steps: int, environment: Environment,
                      all_actions_valid: bool=True,
                      progress_bar: bool=False):
        if progress_bar:
            print("Training Louvain Options")

        level = -1
        for option in self.options:
            if not option.hierarchy_level == level:
                level = option.hierarchy_level
                if progress_bar:
                    print("Training Options for Level " + str(level))
            self.train_option(option, num_training_steps, environment, all_actions_valid, progress_bar)

        return

    def train_options_value_iteration(self, final_delta: float, environment: Environment,
                                      option_rollouts: int=50,
                                      all_actions_valid: bool=True,
                                      progress_bar: bool=False):
        level = -1
        total_options = len(self.options)
        i = 0
        for option in self.options:
            if not option.hierarchy_level == level:
                level = option.hierarchy_level
                if progress_bar:
                    print("Training Options for Level " + str(level))
            if progress_bar:
                print_progress_bar(i, total_options,
                                   '    ',
                                   "Complete")
            self.train_option_value_iteration(option, environment,
                                              final_delta, option_rollouts,
                                              all_actions_valid)
            i += 1

        return
