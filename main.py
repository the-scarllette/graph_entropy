import copy
import json
import math
import os
import random

import networkx as nx
import numpy as np
import scipy.sparse
from scipy import sparse
import time
from typing import Dict, List, Tuple

import environments.environment
from environments.environment import Environment
from environments.lavaflow import LavaFlow
from environments.taxicab import TaxiCab
from environments.tinytown import TinyTown
from environments.waterbucket import WaterBucket
import graphing
from learning_agents.betweennessagent import BetweennessAgent
from learning_agents.eigenoptionagent import EigenOptionAgent
from learning_agents.learningagent import LearningAgent
from learning_agents.louvainagent import LouvainAgent
from learning_agents.multilevelgoalagent import MultiLevelGoalAgent
from learning_agents.optionsagent import Option, OptionsAgent, create_option_goal_initiation_func, \
    generate_option_to_goal
from learning_agents.preparednessagent import PreparednessAgent
from learning_agents.qlearningagent import QLearningAgent
from learning_agents.subgoalagent import SubgoalAgent
from progressbar import print_progress_bar


def add_eigenoptions_to_stg(agent: EigenOptionAgent, environment: Environment):
    environment_filenames = get_filenames(environment)
    stg_filename = filenames[2]
    stg_values_filename = filenames[3]

    with open(stg_values_filename , 'r') as f:
        state_transition_graph_values = json.load(f)

    for node in state_transition_graph_values:
        state_transition_graph_values[node]['eigenoption_goal'] = 'False'
    for i in range(agent.num_options):
        option = agent.options[i]
        state_transition_graph_values[str(option.goal_index)]['eigenoption_goal'] = 'True'

    with open(stg_values_filename, 'w') as f:
        json.dump(state_transition_graph_values, f)
    state_transition_graph = nx.read_gexf(stg_filename)
    nx.set_node_attributes(state_transition_graph, state_transition_graph_values)
    nx.write_gexf(state_transition_graph, stg_filename)
    return


def add_local_maxima_to_file(env_name, key, num_hops=1, compressed_matrix=False, progress_bar=False):
    stg_values_filename = env_name + '_stg_values.json'
    adj_matrix_filename = env_name + '_adj_matrix.txt'

    if compressed_matrix:
        adj_matrix = sparse.load_npz(adj_matrix_filename + '.npz')
    else:
        adj_matrix = np.loadtxt(adj_matrix_filename)

    with open(stg_values_filename, 'r') as f:
        stg_values = json.load(f)

    stg_values = {int(state): stg_values[state] for state in stg_values}
    progress_bar_prefix = None
    if progress_bar:
        progress_bar_prefix = 'Finding local Maxima ' + str(num_hops) + ' hops'
    local_maxima = find_local_maxima(adj_matrix, stg_values, num_hops, key, progress_bar_prefix)

    local_maxima_key = key + ' - local maxima'

    for state in stg_values:
        if adj_matrix[:, state].sum() <= 0:
            local_maxima[int(state)] = False
        stg_values[state][local_maxima_key] = str(local_maxima[int(state)])

    with open(stg_values_filename, 'w') as f:
        json.dump(stg_values, f)

    if compressed_matrix:
        return

    stg_values = {str(state): stg_values[state] for state in stg_values}

    stg_filename = env_name + '_stg.gexf'
    stg = nx.read_gexf(stg_filename)
    nx.set_node_attributes(stg, stg_values)
    nx.write_gexf(stg, stg_filename)
    return


def add_preparedness_subgoals_to_file(env_name, min_num_subgoals, min_num_hops=1, max_num_hops=5,
                                      use_stg=False, beta=None):
    stg_values_filename = env_name + '_stg_values.json'

    if use_stg:
        stg_filename = env_name + '_stg.gexf'
        stg = nx.read_gexf(stg_filename)

        def subgoal_function(x):
            return find_preparedness_subgoals(x, min_num_subgoals, stg=stg)
    else:
        adj_matrix_filename = env_name + '_adj_matrix.txt.npz'
        adj_matrix = sparse.load_npz(adj_matrix_filename)

        def subgoal_function(x):
            return find_preparedness_subgoals(x, min_num_subgoals, adj_matrix=adj_matrix)

    with open(stg_values_filename, 'r') as f:
        data = json.load(f)

    for num_hops in range(min_num_hops, max_num_hops + 1):
        key = 'preparedness - ' + str(num_hops) + ' hops'
        if beta is not None:
            key += ' - beta = ' + str(beta)

        nodes_preparedness = {node: data[node][key] for node in data}
        subgoals = subgoal_function(nodes_preparedness)

        key += ' subgoal'
        for node in data:
            data[node][key] = node in subgoals

    with open(stg_values_filename, 'w') as f:
        json.dump(data, f)

    if not use_stg:
        return
    nx.set_node_attributes(stg, data)
    nx.write_gexf(stg, stg_filename)
    return




def add_subgoals(env_name,
                 compressed_matrix=False,
                 beta=0.5, beta_values=None, min_num_hops=1, max_num_hops=10,
                 log_base=2, add_betweenness=False,
                 accuracy=6):
    adj_matrix_filename = env_name + '_adj_matrix.txt'
    stg_values_filename = env_name + '_stg_values.json'
    stg_filename = env_name + '_stg.gexf'

    if compressed_matrix:
        adj_matrix = sparse.load_npz(adj_matrix_filename + '.npz')
    else:
        adj_matrix = np.loadtxt(adj_matrix_filename)

    with open(stg_values_filename, 'r') as f:
        stg_values = json.load(f)

    if min_num_hops > 0:
        print("Finding Preparedess Values")
        if beta_values is None:
            beta_values = [beta]

        for hops in range(min_num_hops, max_num_hops + 1):
            preparedness_values = preparedness(adj_matrix, None, beta_values,
                                               min_num_hops, hops, log_base, accuracy)
            for state in stg_values:
                stg_values[state].update(preparedness_values[int(state)])

            with open(stg_values_filename, 'w') as f:
                json.dump(stg_values, f)

    if add_betweenness:
        print("Finding betweenness")
        betweenness_values = compute_betweeness(adj_matrix)
        betweenness_local_maxima = find_local_maxima(adj_matrix, betweenness_values, key='betweenness')

        for state in stg_values:
            stg_values[state]['betweenness'] = betweenness_values[int(state)]
            is_betweenness_local_maxima = betweenness_local_maxima[int(state)]
            stg_values[state]['betweenness local maxima'] = str(is_betweenness_local_maxima)

        with open(stg_values_filename, 'w') as f:
            json.dump(stg_values, f)

    if compressed_matrix:
        return

    stg = nx.read_gexf(stg_filename)
    nx.set_node_attributes(stg, stg_values)
    nx.write_gexf(stg, stg_filename)
    return


def betweenness(stg: nx.Graph, existing_stg_values=None):
    # Finding Betweenness Centraility Values
    betweenness_values = nx.betweenness_centrality(stg, normalized=True)

    if existing_stg_values is None:
        existing_stg_values = {key: {'betweenness': betweenness_values[key]}
                               for key in betweenness_values}
    else:
        for key in betweenness_values:
            existing_stg_values[key].update({'betweenness': betweenness_values[key]})

    # Finding Subgoals
    for node in existing_stg_values:
        betweenness_value = existing_stg_values[node]['betweenness']
        local_maxima_str = 'True'
        for adjacent_node  in stg.neighbors(node):
            adjacent_betweenness_value = existing_stg_values[adjacent_node]['betweenness']
            if betweenness_value <= adjacent_betweenness_value:
                local_maxima_str = 'False'
                break
        existing_stg_values[node]['betweenness - local maxima'] = local_maxima_str

    # Updating stg
    nx.set_node_attributes(stg, existing_stg_values)

    return stg, existing_stg_values


def count_clusters(environment: Environment, cluster_key: str, count_states: bool=False)\
    -> Dict[int, int] | Tuple[Dict[int, int], int]:
    env_filenames = get_filenames(environment)
    with open(env_filenames['state transition graph values'], 'r') as f:
        state_transition_graph_values = json.load(f)
    cluster_count = {}
    num_states = 0
    cluster_level = 0
    all_levels_counted = False
    clusters_found = False

    while not all_levels_counted:
        key = cluster_key + "-" + str(cluster_level)
        clusters_at_level = 0

        for node in state_transition_graph_values:
            try:
                node_cluster = state_transition_graph_values[node][key]
                clusters_found = True
            except KeyError:
                if clusters_found:
                    all_levels_counted = True
                break
            if node_cluster > clusters_at_level:
                clusters_at_level = node_cluster
            num_states += 1

        if not all_levels_counted:
            cluster_count[cluster_level] = clusters_at_level
        cluster_level += 1

    if not count_states:
        return cluster_count

    num_states = num_states / cluster_level
    return cluster_count, num_states


def count_subgoals(environment: Environment, subgoal_key: str, multiple_levels: bool=False,
                   count_states: bool=False)\
        -> Dict[int, int] | Tuple[Dict[int, int], int]:
    env_filenames = get_filenames(environment)
    with open(env_filenames['state transition graph values'], 'r') as f:
        state_transition_graph_values = json.load(f)
    subgoal_count = {}
    num_states = 0

    if not multiple_levels:
        subgoal_count[1] = 0
        for node in state_transition_graph_values:
            num_states += 1
            if state_transition_graph_values[node][subgoal_key] == 'True':
                subgoal_count[1] += 1
        if not count_states:
            return subgoal_count
        return subgoal_count, num_states

    for node in state_transition_graph_values:
        num_states += 1
        node_subgoal_level = state_transition_graph_values[node][subgoal_key]

        if node_subgoal_level == 'None':
            continue

        try:
            subgoal_count[int(node_subgoal_level)] += 1
        except KeyError:
            subgoal_count[int(node_subgoal_level)] = 1

    if not count_states:
        return subgoal_count
    return subgoal_count, num_states


def compute_betweeness(adj_matrix):
    env_nodes = range(adj_matrix.shape[0])

    def get_neighbours(node_to_get):
        return [i for i in env_nodes if i != node_to_get and adj_matrix[node_to_get, i] > 0]

    node_betweenness_array = [0 for _ in env_nodes]
    for node in env_nodes:
        print_progress_bar(node, adj_matrix.shape[0],
                           prefix='Betweenness: ', suffix=" complete")
        S = []
        S_len = 0
        Q = [node]
        Q_len = 1
        d = [-1 for _ in env_nodes]
        sigma = [0 for _ in env_nodes]
        d[node] = 0
        sigma[node] = 1
        P = [[] for _ in env_nodes]

        while Q_len > 0:
            v = Q.pop(0)
            Q_len -= 1
            S.append(v)
            S_len += 1
            v_neighbours = get_neighbours(v)
            for w in v_neighbours:
                if d[w] < 0:
                    Q.append(w)
                    Q_len += 1
                    d[w] = d[v] + 1
                if d[w] == d[v] + 1:
                    sigma[w] += sigma[v]
                    P[w].append(v)

        delta = [0 for _ in env_nodes]
        while S_len > 0:
            w = S.pop(S_len - 1)
            S_len -= 1
            for v in P[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
                if w != node:
                    node_betweenness_array[w] += delta[w]

    betweenness_values = {i: {'betweenness': node_betweenness_array[i]}
                          for i in env_nodes}

    return betweenness_values


def compute_entropy(distribution, log_base=10):
    entropy = 0
    for prob in distribution:
        if prob <= 0:
            continue
        entropy -= prob * np.emath.logn(log_base, prob)
    return entropy


def create_subgoal_graph(state_transition_graph: nx.MultiDiGraph,
                         stg_values : Dict[str, Dict[str, str | float | int]],
                         subgoals: Dict[str, List[str]],
                         max_distance: int=np.inf) -> (
        nx.MultiDiGraph, nx.MultiDiGraph, Dict[str, Dict[str, float|str]]):
    def connect_nodes(node_1: str, node_2: str) -> bool:
        return nx.has_path(state_transition_graph, node_1, node_2)
    if max_distance != np.inf:
        def connect_nodes(node_1: str, node_2: str) -> bool:
            shortest_path_distance = nx.shortest_path_length(state_transition_graph, node_1, node_2)
            return shortest_path_distance <= max_distance

    # Building Aggregate Graph
    aggregate_graph = nx.MultiDiGraph()
    subgoal_heights = [int(height) for height in subgoals.keys()]
    min_height = subgoal_heights[0]
    max_height = subgoal_heights[-1]
    for subgoal_height in subgoals:
        for node in subgoals[subgoal_height]:
            aggregate_graph.add_node(node)

        # Path Between Subgoals
        for subgoal_height in range(min_height, max_height + 1):
            for subgoal in subgoals[subgoal_height]:
                # Path of Decreasing Preparedness
                increasing_path_found = False
                start_height = subgoal_height + 1
                while (start_height <= max_height) and (not increasing_path_found):
                    for start_node in subgoals[start_height]:
                        if connect_nodes(start_node, subgoal):
                            aggregate_graph.add_edge(start_node, subgoal, weight=1.0)
                            increasing_path_found = True
                    start_height += 1

                # Paths of Increasing Preparedness
                decreasing_path_found = False
                start_height = subgoal_height - 1
                while (min_height <= start_height) and (not decreasing_path_found):
                    for start_node in subgoals[start_height]:
                        if connect_nodes(start_node, subgoal):
                            aggregate_graph.add_edge(start_node, subgoal, weight=1.0)
                            decreasing_path_found = True
                    start_height -= 1
    nx.set_node_attributes(aggregate_graph, stg_values)

    return state_transition_graph, aggregate_graph, stg_values


def extract_graph_entropy_values(values_dict):
    extracted_graph_entropies = {values_dict[node]['state']: values_dict[node]['graph entropy']
                                 for node in values_dict}
    return extracted_graph_entropies

def find_flat_subgoals(stg_values: Dict[str, Dict[str, str|float]],
                       subgoal_key: str) -> List[str]:
    subgoals = []

    for node in stg_values:
        if stg_values[node][subgoal_key] != "None":
            subgoals.append(node)

    return subgoals


def find_local_maxima(adjacency_matrix, values, num_hops=1, key=None, progress_bar_prefix=None):
    nodes = range(adjacency_matrix.shape[0])
    local_maxima = []

    for node in nodes:
        if progress_bar_prefix is not None:
            print_progress_bar(node, adjacency_matrix.shape[0], progress_bar_prefix)

        connected_nodes = get_neighbours(adjacency_matrix, node, num_hops, directed=False)
        if key is None:
            is_maxima = all([values[node] > values[connected_node] for connected_node in connected_nodes])
        else:
            node_value = values[node][key]
            is_maxima = True
            for connected_node in connected_nodes:
                if connected_node == node:
                    continue
                if values[connected_node][key] >= values[node][key]:
                    is_maxima = False
                    break
        local_maxima.append(is_maxima)
    return local_maxima


def find_preparedness_subgoals(preparedness_values, min_num_subgoals, stg=None, adj_matrix=None):
    if (stg is None) and (adj_matrix is None):
        raise AttributeError("One of stg or adj_matrix must not be None")

    sorted_keys = sorted(preparedness_values, key=lambda x: preparedness_values[x], reverse=True)

    if stg is not None:
        def has_no_in_edges(x):
            return len(stg.in_edges(x)) <= 0
    else:
        def has_no_in_edges(x):
            return adj_matrix.getcol(x).sum() <= 0

    subgoals = []
    subgoals_added = 0
    for subgoal in sorted_keys:
        if has_no_in_edges(subgoal):
            continue

        subgoals.append(subgoal)
        subgoals_added += 1
        if subgoals_added >= min_num_subgoals:
            break

    k = min_num_subgoals
    smallest_passing_value = preparedness_values[subgoals[-1]]
    while smallest_passing_value == preparedness_values[sorted_keys[k]]:
        subgoals.append(sorted_keys[k])
        k += 1

    return subgoals


def find_save_environment_stg(env: Environment, save_file_prefix: str, compressed=False,
                              progress_bar=False):
    adjacency_matrix_save_path = save_file_prefix + '_adj_matrix.npz'
    all_states_save_path = save_file_prefix + '_all_states.npy'
    stg_save_path = save_file_prefix + '_stg.gexf'
    stg_values_save_path = save_file_prefix + '_stg_values.json'

    adjacency_matrix, all_states = env.get_adjacency_matrix(True, True, compressed,
                                                            progress_bar=progress_bar)
    all_states_numpy = np.array(all_states)

    if progress_bar:
        print(str(all_states_numpy.shape[0]) + " states found")

    state_transition_graph_values = {str(node): {'state': np.array2string(all_states_numpy[node])}
                                     for node in range(adjacency_matrix.shape[0])}

    np.save(all_states_save_path, all_states_numpy)
    with open(stg_values_save_path, 'w') as f:
        json.dump(state_transition_graph_values, f)
    if not compressed:
        graph = nx.from_numpy_array(adjacency_matrix)
        nx.set_node_attributes(graph, state_transition_graph_values)
        np.save(adjacency_matrix_save_path, adjacency_matrix)
        nx.write_gexf(graph, stg_save_path)
        return

    graph = nx.from_scipy_sparse_array(adjacency_matrix)
    nx.set_node_attributes(graph, state_transition_graph_values)
    nx.write_gexf(graph, stg_save_path)
    sparse.save_npz(adjacency_matrix_save_path, adjacency_matrix)
    return


def find_save_stg_subgoals(env: Environment, env_name, probability_weights=False, compressed_matrix=False,
                           state_labels=None,
                           beta=0.5, beta_values=None, min_num_hops=1, max_num_hops=10, log_base=2,
                           find_betweenness=False,
                           accuracy=6):
    adj_matrix_filename = env_name + '_adj_matrix.txt'
    stg_values_filename = env_name + '_stg_values.json'
    stg_filename = env_name + '_stg.gexf'

    if beta_values is None:
        beta_values = [beta]

    print("Finding Adjacency Matrix")
    adj_matrix, all_states = env.get_adjacency_matrix(directed=True, probability_weights=probability_weights,
                                                      compressed_matrix=compressed_matrix)
    print("Finding Preparedness")
    preparedness_values = preparedness(adj_matrix, None, beta_values, min_num_hops, max_num_hops, log_base, accuracy)

    if find_betweenness:
        print("Finding Betweenness")
        betweenness_values = compute_betweeness(adj_matrix)
        print("Finding Betweenness Local Maxima")
        betweenness_local_maxima = find_local_maxima(adj_matrix, betweenness_values, key='betweenness')

    num_state_labels = None
    if state_labels is not None:
        num_state_labels = len(state_labels)

    betweeness_subgoal_indexes = []
    for i in preparedness_values:
        state = all_states[i]
        preparedness_values[i]['state'] = np.array2string(state)

        if find_betweenness:
            preparedness_values[i]['betweenness'] = float(betweenness_values[i]['betweenness'])
            is_betweenness_local_maxima = betweenness_local_maxima[i]
            preparedness_values[i]['betweenness local maxima'] = str(is_betweenness_local_maxima)
            if is_betweenness_local_maxima:
                betweeness_subgoal_indexes.append(i)

        if state_labels is not None:
            for k in range(num_state_labels):
                preparedness_values[i][state_labels[k]] = int(state[k])

    f = open(stg_values_filename, 'w')
    f.close()
    with open(stg_values_filename, 'w') as f:
        json.dump(preparedness_values, f)

    if compressed_matrix:
        sparse.save_npz(adj_matrix_filename, adj_matrix)
        return
    np.savetxt(adj_matrix_filename, adj_matrix)

    g = nx.from_numpy_array(adj_matrix, create_using=nx.MultiDiGraph)
    nx.set_node_attributes(g, preparedness_values)
    nx.write_gexf(g, stg_filename)
    return


def get_filenames(env: Environment) -> Dict[str, str]:
    adj_matrix_filename = env.environment_name + '_adj_matrix.npz'
    all_states_filename = env.environment_name + '_all_states.npy'
    stg_filename = env.environment_name + '_stg.gexf'
    stg_values_filename = env.environment_name + '_stg_values.json'
    agent_directory = env.environment_name + '_agents'
    results_directory = env.environment_name + '_episode_results'
    preparedness_aggregate_graph = env.environment_name + '_preparedness_aggregate_graph.gexf'
    frequency_entropy_aggregate_graph = env.environment_name + '_frequency_entropy_subgoal_graph.gexf'
    neighbourhood_entropy_aggregate_graph = env.environment_name + '_neighbourhood_entropy_subgoal_graph.gexf'
    return {'adjacency matrix': adj_matrix_filename,
            'all states': all_states_filename,
            'state transition graph': stg_filename,
            'state transition graph values': stg_values_filename,
            'agents': agent_directory,
            'results': results_directory,
            'preparedness aggregate graph': preparedness_aggregate_graph,
            'frequency entropy subgoal graph': frequency_entropy_aggregate_graph,
            'neighbourhood entropy subgoal graph': neighbourhood_entropy_aggregate_graph}


def get_neighbours(adjacency_matrix: np.matrix, node, num_hops=1, directed=True, compressed_matrix=True):
    if num_hops <= 0:
        return []

    num_hops -= 1
    N = adjacency_matrix.shape[0]
    if directed:
        if not compressed_matrix:
            immediate_neighbours = np.where(adjacency_matrix[node, :] > 0)
        else:
            immediate_neighbours = adjacency_matrix[node, :].nonzero()[1]
    else:
        immediate_neighbours = np.where(adjacency_matrix[node, :] > 0 | adjacency_matrix[:, node] > 0)

    if num_hops <= 0:
        return immediate_neighbours

    neighbours = []
    for neighbour in immediate_neighbours:
        found_neighbours = get_neighbours(adjacency_matrix, neighbour, num_hops)
        for found_neighbour in found_neighbours:
            if found_neighbour in neighbours or found_neighbour in immediate_neighbours:
                continue
            neighbours.append(found_neighbour)

    if compressed_matrix:
        return np.append(neighbours, immediate_neighbours)

    neighbours += immediate_neighbours
    return neighbours


def get_preparedness_key(hops, beta=None):
    key = 'preparedness - ' + str(hops) + ' hops'
    if beta is None:
        return key
    key += ' - beta = ' + str(beta)
    return key


def get_preparedness_subgoals(environment: Environment, beta=None, compressed_matrix=False):
    stg_values_filename = environment.environment_name + '_stg_values.json'
    with open(stg_values_filename, 'r') as f:
        stg_values = json.load(f)

    all_goals_found = False
    subgoals = []
    last_subgoals = []
    max_hops = 1
    while not all_goals_found:
        current_subgoals = []
        preparedness_key = get_preparedness_key(max_hops, beta) + ' - local maxima'
        for node in stg_values:
            try:
                is_subgoal = stg_values[node][preparedness_key]
                if is_subgoal == 'True':
                    subgoal_str = stg_values[node]['state']
                    current_subgoals.append(subgoal_str)
                    if subgoal_str in last_subgoals:
                        last_subgoals.remove(subgoal_str)

            except KeyError:
                all_goals_found = True
                max_hops -= 1
                break

        if (max_hops > 1) and (last_subgoals is None):
            all_goals_found = True
            break

        last_subgoals = current_subgoals.copy()
        subgoals.append(current_subgoals.copy())
        max_hops += 1

    # Pruning Subgoals
    for current_hops in range(max_hops - 2, -1, -1):
        min_hops_to_prune = current_hops + 1
        subgoals_to_prune = []
        for i in range(min_hops_to_prune, max_hops):
            subgoals_to_prune += subgoals[i]

        for subgoal_to_prune in subgoals_to_prune:
            try:
                subgoals[current_hops].remove(subgoal_to_prune)
            except ValueError:
                ()

    # Removing empty lists
    filtered_subgoals = []
    for subgoal_list in subgoals:
        if not subgoal_list:
            max_hops -= 1
            continue
        filtered_subgoals.append(subgoal_list)

    return filtered_subgoals


def get_state_transition_graph(environment: Environment, directed: bool=True, probability_weights: bool=True,
                               compressed_matrix: bool=True, progress_bar: bool=True):
    filenames = get_filenames(environment)
    adjacency_matrix, state_transition_graph, state_transition_graph_values = environment.get_adjacency_matrix(
        directed, probability_weights, compressed_matrix, progress_bar=progress_bar)

    sparse.save_npz(filenames['adjacency matrix'], adjacency_matrix)
    with open(filenames['state transition graph values'], 'w') as f:
        json.dump(state_transition_graph_values, f)
    nx.write_gexf(state_transition_graph, filenames['state transition graph'])
    return


def get_subgoals_from_folder(folder_path):
    subgoal_file_names = os.listdir(folder_path)
    subgoals = [np.loadtxt(folder_path + '/' + subgoal_file_name)
                for subgoal_file_name in subgoal_file_names]
    return subgoals


def get_undirected_connected_nodes(adjacency_matrix, node):
    connected_nodes = []
    for i in range(adjacency_matrix.shape[0]):
        if adjacency_matrix[node, i] > 0 or adjacency_matrix[i, node] > 0:
            connected_nodes.append(i)
    return connected_nodes


def graph_available_skills(env: Environment, agents: List[OptionsAgent],
                           agent_labels: List[str],
                           all_actions_valid: bool=False,
                           graph_name: None|str=None,
                           colours: None|List[str]=None,
                           save_data_path: None|str=None,
                           verbose: bool=True):
    agents_len = len(agents)
    possible_actions = env.possible_actions
    env_filenames = get_filenames(env)
    with open(env_filenames['state transition graph values'], 'r') as f:
        env_state_values = json.load(f)
    plot_data = [[] for _ in range(agents_len)]

    if verbose:
        print("Counting Skills")
        num_nodes = len(env_state_values)

    for i in range(agents_len):
        agent = agents[i]

        if verbose:
            print("Counting " + agent_labels[i] + " skills")

        for node in env_state_values:
            if verbose:
                print_progress_bar(int(node), num_nodes, agent_labels[i], 'Complete')

            state_str = env_state_values[node]['state']
            state = agent.state_str_to_state(state_str)
            if not all_actions_valid:
                possible_actions = env.get_possible_actions(state)

            plot_data[i].append(agent.count_available_skills(state, possible_actions))

    if verbose:
        print("Plotting data")

    graphing.graph_multiple(plot_data,
                            name=graph_name,
                            labels=agent_labels,
                            x_label="State",
                            y_label="Number of Available Skills",
                            colours=colours,
                            no_xticks=True)

    if save_data_path is not None:
        if os.path.exists(save_data_path):
            with open(save_data_path, 'r') as f:
                save_data = json.load(f)
                for i in range(agents_len):
                    save_data[agent_labels[i]] = plot_data[i]
        else:
            save_data = {agent_labels[i]: plot_data[i] for i in range(agents_len)}
        with open(save_data_path, 'w') as save_data_file:
            json.dump(save_data, save_data_file)
    return

def graph_available_skills_from_file(skills_filepath: str,
                                     agent_labels: List[str],
                                     graph_name: None|str=None,
                                     legend: bool=True,
                                     colours: None|List[str]=None,
                                     y_lim: None|List[int]=None,
                                     y_tick: None|int=None,
                                     log_plot: bool=False,
                                     smoothing_window: None|int=None):
    with open(skills_filepath, 'r') as skills_file:
        skills_data = json.load(skills_file)
    plot_data = []

    for agent_label in agent_labels:
        num_skills = skills_data[agent_label]
        to_plot = []

        if not log_plot:
            to_plot = num_skills
        else:
            for skill_count in num_skills:
                if skill_count <= 0:
                    to_plot.append(0)
                    continue
                to_plot.append(np.log(skill_count))

        if smoothing_window is not None:
            smoothed_data = []
            current_sum = 0
            i = 1
            for elm in to_plot:
                current_sum += elm
                if i >= smoothing_window:
                    smoothed_data.append(current_sum / smoothing_window)
                    current_sum -= to_plot[i - smoothing_window]
                i += 1
            to_plot = smoothed_data

        plot_data.append(to_plot)

    y_label = "Number of Available Skills"
    if log_plot:
        y_label = "Log Number of Available Skills"
    if smoothing_window is not None:
        y_label = "Average " + y_label

    if not legend:
        agent_labels = None

    graphing.graph_multiple(
        plot_data,
        name=graph_name,
        labels=agent_labels,
        x_label="State",
        y_label=y_label,
        colours=colours,
        ylim=y_lim,
        y_tick=y_tick,
        no_xticks=True
    )
    return

def graph_average_available_skills_from_file(skills_filepaths: List[str],
                                             agent_labels: List[str],
                                             presentation_labels: List[str],
                                             env_names: List[str],
                                             graph_name: None|str=None,
                                             error_bars: bool=False,
                                             width: float=0.5,
                                             y_lims: None|List[List[int]]=None, y_ticks: None|int=None,
                                             colours: None|List[str]=None,
                                             ):
    num_envs = len(env_names)
    num_agents = len(agent_labels)
    plot_data = [{"Level 1": np.zeros(num_agents)} for _ in range(num_envs)]
    errors = [np.zeros(num_agents) for _ in range(num_envs)]

    for i in range(num_envs):
        with open(skills_filepaths[i], 'r') as skills_file:
            skills_data = json.load(skills_file)
        num_states = None

        for j in range(num_agents):
            agent_label = agent_labels[j]

            try:
                skills_per_state = skills_data[agent_label]
            except KeyError:
                continue

            if num_states is None:
                num_states = len(skills_per_state)

            mean = sum(skills_per_state) / num_states
            plot_data[i]["Level 1"][j] = mean

            standard_dev = 0
            for value in skills_per_state:
                standard_dev += np.square(value - mean)
            errors[i][j] = standard_dev/num_states
        print(plot_data[i]["Level 1"])
        print(errors[i])

    x_label = "Average Number of Available Skills"

    if not error_bars:
        errors = None

    graphing.graph_multiple_stacked_barchart(
        plot_data,
        [tuple(presentation_labels) for _ in range(num_envs)],
        env_names,
        width,
        x_label,
        "Agent",
        y_lims, y_ticks,
        errors=errors,
        name=graph_name,
        colours=colours
    )

    return


def graph_multiple_skill_count(agents: List[List[OptionsAgent]],
                               agent_labels: List[List[str]], env_names: List[str],
                               graph_name: None|str=None,
                               legend_axes: None|int=None, legend_location: None|str=None,
                               width: float=0.5, y_lims: None|List[List[int]]=None, y_ticks: None|List[int]=None,
                               colours: None|List[str]=None):
    num_envs = len(env_names)
    max_level = 0
    skill_counts = [{} for _ in range(num_envs)]
    num_agents = [len(agent_list) for agent_list in agents]

    for i in range(num_envs):
        for j in range(num_agents[i]):
            agent_label = agent_labels[i][j]

            skill_counts[i][agent_label] = agents[i][j].count_skills()

            max_level = max(max(skill_counts[i][agent_label].keys()), max_level)

    plot_data = [{"Level " + str(level): np.zeros(num_agents[i])
                 for level in range(1, max_level + 1)} for i in range(num_envs)]

    for i in range(num_envs):
        for j in range(num_agents[i]):
            agent_label = agent_labels[i][j]
            for level in range(1, max_level + 1):
                try:
                    count = skill_counts[i][agent_label][level]
                except KeyError:
                    count = 0
                plot_data[i]["Level " + str(level)][j] = count

    x_label = "Number of Skills"

    graphing.graph_multiple_stacked_barchart(plot_data,
                                             [tuple(agent_label) for agent_label in agent_labels],
                                             env_names,
                                             width,
                                             x_label,
                                             "Agent",
                                             y_lims, y_ticks,
                                             legend_axes, legend_location,
                                             graph_name,
                                             False,
                                             colours
    )
    return

def graph_multiple_skill_count_by_state_size(agents: List[List[PreparednessAgent]],
                                             agent_labels: List[str],
                                             name: str|None=None,
                                             verbose: bool=False,
                                             colours: List[str]|None=None):
    to_plot = []
    x = []

    if verbose:
        num_agents = str(len(agents) * len(agent_labels))
        i = 1

    x_found = False

    for agent_list in agents:
        y = []
        for agent in agent_list:
            if verbose:
                print("Counting Skills agent: " + str(i) + "/" + num_agents)
                i += 1

            y.append(sum(agent.count_skills().values()))

            if not x_found:
                num_states = agent.state_transition_graph.number_of_nodes()
                x.append(num_states)

        x_found = True

    graphing.graph_multiple(
        to_plot,
        x,
        name,
        agent_labels,
        "Number of States",
        "Number of Skills",
        colours=colours,
        x_ticks=x,
        linestyle=':',
        marker='o'
    )

    return

def graph_multiple_subgoal_count(envs: List[Environment], env_names: List[str],
                                 subgoal_keys: List[str], multiple_levels: List[bool],
                                 clusters: None|List[bool]=None,
                                 plot_percentage: bool=False,
                                 labels: None|List[str]=None, graph_name: None|str=None,
                                 legend_axes: None|int=None, legend_location: str="upper right",
                                 width: float=0.5, y_lims: None|List[List[int]]=None, y_ticks: None|List[int]=None,
                                 percentage: bool=False,
                                 colours: None|List[str]=None):
    num_keys = len(subgoal_keys)
    num_envs = len(envs)
    max_level = 0
    num_states = None
    any_clusters = False

    if clusters is None:
        clusters = [False] * num_keys

    subgoal_counts = [{} for _ in range(num_envs)]
    num_states = [0] * num_envs

    for i in range(num_envs):
        for j in range(num_keys):
            if not clusters[j]:
                subgoal_counts[i][subgoal_keys[j]], state_count = count_subgoals(envs[i], subgoal_keys[j],
                                                                             multiple_levels[j],
                                                                             True)
                num_states[i] = state_count
            else:
                subgoal_counts[i][subgoal_keys[j]] = count_clusters(envs[i], subgoal_keys[j],
                                                                 False)
                any_clusters = True

            height = max(subgoal_counts[i][subgoal_keys[j]].keys())
            if height > max_level:
                max_level = height

    plot_data = [{"Level " + str(level): np.zeros(num_keys) for level in range(1, max_level + 1)}
                 for _ in range(num_envs)]

    for i in range(num_envs):
        for j in range(num_keys):
            subgoal_key = subgoal_keys[j]
            for level in range(1, max_level + 1):
                try:
                    count = subgoal_counts[i][subgoal_key][level]
                    if plot_percentage:
                        count = (count / num_states[i]) * 100
                except KeyError:
                    count = 0
                plot_data[i]["Level " + str(level)][j] = count

    x_label = "Number of Subgoals"
    if plot_percentage and (not any_clusters):
        x_label = "Percentage of States\nIdentified as Subgoals"
    if (not plot_percentage) and any_clusters:
        x_label = "Number of Subgoal States/Clusters"
    if plot_percentage and any_clusters:
        x_label = "Percentage of States\nIdentified as Subgoals/Clusters"

    if labels is None:
        labels = subgoal_keys

    graphing.graph_multiple_stacked_barchart(plot_data,
                                             labels,
                                             env_names,
                                             width,
                                             x_label,
                                             "Subgoal Method",
                                             y_lims,
                                             y_ticks,
                                             legend_axes, legend_location,
                                             graph_name,
                                             percentage,
                                             colours)
    return


def graph_skill_count(agents: List[OptionsAgent], agent_labels: List[str],
                      graph_name: None|str=None, legend_location: None|str=None,
                      width: float=0.5, y_lim: None|List[int]=None, colours: None|List[str]=None):
    num_labels = len(agents)
    skill_counts = {}
    max_level = 0

    for i in range(num_labels):
        agent = agents[i]
        agent_label = agent_labels[i]

        skill_counts[agent_label] = agent.count_skills()

        max_level = max(max(skill_counts[agent_label].keys()), max_level)

    graphing_data = {"Level " + str(level): np.zeros(num_labels) for level in range(1, max_level + 1)}

    for i in range(num_labels):
        agent_label = agent_labels[i]
        for level in range(1, max_level + 1):
            try:
                count = skill_counts[agent_label][level]
            except KeyError:
                count = 0
            graphing_data["Level " + str(level)][i] = count

    graphing.graph_stacked_barchart(graphing_data,
                                    tuple(agent_labels),
                                    width=width,
                                    x_label="Agent",
                                    y_label="Number of Skills",
                                    y_lim=y_lim,
                                    legend_location=legend_location,
                                    name=graph_name,
                                    colours=colours
                                    )
    return

def graph_skill_count_by_state_size(agents: List[PreparednessAgent],
                                    name: str|None=None,
                                    verbose: bool=False):
    y = []
    x = []

    if verbose:
        num_agents = str(len(agents))
        i = 1

    for agent in agents:
        if verbose:
            print("Counting Skills agent: " + str(i) + "/" + num_agents)
            i += 1

        y.append(sum(agent.count_skills().values()))

        num_states = agent.state_transition_graph.number_of_nodes()
        x.append(num_states)

    graphing.graph(
        y,
        x,
        x,
        name,
        y_label="Number of Skills",
        x_label="Number of States",
        marker='o',
        linestyle=':'
    )
    return

def graph_skill_count_by_state_size_from_file(filepath: str,
                                              name: str|None=None):
    with open(filepath, 'r') as f:
        skill_counts = json.load(f)

    y = []
    x = []

    n = 1
    while True:
        try:
            counts = skill_counts[str(n)]
        except KeyError:
            break

        y.append(counts['count'])
        x.append(counts['num states'])

        n += 1

    graphing.graph(
        y,
        x,
        x,
        name,
        y_label="Number of Skills",
        x_label="Number of States",
        marker='o',
        linestyle=':'
    )
    return

def graph_subgoal_count(environment: Environment, subgoal_keys: List[str], multiple_levels: List[bool],
                        clusters: None|List[bool]=None,
                        plot_percentage: bool=False,
                        plot_num_states: bool=False,
                        labels: None|List[str]=None, graph_name: None|str=None, legend: bool=True,
                        width: float=0.5, y_lim: None|List[int]=None, colours: None|List[str]=None):
    num_keys = len(subgoal_keys)
    subgoal_counts = {}
    max_level = 0
    num_states = None
    any_clusters = False

    if clusters is None:
        clusters = [False] * num_keys

    for i in range(num_keys):
        if not clusters[i]:
            subgoal_counts[subgoal_keys[i]], num_states = count_subgoals(environment, subgoal_keys[i], multiple_levels[i],
                                                                         True)
        else:
            subgoal_counts[subgoal_keys[i]] = count_clusters(environment, subgoal_keys[i],
                                                                         False)
            any_clusters = True
        height = max(subgoal_counts[subgoal_keys[i]].keys())
        if height > max_level:
            max_level = height

    graphing_data = {"Level " + str(level): np.zeros(num_keys) for level in range(1, max_level + 1)}

    for i in range(num_keys):
        subgoal_key = subgoal_keys[i]
        for level in range(1, max_level + 1):
            try:
                count = subgoal_counts[subgoal_key][level]
                if plot_percentage:
                    count = (count / num_states) * 100
            except KeyError:
                count = 0
            graphing_data["Level " + str(level)][i] = count

    if labels is None:
        labels = subgoal_keys

    threshold_key = "Total Number of States"
    if not plot_num_states:
        num_states = None
        threshold_key = None

    y_label = "Number of Subgoals"
    if plot_percentage and (not any_clusters):
        y_label = "Percentage of States\nIdentified as Subgoals"
    if (not plot_percentage) and any_clusters:
        y_label = "Number of Subgoals/Clusters"
    if plot_percentage and any_clusters:
        y_label = "Percentage of States\nIdentified as Subgoals/Clusters"

    graphing.graph_stacked_barchart(graphing_data,
                                    tuple(labels),
                                    num_states, threshold_key,
                                    width,
                                    y_label,
                                    "Subgoal Method",
                                    y_lim,
                                    legend,
                                    graph_name,
                                    colours
                                    )
    return


def label_subgoals(adj_matrix: sparse.csr_matrix, stg: nx.MultiDiGraph,
                   stg_values: Dict[str, Dict[str, float|str]], value_key: str,
                   value_key_suffix: str="",
                   min_level: None|int=None, max_level: None|int=None
                   ) -> Tuple[nx.MultiDiGraph, Dict[str, Dict[str, float|str]], Dict[int, List[str]]]:
    subgoal_level_key = value_key + " subgoal level"
    subgoal_found = False
    if min_level is None:
        get_value_key = lambda _: value_key
        get_subgoal_key = lambda _: value_key + " - local maxima"
    else:
        get_value_key = lambda level: value_key + " - " + str(level) + " hops" + value_key_suffix
        get_subgoal_key = lambda level: value_key + " - "+ str(level) + " hops" + value_key_suffix + " - local maxima"

    if min_level is None:
        min_level = 1
        max_level = 1
    elif max_level is None:
        max_level = 1
        max_level_found = False
        while not max_level_found:
            try:
                _ = stg_values['0'][get_value_key(max_level)]
                max_level += 1
            except KeyError:
                max_level_found = True

    subgoals = {level: [] for level in range(min_level, max_level + 1)}
    for level in range(min_level, max_level + 1):
        key = get_value_key(level)
        subgoal_key = get_subgoal_key(level)

        for node in stg_values:
            is_subgoal_str = 'True'
            distance_matrix = sparse.csgraph.dijkstra(adj_matrix, True, indices=int(node),
                                                      unweighted=True)

            in_neighbours = np.where((distance_matrix <= level) &
                                     (0 < distance_matrix))[0]

            if in_neighbours.size <= 0:
                is_subgoal_str = 'False'
            else:
                out_neighbours = np.where(distance_matrix <= level)[0]
                value = float(stg_values[node][key])
                for neighbour in np.append(out_neighbours, in_neighbours):
                    neighbour_str = str(neighbour)
                    if neighbour_str == node:
                        continue
                    neighbour_value = float(stg_values[neighbour_str][key])
                    if neighbour_value >= value:
                        is_subgoal_str = 'False'
                        break

            stg_values[node][subgoal_key] = is_subgoal_str

            if is_subgoal_str == 'True':
                subgoal_found = True
                subgoals[level].append(node)
                stg_values[node][subgoal_level_key] = str(level)
            elif level == min_level:
                stg_values[node][subgoal_level_key] = 'None'

        if level > min_level and subgoal_found and subgoals[level - 1] == subgoals[level]:
            break

    subgoals[level - 1] = subgoals[level].copy()
    subgoals[level] = []
    level -= 1
    for node in subgoals[level]:
        stg_values[node][subgoal_level_key] = str(level)
    for levels_to_prune in range(level, min_level, -1):
        for lower_levels in range(min_level, levels_to_prune):
            for node in subgoals[levels_to_prune]:
                if node in subgoals[lower_levels]:
                    subgoals[lower_levels].remove(node)
    subgoals_no_empty = {}
    level_to_set = 0
    for i in range(min_level, level + 1):
        if subgoals[i]:
            level_to_set += 1
            subgoals_no_empty[level_to_set] = subgoals[i]

    nx.set_node_attributes(stg, stg_values)
    return stg, stg_values, subgoals_no_empty


def label_preparedness_subgoals(adj_matrix: sparse.csr_matrix, stg: nx.MultiDiGraph,
                                stg_values: Dict[str, Dict[str, float|str]], beta: float=0.5,
                                min_level: int=1, max_level: None|int=None
                                ) -> Tuple[nx.MultiDiGraph, Dict[str, float|str], Dict[int, List[str]]]:
    return label_subgoals(adj_matrix, stg, stg_values, "preparedness", " - beta = " + str(beta), min_level, max_level)


def make_entropy_intrinsic_reward(graph_entropies):
    def intrinsic_reward_func(state):
        return graph_entropies[np.array2string(np.ndarray.astype(state, dtype=int))]

    return intrinsic_reward_func


def node_frequency_entropy(adjacency_matrix: np.matrix, node, num_hops=1,
                           log_base=10, accuracy=4, compressed_matrix=False,
                           neighbours=None):
    N = adjacency_matrix.shape[0]
    if neighbours is None:
        neighbours = get_neighbours(adjacency_matrix, node, num_hops, True, compressed_matrix)

    # P(S_t+n | s_t)
    def prob(start_node, goal_node, hops_away):
        p = 0
        W_start_node = 0.0
        for j in neighbours:
            W_start_node += adjacency_matrix[int(start_node), int(j)]
        if (W_start_node <= 0)  and (start_node == goal_node):
            return 1.0
        if hops_away == 1:
            if W_start_node <= 0:
                return 0
            p = adjacency_matrix[int(start_node), int(goal_node)] / W_start_node
            return p

        for j in neighbours:
            w_start_node_j = adjacency_matrix[int(start_node), int(j)]
            if w_start_node_j <= 0:
                continue
            p += (w_start_node_j / W_start_node) * prob(j, goal_node, hops_away - 1)

        return p

    neighbour_probabilities = [prob(node, neighbour, num_hops) for neighbour in neighbours]

    return round(compute_entropy(neighbour_probabilities, log_base), accuracy)


def node_preparedness(adjacency_matrix: np.matrix, node, beta, num_hops=1, log_base=10, accuracy=4):
    frequency_entropy = node_frequency_entropy(adjacency_matrix, node, num_hops, log_base, accuracy)
    structural_entropy = node_structural_entropy(adjacency_matrix, node, num_hops, log_base, accuracy)

    preparedness_found = (beta * frequency_entropy) + ((1 - beta) * structural_entropy)
    return round(preparedness_found, accuracy)


def node_structural_entropy(adjacency_matrix: np.matrix, node, num_hops=1,
                            log_base=10, accuracy=4, compressed_matrix=False,
                            neighbours=None):
    # Find nodes in Neighbourhood
    if neighbours is None:
        neighbours = get_neighbours(adjacency_matrix, node, num_hops, True, compressed_matrix)
        if node not in neighbours:
            neighbours.append(node)

    num_nodes = adjacency_matrix.shape[0]

    # Compute All Hops
    # W_n_i_j
    def weights_out(start_node, hops_away):
        W = 0.0
        if hops_away == 1:
            for j in neighbours:
                W += adjacency_matrix[int(start_node), int(j)]
            return W

        for j in neighbours:
            w_start_node_j = adjacency_matrix[int(start_node), int(j)]
            if w_start_node_j <= 0:
                continue
            W += (w_start_node_j * weights_out(j, hops_away - 1))
        return W

    T = 0
    for neighbour in neighbours:
        T += weights_out(neighbour, num_hops)

    # Compute Hops to each neighbourhood
    def weights_to_node(start_node, goal_node, hops_away):
        if hops_away == 1:
            return adjacency_matrix[int(start_node), int(goal_node)]

        P_hat = 0.0
        if hops_away == 2:
            for j in neighbours:
                P_hat += (adjacency_matrix[int(start_node), int(j)] * adjacency_matrix[int(j), int(goal_node)])
            return P_hat

        for j in neighbours:
            w_start_node_j = adjacency_matrix[int(start_node), int(j)]
            if w_start_node_j <= 0:
                continue
            P_hat += (w_start_node_j * weights_to_node(j, goal_node, hops_away - 1))
        return P_hat

    neighbour_probabilities = []
    if T == 0:
        return 0.0
    for goal_neighbour in neighbours:
        P = 0
        for start_neighbour in neighbours:
            P += weights_to_node(start_neighbour, goal_neighbour, num_hops)
        neighbour_probabilities.append(P / T)

    # Compute Entropy
    return round(compute_entropy(neighbour_probabilities, log_base), accuracy)


def outweight_sum(adjacency_matrix: np.matrix, node):
    return adjacency_matrix[node].sum()


def preparedness(adjacency_matrix, beta=None, beta_values=None,
                 min_num_hops=1, max_num_hops=1, log_base=10, accuracy=4,
                 compressed_matrix=False, distance_matrix=None):
    if (beta is None) and (beta_values is None):
        raise ValueError("One of beta or beta values must not be None")

    def get_name_suffix(x):
        return '- ' + str(x) + ' hops'

    name_suffix = get_name_suffix(min_num_hops)
    num_nodes = adjacency_matrix.shape[0]
    num_hops = min_num_hops

    preparedness_values = {}
    for i in range(num_nodes):
        print_progress_bar(i, num_nodes, prefix="     " + str(num_hops) + " hops:", suffix='Complete', length=100)

        if distance_matrix is None:
            neighbours = get_neighbours(adjacency_matrix, i, min_num_hops, True, compressed_matrix)
        else:
            neighbours = np.where(distance_matrix[i, :] <= min_num_hops)

        preparedness_values[i] = {'frequency entropy ' + name_suffix:
                                      node_frequency_entropy(adjacency_matrix, i, min_num_hops, log_base, accuracy,
                                                             compressed_matrix, neighbours),
                                  'structural entropy ' + name_suffix:
                                      node_structural_entropy(adjacency_matrix, i, min_num_hops, log_base, accuracy,
                                                              compressed_matrix, neighbours)}

    num_hops += 1
    while num_hops <= max_num_hops:
        name_suffix = get_name_suffix(num_hops)

        for i in range(num_nodes):
            print_progress_bar(i, num_nodes, prefix="     " + str(num_hops) + " hops:", suffix='Complete', length=100)

            if distance_matrix is None:
                neighbours = get_neighbours(adjacency_matrix, i, num_hops, True, compressed_matrix)
            else:
                neighbours = np.where(distance_matrix[i, :] <= num_hops)

            preparedness_values[i]['frequency entropy ' + name_suffix] = \
                node_frequency_entropy(adjacency_matrix, i, num_hops, log_base, accuracy, compressed_matrix,
                                       neighbours)
            preparedness_values[i]['structural entropy ' + name_suffix] = \
                node_structural_entropy(adjacency_matrix, i, num_hops, log_base, accuracy, compressed_matrix,
                                        neighbours)

        num_hops += 1

    if beta is not None:
        beta_values = [beta]

    for beta in beta_values:
        for num_hops in range(min_num_hops, max_num_hops + 1):
            name_suffix = get_name_suffix(num_hops)
            preparedness_key = get_preparedness_key(num_hops, beta)
            for i in range(num_nodes):
                preparedness_values[i][preparedness_key] = \
                    (beta * preparedness_values[i]['frequency entropy ' + name_suffix]) + \
                    ((1 - beta) * preparedness_values[i]['structural entropy ' + name_suffix])
    return preparedness_values


def preparedness_aggregate_graph(environment: Environment,
                                 adjacency_matrix: sparse.csr_matrix,
                                 state_transition_graph: nx.MultiDiGraph,
                                 stg_values : Dict[str, Dict[str, str | float | int]],
                                 preparedness_subgoals: Dict[str, List[str]] | None=None,
                                 min_hop: int=1,
                                 max_hop: int | None=None,
                                 beta: float=0.5,
                                 max_distance: int=np.inf) -> (nx.MultiDiGraph, nx.MultiDiGraph, Dict[str, Dict]):
    def connect_nodes(node_1, node_2):
        return nx.has_path(state_transition_graph, node_1, node_2)
    if max_distance != np.inf:
        def connect_nodes(node_1, node_2):
            shortest_path_distance = nx.shortest_path_length(state_transition_graph, node_1, node_2)
            return shortest_path_distance <= max_distance

    if preparedness_subgoals is None:
        state_transition_graph, stg_values, preparedness_subgoals = label_preparedness_subgoals(adjacency_matrix,
                                                                                                state_transition_graph,
                                                                                                stg_values,
                                                                                                beta,
                                                                                                min_hop,
                                                                                                max_hop)

    # Getting Start Nodes
    start_states = environment.get_start_states()
    start_nodes = []
    for start_state in start_states:
        start_state_str = np.array2string(start_state)
        for node in stg_values:
            if stg_values[node]['state'] == start_state_str:
                start_nodes.append(node)
                break

    # Building Aggregate Graph
    aggregate_graph = nx.MultiDiGraph()
    subgoal_heights = [int(height) for height in preparedness_subgoals.keys()]
    min_height = subgoal_heights[0]
    max_height = subgoal_heights[-1]
    for subgoal_height in preparedness_subgoals:
        for node in preparedness_subgoals[subgoal_height]:
            aggregate_graph.add_node(node)

    # Path Between Subgoals
    for subgoal_height in range(min_height, max_height + 1):
        for subgoal in preparedness_subgoals[subgoal_height]:
            # Path of Decreasing Preparedness
            increasing_path_found = False
            start_height = subgoal_height + 1
            while (start_height <= max_height) and (not increasing_path_found):
                for start_node in preparedness_subgoals[start_height]:
                    if connect_nodes(start_node, subgoal):
                        aggregate_graph.add_edge(start_node, subgoal, weight=1.0)
                        increasing_path_found = True
                start_height += 1

            # Paths of Increasing Preparedness
            decreasing_path_found = False
            start_height = subgoal_height - 1
            while (min_height <= start_height) and (not decreasing_path_found):
                for start_node in preparedness_subgoals[start_height]:
                    if connect_nodes(start_node, subgoal):
                        aggregate_graph.add_edge(start_node, subgoal, weight=1.0)
                        decreasing_path_found = True
                start_height -= 1

            # Paths from Start States to Subgoals (with no in-paths)
            '''
            if not (increasing_path_found or decreasing_path_found):
                for initial_node in start_nodes:
                    if nx.has_path(state_transition_graph, initial_node, subgoal):
                        aggregate_graph.add_node(initial_node)
                        aggregate_graph.add_edge(initial_node, subgoal, weight=1.0)
            '''

    nx.set_node_attributes(aggregate_graph, stg_values)

    return state_transition_graph, aggregate_graph, stg_values


def preparedness_efficient(adjacency_matrix, beta=None, beta_values=None,
                           min_num_hops=1, max_num_hops=1, log_base=10, accuracy=4,
                           compressed_matrix=False,
                           existing_stg_values=None, computed_hops_range=None,
                           progress_bar=True):
    if (beta is None) and (beta_values is None):
        raise ValueError("One of beta or beta values must not be None")

    def get_name_suffix(x):
        return ' - ' + str(x) + ' hops'

    num_nodes = adjacency_matrix.shape[0]
    if beta is not None:
        beta_values = [beta]

    # Computing Preparedness
    preparedness_values = {}
    min_computed_hops = min_num_hops
    max_computed_hops = min_computed_hops - 1
    if existing_stg_values is not None:
        preparedness_values = existing_stg_values
    if computed_hops_range is not None:
        min_computed_hops = computed_hops_range[0]
        max_computed_hops = computed_hops_range[1]

    for node in range(num_nodes):
        if progress_bar:
            print(str(node) + '/' + str(num_nodes))
            print_progress_bar(node, num_nodes, 'Computing Preparedness: ', 'Complete')
        distances = sparse.csgraph.dijkstra(adjacency_matrix, directed=True, indices=node, unweighted=True,
                                            limit=max_num_hops + 1)
        if existing_stg_values is None:
            preparedness_values[str(node)] = {}
        for num_hops in range(min_num_hops, max_num_hops + 1):
            name_suffix = get_name_suffix(num_hops)

            neighbours = np.where((0 < distances) & (distances <= num_hops))[0]

            frequency_entropy = None
            if (num_hops > min_num_hops) and (neighbours.size == 1):
                try:
                    frequency_entropy = (
                        preparedness_values)[str(neighbours[0])]['frequency entropy ' + get_name_suffix(num_hops - 1)]
                except KeyError:
                    frequency_entropy = None
            if frequency_entropy is None:
                frequency_entropy = \
                    node_frequency_entropy(adjacency_matrix, node, min_num_hops, log_base,
                                           accuracy, compressed_matrix, neighbours)
            preparedness_values[str(node)]['frequency entropy ' + name_suffix] = frequency_entropy
            if min_computed_hops <= num_hops <= max_computed_hops:
                continue
            preparedness_values[str(node)]['structural entropy ' + name_suffix] = \
                node_structural_entropy(adjacency_matrix, node, min_num_hops, log_base,
                                        accuracy, compressed_matrix, neighbours)

            for beta_value in beta_values:
                preparedness_key = get_preparedness_key(num_hops, beta_value)
                preparedness_values[str(node)][preparedness_key] = \
                    (beta * preparedness_values[str(node)]['frequency entropy ' + name_suffix]) + \
                    ((1 - beta) * preparedness_values[str(node)]['structural entropy ' + name_suffix])
    return preparedness_values


def print_eigenoptions_subgoals(state_transition_graph_values):
    for node in state_transition_graph_values:
        if state_transition_graph_values[node]['eigenoption_goal'] == 'True':
            print(state_transition_graph_values[node]['state'])
    return


def print_preparedness_subgoals(environment: Environment, subgoal_level: int, beta: float):
    #all_states_filename = environment.environment_name + '_all_states.npy'
    values_filename = environment.environment_name + '_stg_values.json'

    #all_states = np.load(all_states_filename)
    with open(values_filename, 'r') as f:
        values = json.load(f)

    num_states = len(values)
    subgoal_key = get_preparedness_key(subgoal_level, beta) + ' - local maxima'

    subgoal_indexes = [str(node) for node in range(num_states)
                       if values[str(node)][subgoal_key] == 'True']

    num_subgoal_states = 0
    for i in subgoal_indexes:
        print(values[i]['state'])
        num_subgoal_states += 1

    print(str(num_subgoal_states) + '/' + str(num_states) + " subgoals (" +\
          str(round((num_subgoal_states/num_states) * 100, 3)) + "%)")
    return


def print_subgoals(state_values, subgoal_key, value_key=None, state_labels=None):
    subgoal_count = 0
    total_states = 0

    def print_subgoal(subgoal):
        print(subgoal)
        return

    if state_labels is not None:
        num_state_labels = len(state_labels)

        def print_subgoal(subgoal):
            subgoal_holder = subgoal[1: len(subgoal) - 1].split(' ')
            subgoal = []
            for elm in subgoal_holder:
                try:
                    subgoal.append(int(elm))
                except ValueError:
                    continue
            for i in range(num_state_labels):
                print(state_labels[i] + ": " + str(subgoal[i]))
            return

    to_print = {}
    for state in state_values:
        total_states += 1

        is_subgoal = state_values[state][subgoal_key]
        if is_subgoal in ['True', 'False']:
            is_subgoal = is_subgoal == 'True'
        if is_subgoal:
            subgoal_count += 1
            if value_key is None:
                print("Subgoal " + str(subgoal_count) + ":")
                print_subgoal(state_values[state]['state'])
                continue

            to_print[state_values[state]['state']] = state_values[state][value_key]

    if value_key is None:
        print(str(subgoal_count) + "/" + str(total_states) + " subgoals")
        print(str((subgoal_count / total_states) * 100) + "% of states are subgoals")
        return

    subgoals_sorted = list(to_print.keys())
    subgoals_sorted.sort(key=lambda x: to_print[x], reverse=True)

    for i in range(0, subgoal_count):
        print("Subgoal " + str(i + 1) + ":")
        print_subgoal(subgoals_sorted[i])
        print(value_key + ": " + str(to_print[subgoals_sorted[i]]))
    print(str(subgoal_count) + "/" + str(total_states) + " subgoals")
    print(str((subgoal_count / total_states) * 100) + "% of states are subgoals")
    return


def rank_array_dictionary(to_rank, key1, key2):
    def get_key_func(k):
        def get_value_k(d):
            return d[k]

        return get_value_k

    to_rank.sort(key=get_key_func(key2), reverse=True)

    rankings = []
    to_add = [to_rank[0][key1]]
    current_value = to_rank[0][key2]
    ranking_values = [current_value]
    for i in range(1, len(to_rank)):
        value = to_rank[i][key2]
        if value < current_value:
            rankings.append(to_add)
            ranking_values.append(value)
            current_value = value
            to_add = [to_rank[i][key1]]
            continue
        to_add.append(to_rank[i][key1])
    rankings.append(to_add)
    return rankings, ranking_values


def rank_environment_evc(env: Environment, accuracy=None):
    adj_matrix = env.get_adjacency_matrix()

    eigenvalues, eigenvectors = np.linalg.eig(adj_matrix)

    max_value = eigenvalues[0]
    index = 0
    for i in range(1, eigenvalues.shape[0]):
        value = eigenvalues[i]
        if value > max_value:
            index = i
            max_value = value
    vector_chosen = np.array([eigenvectors[i][index] for i in range(eigenvalues.shape[0])])
    vector_chosen_real = np.array([np.absolute(item) for item in vector_chosen])

    if accuracy is not None:
        nodes_rankings = [{'node': env.index_to_state(i), 'value': round(vector_chosen_real[i], accuracy)}
                          for i in range(eigenvalues.shape[0])]
    else:
        nodes_rankings = [{'node': env.index_to_state(i), 'value': vector_chosen_real[i]}
                          for i in range(eigenvalues.shape[0])]

    nodes_ranked, values = rank_array_dictionary(nodes_rankings, 'node', 'value')

    return nodes_ranked, values


def train_agent(env: Environment, agent, num_steps,
                evaluate_policy_window=np.inf,
                all_actions_valid=False, agent_save_path=None,
                total_eval_steps=np.inf,
                copy_agent=True,
                progress_bar=False):
    current_possible_actions = env.possible_actions
    epoch_returns = []
    evaluate_agent = copy.copy(agent)
    evaluate_agent.alpha = 0.0
    evaluate_agent.epsilon = 0.0
    evaluate_env = copy.deepcopy(env)
    total_steps = 0
    training_returns = []
    window_steps = 0
    if evaluate_policy_window == np.inf:
        window_steps = np.inf

    while total_steps < num_steps:
        done = False
        state = env.reset()
        if not all_actions_valid:
            current_possible_actions = env.get_possible_actions(state)

        while not done:
            if progress_bar:
                print_progress_bar(total_steps, num_steps, decimals=3,
                                   prefix='Agent Training: ', suffix='Complete')
            if window_steps <= 0:
                if copy_agent:
                    evaluate_agent.copy_agent(agent)
                else:
                    agent.save(agent_save_path)
                    evaluate_agent.load(agent_save_path)
                epoch_return = run_epoch(evaluate_env, evaluate_agent, total_eval_steps,
                                         total_steps,
                                         all_actions_valid, progress_bar)
                epoch_returns.append(epoch_return)
                window_steps = evaluate_policy_window

            action = agent.choose_action(state, False, possible_actions=current_possible_actions)
            if action is None:
                done = True
                continue
            next_state, reward, done, _ = env.step(action)

            if all_actions_valid:
                agent.learn(state, action, reward, next_state, done)
            else:
                current_possible_actions = env.get_possible_actions(next_state)
                agent.learn(state, action, reward, next_state, terminal=done,
                            next_state_possible_actions=current_possible_actions)

            if total_steps >= num_steps:
                done = True

            state = next_state
            total_steps += 1
            training_returns.append(reward)
            window_steps -= 1

    if agent_save_path is not None:
        agent.save(agent_save_path)

    return agent, training_returns, epoch_returns


def run_episode(env: Environment,
                agent: LearningAgent = None,
                all_actions_valid=True,
                max_steps=np.inf):
    current_possible_actions = env.possible_actions
    done = False
    episode_return = 0
    state = env.reset()
    total_steps = 0

    if not all_actions_valid:
        current_possible_actions = env.get_possible_actions()

    while (not done) and (total_steps < max_steps):
        if agent is None:
            env.print_state()
            if not all_actions_valid:
                print("Possible actions: " + str(current_possible_actions))
            action = int(input("Input action: "))
        else:
            action = agent.choose_action(state, True,
                                         possible_actions=current_possible_actions)
        if action is None:
            done = True
            break
        next_state, reward, done, _ = env.step(action)

        if not all_actions_valid:
            current_possible_actions = env.get_possible_actions()

        if agent is not None:
            agent.learn(state, action, reward, next_state, done,
                        current_possible_actions)
        total_steps += 1

        episode_return += reward
        state = next_state

    if agent is None:
        env.print_state(state)
        print("Total reward: " + str(episode_return))

    return episode_return


def run_epoch(env: Environment,
              agent: LearningAgent,
              num_steps: int,
              seed: None | int=None,
              all_actions_valid: bool=True,
              progress_bar: bool=True):
    current_possible_actions = env.possible_actions
    epoch_return = 0
    total_steps = 0

    if seed is not None:
        state = env.reset(seed=seed)
        seed += 1
    else:
        state = env.reset()
    done = False
    if not all_actions_valid:
        current_possible_actions = env.get_possible_actions(state)

    while total_steps < num_steps:
        if progress_bar:
            print_progress_bar(total_steps, num_steps, '    Running Epoch:')
        if done:
            if num_steps >= np.inf:
                break
            if seed is not None:
                state = env.reset(seed=seed)
                seed += 1
            else:
                state = env.reset()
            if not all_actions_valid:
                current_possible_actions = env.get_possible_actions(state)

        action = agent.choose_action(state, True, current_possible_actions)

        next_state, reward, done, _ = env.step(action)

        if not all_actions_valid:
            current_possible_actions = env.get_possible_actions(next_state)

        agent.learn(state, action, reward, next_state, done,
                    current_possible_actions)
        total_steps += 1

        epoch_return += reward
        state = next_state

    return epoch_return

def train_betweenness_agents(base_agent_save_path: str,
                             environment: Environment,
                             training_timesteps, num_agents, evaluate_policy_window=10,
                             all_actions_valid=True,
                             total_eval_steps=np.inf,
                             continue_training=False,
                             overwrite_existing_agents=True,
                             alpha=0.9, epsilon=0.1, gamma=0.9, subgoal_distance: int=30,
                             progress_bar=False):
    all_agent_training_returns = {str(i): [] for i in range(num_agents)}
    all_agent_returns = {str(i): [] for i in range(num_agents)}
    filenames = get_filenames(environment)
    agent_training_results_file = filenames['results'] + '/betweenness_training_returns.json'
    agent_results_file = filenames['results'] + '/betweenness_epoch_returns.json'
    stg = nx.read_gexf(filenames['state transition graph'])

    directories_to_make = [filenames['agents'], filenames['results']]
    for directory in directories_to_make:
        if not os.path.isdir(directory):
            os.mkdir(directory)

    existing_results = False
    if continue_training or not overwrite_existing_agents:
        if os.path.exists(agent_results_file):
            existing_results = True
            with open(agent_results_file, 'r') as f:
                all_agent_returns = json.load(f)
        if os.path.exists(agent_training_results_file):
            existing_results = True
            with open(agent_training_results_file, 'r') as f:
                all_agent_training_returns = json.load(f)

    agent_start_index = 0
    existing_agents_index = 0
    if (not overwrite_existing_agents) and existing_results:
        existing_agents_index = len(all_agent_training_returns)
        num_agents += existing_agents_index
        for i in range(existing_agents_index, num_agents):
            all_agent_returns[str(i)] = []
            all_agent_training_returns[str(i)] = []
        if not continue_training:
            agent_start_index = existing_agents_index

    base_agent = BetweennessAgent(environment.possible_actions,
                                  alpha, epsilon, gamma,
                                  environment.state_shape, environment.state_dtype,
                                  stg, subgoal_distance)
    base_agent.load(filenames['agents'] + '/' + base_agent_save_path)

    # Training Agent
    for i in range(agent_start_index, num_agents):
        if progress_bar:
            print("Training betweenness agent " + str(i))
        agent_filename = '/betweenness_agent_' + str(i) + '.json'
        agent = BetweennessAgent(environment.possible_actions,
                                 alpha, epsilon, gamma,
                                 environment.state_shape, environment.state_dtype,
                                 stg, subgoal_distance)
        if continue_training and (i < existing_agents_index):
            agent.load(filenames['agents'] + '/' + agent_filename)
        else:
            agent.copy_agent(base_agent)

        # Train Agent
        agent, agent_training_returns, agent_returns = train_agent(environment, agent, training_timesteps,
                                                                   evaluate_policy_window,
                                                                   all_actions_valid,
                                                                   agent_save_path=(filenames['agents'] +
                                                                                    agent_filename),
                                                                   total_eval_steps=total_eval_steps,
                                                                   copy_agent=True,
                                                                   progress_bar=progress_bar)

        all_agent_training_returns[str(i)] += agent_training_returns
        all_agent_returns[str(i)] += agent_returns

        # Saving Results
        with open(agent_training_results_file, 'w') as f:
            json.dump(all_agent_training_returns, f)
        with open(agent_results_file, 'w') as f:
            json.dump(all_agent_returns, f)

    return


def train_eigenoption_agents(base_agent_save_path,
                             environment: Environment,
                             training_timesteps, num_agents, evaluate_policy_window,
                             all_actions_valid=True,
                             total_eval_steps=np.inf,
                             continue_training=False,
                             overwrite_existing_agents=False,
                             num_options=64, alpha=0.9, epsilon=0.1, gamma=0.9,
                             progress_bar=False):
    all_agent_training_returns = {str(i): [] for i in range(num_agents)}
    all_agent_returns = {str(i): [] for i in range(num_agents)}
    filenames = get_filenames(environment)
    adjacency_matrix_filename = filenames['adjacency matrix']
    agent_directory = filenames['agents']
    state_transition_graph = nx.read_gexf(filenames['state transition graph'])
    results_directory = filenames['results']
    agent_training_results_file = 'eigenoptions_training_returns.json'
    agent_results_file = 'eigenoptions_epoch_returns.json'

    directories_to_make = [agent_directory, results_directory]
    for directory in directories_to_make:
        if not os.path.isdir(directory):
            os.mkdir(directory)
    if not os.path.isdir(agent_directory):
        os.mkdir(agent_directory)

    existing_results = False
    if continue_training or not overwrite_existing_agents:
        if os.path.exists(agent_results_file):
            existing_results = True
            with open(agent_results_file, 'r') as f:
                all_agent_returns = json.load(f)
        if os.path.exists(agent_training_results_file):
            existing_results = True
            with open(agent_training_results_file, 'r') as f:
                all_agent_training_returns = json.load(f)

    agent_start_index = 0
    existing_agents_index = 0
    if (not overwrite_existing_agents) and existing_results:
        existing_agents_index = len(all_agent_training_returns)
        num_agents += existing_agents_index
        for i in range(existing_agents_index, num_agents):
            all_agent_returns[str(i)] = []
            all_agent_training_returns[str(i)] = []
        if not continue_training:
            agent_start_index = existing_agents_index

    adjacency_matrix = sparse.load_npz(adjacency_matrix_filename)

    # Training Agent
    for i in range(agent_start_index, num_agents):
        if progress_bar:
            print("Training eigenoptions agent " + str(i))

        # Load agent
        agent = EigenOptionAgent(adjacency_matrix, state_transition_graph,
                                 alpha, epsilon, gamma,
                                 environment.possible_actions,
                                 environment.state_dtype,
                                 num_options)
        if continue_training and (i < existing_agents_index):
            agent.load(agent_directory + '/eigenoption_agent_' + str(i) + '.json')
        else:
            agent.load(base_agent_save_path)

        # Train Agent
        agent, agent_training_returns, agent_returns = train_agent(environment, agent, training_timesteps,
                                                                   evaluate_policy_window,
                                                                   all_actions_valid,
                                                                   agent_save_path=(agent_directory +
                                                                                    '/eigenoption_agent_' + str(i) +
                                                                                    '.json'),
                                                                   total_eval_steps=total_eval_steps,
                                                                   copy_agent=True,
                                                                   progress_bar=progress_bar)

        all_agent_training_returns[str(i)] += agent_training_returns
        all_agent_returns[str(i)] += agent_returns

        # Save results
        with open(results_directory + '/' + agent_training_results_file, 'w') as f:
            json.dump(all_agent_training_returns, f)
        with open(results_directory + '/' + agent_results_file, 'w') as f:
            json.dump(all_agent_returns, f)
    return


def train_louvain_agents(environment: Environment, file_name_prefix,
                         agent_directory, results_directory,
                         training_timesteps, num_agents, evaluate_policy_window=10,
                         initial_load_path: str | None = None,
                         all_actions_valid=False,
                         total_eval_steps=np.inf,
                         overwrite_existing_agents=False,
                         alpha=0.9, epsilon=0.1, gamma=0.9,
                         state_dtype=int, state_shape=None,
                         progress_bar=False):
    all_agent_training_returns = {str(i): [] for i in range(num_agents)}
    all_agent_returns = {str(i): [] for i in range(num_agents)}
    stg_filename = file_name_prefix + '_stg.gexf'
    agent_training_results_file = 'louvain agent training returns'
    agent_results_file = 'louvain agent returns'
    agent_save_directory = "louvain_agents"

    stg = nx.read_gexf(stg_filename)

    directories_to_make = [agent_directory, results_directory]
    for directory in directories_to_make:
        if not os.path.isdir(directory):
            os.mkdir(directory)

    if not os.path.isdir(agent_directory + '/' + agent_save_directory):
        os.mkdir(agent_directory + '/' + agent_save_directory)

    if initial_load_path is None:
        initial_agent = LouvainAgent(environment.possible_actions, stg,
                                     state_dtype, state_shape,
                                     alpha, epsilon, gamma)
        initial_agent.apply_louvain()
        initial_load_path = environment.environment_name + "_agents/initial_louvain_agent.json"
        initial_agent.save(initial_load_path)

    existing_results = False
    if not overwrite_existing_agents:
        if os.path.exists(results_directory + '/' + agent_results_file):
            existing_results = True
            with open(results_directory + '/' + agent_results_file, 'r') as f:
                all_agent_returns = json.load(f)
        if os.path.exists(results_directory + '/' + agent_training_results_file):
            existing_results = True
            with open(results_directory + '/' + agent_training_results_file, 'r') as f:
                all_agent_training_returns = json.load(f)

    agent_start_index = 0
    if (not overwrite_existing_agents) and existing_results:
        agent_start_index = len(all_agent_training_returns)
        num_agents += agent_start_index
        for i in range(agent_start_index, num_agents):
            all_agent_returns[i] = []
            all_agent_training_returns[i] = []

    agent = LouvainAgent(environment.possible_actions, stg, state_dtype, state_shape,
                         alpha, epsilon, gamma)

    for i in range(agent_start_index, num_agents):
        if progress_bar:
            print("Training Louvain Agent " + str(i))
        agent.load(initial_load_path)
        agent, agent_training_returns, agent_returns = train_agent(environment, agent,
                                                                   training_timesteps, evaluate_policy_window,
                                                                   all_actions_valid,
                                                                   agent_save_path=(
                                                                           agent_directory
                                                                           +'/louvain_agent_' + str(i) + '.json'),
                                                                   total_eval_steps=total_eval_steps,
                                                                   copy_agent=False,
                                                                   progress_bar=progress_bar)

        # Saving result
        all_agent_training_returns[i] = agent_training_returns
        all_agent_returns[i] = agent_returns
        with open(results_directory + '/' + agent_training_results_file, 'w') as f:
            json.dump(all_agent_training_returns, f)
        with open(results_directory + '/' + agent_results_file, 'w') as f:
            json.dump(all_agent_returns, f)

    return

def train_preparedness_agents(base_agent_save_path: str,
                              option_onboarding: str,
                              environment: Environment,
                              training_timesteps: int, num_agents: int, evaluate_policy_window: int=10,
                              all_actions_valid: bool=True,
                              total_eval_steps: int=np.inf,
                              continue_training: bool=False,
                              overwrite_existing_agents=False,
                              alpha: float=0.9, epsilon: float=0.1, gamma: float=0.9,
                              max_option_length: int=np.inf, max_hierarchy_height: None | int=None,
                              progress_bar: bool=False) -> None:
    agent_results_file = 'preparedness_agent_returns_' + option_onboarding + '_onboarding.json'
    agent_training_results_file = 'preparedness_agent_training_returns_' + option_onboarding + '_onboarding.json'
    filenames = get_filenames(environment)

    state_transition_graph = nx.read_gexf(filenames['state transition graph'])
    aggregate_graph = nx.read_gexf(filenames['preparedness aggregate graph'])

    base_agent = PreparednessAgent(environment.possible_actions,
                                   alpha, epsilon, gamma,
                                   environment.state_dtype,
                                   environment.state_shape,
                                   state_transition_graph, aggregate_graph,
                                   option_onboarding,
                                   max_option_length, max_hierarchy_height)
    training_agent = PreparednessAgent(environment.possible_actions,
                                   alpha, epsilon, gamma,
                                   environment.state_dtype,
                                   environment.state_shape,
                                   state_transition_graph, aggregate_graph,
                                   option_onboarding,
                                   max_option_length, max_hierarchy_height)
    base_agent.load(base_agent_save_path)

    directories_to_make = [filenames['agents'], filenames['results']]
    for directory in directories_to_make:
        if not os.path.isdir(directory):
            os.mkdir(directory)

    all_agent_returns = {str(i): [] for i in range(num_agents)}
    all_agent_training_returns = {str(i): [] for i in range(num_agents)}
    existing_results = False
    if continue_training or not overwrite_existing_agents:
        if os.path.exists(agent_results_file):
            existing_results = True
            with open(agent_results_file, 'r') as f:
                all_agent_returns = json.load(f)
        if os.path.exists(agent_training_results_file):
            existing_results = True
            with open(agent_training_results_file, 'r') as f:
                all_agent_training_returns = json.load(f)

    agent_start_index = 0
    existing_agents_index = 0
    if (not overwrite_existing_agents) and existing_results:
        existing_agents_index = len(all_agent_training_returns)
        num_agents += existing_agents_index
        for i in range(existing_agents_index, num_agents):
            all_agent_returns[str(i)] = []
            all_agent_training_returns[str(i)] = []
        if not continue_training:
            agent_start_index = existing_agents_index

    for i in range(agent_start_index, num_agents):
        i_str = str(i)
        if progress_bar:
            print("Training Preparedness agent " + option_onboarding + " onboarding: " +
                  i_str + '/' + str(num_agents - 1))

        agent_save_path = 'preparedness_agent_' + option_onboarding + '_' + i_str

        training_agent.specific_onboarding_possible = True
        training_agent.set_onboarding(option_onboarding)
        if continue_training and (i < existing_agents_index):
            training_agent.load(agent_save_path)
        else:
            training_agent.copy_agent(base_agent)
        training_agent, agent_training_returns, agent_returns = train_agent(environment, training_agent,
                                                                            training_timesteps,
                                                                            evaluate_policy_window,
                                                                            all_actions_valid,
                                                                            agent_save_path=(filenames['agents'] + '/' +
                                                                                             agent_save_path),
                                                                            total_eval_steps=total_eval_steps,
                                                                            copy_agent=True,
                                                                            progress_bar=progress_bar)
        all_agent_training_returns[i_str] += agent_training_returns
        all_agent_returns[i_str] += agent_returns

        # Saving Results
        with open(filenames['results'] + '/' + agent_training_results_file, 'w') as f:
            json.dump(all_agent_training_returns, f)
        with open(filenames['results'] + '/' + agent_results_file, 'w') as f:
            json.dump(all_agent_returns, f)

    return

def train_preparedness_flat_agents(base_agent_save_path: str,
                                   environment: Environment,
                                   alpha: float, epsilon: float, gamma: float,
                                   num_agents: int,
                                   evaluate_policy_window: int,
                                   training_timesteps: int,
                                   total_eval_steps: int,
                                   all_actions_valid: bool=False,
                                   continue_training: bool=False, overwrite_existing_agents: bool=False,
                                   progress_bar: bool=False
                                   ):
    all_agent_training_returns = {str(i): [] for i in range(num_agents)}
    all_agent_returns = {str(i): [] for i in range(num_agents)}
    filenames = get_filenames(environment)
    agent_training_results_file = filenames['results'] + '/preparedness_flat_training_returns.json'
    agent_results_file = filenames['results'] + '/preparedness_flat_epoch_returns.json'
    stg = nx.read_gexf(filenames['state transition graph'])
    with open(filenames['state transition graph values'], 'r') as f:
        stg_values = json.load(f)

    directories_to_make = [filenames['agents'], filenames['results']]
    for directory in directories_to_make:
        if not os.path.isdir(directory):
            os.mkdir(directory)

    existing_results = False
    if continue_training or not overwrite_existing_agents:
        if os.path.exists(agent_results_file):
            existing_results = True
            with open(agent_results_file, 'r') as f:
                all_agent_returns = json.load(f)
        if os.path.exists(agent_training_results_file):
            existing_results = True
            with open(agent_training_results_file, 'r') as f:
                all_agent_training_returns = json.load(f)

    agent_start_index = 0
    existing_agents_index = 0
    if (not overwrite_existing_agents) and existing_results:
        existing_agents_index = len(all_agent_training_returns)
        num_agents += existing_agents_index
        for i in range(existing_agents_index, num_agents):
            all_agent_returns[str(i)] = []
            all_agent_training_returns[str(i)] = []
        if not continue_training:
            agent_start_index = existing_agents_index

    subgoals = find_flat_subgoals(stg_values, "preparedness subgoal level")

    base_agent = SubgoalAgent(
        environment.possible_actions,
        alpha,
        epsilon,
        gamma,
        environment.state_shape,
        environment.state_dtype,
        stg,
        subgoals
    )
    base_agent.load(filenames['agents'] + '/' + base_agent_save_path)

    # Training Agent
    for i in range(agent_start_index, num_agents):
        if progress_bar:
            print("Training preparedness flat agent " + str(i))
        agent_filename = '/preparedness_flat_agent_' + str(i) + '.json'
        agent = SubgoalAgent(
            environment.possible_actions,
            alpha,
            epsilon,
            gamma,
            environment.state_shape,
            environment.state_dtype,
            stg,
            subgoals
        )
        if continue_training and (i < existing_agents_index):
            agent.load(filenames['agents'] + '/' + agent_filename)
        else:
            agent.copy_agent(base_agent)

        # Train Agent
        agent, agent_training_returns, agent_returns = train_agent(environment, agent, training_timesteps,
                                                                   evaluate_policy_window,
                                                                   all_actions_valid,
                                                                   agent_save_path=(filenames['agents'] +
                                                                                    agent_filename),
                                                                   total_eval_steps=total_eval_steps,
                                                                   copy_agent=True,
                                                                   progress_bar=progress_bar)

        all_agent_training_returns[str(i)] += agent_training_returns
        all_agent_returns[str(i)] += agent_returns

        # Saving Results
        with open(agent_training_results_file, 'w') as f:
            json.dump(all_agent_training_returns, f)
        with open(agent_results_file, 'w') as f:
            json.dump(all_agent_returns, f)

    return


def train_q_learning_agent(environment: Environment,
                           training_timesteps, num_agents, evaluate_policy_window=10,
                           all_actions_valid=True,
                           total_eval_steps=np.inf,
                           continue_training=False,
                           overwrite_existing_agents=False,
                           alpha=0.9, epsilon=0.1, gamma=0.9,
                           intrinsic_reward=None, intrinsic_reward_lambda=None,
                           file_save_name='q_learning',
                           progress_bar=False):
    all_agent_returns = {str(i): [] for i in range(num_agents)}
    all_agent_training_returns = {str(i): [] for i in range(num_agents)}
    filenames = get_filenames(environment)
    agent_results_file = filenames['results'] + '/' + file_save_name + '_epoch_returns.json'
    agent_training_results_file = filenames['results'] + '/' + file_save_name + '_training_returns.json'

    directories_to_make = [filenames['agents'], filenames['results']]
    for directory in directories_to_make:
        if not os.path.isdir(directory):
            os.mkdir(directory)

    existing_results = False
    if continue_training or not overwrite_existing_agents:
        if os.path.exists(agent_results_file):
            existing_results = True
            with open(agent_results_file, 'r') as f:
                all_agent_returns = json.load(f)
        if os.path.exists(agent_training_results_file):
            existing_results = True
            with open(agent_training_results_file, 'r') as f:
                all_agent_training_returns = json.load(f)

    agent_start_index = 0
    existing_agents_index = 0
    if (not overwrite_existing_agents) and existing_results:
        existing_agents_index = len(all_agent_training_returns)
        num_agents += existing_agents_index
        for i in range(existing_agents_index, num_agents):
            all_agent_returns[str(i)] = []
            all_agent_training_returns[str(i)] = []
        if not continue_training:
            agent_start_index = existing_agents_index

    for i in range(agent_start_index, num_agents):
        if progress_bar:
            print("Training Q-Learning Agent " + str(i))

        agent = QLearningAgent(environment.possible_actions, alpha, epsilon, gamma,
                               intrinsic_reward, intrinsic_reward_lambda)
        if continue_training and (i < existing_agents_index):
            agent.load_policy(filenames['agents'] + '/q_learning_agent_' + str(i) + '.json')
        agent, training_returns, epoch_returns = train_agent(environment, agent, training_timesteps,
                                                             evaluate_policy_window,
                                                             all_actions_valid,
                                                             filenames['agents'] +
                                                             '/q_learning_agent_' + str(i) + '.json',
                                                             total_eval_steps,
                                                             True,
                                                             progress_bar)

        all_agent_returns[str(i)] += epoch_returns
        all_agent_training_returns[str(i)] += training_returns

        with open(filenames['results'] + '/' + file_save_name + '_epoch_returns.json', 'w') as f:
            json.dump(all_agent_returns, f)
        with open(filenames['results'] + '/' + file_save_name + '_training_returns.json', 'w') as f:
            json.dump(all_agent_training_returns, f)

    return

def update_graph_attributes(environment: Environment,
                            attributes: Dict[str, Dict[str, str|float]]) -> None:
    graph_filenames = get_filenames(environment)

    for path in [graph_filenames['state transition graph'],
                 graph_filenames['preparedness aggregate graph'],
                 graph_filenames['frequency entropy subgoal graph'],
                 graph_filenames['neighbourhood entropy subgoal graph']]:
        try:
            g = nx.read_gexf(path)
        except FileNotFoundError:
            continue
        nx.set_node_attributes(g, attributes)
        nx.write_gexf(g, path)

    with open(graph_filenames['state transition graph values'], 'w') as f:
        json.dump(attributes, f)
    return


# Comparators: DIAYN, DADS, Hierarchical Empowerment, Betweenness, Eigenoptions, Louvain
# Environments: Taxicab (modified), Lavaworld, tiny towns (2x2, 3x3), SimpleWindGridworld (4x7x7, 4x10x10)

# Writing: Related Work, future work

# TODO: get diayn running
# TODO: fix run agent for DADS and DIAYN so not learning on evaluation steps

if __name__ == "__main__":
    # lavaflow = LavaFlow(None, None, (0, 0))
    # taxicab = TaxiCab(
    #       False,
    #       False,
    #       [0.25, 0.01, 0.01, 0.01, 0.72],
    #       continuous=True
    #)
    tinytown = TinyTown(2, 3, pick_every=1)

    option_onboarding = 'specific'
    # Taxicab=25, tinytown2x2=25, tinytown2x3=50, lavaflow=50
    graphing_window = 50
    evaluate_policy_window = 10
    hops = 5
    min_num_hops = 1
    max_num_hops = 4
    num_agents = 5
    # Taxicab=100, Simple_wind_gridworld_4x7x7=25, tinytown_3x3=100, tinytown_2x2=np.inf, tinytown_2x3=35, lavaflow_room=50
    total_evaluation_steps = 35
    # tinytown 2x2: 25_000, tinytown(choice)2x3=50_000, taxicab_arrival-prob 500_000, lavaflow_room=100_000, lavaflow_pipes=2_000
    options_training_timesteps = 50_000
    #tinytown_2x2=20_000, tinytown_2x3(choice)=200_000, tinytown_3x3=1_000_000, simple_wind_gridworld_4x7x7=50_000
    #lavaflow_room=50_000, lavaflow_pipes=50_000 taxicab=50_000
    training_timesteps = 50_000
    # Min Hops: Taxicab=1, lavaflow=1, tinytown(2x2)=2, tinytown(2x3)=1(but all level 1 subgoals are level 2)

    # Graph Ordering + Colouring:
    # None Onboarding - 332288 - 1
    # Generic - 117733 - 2
    # Specific - 88CCEE - 3
    # Eigenoptions/Preparedness Flat - DDCC77 - 4
    # Louvain - CC6677 - 5
    # Betweenness - AA4499 - 6
    # Primitives - 555555 - 7
    # _ - EE3377 - 8

    filenames_tinytown = get_filenames(tinytown)
    data = graphing.extract_data(
        filenames_tinytown['results'],
        [
            'preparedness_agent_returns_none_onboarding.json',
            'preparedness_agent_returns_generic_onboarding.json',
            'preparedness_agent_returns_specific_onboarding.json',
            #'eigenoptions_epoch_returns.json',
            #'louvain agent returns',
            #'betweenness_epoch_returns.json',
            'preparedness_flat_epoch_returns.json',
            'q_learning_epoch_returns.json'
        ]
    )
    graphing.graph_reward_per_epoch(
        data,
        graphing_window,
        evaluate_policy_window,
        name='TinyTown',
        x_label='Timesteps',
        y_label='Average Epoch Return',
        error_bars=True,
        colours=['#332288',
                 '#117733',
                 '#88CCEE',
                 '#DDCC77',
                 #'#CC6677',
                 #'#AA4499',
                 '#555555'
                 ]
    )
    exit()

    print("Training Preparedness Flat Agent Lavaflow")
    train_preparedness_flat_agents(
        '/preparedness_flat_agent.json',
        lavaflow,
        0.9,
        0.15,
        0.9,
        5,
        evaluate_policy_window,
        training_timesteps,
        total_evaluation_steps,
        True,
        False,
        False,
        True
    )
    exit()

    filenames = get_filenames(tinytown)
    stg = nx.read_gexf(filenames['state transition graph'])
    with open(filenames['state transition graph values'], 'r') as f:
        stg_values = json.load(f)

    preparedness_flat_agent = SubgoalAgent(
        tinytown.possible_actions,
        0.9,
        0.15,
        0.9,
        tinytown.state_shape,
        tinytown.state_dtype,
        stg,
        find_flat_subgoals(stg_values, 'preparedness subgoal level'),
        30
    )

    preparedness_flat_agent.load(filenames['agents'] + '/preparedness_flat_agent.json')
    print("Training TinyTown Preparedness Flat Agent")
    preparedness_flat_agent.train_options(
        tinytown,
        options_training_timesteps,
        False,
        True
    )
    preparedness_flat_agent.save(filenames['agents'] + '/preparedness_flat_agent.json')

    exit()

    start_n = 9
    end_n = 9

    lavaflow = LavaFlow(
        LavaFlow.generate_empty_board(7),
        "8_square",
        (0, 0)
    )

    filenames = get_filenames(lavaflow)
    stg = nx.read_gexf(filenames['state transition graph'])
    with open(filenames['state transition graph values'], 'r') as f:
        stg_values = json.load(f)
    subgoal_graph = nx.read_gexf(filenames['preparedness aggregate graph'])

    with open("skill_counts_scaling.json", 'r') as f:
        skill_counts = json.load(f)

    preparedness_agent = PreparednessAgent(
        lavaflow.possible_actions,
        0.9,
        0.15,
        0.9,
        lavaflow.state_dtype,
        lavaflow.state_shape,
        stg,
        subgoal_graph,
        'none'
    )
    preparedness_agent.load(filenames['agents'] + '/preparedness_agent.json')

    skill_counts['5'] = {'count': sum(preparedness_agent.count_skills().values()),
                         'num states': stg.number_of_nodes()}
    with open("skill_counts_scaling.json", 'w') as f:
        json.dump(skill_counts, f)

    exit()

    lavaflow_envs = [LavaFlow(LavaFlow.generate_scatter_board(n), str(n) + "_scatter", (0, 0))
                     for n in range(start_n, end_n + 1, 1)]
    lavaflow_agents = []
    with open("skill_counts_scaling.json", 'r') as f:
        skill_counts = json.load(f)

    print("Creating Agents")
    n = start_n
    for lavaflow_env in lavaflow_envs:
        print("Creating Agent " + str(n) + "/" + str(end_n))

        filenames = get_filenames(lavaflow_env)
        stg = nx.read_gexf(filenames['state transition graph'])
        with open(filenames['state transition graph values'], 'r') as f:
            stg_values = json.load(f)
        subgoal_graph = nx.read_gexf(filenames['preparedness aggregate graph'])

        preparedness_agent = PreparednessAgent(
            lavaflow_env.possible_actions,
            0.9,
            0.15,
            0.9,
            lavaflow_env.state_dtype,
            lavaflow_env.state_shape,
            stg,
            subgoal_graph,
            'none'
        )
        preparedness_agent.create_options(lavaflow_env)
        skill_counts[str(n)] = {
            'count': sum(preparedness_agent.count_skills().values()),
            'num states': stg.number_of_nodes()
        }
        with open("skill_counts_scaing.json", 'w') as f:
            json.dump(skill_counts, f)

        n += 1

    exit()

    print("Counting skills")
    graph_skill_count_by_state_size(
        lavaflow_agents,
        'Preparedness Skill Count by State Space Size',
        verbose=True
    )
    exit()

    lavaflow_envs = [LavaFlow(LavaFlow.generate_scatter_board(n), str(n) + "_scatter", (0, 0))
                     for n in range(start_n, end_n + 1, 1)]
    lavaflow_agents = []
    n = start_n
    for lavaflow_env in lavaflow_envs:
        print(n)
        filenames = get_filenames(lavaflow_env)
        adj_matrix = sparse.load_npz(filenames['adjacency matrix'])
        stg = nx.read_gexf(filenames['state transition graph'])
        with open(filenames['state transition graph values'], 'r') as f:
            stg_values = json.load(f)

        stg_values = preparedness_efficient(adj_matrix, 0.5,
                                            max_num_hops=4,
                                            compressed_matrix=True,
                                            existing_stg_values=stg_values,
                                            progress_bar=True)
        nx.set_node_attributes(stg, stg_values)
        with open(filenames['state transition graph values'], 'w') as f:
            json.dump(stg_values, f)
        nx.write_gexf(stg, filenames['state transition graph'])

        stg, stg_values, subgoals = label_preparedness_subgoals(adj_matrix, stg, stg_values,
                                                                max_level=4)
        print(subgoals)
        with open(filenames['state transition graph values'], 'w') as f:
            json.dump(stg_values, f)
        nx.write_gexf(stg, filenames['state transition graph'])

        stg, aggregate_graph, stg_values = preparedness_aggregate_graph(
            lavaflow_env,
            adj_matrix,
            stg,
            stg_values,
            subgoals,
            max_hop=3
        )
        with open(filenames['state transition graph values'], 'w') as f:
            json.dump(stg_values, f)
        nx.write_gexf(stg, filenames['state transition graph'])
        nx.write_gexf(aggregate_graph, filenames['preparedness aggregate graph'])

        n += 1

    filenames = get_filenames(lavaflow_env)
    stg = nx.read_gexf(filenames['state transition graph'])
    subgoal_graph = nx.read_gexf(filenames['preparedness aggregate graph'])
    with open(filenames['state transition graph values'], 'r') as f:
        stg_values = json.load(f)

    exit()

    adj_matrix, stg, stg_values = lavaflow.get_adjacency_matrix(
        True,
        True,
        True,
        progress_bar=True
    )
    sparse.save_npz(filenames['adjacency matrix'], adj_matrix)
    nx.set_node_attributes(stg, stg_values)
    nx.write_gexf(stg, filenames['state transition graph'])
    with open(filenames['state transition graph values'], 'w') as f:
        json.dump(stg_values, f)

    exit()

    graph_average_available_skills_from_file(
        [
            "lavaflow_num_available_skills.json",
            "taxicab_num_available_skills.json",
            "tinytown2x3_num_available_skills.json"
        ],
        [
            'Preparedness (No Onboarding)',
            'Preparedness (Generic Onboarding)',
            'Preparedness (Specific Onboarding)',
            'Betweenness',
            'Louvain',
        ],
        [
            'Preparedness',
            'Preparedness\nGeneric\nOnboarding',
            'Preparedness\nSpecific\nOnboarding',
            'Betweenness',
            'Louvain',
        ],
        [
            "Lavaflow",
            "Taxicab",
            "Tinytown"
        ],
        "Average Available Skills",
        error_bars=False,
        y_lims=[
            [0, 1],
            [0, 30],
            [0, 3]
        ],
        y_ticks=[
            0.2,
            5,
            0.5
        ],
        colours=[
            '#332288',
            '#117733',
            '#88CCEE',
            '#CC6677',
            '#AA4499',
        ]
    )
    exit()

    filenames_tinytown = get_filenames(tinytown)

    plotting_colours = ['#332288',
                        '#117733',
                        '#88CCEE',
                        '#DDCC77',
                        '#CC6677',
                        '#AA4499',
                        '#555555',
                        '#EE3377']
    graph_available_skills_from_file(
        'tinytown2x3_num_available_skills.json',
        [
            'Preparedness (No Onboarding)',
            'Preparedness (Generic Onboarding)',
            'Preparedness (Specific Onboarding)',
            'Betweenness',
            'Louvain',
        ],
        'Taxicab Available Skills',
        True,
        [
            '#332288',
            '#117733',
            '#88CCEE',
            '#CC6677',
            '#AA4499',
        ],
        log_plot=True,
        smoothing_window=100
    )
    exit()

    print("Training tinytown Louvain Agents")
    train_louvain_agents(tinytown, tinytown.environment_name,
                         filenames_tinytown['agents'], filenames_tinytown['results'],
                         training_timesteps, 3, evaluate_policy_window,
                         initial_load_path=filenames_tinytown['agents'] + '/louvain_base_agent.json',
                         all_actions_valid=False,
                         overwrite_existing_agents=False,
                         total_eval_steps=total_evaluation_steps,
                         state_dtype=tinytown.state_dtype, state_shape=tinytown.state_shape, progress_bar=True)

    exit()

    print("Betweenness agent " + tinytown.environment_name + " agent training")
    train_betweenness_agents('/betweenness_base_agent.json', tinytown,
                             training_timesteps, 5, evaluate_policy_window,
                             False, total_evaluation_steps, False,
                             True,
                             0.9, 0.1, 0.9, 30, True)
    print("Betweenness agent " + tinytown.environment_name + " agent training")
    exit()

    adj_matrix_tinytown = sparse.load_npz(filenames_tinytown['adjacency matrix'])
    preparednesss_tinytown_subgoal_graph = nx.read_gexf(filenames_tinytown['preparedness aggregate graph'])
    state_transition_graph_tinytown = nx.read_gexf(filenames_tinytown['state transition graph'])

    print("Preparedness None Agent loading")
    preparedness_agent_tinytown_none = PreparednessAgent(tinytown.possible_actions,
                                                         0.9, 0.15, 0.9,
                                                         tinytown.state_dtype, tinytown.state_shape,
                                                         state_transition_graph_tinytown,
                                                         preparednesss_tinytown_subgoal_graph,
                                                         option_onboarding='none')
    preparedness_agent_tinytown_none.load(filenames_tinytown['agents'] + '/preparedness_agent_none_0')
    print("Preparedness Generic Agent loading")
    preparedness_agent_tinytown_generic = PreparednessAgent(tinytown.possible_actions,
                                                            0.9, 0.15, 0.9,
                                                            tinytown.state_dtype, tinytown.state_shape,
                                                            state_transition_graph_tinytown,
                                                            preparednesss_tinytown_subgoal_graph,
                                                            option_onboarding='generic')
    preparedness_agent_tinytown_generic.load(filenames_tinytown['agents'] + '/preparedness_agent_generic_0')
    print("Preparedness Specific Agent loading")
    preparedness_agent_tinytown_specific = PreparednessAgent(tinytown.possible_actions,
                                                             0.9, 0.15, 0.9,
                                                             tinytown.state_dtype, tinytown.state_shape,
                                                             state_transition_graph_tinytown,
                                                             preparednesss_tinytown_subgoal_graph,
                                                             option_onboarding='specific')
    preparedness_agent_tinytown_specific.load(filenames_tinytown['agents'] + '/preparedness_agent_specific_0')

    graph_available_skills(
        tinytown,
        [
            preparedness_agent_tinytown_none,
            preparedness_agent_tinytown_generic,
            preparedness_agent_tinytown_specific
        ],
        [
            'Preparedness (No Onboarding)',
            'Preparedness (Generic Onboarding)',
            'Preparedness (Specific Onboarding)'
        ],
        False,
        "TinyTown Available Skills",
        [
            '#332288',
            '#117733',
            '#88CCEE'
        ],
        "tinytown2x3_num_available_skills.json",
        True
    )
    exit()

    graph_multiple_subgoal_count([lavaflow, taxicab, tinytown, tinytown],
                                 [
                                     'Lavaflow',
                                     'Taxicab',
                                     'TinyTown\n(Truncated)',
                                     'TinyTown'
                                 ],
                                 [
                                     'preparedness subgoal level',
                                     'frequency entropy  subgoal level',
                                     'structural entropy  subgoal level',
                                     'node betweenness subgoal'
                                 ],
                                 [
                                     True,
                                     True,
                                     True,
                                     False
                                 ],
                                 [
                                     False,
                                     False,
                                     False,
                                     False
                                 ],
                                 True,
                                 [[
                                     'Preparedness',
                                     'Frequency\nEntropy',
                                     'Neighbourhood\nEntropy',
                                     'Betweenness',
                                 ], [
                                     'Preparedness',
                                     'Frequency\nEntropy',
                                     'Neighbourhood\nEntropy',
                                     'Betweenness',
                                 ],
                                     [
                                         'Preparedness',
                                         'Frequency\nEntropy',
                                         'Neighbourhood\nEntropy',
                                         'Betweenness'
                                     ], [
                                     'Preparedness',
                                     'Frequency\nEntropy',
                                     'Neighbourhood\nEntropy',
                                     'Betweenness'
                                 ]
                                 ],
                                 "Percentage Subgoals",
                                 y_lims=[[0, 11],
                                         [0, 6],
                                         [0, 3],
                                         [0, 45]],
                                 y_ticks=[3.0, 2.0, 1.0, 15.0],
                                 percentage=True,
                                 legend_axes=None,
                                 colours=[
                                     '#332288',
                                     '#117733',
                                     '#88CCEE',
                                     '#DDCC77',
                                     '#CC6677',
                                     '#AA4499',
                                     '#555555',
                                     '#EE3377'
                                 ]
                                 )
    exit()

    graph_multiple_skill_count([
        [preparedness_agent_lavaflow_none,
         preparedness_agent_lavaflow_generic,
         preparedness_agent_lavaflow_specific,
         betweenness_agent_lavaflow,
         louvain_agent_lavaflow],
        [preparedness_agent_taxicab_none,
         preparedness_agent_taxicab_generic,
         betweenness_agent_taxicab,
         louvain_agent_taxicab],
        [preparedness_agent_tinytown_none,
         preparedness_agent_tinytown_generic,
         preparedness_agent_tinytown_specific,
         betweenness_agent_tinytown,
         louvain_agent_tinytown],
    ],
        [['Preparedness',
          'Preparedness\nGeneric\nOnboarding',
          'Preparedness\nSpecific\nOnboarding',
          'Betweenness',
          'Louvain'
          ],
         ['Preparedness',
          'Preparedness\nGeneric\nOnboarding',
          'Betweenness',
          'Louvain',
          ],
         ['Preparedness',
          'Preparedness\nGeneric\nOnboarding',
          'Preparedness\nSpecific\nOnboarding',
          'Betweenness',
          'Louvain'
          ],
         ],
        [
            'Lavaflow',
            'Taxicab',
            'Tinytown'
        ],
        "Number of Skills Discovered",
        y_lims=[
            [0, 500],
            [0, 700],
            [0, 550]
        ],
        y_ticks=[
            100,
            100,
            100
        ],
        legend_axes=2,
        legend_location='lower right',
        colours=[
            '#332288',
            '#117733',
            '#88CCEE',
            '#DDCC77',
            '#CC6677',
            '#AA4499',
            '#555555',
            '#EE3377'
        ]
    )
    exit()

    print("TinyTown Louvain Training Options")
    louvain_agent = LouvainAgent(tinytown.possible_actions,
                                 state_transition_graph,
                                 tinytown.state_dtype, tinytown.state_shape,
                                 min_hierarchy_level=0)
    print("Applying Louvain")
    louvain_agent.apply_louvain(first_levels_to_skip=2)
    louvain_agent.load(filenames['agents'] + '/louvain_base_agent.json')
    louvain_agent.train_options_value_iteration(0.001, tinytown, 1, False, True)
    # louvain_agent.train_options(options_training_timesteps,
    #                             tinytown, False, True)
    louvain_agent.save(filenames['agents'] + '/louvain_base_agent.json')
    exit()

    graph_subgoal_count(
        tinytown,
        [
            'preparedness subgoal level',
            'frequency entropy  subgoal level',
            'structural entropy  subgoal level',
            'node betweenness subgoal',
            'cluster'
        ],
        [
            True,
            True,
            True,
            False,
            True
        ],
        [
            False,
            False,
            False,
            False,
            True
        ],
        True,
        labels=[
             'Preparedness',
             'Frequency Entropy',
             'Neighbourhood Entropy',
             'Betweenness',
             'Louvain'
         ],
        graph_name="TinyTown Percentage Subgoals/Clusters",
        legend=True,
        y_lim=[0, 10.5],
        colours=[
                 '#332288',
                 '#117733',
                 '#88CCEE',
                 '#DDCC77',
                 '#CC6677',
                 '#AA4499',
                 '#555555',
                 '#EE3377'
             ]
    )
    exit()

    print("Labeling frequency entropy subgoals")
    state_transition_graph, stg_values, frequency_entropy_subgoals = label_subgoals(
        adj_matrix,
        state_transition_graph,
        stg_values,
        'frequency entropy ',
        min_level=1
    )
    print("Labeling neighbourhood entropy subgoals")
    state_transition_graph, stg_values, neighbourhood_entropy_subgoals = label_subgoals(
        adj_matrix,
        state_transition_graph,
        stg_values,
        'structural entropy ',
        min_level=1
    )

    nx.write_gexf(state_transition_graph, filenames['state transition graph'])
    with open(filenames["state transition graph values"], 'w') as f:
        json.dump(stg_values, f)

    print("Creating frequency entropy subgoal graph")
    state_transition_graph, frequency_subgoal_graph, stg_values = create_subgoal_graph(
        state_transition_graph,
        stg_values,
        frequency_entropy_subgoals
    )
    nx.write_gexf(frequency_subgoal_graph, filenames['frequency entropy subgoal graph'])
    print("Creating neighbourhood entropy subgoal graph")
    state_transition_graph, neighbourhood_subgoal_graph, stg_values = create_subgoal_graph(
        state_transition_graph,
        stg_values,
        neighbourhood_entropy_subgoals
    )
    nx.write_gexf(neighbourhood_subgoal_graph, filenames['neighbourhood entropy subgoal graph'])

    visual_graph = nx.read_graphml("lavaflow_room_visual_graph.graphml")
    nx.set_node_attributes(visual_graph, stg_values)
    nx.write_graphml(visual_graph, "lavaflow_room_visual_graph.graphml")
    exit()

    stg_values = preparedness_efficient(adj_matrix, 0.5,
                                        max_num_hops=8,
                                        compressed_matrix=True,
                                        existing_stg_values=stg_values,
                                        computed_hops_range=[1, 6], progress_bar=True)
    with open(filenames["state transition graph values"], 'w') as f:
        json.dump(stg_values, f)

    graph_skill_count(
        [
            preparedness_agent_none,
            preparedness_agent_generic,
            preparedness_agent_specific,
            betweenness_agent,
            louvain_agent
        ],
        [
            "Preparedness\nNo Onboarding",
            "Preparedness\nGeneric Onboarding",
            "Preparedness\nSpecific Onboarding",
            "Betweenness",
            'Louvain'

        ],
        "Taxicab Number of Skills Discovered",
        "upper left",
        colours=[
            '#332288',
            '#117733',
            '#88CCEE',
            '#DDCC77',
            '#CC6677',
            '#AA4499',
            '#555555',
            '#EE3377'
        ]
    )
    exit()

    louvain_agent = LouvainAgent(tinytown.possible_actions, state_transition_graph,
                                 tinytown.state_dtype, tinytown.state_shape,
                                 0.9, 0.15, 0.9,
                                 min_hierarchy_level=0)
    print("Applying Louvain")
    stg_values = louvain_agent.apply_louvain(first_levels_to_skip=2,
                                             state_transition_graph_values=stg_values)
    nx.write_gexf(state_transition_graph, filenames['state transition graph'])
    with open(filenames['state transition graph values'], 'w') as f:
        json.dump(stg_values, f)

    print("Creating Options")
    louvain_agent.create_options()
    print("Saving Agent")
    louvain_agent.save(filenames['agents'] + '/louvain_base_agent.json')
    exit()

    louvain_agent.train_options(options_training_timesteps, lavaflow,
                                False, True)
    louvain_agent.save(filenames['agents'] + '/louvain_base_agent.json')
    exit()

    graph_subgoal_count(lavaflow, ['preparedness subgoal level',
                                   # 'frequency entropy  subgoal level',
                                   # 'structural entropy  subgoal level',
                                   'node betweenness subgoal',
                                   'cluster'
                                   ],
                        [
                            True,
                            # True,
                            # True,
                            False,
                            True
                        ],
                        clusters=[
                            False,
                            # False,
                            # False,
                            False,
                            True
                        ],
                        plot_percentage=True,
                        legend=True,
                        y_lim=[0, 8.5],
                        labels=[
                            'Preparedness',
                            # 'Frequency\nEntropy',
                            # 'Neighbourhood\nEntropy',
                            'Betweenness',
                            'Louvain'
                        ],
                        colours=['#332288',
                                 '#117733',
                                 '#88CCEE',
                                 '#DDCC77',
                                 '#CC6677',
                                 '#AA4499',
                                 '#555555',
                                 '#EE3377'
                                 ],
                        graph_name="Lavaflow Percentage of Clusters/Subgoals States"
                        )
    exit()

    betweenness_agent = BetweennessAgent(tinytown.possible_actions,
                                         0.9, 0.1, 0.9,
                                         tinytown.state_shape, tinytown.state_dtype,
                                         state_transition_graph, 30)
    stg_values = betweenness_agent.find_betweenness_subgoals(stg_values)
    update_graph_attributes(tinytown, stg_values)
    exit()

    print("Labeling frequency entropy subgoals")
    state_transition_graph, stg_values, frequency_entropy_subgoals = label_subgoals(
        adj_matrix,
        state_transition_graph,
        stg_values,
        'frequency entropy ',
        min_level=1
    )
    print("Labeling neighbourhood entropy subgoals")
    state_transition_graph, stg_values, neighbourhood_entropy_subgoals = label_subgoals(
        adj_matrix,
        state_transition_graph,
        stg_values,
        'structural entropy ',
        min_level=1
    )
    nx.write_gexf(state_transition_graph, filenames['state transition graph'])
    with open(filenames["state transition graph values"], 'w') as f:
        json.dump(stg_values, f)

    print("Creating frequency entropy subgoal graph")
    state_transition_graph, frequency_subgoal_graph, stg_values = create_subgoal_graph(
        state_transition_graph,
        stg_values,
        frequency_entropy_subgoals
    )
    nx.write_gexf(frequency_subgoal_graph, filenames['frequency entropy subgoal graph'])
    print("Creating neighbourhood entropy subgoal graph")
    state_transition_graph, neighbourhood_subgoal_graph, stg_values = create_subgoal_graph(
        state_transition_graph,
        stg_values,
        neighbourhood_entropy_subgoals
    )
    nx.write_gexf(neighbourhood_subgoal_graph, filenames['neighbourhood entropy subgoal graph'])

    holder_graph = nx.read_graphml('tinytown_2x2x3_visual_graph.graphml')
    nx.set_node_attributes(holder_graph, stg_values)
    nx.write_graphml(holder_graph, 'tinytown_2x2x3_visual_graph.graphml')
    exit()

    print("Preparedness " + tinytown.environment_name + " computing")
    stg_values = preparedness_efficient(adj_matrix, 0.5,
                                        max_num_hops=9,
                                        compressed_matrix=True,
                                        existing_stg_values=stg_values,
                                        computed_hops_range=[1, 8],
                                        progress_bar=True)
    nx.set_node_attributes(state_transition_graph, stg_values)
    nx.write_gexf(state_transition_graph, filenames['state transition graph'])
    with open(filenames["state transition graph values"], 'w') as f:
        json.dump(stg_values, f)
    exit()

    print("Training " + lavaflow.environment_name + " " + option_onboarding + " Preparedness Agent")
    train_preparedness_agents(filenames['agents'] + '/preparedness_base_agent.json',
                              option_onboarding, lavaflow,
                              training_timesteps, 5, evaluate_policy_window,
                              True, total_evaluation_steps,
                              continue_training=False, overwrite_existing_agents=True,
                              progress_bar=True)
    exit()

    print("Labeling preparedness subgoals")
    state_transition_graph, stg_values, preparedness_subgoals = label_preparedness_subgoals(
        adj_matrix, state_transition_graph, stg_values,0.5)

    print("Creating preparedness subgoal graph")
    state_transition_graph, preparedness_subgoal_graph, stg_values = preparedness_aggregate_graph(
        taxicab, adj_matrix, state_transition_graph, stg_values,
        preparedness_subgoals
    )

    adj_matrix, state_transition_graph, stg_values = taxicab.get_adjacency_matrix(True, True, True, progress_bar=True)

    state_transition_graph, stg_values, preparedness_subgoals = label_preparedness_subgoals(
        adj_matrix, state_transition_graph, stg_values
    )
    with open(filenames["state transition graph values"], 'w') as f:
        json.dump(stg_values, f)

    update_graph_attributes(taxicab, stg_values)
    exit()

    stg_values = preparedness_efficient(adj_matrix, beta=0.5, max_num_hops=4,
                                        compressed_matrix=True,
                                        existing_stg_values=stg_values
                                        )
    nx.set_node_attributes(state_transition_graph, stg_values)
    nx.write_gexf(state_transition_graph, filenames['state transition graph'])
    with open(filenames["state transition graph values"], 'w') as f:
        json.dump(stg_values, f)
    exit()

    preparedness_agent = PreparednessAgent(tinytown.possible_actions,
                                           0.9, 0.15, 0.9,
                                           tinytown.state_dtype, tinytown.state_shape,
                                           state_transition_graph, preparednesss_subgoal_graph,
                                           option_onboarding='none')
    preparedness_agent.load(filenames['agents'] + '/preparedness_base_agent.json')
    preparedness_agent.train_options(tinytown, options_training_timesteps,
                                     train_between_options=True,
                                     train_onboarding_options=True,
                                     train_subgoal_options=True,
                                     progress_bar=True)
    preparedness_agent.save(filenames['agents'] + '/preparedness_base_agent.json')
    print(tinytown.environment_name + " preparedness training options")
    exit()

    print("Training tinytown eigenoptions agent")
    train_eigenoption_agents(filenames['agents'] + '/eigenoptions_base_agent.json', tinytown,
                             training_timesteps, 5, evaluate_policy_window,
                             False, total_evaluation_steps,
                             continue_training=False,
                             overwrite_existing_agents=True,
                             progress_bar=True)
    exit()

    print("Training eigenoptions taxicab options")
    eigenoptions_agent = EigenOptionAgent(adj_matrix, state_transition_graph,
                                          0.9, 0.15, 0.9,
                                          taxicab.possible_actions,
                                          taxicab.state_dtype, taxicab.state_shape,
                                          64)
    eigenoptions_agent.find_options(True)
    eigenoptions_agent.save(filenames['agents'] + '/eigenoptions_base_agent.json')
    eigenoptions_agent.train_options(taxicab, options_training_timesteps,
                                     True, True)
    eigenoptions_agent.save(filenames['agents'] + '/eigenoptions_base_agent.json')
    exit()

    print("Training taxicab primitives")
    train_q_learning_agent(taxicab,
                           training_timesteps, 5,
                           continue_training=False,
                           progress_bar=True,
                           overwrite_existing_agents=True,
                           all_actions_valid=True,
                           total_eval_steps=total_evaluation_steps)
    exit()

    print(tinytown.environment_name + " preparedness training options")
    state_transition_graph, preparedness_subgoal_graph, stg_values = (
        preparedness_aggregate_graph(tinytown, adj_matrix, state_transition_graph, stg_values,
                                     min_hop=3))
    nx.write_gexf(state_transition_graph, filenames['state transition graph'])
    nx.write_gexf(preparedness_subgoal_graph, filenames['preparedness aggregate graph'])
    exit()

    print("Betweenness Agent " + taxicab.environment_name + " training options")
    betweennessagent = BetweennessAgent(taxicab.possible_actions, 0.9, 0.3, 0.9,
                                        taxicab.state_shape, taxicab.state_dtype,
                                        state_transition_graph, 30)
    betweennessagent.load(filenames['agents'] + '/betweenness_base_agent.json')
    betweennessagent.train_options(taxicab, options_training_timesteps,
                                   True, True)
    betweennessagent.save(filenames['agents'] + '/betweenness_base_agent.json')
    exit()

    stg_values = betweennessagent.find_betweenness_subgoals(stg_values)
    with open(filenames['state transition graph values'], 'w') as f:
        json.dump(stg_values, f)
    nx.set_node_attributes(state_transition_graph, stg_values)
    nx.write_gexf(state_transition_graph, filenames['state transition graph'])
    betweennessagent.create_options()
    betweennessagent.save(filenames['agents'] + '/betweenness_base_agent.json')
    exit()

    adj_matrix, state_transition_graph, stg_values = taxicab.get_adjacency_matrix(True, True,
                                                                                  True,
                                                                                  progress_bar=True)
    with open(filenames['state transition graph values'], 'w') as f:
        json.dump(stg_values, f)
    sparse.save_npz(filenames['adjacency matrix'], adj_matrix)
    nx.set_node_attributes(state_transition_graph, stg_values)
    nx.write_gexf(state_transition_graph, filenames['state transition graph'])
    exit()

    state_transition_graph, preparedness_subgoal_graph, stg_values = (
        preparedness_aggregate_graph(taxicab, adj_matrix,
                                     state_transition_graph, stg_values, min_hop=1, max_hop=None))
    nx.write_gexf(state_transition_graph, filenames['state transition graph'])
    nx.write_gexf(preparedness_subgoal_graph, filenames['preparedness aggregate graph'])
    with open(filenames['state transition graph values'], 'w') as f:
        json.dump(stg_values, f)
    exit()

    state_transition_graph, stg_values, preparedness_subgoals = label_preparedness_subgoals(adj_matrix,
                                                                                            state_transition_graph,
                                                                                            stg_values,
                                                                                            0.5,
                                                                                            min_num_hops, max_num_hops)
    with open(filenames['state transition graph values'], 'w') as f:
        json.dump(stg_values, f)
    nx.write_gexf(state_transition_graph, filenames['state transition graph'])
    exit()

    print(taxicab.environment_name + " preparedness hops 1 - 4")
    stg_values = preparedness_efficient(adj_matrix, beta=0.5,
                                        min_num_hops=1, max_num_hops=4, compressed_matrix=True,
                                        existing_stg_values=stg_values)
    with open(filenames['state transition graph values'], 'w') as f:
        json.dump(stg_values, f)
    nx.set_node_attributes(state_transition_graph, stg_values)
    nx.write_gexf(state_transition_graph, filenames['state transition graph'])
    print(taxicab.environment_name + " preparedness hops 1 - 4")
    exit()

    train_preparedness_agents(filenames['agents'] + "/preparedness_base_agent.json",
                              option_onboarding, taxicab,
                              training_timesteps, num_agents, evaluate_policy_window,
                              False, total_evaluation_steps,
                              continue_training=True, progress_bar=True)
    exit()

    preparedness_agent.set_options_by_pathing(levels_to_set=[1, 2], options_to_set=untrained_options)
    preparedness_agent.save(filenames['agents'] + '/preparedness_base_agent.json')
    print(taxicab.environment_name + " preparedness training options")
    exit()

    print("Simple Wind Gridworld")
    train_betweenness_agents(simple_wind_gridworld,
                             training_timesteps, num_agents, evaluate_policy_window,
                             True, 'options_trained.json',
                             total_evaluation_steps, False,
                             progress_bar=True)
    exit()

    print("Training Eigenoptions")
    train_eigenoption_agents(taxicab_env,
                             taxicab_env.environment_name, agent_directory, results_directory,
                             options_training_timesteps, training_timesteps,
                             num_agents, evaluate_policy_window,
                             all_actions_valid=False,
                             total_eval_steps=total_evaluation_steps,
                             progress_bar=True)
    exit()
