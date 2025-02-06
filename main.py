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
from typing import Dict, List

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


def create_preparedness_reward_function(environment_name, hops, beta=None):
    with open(environment_name + '_stg_values.json', 'r') as f:
        stg_values = json.load(f)

    key = 'preparedness - ' + str(hops) + ' hops'
    if beta is not None:
        key += ' - beta = ' + str(beta)

    stg_lookup = {stg_values[state_index]['state']: stg_values[state_index][key]
                  for state_index in stg_values}

    return lambda state: stg_lookup[np.array2string(state)]


def extract_graph_entropy_values(values_dict):
    extracted_graph_entropies = {values_dict[node]['state']: values_dict[node]['graph entropy']
                                 for node in values_dict}
    return extracted_graph_entropies


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
    return {'adjacency matrix': adj_matrix_filename,
            'all states': all_states_filename,
            'state transition graph': stg_filename,
            'state transition graph values': stg_values_filename,
            'agents': agent_directory,
            'results': results_directory,
            'preparedness aggregate graph': preparedness_aggregate_graph}


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


def label_preparedness_subgoals(adj_matrix, stg, stg_values, beta=0.5,
                                min_hops=1, max_hop=None):
    subgoal_level_key = 'preparedness subgoal level'
    def get_preparedness_key(x):
        return 'preparedness - ' + str(x) + ' hops - beta = ' + str(beta)
    def get_local_maxima_key(x):
        return get_preparedness_key(x) + ' - local maxima'

    if max_hop is None:
        max_hop = 1
        max_hop_found = False
        while not max_hop_found:
            try:
                _ = stg_values['0'][get_local_maxima_key(max_hop)]
                max_hop += 1
            except KeyError:
                max_hop_found = True
    distance_matrix = sparse.csgraph.dijkstra(adj_matrix, True,
                                              unweighted=True, limit=max_hop)

    subgoals = {hop: [] for hop in range(1, max_hop + 1)}
    hops = min_hops
    while hops <= max_hop:
        local_maxima_key = get_local_maxima_key(hops)
        preparedness_key = get_preparedness_key(hops)
        for node in stg_values:
            is_subgoal_str = 'True'

            in_neighbours = np.where((distance_matrix[:, int(node)] <= hops) &
                                     (0 < distance_matrix[:, int(node)]))[0]

            if in_neighbours.size <= 0:
                is_subgoal_str = 'False'
            else:
                out_neighbours = np.where(distance_matrix[int(node)] <= hops)[0]
                preparedness_value = float(stg_values[node][preparedness_key])
                for neighbour in np.append(out_neighbours, in_neighbours):
                    neighbour_str = str(neighbour)
                    if neighbour_str == node:
                        continue
                    neighbour_preparedness = float(stg_values[neighbour_str][preparedness_key])
                    if neighbour_preparedness > preparedness_value:
                        is_subgoal_str = 'False'
                        break

            stg_values[node][local_maxima_key] = is_subgoal_str

            if is_subgoal_str == 'True':
                subgoals[hops].append(node)
                stg_values[node][subgoal_level_key] = str(hops)
            elif hops == min_hops:
                stg_values[node][subgoal_level_key] = 'None'

        if hops > min_hops and (subgoals[hops - 1] == subgoals[hops] or subgoals[hops] == []):
            break
        hops += 1

    subgoals[hops - 1] = subgoals[hops].copy()
    subgoals[hops] = []
    hops -= 1
    for node in subgoals[hops]:
        stg_values[node][subgoal_level_key] = str(hops)
    for hops_to_prune in range(hops, min_hops, -1):
        for lower_hops in range(min_hops, hops_to_prune):
            for node in subgoals[hops_to_prune]:
                if node in subgoals[lower_hops]:
                    subgoals[lower_hops].remove(node)
    subgoals_no_empty = {}
    level = 0
    for i in range(min_hops, hops + 1):
        if subgoals[i]:
            level += 1
            subgoals_no_empty[level] = subgoals[i]

    '''
    final_subgoals = {l: [] for l in range(1, level + 1)}
    final_subgoals[1] = subgoals_no_empty[1]
    l = 2
    for l in range(2, level + 1):
        for node in subgoals_no_empty[l]:
            valid_subgoal = False
            for prior_level in range(l - 1, 0, -1):
                for start_node in final_subgoals[prior_level]:
                    if distance_matrix[int(start_node), int(node)] < np.inf:
                        valid_subgoal = True
                        break
                if valid_subgoal:
                    break

            if valid_subgoal:
                final_subgoals[l].append(node)
            else:
                stg_values[node][subgoal_level_key] = 'None'
    '''

    nx.set_node_attributes(stg, stg_values)
    return stg, stg_values, subgoals_no_empty


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

    # Finding Subgoals
    '''
    subgoals = {i: [] for i in range(min(min_computed_hops, min_num_hops), max(max_computed_hops, max_num_hops) + 1)}
    all_subgoals = []
    for node in range(num_nodes):
        distances = sparse.csgraph.dijkstra(adjacency_matrix, directed=False, indices=node, unweighted=True,
                                            limit=max_num_hops + 1)

        for num_hops in range(min(min_num_hops, min_computed_hops), max(max_num_hops, max_computed_hops) + 1):
            preparedness_key = get_preparedness_key(num_hops, beta) + ' - local maxima'
            try:
                if preparedness_values[str(node)][preparedness_key] == 'True':
                    subgoals[num_hops].append(node)
                    all_subgoals.append(node)
                    continue
            except KeyError:
                ()
            is_subgoal_str = 'False'

            neighbours = np.where((0 < distances) & (distances <= num_hops))[0]
            preparedness_key = get_preparedness_key(num_hops, beta)

            if neighbours.shape[0] > 0:
                sorted_values = np.sort([preparedness_values[str(neighbour)][get_preparedness_key(num_hops, beta)]
                                         for neighbour in neighbours])

                is_subgoal_str = 'False'
                if preparedness_values[str(node)][preparedness_key] > sorted_values[-1]:
                    is_subgoal_str = 'True'
                    subgoals[num_hops].append(node)
                    all_subgoals.append(node)

            preparedness_values[str(node)][preparedness_key + ' - local maxima'] = is_subgoal_str

    # Pruning Subgoals
    pruned_subgoals = {i: [] for i in range(min(min_computed_hops, min_num_hops),
                                            max(max_computed_hops, max_num_hops) + 1)}


    # Finding hierarchy level
    level = min_num_hops + 1
    if computed_hops_range is not None:
        level = min_computed_hops + 1
    hierarchy_level_found = False
    while (not hierarchy_level_found) and (level <= max_num_hops):
        index = num_hops + max_computed_hops - min_computed_hops + 1 - min_num_hops
        if computed_hops_range is not None and (min_computed_hops <= level <= max_computed_hops):
            index = level - min_computed_hops

        if num_subgoals[index - 1] != num_subgoals[index]:
            level += 1
            continue

        arrays_equal = True
        for i in range(num_subgoals[index - 1]):
            if subgoals[index - 1] != subgoals[index]:
                arrays_equal = False
                break
        if not arrays_equal:
            level += 1
            continue

        hierarchy_level_found = True

    if not hierarchy_level_found:
        level -= 1
        print("Hierarchy level maxed-out")
    '''
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
              progress_bar: bool=False):
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

    all_agent_training_returns = {}
    all_agent_returns = {}
    if initial_load_path is None:
        initial_agent = LouvainAgent(environment.possible_actions, stg,
                                     state_dtype, state_shape,
                                     alpha, epsilon, gamma)
        initial_agent.apply_louvain()
        initial_load_path = environment.environment_name + "_agents/initial_louvain_agent.json"
        initial_agent.save(initial_load_path)

    existing_results = False
    if not overwrite_existing_agents:
        if os.path.exists(agent_results_file):
            existing_results = True
            with open(agent_results_file, 'r') as f:
                all_agent_returns = json.load(f)
        if os.path.exists(agent_training_results_file):
            existing_results = True
            with open(agent_training_results_file, 'r') as f:
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


# Comparators: DIAYN, DADS, Hierarchical Empowerment, Betweenness, Eigenoptions, Louvain
# Environments: Taxicab (modified), Lavaworld, tiny towns (2x2, 3x3), SimpleWindGridworld (4x7x7, 4x10x10)

# Writing: Related Work, future work

# TODO: get diayn running
# TODO: fix run agent for DADS and DIAYN so not learning on evaluation steps

if __name__ == "__main__":
    '''
    board = np.array([[3, 3, 3, 3, 3, 3, 3, 3],
                      [3, 0, 0, 0, 0, 0, 0, 3],
                      [3, 3, 3, 0, 3, 3, 3, 2],
                      [3, 3, 3, 0, 0, 0, 0, 0],
                      [3, 0, 0, 0, 0, 0, 3, 3],
                      [3, 0, 3, 0, 0, 0, 2, 3],
                      [3, 0, 3, 3, 1, 3, 3, 3],
                      [3, 3, 3, 3, 3, 3, 3, 3]])
    board_name = 'three_corridor'

    board = np.array([[3, 3, 3, 3, 3, 3, 3, 3    # TODO: Add output that flags untrained options],
                      [3, 0, 0, 0, 2, 3, 2, 3],
                      [3, 0, 3, 0, 3, 3, 0, 3],
                      [3, 1, 0, 0, 0, 0, 0, 3],
                      [3, 0, 3, 0, 3, 0, 3, 3],
                      [3, 0, 0, 0, 0, 0, 0, 3],
                      [3, 3, 3, 3, 3, 3, 3, 3]
                      ])
    board_name = "pipes"
    '''

    board = np.array([[3, 3, 3, 3, 3, 3, 3],
                      [3, 0, 0, 0, 0, 0, 3],
                      [3, 0, 3, 0, 3, 0, 3],
                      [3, 0, 0, 0, 2, 0, 3],
                      [3, 0, 3, 0, 3, 0, 3],
                      [3, 0, 0, 0, 0, 0, 3],
                      [3, 3, 3, 3, 3, 3, 3]
                      ])
    board_name = 'blocks'

    lavaflow = LavaFlow(None, None, (0, 0))
    # taxicab = TaxiCab(False, False, [0.25, 0.01, 0.01, 0.01, 0.72],
    #                   continuous=False)
    # tinytown = TinyTown(2, 3, pick_every=1)

    option_onboarding = 'specific'
    graphing_window = 25
    evaluate_policy_window = 10
    hops = 5
    min_num_hops = 1
    max_num_hops = 4
    num_agents = 3
    # Taxicab=100, Simple_wind_gridworld_4x7x7=25, tinytown_3x3=100, tinytown_2x2=np.inf, tinytown_2x3=35, lavaflow_room=50
    total_evaluation_steps = 50
    # tinytown 2x2: 25_000, tinytown(choice)2x3=50_000, taxicab_arrival-prob 500_000, lavaflow_room=100_000, lavaflow_pipes=2_000
    options_training_timesteps = 1_000_000
    #tinytown_2x2=20_000, tinytown_2x3(choice)=200_000, tinytown_2x3(random)=150_000 tinytown_3x3=1_000_000, simple_wind_gridworld_4x7x7=50_000
    #lavaflow_room=50_000, lavaflow_pipes=50_000 taxicab=50_000
    training_timesteps = 50_000
    # Min Hops: Taxicab=1, lavaflow=1, tinytown(2x2)=2, tinytown(2x3)=1(but all level 1 subgoals are level 2)

    # Graph Ordering + Colouring:
    # None Onboarding - 332288
    # Generic - 117733
    # Specific - 88CCEE
    # Eigenoptions - DDCC77
    # Louvain - CC6677
    # Betweenness - AA4499
    # Primitives - 555555

    filenames = get_filenames(lavaflow)
    adj_matrix = sparse.load_npz(filenames['adjacency matrix'])
    preparednesss_subgoal_graph = nx.read_gexf(filenames['preparedness aggregate graph'])
    state_transition_graph = nx.read_gexf(filenames['state transition graph'])
    with open(filenames['state transition graph values'], 'r') as f:
        stg_values = json.load(f)

    train_q_learning_agent(lavaflow,
                           training_timesteps, num_agents,
                           continue_training=False,
                           progress_bar=True,
                           overwrite_existing_agents=False,
                           all_actions_valid=True,
                           total_eval_steps=total_evaluation_steps)
    exit()

    data = graphing.extract_data(filenames['results'],
                                 [
                                     'preparedness_agent_returns_none_onboarding.json',
                                     'preparedness_agent_returns_generic_onboarding.json',
                                     'preparedness_agent_returns_specific_onboarding.json',
                                     'eigenoptions_epoch_returns.json',
                                     'louvain agent returns',
                                     'betweenness_epoch_returns.json',
                                     'q_learning_epoch_returns.json'
                                 ])
    graphing.graph_reward_per_epoch(data, graphing_window, evaluate_policy_window,
                                    name='Lavaflow',
                                    x_label='Decision Stages',
                                    y_label='Average Epoch Return',
                                    error_bars='st_error',
                                    labels=[
                                        'No Onboarding',
                                        'Generic Onboarding',
                                        'Specific Onboarding',
                                        'Eigenoptions',
                                        'Louvain',
                                        'Betweenness',
                                        'Primitives'
                                    ])
    exit()

    louvain_agent = LouvainAgent(tinytown.possible_actions, state_transition_graph,
                                 tinytown.state_dtype, tinytown.state_shape,
                                 0.9, 0.15, 0.9)
    louvain_agent.load(filenames['agents'] + '/louvain_base_agent.json')
    louvain_agent.train_options(options_training_timesteps, tinytown,
                                False, True)
    louvain_agent.save(filenames['agents'] + '/louvain_base_agent.json')
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

    print(tinytown.environment_name + " preparedness training options")
    state_transition_graph, preparedness_subgoal_graph, stg_values = (
        preparedness_aggregate_graph(tinytown, adj_matrix, state_transition_graph, stg_values,
                                     min_hop=3))
    nx.write_gexf(state_transition_graph, filenames['state transition graph'])
    nx.write_gexf(preparedness_subgoal_graph, filenames['preparedness aggregate graph'])
    exit()

    train_betweenness_agents('/betweenness_base_agent.json', tinytown,
                             training_timesteps, num_agents, evaluate_policy_window,
                             False, total_evaluation_steps, False,
                             0.9, 0.1, 0.9, 30, True)
    print("Betweenness agent " + tinytown.environment_name + " agent training")
    exit()

    train_preparedness_agents(filenames['agents'] + '/preparedness_base_agent.json',
                              option_onboarding, tinytown,
                              training_timesteps, num_agents, evaluate_policy_window,
                              False, total_evaluation_steps,
                              continue_training=False, progress_bar=True)
    exit()

    train_eigenoption_agents(filenames['agents'] + '/eigenoptions_base_agent.json', tinytown,
                             training_timesteps, num_agents, evaluate_policy_window,
                             False, total_evaluation_steps,
                             continue_training=False,
                             progress_bar=True)
    exit()

    eigenoptions_agent = EigenOptionAgent(adj_matrix, state_transition_graph,
                                          0.9, 0.15, 0.9,
                                          tinytown.possible_actions,
                                          tinytown.state_dtype, tinytown.state_shape,
                                          64)
    eigenoptions_agent.load(filenames['agents'] + '/eigenoptions_base_agent.json')
    eigenoptions_agent.train_options(tinytown, options_training_timesteps,
                                     False, True)
    eigenoptions_agent.save(filenames['agents'] + '/eigenoptions_base_agent.json')
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

    train_louvain_agents(lavaflow, lavaflow.environment_name,
                         filenames['agents'], filenames['results'],
                         training_timesteps, num_agents, evaluate_policy_window,
                         filenames['agents'] + '/louvain_base_agent.json',
                         initial_load_path=filenames['agents'] + '/louvain_base_agent.json',
                         all_actions_valid=True,
                         total_eval_steps=total_evaluation_steps,
                         state_dtype=lavaflow.state_dtype, state_shape=lavaflow.state_shape, progress_bar=True)

    exit()

    louvain_agent = LouvainAgent(lavaflow.possible_actions,
                                 state_transition_graph,
                                 lavaflow.state_dtype, lavaflow.state_shape)
    print("Applying Louvain")
    louvain_agent.apply_louvain(first_levels_to_skip=1)
    print("Creating Options")
    louvain_agent.create_options()
    louvain_agent.save(filenames['agents'] + '/louvain_base_agent.json')
    louvain_agent.train_options_value_iteration(0.001, lavaflow, 1, True, True)
    louvain_agent.save(filenames['agents'] + '/louvain_base_agent.json')
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
