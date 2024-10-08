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

import environments.environment
from environments.environment import Environment
from environments.fourroom import FourRoom
from environments.game2048 import Game2048
from environments.GridworldTron import GridworldTron
from environments.keycard import KeyCard
from environments.lavaflow import LavaFlow
from environments.MasterMind import MasterMind
from environments.potionmaking import PotionMaking
from environments.railroad import RailRoad
from environments.simplewindgridworld import SimpleWindGridWorld
from environments.taxicab import TaxiCab
from environments.tinytown import TinyTown
from environments.waterbucket import WaterBucket
import graphing
from learning_agents.betweennessagent import BetweennessAgent
from learning_agents.dadsagent import DADSAgent
from learning_agents.diaynagent import DIAYNAgent
from learning_agents.eigenoptionagent import EigenOptionAgent
from learning_agents.learningagent import LearningAgent
from learning_agents.louvainagent import LouvainAgent
from learning_agents.multilevelgoalagent import MultiLevelGoalAgent
from learning_agents.optionsagent import Option, OptionsAgent, create_option_goal_initiation_func, \
    generate_option_to_goal
from learning_agents.qlearningagent import QLearningAgent
from learning_agents.softactorcritic import SoftActorCritic
from learning_agents.vicagent import VICAgent
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


def find_betweenness_subgoals(env: Environment, stg_values=None):
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


def get_filenames(env: Environment):
    adj_matrix_filename = env.environment_name + '_adj_matrix.npz'
    all_states_filename = env.environment_name + '_all_states.npy'
    stg_filename = env.environment_name + '_stg.gexf'
    stg_values_filename = env.environment_name + '_stg_values.json'
    agent_directory = env.environment_name + '_agents'
    results_directory = env.environment_name + '_episode_results'
    return [adj_matrix_filename, all_states_filename,
            stg_filename, stg_values_filename,
            agent_directory, results_directory]


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


def get_stg(env_name, no_self_loops=False):
    adj_matrix_filename = env_name + '_adj_matrix.txt'
    stg_values_filename = env_name + '_stg_values.json'
    stg_filename = env_name + '_stg.gexf'

    adj_matrix = np.loadtxt(adj_matrix_filename)
    if no_self_loops:
        id = np.identity(adj_matrix.shape[0])
        adj_matrix = adj_matrix - id

    with open(stg_values_filename, 'r') as f:
        data = json.load(f)
    reformatted_data = {int(i): data[i] for i in data}
    g = nx.from_numpy_array(adj_matrix, create_using=nx.MultiDiGraph)
    nx.set_node_attributes(g, reformatted_data)
    nx.write_gexf(g, stg_filename)
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
        if hops_away == 1:
            if W_start_node == 0:
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
                neighbours = np.where(distance_matrix[i, :] >= num_hops)

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


def preparedness_efficient(adjacency_matrix, beta=None, beta_values=None,
                           min_num_hops=1, max_num_hops=1, log_base=10, accuracy=4,
                           compressed_matrix=False,
                           existing_stg_values=None, computed_hops_range=None):
    if (beta is None) and (beta_values is None):
        raise ValueError("One of beta or beta values must not be None")

    def get_name_suffix(x):
        return '- ' + str(x) + ' hops'

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
        distances = sparse.csgraph.dijkstra(adjacency_matrix, indices=node, unweighted=True,
                                            limit=max_num_hops + 1)
        print_progress_bar(node, num_nodes, prefix="Preparedness :",
                           suffix='Complete', length=100)
        if existing_stg_values is None:
            preparedness_values[str(node)] = {}
        for num_hops in range(min_num_hops, max_num_hops + 1):
            name_suffix = get_name_suffix(num_hops)

            neighbours = np.where((0 < distances) & (distances <= num_hops))[0]

            preparedness_values[str(node)]['frequency entropy ' + name_suffix] = \
                node_frequency_entropy(adjacency_matrix, node, min_num_hops, log_base,
                                       accuracy, compressed_matrix, neighbours)
            preparedness_values[str(node)]['structural entropy ' + name_suffix] = \
                node_structural_entropy(adjacency_matrix, node, min_num_hops, log_base,
                                        accuracy, compressed_matrix, neighbours)

            for beta_value in beta_values:
                preparedness_key = get_preparedness_key(num_hops, beta_value)
                preparedness_values[str(node)][preparedness_key] = \
                    (beta * preparedness_values[str(node)]['frequency entropy ' + name_suffix]) + \
                    ((1 - beta) * preparedness_values[str(node)]['structural entropy ' + name_suffix])

    # Finding Subgoals
    subgoals = [[] for _ in range(min_computed_hops, max_computed_hops + 1)] +\
            [[] for _ in range(min_num_hops, max_num_hops + 1)]
    num_subgoals = [0 for _ in range(min_computed_hops, max_computed_hops + 1)] +\
                   [0 for _ in range(min_num_hops, max_num_hops + 1)]
    for node in range(num_nodes):
        distances = sparse.csgraph.dijkstra(adjacency_matrix, indices=node, unweighted=True,
                                            limit=max_num_hops + 1)

        for num_hops in range(min_computed_hops, max_computed_hops + 1):
            preparedness_key = get_preparedness_key(num_hops, beta) + ' - local maxima'
            try:
                if preparedness_values[str(node)][preparedness_key] == 'True':
                    index = num_hops - min_computed_hops
                    subgoals[index].append(node)
                    num_subgoals[index] += 1
            except KeyError:
                preparedness_values[str(node)][preparedness_key] = 'False'

        for num_hops in range(min_num_hops, max_num_hops + 1):
            is_subgoal_str = 'False'

            neighbours = np.where((0 < distances) & (distances <= num_hops))[0]
            preparedness_key = get_preparedness_key(num_hops, beta)

            if neighbours.shape[0] > 0:
                sorted_values = np.sort([preparedness_values[str(neighbour)][get_preparedness_key(num_hops, beta)]
                                         for neighbour in neighbours])

                is_subgoal_str = 'False'
                if preparedness_values[str(node)][preparedness_key] > sorted_values[-1]:
                    is_subgoal_str = 'True'
                    index = num_hops + max_computed_hops - min_computed_hops + 1 - min_num_hops
                    subgoals[index].append(node)
                    num_subgoals[index] += 1

            preparedness_values[str(node)][preparedness_key + ' - local maxima'] = is_subgoal_str

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

    return preparedness_values, level


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
                total_eval_steps=100,
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
            current_possible_actions = env.get_possible_actions()

        while not done:
            if progress_bar:
                print_progress_bar(total_steps, num_steps,
                                   prefix='Agent Training: ', suffix='Complete')
            if window_steps <= 0:
                evaluate_agent.copy_agent(agent)
                epoch_return = run_epoch(evaluate_env, evaluate_agent, total_eval_steps,
                                         all_actions_valid)
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
                current_possible_actions = env.get_possible_actions()
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
              all_actions_valid: bool = True):
    current_possible_actions = env.possible_actions
    done = True
    epoch_return = 0
    total_steps = 0

    while total_steps < num_steps:
        if done:
            state = env.reset()
            done = False
            if not all_actions_valid:
                current_possible_actions = env.get_possible_actions()

        action = agent.choose_action(state, True, current_possible_actions)

        next_state, reward, done, _ = env.step(action)

        if not all_actions_valid:
            current_possible_actions = env.get_possible_actions()

        agent.learn(state, action, reward, next_state, done,
                    current_possible_actions)
        total_steps += 1

        epoch_return += reward
        state = next_state

    return epoch_return


def train_betweenness_agents(environment: Environment,
                             training_timesteps, num_agents, evaluate_policy_window=10,
                             all_actions_valid=True,
                             base_agent_save_path: str=None,
                             total_eval_steps=np.inf,
                             continue_training=False,
                             alpha=0.9, epsilon=0.1, gamma=0.9,
                             progress_bar=False):
    all_agent_training_returns = {}
    all_agent_returns = {}
    filenames = get_filenames(environment)
    state_transition_graph = nx.read_gexf(filenames[2]) # nx.from_scipy_sparse_array(adj_matrix, create_using=nx.DiGraph)
    with open(filenames[3], 'r') as f:
        stg_values = json.load(f)
    agent_directory = filenames[4] + '/betweenness_agents'
    results_directory = filenames[5]
    agent_training_results_file = 'betweenness_training_returns.json'
    agent_results_file = 'betweenness_epoch_returns.json'

    directories_to_make = [agent_directory, results_directory]
    for directory in directories_to_make:
        if not os.path.isdir(directory):
            os.mkdir(directory)

    if continue_training:
        with open(results_directory + '/' + agent_results_file, 'r') as f:
            all_agent_returns = json.load(f)
        with open(results_directory + '/' + agent_training_results_file, 'r') as f:
            all_agent_training_returns = json.load(f)

    # Training Agent
    for i in range(num_agents):
        if progress_bar:
            print("Training betweenness agent " + str(i))
        agent_filename = '/betweenness_agent_' + str(i) + '.json'
        agent = BetweennessAgent(environment.possible_actions,
                                 alpha, epsilon, gamma,
                                 state_transition_graph, environment.state_shape,
                                 environment.state_dtype)
        if continue_training:
            agent.load(agent_directory + agent_filename)
        else:
            agent.load(agent_directory + '/' + base_agent_save_path)

        # Train Agent
        agent, agent_training_returns, agent_returns = train_agent(environment, agent, training_timesteps,
                                                                   evaluate_policy_window,
                                                                   all_actions_valid,
                                                                   agent_save_path=(agent_directory +
                                                                                    agent_filename),
                                                                   total_eval_steps=total_eval_steps,
                                                                   progress_bar=progress_bar)

        if continue_training:
            all_agent_training_returns[str(i)] += agent_training_returns
            all_agent_returns[str(i)] += agent_returns
        else:
            all_agent_training_returns[str(i)] = agent_training_returns
            all_agent_returns[str(i)] = agent_returns

        # Saving Results
        with open(results_directory + '/' + agent_training_results_file, 'w') as f:
            json.dump(all_agent_training_returns, f)
        with open(results_directory + '/' + agent_results_file, 'w') as f:
            json.dump(all_agent_returns, f)

    return


def train_betweenness_options(environment: Environment, file_name_prefix, training_timesteps,
                              all_actions_valid=True, compressed_matrix=False,
                              options_save_directory=None,
                              alpha=0.9, epsilon=0.1, gamma=0.9,
                              progress_bar=False, progress_bar_prefix=None):
    stg_values_filename = file_name_prefix + '_stg_values.json'
    with open(stg_values_filename, 'r') as f:
        stg_values = json.load(f)

    adj_matrix = None
    stg = None
    state_indexer = None

    if compressed_matrix:
        adj_matrix_filename = file_name_prefix + '_adj_matrix.txt.npz'
        adj_matrix = sparse.load_npz(adj_matrix_filename)

        state_indexer = {stg_values[index]['state']: index
                         for index in stg_values}
    else:
        stg_filename = file_name_prefix + '_stg.gexf'
        stg = nx.read_gexf(stg_filename)

    if (options_save_directory is not None) and (not os.path.isdir(options_save_directory)):
        os.mkdir(options_save_directory)

    subgoals_with_options = []
    subgoals = [node for node in stg_values
                if stg_values[node]['betweenness local maxima'] == 'True']

    for subgoal in subgoals:
        save_path = options_save_directory + "/subgoal - " + str(subgoal)
        if (subgoal in subgoals_with_options) or os.path.isfile(save_path):
            continue
        if progress_bar_prefix is not None:
            print(progress_bar_prefix)

        option = generate_option_to_goal(environment, subgoal,
                                         training_timesteps,
                                         stg, adj_matrix, state_indexer,
                                         all_actions_valid,
                                         alpha, epsilon, gamma,
                                         progress_bar,
                                         save_path)

    return


def train_dads_agent(environment: Environment,
                     results_directory,
                     num_skills,
                     training_timesteps, num_agents, evaluate_policy_window=10,
                     skill_training_cycles=10, skill_training_steps=1000,
                     skill_length=10, model_layers=[128, 128],
                     all_actions_valid=True,
                     progress_bar=False, progress_bar_prefix=None):
    if progress_bar_prefix is None:
        progress_bar_prefix = ""

    directories_to_make = [agent_directory, results_directory]
    for directory in directories_to_make:
        if not os.path.isdir(directory):
            os.mkdir(directory)

    base_dads_agent = DADSAgent(environment.possible_actions, environment.state_shape, num_skills, skill_length,
                                model_layers)

    # Train Skills
    start_skills_time = time.time()
    if progress_bar:
        print("Training DADS skills " + progress_bar_prefix)
    base_dads_agent.learn_skills(environment, skill_training_cycles, skill_training_steps, all_actions_valid)
    end_skills_time = time.time()

    # Train agent
    all_agent_training_returns = {}
    all_agent_returns = {}
    agent_training_results_file = 'dads training returns'
    agent_results_file = 'dads returns'
    agent_training_times = [0 for _ in range(num_agents)]
    for i in range(num_agents):
        agent_start_time = time.time()
        if progress_bar:
            print("Training DADS agent " + progress_bar_prefix)
        dads_agent = copy.deepcopy(base_dads_agent)
        dads_agent, training_returns, episode_returns = train_agent(environment, dads_agent,
                                                                    training_timesteps, evaluate_policy_window,
                                                                    all_actions_valid,
                                                                    total_eval_steps=total_evaluation_steps,
                                                                    progress_bar=progress_bar)
        all_agent_training_returns[i] = training_returns
        all_agent_returns[i] = episode_returns
        agent_training_times[i] = time.time() - agent_start_time

    # Saving Results
    with open(results_directory + '/' + agent_training_results_file, 'w') as f:
        json.dump(all_agent_training_returns, f)
    with open(results_directory + '/' + agent_results_file, 'w') as f:
        json.dump(all_agent_returns, f)

    print("Skill Training Time: " + str(end_skills_time - start_skills_time))
    print("Agent Training Times: ")
    for i in range(num_agents):
        print("Agent " + str(i) + ": " + str(agent_training_times[i]))
    print("Average Agent Training Time: " + str(sum(agent_training_times) / num_agents))

    return


def train_diayn_agent(environment: Environment,
                      results_directory,
                      num_skills,
                      training_timesteps, num_agents, evaluate_policy_window=10,
                      skill_training_episodes=10, skill_training_max_steps_per_episode=1000,
                      skill_length=10, model_layers=[128, 128],
                      all_actions_valid=True,
                      progress_bar=False, progress_bar_prefix=None):
    if progress_bar_prefix is None:
        progress_bar_prefix = ""

    directories_to_make = [agent_directory, results_directory]
    for directory in directories_to_make:
        if not os.path.isdir(directory):
            os.mkdir(directory)

    base_diayn_agent = DIAYNAgent(environment.possible_actions, environment.state_shape,
                                  num_skills, skill_length, model_layers)

    # Train Skills
    start_skills_time = time.time()
    if progress_bar:
        print("Training DIAYN skills " + progress_bar_prefix)
    base_diayn_agent.learn_skills(environment, skill_training_episodes, skill_training_max_steps_per_episode,
                                  all_actions_valid)
    end_skills_time = time.time()

    # Train Agent
    all_agent_training_returns = {}
    all_agent_returns = {}
    agent_training_results_file = 'diayn training returns'
    agent_results_file = 'diayn returns'
    agent_training_times = [0 for _ in range(num_agents)]
    for i in range(num_agents):
        agent_start_time = time.time()
        if progress_bar:
            print("Training DIAYN agent " + progress_bar_prefix)
        diayn_agent = copy.deepcopy(base_diayn_agent)
        dads_agent, training_returns, episode_returns = train_agent(environment, diayn_agent,
                                                                    training_timesteps, evaluate_policy_window,
                                                                    all_actions_valid,
                                                                    total_eval_steps=total_evaluation_steps,
                                                                    progress_bar=progress_bar)
        all_agent_training_returns[i] = training_returns
        all_agent_returns[i] = episode_returns
        agent_training_times[i] = time.time() - agent_start_time

        # Saving Results
        with open(results_directory + '/' + agent_training_results_file, 'w') as f:
            json.dump(all_agent_training_returns, f)
        with open(results_directory + '/' + agent_results_file, 'w') as f:
            json.dump(all_agent_returns, f)

    print("Skill Training Time: " + str(end_skills_time - start_skills_time))
    print("Agent Training Times: ")
    for i in range(num_agents):
        print("Agent " + str(i) + ": " + str(agent_training_times[i]))
    print("Average Agent Training Time: " + str(sum(agent_training_times) / num_agents))
    return


def train_eigenoption_agents(base_agent_save_path,
                             environment: Environment,
                             training_timesteps, num_agents, evaluate_policy_window,
                             all_actions_valid=True,
                             total_eval_steps=np.inf,
                             continue_training=False,
                             num_options=64, alpha=0.9, epsilon=0.1, gamma=0.9,
                             progress_bar=False):
    all_agent_training_returns = {}
    all_agent_returns = {}
    filenames = get_filenames(environment)
    adjacency_matrix_filename = filenames[0]
    all_states_filename = filenames[1]
    agent_directory = filenames[4]
    results_directory = filenames[5]
    agent_training_results_file = 'eigenoptions_training_returns.json'
    agent_results_file = 'eigenoptions_epoch_returns.json'

    directories_to_make = [agent_directory, results_directory]
    for directory in directories_to_make:
        if not os.path.isdir(directory):
            os.mkdir(directory)
    if not os.path.isdir(agent_directory):
        os.mkdir(agent_directory)

    if continue_training:
        with open(results_directory + '/' + agent_results_file, 'r') as f:
            all_agent_returns = json.load(f)
        with open(results_directory + '/' + agent_training_results_file, 'r') as f:
            all_agent_training_returns = json.load(f)

    adjacency_matrix = sparse.load_npz(adjacency_matrix_filename)
    all_states = np.load(all_states_filename)

    # Training Agent
    for i in range(num_agents):
        if progress_bar:
            print("Training eigenoptions agent " + str(i))

        # Load agent
        agent = EigenOptionAgent(adjacency_matrix, all_states,
                                 alpha, epsilon, gamma,
                                 environment.possible_actions,
                                 environment.state_dtype,
                                 num_options)
        if continue_training:
            agent.load(agent_directory + '/eigenoption_agent_' + str(i) + '.json')
        else:
            agent.load(agent_directory + '/' + base_agent_save_path)

        # Train Agent
        agent, agent_training_returns, agent_returns = train_agent(environment, agent, training_timesteps,
                                                                   evaluate_policy_window,
                                                                   all_actions_valid,
                                                                   agent_save_path=(agent_directory +
                                                                                    '/eigenoption_agent_' + str(i) +
                                                                                    '.json'),
                                                                   total_eval_steps=total_eval_steps,
                                                                   progress_bar=progress_bar)

        if continue_training:
            all_agent_training_returns[str(i)] += agent_training_returns
            all_agent_returns[str(i)] += agent_returns
        else:
            all_agent_training_returns[str(i)] = agent_training_returns
            all_agent_returns[str(i)] = agent_returns

        # Save results
        with open(results_directory + '/' + agent_training_results_file, 'w') as f:
            json.dump(all_agent_training_returns, f)
        with open(results_directory + '/' + agent_results_file, 'w') as f:
            json.dump(all_agent_returns, f)
    return


def train_louvain_agents(environment: Environment, file_name_prefix,
                         agent_directory, results_directory,
                         training_timesteps, num_agents, evaluate_policy_window=10,
                         agent_load_file=None,
                         initial_agent=None,
                         all_actions_valid=False,
                         total_eval_steps=np.inf,
                         alpha=0.9, epsilon=0.1, gamma=0.9,
                         state_dtype=int, state_shape=None,
                         progress_bar=False):
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
    if initial_agent is None:
        initial_agent = LouvainAgent(environment.possible_actions, stg,
                                     state_dtype, state_shape,
                                     alpha, epsilon, gamma)
        initial_agent.apply_louvain()

    for i in range(num_agents):
        if progress_bar:
            print("Training Louvain Agent " + str(i))
        agent = copy.deepcopy(initial_agent)
        if agent_load_file is not None:
            agent.load_policy(agent_load_file)
        agent, agent_training_returns, agent_returns = train_agent(environment, agent,
                                                                   training_timesteps, evaluate_policy_window,
                                                                   all_actions_valid,
                                                                   total_eval_steps=total_eval_steps,
                                                                   progress_bar=progress_bar)

        # Saving result
        all_agent_training_returns[i] = agent_training_returns
        all_agent_returns[i] = agent_returns
        with open(results_directory + '/' + agent_training_results_file, 'w') as f:
            json.dump(all_agent_training_returns, f)
        with open(results_directory + '/' + agent_results_file, 'w') as f:
            json.dump(all_agent_returns, f)

    return


def train_multi_level_preparedness_agents(environment: Environment, file_name_prefix,
                                          agent_directory, results_directory,
                                          training_timesteps, num_agents, evaluate_policy_window=10,
                                          agent_load_file=None,
                                          initial_agent=None,
                                          all_actions_valid=False,
                                          total_eval_steps=np.inf,
                                          alpha=0.9, epsilon=0.1, gamma=0.9,
                                          state_dtype=int,
                                          progress_bar=False):
    stg_filename = file_name_prefix + '_stg.gexf'
    agent_training_results_file = 'preparedness multi level agent training returns'
    agent_results_file = 'preparedness multi level agent returns'
    agent_save_directory = "preparedness_multi_level_agents"

    stg = nx.read_gexf(stg_filename)

    directories_to_make = [agent_directory, results_directory]
    for directory in directories_to_make:
        if not os.path.isdir(directory):
            os.mkdir(directory)

    if not os.path.isdir(agent_directory + '/' + agent_save_directory):
        os.mkdir(agent_directory + '/' + agent_save_directory)

    preparedness_subgoals = get_preparedness_subgoals(environment, beta=0.5)

    all_agent_training_returns = {}
    all_agent_returns = {}
    if initial_agent is None:
        initial_agent = MultiLevelGoalAgent(environment.possible_actions,
                                            alpha, epsilon, gamma,
                                            preparedness_subgoals, stg,
                                            state_dtype=state_dtype)
    for i in range(num_agents):
        if progress_bar:
            print("Training agent " + str(i))

        agent = copy.deepcopy(initial_agent)
        if agent_load_file is not None:
            agent.load_policy(agent_load_file)
        agent, agent_training_returns, agent_returns = train_agent(environment, agent,
                                                                   training_timesteps, evaluate_policy_window,
                                                                   all_actions_valid,
                                                                   total_eval_steps=total_eval_steps,
                                                                   progress_bar=progress_bar)

        # Saving result
        all_agent_training_returns[i] = agent_training_returns
        all_agent_returns[i] = agent_returns
        with open(results_directory + '/' + agent_training_results_file, 'w') as f:
            json.dump(all_agent_training_returns, f)
        with open(results_directory + '/' + agent_results_file, 'w') as f:
            json.dump(all_agent_returns, f)

    return


def train_preparedness_agents(environment: Environment, file_name_prefix,
                              options_directory, agent_directory, results_directory,
                              training_timesteps, num_agents, evaluate_policy_window=10,
                              all_actions_valid=True,
                              total_eval_steps=np.inf,
                              min_num_hops=1, max_num_hops=5,
                              alpha=0.9, epsilon=0.1, gamma=0.9, beta=None,
                              progress_bar=False,
                              compressed_matrix=False):
    stg_values_filename = file_name_prefix + '_stg_values.json'
    with open(stg_values_filename, 'r') as f:
        preparedness_values = json.load(f)

    adj_matrix = None
    stg = None
    state_indexer = None

    if compressed_matrix:
        adj_matrix_filename = file_name_prefix + '_adj_matrix.txt.npz'
        adj_matrix = sparse.load_npz(adj_matrix_filename)

        state_indexer = {preparedness_values[index]['state']: index
                         for index in preparedness_values}
    else:
        stg_filename = file_name_prefix + '_stg.gexf'
        stg = nx.read_gexf(stg_filename)

    directories_to_make = [agent_directory, results_directory]
    for directory in directories_to_make:
        if not os.path.isdir(directory):
            os.mkdir(directory)

    primitive_options = [Option(actions=[possible_action]) for possible_action in environment.possible_actions]

    for num_hops in range(min_num_hops, max_num_hops + 1):
        # Creating Directory
        agent_save_directory = "preparedness - " + str(num_hops) + ' hops agents'
        if not os.path.isdir(agent_directory + '/' + agent_save_directory):
            os.mkdir(agent_directory + '/' + agent_save_directory)

        # Collect Options
        options = []
        key = 'preparedness - ' + str(num_hops) + ' hops '
        if beta is not None:
            key += '- beta = ' + str(beta) + ' subgoal'
        else:
            key += 'subgoal'
        for node in preparedness_values:
            if preparedness_values[node][key]:
                policy = QLearningAgent(environment.possible_actions,
                                        alpha, epsilon, gamma)
                policy.load_policy(options_directory + "/subgoal - " + str(node))
                initiation_func = create_option_goal_initiation_func(node, stg, adj_matrix, state_indexer)
                option = Option(policy=policy, initiation_func=initiation_func,
                                terminating_func=lambda x: not initiation_func(x))
                options.append(option)
        options += primitive_options

        # Training Agents
        all_agent_training_returns = {}
        all_agent_returns = {}
        for i in range(num_agents):
            if progress_bar:
                print("Training " + str(num_hops) + " hops agent " + str(i))

            agent = OptionsAgent(alpha, epsilon, gamma, options)
            agent, agent_training_returns, agent_returns = train_agent(environment, agent, training_timesteps,
                                                                       evaluate_policy_window,
                                                                       all_actions_valid,
                                                                       agent_save_path=(agent_directory + '/' +
                                                                                        agent_save_directory + '/' +
                                                                                        'agent ' + str(i)),
                                                                       total_eval_steps=total_eval_steps,
                                                                       progress_bar=progress_bar)

            all_agent_training_returns[i] = agent_training_returns
            all_agent_returns[i] = agent_returns

        # Saving Results
        agent_training_results_file = 'preparedness - ' + str(num_hops) + ' hops training returns'
        agent_results_file = 'preparedness - ' + str(num_hops) + ' hops returns'
        with open(results_directory + '/' + agent_training_results_file, 'w') as f:
            json.dump(all_agent_training_returns, f)
        with open(results_directory + '/' + agent_results_file, 'w') as f:
            json.dump(all_agent_returns, f)

    return


def train_preparedness_options(environment: Environment, file_name_prefix, training_timesteps,
                               all_actions_valid=True, compressed_matrix=False,
                               min_num_hops=1, max_num_hops=5,
                               options_save_directory=None,
                               alpha=0.9, epsilon=0.1, gamma=0.9, beta=None,
                               progress_bar=False, progress_bar_prefix=None):
    stg_values_filename = file_name_prefix + '_stg_values.json'
    with open(stg_values_filename, 'r') as f:
        preparedness_values = json.load(f)

    adj_matrix = None
    stg = None
    state_indexer = None

    if compressed_matrix:
        adj_matrix_filename = file_name_prefix + '_adj_matrix.txt.npz'
        adj_matrix = sparse.load_npz(adj_matrix_filename)

        state_indexer = {preparedness_values[index]['state']: index
                         for index in preparedness_values}
    else:
        stg_filename = file_name_prefix + '_stg.gexf'
        stg = nx.read_gexf(stg_filename)

    if (options_save_directory is not None) and (not os.path.isdir(options_save_directory)):
        os.mkdir(options_save_directory)

    subgoals_with_options = []
    for num_hops in range(min_num_hops, max_num_hops + 1):
        if progress_bar:
            print("Training Options for " + str(num_hops) + " hops")
        key = 'preparedness - ' + str(num_hops) + ' hops '
        if beta is not None:
            key += '- beta = ' + str(beta) + ' subgoal'
        else:
            key += 'subgoal'
        subgoals = [node for node in preparedness_values
                    if preparedness_values[node][key]]

        for subgoal in subgoals:
            if subgoal in subgoals_with_options:
                continue
            save_path = None
            if options_save_directory is not None:
                save_path = options_save_directory + "/subgoal - " + str(subgoal)
            if progress_bar_prefix is not None:
                print(progress_bar_prefix)
            option = generate_option_to_goal(environment, subgoal,
                                             training_timesteps,
                                             stg, adj_matrix, state_indexer,
                                             all_actions_valid,
                                             alpha, epsilon, gamma,
                                             progress_bar,
                                             save_path=save_path)
            subgoals_with_options.append(subgoal)

    return


def train_q_learning_agent(environment: Environment,
                           training_timesteps, num_agents, evaluate_policy_window=10,
                           all_actions_valid=True,
                           total_eval_steps=np.inf,
                           continue_training=False,
                           alpha=0.9, epsilon=0.1, gamma=0.9,
                           intrinsic_reward=None, intrinsic_reward_lambda=None,
                           file_save_name='q_learning',
                           progress_bar=False):
    all_epoch_returns = {}
    all_training_returns = {}
    filenames = get_filenames(environment)
    agent_directory = filenames[4]
    results_directory = filenames[5]
    directories_to_make = [agent_directory, results_directory]

    for directory in directories_to_make:
        if not os.path.isdir(directory):
            os.mkdir(directory)

    if continue_training:
        with open(results_directory + '/' + file_save_name + '_epoch_returns.json', 'r') as f:
            all_epoch_returns = json.load(f)
        with open(results_directory + '/' + file_save_name + '_training_returns.json', 'r') as f:
            all_training_returns = json.load(f)

    for i in range(num_agents):
        if progress_bar:
            print("Training Q-Learning Agent " + str(i))

        agent = QLearningAgent(environment.possible_actions, alpha, epsilon, gamma,
                               intrinsic_reward, intrinsic_reward_lambda)
        if continue_training:
            agent.load_policy(agent_directory + '/q_learning_agent ' + str(i))
        agent, training_returns, epoch_returns = train_agent(environment, agent, training_timesteps,
                                                               evaluate_policy_window,
                                                               all_actions_valid,
                                                               agent_directory + '/q_learning_agent ' + str(i),
                                                               total_eval_steps,
                                                               progress_bar)

        if continue_training:
            all_epoch_returns[str(i)] += epoch_returns
            all_training_returns[str(i)] += training_returns
        else:
            all_epoch_returns[i] = epoch_returns
            all_training_returns[i] = training_returns

        with open(results_directory + '/' + file_save_name + '_epoch_returns.json', 'w') as f:
            json.dump(all_epoch_returns, f)
        with open(results_directory + '/' + file_save_name + '_training_returns.json', 'w') as f:
            json.dump(all_training_returns, f)

    return


def train_sac_agent(environment: Environment, state_shape,
                    results_directory,
                    training_timesteps, num_agents, evaluate_policy_window=10,
                    all_actions_valid=True,
                    total_eval_steps=np.inf,
                    file_save_name='sac',
                    progress_bar=False):
    all_episode_returns = {}
    all_training_returns = {}
    directories_to_make = [results_directory]

    for directory in directories_to_make:
        if not os.path.isdir(directory):
            os.mkdir(directory)

    for i in range(num_agents):
        if progress_bar:
            print("Training SAC Agent " + str(i))

        agent = SoftActorCritic(environment.possible_actions, state_shape,
                                model_layers=[8, 8],
                                update_steps=10)
        agent, training_returns, episode_returns = train_agent(env, agent,
                                                               training_timesteps, evaluate_policy_window,
                                                               all_actions_valid=False,
                                                               total_eval_steps=total_evaluation_steps,
                                                               progress_bar=True)

        all_episode_returns[i] = episode_returns
        all_training_returns[i] = training_returns

        with open(results_directory + '/' + file_save_name + '_episode_returns.json', 'w') as f:
            json.dump(all_episode_returns, f)
        with open(results_directory + '/' + file_save_name + '_training_returns.json', 'w') as f:
            json.dump(all_training_returns, f)

    return


def train_subgoal_agent(environment: Environment, keys, file_name_prefix,
                        options_directory, agent_directory, results_directory,
                        training_timesteps, num_agents, evaluate_policy_window=10,
                        all_actions_valid=True,
                        total_eval_steps=np.inf,
                        alpha=0.9, epsilon=0.1, gamma=0.9,
                        progress_bar=False,
                        compressed_matrix=False):
    stg_values_filename = file_name_prefix + '_stg_values.json'
    with open(stg_values_filename, 'r') as f:
        stg_values = json.load(f)

    adj_matrix = None
    stg = None
    state_indexer = None

    if compressed_matrix:
        adj_matrix_filename = file_name_prefix + '_adj_matrix.txt.npz'
        adj_matrix = sparse.load_npz(adj_matrix_filename)

        state_indexer = {stg_values[index]['state']: index
                         for index in stg_values}
    else:
        stg_filename = file_name_prefix + '_stg.gexf'
        stg = nx.read_gexf(stg_filename)

    directories_to_make = [agent_directory, results_directory]
    for directory in directories_to_make:
        if not os.path.isdir(directory):
            os.mkdir(directory)

    primitive_options = [Option(actions=[possible_action]) for possible_action in environment.possible_actions]

    for key in keys:
        # Creating Directory
        dir = agent_directory + '/' + key
        if not os.path.isdir(dir):
            os.mkdir(dir)

        # Collect Options
        options = []
        has_options = False
        for node in stg_values:
            if stg_values[node][key] == 'True':
                policy = QLearningAgent(environment.possible_actions,
                                        alpha, epsilon, gamma)
                try:
                    policy.load_policy(options_directory + "/subgoal - " + str(node))
                except FileNotFoundError:
                    stg_values[node][key] = 'False'
                    continue

                initiation_func = create_option_goal_initiation_func(node, stg, adj_matrix, state_indexer)
                option = Option(policy=policy, initiation_func=initiation_func,
                                terminating_func=lambda x: not initiation_func(x))
                has_options = True
                options.append(option)

        if not has_options:
            print("No options for " + key + ' agent')
            continue
        options += primitive_options

        # Training Agents
        all_agent_training_returns = {}
        all_agent_returns = {}
        agent_training_results_file = key + ' training returns'
        agent_results_file = key + ' returns'
        for i in range(num_agents):
            print("Training " + key + " agent " + str(i))

            agent = OptionsAgent(alpha, epsilon, gamma, options)
            agent, agent_training_returns, agent_returns = train_agent(environment, agent, training_timesteps,
                                                                       evaluate_policy_window,
                                                                       all_actions_valid,
                                                                       agent_save_path=(dir + '/agent ' + str(i)),
                                                                       total_eval_steps=total_eval_steps,
                                                                       progress_bar=progress_bar)

            all_agent_training_returns[i] = agent_training_returns
            all_agent_returns[i] = agent_returns

            # Saving Results
            with open(results_directory + '/' + agent_training_results_file, 'w') as f:
                json.dump(all_agent_training_returns, f)
            with open(results_directory + '/' + agent_results_file, 'w') as f:
                json.dump(all_agent_returns, f)

    return


def train_subgoal_options(environment: Environment, file_name_prefix, training_timesteps,
                          keys,
                          all_actions_valid=True, compressed_matrix=False,
                          options_save_directory=None,
                          alpha=0.9, epsilon=0.1, gamma=0.9,
                          progress_bar=False, progress_bar_prefix=None):
    stg_values_filename = file_name_prefix + '_stg_values.json'
    with open(stg_values_filename, 'r') as f:
        stg_values = json.load(f)

    adj_matrix = None
    stg = None
    state_indexer = None

    if compressed_matrix:
        adj_matrix_filename = file_name_prefix + '_adj_matrix.txt.npz'
        adj_matrix = sparse.load_npz(adj_matrix_filename)

        state_indexer = {stg_values[index]['state']: index
                         for index in stg_values}
    else:
        stg_filename = file_name_prefix + '_stg.gexf'
        stg = nx.read_gexf(stg_filename)

    if (options_save_directory is not None) and (not os.path.isdir(options_save_directory)):
        os.mkdir(options_save_directory)

    subgoals_with_options = []
    for key in keys:
        if progress_bar:
            print("Training Options for " + key)

        subgoals = [node for node in stg_values
                    if stg_values[node][key] == 'True']

        for subgoal in subgoals:
            if subgoal in subgoals_with_options:
                continue
            save_path = None
            if options_save_directory is not None:
                save_path = options_save_directory + "/subgoal - " + str(subgoal)
            if progress_bar_prefix is not None:
                print(progress_bar_prefix)
                print(key)
            option = generate_option_to_goal(environment, subgoal,
                                             training_timesteps,
                                             stg, adj_matrix, state_indexer,
                                             all_actions_valid,
                                             alpha, epsilon, gamma,
                                             progress_bar,
                                             save_path=save_path)
            subgoals_with_options.append(subgoal)

    return


# Comparators: DIAYN, DADS, Hierarchical Empowerment, Betweenness, Eigenoptions, Louvain
# Environments: Taxicab (modified), Lavaworld, tiny towns (2x2, 3x3), SimpleWindGridworld (4x7x7, 4x10x10)

# Writing: Related Work, future work

# TODO: get diayn running
# TODO: fix run agent for DADS and DIAYN so not learning on evaluation steps

if __name__ == "__main__":
    board = np.array([[3, 2, 3],
                      [0, 0, 0],
                      [4, 0, 4],
                      [0, 1, 0]])
    board_name = 'room'
    simple_wind_gridworld = SimpleWindGridWorld((7, 7), 4)

    beta = 0.5
    graphing_window = 5
    evaluate_policy_window = 10
    intrinsic_reward_lambda = 0.5
    hops = 5
    min_num_hops = 1
    max_num_hops = 1
    num_agents = 3
    total_evaluation_steps = 25 #Simple_wind_gridworld_4x7x7 = 25, tinytown_3x3 = 100
    options_training_timesteps = 10_000
    training_timesteps = 50_000 #1_000_000

    #filenames = get_filenames(tinytown)
    #adj_matrix = sparse.load_npz(filenames[0])
    #all_states = np.load(filenames[1])
    #state_transition_graph = nx.read_gexf(filenames[2]) # nx.from_scipy_sparse_array(adj_matrix, create_using=nx.DiGraph)
    #with open(filenames[3], 'r') as f:
    #    stg_values = json.load(f)

    print("Simple Wind Gridworld")
    train_betweenness_agents(simple_wind_gridworld,
                             training_timesteps, num_agents, evaluate_policy_window,
                             True, 'options_trained.json',
                             total_evaluation_steps, False,
                             progress_bar=True)
    exit()

    print("Tiny Town 3x3: Eigenoptions Training Agent")
    eigenoptions_agent = EigenOptionAgent(adj_matrix, all_states, 0.9, 0.1, 0.9,
                                          tinytown.possible_actions)
    train_eigenoption_agents('eigenoptions_options_trained_agent.json', tinytown,
                             training_timesteps, num_agents, evaluate_policy_window,
                             False, total_evaluation_steps,
                             continue_training=True,
                             progress_bar=True)
    eigenoptions_agent.save(filenames[4] + '/eigenoptions_options_trained_agent.json')
    exit()

    preparedness_values, hierarchy = preparedness_efficient(adj_matrix, 0.5, min_num_hops=1, max_num_hops=10,
                                                            compressed_matrix=True, existing_stg_values=stg_values,
                                                            computed_hops_range=[1, 8])

    print("Hierarchy height: " + str(hierarchy))

    with open(filenames[3], 'w') as f:
        json.dump(preparedness_values, f)
    exit()

    data = graphing.extract_data(filenames[5])
    graphing.graph_reward_per_timestep(data, graphing_window,
                                       name='Tiny Town (3x3)',
                                       x_label='Epoch',
                                       y_label='Average Epoch Return',
                                       error_bars='std')
    exit()

    train_eigenoption_agents('eigenoptions_options_trained_agent.json', simple_wind_gridworld,
                             training_timesteps, num_agents,
                             evaluate_policy_window,
                             total_eval_steps=total_evaluation_steps,
                             continue_training=False,
                             progress_bar=True)
    exit()

    eigenoptions_agent = EigenOptionAgent(adj_matrix, all_states,
                                          0.9, 0.1, 0.9,
                                          simple_wind_gridworld.possible_actions)
    eigenoptions_agent.load(filenames[4] + '/eigenoptions_options_trained_agent.json')

    preparedness_values, hierarchy = preparedness_efficient(adj_matrix, 0.5, min_num_hops=1, max_num_hops=2,
                                                            compressed_matrix=True,
                                                            existing_stg_values=stg_values, computed_hops_range=None)

    eigenoptions_agent.train_options(simple_wind_gridworld, options_training_timesteps,
                                     True, True)
    eigenoptions_agent.save(filenames[4] + '/eigenoptions_options_trained_agent')
    exit()
    add_eigenoptions_to_stg(eigenoptions_agent, simple_wind_gridworld)
    exit()

    print("Creating Agent")
    eignoptions_agent = EigenOptionAgent(adj_matrix, all_states,
                                         0.9, 0.1, 0.9,
                                         simple_wind_gridworld.possible_actions)
    print("Finding Eigenoptions")
    eignoptions_agent.find_options(True)
    eignoptions_agent.save(filenames[4] + '/eigenoptions_base_agent')
    exit()

    train_q_learning_agent(simple_wind_gridworld,
                           training_timesteps, num_agents,
                           progress_bar=True,
                           total_eval_steps=total_evaluation_steps)
    exit()

    louvain_agent = LouvainAgent(tiny_town_env.possible_actions,
                                 state_transition_graph, int,
                                 (3, 3))
    louvain_agent.apply_louvain()
    louvain_agent.create_options()
    louvain_agent.train_options_value_iteration(0.001, 100,
                                                tiny_town_env, False)
    exit()

    state_transition_graph, stg_values = betweenness(state_transition_graph, stg_values)

    nx.write_gexf(state_transition_graph, stg_filename)
    with open(stg_values_filename, 'w') as f:
        json.dump(stg_values, f)
    exit()

    train_q_learning_agent(tiny_town_env,
                           agent_directory, results_directory,
                           training_timesteps, num_agents, evaluate_policy_window,
                           all_actions_valid=False, continue_training=True,
                           progress_bar=True)
    exit()

    stg = nx.read_gexf(stg_filename)

    louvain_agent = LouvainAgent(tiny_town_env.possible_actions, stg, tiny_town_env.state_dtype, (5, 1),
                                 min_hierarchy_level=0)
    louvain_agent.apply_louvain(graph_save_path=stg_filename)
    louvain_agent.create_options()
    louvain_agent.print_options()

    # Training Louvain Subgoals
    louvain_agent.train_options(options_training_timesteps, tiny_town_env,
                                True, True)

    # Training Louvain Agent
    train_louvain_agents(tiny_town_env, tiny_town_env.environment_name,
                         agent_directory, results_directory,
                         training_timesteps, num_agents, evaluate_policy_window,
                         initial_agent=louvain_agent,
                         all_actions_valid=True,
                         total_eval_steps=total_evaluation_steps,
                         progress_bar=True)

    exit()

    find_save_stg_subgoals(tiny_town_env, tiny_town_env.environment_name,
                           True, max_num_hops=1
                           )
    exit()

    print("Training DIAYN")
    train_diayn_agent(taxicab_env, results_directory, 2,
                      training_timesteps, num_agents, evaluate_policy_window,
                      skill_training_episodes=2, skill_length=3, model_layers=[4, 4],
                      skill_training_max_steps_per_episode=50,
                      progress_bar=True)
    exit()

    # stg = nx.read_gexf(stg_filename)
    # aggregate_graphs, stg = generate_aggregate_graphs(stg, apply_louvain, {'return_aggregate_graphs': True})

    train_multi_level_agent(
        (TaxiCab, {'use_time': False, 'use_fuel': False, 'arrival_probabilities': [0.25, 0.01, 0.01, 0.01, 0.72],
                   'hashable_states': True}, taxicab_env.environment_name),
        0.1, 0.9, 0.9, 0,
        False, 3, training_timesteps, 1, options_training_timesteps, 250,
        1, False,
        results_directory, aggregate_graphs, stg, 0
    )
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

    print("Training DADS")
    train_dads_agent(taxicab_env, results_directory, 2,
                     training_timesteps, num_agents, evaluate_policy_window,
                     skill_training_cycles=200, skill_length=3, model_layers=[16, 16],
                     skill_training_steps=10,
                     progress_bar=True)
    exit()

    train_sac_agent(env, state_shape,
                    results_directory,
                    training_timesteps, num_agents, evaluate_policy_window,
                    all_actions_valid=True,
                    total_eval_steps=total_evaluation_steps,
                    progress_bar=True)
    exit()

    print(env.environment_name + ' finding stg and subgoals')
    find_save_stg_subgoals(env, env.environment_name, True,
                           beta_values=[x / 10 for x in range(1, 10)],
                           max_num_hops=max_num_hops,
                           find_betweenness=True)

    for hops in range(1, max_num_hops + 1):
        add_local_maxima_to_file(env.environment_name,
                                 'preparedness - ' + str(hops) + ' hops - beta = 0.5', hops,
                                 progress_bar=True)

    exit()

    if not os.path.isdir(agent_directory):
        os.mkdir(agent_directory)

    stg = nx.read_gexf(stg_filename)

    preparedness_subgoals = get_preparedness_subgoals(env, 0.5)
    agent = MultiLevelGoalAgent(env.possible_actions, 0.9, 0.1, 0.9,
                                preparedness_subgoals, stg, state_dtype=int)
    agent.print_options()
    agent.train_options(env, options_training_timesteps, all_actions_valid=False, progress_bar=True)
    train_multi_level_preparedness_agents(env, env.environment_name,
                                          agent_directory, results_directory,
                                          training_timesteps, num_agents, evaluate_policy_window,
                                          total_eval_steps=total_evaluation_steps,
                                          initial_agent=agent, all_actions_valid=False,
                                          progress_bar=True)
    exit()

    print(env.environment_name + ' training subgoal options')

    train_subgoal_options(env, env.environment_name,
                          options_training_timesteps,
                          ['preparedness - ' + str(hops) + ' hops - beta = 0.5 - local maxima'
                           for hops in range(min_num_hops, max_num_hops + 1)] +
                          ['betweenness local maxima'],
                          options_save_directory=options_save_directory,
                          all_actions_valid=True,
                          progress_bar=True)
    exit()

    train_subgoal_agent(env,
                        ['preparedness - ' + str(hops) + ' hops - beta = 0.5 - local maxima'
                         for hops in range(min_num_hops, max_num_hops + 1)] +
                        ['betweenness local maxima'],
                        env.environment_name,
                        options_save_directory, agent_directory, results_directory,
                        training_timesteps, num_agents, evaluate_policy_window,
                        total_eval_steps=100,
                        all_actions_valid=True,
                        progress_bar=True)
    exit()

    train_q_learning_agent(env,
                           agent_directory, results_directory,
                           training_timesteps, num_agents, evaluate_policy_window,
                           True,
                           total_eval_steps=100,
                           progress_bar=True)
    exit()

    with open(env.environment_name + '_stg_values.json', 'r') as f:
        data = json.load(f)

    print_subgoals(data, 'preparedness - 4 hops - beta = 0.5 - local maxima')
    exit()

    print(env.environment_name + ' training subgoal agents')

    env.visualise_subgoals('betweenness local maxima')
    exit()

    print(env.environment_name + " - Training Preparedness - 1 hops Agent")

    train_subgoal_agent(env, ['preparedness - 5 hops - beta = 0.5 - local maxima'],
                        env.environment_name,
                        options_save_directory, agent_directory, results_directory,
                        training_timesteps, num_agents, evaluate_policy_window,
                        all_actions_valid=False,
                        progress_bar=False)
    exit()

    print(env.environment_name + ' stg - subgoals')

    for hops in range(1, max_num_hops + 1):
        intrinsic_reward = create_preparedness_reward_function(env.environment_name, hops,
                                                               beta=0.5)
        train_q_learning_agent(env, agent_directory, results_directory,
                               training_timesteps, num_agents, evaluate_policy_window,
                               all_actions_valid=False,
                               intrinsic_reward=intrinsic_reward,
                               intrinsic_reward_lambda=intrinsic_reward_lambda,
                               progress_bar=True,
                               file_save_name='preparedness_' + str(hops) + '_hops_intrinsic_reward')
    exit()

    train_betweenness_agents(env, env.environment_name,
                             options_save_directory, agent_directory, results_directory,
                             training_timesteps, num_agents, evaluate_policy_window,
                             all_actions_valid=False,
                             progress_bar=False)
    exit()
