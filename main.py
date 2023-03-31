import json
import math
import os

import networkx as nx
from networkx import DiGraph
import numpy as np
import random as rand

import environments.taxicab
from environments.environment import Environment
from environments.deepseaexplore import DeepSeaExplore
from environments.fourroom import FourRoom
from environments.hanoi import HanoiEnvironment
from environments.keycard import KeyCard
from environments.railroad import RailRoad
from environments.sixroom import SixRoom
from environments.taxicab import TaxiCab
from environments.tinytown import TinyTown
from environments.waterbucket import WaterBucket
from graph import Graph
import graphing
from learning_agents.optionsagent import Option, OptionsAgent, generate_options_from_goals
from learning_agents.qlearningagent import QLearningAgent


def find_goals(env: Environment, num_steps, assign_frequency, t_o, t_p, alpha, beta):
    goals = []
    state_hit_obs = {}

    def add_to_state_hit_obs(s, obs_to_add, hits_to_add):
        try:
            state_hit_obs[s][0] += obs_to_add
            state_hit_obs[s][1] += hits_to_add
        except KeyError:
            state_hit_obs[s] = [obs_to_add, hits_to_add]
        return

    def add_transition_to_graph(graph, s, s_prime):
        s_added = graph.add_node(s)
        s_prime_added = graph.add_node(s_prime)

        s_node = graph.get_node(s)
        try:
            s_node.pointers[s_prime] += 1
        except KeyError:
            s_prime_node = graph.get_node(s_prime)
            s_node.add_pointer(s_prime_node, 1)

        if s_added:
            add_to_state_hit_obs(s, 1, 0)
        if s_prime_added:
            add_to_state_hit_obs(s_prime, 1, 0)

        return

    def rank_states(graph):
        adjacency_matrix = np.zeros((graph.num_nodes, graph.num_nodes))

        for i in range(graph.num_nodes):
            node = graph.nodes[i]
            for j in range(graph.num_nodes):
                pointing_to = graph.nodes[j]
                if j == i:
                    weight = 1.0
                else:
                    try:
                        weight = node.pointers[pointing_to]
                    except KeyError:
                        weight = 0.0
                adjacency_matrix.itemset(i, j, weight)

        def get_key_func(k):
            def get_value_k(d):
                return d[k]

            return get_value_k

        li_array = [find_local_influence(adjacency_matrix, node, alpha=alpha) for node in range(graph.num_nodes)]
        ii_array = [find_indirect_influence(adjacency_matrix, node, li_array) for node in range(graph.num_nodes)]
        influences_array = [{'state': graph.nodes[node].name,
                             'local': li_array[node],
                             'indirect': ii_array[node],
                             'total': (beta * li_array[node] + (1 - beta) * ii_array[node])}
                            for node in range(graph.num_nodes)]
        influences_array.sort(key=get_key_func('total'), reverse=True)
        return [dic['state'] for dic in influences_array]

    possible_actions = env.possible_actions

    done = True
    trajectory_graph = None
    for i in range(num_steps):
        if done or i % assign_frequency == 0:
            if trajectory_graph is not None and trajectory_graph.num_nodes > 1:
                ranked_states = rank_states(trajectory_graph)
                max_power_state = ranked_states[0]

                add_to_state_hit_obs(max_power_state, 0, 1)
                num_observations = state_hit_obs[max_power_state][0]
                if num_observations > t_o:
                    hit_rate = state_hit_obs[max_power_state][1] / num_observations
                    if hit_rate > t_p:
                        if max_power_state not in goals:
                            goals.append(max_power_state)

            trajectory_graph = Graph()
        if done:
            state = env.reset()
            done = False
            trajectory_graph.add_node(state)
            add_to_state_hit_obs(state, 1, 0)

        action = rand.choice(possible_actions)
        next_state, _, done, _ = env.step(action)

        add_transition_to_graph(trajectory_graph, state, next_state)
        state = next_state

    return goals


def find_indirect_influence(adjacency_matrix: np.matrix, node, local_influences=None):
    nodes = list(range(adjacency_matrix.shape[0]))

    if local_influences is None:
        local_influences = [find_local_influence(adjacency_matrix, i) for i in nodes]

    connected_nodes = get_connected_nodes(adjacency_matrix, node)

    two_hop_paths = {}
    for i in connected_nodes:
        connected_to_i = get_connected_nodes(adjacency_matrix, i)
        for j in connected_to_i:
            if i == j or j in connected_nodes:
                continue
            try:
                two_hop_paths[j].append((node, i))
            except KeyError:
                two_hop_paths[j] = [(node, i)]

    ii = 0
    num_two_hop_neighbours = 0
    for two_hop_neighbour in two_hop_paths:
        num_two_hop_neighbours += 1
        paths = two_hop_paths[two_hop_neighbour]
        to_add = sum([local_influences[path[0]] * local_influences[path[1]] for path in paths]) / len(paths)
        ii += to_add
    if num_two_hop_neighbours == 0:
        return 0
    ii *= 1 / num_two_hop_neighbours
    return ii


def find_local_influence(adjacency_matrix: np.matrix, node, alpha=1, log_base=10, weighted=False):
    if not weighted:
        alpha = 0

    connected_nodes = get_connected_nodes(adjacency_matrix, node)
    num_one_hop_nodes = len(connected_nodes)
    one_hop_nodes = np.zeros((num_one_hop_nodes, num_one_hop_nodes))
    for i in range(num_one_hop_nodes):
        for j in range(num_one_hop_nodes):
            one_hop_nodes.itemset(i, j, adjacency_matrix.item(connected_nodes[i], connected_nodes[j]))

    sdc_values = []
    for i in range(num_one_hop_nodes):
        sdc = 0
        k = 0
        for j in range(num_one_hop_nodes):
            if i == j:
                continue
            weight_j_i = one_hop_nodes.item(j, i)
            sdc += weight_j_i
            if weight_j_i > 0:
                k += 1
        sdc *= (k / num_one_hop_nodes) ** alpha
        sdc_values.append(sdc)
    sdc_sum = sum(sdc_values)

    if sdc_sum <= 0:
        return 0
    li = math.log(sdc_sum, log_base)
    to_add = 0
    for sdc in sdc_values:
        if sdc > 0:
            to_add += sdc * math.log(sdc, log_base)
    li -= to_add / sdc_sum

    return li


def find_local_maxima(adjacency_matrix: np.matrix, values):
    nodes = range(adjacency_matrix.shape[0])
    local_maxima = []

    for node in nodes:
        connected_nodes = get_undirected_connected_nodes(adjacency_matrix, node)
        is_maxima = all([values[node] > values[connected_node] for connected_node in connected_nodes])
        local_maxima.append(is_maxima)
    return local_maxima


def find_weighted_local_influence(adjacency_matrix: np.matrix, node, theta=0.5, log_base=10, accuracy=4):
    connected_nodes = get_connected_nodes(adjacency_matrix, node)
    num_one_hop_nodes = len(connected_nodes)
    one_hop_nodes = np.zeros((num_one_hop_nodes, num_one_hop_nodes))
    for i in range(num_one_hop_nodes):
        for j in range(num_one_hop_nodes):
            one_hop_nodes.itemset(i, j, adjacency_matrix.item(connected_nodes[i], connected_nodes[j]))

    out_degree_values = [0 for _ in range(num_one_hop_nodes)]
    in_degree_values = [0 for _ in range(num_one_hop_nodes)]
    for i in range(num_one_hop_nodes):
        for j in range(num_one_hop_nodes):
            if i == j:
                continue
            weight_i_j = one_hop_nodes.item(i, j)
            if weight_i_j > 0:
                out_degree_values[i] += 1
                in_degree_values[j] += 1
    sdc_values = [out_degree_values[i] + in_degree_values[i] for i in range(num_one_hop_nodes)]
    sdc_sum = sum(sdc_values)

    if sdc_sum <= 0:
        return 0

    struc_ent = math.log(sdc_sum, log_base)
    to_add = 0
    for sdc in sdc_values:
        if sdc > 0:
            to_add += sdc * math.log(sdc, log_base)
    struc_ent -= to_add / sdc_sum
    struc_ent = round(struc_ent, accuracy)

    trans_ent = 0
    weight_sum = 0.0
    weight_log_sum = 0.0
    for j in connected_nodes:
        if node == j:
            continue
        weight = adjacency_matrix.item(node, j)
        weight_sum += weight
        weight_log_sum += weight * math.log(weight, log_base)
    if weight_sum > 0:
        trans_ent = math.log(weight_sum, log_base) - (weight_log_sum / weight_sum)
    trans_ent = round(trans_ent, accuracy)

    li = theta * struc_ent + (1 - theta) * trans_ent

    return round(li, accuracy)


def get_undirected_connected_nodes(adjacency_matrix: np.matrix, node):
    nodes = list(range(adjacency_matrix.shape[0]))

    connected_nodes = []
    for i in nodes:
        if i == node:
            continue
        if adjacency_matrix.item(node, i) > 0 or adjacency_matrix.item(i, node) > 0:
            connected_nodes.append(i)
    return connected_nodes


def get_connected_nodes(adjacency_matrix: np.matrix, node):
    nodes = list(range(adjacency_matrix.shape[0]))

    connected_nodes = []
    for i in nodes:
        if adjacency_matrix.item(node, i) > 0 or node == i:
            connected_nodes.append(i)
    return connected_nodes


def group_graph_by_local_maxima(adjacency_matrix: np.matrix, values):
    n = adjacency_matrix.shape[0]
    groups = [None for _ in range(n)]
    num_groups = 0

    all_grouped = False

    def get_value(v):
        return values[v]

    while not all_grouped:

        num_none = sum([1 for g in groups if g is None])
        all_grouped = True

        for node in range(n):
            group = groups[node]
            if group is not None:
                continue

            all_grouped = False
            sorted_connected_nodes = get_undirected_connected_nodes(adjacency_matrix, node)
            sorted_connected_nodes.sort(key=get_value, reverse=True)
            node_value = values[node]
            connected_groups = [groups[connected_node] for connected_node in sorted_connected_nodes]
            connected_values = [values[connected_node] for connected_node in sorted_connected_nodes]

            if len(sorted_connected_nodes) == 0 or node_value > values[sorted_connected_nodes[0]]:
                groups[node] = num_groups
                num_groups += 1
                continue

            group_to_set = None
            for i in range(1, len(sorted_connected_nodes)):
                node_bigger = sorted_connected_nodes[i - 1]
                if values[sorted_connected_nodes[i]] < node_value <= values[node_bigger]:
                    group_to_set = groups[node_bigger]
                    break

            if group_to_set is None:
                group_to_set = groups[sorted_connected_nodes[-1]]

            groups[node] = group_to_set

    return groups


def partition_env_by_ncut(env: Environment, num_partitions):
    # get matricies
    # find eigenvalues and vectors, pick second eigenvalue for partition
    # for each possible partition, find NCUT
    # choose partition with lowest NCUT value

    # Return array of partitions
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


def compute_betweeness(adj_matrix):
    env_nodes = range(adj_matrix.shape[0])

    def get_neighbours(node_to_get):
        return [i for i in env_nodes if i != node_to_get and adj_matrix[node_to_get][i] > 0]

    node_betweenness_array = [0 for _ in env_nodes]
    for node in env_nodes:
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


def rank_environment_by_influences(env, alpha=1.0, beta=0.5, accuracy=3, weighted=False):
    adj_matrix = env.get_adjacency_matrix()

    influences = rank_graph_by_influence(adj_matrix, alpha=alpha, beta=beta, weighted=weighted)

    nodes = []
    num_nodes = 0
    for item in influences:
        x_y = env.index_to_state(item['node'])
        try:
            is_valid = env.valid_state(x_y['x'], x_y['y'])
        except:
            is_valid = True
        if is_valid:
            nodes.append({'node': x_y, 'influence': round(item['total'], accuracy)})
            num_nodes += 1
    rankings = []
    to_add = [nodes[0]['node']]
    current_influence = nodes[0]['influence']
    ranking_influences = []
    for i in range(1, num_nodes):
        node = nodes[i]
        influence = node['influence']
        if influence < current_influence:
            rankings.append(to_add)
            to_add = [node['node']]
            ranking_influences.append(current_influence)
            current_influence = influence
        else:
            to_add.append(node['node'])
    rankings.append(to_add)
    ranking_influences.append(current_influence)
    print('Influences')
    for influence in ranking_influences:
        print(influence)
    return rankings, ranking_influences


def rank_environment_by_influences_multiple(env: Environment, num_rankings,
                                            steps_per_ranking, assign_freq, min_obs, min_hit_rate, a, b):
    print("Finding Goals")
    goal_results = []
    for _ in range(num_rankings):
        env_goals = find_goals(env, steps_per_ranking, assign_freq, min_obs, min_hit_rate, a, b)
        for goal in env_goals:
            goal_result_found = False
            for goal_result in goal_results:
                if goal_result['goal'] == goal:
                    goal_result['result'] += 1
                    goal_result_found = True
                    break
            if not goal_result_found:
                goal_results.append({'goal': goal, 'result': 1})
    print("Goals Found:")

    def get_key_func(k):
        def get_value_k(d):
            return d[k]

        return get_value_k

    goal_results.sort(key=get_key_func('result'), reverse=True)
    for goal_result in goal_results:
        print(goal_result['goal'], ' | ', goal_result['result'])

    rankings = []
    to_add = [goal_results[0]['goal']]
    current_value = goal_results[0]['result']
    ranking_values = [current_value]
    for i in range(1, len(goal_results)):
        value = goal_results[i]['result']
        if value < current_value:
            rankings.append(to_add)
            ranking_values.append(value)
            current_value = value
            to_add = [goal_results[i]['goal']]
            continue
        to_add.append(goal_results[i]['goal'])
    rankings.append(to_add)

    rankings_adjusted = []
    for ranking in rankings:
        ranking = [env.code_to_state(goal) for goal in ranking]
        try:
            ranking = [{'x': goal['taxi_x'],
                        'y': goal['taxi_y'],
                        'stop': goal['passenger_location']}
                       for goal in ranking]
            for goal in ranking:
                if goal['stop'] < 4:
                    goal['stop'] += 1
                elif goal['stop'] == 4:
                    goal['stop'] = 0
        except KeyError:
            ()
        rankings_adjusted.append(ranking)
    return rankings_adjusted, ranking_values


def rank_graph_by_influence(adjacency_matrix, alpha=1, beta=0.5, log_base=10, weighted=False):
    def get_key_func(k):
        def get_value_k(d):
            return d[k]

        return get_value_k

    num_nodes = range(adjacency_matrix.shape[0])

    li_array = [find_local_influence(adjacency_matrix, node, alpha=alpha, log_base=log_base, weighted=weighted)
                for node in num_nodes]
    ii_array = [find_indirect_influence(adjacency_matrix, node, li_array)
                for node in num_nodes]
    influences_array = [{'node': node,
                         'local': li_array[node],
                         'indirect': ii_array[node],
                         'total': (beta * li_array[node] + (1 - beta) * ii_array[node])}
                        for node in num_nodes]
    influences_array.sort(key=get_key_func('total'), reverse=True)
    return influences_array


def train_agent(env: Environment, agent, num_episodes, true_state=False, all_actions_valid=True, agent_save_path=None):
    agent_returns = []
    agent_successes = []

    track_successes = True

    for _ in range(num_episodes):
        done = False
        state = env.reset(true_state)
        if not all_actions_valid:
            current_possible_actions = env.get_possible_actions()

        while not done:
            if all_actions_valid:
                action = agent.choose_action(state)
            else:
                action = agent.choose_action(state, possible_actions=current_possible_actions)
            next_state, reward, done, info = env.step(action, true_state)

            if all_actions_valid:
                agent.learn(state, action, reward, next_state)
            else:
                current_possible_actions = env.get_possible_actions()
                agent.learn(state, action, reward, next_state, terminal=done,
                            next_state_possible_actions=current_possible_actions)

                agent_returns.append(reward)
                state = next_state

        if track_successes:
            try:
                agent_successes.append(info['success'])
            except TypeError:
                track_successes = False

    if agent_save_path is not None:
        agent.save(agent_save_path)

    if track_successes:
        return agent, agent_returns, agent_successes
    return agent, agent_returns


def compute_graph_entropy_values(adjacency_matrix, weighted=False, theta=0.5, beta=0.5, accuracy=10,
                                 print_values=False):
    if weighted:
        li_values = [round(find_weighted_local_influence(adjacency_matrix=adjacency_matrix, node=node, theta=theta),
                           accuracy)
                     for node in range(adjacency_matrix.shape[0])]
    else:
        li_values = [round(find_local_influence(adjacency_matrix=adjacency_matrix,
                                                node=node), accuracy)
                     for node in range(adjacency_matrix.shape[0])]

    ii_values = [round(find_indirect_influence(adjacency_matrix=adjacency_matrix,
                                               node=node, local_influences=li_values), accuracy)
                 for node in range(adjacency_matrix.shape[0])]

    if print_values:
        for i in range(adjacency_matrix.shape[0]):
            print(str(i + 1) + " " + str(li_values[i]) + " " + str(round(ii_values[i], accuracy)) + ' ' +
                  str(round(beta * li_values[i] + (1 - beta) * ii_values[i], accuracy)))

    influences = {i: {'local influence': li_values[i],
                      'indirect influence': ii_values[i],
                      'total influence': round(beta * li_values[i] + (1 - beta) * ii_values[i], 4)}
                  for i in range(adjacency_matrix.shape[0])}
    return influences


def get_results(environment: Environment, num_timesteps, options_training_timesteps, num_agents,
                entropy_cutoff, hand_crafted_goals=None, all_actions_valid=True,
                agent_save_directory="", results_save_directory="", graph_name="",
                q_learning_alpha=0.9, q_learning_epsilon=0.1, q_learning_gamma=0.9):
    adj_matrix, all_states = environment.get_adjacency_matrix(directed=True)
    all_states_bytes = [state.tobytes() for state in all_states]
    stg = nx.from_numpy_array(adj_matrix, create_using=nx.MultiDiGraph())
    print("Finding Goals")
    print("    Adjacency Matrix Found")
    inf_values = compute_graph_entropy_values(adj_matrix, weighted=True, beta=1.0)
    print("    Influence Values Computed")
    betweenness = compute_betweeness(adj_matrix)
    print("    Betweenness Values Computed")

    is_betweeness_local_maxima = find_local_maxima(adj_matrix,
                                                   [betweenness[i]['betweenness'] for i in range(adj_matrix.shape[0])])
    betweeness_local_maxima = [all_states[i] for i in range(adj_matrix.shape[0])
                               if is_betweeness_local_maxima[i]]

    print("    Entropy Goals")
    entropy_goals = []
    for i in inf_values:
        if inf_values[i]['local influence'] >= entropy_cutoff:
            entropy_goals.append(all_states[i])
            print(np.array2string(all_states[i]))
    print(str(len(entropy_goals)) + " Entropy Goals Found")

    print("    Betweenness Goals")
    for state in betweeness_local_maxima:
        print(np.array2string(state))
    print(str(len(betweeness_local_maxima)) + " Betweenness goals found")

    if hand_crafted_goals is not None:
        print("    Handcrafted Goals")

    all_q_learner_agent_returns = []
    print("Training Q Learner")
    for agent in range(num_agents):
        q_learning_agent = QLearningAgent(environment.possible_actions,
                                          q_learning_alpha, q_learning_epsilon, q_learning_gamma)
        save_path = agent_save_directory + 'q_learner' + str(agent) + '.json'
        q_learning_agent, q_learning_agent_returns = train_agent(environment, q_learning_agent,
                                                                 num_timesteps, true_state=True,
                                                                 all_actions_valid=all_actions_valid,
                                                                 agent_save_path=save_path)
        all_q_learner_agent_returns.append(q_learning_agent_returns)
    with open(results_save_directory + 'q_learner_returns.json', 'w') as f:
        q_learner_returns_data = {i: all_q_learner_agent_returns[i] for i in range(num_agents)}
        json.dump(q_learner_returns_data, f)
    print("    Q learner Trained")

    if hand_crafted_goals is not None:
        print("Training Handcrafted Options")
        handcrafted_options = generate_options_from_goals(environment, stg, hand_crafted_goals, all_states_bytes,
                                                          options_training_timesteps,
                                                          true_state=True, all_actions_valid=all_actions_valid)

        print("    Handcrafted Options Trained")
        for a in environment.possible_actions:
            handcrafted_options.append(Option(actions=[a]))

        all_handcrafted_agent_returns = []
        for agent in range(num_agents):
            handcrafted_agent = OptionsAgent(alpha=q_learning_alpha, epsilon=q_learning_epsilon, gamma=q_learning_gamma,
                                             options=handcrafted_options)
            save_path = agent_save_directory + 'handcrafted_goals' + str(agent) + '.json'
            handcrafted_agent, handcrafted_agent_returns = train_agent(environment, handcrafted_agent, num_timesteps,
                                                                       true_state=True,
                                                                       all_actions_valid=all_actions_valid,
                                                                       agent_save_path=save_path)
            all_handcrafted_agent_returns.append(handcrafted_agent_returns)
        with open(results_save_directory + 'handcrafted_returns.json', 'w') as f:
            handcrafted_returns_data = {i: all_handcrafted_agent_returns[i] for i in range(num_agents)}
            json.dump(handcrafted_returns_data, f)
        print("    Handcrafted Agent Trained")

    print("Entropy Influence Agent")

    entropy_options = generate_options_from_goals(environment, stg, entropy_goals, all_states_bytes,
                                                  options_training_timesteps,
                                                  true_state=True, all_actions_valid=all_actions_valid)
    print("    Entropy Options Trained")
    for a in environment.possible_actions:
        entropy_options.append(Option(actions=[a]))

    all_option_agent_returns = []
    for _ in range(num_agents):
        options_agent = OptionsAgent(alpha=q_learning_alpha, epsilon=q_learning_epsilon, gamma=q_learning_gamma,
                                     options=entropy_options)
        save_path = agent_save_directory + 'entropy_goals' + str(agent) + '.json'
        options_agent, options_agent_returns, = train_agent(environment, options_agent, num_timesteps,
                                                            true_state=True, all_actions_valid=all_actions_valid,
                                                            agent_save_path=save_path)
        all_option_agent_returns.append(options_agent_returns)
    with open(results_save_directory + 'entropy_returns.json', 'w') as f:
        entropy_returns_data = {i: all_option_agent_returns[i] for i in range(num_agents)}
        json.dump(entropy_returns_data, f)
    print("    Entropy Options agent trained")

    print("Betweenness Agent")

    betweenness_options = generate_options_from_goals(environment, stg,
                                                      betweeness_local_maxima, all_states_bytes,
                                                      options_training_timesteps,
                                                      true_state=True, all_actions_valid=all_actions_valid)
    print("    Betweenness Options Trained")

    for a in environment.possible_actions:
        betweenness_options.append(Option(actions=[a]))

    all_betweenness_agent_returns = []
    for _ in range(num_agents):
        betweenness_agent = OptionsAgent(alpha=q_learning_alpha, epsilon=q_learning_epsilon, gamma=q_learning_gamma,
                                         options=betweenness_options)
        save_path = agent_save_directory + 'betweenness_goals' + str(agent) + '.json'
        betweenness_agent, betweenness_agent_returns = train_agent(environment, betweenness_agent, num_timesteps,
                                                                   true_state=True,
                                                                   all_actions_valid=all_actions_valid,
                                                                   agent_save_path=save_path)
        all_betweenness_agent_returns.append(betweenness_agent_returns)
    with open(results_save_directory + 'betweenness_returns.json', 'w') as f:
        betweenness_returns_data = {i: all_betweenness_agent_returns[i] for i in range(num_agents)}
        json.dump(betweenness_returns_data, f)
    print("    Betweenness Agent Trained")
    return


if __name__ == "__main__":
    
    print("Hello world")
    exit()

    tiny_town_env = TinyTown(3, 3, pick_every=1)
    adj_matrix, all_states = tiny_town_env.get_adjacency_matrix()

    print("Finding Goals")
    print("    Adjacency Matrix Found")
    inf_values = compute_graph_entropy_values(adj_matrix, weighted=True, beta=1.0)
    print("    Influence Values Computed")
    betweenness = compute_betweeness(adj_matrix)
    print("    Betweenness Values Computed")
    is_betweeness_local_maxima = find_local_maxima(adj_matrix,
                                                   [betweenness[i]['betweenness'] for i in range(adj_matrix.shape[0])])
    betweeness_local_maxima = [all_states[i] for i in range(adj_matrix.shape[0])
                               if is_betweeness_local_maxima[i]]

    for i in inf_values:
        state = all_states[i]
        name = ""
        for x in range(tiny_town_env.width):
            for y in range(tiny_town_env.height):
                name += str(state[y, x])
            name += '\n'
        tile = str(state[tiny_town_env.height, tiny_town_env.width])

        inf_values[i]['name'] = name
        inf_values[i]['tile'] = tile
        inf_values[i]['betweenness'] = float(betweenness[i]['betweenness'])
        inf_values[i]['betweenness local maxima'] = str(betweeness_local_maxima[i])

    pick_key = {0: 'random', 1: 'choice'}
    g = nx.from_numpy_matrix(adj_matrix, create_using=nx.MultiDiGraph())
    nx.set_node_attributes(g, inf_values)
    nx.write_gexf(g, 'tiny_towns_' + str(tiny_town_env.height) + 'x' + str(tiny_town_env.width) + '_' +
                  pick_key[tiny_town_env.pick_every] + '.gexf')

    '''

    data = graphing.extract_data('results/railroadrandom')
    graphing.graph_reward_per_timestep(data, window=500,
                                       name='Railroad 3x3 Random',
                                       labels=['Betweenness', 'Entropy', 'Q-Learner'],
                                       x_label='Timestep', y_label='Reward per Timestep')
    exit()

    environments_to_train = [TinyTown(2, 2, pick_every=1), TinyTown(2, 2, pick_every=0),
                             RailRoad(3, 3, tile_generation='choice'),
                             RailRoad(3, 3, tile_generation='random')]
    save_file_starts = ['learning_agents/tinytown/',
                        'learning_agents/tinytownrandom2x2/',
                        'learning_agents/railroadchoice/',
                        'learning_agents/railroadrandom/']
    results_file_starts = ['results/tinytown/',
                           'results/tinytownrandom2x2/',
                           'results/railroadchoice/',
                           'results/railroadrandom/']
    graph_names = ['Tiny Town Choice 2x2', 'Tiny Town Random 2x2',
                   'Railroad Choice 3x3',
                   'Railroad Random 3x3']
    entropy_thresholds = [0.8, 0.7, 0.9, 1.0]

    q_learning_alpha = 0.9
    q_learning_epsilon = 0.1
    q_learning_gamma = 0.9

    training_episodes = [10000, 10000, 10000, 10000]
    num_agents = 20

    for i in range(3, 4):
        print(graph_names[i])
        get_results(environments_to_train[i],
                    training_episodes[i], 50000, num_agents,
                    entropy_cutoff=entropy_thresholds[i], all_actions_valid=False,
                    agent_save_directory=save_file_starts[i], results_save_directory=results_file_starts[i],
                    q_learning_alpha=q_learning_alpha, q_learning_gamma=q_learning_gamma,
                    q_learning_epsilon=q_learning_epsilon, graph_name=graph_names[i])

    def get_rolling_sum(a):
        total = 0
        rolling_sum = []
        for elm in a:
            total += elm
            rolling_sum.append(total)
        return rolling_sum


    def get_total_reward_per_window(data, window):
        reward_per_timestep = []
        current_sum = 0
        for i in range(len(data)):
            if (i + 1) % window == 0:
                reward_per_timestep.append(current_sum)
                current_sum = 0
            current_sum += data[i]
        return reward_per_timestep


    f = open('results/tinytownq_learner_returns.json')
    q_learning_data = json.load(f)
    f.close()

    f = open('results/tinytownbetweenness_returns.json')
    betweenness_data = json.load(f)
    f.close()

    f = open('results/tinytownentropy_returns.json')
    entropy_data = json.load(f)
    f.close()

    k = 10
    window = 100
    q_learning_averaged = graphing.get_averages(list(q_learning_data.values()))
    q_learning_rolling_sum = get_rolling_sum(q_learning_averaged)
    q_learning_rolling_average = graphing.get_rolling_average(q_learning_averaged, k)
    q_learning_total_reward_per_window = get_total_reward_per_window(q_learning_averaged, window)
    q_learning_reward_per_timestep = get_reward_per_timstep(q_learning_averaged, window)
    betweenness_averaged = graphing.get_averages(list(betweenness_data.values()))
    betweenness_rolling_sum = get_rolling_sum(betweenness_averaged)
    betweenness_rolling_average = graphing.get_rolling_average(betweenness_averaged, k)
    betweenness_reward_per_timestep = get_reward_per_timstep(betweenness_averaged, window)
    entropy_averaged = graphing.get_averages(list(entropy_data.values()))
    entropy_rolling_sum = get_rolling_sum(entropy_averaged)
    entropy_rolling_average = graphing.get_rolling_average(entropy_averaged, k)
    entropy_reward_per_timestep = get_reward_per_timstep(entropy_averaged, window)

    to_graph = list(map(lambda x: get_reward_per_timstep(x, window),
                        [q_learning_averaged, betweenness_averaged, entropy_averaged]))

    graphing.graph_multiple(to_graph,
                            name='Tiny Town Choice 2x2',
                            labels=['Q-learner', 'Betweenness', 'Entropy'],
                            x_label='timestep', y_label='Reward per Timestep')

    exit()
    
    k = 5000
    window = 100
    q_learning_averaged = graphing.get_averages(list(q_learning_data.values()))
    q_learning_rolling_sum = get_rolling_sum(q_learning_averaged)
    q_learning_rolling_average = graphing.get_rolling_average(q_learning_averaged, k)
    q_learning_reward_per_timestep = get_reward_per_timstep(q_learning_averaged, window)
    betweenness_averaged = graphing.get_averages(list(betweeness_data.values()))
    betweenness_rolling_sum = get_rolling_sum(betweenness_averaged)
    betweenness_rolling_average = graphing.get_rolling_average(betweenness_averaged, k)
    betweenness_reward_per_timestep = get_reward_per_timstep(betweenness_averaged, window)
    entropy_averaged = graphing.get_averages(list(entropy_data.values()))
    entropy_rolling_sum = get_rolling_sum(entropy_averaged)
    entropy_rolling_average = graphing.get_rolling_average(entropy_averaged, k)
    entropy_reward_per_timestep = get_reward_per_timstep(entropy_averaged, window)

    n = 5001 - k
    graphing.graph_multiple([betweenness_reward_per_timestep,
                             entropy_reward_per_timestep],
                            name='Tiny Town Random (2x2)',
                            labels=['Betweenness', 'Entropy'],
                            x_label='20 n timestep', y_label='Average Reward per Timestep')
    
    print("Finding Goals")
    print("    Adjacency Matrix Found")
    inf_values = compute_graph_entropy_values(adj_matrix, weighted=True, beta=1.0)
    print("    Influence Values Computed")
    betweenness = compute_betweeness(adj_matrix)
    print("    Betweenness Values Computed")
    is_betweeness_local_maxima = find_local_maxima(adj_matrix,
                                                   [betweenness[i]['betweenness'] for i in range(adj_matrix.shape[0])])
    betweeness_local_maxima = [all_states[i] for i in range(adj_matrix.shape[0])
                               if is_betweeness_local_maxima[i]]


    for i in inf_values:
        state = all_states[i]
        name = ""
        for x in range(tiny_town_env.width):
            for y in range(tiny_town_env.height):
                name += str(state[y, x])
            name += '\n'
        tile = str(state[tiny_town_env.height, tiny_town_env.width])

        inf_values[i]['name'] = name
        inf_values[i]['tile'] = tile
        inf_values[i]['betweenness'] = float(betweeness[i]['betweenness'])
        inf_values[i]['influence group'] = inf_groups[i]
        inf_values[i]['betweenness group'] = betweeness_groups[i]
        inf_values[i]['influence local maxima'] = str(inf_local_maxima[i])
        inf_values[i]['betweenness local maxima'] = str(betweeness_local_maxima[i])

    pick_key = {0: 'random', 1: 'choice'}
    g = nx.from_numpy_matrix(adj_matrix, create_using=nx.MultiDiGraph())
    nx.set_node_attributes(g, inf_values)
    nx.write_gexf(g, 'tiny_towns_' + str(tiny_town_env.height) + 'x' + str(tiny_town_env.width) + '_' +
                  pick_key[tiny_town_env.pick_every] + '.gexf')

    
    tile_generation = 'random'
    railroad_env = RailRoad(width=3, height=3, stations=[(0, 0), (2, 2)], tile_generation=tile_generation)

    adj_matrix, all_states = railroad_env.get_adjacency_matrix()
    print("Graph formed")

    inf_values = compute_graph_entropy_values(adj_matrix, weighted=True, beta=1.0)
    print("Influences Found")

    betweeness = compute_betweeness(adj_matrix)
    print("Betweenness computed")

    inf_local_maxima = find_local_maxima(adj_matrix,
                                         [inf_values[i]['local influence'] for i in range(adj_matrix.shape[0])])
    print("Influence groups found")

    betweeness_groups = group_graph_by_local_maxima(adj_matrix,
                                                    [betweeness[i]['betweenness'] for i in range(adj_matrix.shape[0])])
    betweeness_local_maxima = find_local_maxima(adj_matrix,
                                                [betweeness[i]['betweenness'] for i in range(adj_matrix.shape[0])])
    print("Betweenness groups found")

    name_key = {0: '0', 1: ' | ', 2: '-', 3: '+'}
    for i in inf_values:
        state = all_states[i]
        name = ""
        for x in range(railroad_env.width):
            for y in range(railroad_env.height):
                name += name_key[state[y, x]]
            name += '\n'
        inf_values[i]['name'] = name
        if tile_generation == 'random':
            inf_values[i]['next tile'] = name_key[state[railroad_env.height, railroad_env.width]]
        to_set = float(betweeness[i]['betweenness'])
        inf_values[i]['betweenness'] = to_set
        inf_values[i]['betweenness group'] = betweeness_groups[i]
        inf_values[i]['influence local maxima'] = str(inf_local_maxima[i])
        inf_values[i]['betweenness local maxima'] = str(betweeness_local_maxima[i])

    g = nx.from_numpy_matrix(adj_matrix, create_using=nx.MultiDiGraph7())
    nx.set_node_attributes(g, inf_values)

    nx.write_gexf(g, 'railroad_graph_3x3x2_' + tile_generation + '.gexf')

    water_bucket_env = WaterBucket(buckets=[12, 8, 5, 3], start=np.array([[12, 12],
                                                                          [8, 0],
                                                                          [5, 0],
                                                                          [3, 0]]))

    adj_matrix, all_states = water_bucket_env.get_adjacency_matrix()

    inf_values = compute_graph_entropy_values(adj_matrix, weighted=True, beta=0.5, accuracy=4, print_values=True)
    betweenness = compute_betweeness(adj_matrix)

    name_key = {0: '0', 1: ' | ', 2: '-', 3: '+'}
    for i in inf_values:
        state = all_states[i]
        inf_values[i]['name'] = np.array2string(state)
        to_set = float(betweenness[i]['betweenness'])
        inf_values[i]['betweenness'] = to_set

    g = nx.from_numpy_matrix(adj_matrix, create_using=nx.MultiDiGraph())
    nx.set_node_attributes(g, inf_values)
    '''
