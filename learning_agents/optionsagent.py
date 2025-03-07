import json
import random as rand
import typing
import networkx as nx
import numpy as np
from scipy import sparse
from typing import Dict, List, Tuple

from environments.environment import Environment
from learning_agents.qlearningagent import QLearningAgent
from progressbar import print_progress_bar


def create_option_goal_initiation_func(goal_index, stg=None, adj_matrix=None, state_indexer=None):
    if (stg is None) and (adj_matrix is None):
        raise AttributeError("A state transition graph or adjacency matrix is required"
                             "to create an initiation function")

    if (adj_matrix is not None) and (state_indexer is None):
        raise AttributeError("A state-indexer is required to create an initiation function using"
                             "an adjacency matrix")

    if stg is not None:
        goal_state = stg.nodes[goal_index]['state']

        def get_state_index(state):
            s_str = np.array2string(np.ndarray.astype(state, dtype=int))
            for node in stg.nodes:
                if stg.nodes[node]['state'] == s_str:
                    return node
            return None

        def path_function(state_index):
            if state_index == goal_state:
                return False
            return nx.has_path(stg, state_index, goal_index)
    else:
        for goal_state in state_indexer:
            if state_indexer[goal_state] == goal_index:
                break

        distance_matrix = sparse.csgraph.shortest_path(adj_matrix)

        def get_state_index(state):
            return state_indexer[np.array2string(np.ndarray.astype(state, dtype=int))]

        def path_function(state_index):
            distance = distance_matrix[int(state_index), int(goal_index)]
            return (distance != 0) and (distance != np.inf)

    def initiation_func(state):
        path_exists = path_function(get_state_index(state))
        return path_exists

    return initiation_func


def generate_option_to_goal(environment: Environment, goal_index,
                                training_steps,
                                stg=None, adj_matrix=None, state_indexer=None,
                                all_actions_valid=True,
                                alpha=0.9, epsilon=0.1, gamma=0.9,
                                progress_bar=False, save_path=None):
    if (stg is None) and (adj_matrix is None):
        raise AttributeError("Require a state transition graph or adjacency matrix to create option")

    if (adj_matrix is not None) and (state_indexer is None):
        raise AttributeError("Require a state-indexer if using adjacency matrix to create option")

    if stg is not None:
        in_edges = stg.in_edges(goal_index)
        if len(in_edges) <= 0:
            return None

        def get_state_index(s):
            try:
                s_str = np.array2string(np.ndarray.astype(s, dtype=int))
            except TypeError:
                s_str = s
            for node in stg.nodes:
                if stg.nodes[node]['state'] == s_str:
                    return node
            return None
    else:
        if adj_matrix.getcol(goal_index).sum() <= 0:
            return None

        def get_state_index(s):
            return state_indexer[s]

    initiation_func = create_option_goal_initiation_func(goal_index, stg, adj_matrix, state_indexer)

    def terminating_func(s):
        return not initiation_func(s)

    policy = QLearningAgent(environment.possible_actions, alpha, epsilon, gamma)

    training_complete = False
    current_step = 0

    if progress_bar:
        print("Training option to goal " + str(goal_index))

    while not training_complete:
        done = False
        state = environment.reset()
        if terminating_func(state):
            continue

        if not all_actions_valid:
            current_possible_actions = environment.get_possible_actions()

        while not done:
            if all_actions_valid:
                action = policy.choose_action(state)
            else:
                action = policy.choose_action(state, possible_actions=current_possible_actions)

            next_state, _, done, _ = environment.step(action)
            current_step += 1

            if progress_bar:
                print_progress_bar(current_step, training_steps,
                                   prefix='Option Training: ', suffix='Complete')

            reward = -0.001
            np_array = np.ndarray.astype(next_state, dtype=int)
            target_index = get_state_index(np.array2string(np_array))
            if goal_index == target_index:
                done = True
                reward = 1.0
            elif terminating_func(next_state) or done:
                done = True
                reward = -1.0

            if all_actions_valid:
                policy.learn(state, action, reward, next_state, terminal=done)
            else:
                current_possible_actions = environment.get_possible_actions()
                policy.learn(state, action, reward, next_state, terminal=done,
                             next_state_possible_actions=current_possible_actions)

            state = next_state

            if current_step >= training_steps:
                training_complete = True
                break

    option = Option(environment.possible_actions, policy,
                    initiation_func=initiation_func,
                    terminating_func=terminating_func)

    if save_path is not None:
        policy.save(save_path)

    return option


class Option:

    def __init__(self, actions=[], policy=None, initiation_func=None, terminating_func=None):
        self.actions = actions
        self.policy = policy

        self.initiation_func = initiation_func
        self.terminating_func = terminating_func
        return

    def choose_action(self, state, possible_actions=None):
        if self.policy is None:
            try:
                action = self.actions[0]
                return action
            except IndexError:
                raise AttributeError('Option must have a defined policy or a set of actions')
        return self.policy.choose_action(state, True, possible_actions=possible_actions)

    def has_policy(self):
        return not (self.policy is None)

    def initiated(self, state: np.ndarray) -> bool:
        if self.initiation_func is None:
            return True
        return self.initiation_func(state)

    def save(self, save_path):
        if self.policy is None:
            return
        self.policy.save(save_path)
        return

    def terminated(self, state):
        if self.terminating_func is None:
            return True
        return self.terminating_func(state)


class OptionsAgent:

    def __init__(self, alpha, epsilon, gamma, options, step_size=None, state_dtype=int,
                 termination_func=None, intra_option=True):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        self.options = options

        self.step_size = step_size
        self.current_step = 0

        self.current_option = None
        self.current_option_index = None
        self.option_start_state = None
        self.total_option_reward = 0
        self.current_option_step = 0

        self.state_option_values = {}
        self.intra_state_option_values = {}

        self.intra_option = intra_option
        self.state_dtype = state_dtype

        self.termination_func = termination_func

        self.last_possible_actions = None
        return

    def choose_action(self, state, optimal_choice=False, possible_actions=None):
        self.last_possible_actions = possible_actions

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

        if chosen_action == -1:
            self.current_option = None
            self.current_option_index = None
            self.current_option_step = 0
            self.current_step = 0
            return self.choose_action(state, optimal_choice)

        self.current_option_step += 1
        self.current_step += 1
        return chosen_action

    def choose_option(self, state, no_random, possible_actions=None):
        self.current_option_step = 0
        self.option_start_state = state

        available_options = self.get_available_options(state, possible_actions=possible_actions)
        num_available_options = len(available_options)
        if num_available_options == 0:
            return None

        if not no_random and rand.uniform(0, 1) < self.epsilon:
            self.current_option_index = rand.choice(available_options)
            return self.options[int(self.current_option_index)]

        option_values = self.get_state_option_values(state, available_options)
        total_values = min(len(option_values), num_available_options)

        ops = [available_options[0]]
        try:
            max_value = option_values[available_options[0]]
        except KeyError:
            option_data = list(option_values.values())
            option_values = {}
            for i in range(total_values):
                option_values[available_options[i]] = option_data[i]
            max_value = option_data[0]
        for i in range(1, total_values):
            op = available_options[i]
            value = option_values[op]
            if value > max_value:
                max_value = value
                ops = [op]
            elif value == max_value:
                ops.append(op)

        self.current_option_index = rand.choice(ops)
        return self.options[int(self.current_option_index)]

    def copy_agent(self, copy_from):
        self.state_option_values = copy_from.state_option_values.copy()
        self.current_option = None
        self.current_option_index = None
        return

    def count_available_skills(self, state: np.ndarray, possible_actions: None|List[int]=None) -> int:
        available_skills = 0
        for option in self.options:
            if not option.has_policy():
                continue
            if option.initiated(state):
                available_skills += 1
        return available_skills

    def count_skills(self) -> Dict[int, int]:
        num_skills = 0
        for option in self.options:
            if option.has_policy():
                num_skills += 1
        return {1: num_skills}

    def get_available_options(self, state: np.ndarray, possible_actions: None|List[int]=None) -> List[int]:
        available_options = []
        option_index = 0
        for option in self.options:
            if (possible_actions is not None) and (not option.has_policy()):
                if option.actions[0] in possible_actions:
                    available_options.append(option_index)
                    option_index += 1
                    continue
            elif option.initiated(state):
                available_options.append(option_index)
            option_index += 1
        return available_options

    def get_intra_state_option_values(self, state, available_options=None):
        state_str = np.array2string(np.ndarray.astype(state, dtype=self.state_dtype))

        try:
            option_values = self.intra_state_option_values[state_str]
        except KeyError:
            if available_options is None:
                available_options = self.get_available_options(state)
            option_values = {option: 0.0 for option in available_options}
            self.intra_state_option_values[state_str] = option_values
        return option_values

    def get_state_option_values(self, state, available_options=None):
        state_str = np.array2string(np.ndarray.astype(state, dtype=self.state_dtype))

        try:
            option_values = {option: self.state_option_values[state_str][option]
                             for option in self.state_option_values[state_str]}
            if option_values == {}:
                raise KeyError
        except KeyError:
            if available_options is None:
                available_options = self.get_available_options(state)
            option_values = {option: 0.0 for option in available_options}
            self.state_option_values[state_str] = option_values
        return option_values

    def intra_option_learning(self, state, action, reward, next_state,
                              terminal=None,
                              next_state_possible_actions=None):
        return

    def learn(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray,
              terminal: bool|None=None, next_state_possible_actions: List[int]|None=None):
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
            try:
                next_state_option_values_list = [next_state_option_values[option] for option in next_available_options]
            except KeyError:
                next_state_option_values_list = [next_state_option_values[str(option)] for option in next_available_options]
        max_next_state_option_value = max(next_state_option_values_list)

        for option_index in available_options:
            option = self.options[option_index]
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

                try:
                    self.state_option_values[state_str][option_index] += self.alpha * (reward -
                                                                                       state_option_values[option_index] +
                                                                                       self.gamma * gamma_product)
                except KeyError:
                    self.state_option_values[state_str][str(option_index)] += self.alpha * (reward -
                                                                                       state_option_values[
                                                                                           str(option_index)] +
                                                                                       self.gamma * gamma_product)

        if not (terminal or self.current_option.terminated(next_state)):
            return

        try:
            option_value = self.get_state_option_values(self.option_start_state)[self.current_option_index]
        except KeyError:
            option_value = self.get_state_option_values(self.option_start_state)[str(self.current_option_index)]
        option_start_state_str = self.state_to_state_str(self.option_start_state)
        try:
            self.state_option_values[option_start_state_str][self.current_option_index] \
                += self.alpha * (self.total_option_reward + (self.gamma ** self.current_option_step) *
                                 max_next_state_option_value
                                 - option_value)
        except KeyError:
            self.state_option_values[option_start_state_str][str(self.current_option_index)] \
                += self.alpha * (self.total_option_reward + (self.gamma ** self.current_option_step) *
                                 max_next_state_option_value
                                 - option_value)
        self.current_option = None
        self.option_start_state = None
        self.current_option_index = None
        self.total_option_reward = 0
        return

    def learn_no_intra(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray,
              terminal: bool|None=None, next_state_possible_actions: List[int]|None=None) -> None:
        # Q(s, o) = Q(s, o) + \alpha(r - Q(s, o) + \gamma * MAXQ(next_state, o_prime))
        if not (terminal or self.current_option.terminated(next_state)):
            return

        next_available_options = []
        if not terminal:
            next_available_options = self.get_available_options(next_state, next_state_possible_actions)

        next_state_option_values_list = [0.0]
        next_state_option_values = self.get_state_option_values(next_state, next_available_options)
        if next_available_options:
            next_state_option_values_list = [next_state_option_values[option] for option in next_available_options]
        max_next_state_option_value = max(next_state_option_values_list)

        option_value = self.get_state_option_values(self.option_start_state)[self.current_option_index]
        option_start_state_str = self.state_to_state_str(self.option_start_state)
        self.state_option_values[option_start_state_str][self.current_option_index] \
            += self.alpha * (self.total_option_reward + (self.gamma ** self.current_option_step) *
                             max_next_state_option_value
                             - option_value)
        self.current_option = None
        self.option_start_state = None
        self.current_option_index = None
        self.total_option_reward = 0
        return

    def save(self, save_path):
        try:
            f = open(save_path, 'x')
            f.close()
        except FileExistsError:
            ()

        data = {'options': {}}
        num_options = len(self.options)
        for i in range(num_options):
            option = self.options[i]
            if not option.has_policy():
                continue
            data['options'][i] = option.policy.q_values

        data['option values'] = {}
        for state in self.state_option_values:
            data['option values'][state] = {self.options.index(option): self.state_option_values[state][option]
                                            for option in self.state_option_values[state]}

        with open(save_path, 'w') as f:
            json.dump(data, f)
        return

    def set_state_option_values(self, values: Dict[int, float], state: np.ndarray) -> None:
        state_str = self.state_to_state_str(state)
        self.state_option_values[state_str] = {str(option): values[option]
                                               for option in values}
        return

    def state_str_to_state(self, state_str):
        if '\n' not in state_str:
            state = np.fromstring(state_str[1: len(state_str) - 1],
                                  sep=' ', dtype=self.state_dtype)
            return state

        state_str = state_str.replace('[', '')
        state_str = state_str.replace(']', '')
        state_str = state_str.replace('\n', '')
        state = np.fromstring(state_str,
                              sep=' ', dtype=self.state_dtype)
        state = state.reshape(self.state_shape)
        return state
    
    def state_to_state_str(self, state: np.ndarray) -> str:
        return np.array2string(np.ndarray.astype(state, dtype=self.state_dtype))

    def terminated(self, state):
        if self.terminating_func is None:
            return True
        return self.terminating_func(state)

