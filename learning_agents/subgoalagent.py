import copy
import json
import random as rand
import typing
import networkx as nx
import numpy as np
from scipy import sparse
from typing import Callable, List, Tuple

from environments.environment import Environment
from learning_agents.optionsagent import Option, OptionsAgent
from learning_agents.qlearningagent import QLearningAgent
from progressbar import print_progress_bar


class SubgoalOption(Option):

    def __init__(self, actions: List[int], subgoal: str, path_function: Callable[[np.ndarray, str], bool]) -> None:
        self.actions = None
        self.policy = QLearningAgent(actions, self.alpha, self.epsilon, self.gamma)
        self.subgoal = subgoal
        self.initiation_func = lambda state: path_function(state, self.subgoal)
        self.terminating_func = lambda state: not path_function(state, self.subgoal)
        return


class SubgoalAgent(OptionsAgent):

    def __init__(self, actions: List[int], alpha: float, epsilon: float, gamma: float,
                 state_shape: Tuple[int, int],
                 state_transition_graph: nx.MultiDiGraph,
                 subgoals: List[str],
                 subgoal_distance: int=30) -> None:
        self.actions = actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.state_shape = state_shape
        self.state_transition_graph = state_transition_graph
        self.subgoals = subgoals
        self.subgoal_distance = subgoal_distance

        self.options = [SubgoalOption(self.actions, subgoal, self.option_initiation) for subgoal in self.subgoals]

        self.state_node_lookup = {}
        self.path_lookup = {node: {} for node in self.state_transition_graph.nodes()}

        self.current_option = None
        self.current_option_index = None
        self.option_start_state = None
        self.las_possible_actions = None
        self.total_option_reward = 0
        self.current_option_step = 0
        self.state_option_values = {}
        return

    def copy_agent(self, copy_from: 'SubgoalAgent') -> None:
        self.options = copy.deepcopy(self.options)
        self.state_node_lookup = copy.deepycopy(self.state_node_lookup)
        self.path_lookup = copy.deepcopy(self.path_lookup)
        self.state_option_values = copy.deepcopy(self.state_option_values)
        return

    def get_state_node(self, state: np.ndarray) -> str:
        state_str = self.state_to_state_str(state)
        try:
            node = self.state_node_lookup[state_str]
        except KeyError:
            for node, values in self.state_transition_graph.nodes(data=True):
                if state_str == values['state']:
                    break
            self.state_node_lookup[state_str] = node
        return node

    def load(self, save_path: str) -> None:
        with open(save_path, 'r') as f:
            agent_save_file = json.load(f)

        self.options = []
        self.subgoals = []
        for subgoal, option_dict in agent_save_file['options']:
            option = SubgoalOption(self.actions, subgoal, self.option_initiation)
            option.policy.q_values = option_dict['q_values']
            self.options.append(option)

        self.state_node_lookup = agent_save_file['state_node_lookup']
        self.path_lookup = agent_save_file['path_lookup']
        self.state_option_values = agent_save_file['state_option_values']
        return

    def option_initiation(self, state: np.ndarray, subgoal: str) -> bool:
        state_str = self.state_to_state_str(state)

        try:
            option_initiated_str = self.path_lookup[subgoal][state_str]
            option_initiated = option_initiated_str == 'True'
        except KeyError:
            state_node = self.get_state_node(state)
            if state_node == subgoal:
                option_initiated = False
            else:
                path_distance = nx.shortest_path_length(self.state_transition_graph, state_node, subgoal)
                option_initiated = path_distance <= self.subgoal_distance
            option_initiated_str = 'True' if option_initiated else 'False'
            self.path_lookup[subgoal][state_str] = option_initiated_str

        return option_initiated

    def save(self, save_path: str) -> None:
        agent_save_file = {'options': {option.subgoal: {'policy': option.policy.q_values}
                                       for option in self.options},
                           'state_node_lookup': self.state_node_lookup,
                           'path_lookup': self.path_lookup,
                           'state_option_values': self.state_option_values}

        with open(save_path, 'w') as f:
            json.dump(agent_save_file, f)
        return
