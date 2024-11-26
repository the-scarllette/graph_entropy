import json
import random as rand
import typing
import networkx as nx
import numpy as np
from scipy import sparse
from typing import List, Tuple

from environments.environment import Environment
from learning_agents.optionsagent import Option, OptionsAgent
from learning_agents.qlearningagent import QLearningAgent
from progressbar import print_progress_bar


class SubgoalOption(Option):

    def __init__(self, actions: List[int], subgoal: str, initiation_set: List[str]) -> None:
        self.actions = None
        self.policy = QLearningAgent(actions, self.alpha, self.epsilon, self.gamma)
        self.subgoal = subgoal
        self.initiation_set = initiation_set.copy()
        self.initiation_func = lambda s: s in self.initiation_set
        self.terminating_func = lambda s: (s == self.subgoal) or (s not in self.initiation_set)
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

        self.options = [self.create_option(subgoal) for subgoal in self.subgoals]

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

    # TODO: Create Copy Agent Method
    def copy_agent(self, copy_from: 'SubgoalAgent') -> None:
        return

    def create_option(self, subgoal: str) -> SubgoalOption:
        path_lengths = nx.shortest_path_length(self.state_transition_graph, target=subgoal)
        initiation_set = [node for node, distance in path_lengths
                       if distance <= self.subgoal_distance and node != subgoal]
        option = SubgoalOption(self.actions, subgoal, initiation_set)
        return option

    # TODO: Create get_available_options method
    def get_available_options(self, state: np.ndarray, possible_actions: None|List[int]=None) -> List[str]:
        return

    # TODO: Create load method
    def load(self, save_path: str) -> None:
        return

    # TODO: Create save method
    def save(self, save_path: str) -> None:
        return
