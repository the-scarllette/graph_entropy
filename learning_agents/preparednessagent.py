import networkx as nx
import numpy as np
from typing import Callable, Dict, List, Type

from optionsagent import Option
from learningagent import LearningAgent


class PreparednessOption(Option):

    def __init__(self, actions: List[Option] | List[int], start_node: None | str, end_node: None | str,
                 start_state_str: str, end_state_str: str,
                 initiation_func: Callable[[np.ndarry], bool]):
        self.actions = actions
        self.start_node = start_node
        self.start_state_str = start_state_str
        self.end_node = end_node
        self.end_state_str = end_state_str
        self.initiation_func = initiation_func
        return

    # TODO: Create Choose Action Function
    def choose_action(self, state: np.ndarray, possible_actions: List[int]) -> int:
        return -1

    def initiated(self, state: np.ndarray) -> bool:
        if self.start_node is not None:
            return self.start_state_str == np.array2string(state)
        return self.initiation_func(state)

    def terminated(self, state: np.ndarray) -> bool:
        if self.end_state_str == np.array2string(state):
            return True
        return not self.initiation_func(state)


class PreparednessAgent(LearningAgent):

    def __init__(self, actions: List[int], alpha: float, epsilon: float, gamma: float, state_dtype: Type,
                 state_transition_graph: nx.MultiDiGraph,
                 aggregate_graph: nx.MultiDiGraph):
        self.actions = actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.state_dtype = state_dtype
        self.state_transition_graph = state_transition_graph
        self.aggregate_graph = aggregate_graph
        return

    def choose_action(self, state, optimal_choice=False, possible_actions=None):
        return

    def copy_agent(self, copy_from):
        return

    def create_option(self, start_node: None | str, end_node: str, hierarchy_level: int):
        option = Option()
        return

    def create_options(self):
        # First Level: all edges in aggregate graph (paths from one subgoal to another)
        for start_node, end_node in self.state_transition_graph.edges:
            self.create_option(start_node, end_node, 1)


        return

    def learn(self, state, action, reward, next_state,
              terminal=None, next_state_possible_actions=None):
        return

    def save(self, save_path):
