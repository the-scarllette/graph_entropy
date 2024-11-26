import networkx as nx
from typing import Dict, List, Tuple, Type

from learning_agents.subgoalagent import SubgoalAgent

class BetweennessAgent(SubgoalAgent):

    def __init__(self, actions: List[int], alpha: float, epsilon: float, gamma: float,
                 state_shape: Tuple[int, int],
                 state_dtype: Type,
                 state_transition_graph: nx.MultiDiGraph,
                 subgoal_distance: int=30) -> None:
        self.actions = actions
        self.num_actions = len(actions)
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.state_shape = state_shape
        self.state_dtype = state_dtype
        self.state_transition_graph = state_transition_graph
        self.subgoals = []
        self.subgoal_distance = subgoal_distance

        self.options = []

        self.state_node_lookup = {}
        self.option_initiation_lookup = {}

        self.current_option = None
        self.current_option_index = None
        self.option_start_state = None
        self.las_possible_actions = None
        self.total_option_reward = 0
        self.current_option_step = 0
        self.state_option_values = {}
        return

    def find_betweenness_subgoals(self, state_transition_graph_values: Dict[str, Dict[str, str]]|None=None) -> Dict[str, Dict[str, str]]|None:
        betweenness_values = nx.betweenness_centrality(self.state_transition_graph)
        self.subgoals = []

        for node in betweenness_values.keys():
            betweenness = betweenness_values[node]
            is_subgoal = True
            for neighbour in nx.all_neighbors(self.state_transition_graph, node):
                if neighbour == node:
                    continue
                neighbour_betweenness = betweenness_values[neighbour]
                if neighbour_betweenness >= betweenness:
                    is_subgoal = False
                    break

            if is_subgoal:
                self.subgoals.append(node)
            if state_transition_graph_values is not None:
                state_transition_graph_values[node]['node betweenness'] = betweenness_values[node]
                state_transition_graph_values[node]['node betweenness subgoal'] = str(is_subgoal)

        if state_transition_graph_values is None:
            return

        return state_transition_graph_values
