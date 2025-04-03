import copy
import json
import networkx as nx
import numpy as np
import random as rand
import sys
from scipy import sparse
from typing import Callable, Dict, List, Tuple, Type

from environments.environment import Environment
from learning_agents.agentbehaviour import AgentBehaviour
from learning_agents.optionsagent import Option, OptionsAgent
from learning_agents.preparednessagent import PreparednessAgent
from learning_agents.qlearningagent import QLearningAgent
from learning_agents.rodagent import RODAgent
from progressbar import print_progress_bar

class PreparednessIncremental(RODAgent, PreparednessAgent):

    def __init__(
            self,
            actions: List[int],
            skill_training_window: int,
            alpha: float,
            epsilon: float,
            gamma: float,
            state_dtype: Type,
            state_shape: Tuple[int, int],
            max_subgoal_height: int,
            option_onboarding: str,
            option_discovery_method: str
    ):
        assert actions is not None
        assert option_onboarding == 'none' or option_onboarding == 'specific' or option_onboarding == 'generic'
        assert option_discovery_method == "update" or "replace"

        self.actions = actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.state_dtype = state_dtype
        self.state_shape = state_shape

        self.behaviour_mode = AgentBehaviour.EXPLORE
        self.adjacency_matrix = None
        self.preparedness_values = {}
        self.state_transition_graph = nx.MultiDiGraph()
        self.subgoal_graph = nx.MultiDiGraph()

        self.max_subgoal_height = max_subgoal_height
        self.option_onboarding = option_onboarding
        self.option_discovery_method = option_discovery_method

        self.options = []
        self.primitive_options = [Option([action]) for action in self.actions]
        self.specific_onboarding_options = []
        self.generic_onboarding_subgoal_options = []
        self.specific_onboarding_subgoal_options = []
        self.state_node_lookup = {}
        self.path_lookup = {node: {} for node in self.state_transition_graph.nodes()}

        self.current_step = 0
        self.current_option = None
        self.current_option_index = None
        self.option_start_state = None
        self.last_possible_actions = None
        self.total_option_reward = 0
        self.current_option_step = 0
        self.current_skill_training_step = 0
        self.skill_training_window = skill_training_window
        self.state_option_values = {'none': {}, 'generic': {}, 'specific': {}}
        return

    # TODO: Copy Agent
    def copy_agent(
            self,
            copy_from: 'PreparednessIncremental'
    ):
        pass

    def compute_graph_preparedness(self, hop: int):
        return

    def discover_skills(self):
        # REPLACE:
        # Find subgoals
        # Make subgoal graph
        # generate set of skills
        # skip over any skills that already exist

        # UPDATE:
        # Find subgoals
        # add subgoals onto existing subgoal graph
        # generate new set of skills
        # add set of skills to existing set
        pass

    @staticmethod
    def entropy(
            distribution: List[float],
            log_base: float=2
    ) -> float:
        entropy = 0
        for prob in distribution:
            if prob <= 0:
                continue
            entropy -= prob * np.emath.logn(log_base, prob)
        return entropy

    def find_subgoals(self):
        # Look for subgoals up to the max height
        self.adjacency_matrix = nx.to_scipy_sparse_array(self.state_transition_graph)
        pass

    def frequency_entropy(
            self,
            node: str,
            hops: int,
            neighbours: np.ndarray,
            accuracy: int=4
    ) -> float:
        if hops > 0 and neighbours.size == 1:
            try:
                frequency_entropy = self.preparedness_values[node][self.frequency_entropy_key(hops)]
                return frequency_entropy
            except KeyError:
                ()

        # P(S_t+n | s_t)
        def prob(start_node, goal_node, hops_away):
            p = 0
            W_start_node = 0.0
            for j in neighbours:
                W_start_node += self.adjacency_matrix[int(start_node), int(j)]
            if (W_start_node <= 0) and (start_node == goal_node):
                return 1.0
            if hops_away == 1:
                if W_start_node <= 0:
                    return 0
                p = self.adjacency_matrix[int(start_node), int(goal_node)] / W_start_node
                return p

            for j in neighbours:
                w_start_node_j = self.adjacency_matrix[int(start_node), int(j)]
                if w_start_node_j <= 0:
                    continue
                p += (w_start_node_j / W_start_node) * prob(j, goal_node, hops_away - 1)

            return p

        neighbour_probabilities = [prob(node, neighbour, hops) for neighbour in neighbours]
        return round(self.entropy(neighbour_probabilities), accuracy)

    @staticmethod
    def frequency_entropy_key(
            hops: int
    ) -> str:
        return 'frequency_entropy ' + str(hops) + ' hops'

    def neighbourhood_entropy(
            self,
            node: str,
            hops: int,
            neighbours: np.ndarray,
            accuracy: int=4
    ) -> float:

        # W_n_i_j
        def weights_out(start_node, hops_away):
            W = 0.0
            if hops_away == 1:
                for j in neighbours:
                    W += self.adjacency_matrix[int(start_node), int(j)]
                return W

            for j in neighbours:
                w_start_node_j = self.adjacency_matrix[int(start_node), int(j)]
                if w_start_node_j <= 0:
                    continue
                W += (w_start_node_j * weights_out(j, hops_away - 1))
            return W

        T = 0
        for neighbour in neighbours:
            T += weights_out(neighbour, hops)

            # Compute Hops to each neighbourhood
            def weights_to_node(start_node, goal_node, hops_away):
                if hops_away == 1:
                    return self.adjacency_matrix[int(start_node), int(goal_node)]

                P_hat = 0.0
                if hops_away == 2:
                    for j in neighbours:
                        P_hat += (self.adjacency_matrix[int(start_node), int(j)] * self.adjacency_matrix[int(j), int(goal_node)])
                    return P_hat

                for j in neighbours:
                    w_start_node_j = self.adjacency_matrix[int(start_node), int(j)]
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
                    P += weights_to_node(start_neighbour, goal_neighbour, hops)
                neighbour_probabilities.append(P / T)

            # Compute Entropy
            return round(self.entropy(neighbour_probabilities), accuracy)

    @staticmethod
    def neighbourhood_entropy_key(
            hops: int
    ) -> str:
        return "neighbourhood_entropy " + str(hops) + ' hops'

    def policy_learn(
            self,
            state: np.ndarray,
            action: int,
            reward: float,
            next_state: np.ndarray,
            terminal: bool | None = None,
            next_state_possible_actions: List[int] | None = None
    ):
        pass

    def preparedness(self, node: str, hop: int) -> float:
        distances = sparse.csgraph.dijkstra(
            self.adjacency_matrix, directed=True, indicies=int(node), unweighted=True, limit=hop+1
        )
        neighbours = np.where((0 < distances) & (distances <= hop))[0]

        frequency_entropy = self.frequency_entropy(
            node,
            hop,
            neighbours
        )
        neighbourhood_entropy = self.neighbourhood_entropy(
            node,
            hop,
            neighbours
        )
        node_preparedness = frequency_entropy + neighbourhood_entropy

        try:
            existing_values = self.preparedness_values[node]
        except KeyError:
            existing_values = {}
        existing_values[self.frequency_entropy_key(hop)] = frequency_entropy
        existing_values[self.neighbourhood_entropy_key(hop)] = neighbourhood_entropy
        existing_values[self.preparedness_key(hop)] = node_preparedness

        return node_preparedness

    @staticmethod
    def preparedness_key(
            hops: int
    ) -> str:
        return "preparedness " + str(hops) + ' hops'

    def save_representation(
            self,
            save_path: str
    ):
        pass

    def train_skill(
            self,
            skill: Option,
            state: np.ndarray,
            action: int,
            reward: float,
            next_state: np.ndarray,
            terminal: bool | None = None
    ):
        pass

    def update_representation(
            self,
            state: np.ndarray,
            action: int,
            reward: float,
            next_state: np.ndarray,
            terminal: bool | None = None
    ):
        pass
