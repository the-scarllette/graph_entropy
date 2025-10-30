import json
import networkx as nx
import numpy as np
import random as rand
from scipy import sparse
from typing import Dict, List, Tuple, Type

from learning_agents.agentbehaviour import AgentBehaviour
from learning_agents.optionsagent import Option, OptionsAgent

class PreparednessIncremental(OptionsAgent):

    skill_training_failure_reward: float = -1.0
    skill_training_step_reward: float = -0.01
    skill_training_success_reward: float = 1.0

    def __init__(
            self,
            actions: List[int],
            alpha: float,
            epsilon: float,
            gamma: float,
            state_dtype: Type,
            state_shape: Tuple[int, ...],
            graph_save_paths_prefix: str,
    ):
        self.actions: List[int] = actions
        self.alpha: float = alpha
        self.epsilon: float = epsilon
        self.gamma: float = gamma
        self.state_dtype: Type = state_dtype
        self.state_shape: Tuple[int, ...] = state_shape
        self.graph_save_paths_prefix: str = graph_save_paths_prefix

        self.state_transition_graph: nx.DiGraph = nx.DiGraph()
        self.num_nodes: int = 0
        self.stg_values: Dict[str, Dict[str, str|float]] = {}
        self.subgoal_graph: nx.DiGraph = nx.DiGraph()
        self.state_node_lookup: Dict[str, str] = {}
        self.node_state_lookup: Dict[str, str] = {}
        # node -> next_node -> num observations
        self.total_transitions: Dict[str, Dict[str, int]] = {}

        self.state_option_values: Dict[str, Dict[str, float]] = {}

        self.behaviour: AgentBehaviour = AgentBehaviour.EXPLORE
        return

    def choose_action(
            self,
            state: np.ndarray,
            optimal_choice: bool=False,
            possible_actions: None | List[int] = None
    ) -> int:
        if possible_actions is None:
            possible_actions = self.actions

        if self.behaviour == AgentBehaviour.EXPLORE:
            action = rand.choice(possible_actions)
            return action
        pass

    def copy_agent(
            self,
            copy_from: 'PreparednessIncremental'
    ):
        pass

    def learn(
            self,
            state: np.ndarray,
            action: int,
            reward: float,
            next_state: np.ndarray,
            terminal: bool|None = None,
            next_state_possible_actions: List[int]|None = None
    ):
        if self.behaviour == AgentBehaviour.EXPLORE:
            self.update_state_transition_graph(state, next_state)
            return
        pass

    def load(
            self,
            save_path: str
    ):
        with open(save_path, 'r') as f:
            agent_data = json.load(f)

        self.num_nodes = agent_data['num_nodes']
        self.stg_values = agent_data['stg_values']
        self.state_node_lookup = agent_data['state_node_lookup']
        self.node_state_lookup = agent_data['node_state_lookup']
        self.total_transitions = agent_data['total_transitions']

        self.state_transition_graph = nx.read_gexf(agent_data['state_transition_graph_save_path'])
        self.subgoal_graph = nx.read_gexf(agent_data['subgoal_graph_save_path'])
        nx.set_node_attributes(self.state_transition_graph, self.stg_values)
        nx.set_node_attributes(self.subgoal_graph, self.stg_values)
        return

    def save(
            self,
            save_path: str,
    ):
        stg_save_path: str = self.graph_save_paths_prefix + "_state_transition_graph.gexf"
        subgoal_graph_save_path: str = self.graph_save_paths_prefix + "_subgoal_graph.gexf"

        agent_save_file = {
            'state_transition_graph_save_path': stg_save_path,
            'num_nodes': self.num_nodes,
            'stg_values': self.stg_values,
            'subgoal_graph_save_path': subgoal_graph_save_path,
            'state_node_lookup': self.state_node_lookup,
            'node_state_lookup': self.node_state_lookup,
            'total_transitions': self.total_transitions
        }

        with open(save_path, 'w') as f:
            json.dump(agent_save_file, f)

        nx.write_gexf(stg_save_path, self.state_transition_graph)
        nx.write_gexf(subgoal_graph_save_path, self.subgoal_graph)
        return

    def state_to_node(
            self,
            state: np.ndarray
    ) -> str:
        state_str = self.state_to_state_str(state)
        try:
            state_node = self.state_node_lookup[state_str]
        except KeyError:
            self.state_node_lookup[state_str] = str(self.num_nodes)
            self.num_nodes += 1
        return state_node

    def update_state_transition_graph(
            self,
            state: np.ndarray,
            next_state: np.ndarray
    ):
        state_str: str = self.state_to_state_str(state)
        next_state_str: str = self.state_to_str(next_state)
        state_node: str = self.state_to_node(state)
        next_state_node: str = self.state_to_node(next_state)

        if not self.state_transition_graph.has_node(state_node):
            self.state_transition_graph.add_node(state_node)
            self.total_transitions[state_node] = {}
            self.stg_values[state_node] = {}

        if not self.state_transition_graph.has_node(next_state_node):
            self.state_transition_graph.add_node(next_state_node)
            self.stg_values[next_state_node] = {}

        if not self.state_transition_graph.has_edge(state_node, next_state_node):
            self.state_transition_graph.add_edge(state_node, next_state_node)
            self.total_transitions[state_str][next_state_str] = 1
        else:
            self.total_transitions[state_str][next_state_str] += 1

        return
