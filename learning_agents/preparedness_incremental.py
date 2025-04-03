import copy
import json
import networkx as nx
import numpy as np
import random as rand
import sys
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
            option_onboarding: str
    ):
        assert actions is not None
        assert option_onboarding == 'none' or option_onboarding == 'specific' or option_onboarding == 'generic'

        self.actions = actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.state_dtype = state_dtype
        self.state_shape = state_shape

        self.behaviour_mode = AgentBehaviour.EXPLORE
        self.state_transition_graph = nx.MultiDiGraph()
        self.subgoal_graph = nx.MultiDiGraph()

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

    def choose_action(self,
        state: np.ndarray,
        optimal_choice: bool=False,
        possible_actions: None|List[int]=None
    ) -> int:
        pass

    def choose_option(
            self,
            state: np.ndarray,
            no_random: bool,
            possible_actions: None|List[int]=None
    ):
        pass

    def copy_agent(
            self,
            copy_from: 'PreparednessIncremental'
    ):
        pass

    def count_skills(self) -> Dict[int, int]:
        pass

    def discover_skills(self):
        pass

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
