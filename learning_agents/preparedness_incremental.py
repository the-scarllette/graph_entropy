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
from learning_agents.qlearningagent import QLearningAgent
from progressbar import print_progress_bar

class PreparednessIncremental(OptionsAgent):

    def __init__(
            self,
            actions: List[int],
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
        self.state_option_values = {'none': {}, 'generic': {}, 'specific': {}}
        return

    def choose_action(self,
        state: np.ndarray,
        optimal_choice: bool=False,
        possible_actions: None|List[int]=None
    ) -> int:

        if possible_actions is None:
            possible_actions = self.actions

        if self.behaviour_mode == Behaviour.EXPLORE:
            return rand.choice(possible_actions)



        return None

    def set_explore_behaviour(self):

    def learn(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray,
              terminal: bool|None=None, next_state_possible_actions: List[int]|None=None):
        # Update state transition graph

        # If count % learn_skills_epoch == 0:
        #   find new preparedness values
        #   if REPLACE SKILLS:
        #       Find new subgoals
        #       build new subgoal graph
        #       Find new skills
        #       Generate new skill hierarchy
        #   if UPDATE SKILLS:
        #       find new subgoals
        #       add new subgoals to subgoal graph
        #       remove non-existing skills based on subgoal graph
        #       add new skills based on subgoal graph
        #       re-define skills based on subgoal graph

        #
        return
