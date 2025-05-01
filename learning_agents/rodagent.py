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

class RODAgent(OptionsAgent):

    def __init__(
            self,
            actions: List[int],
            options: List[Option],
            skill_training_window: int,
            alpha: float,
            epsilon: float,
            gamma: float,
            state_dtype: Type
        ):

        self.actions = actions
        self.options = options
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        self.environment_start_states = []

        self.current_skill = None
        self.current_option_index = None
        self.option_start_state = None
        self.total_option_reward = 0
        self.current_option_step = 0
        self.current_skill_training_step = 0
        self.skill_training_window = skill_training_window
        self.skill_training_reward_sum = 0

        self.state_option_values = {}
        self.intra_state_option_values = {}

        self.state_dtype = state_dtype

        self.behaviour = AgentBehaviour.EXPLORE

        self.training_skill = None
        return


    def add_start_state(
            self,
            state: np.ndarray,
    ):
        if state not in self.environment_start_states:
            self.environment_start_states.append(state)
        return

    def choose_action(self,
            state: np.ndarray,
            optimal_choice: bool = False,
            possible_actions: None | List[int] = None
        ) -> int:

        if self.behaviour == AgentBehaviour.EXPLORE:
            return rand.choice(self.actions)

        if self.behaviour == AgentBehaviour.TRAIN_SKILLS:
            if self.training_skill is None:
                self.training_skill = self.choose_training_skill(state)

            return self.training_skill.choose_action(state, possible_actions)

        return super(self).choose_action(state, optimal_choice, possible_actions)

    def choose_training_skill(self, state: np.ndarray) -> Option:
        possible_skills = []
        for skill in self.options:
            if skill.has_policy and skill.initiated(state):
                possible_skills.append(skill)
        return rand.choice(possible_skills)

    def discover_skills(self):
        return

    def learn(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray,
              terminal: bool | None = None, next_state_possible_actions: List[int] | None = None):
        if self.behaviour == AgentBehaviour.EXPLORE:
            self.update_representation(state, action, reward, next_state, terminal)
            return

        if self.behaviour == AgentBehaviour.TRAIN_SKILLS:
            self.current_skill_training_step += 1
            if self.current_skill is not None:
                self.current_skill(
                    self.training_skill,
                    state,
                    action,
                    reward,
                    next_state,
                    terminal,
                    next_state_possible_actions
                )
            if (self.current_skill is None) or (self.current_skill_training_step == self.skill_training_window):
                self.skill_training_reward_sum = 0.0
                self.choose_training_skill(state)
                self.current_skill_training_step = 0
            return

        self.policy_learn(state, action, reward, next_state, terminal, next_state_possible_actions)
        return

    def load(
            self,
            save_path: str
    ):
        return

    def policy_learn(
            self,
            state: np.ndarray,
            action: int,
            reward: float,
            next_state: np.ndarray,
            terminal: bool | None = None,
            next_state_possible_actions: List[int] | None = None
    ):
        return

    def save_representation(
            self,
            save_path: str
    ):
        return

    def set_behaviour(self, behaviour: AgentBehaviour):
        if self.behaviour == AgentBehaviour.EXPLORE:
            self.discover_skills()
        elif self.behaviour == AgentBehaviour.TRAIN_SKILLS:
            self.training_skill = None

        self.behaviour = behaviour
        return

    def set_behaviour_explore(self):
        return self.set_behaviour(AgentBehaviour.EXPLORE)

    def set_behaviour_learn(self):
        return self.set_behaviour(AgentBehaviour.LEARN)

    def set_behaviour_train_skills(self):
        return self.set_behaviour(AgentBehaviour.TRAIN_SKILLS)

    def train_skill(
            self,
            skill: Option,
            state: np.ndarray,
            action: int,
            reward: float,
            next_state: np.ndarray,
            terminal: bool | None = None,
            next_state_possible_actions: List[int]|None = None
    ):
        return

    def update_representation(
            self,
            state: np.ndarray,
            action: int,
            reward: float,
            next_state: np.ndarray,
            terminal: bool | None = None
    ):
        return
