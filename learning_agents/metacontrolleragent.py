
from learning_agents.learningagent import LearningAgent

import copy
import numpy as np
import random


class MetaControllerAgent(LearningAgent):

    def __init__(self, actions):
        self.actions = actions
        self.current_skill = None
        self.current_skill_length = 0
        self.meta_controller = None
        self.policy = None
        self.skill_length = None
        return

    def choose_action(self, state, optimal_choice=False,
                      possible_actions=None) -> int:
        if self.current_skill is None:
            self.choose_skill(state)

        skill_state_vector = self.create_skill_state_vector(self.current_skill, state)

        action = self.policy.choose_action(skill_state_vector, optimal_choice, possible_actions)

        return action

    def choose_skill(self, state, optimal_choice=False):
        self.current_skill = self.meta_controller.choose_action(state, optimal_choice)
        return

    def copy_agent(self, copy_from):
        self = copy.deepcopy(copy_from)
        return

    def create_skill_state_vector(self, skill, state):
        skill_vector = np.zeros(self.num_skills)
        skill_vector[skill] = 1.0
        skill_state_vector = np.append(skill_vector, state)
        return skill_state_vector

    def learn(self, state, action, reward, next_state, terminal=None,
              next_state_possible_actions=None, no_learning=False):
        self.meta_controller.learn(state, self.current_skill, reward, next_state, terminal, no_learning)

        self.current_skill_length += 1
        if self.current_skill_length >= self.skill_length or terminal:
            self.current_skill = None
            self.current_skill_length = 0
        return

    def sample_skill(self):
        self.current_skill = random.randint(0, self.num_skills - 1)
        return
