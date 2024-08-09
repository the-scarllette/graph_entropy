
import numpy as np


class LearningAgent:

    def __init__(self, actions):
        self.actions = actions
        return

    def choose_action(self, state, optimal_choice=False,
                      possible_actions=None) -> int:
        return None

    def learn(self, state, action, reward, next_state, terminal=None,
              next_state_possible_actions=None,
              no_learning=False):
        return


