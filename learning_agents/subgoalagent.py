import copy
import json
import random as rand
import typing
import networkx as nx
import numpy as np
from scipy import sparse
from typing import Callable, List, Tuple, Type

from environments.environment import Environment
from learning_agents.optionsagent import Option, OptionsAgent
from learning_agents.qlearningagent import QLearningAgent
from progressbar import print_progress_bar


class SubgoalOption(Option):

    def __init__(self, actions: List[int], alpha, epsilon, gamma, subgoal: str, initiation_set: List[np.ndarray]) -> None:
        self.actions = None
        self.policy = QLearningAgent(actions, alpha, epsilon, gamma)
        self.subgoal = subgoal
        self.initiation_set = initiation_set
        return

    def initiated(self, state: np.ndarray) -> bool:
        for initiation_state in self.initiation_set:
            if np.array_equal(state, initiation_state):
                return True
        return False

    def terminated(self, state: np.ndarray) -> bool:
        return not self.initiated(state)

class SubgoalAgent(OptionsAgent):

    option_failure_reward = -1.0
    option_step_reward = -0.001
    option_success_reward = 1.0

    def __init__(self, actions: List[int], alpha: float, epsilon: float, gamma: float,
                 state_shape: Tuple[int, int],
                 state_dtype: Type,
                 state_transition_graph: nx.MultiDiGraph,
                 subgoals: List[str],
                 subgoal_distance: int=30) -> None:
        self.actions = actions
        self.num_actions = len(actions)
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.state_shape = state_shape
        self.state_dtype = state_dtype
        self.state_transition_graph = state_transition_graph
        self.subgoals = subgoals
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
        self.current_step = 0
        self.intra_option = False
        return

    def copy_agent(self, copy_from: 'SubgoalAgent') -> None:
        self.options = copy.deepcopy(copy_from.options)
        self.state_node_lookup = copy.deepcopy(copy_from.state_node_lookup)
        self.option_initiation_lookup = copy.deepcopy(copy_from.option_initiation_lookup)
        self.state_option_values = copy.deepcopy(copy_from.state_option_values)
        return

    def create_options(self) -> None:
        self.options = []
        for action in self.actions:
            option = Option([action])
            self.options.append(option)

        for subgoal in self.subgoals:
            distances = nx.shortest_path_length(self.state_transition_graph, target=subgoal)
            initiation_set = [self.node_to_state(node) for node in list(distances.keys())
                                if (node != subgoal) and (distances[node] <= self.subgoal_distance)]
            option = SubgoalOption(self.actions, self.alpha, self.epsilon, self.gamma, subgoal, initiation_set)
            self.options.append(option)
        return

    # TODO FIX FOR ONLY SOME POSSIBLE ACTIONS
    def get_available_options(self, state: np.ndarray, possible_actions: None|List[int]=None) -> List[int]:
        state_str = self.state_to_state_str(state)

        try:
            available_options = self.option_initiation_lookup[state_str]
        except KeyError:
            option_index = 0
            available_options = []

            for option in self.options:
                if option.actions is not None:
                    if possible_actions is None or option.actions[0] in possible_actions:
                        available_options.append(option_index)
                elif option.initiated(state):
                    available_options.append(option_index)
                option_index += 1

            self.option_initiation_lookup[state_str] = available_options

        return available_options

    def get_state_node(self, state: np.ndarray) -> str:
        state_str = self.state_to_state_str(state)
        try:
            node = self.state_node_lookup[state_str]
        except KeyError:
            for node, values in self.state_transition_graph.nodes(data=True):
                if state_str == values['state']:
                    break
            self.state_node_lookup[state_str] = node
        return node

    def load(self, save_path: str) -> None:
        with open(save_path, 'r') as f:
            agent_save_file = json.load(f)

        self.options = []
        self.subgoals = []
        for action in self.actions:
            option = Option([action])
            self.options.append(option)
        for subgoal in list(agent_save_file['options'].keys()):
            option_dict = agent_save_file['options'][subgoal]
            option = SubgoalOption(self.actions, self.alpha, self.epsilon, self.gamma, subgoal, [self.state_str_to_state(state_str)
                                                           for state_str in option_dict['initiation_set']])
            option.policy.q_values = option_dict['policy']
            self.options.append(option)

        self.state_node_lookup = agent_save_file['state_node_lookup']
        self.option_initiation_lookup = agent_save_file['option_initiation_lookup']
        self.state_option_values = agent_save_file['state_option_values']
        return

    def node_to_state(self, node: str) -> np.ndarray:
        state_str = self.state_transition_graph.nodes(data=True)[node]['state']
        return self.state_str_to_state(state_str)

    def save(self, save_path: str) -> None:
        agent_save_file = {'options': {self.options[i].subgoal: {'initiation_set': [self.state_to_state_str(state)
                                                                           for state in self.options[i].initiation_set],
                                                        'policy': self.options[i].policy.q_values}
                                       for i in range(self.num_actions, len(self.options))},
                           'state_node_lookup': self.state_node_lookup,
                           'option_initiation_lookup': self.option_initiation_lookup,
                           'state_option_values': self.state_option_values}

        with open(save_path, 'w') as f:
            json.dump(agent_save_file, f)
        return

    def train_option(self, option: Option, environment: Environment,
                     training_timesteps: int,
                     all_actions_possible: bool=False,
                     progress_bar: bool=False) -> Tuple[int, int]:

        # Getting Start States
        terminated = True
        possible_actions = self.actions
        total_end_states = 0
        total_successes = 0

        for current_timesteps in range(training_timesteps):
            if progress_bar:
                print_progress_bar(current_timesteps, training_timesteps,
                                   '        >')

            if terminated:
                state = rand.choice(option.initiation_set)
                state = environment.reset(state)
                if not all_actions_possible:
                    possible_actions = environment.get_possible_actions(state)

            action = option.choose_action(state, possible_actions)

            next_state, _, terminated, _ = environment.step(action)
            next_state_node = self.get_state_node(next_state)

            reward = self.option_step_reward
            if next_state_node == option.subgoal:
                terminated = True
                reward = self.option_success_reward
                total_successes += 1
            elif terminated or option.terminated(next_state):
                terminated = True
                reward = self.option_failure_reward

            if terminated:
                total_end_states += 1

            if not all_actions_possible:
                possible_actions = environment.get_possible_actions(next_state)

            option.policy.learn(state, action, reward, next_state, terminated, possible_actions)

            state = next_state

        return total_end_states, total_successes

    def train_options(self, environment: Environment,
                      training_timesteps: int,
                      all_actions_possible: bool,
                      progress_bar: bool=False) -> None:
        def percentage(x: int, y: int) -> float:
            if y <= 0:
                return -1.0
            return round((x/y) * 100, 1)

        if progress_bar:
            print("Training Subgoal Options")

        for i in range(self.num_actions, len(self.options)):
            option = self.options[i]
            if progress_bar:
                print("     Option -> " + option.subgoal)
            total_end_states, total_successes = self.train_option(option, environment, training_timesteps,
                                                                  all_actions_possible,
                                                                  progress_bar)
            if progress_bar:
                percentage_hits = percentage(total_successes, total_end_states)
                print("     Option -> " + option.subgoal + " " + str(percentage_hits) + "% hits")
        return
