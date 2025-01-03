import random

from environments.environment import Environment
from progressbar import print_progress_bar
from learning_agents.optionsagent import Option, OptionsAgent
from learning_agents.qlearningagent import QLearningAgent

import json
import networkx as nx
import numpy as np


class MultiLevelGoalAgent(OptionsAgent):

    def __init__(self, primitive_actions, alpha, epsilon, gamma, goals, stg, compressed_stg=False, state_dtype=int):
        self.primitive_actions = primitive_actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.state_dtype = state_dtype
        self.stg = stg
        self.compressed_stg = compressed_stg
        self.current_step = 0

        self.current_option = None
        self.option_start_state = None
        self.total_option_reward = 0
        self.current_option_step = 0

        self.state_option_values = {}
        self.intra_state_option_values = {}

        self.last_possible_actions = None
        self.intra_option = True

        self.goals = goals
        self.num_option_levels = len(self.goals)
        self.goal_indexes = [[self.get_state_index(goal) for goal in self.goals[i]]
                             for i in range(self.num_option_levels)]
        self.num_goals_per_level = [len(self.goals[i]) for i in range(self.num_option_levels)]

        # Define Options
        self.primitive_options = [Option([action]) for action in primitive_actions]
        self.options = []
        self.options_between_goals = []

        # First level options: all states to first level goals
        first_level_options = []
        for i in range(self.num_goals_per_level[0]):
            goal = self.goals[0][i]
            goal_index = self.goal_indexes[0][i]

            first_level_options.append({"start_state": None,
                                        "start_state_index": None,
                                        "goal": goal,
                                        "goal_index": goal_index,
                                        "option": self.create_option(goal, goal_index)})
        self.options.append(first_level_options)

        # Options between a goal and the goals of the next level
        for level in range(self.num_option_levels - 1):
            options_between_goals = []
            for i in range(self.num_goals_per_level[level]):
                start_state = self.goals[level][i]
                start_state_index = self.goal_indexes[level][i]
                for j in range(self.num_goals_per_level[level + 1]):
                    goal_index = self.goal_indexes[level + 1][j]
                    goal = self.goals[level + 1][j]

                    if not nx.has_path(self.stg, start_state_index, goal_index):
                        continue

                    options_between_goals.append({"start_state": start_state,
                                                  "start_state_index": start_state_index,
                                                  "goal": goal,
                                                  "goal_index": goal_index,
                                                  "option": self.create_option(goal, goal_index, start_state)})

            self.options_between_goals.append(options_between_goals.copy())

        # Multi-level-options: from all states to a goal
        for level in range(1, self.num_option_levels):
            options = []
            for i in range(self.num_goals_per_level[level]):
                goal_state = self.goals[level][i]
                goal_index = self.goal_indexes[level][i]

                # Options available are:
                # Options from lower_level goal states, to this goal state
                # if this level does not work, search in a previous level
                level_to_search = level - 1
                level_found = False
                while (level_to_search >= 0) and (not level_found):
                    options_to_subgoal = []
                    options_from_subgoal = []

                    for option in self.options_between_goals[level_to_search]:
                        if option['goal'] == goal_state:
                            # Options from all states to the lower level goal states that has a path to this goal state
                            for option_to_goal in self.options[level_to_search]:
                                if option['start_state'] == option_to_goal['goal']:
                                    options_to_subgoal.append(option_to_goal['option'])
                                    options_from_subgoal.append(option['option'])
                                    level_found = True
                                    break
                    level_to_search -= 1

                if not level_found:
                    self.options[0].append({"start_state": None,
                                            "start_state_index": None,
                                            "goal": goal_state,
                                            "goal_index": goal_index,
                                            "option": self.create_option(goal_state, goal_index)})
                    continue

                options_available = options_from_subgoal + options_to_subgoal

                options.append({"start_state": None,
                                "start_state_index": None,
                                "goal": goal_state,
                                "goal_index": goal_index,
                                "option": self.create_option(goal_state, goal_index, options=options_available.copy())})
            self.options.append(options.copy())
        return

    def choose_action(self, state, optimal_choice=False, possible_actions=None):
        if self.current_option is None:
            self.current_option = self.choose_option(state, optimal_choice, possible_actions)

        if not self.current_option.has_policy():
            return self.current_option.actions[0]
        else:
            try:
                if self.current_option.policy.current_option is not None:
                    if self.current_option.policy.current_option.terminated(state):
                        self.current_option.policy.current_option = None
            except AttributeError:
                ()

        chosen_action = self.current_option.choose_action(state, possible_actions)
        if chosen_action == -1:
            self.current_option = None
            return self.choose_action(state, optimal_choice)

        return chosen_action

    def create_option(self, goal_state, goal_index, start_state=None, options=None):

        def create_initiation_func():
            if start_state is None:
                def initiation_func(state):
                    s_str = np.array2string(np.ndarray.astype(state, dtype=self.state_dtype))
                    if s_str == goal_state:
                        return False
                    state_index = self.get_state_index(s_str)
                    return nx.has_path(self.stg, state_index, goal_index)

                return initiation_func

            def initiation_func(state):
                return np.array2string(np.ndarray.astype(state, dtype=self.state_dtype)) == start_state

            return initiation_func

        def termination_func(state):
            s_str = np.array2string(np.ndarray.astype(state, dtype=self.state_dtype))
            if goal_state == s_str:
                return True
            state_index = self.get_state_index(s_str)
            return not nx.has_path(self.stg, state_index, goal_index)

        if options is not None:
            policy = OptionsAgent(self.alpha, self.epsilon, self.gamma,
                                  options, intra_option=False)
        else:
            policy = QLearningAgent(self.primitive_actions, self.alpha, self.epsilon, self.gamma)

        option = Option(policy=policy, initiation_func=create_initiation_func(), terminating_func=termination_func)
        return option

    def get_available_options(self, state, possible_actions=None):
        available_options = []

        if possible_actions is None:
            available_options = [primitive_option for primitive_option in self.primitive_options]
        else:
            for primitive_option in self.primitive_options:
                action = primitive_option.actions[0]
                if action in possible_actions:
                    available_options.append(primitive_option)

        for level in range(self.num_option_levels):
            for option_dict in self.options[level]:
                option = option_dict['option']
                if option.initiation_func(state):
                    available_options.append(option)

        return available_options

    def get_state_index(self, state):
        for node in self.stg.nodes:
            if self.stg.nodes[node]['state'] == state:
                return node
        return None

    def learn(self, state, action, reward, next_state,
              terminal=None, next_state_possible_actions=None):
        # Q(s, o) = Q(s, o) + \alpha(r - Q(s, o) + \gamma((1 - \beta)Q(s_prime, o) + \beta(MAXQ(s_prime, o_prime)))))

        # if terminal in next_state
        # Q(s, o) = Q(s, o) + \alpha(r - Q(s, o) + \gamma*MAXQ(next_state, o_prime))

        # if not terminal in next_state
        # Q(s, o) = Q(s, o) + \alpha*(r - Q(s, o) + \gamma*Q(s_prime, o))

        state_str = np.array2string(np.ndarray.astype(state, dtype=self.state_dtype))

        available_options = self.get_available_options(state, self.last_possible_actions)
        state_option_values = self.get_state_option_values(state, available_options)

        next_available_options = []
        if not terminal:
            next_available_options = self.get_available_options(next_state, next_state_possible_actions)

        next_state_option_values_list = [0.0]
        next_state_option_values = self.get_state_option_values(next_state)
        if next_available_options:
            next_state_option_values_list = [next_state_option_values[option] for option in next_available_options]

        max_next_state_option_value = max(next_state_option_values_list)

        for option in available_options:
            if option.has_policy():
                train_option = option.choose_action(state, self.last_possible_actions) == action
                try:
                    reset_inner_option_policy = option.policy.current_option is None
                except AttributeError:
                    reset_inner_option_policy = False
            else:
                train_option = option.actions[0] == action
                reset_inner_option_policy = False

            if train_option:
                if reset_inner_option_policy:
                    option.policy.current_option = None

                gamma_product = max_next_state_option_value
                if not option.terminated(next_state):
                    try:
                        gamma_product = next_state_option_values[option]
                    except KeyError:
                        gamma_product = max_next_state_option_value

                self.state_option_values[state_str][option] += self.alpha * (reward - state_option_values[option] +
                                                                             self.gamma * gamma_product)

        if not (terminal or self.current_option.terminated(next_state)):
            return

        option_value = self.get_state_option_values(self.option_start_state)[self.current_option]
        option_start_state_str = np.array2string(np.ndarray.astype(self.option_start_state, dtype=self.state_dtype))
        self.state_option_values[option_start_state_str][self.current_option] \
            += self.alpha * (self.total_option_reward + (self.gamma ** self.current_option_step) *
                             max_next_state_option_value
                             - option_value)
        self.current_option = None
        self.option_start_state = None
        self.total_option_reward = 0
        return

    def load_policy(self, load_path):
        with open(load_path, "r") as f:
            data = json.load(f)

        # Getting agent policy
        self.state_option_values = data['agent_policy']

        # Getting policy of options to subgoals
        for level in range(self.num_option_levels):
            for i in range(self.num_goals_per_level):
                if i == 0:
                    self.options[level][i]['option'].policy.q_values = data['options_between_subgoals'][level][i]
                else:
                    self.options[level][i]['option'].policy.state_option_values = \
                        data['options_between_subgoals'][level][i]

        # Getting policy of options between subgoals
        for level in range(self.num_option_levels - 1):
            for i in range(len(self.options_between_goals[level])):
                self.options_between_goals[level][i]['option'].policy.q_values = \
                    data['options_between_subgoals'][level][i]
        return

    def print_options(self):
        level = 1
        for options in self.options:
            print("Options at level " + str(level))
            for option in options:
                from_state = "all states"
                if option['start_state'] is not None:
                    from_state = option['start_state']
                print("From " + from_state + " to " + option['goal'])
            level += 1

        level = 1
        for options in self.options_between_goals:
            print("Options at level " + str(level))
            for option in options:
                from_state = "all states"
                if option['start_state'] is not None:
                    from_state = option['start_state']
                print("From " + from_state + " to " + option['goal'])
            level += 1
        return

    def save(self, save_path):
        data = {'options_to_subgoals': {},
                'agent_policy': self.state_option_values}

        # Get options to subgoals
        for level in range(self.num_option_levels):
            option_data = {}
            for i in range(self.num_goals_per_level[level]):
                if level == 0:
                    option_data[i] = self.options[level][i]['option'].policy.q_values
                else:
                    option_data[i] = self.options[level][i]['option'].policy.state_option_values
            data['options_to_subgoals'][level] = option_data

        # Get options between subgoals
        option_data = {}
        for level in range(self.num_option_levels - 1):
            data_to_add = {}
            for i in range(len(self.options_between_goals[level])):
                data_to_add[i] = self.options_between_goals[level][i]['option'].policy.q_values
            option_data[level] = data_to_add
        data['options_between_subgoals'] = option_data

        with open(save_path, 'w') as f:
            json.dump(data, f)
        return

    def train_options(self, environment: Environment, training_steps,
                      all_actions_valid=True,
                      progress_bar=False):

        def train_option(option_dict):
            option = option_dict['option']

            done = True
            total_steps = 0

            current_possible_actions = environment.get_possible_actions()

            while total_steps < training_steps:
                if progress_bar:
                    progress_bar_str = "all states"
                    if option_dict['start_state'] is not None:
                        progress_bar_str = option_dict['start_state']
                    progress_bar_str += " to " + option_dict["goal"]
                    print_progress_bar(total_steps, training_steps,
                                       prefix='Training Option from ' + progress_bar_str + ': ', suffix='Complete')

                if done:
                    start_state = option_dict['start_state']
                    if start_state is not None:
                        if '\n' not in start_state:
                            start_state = np.fromstring(start_state[1: len(start_state) - 1],
                                                        sep=' ', dtype=self.state_dtype)
                        else:
                            start_state = start_state.replace('[', '')
                            start_state = start_state.replace(']', '')
                            start_state = start_state.replace('\n', '')
                            start_state = np.fromstring(start_state,
                                                        sep=' ', dtype=self.state_dtype)
                            start_state = start_state.reshape((3, 3))
                    while start_state is None:
                        start_state = environment.generate_random_state()
                        start_state_str = np.array2string(np.ndarray.astype(start_state, dtype=self.state_dtype))
                        state_index = self.get_state_index(start_state_str)
                        if not nx.has_path(self.stg, state_index, option_dict['goal_index']):
                            start_state = None
                    state = environment.reset(start_state)
                    done = False

                if option_dict['goal_index'] == '17' and option_dict['start_state'] is None:
                    k = 9

                if not all_actions_valid:
                    current_possible_actions = environment.get_possible_actions()

                action = option.policy.choose_action(state, possible_actions=current_possible_actions)
                if action is None:
                    done = True
                    continue
                next_state, _, done, _ = environment.step(action)

                reward = 0.0
                if np.array_equal(state, next_state):
                    reward = -0.1
                state_str = np.array2string(np.ndarray.astype(next_state, dtype=self.state_dtype))
                state_index = self.get_state_index(state_str)

                if state_str == option_dict['goal']:
                    try:
                        if option.policy.current_option.terminated(next_state):
                            done = True
                            reward = 1.0
                    except AttributeError:
                        done = True
                        reward = 1.0
                elif done or not nx.has_path(self.stg, state_index, option_dict['goal_index']):
                    done = True
                    reward = -1.0

                if all_actions_valid:
                    option.policy.learn(state, action, reward, next_state, terminal=done)
                else:
                    current_possible_actions = environment.get_possible_actions()
                    option.policy.learn(state, action, reward, next_state, terminal=done,
                                        next_state_possible_actions=current_possible_actions)

                state = next_state
                total_steps += 1
            return

        # Training first level options: States to first level goals
        for option_dict in self.options[0]:
            train_option(option_dict)

        # Training first level options: between goals and the goals of the next level
        for level in range(self.num_option_levels - 1):
            for option_dict in self.options_between_goals[level]:
                train_option(option_dict)

        # Training higher-level options: from states to goals
        for level in range(1, self.num_option_levels):
            for option_dict in self.options[level]:
                train_option(option_dict)

        return
