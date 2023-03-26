import json
import random as rand
import networkx as nx
import numpy as np

from environments.environment import Environment
from learning_agents.qlearningagent import QLearningAgent


def generate_options_from_goals(environment: Environment, stg: nx.Graph, goals, all_states, training_episodes,
                                true_state=True, all_actions_valid=True):

    def generate_initiation_func(target):
        target_index = all_states.index(target.tobytes())
        def initiation_func(s):
            if np.array_equal(s, target):
                return False
            s_index = all_states.index(s.tobytes())
            return nx.has_path(stg, s_index, target_index)
        return initiation_func

    def generate_terminating_func(target):
        target_index = all_states.index(target.tobytes())
        def terminating_func(s):
            if np.array_equal(s, target):
                return True
            s_index = all_states.index(s.tobytes())
            return not nx.has_path(stg, s_index, target_index)
        return terminating_func


    options = []
    start_states = [s.tobytes() for s in environment.get_start_states()]
    for goal in goals:
        if goal.tobytes() in start_states:
            continue

        policy = QLearningAgent(environment.possible_actions, alpha=0.9, epsilon=0.1, gamma=0.9)
        option_terminating_func = generate_terminating_func(goal)

        for _ in range(training_episodes):
            done = False
            state = environment.reset(true_state)
            if not all_actions_valid:
                current_possible_actions = environment.get_possible_actions()
            while not done:
                if all_actions_valid:
                    action = policy.choose_action(state)
                else:
                    action = policy.choose_action(state, possible_actions=current_possible_actions)
                next_state, _, done, _ = environment.step(action, true_state=true_state)
                reward = 0.0
                if np.array_equal(next_state, goal):
                    done = True
                    reward = 1.0
                elif option_terminating_func(next_state):
                    done = True

                if all_actions_valid:
                    policy.learn(state, action, reward, next_state)
                else:
                    current_possible_actions = environment.get_possible_actions()
                    policy.learn(state, action, reward, next_state,
                                 next_state_possible_actions=current_possible_actions)
                state = next_state

        option = Option(environment.possible_actions, policy,
                        initiation_func=generate_initiation_func(goal),
                        terminating_func=option_terminating_func)
        options.append(option)
    return options


class Option:

    def __init__(self, actions=[], policy=None, initiation_func=None, terminating_func=None):
        self.actions = actions
        self.policy = policy

        self.initiation_func = initiation_func
        self.terminating_func = terminating_func
        return

    def choose_action(self, state, possible_actions=None):
        if self.policy is None:
            raise AttributeError('Option must have a defined policy')
        return self.policy.choose_action(state, True, possible_actions=possible_actions)

    def has_policy(self):
        return self.policy is not None

    def initiated(self, state):
        if self.initiation_func is None:
            return True
        return self.initiation_func(state)

    def save(self, save_path):
        if self.policy is None:
            return
        self.policy.save(save_path)
        return

    def terminated(self, state):
        if self.terminating_func is None:
            return True
        return self.terminating_func(state)


class OptionsAgent:

    def __init__(self, alpha, epsilon, gamma, options, step_size=None, intra_option=False):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        self.options = options

        self.step_size = step_size
        self.current_step = 0

        self.current_option = None
        self.option_start_state = None
        self.total_option_reward = 0
        self.current_option_step = 0

        self.state_option_values = {}

        self.intra_option = intra_option
        return

    def choose_action(self, state, optimal_choice=False, possible_actions=None):
        if self.current_option is None or (self.step_size is not None and self.current_step % self.step_size == 0):
            self.current_option = self.choose_option(state, optimal_choice, possible_actions)

        if self.current_option.has_policy():
            chosen_action = self.current_option.choose_action(state, possible_actions)
        else:
            chosen_action = self.current_option.actions[self.current_option_step]

        if chosen_action == -1:
            self.current_option = None
            return self.choose_action(state, optimal_choice)

        self.current_option_step += 1
        self.current_step += 1
        return chosen_action

    def choose_option(self, state, no_random, possible_actions=None):
        self.current_option_step = 0
        self.option_start_state = state

        available_options = self.get_available_options(state, possible_actions=possible_actions)

        if not no_random and rand.uniform(0, 1) < self.epsilon:
            return rand.choice(available_options)

        option_values = self.get_state_option_values(state, available_options)

        ops = [available_options[0]]
        max_value = option_values[available_options[0]]
        for i in range(1, len(available_options)):
            op = available_options[i]
            value = option_values[op]
            if value > max_value:
                max_value = value
                ops = [op]
            elif value == max_value:
                ops.append(op)
        return rand.choice(ops)

    def get_available_options(self, state, possible_actions=None):
        available_options = []
        for option in self.options:
            if possible_actions is not None:
                if not option.has_policy():
                    if option.actions[0] in possible_actions:
                        available_options.append(option)
                    continue
            if option.initiated(state):
                available_options.append(option)
        return available_options

    def get_state_option_values(self, state, available_options=None):
        state_str = np.array2string(state)

        try:
            option_values = self.state_option_values[state_str]
        except KeyError:
            if self.intra_option:
                available_options = self.options
            elif available_options is None:
                available_options = self.get_available_options(state)
            option_values = {option: 0.0 for option in available_options}
            self.state_option_values[state_str] = option_values
        return option_values

    def intra_option_learning(self, state, action, reward, next_state, terminal):
        state_option_value = self.get_state_option_values(state)[self.current_option]
        next_state_option_values = self.get_state_option_values(next_state)

        option = self.current_option
        if not terminal and not self.current_option.terminated(next_state):
            u = next_state_option_values[self.current_option]
        else:
            u = max(list(next_state_option_values.values()))
            self.current_option = None
            self.option_start_state = None

        state_str = np.array2string(state)
        self.state_option_values[state_str][option] += self.alpha * \
                                                       (reward + (self.gamma * u) - state_option_value)
        return

    def learn(self, state, action, reward, next_state,
              terminal=None, next_state_possible_actions=None):
        if self.intra_option:
            self.intra_option_learning(state, action, reward, next_state, terminal)
            return

        self.total_option_reward += reward
        if not (terminal or self.current_option.terminated(next_state)):
            return

        option_value = self.get_state_option_values(self.option_start_state)[self.current_option]
        all_next_options = self.get_state_option_values(next_state)

        next_options = all_next_options
        if next_state_possible_actions is not None:
            next_options = []
            for option in self.options:
                if not option.has_policy():
                    if option.actions[0] in next_state_possible_actions:
                        next_options.append(option)
                    continue

                if option.initiated(next_state):
                    next_options.append(option)

        next_option_values = [all_next_options[option] for option in next_options]
        if not next_options:
            next_option_values = [0.0]
        max_next_option = max(next_option_values)

        state_str = np.array2string(self.option_start_state)

        self.state_option_values[state_str][self.current_option] += self.alpha * \
                                                                    (self.total_option_reward +
                                                                     (self.gamma ** self.current_option_step) *
                                                                     max_next_option
                                                                     - option_value)
        self.current_option = None
        self.option_start_state = None
        self.total_option_reward = 0
        return

    def save(self, save_path):
        try:
            f = open(save_path, 'x')
            f.close()
        except FileExistsError:
            ()

        data = {'options': {}}
        num_options = len(self.options)
        for i in range(num_options):
            option = self.options[i]
            if not option.has_policy():
                continue
            data['options'][i] = option.policy.q_values

        data['option values'] = {}
        for state in self.state_option_values:
            data['option values'][state] = {self.options.index(option): self.state_option_values[state][option]
                                            for option in self.state_option_values[state]}

        with open(save_path, 'w') as f:
            json.dump(data, f)
        return
