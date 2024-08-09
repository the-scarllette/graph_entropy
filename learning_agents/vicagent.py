import numpy as np

from learning_agents.qlearningagent import QLearningAgent


def get_max_keys(dict):
    max_value = -np.inf
    max_keys = []
    for key in dict:
        value = dict[key]
        if value > max_value:
            max_value = value
            max_keys = [key]
        elif value == max_value:
            max_keys.append(key)
    return max_keys


class VICAgent:

    def __init__(self, actions, alpha, epsilon, gamma, reward_lambda):
        self.policy = QLearningAgent(actions, alpha, epsilon, gamma)
        self.reward_lambda = reward_lambda
        self.state_possible_actions = None
        self.total_num_actions = len(actions)
        return

    def choose_action(self, state, optimal_choice=False, possible_actions=None):
        self.state_possible_actions = possible_actions
        return self.policy.choose_action(state, optimal_choice, possible_actions)

    def get_prob_action_state(self, action, state, possible_actions=None):
        action_values = self.policy.get_action_values(state)
        max_actions = get_max_keys(action_values)
        num_actions = self.total_num_actions
        if possible_actions is not None:
            num_actions = len(possible_actions)

        prob_action_state = 0
        if num_actions > 0:
            prob_action_state = self.policy.epsilon * (1 / num_actions)

        if action in max_actions:
            prob_action_state += (1 - self.policy.epsilon) * (1 / len(max_actions))
        return prob_action_state

    def learn(self, state, action, reward, next_state, terminal=None, next_state_possible_actions=None):
        prob_action_state = self.get_prob_action_state(action, state, self.state_possible_actions)
        prob_action_next_state = 0
        if not terminal:
            prob_action_next_state = self.get_prob_action_state(action, next_state, next_state_possible_actions)
        intrinsic_reward = 0.0
        if prob_action_state > 0:
            intrinsic_reward -= np.log(prob_action_state)
        if prob_action_next_state > 0:
            intrinsic_reward += np.log(prob_action_next_state)

        total_reward = reward + (self.reward_lambda * intrinsic_reward)

        self.policy.learn(state, action, total_reward, next_state, terminal, next_state_possible_actions)
        return

    def save(self, save_path):
        self.policy.save(save_path)
        return
