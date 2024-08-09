import numpy as np
import random
from itertools import product
from typing import Any

from environments.environment import Environment


class MasterMind(Environment):

    no_value_number = -1
    default_code_length = 3
    default_possible_numbers = [0, 1, 3]
    default_maximum_guesses = 10

    failure_reward = -1.0
    step_reward = -0.01
    success_reward = 1.0

    def __init__(self, code_length=None, possible_numbers=None, maximum_guesses=None):

        self.code_length = code_length
        if self.code_length is None:
            self.code_length = self.default_code_length
        self.possible_numbers = possible_numbers
        if self.possible_numbers is None:
            self.possible_numbers = self.default_possible_numbers
        self.maximum_guesses = maximum_guesses
        if self.maximum_guesses is None:
            self.maximum_guesses = self.default_maximum_guesses
        self.num_possible_numbers = len(self.possible_numbers)

        self.num_possible_actions = pow(self.num_possible_numbers, self.code_length)
        self.possible_actions = list(range(self.num_possible_actions))
        self.action_code_lookup = list(product(self.possible_numbers, repeat=self.code_length))

        self.code = None
        self.current_state = None
        self.guess_count = 0
        self.terminal = True

        self.state_len = self.maximum_guesses * (self.code_length + 2)

        self.environment_name = 'mastermind_' + str(self.code_length)
        return

    def get_clue_from_guess(self, guess, code=None):
        if code is None:
            code = self.code

        correct_number_place = 0
        correct_number = 0

        code_holder = list(code).copy()
        guess = list(guess)

        for i in range(self.code_length):
            guess_number = guess[i]
            if guess_number == code_holder[i]:
                correct_number_place += 1
                code_holder[i] = None
                guess[i] = None

        for i in range(self.code_length):
            guess_number = guess[i]
            if guess_number is None:
                continue

            try:
                index = code_holder.index(guess_number)
                correct_number += 1
                code_holder[index] = None
            except ValueError:
                continue

        return [correct_number_place, correct_number]

    def get_previous_clue(self, clue_index, state=None):
        if state is None:
            state = self.current_state

        guess_start_index = clue_index * (self.code_length + 2)

        return state[guess_start_index + self.code_length: guess_start_index + self.code_length + 2]

    def get_previous_guess(self, guess_index, state=None):
        if state is None:
            state = self.current_state

        guess_start_index = guess_index * (self.code_length + 2)

        return state[guess_start_index: guess_start_index + self.code_length]

    def get_previous_guess_clue(self, guess_clue_index, state=None):
        if state is None:
            state = self.current_state

        guess_start_index = guess_clue_index * (self.code_length + 2)

        return state[guess_start_index: guess_start_index + self.code_length + 2]

    def get_start_states(self):
        return [np.full(self.state_len, self.no_value_number)]

    def get_successor_states(self, state, probability_weights=False):
        if self.is_terminal(state):
            return [], []

        successor_states = []
        successor_states_bytes = {}
        ways_to_reach_successor = []
        possible_codes = 0
        num_successors = 0

        num_previous_guesses = self.maximum_guesses
        guess_start_index = self.state_len - 1
        while 0 < guess_start_index:
            if (state[guess_start_index] != self.no_value_number) and (
                    state[guess_start_index - 1] != self.no_value_number):
                break
            num_previous_guesses -= 1
            guess_start_index -= self.code_length + 2
        guess_start_index += 1

        for possible_code in self.action_code_lookup:
            code_possible = True
            for guess_index in range(num_previous_guesses):
                guess_clue = self.get_previous_guess_clue(guess_index, state)
                guess = guess_clue[0: self.code_length]
                clue = guess_clue[self.code_length: self.code_length + 2]
                clue_from_code = self.get_clue_from_guess(guess, possible_code)

                if not np.array_equal(clue, clue_from_code):
                    code_possible = False
                    break

            if not code_possible:
                continue
            possible_codes += 1

            for possible_guess in self.action_code_lookup:
                clue_from_guess = self.get_clue_from_guess(possible_guess, possible_code)

                successor_state = state.copy()
                successor_state[guess_start_index: guess_start_index + self.code_length] = possible_guess
                successor_state[guess_start_index + self.code_length:
                                guess_start_index + self.code_length + 2] = clue_from_guess

                successor_state_bytes = successor_state.tobytes()
                try:
                    successor_state_index = successor_states_bytes[successor_state_bytes]
                    ways_to_reach_successor[successor_state_index] += 1
                except KeyError:
                    successor_states_bytes[successor_state_bytes] = num_successors
                    successor_states.append(successor_state)
                    ways_to_reach_successor.append(1)
                    num_successors += 1

        if not probability_weights:
            weights = [1.0] * num_successors
        else:
            weights = [elm / (possible_codes * self.num_possible_actions) for elm in ways_to_reach_successor]

        return successor_states, weights

    def play_text_based(self, code=None):
        _ = self.reset(code)

        while not self.terminal:
            print("Guess Numnber: " + str(self.guess_count + 1) + '/' + str(self.maximum_guesses) + ':\n')
            guess_str = input()
            guess = [int(elm) for elm in guess_str.split(' ')]

            if len(guess) != self.code_length:
                print("Invalid guess")
                continue

            guess_index = self.action_code_lookup.index(tuple(guess))

            _, _, _, _ = self.step(guess_index)

            clue_start_index = ((self.guess_count - 1) * (self.code_length + 2)) + self.code_length
            clue = self.current_state[clue_start_index: clue_start_index + 2]
            print("Correct Number and Position: " + str(clue[0]))
            print("Correct Number: " + str(clue[1]))

        if clue[0] >= self.code_length:
            print("Congrats, you found the code!")
            return

        print("Bad luck!")
        to_print = "The code was: "
        for elm in self.code:
            to_print += str(elm) + ' '
        print(to_print)
        return

    def is_terminal(self, state):
        if (state[self.state_len - 2] != self.no_value_number) or (state[self.state_len - 1] != self.no_value_number):
            return True

        last_clue_index = self.state_len - 2
        while self.code_length <= last_clue_index:
            if state[last_clue_index] == self.code_length:
                return True
            elif (state[last_clue_index] == self.no_value_number) and (
                    state[last_clue_index + 1] == self.no_value_number):
                last_clue_index -= self.code_length + 2
            else:
                return False
        return False

    def reset(self, code=None) -> Any:
        self.current_state = np.full(self.state_len, self.no_value_number)

        self.guess_count = 0
        self.terminal = False

        self.code = code
        if self.code is None:
            self.code = random.choice(self.action_code_lookup)
        return self.current_state

    def step(self, action) -> (Any, float, bool, Any):
        guess = self.action_code_lookup[action]
        start_index = self.guess_count * (self.code_length + 2)

        # Getting clue
        clue = self.get_clue_from_guess(guess)

        # Return pins
        self.current_state[start_index + self.code_length] = clue[0]
        self.current_state[start_index + self.code_length + 1] = clue[1]

        # append last guess to state
        for i in range(self.code_length):
            self.current_state[start_index + i] = guess[i]

        # Incrementing number of guesses
        self.guess_count += 1

        # Success or failure check
        reward = self.step_reward
        if clue[0] >= self.code_length:
            self.terminal = True
            reward = self.success_reward
        if self.guess_count >= self.maximum_guesses:
            self.terminal = True
            reward = self.failure_reward

        return self.current_state, reward, self.terminal, None
