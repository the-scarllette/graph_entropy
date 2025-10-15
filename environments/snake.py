import numpy as np
import random as rand
from typing import Dict,List, Tuple

from environments.environment import Environment

# STATE: food coords, head coords, body coords (fill until width X height values with 0)

class Snake(Environment):

    EMPTY_COORDS: np.ndarray = np.array([-1, -1])

    NORTH: int=0
    SOUTH: int = 1
    EAST: int = 2
    WEST: int=3

    possible_actions = [NORTH, SOUTH, EAST, WEST]

    EMPTY_TILE: int=0
    HEAD_TILE: int=1
    FOOD_TILE: int=2
    BODY_TILE: int=3

    collect_food_reward: float = 0.0
    failure_reward: float = -0.5
    step_reward: float = 0.0
    terminal_reward_per_food: float = 0.5

    head_coords: (int, int)=(-1, -1)
    body_coords: List[Tuple[int, int]]=[]
    food_coords: (int, int)=(-1, -1)

    def __init__(
            self,
            width: int,
            height: int,
            start_length: int
    ):
        self.width: int = width
        self.height: int = height
        self.start_length: int = start_length
        self.max_body_length: int = (self.height * self.width) - 1

        if (self.start_length >= self.width) and (self.start_length >= self.height):
            raise ValueError("start_length must be less than either width or height of environment")

        self.environment_name: str = (
                "snake_" + str(self.width) + "x" + str(self.height) + "_start_length_" + str(self.start_length)
        )

        self.state_dtype: type=int
        self.state_shape: (int, ) = (2, 2 + self.max_body_length)

        self.current_state: None|np.ndarray = None
        self.terminal: bool=True
        return

    def get_start_states(
            self
    ) -> List[np.ndarray]:
        start_states: List[np.ndarray] = []
        potential_start_states: List[np.ndarray] = []
        start_state: np.ndarray
        start_state_valid: bool=False

        for head_i in range(self.height):
            for head_j in range(self.width):
                start_state = np.full(self.state_shape, -1)
                start_state[:, 0] = [head_i, head_j]
                potential_start_states = []
                if self.start_length <= 0:
                    potential_start_states.append(np.copy(start_state))
                    start_state_valid = True
                else:
                    if head_i - self.start_length >= 0:
                        new_start_state = np.copy(start_state)
                        for body_length in range(2, 2 + self.start_length):
                            new_start_state[:, body_length] = [head_i - body_length + 1, head_j]
                        potential_start_states.append(np.copy(new_start_state))
                        start_state_valid = True
                    if head_i + self.start_length < self.height:
                        new_start_state = np.copy(start_state)
                        for body_length in range(2, 2 + self.start_length):
                            new_start_state[:, body_length] = [head_i + body_length -  1, head_j]
                        potential_start_states.append(np.copy(new_start_state))
                        start_state_valid = True
                    if head_j - self.start_length >= 0:
                        new_start_state = np.copy(start_state)
                        for body_length in range(2, 2 + self.start_length):
                            new_start_state[:, body_length] = [head_i, head_j - body_length + 1]
                        potential_start_states.append(np.copy(new_start_state))
                        start_state_valid = True
                    if head_j + self.start_length < self.width:
                        new_start_state = np.copy(start_state)
                        for body_length in range(2, 2 + self.start_length):
                            new_start_state[:, body_length] = [head_i, head_j + body_length - 1]
                        potential_start_states.append(np.copy(new_start_state))
                        start_state_valid = True

                if not start_state_valid:
                    continue

                for potential_start_state in potential_start_states:
                    for food_i in range(self.height):
                        for food_j in range(self.width):
                            if (head_i == food_i) and (head_j == food_j):
                                continue
                            can_place_food = True
                            for body_index in range(2, 2 + self.start_length):
                                if np.array_equal(potential_start_state[:, body_index], [food_i, food_j]):
                                    can_place_food = False
                                    break
                            if not can_place_food:
                                continue

                            potential_start_state[:, 1] = [food_i, food_j]
                            start_states.append(np.copy(potential_start_state))
                            potential_start_state[:, 1] = [-1, -1]

        return start_states

    def get_state_features(
            self,
            state: np.ndarray
    ) -> Dict[str, str]:
        state_features: Dict[str, str] = {
            'state': np.array2string(state),
            'head y': str(state[0, 0]),
            'head x': str(state[1, 0]),
            'food y': str(state[0, 1]),
            'food x': str(state[1, 1]),
            'head coords': str((state[0, 0], state[1, 0])),
            'food coords': str((state[0, 1], state[1, 1]))
        }

        body_length: int = 0
        for i in range(2, self.max_body_length + 2):
            if self.is_empty_coords(state[:, i]):
                break
            body_length += 1

        state_features['body length'] = str(body_length)

        return state_features

    def get_successor_states(
            self,
            state: np.ndarray,
            probability_weights: bool = False
    ) -> (List[np.ndarray], List[float]):

        if self.is_terminal(state):
            return [], []

        successor_states: List[np.ndarray] = []
        weights: List[float] = []
        move_successors: List[np.ndarray] = []
        num_successor_states: int = 0

        original_next_body_location: np.ndarray = np.copy(state[:, 0])

        successor_state = np.copy(state)
        successor_state[0, 0] -= 1
        move_successors.append(successor_state)
        successor_state = np.copy(state)
        successor_state[0, 0] += 1
        move_successors.append(successor_state)
        successor_state = np.copy(state)
        successor_state[1, 0] += 1
        move_successors.append(successor_state)
        successor_state = np.copy(state)
        successor_state[1, 0] -= 1
        move_successors.append(successor_state)

        successor_terminal: bool
        successor_food_generated: bool
        next_body_location: np.ndarray
        index: int
        weight: float = 1.0
        for successor_state in move_successors:
            successor_terminal = False
            successor_food_generated = True
            next_body_location = np.copy(original_next_body_location)

            body_length: int = 1
            for index in range(2, self.max_body_length + 2):
                body_location = np.copy(successor_state[:, index])

                if self.is_empty_coords(body_location):
                    break

                body_length += 1

                successor_state[:, index] = np.copy(next_body_location)
                next_body_location = np.copy(body_location)

            for i in range(2, 2 + body_length):
                if np.array_equal(successor_state[:, 0], successor_state[:, i]):
                    successor_terminal = True

            if np.array_equal(successor_state[:, 0], successor_state[:, 1]):
                successor_state[:, index] = np.copy(next_body_location)

                if index >= self.max_body_length + 1:
                    successor_terminal = True
                    successor_state[:, 1] = [-1, -1]
                    successor_food_generated = True
                else:
                    successor_food_generated = False
            elif not ((0 <= successor_state[0, 0] < self.height) and (0 <= successor_state[1, 0] < self.width)):
                successor_terminal = True
                successor_state[:, 0] = [-1, -1]
                successor_state[:, 1] = [-1, -1]

            if (not successor_terminal) and (not successor_food_generated):
                # Generate Successors from collecting food
                num_food_locations: int = self.max_body_length - index + 2
                can_place_food: bool
                true_successor: np.ndarray

                for i in range(self.height):
                    for j in range(self.width):
                        can_place_food = True
                        potential_food_location = np.array([i, j])

                        if np.array_equal(successor_state[:, 0], potential_food_location):
                            continue

                        for body_index in range(2, 2 + body_length):
                            if np.array_equal(successor_state[:, body_index], potential_food_location):
                                can_place_food = False
                                break

                        if can_place_food:
                            true_successor = np.copy(successor_state)
                            true_successor[:, 1] = np.copy(potential_food_location)

                            num_successor_states += 1
                            successor_states.append(np.copy(true_successor))
                            if probability_weights:
                                weight = 0.25 * (1/num_food_locations)
                            weights.append(weight)

            else:
                num_successor_states += 1
                successor_states.append(np.copy(successor_state))
                if probability_weights:
                    weight = 0.25
                weights.append(weight)

        successors_no_duplicates: List[np.ndarray] = []
        weights_no_duplicates: List[float] = []
        equal_indexes: List[int]
        for i in range(num_successor_states):
            successor_state = successor_states[i]
            if successor_state is None:
                continue

            equal_indexes = []
            for j in range(i + 1, num_successor_states):
                if successor_states[j] is None:
                    continue

                if np.array_equal(successor_state, successor_states[j]):
                    equal_indexes.append(j)
                    successor_states[j] = None

            successors_no_duplicates.append(np.copy(successor_state))
            weight = 1.0
            if probability_weights:
                weight = weights[i]
                for equal_index in equal_indexes:
                    weight += weights[equal_index]
            weights_no_duplicates.append(weight)

        return successors_no_duplicates, weights_no_duplicates

    def is_empty_coords(
            self,
            coords: np.ndarray
    ) -> bool:
        return np.array_equal(self.EMPTY_COORDS, coords)

    def is_terminal(
            self,
            state: None|np.ndarray=None,
    ) -> bool:
        if state is None:
            state = self.current_state
            if state is None:
                raise AttributeError("Must provide a state or environment must be initialised with reset method")

        # Head is greater than width and height
        head_coords = state[:, 0]
        if (
                (not ((0 <= head_coords[0] < self.height) and (0 <= head_coords[1] < self.width))) or
                self.is_empty_coords(head_coords)
        ):
            return True

        # Head is same tile as body
        maximum_length_reached: bool = True
        for index in range(2, self.max_body_length + 2):
            body_coords = state[:, index]
            if self.is_empty_coords(body_coords):
                maximum_length_reached = False
                break
            if np.array_equal(head_coords, body_coords):
                return True

        # nowhere else to spawn food (so entire body queue is full and food location is (-1, -1))
        return maximum_length_reached

    def print_state(
            self,
            state: None|np.ndarray=None
    ):
        if state is None:
            if self.terminal and (self.current_state is None):
                raise AttributeError("Either provide a state or print state while environment is not terminal.")
            state = self.current_state

        print_state: np.ndarray = np.full((self.height, self.width), self.EMPTY_TILE)

        if (0 <= state[0, 0] < self.height) and (0 <= state[1, 0] < self.width):
            print_state[state[0, 0], state[1, 0]] = self.HEAD_TILE

        if not self.is_empty_coords(state[:, 1]):
            print_state[state[0, 1], state[1, 1]] = self.FOOD_TILE

        for index in range(2, self.max_body_length + 2):
            body_coords = state[:, index]
            if self.is_empty_coords(body_coords):
                break
            print_state[body_coords[0], body_coords[1]] = self.BODY_TILE

        print(np.array2string(print_state))
        return

    def reset(
            self,
            start_state: None|np.ndarray=None,
            seed: None|int=None
    ) -> np.ndarray:
        if start_state is not None:
            self.current_state = start_state
            self.terminal = self.is_terminal()
            return self.current_state

        self.terminal = False

        if seed is not None:
            rand.seed(seed)

        self.current_state = np.full(self.state_shape, -1)

        if self.start_length <=0:
            head_row = rand.randint(0, self.height - 1)
            head_col = rand.randint(0, self.width - 1)
            self.current_state[:, 0] = [head_row, head_col]
        else:
            start_state_found = False
            while not start_state_found:
                head_row = rand.randint(0, self.height - 1)
                head_col = rand.randint(0, self.width - 1)
                self.current_state[:, 0] = [head_row, head_col]
                potential_directions = []
                if head_row - self.start_length >= 0:
                    start_state_found = True
                    potential_directions.append(0)
                if head_row + self.start_length < self.height:
                    start_state_found = True
                    potential_directions.append(1)
                if head_col - self.start_length >= 0:
                    start_state_found = True
                    potential_directions.append(2)
                if head_col + self.start_length < self.width:
                    start_state_found = True
                    potential_directions.append(3)

            direction = rand.choice(potential_directions)
            if direction == 0:
                for body_length in range(2, 2 + self.start_length):
                    self.current_state[:, body_length] = [head_row - body_length + 1, head_col]
            elif direction == 1:
                for body_length in range(2, 2 + self.start_length):
                    self.current_state[:, body_length] = [head_row + body_length - 1, head_col]
            elif direction == 2:
                for body_length in range(2, 2 + self.start_length):
                    self.current_state[:, body_length] = [head_row, head_col - body_length + 1]
            elif direction == 3:
                for body_length in range(2, 2 + self.start_length):
                    self.current_state[:, body_length] = [head_row, head_col + body_length - 1]

        food_placed = False
        while not food_placed:
            food_row = rand.randint(0, self.height - 1)
            food_col = rand.randint(0, self.width - 1)
            if (food_row == head_row) and (food_col == head_col):
                continue
            food_placed = True
            for body_index in range(2, self.start_length + 2):
                if np.array_equal(self.current_state[:, body_index], [food_row, food_col]):
                    food_placed = False
                    break
        self.current_state[:, 1] = [food_row, food_col]

        return np.copy(self.current_state)

    def step(
            self,
            action: int
    ) -> (np.ndarray, float, bool, None):
        # Move head location
        # Move body forward
        # check if head and food are same location:
        #   if yes, add body length
        # check if environment terminal
        #   if not terminal: produce new food location

        if self.terminal:
            raise AttributeError("Environment is terminated, must be initialised with reset method")

        next_body_location: np.ndarray = np.copy(self.current_state[:, 0])
        food_generated: bool = True

        if action == self.NORTH:
            self.current_state[0, 0] -= 1
        elif action == self.SOUTH:
            self.current_state[0, 0] += 1
        elif action == self.EAST:
            self.current_state[1, 0] += 1
        elif action == self.WEST:
            self.current_state[1, 0] -= 1
        else:
            raise ValueError(f"Invalid action {action}")

        body_length: int = 1
        for index in range(2, self.max_body_length + 2):
            body_location = np.copy(self.current_state[:, index])

            if self.is_empty_coords(body_location):
                break

            body_length += 1

            self.current_state[:, index] = np.copy(next_body_location)
            next_body_location = np.copy(body_location)

        reward: float = self.step_reward

        for i in range(2, body_length + 2):
            if np.array_equal(self.current_state[:, 0], self.current_state[:, i]):
                reward += self.failure_reward
                self.terminal = True

        if np.array_equal(self.current_state[:, 0], self.current_state[:, 1]):
            reward += self.collect_food_reward
            self.current_state[:, index] = np.copy(next_body_location)

            if index >= self.max_body_length + 1:
                self.terminal = True
                self.current_state[:, 1] = [-1, -1]
                food_generated = True
            else:
                food_generated = False
        elif not ((0 <= self.current_state[0, 0] < self.height) and (0 <= self.current_state[1, 0] < self.width)):
            self.terminal = True
            self.current_state[:, 0] = [-1, -1]
            self.current_state[:, 1] = [-1, -1]
            reward += self.failure_reward

        if (not self.terminal) and (not food_generated):
            food_array: None|np.ndarray = None
            while not food_generated:
                food_array = np.array([rand.randint(0, self.height - 1), rand.randint(0, self.width - 1)])

                if np.array_equal(food_array, self.current_state[:, 0]):
                    continue

                food_generated = True
                for index in range(2, self.max_body_length + 2):
                    if self.is_empty_coords(self.current_state[:, index]):
                        break
                    if np.array_equal(self.current_state[:, index], food_array):
                        food_generated = False
                        break

            self.current_state[:, 1] = np.copy(food_array)

        if self.terminal:
            reward += (self.terminal_reward_per_food * (body_length - self.start_length))

        return np.copy(self.current_state), reward, self.terminal, None
