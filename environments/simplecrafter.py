import numpy as np
import random
from typing import List, Tuple

from environments.environment import Environment


## Goal
# Get diamond in shortest time

## Actions
# NSEW
# Act with item
# Place table
# Make wood pickaxe
# Make stone pickaxe
# Make iron pickaxe

class SimpleCrafter(Environment):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3
    COLLECT = 4
    PLACE_TABLE = 5
    WOOD_PICKAXE = 6
    STONE_PICKAXE = 7
    IRON_PICKAXE = 8

    possible_actions = [
        NORTH,
        SOUTH,
        EAST,
        WEST,
        COLLECT,  # collect wood, collect stone, collect iron, collect diamond,
        PLACE_TABLE,  # Place table north
        WOOD_PICKAXE,
        STONE_PICKAXE,
        IRON_PICKAXE
    ]

    EMPTY = 0
    AGENT = 1
    WOOD = 2
    STONE = 3
    IRON = 4
    DIAMOND = 5
    TABLE = 6

    BLOCKS = [
        EMPTY,
        AGENT,
        WOOD,
        STONE,
        IRON,
        DIAMOND,
        TABLE
    ]

    MIN_WOOD = 4
    MIN_STONE = 1
    MIN_IRON = 1
    MIN_DIAMOND = 1

    default_start_state = np.array(
        [
            [WOOD, WOOD, WOOD, EMPTY],
            [WOOD, EMPTY, EMPTY, DIAMOND],
            [EMPTY, EMPTY, STONE, IRON],
            [ WOOD, EMPTY, EMPTY, STONE]
        ]
    )

    # Terminal when:
    # Has diamond
    # total wood in environment is < 4

    invalid_action_reward = -0.5
    step_reward = -0.01
    success_reward = 2.0
    failure_reward = -3.0

    # state:
    #   [flattened image of board]
    #   [num wood]
    #   [num stone]
    #   [num iron]
    #   [num diamond]
    #   [has wood pickaxe]
    #   [has stone pickaxe]
    #   [has iron pickaxe]

    def __init__(
            self,
            start_state: np.ndarray = default_start_state,
    ):
        self.action_space = len(self.possible_actions)
        self.environment_name = "simple_crafter"

        self.current_state = None
        self.start_state = start_state
        self.grid_len = self.start_state.shape[0]
        self.grid_size = self.grid_len * self.grid_len

        self.wood_index = self.grid_size
        self.stone_index = self.grid_size + 1
        self.iron_index = self.grid_size + 2
        self.diamond_index = self.grid_size + 3
        self.wood_pickaxe_index = self.grid_size + 4
        self.stone_pickaxe_index = self.grid_size + 5
        self.iron_pickaxe_index = self.grid_size + 6
        self.block_index_lookup = {
            SimpleCrafter.WOOD: self.wood_index,
            SimpleCrafter.STONE: self.stone_index,
            SimpleCrafter.IRON: self.iron_index,
            SimpleCrafter.DIAMOND: self.diamond_index
        }

        self.state_shape = (self.grid_size + 7,)
        self.state_dtype = float

        self.terminal = True
        pass

    def get_start_states(
            self
    ) -> List[np.ndarray]:
        start_state_template = np.zeros(self.state_shape)
        start_state_template[:self.grid_size] = np.reshape(self.start_state, (self.grid_size,))

        start_states = []
        for i in range(self.grid_size):
            if start_state_template[i] != SimpleCrafter.EMPTY:
                continue

            new_start_state = start_state_template.copy()
            new_start_state[i] = SimpleCrafter.AGENT
            start_states.append(new_start_state)

        return start_states

    def get_successor_states(
            self,
            state: np.ndarray,
            probability_weights: bool = False
    ) -> (List[np.ndarray], List[float]):
        if self.is_terminal(state):
            return [], []

        (
            unflattened_state,
            agent_x,
            agent_y,
            num_wood,
            num_stone,
            num_iron,
            num_diamond,
            wood_pickaxe,
            stone_pickaxe,
            iron_pickaxe
        ) = self.unflatten_state(state)

        successor_states = []
        stationary_actions = 0
        adj_cords = [ adj_cord
                      for adj_cord in [
                (agent_y - 1, agent_x),
                (agent_y + 1, agent_x),
                (agent_y, agent_x + 1),
                (agent_y, agent_x - 1)
            ]
                      if 0 <= adj_cord[0] < self.grid_len and 0 <= adj_cord[1] < self.grid_len
        ]

        if 0 < agent_y and unflattened_state[agent_y - 1][agent_x] == SimpleCrafter.EMPTY:
            successor_state = state.copy()
            successor_state[(agent_y * self.grid_len) + agent_x] = SimpleCrafter.EMPTY
            successor_state[((agent_y - 1) * self.grid_len) + agent_x] = SimpleCrafter.AGENT
            successor_states.append(successor_state)
        else:
            stationary_actions += 1

        if agent_y < self.grid_len - 1 and unflattened_state[agent_y + 1][agent_x] == SimpleCrafter.EMPTY:
            successor_state = state.copy()
            successor_state[(agent_y * self.grid_len) + agent_x] = SimpleCrafter.EMPTY
            successor_state[((agent_y + 1) * self.grid_len) + agent_x] = SimpleCrafter.AGENT
            successor_states.append(successor_state)
        else:
            stationary_actions += 1

        if agent_x < self.grid_len - 1 and unflattened_state[agent_y][agent_x + 1] == SimpleCrafter.EMPTY:
            successor_state = state.copy()
            successor_state[(agent_y * self.grid_len) + agent_x] = SimpleCrafter.EMPTY
            successor_state[(agent_y * self.grid_len) + agent_x + 1] = SimpleCrafter.AGENT
            successor_states.append(successor_state)
        else:
            stationary_actions += 1

        if 0 < agent_x and unflattened_state[agent_y][agent_x - 1] == SimpleCrafter.EMPTY:
            successor_state = state.copy()
            successor_state[(agent_y * self.grid_len) + agent_x] = SimpleCrafter.EMPTY
            successor_state[(agent_y * self.grid_len) + agent_x - 1] = SimpleCrafter.AGENT
            successor_states.append(successor_state)
        else:
            stationary_actions += 1

        collectable_blocks = [SimpleCrafter.WOOD]
        if wood_pickaxe == 1:
            collectable_blocks.append(SimpleCrafter.STONE)
        if stone_pickaxe == 1:
            collectable_blocks.append(SimpleCrafter.IRON)
        if iron_pickaxe == 1:
            collectable_blocks.append(SimpleCrafter.DIAMOND)
        successor_state = state.copy()
        collect_successor_state = False
        for adj_cord in adj_cords:
            if unflattened_state[adj_cord] in collectable_blocks:
                collect_successor_state = True
                successor_state[(adj_cord[0] * self.grid_len) + adj_cord[1]] = SimpleCrafter.EMPTY
                successor_state[self.block_index_lookup[unflattened_state[adj_cord]]] += 1
        if collect_successor_state:
            successor_states.append(successor_state)
        else:
            stationary_actions += 1

        if 0 < agent_y and 1 <= num_wood and unflattened_state[agent_y - 1][agent_x] == SimpleCrafter.EMPTY:
            successor_state = state.copy()
            successor_state[((agent_y - 1) * self.grid_len) + agent_x] = SimpleCrafter.TABLE
            successor_state[self.block_index_lookup[SimpleCrafter.WOOD]] -= 1
            successor_states.append(successor_state)
        else:
            stationary_actions += 1

        table_adjacent = False
        for adj_cord in adj_cords:
            if unflattened_state[adj_cord] == SimpleCrafter.TABLE:
                table_adjacent = True
        if not table_adjacent:
            stationary_actions += 3
        else:
            if 1 <= num_wood:
                successor_state = state.copy()
                successor_state[self.block_index_lookup[SimpleCrafter.WOOD]] -= 1
                successor_state[self.wood_pickaxe_index] = 1
                successor_states.append(successor_state)
            else:
                stationary_actions += 1
            if 1 <= num_stone and 1 <= num_wood:
                successor_state = state.copy()
                successor_state[self.block_index_lookup[SimpleCrafter.STONE]] -= 1
                successor_state[self.block_index_lookup[SimpleCrafter.WOOD]] -= 1
                successor_state[self.stone_pickaxe_index] = 1
                successor_states.append(successor_state)
            else:
                stationary_actions += 1
            if 1 <= num_iron and 1 <= num_wood:
                successor_state = state.copy()
                successor_state[self.block_index_lookup[SimpleCrafter.IRON]] -= 1
                successor_state[self.block_index_lookup[SimpleCrafter.WOOD]] -= 1
                successor_state[self.iron_pickaxe_index] = 1
                successor_states.append(successor_state)
            else:
                stationary_actions += 1

        if stationary_actions > 0:
            successor_states.append(state.copy())
        if not probability_weights:
            return successor_states, [0.0 for _ in successor_states]
        prob_weights = [1/self.action_space for _ in successor_states]
        if stationary_actions > 0:
            prob_weights.append(stationary_actions / self.action_space)
        return successor_states, prob_weights

    def is_terminal(
            self,
            state: np.ndarray
    ) -> bool:
        if state[self.block_index_lookup[SimpleCrafter.DIAMOND]]  >= 1:
            return True

        # Need enough wood in the domain to make an iron pickaxe
        total_wood = state[self.block_index_lookup[SimpleCrafter.WOOD]]
        if total_wood >= 4:
            return False

        unique, counts = np.unique(state[:self.grid_size], return_counts=True)
        block_counts = dict(zip(unique, counts))
        try:
            wood_block_count = block_counts[SimpleCrafter.WOOD]
        except KeyError:
            wood_block_count = 0
        total_wood += wood_block_count
        if total_wood >= 4:
            return False
        if block_counts[SimpleCrafter.TABLE] < 1:
            return True
        if total_wood >= 3:
            return False
        if total_wood > 2 and state[self.wood_pickaxe_index] >= 1:
            return False
        if total_wood >=1 and state[self.stone_pickaxe_index] >= 1:
            return False
        if state[self.iron_pickaxe_index] >= 1:
            return False

        return True

    def print_state(
            self,
            state: None|np.ndarray=None
    ):
        if state is None:
            if self.terminal:
                raise AttributeError("Either provide a state or print state while environment is not terminal.")
            state = self.current_state

        (
            unflattened_state,
            agent_x,
            agent_y,
            num_wood,
            num_stone,
            num_iron,
            num_diamond,
            wood_pickaxe,
            stone_pickaxe,
            iron_pickaxe
        ) = self.unflatten_state(state)

        print(np.array2string(unflattened_state))
        print(
            "Wood: " + str(num_wood)
            + " Stone: " + str(num_stone)
            + " Iron: " + str(num_iron)
            + " Diamond: " + str(num_diamond)
        )
        equipment_list = ""
        if wood_pickaxe >= 1:
            equipment_list += "Wood Pickaxe "
        if stone_pickaxe >= 1:
            equipment_list += "Stone Pickaxe "
        if iron_pickaxe >= 1:
            equipment_list += "Iron Pickaxe "
        print("Equipment: " + equipment_list)
        return

    def reset(
            self,
            start_state: None | np.ndarray = None,
            seed: None | int = None
    ) -> np.ndarray:
        self.terminal = False

        self.current_state = start_state
        if self.current_state is not None:
            return self.current_state.copy()

        unflattened_start_state = self.start_state.copy()
        agent_placed = False
        while not agent_placed:
            agent_y = random.randint(0, self.grid_len - 1)
            agent_x = random.randint(0, self.grid_len - 1)
            if unflattened_start_state[agent_y][agent_x] == SimpleCrafter.EMPTY:
                unflattened_start_state[agent_y][agent_x] = SimpleCrafter.AGENT
                agent_placed = True

        self.current_state = np.zeros(self.state_shape, dtype=self.state_dtype)
        self.current_state[:self.grid_size] = np.reshape(unflattened_start_state, (self.grid_size,))
        return self.current_state.copy()

    def step(
            self,
            action: int
    ) -> (np.ndarray, float, bool, None):
        if self.terminal:
            raise AttributeError("Cannot step when environment is terminal")

        (
            unflattened_state,
            agent_x,
            agent_y,
            num_wood,
            num_stone,
            num_iron,
            num_diamond,
            wood_pickaxe,
            stone_pickaxe,
            iron_pickaxe
        ) = self.unflatten_state(self.current_state)
        reward = self.step_reward
        invalid_action = False

        def alter_state(
                coords: List[Tuple[int, int]],
                new_values: List[int],
        ):
            for i in range(len(coords)):
                unflattened_state[coords[i]] = new_values[i]
            self.current_state[:self.grid_size] = np.reshape(unflattened_state, (self.grid_size,))
            return

        if action == SimpleCrafter.NORTH:
            if 0 < agent_y and unflattened_state[agent_y - 1][agent_x] == SimpleCrafter.EMPTY:
                alter_state(
                    [(agent_y, agent_x), (agent_y - 1, agent_x)],
                    [SimpleCrafter.EMPTY, SimpleCrafter.AGENT]
                )
            else:
                invalid_action = True
        elif action == SimpleCrafter.SOUTH:
            if agent_y < self.grid_len - 1 and unflattened_state[agent_y + 1][agent_x] == SimpleCrafter.EMPTY:
                alter_state(
                    [(agent_y, agent_x), (agent_y + 1, agent_x)],
                    [SimpleCrafter.EMPTY, SimpleCrafter.AGENT]
                )
            else:
                invalid_action = True
        elif action == SimpleCrafter.EAST:
            if agent_x < self.grid_len - 1 and unflattened_state[agent_y][agent_x + 1] == SimpleCrafter.EMPTY:
                alter_state(
                    [(agent_y, agent_x), (agent_y, agent_x + 1)],
                    [SimpleCrafter.EMPTY, SimpleCrafter.AGENT]
                )
            else:
                invalid_action = True
        elif action == SimpleCrafter.WEST:
            if 0 < agent_x and unflattened_state[agent_y][agent_x - 1] == SimpleCrafter.EMPTY:
                alter_state(
                    [(agent_y, agent_x), (agent_y, agent_x - 1)],
                    [SimpleCrafter.EMPTY, SimpleCrafter.AGENT]
                )
            else:
                invalid_action = True
        elif action == SimpleCrafter.PLACE_TABLE:
            if 1 < num_wood and 0 < agent_y and unflattened_state[agent_y - 1][agent_x] == SimpleCrafter.EMPTY:
                alter_state(
                    [(agent_y - 1, agent_x)],
                    [SimpleCrafter.TABLE]
                )
                self.current_state[self.block_index_lookup[SimpleCrafter.WOOD]] -= 1
            else:
                invalid_action = True
        else:
            adj_cords = [adj_cord
                         for adj_cord in
                         [
                             (agent_y - 1, agent_x),
                             (agent_y + 1, agent_x),
                             (agent_y, agent_x - 1),
                             (agent_y, agent_x + 1)
                         ]
                         if 0 <= adj_cord[0] < self.grid_len and 0 <= adj_cord[1] < self.grid_len
                         ]
            if action == SimpleCrafter.COLLECT:
                collectable_blocks = [SimpleCrafter.WOOD]
                coords_to_empty = []
                if self.current_state[self.wood_pickaxe_index] >= 1:
                    collectable_blocks.append(SimpleCrafter.STONE)
                if self.current_state[self.stone_pickaxe_index] >= 1:
                    collectable_blocks.append(SimpleCrafter.IRON)
                if self.current_state[self.iron_pickaxe_index] >= 1:
                    collectable_blocks.append(SimpleCrafter.DIAMOND)
                blocks_collected = False
                for adj_cord in adj_cords:
                    if (
                            0 <= adj_cord[0] < self.grid_len and
                            0 <= adj_cord[1] < self.grid_len and
                            unflattened_state[adj_cord] in collectable_blocks
                    ):
                        blocks_collected = True
                        coords_to_empty.append(adj_cord)
                        self.current_state[self.block_index_lookup[unflattened_state[adj_cord]]] += 1
                        if unflattened_state[adj_cord] == SimpleCrafter.DIAMOND:
                            self.terminal = True
                            reward += self.success_reward

                if blocks_collected:
                    alter_state(
                        coords_to_empty,
                        [SimpleCrafter.EMPTY for _ in coords_to_empty]
                    )
                else:
                    invalid_action = True

            table_adjacent = False
            for adj_cord in adj_cords:
                if unflattened_state[adj_cord] == SimpleCrafter.TABLE:
                    table_adjacent = True
                    break
            if not table_adjacent:
                invalid_action = True
            elif action == SimpleCrafter.WOOD_PICKAXE:
                if num_wood >= 1:
                    self.current_state[self.wood_pickaxe_index] = 1
                    self.current_state[self.block_index_lookup[SimpleCrafter.WOOD]] -= 1
                else:
                    invalid_action = True
            elif action == SimpleCrafter.STONE_PICKAXE:
                if num_stone >= 1 and num_wood >= 1:
                    self.current_state[self.stone_pickaxe_index] = 1
                    self.current_state[self.block_index_lookup[SimpleCrafter.WOOD]] -= 1
                    self.current_state[self.block_index_lookup[SimpleCrafter.STONE]] -= 1
                else:
                    invalid_action = True
            elif action == SimpleCrafter.IRON_PICKAXE:
                if num_iron >= 1 and num_wood >= 1:
                    self.current_state[self.iron_pickaxe_index] = 1
                    self.current_state[self.block_index_lookup[SimpleCrafter.WOOD]] -= 1
                    self.current_state[self.block_index_lookup[SimpleCrafter.IRON]] -= 1
                else:
                    invalid_action = True

        if invalid_action:
            reward += self.invalid_action_reward

        if not self.terminal:
            self.terminal = self.is_terminal(self.current_state)
            if self.terminal:
                reward += self.failure_reward

        return self.current_state.copy(), reward, self.terminal, None

    def unflatten_state(
            self,
            state: np.ndarray
    ) -> (np.ndarray, int, int, int, int, int, int, int, int, int):
        unflattened_state = np.reshape(state[:self.grid_size], (self.grid_len, self.grid_len))

        agent_cords = np.argwhere(unflattened_state == SimpleCrafter.AGENT)[0]
        agent_y = agent_cords[0]
        agent_x = agent_cords[1]

        return (
            unflattened_state,
            agent_x,
            agent_y,
            state[self.wood_index],
            state[self.stone_index],
            state[self.iron_index],
            state[self.diamond_index],
            state[self.wood_pickaxe_index],
            state[self.stone_pickaxe_index],
            state[self.iron_pickaxe_index]
        )

