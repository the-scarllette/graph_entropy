import numpy as np

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
    DO = 4
    PLACE_TABLE = 5
    WOOD_PICKAXE = 6
    STONE_PICKAXE = 7
    IRON_PICKAXE = 8

    possible_actions = [
        NORTH,
        SOUTH,
        EAST,
        WEST,
        DO,
        PLACE_TABLE,
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

    default_grid_len = 6

    # Terminal when:
    # Has diamond
    # total wood in environment is < 4

    invalid_action_reward = -0.1
    step_reward = -0.001
    terminal_reward = 1.0

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
            grid_len: int=default_grid_len
    ):
        self.current_state = None
        self.grid_len = grid_len
        self.grid_size = self.grid_len * self.grid_len

        self.wood_index = self.grid_size
        self.stone_index = self.grid_size + 1
        self.iron_index = self.grid_size + 2
        self.diamond_index = self.grid_size + 3
        self.wood_pickaxe_index = self.grid_size + 4
        self.stone_pickaxe_index = self.grid_size + 5
        self.iron_pickaxe_index = self.grid_size + 6

        self.state_shape = (self.grid_size + 7,)
        self.state_dtype = int
        pass

    def unflatten_state(
            self,
            state: np.ndarray
    ) -> np.ndarray:
        unflattened_state = np.reshape(state[:self.grid_size], (self.grid_len, self.grid_len))
        return unflattened_state

    def reset(
            self,
            start_state: None|np.ndarray=None,
            seed: None|int=None
    ) -> np.ndarray:
        self.current_state = start_state
        if self.current_state is not None:
            return self.current_state
        pass
