import numpy as np
import random as rand
from typing import Tuple

from environments.environment import Environment

class OfficeWorld(Environment):

    EMPTY = 0
    BLOCK = 1
    ELEVATOR = 2

    NORTH = 0
    SOUTH = 1
    WEST = 2
    EAST = 3
    ELEVATOR_UP = 4
    ELEVATOR_DOWN = 5

    possible_actions = [
        NORTH,
        SOUTH,
        WEST,
        EAST,
        ELEVATOR_UP,
        ELEVATOR_DOWN,
    ]

    default_floor_map = np.array(
        [
            [BLOCK, BLOCK, BLOCK, BLOCK, BLOCK],
            [BLOCK, EMPTY, EMPTY, BLOCK, EMPTY],
            [BLOCK, EMPTY, EMPTY, BLOCK, EMPTY],
            [BLOCK, EMPTY, EMPTY, BLOCK, BLOCK],
            [BLOCK, EMPTY, EMPTY, BLOCK, EMPTY],
            [BLOCK, EMPTY, EMPTY, BLOCK, EMPTY],
            [BLOCK, EMPTY, EMPTY, EMPTY, EMPTY],
            [BLOCK, BLOCK, BLOCK, BLOCK, BLOCK],
            [BLOCK, EMPTY, EMPTY, BLOCK, EMPTY],
            [BLOCK, EMPTY, EMPTY, BLOCK, EMPTY],
            [BLOCK, EMPTY, EMPTY, BLOCK, EMPTY],
            [BLOCK, EMPTY, BLOCK, BLOCK, EMPTY],
            [BLOCK, EMPTY, EMPTY, EMPTY],
            [BLOCK, EMPTY, EMPTY, EMPTY],
            [BLOCK, EMPTY, BLOCK, BLOCK],
            [BLOCK, EMPTY, EMPTY, BLOCK],
            [BLOCK, EMPTY, EMPTY, BLOCK],
            [BLOCK, EMPTY, EMPTY, BLOCK],
            [BLOCK, BLOCK, BLOCK, BLOCK],
            [BLOCK, EMPTY, EMPTY, BLOCK],
            [BLOCK, EMPTY, EMPTY, BLOCK],
            [BLOCK, EMPTY, BLOCK, BLOCK],
            [BLOCK, EMPTY, EMPTY, EMPTY],
            [BLOCK, EMPTY, EMPTY, EMPTY],
            [BLOCK, EMPTY, BLOCK, BLOCK],
            [BLOCK, EMPTY, EMPTY, BLOCK],
            [BLOCK, EMPTY, EMPTY, BLOCK],
            [BLOCK, EMPTY, EMPTY, BLOCK],
            [BLOCK, EMPTY, BLOCK, BLOCK],
            [BLOCK, EMPTY, EMPTY, BLOCK],
            [BLOCK, EMPTY, EMPTY, BLOCK],
            [BLOCK, EMPTY, EMPTY, BLOCK],
            [BLOCK, BLOCK, BLOCK, BLOCK],
        ]
    )

    invalid_action_reward = -1
    success_reward = 2.0

    def __init__(
            self,
            floor_map: None|np.ndarray=None,
            elevator_coord: None|Tuple[int]=None
    ):
        return


