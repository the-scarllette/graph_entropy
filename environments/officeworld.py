import numpy as np
import random as rand
from typing import Tuple

from environments.environment import Environment

class OfficeWorld(Environment):

    EMPTY = 0
    BLOCK = 1
    ELEVATOR = 2
    AGENT = 3
    GOAL = 4

    NORTH = 0
    SOUTH = 1
    WEST = 2
    EAST = 3
    ELEVATOR_UP = 4
    ELEVATOR_DOWN = 5

    SQUARE_ROOM = np.array(
        [
            [BLOCK, BLOCK, BLOCK, BLOCK, BLOCK],
            [BLOCK, EMPTY, EMPTY, EMPTY, BLOCK],
            [BLOCK, EMPTY, EMPTY, EMPTY, BLOCK],
            [BLOCK, EMPTY, EMPTY, EMPTY, BLOCK],
            [BLOCK, BLOCK, BLOCK, BLOCK, BLOCK]
        ]
    )

    SQUARE_ROOM_NORTH = np.array(
        [
            [BLOCK, BLOCK, EMPTY, BLOCK, BLOCK],
            [BLOCK, EMPTY, EMPTY, EMPTY, BLOCK],
            [BLOCK, EMPTY, EMPTY, EMPTY, BLOCK],
            [BLOCK, EMPTY, EMPTY, EMPTY, BLOCK],
            [BLOCK, BLOCK, BLOCK, BLOCK, BLOCK]
        ]
    )

    SQUARE_ROOM_SOUTH = np.array(
        [
            [BLOCK, BLOCK, BLOCK, BLOCK, BLOCK],
            [BLOCK, EMPTY, EMPTY, EMPTY, BLOCK],
            [BLOCK, EMPTY, EMPTY, EMPTY, BLOCK],
            [BLOCK, EMPTY, EMPTY, EMPTY, BLOCK],
            [BLOCK, BLOCK, EMPTY, BLOCK, BLOCK]
        ]
    )

    SQUARE_ROOM_EAST = np.array(
        [
            [BLOCK, BLOCK, BLOCK, BLOCK, BLOCK],
            [BLOCK, EMPTY, EMPTY, EMPTY, BLOCK],
            [BLOCK, EMPTY, EMPTY, EMPTY, EMPTY],
            [BLOCK, EMPTY, EMPTY, EMPTY, BLOCK],
            [BLOCK, BLOCK, BLOCK, BLOCK, BLOCK]
        ]
    )

    SQUARE_ROOM_WEST = np.array(
        [
            [BLOCK, BLOCK, BLOCK, BLOCK, BLOCK],
            [BLOCK, EMPTY, EMPTY, EMPTY, BLOCK],
            [EMPTY, EMPTY, EMPTY, EMPTY, BLOCK],
            [BLOCK, EMPTY, EMPTY, EMPTY, BLOCK],
            [BLOCK, BLOCK, BLOCK, BLOCK, BLOCK]
        ]
    )

    RECT_ROOM_NORTH = np.array(
        [
            [BLOCK, EMPTY, BLOCK],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [BLOCK, BLOCK, BLOCK]
        ]
    )

    RECT_ROOM_VERT = np.array(
        [
            [BLOCK, EMPTY, EMPTY, EMPTY, BLOCK],
            [BLOCK, EMPTY, EMPTY, EMPTY, BLOCK],
            [BLOCK, EMPTY, EMPTY, EMPTY, BLOCK],
            [BLOCK, EMPTY, EMPTY, EMPTY, BLOCK],
            [BLOCK, EMPTY, EMPTY, EMPTY, BLOCK]
        ]
    )
    RECT_ROOM_VERT_EAST_EXIT = RECT_ROOM_VERT.copy()
    RECT_ROOM_VERT_EAST_EXIT[2, 4] = EMPTY
    RECT_ROOM_VERT_WEST_EXIT = RECT_ROOM_VERT.copy()
    RECT_ROOM_VERT_WEST_EXIT[2, 0] = EMPTY

    RECT_ROOM_HOR = np.array(
        [
            [BLOCK, BLOCK, BLOCK, BLOCK, BLOCK],
            [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
            [BLOCK, BLOCK, BLOCK, BLOCK, BLOCK]
        ]
    )
    RECT_ROOM_HOR_NORTH_EXIT = RECT_ROOM_HOR.copy()
    RECT_ROOM_HOR_NORTH_EXIT[0, 2] = EMPTY
    RECT_ROOM_HOR_SOUTH_EXIT = RECT_ROOM_HOR.copy()
    RECT_ROOM_HOR_SOUTH_EXIT[4, 2] = EMPTY

    CORRIDOR_VERT = np.array(
        [
            [EMPTY, EMPTY],
            [EMPTY, EMPTY],
            [EMPTY, EMPTY],
            [EMPTY, EMPTY],
            [EMPTY, EMPTY]
        ]
    )

    CORRIDOR_HOR = np.array(
        [
            [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY]
        ]
    )

    possible_actions = [
        NORTH,
        SOUTH,
        WEST,
        EAST,
        ELEVATOR_UP,
        ELEVATOR_DOWN,
    ]

    default_floor_map = SQUARE_ROOM.copy()
    default_floor_map[2, 4] = EMPTY
    default_floor_map[4, 2] = EMPTY

    default_floor_map = np.concatenate(
        [
            default_floor_map,
            RECT_ROOM_VERT_EAST_EXIT,
            SQUARE_ROOM_EAST
        ],
        axis=0
    )
    default_room_slices = [
        [
            CORRIDOR_VERT,
            CORRIDOR_VERT,
            CORRIDOR_VERT
        ],
        [
            SQUARE_ROOM_WEST,
            RECT_ROOM_HOR,
            SQUARE_ROOM_WEST
        ],
        [
            SQUARE_ROOM_EAST,
            RECT_ROOM_HOR,
            SQUARE_ROOM_EAST
        ],
        [
            CORRIDOR_VERT,
            CORRIDOR_VERT,
            CORRIDOR_VERT
        ],
        [
            SQUARE_ROOM_WEST,
            RECT_ROOM_VERT_WEST_EXIT,
            SQUARE_ROOM_WEST
        ]
    ]

    for slice in default_room_slices:
        to_concatenate = np.concatenate(
            slice,
            axis=0
        )
        default_floor_map = np.concatenate(
            [
                default_floor_map,
                to_concatenate
            ],
            axis=1
        )
    default_floor_map = np.concatenate(
        [
            default_floor_map,
            default_floor_map
        ],
        axis=0
    )

    invalid_action_reward = -1
    step_reward = -0.001
    success_reward = 2.0

    def __init__(
            self,
            floor_map: None|np.ndarray=None,
            elevator_coord: None|Tuple[int]=None
    ):
        self.state_dtype = None
        self.state_shape = None
        return


