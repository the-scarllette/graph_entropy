import numpy as np
import random as rand
from typing import Any, List, Tuple, Type

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

    default_num_floors: int=3

    x_locations = [2, 9, 14, 21]
    y_locations = [2, 7, 12, 17, 22, 27]
    default_goal_locations: List[Tuple[int, int, int]]=[
        (y, x, floor)
        for y in y_locations
        for x in x_locations
        for floor in range(default_num_floors)
    ]

    default_elevator_coord: Tuple[int, int, int]=(0, 5, 0)

    default_start_coords: List[Tuple[int, int, int]]=[
        (0, 17),
        (0, 18),
        (30, 5),
        (30, 6),
        (30, 17),
        (30, 18)
    ]

    invalid_action_reward = -1
    step_reward = -0.001
    success_reward = 2.0

    def __init__(
            self,
            floor_map: None|np.ndarray=None,
            elevator_coord: None|Tuple[int, int, int]=None,
            start_coords: None|List[Tuple[int, int, int]]=None,
            num_floors: None|int=None
    ):
        self.floor_map: np.ndarray = floor_map
        if floor_map is None:
            self.floor_map = self.default_floor_map
        self.elevator_coord = elevator_coord
        if elevator_coord is None:
            self.elevator_coord = self.default_elevator_coord
        self.start_coords = start_coords
        if start_coords is None:
            self.start_coords = self.default_start_coords
        self.num_floors = num_floors
        if num_floors is None:
            self.num_floors = self.default_num_floors

        self.state_dtype: Type = int
        self.state_shape: Tuple[int, ...]=(self.floor_map.flatten().shape[0] + 1,)
        self.unflattened_state_shape: Tuple[int, ...]=self.floor_map.shape

        self.current_floor: None|int=None
        self.current_state: None|np.ndarray=None
        self.terminal: bool=True
        return

    def get_current_state(self) -> np.ndarray:
        if self.terminal:
            raise ValueError("Cannot get current state when environment is terminal")
        return self.flatten_state(self.current_state, self.current_floor)

    def get_transition_probability(
            self,
            state: np.ndarray,
            action: int,
            next_state: np.ndarray
    ) -> float:
        pass

    def flatten_state(
            self,
            state: np.ndarray,
            floor: int
    ) -> np.ndarray:
        flattened_state = np.zeros(self.state_shape, dtype=self.state_dtype)
        flattened_state[:self.state_shape[0] - 1] = state.flatten()
        flattened_state[self.state_shape[0]] = floor
        return flattened_state

    def get_start_states(self) -> List[np.ndarray]:
        pass

    def get_success_states(
            self,
            state: np.ndarray,
    ) -> (List[np.ndarray], List[float]):
        pass

    def is_terminal(
            self,
            state: None|np.ndarray=None
    ) -> bool:
        pass

    def step(
            self,
            action: int
    ) -> (np.ndarray, float, bool, Any):
        pass

    def reset(
            self,
            start_state: None|np.ndarray=None,
            seed: None|int=None
    ) -> np.ndarray:
        pass

    def unflatten_state(
            self,
            state: np.ndarray,
    ) -> np.ndarray:
        unflattened_state = np.delete(state, self.state_shape[0])
        unflattened_state = unflattened_state.reshape(self.unflattened_state_shape)
        return unflattened_state
