from typing import Any

from environments.environment import Environment

'''
Action Index
0: Look at random object
1: Move Hand to Eye
2: Move Marker to Eye
3: Move Eye to Hand
4: Move Eye to Marker
5: Interact with object under marker

State representation: (Eye Item, Hand Item, Marker Item, Light, Music, Bell)
'''


class Playroom(Environment):

    possible_actions = [0, 1, 2, 3, 4, 5]

    def __int__(self):
        self.current_state = None
        self.environment_name = 'playroom'
        self.terminal = True
        return

    def get_current_state(self):
        return

    def get_start_states(self):
        return

    def is_terminal(self):
        return

    def reset(self) -> Any:
        return

    def step(self, action) -> (Any, float, bool, Any):
        return
