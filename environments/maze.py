import random as rand

from environments.environment import Environment


class Maze(Environment):
    possible_actions = [0, 1, 2, 3]

    def __init__(self, width, height):
        self.current_state = None

        self.height = height
        self.width = width

        self.maze_squares = []

        self.terminal = True

        self.goal = None
        return

    def get_adjacent_squares(self, square):
        adj_states = []
        for i in range(2):
            for m in [-1, 1]:
                adj = square
                adj[i] += m
                if 0 <= adj[0] <= self.width and 0 <= adj[1] <= self.height:
                    adj_states.append(adj)
        return adj_states

    def get_current_state(self):
        return

    def get_successor_states(self, state):
        successor_states = [successor for successor in self.get_adjacent_squares(state)
                            if successor in self.maze_squares]
        return successor_states

    def generate_maze(self):
        self.maze_squares.append((0, 0))

        walls = self.get_adjacent_squares((0, 0))

        while len(walls) > 0:
            wall_index = rand.randint(0, len(walls) - 1)
            wall = walls.pop(wall_index)
            for cell

        return

    def step(self, action, true_state=False):
        return None, 0.0, False, None

    def reset(self, true_state=False):
        self.generate_maze()
        self.terminal = False
        self.current_state = (0, 0)
        self.goal = rand.choice(self.maze_squares)
        return None
