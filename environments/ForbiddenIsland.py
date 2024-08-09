from environments.environment import Environment


class ForbiddenIsland(Environment):

    possible_actions = [0, 1, 2, 3, #North, South, East, West
                        4, # Shore Up
                        5, # Get Treasure]

    def __init__(self):
        return
