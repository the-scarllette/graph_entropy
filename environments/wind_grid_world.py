import random
import time
from typing import Tuple, List

import matplotlib
from matplotlib.patches import Circle
from sys import platform

if platform == "linux" or platform == "linux2":
    # linux
    matplotlib.use('TKAGG')
elif platform == "darwin":
    # OS X
    matplotlib.use('MacOSX')
elif platform == "win32":
    # Windows...
    pass

import matplotlib.pyplot as plt
import numpy as np

class Agent(object):

    def __init__(self, position: np.ndarray):
        self.position = position

class WindGridWorld(object):
    def __init__(self, size: Tuple[int, int, int], num_target_points: int, num_agents: int,
                 agent_camera_visualisation_radius: float, random_spawn: bool, visualise: bool,
                 set_start: bool):
        '''
        :param size: (height, width)
        :param num_target_points:
        '''
        self.size = size
        self.num_target_points = num_target_points
        self.visualise = visualise
        self.num_agents = num_agents
        self.random_spawn = random_spawn
        self.agent_camera_visualisation_radius = agent_camera_visualisation_radius

        if (self.visualise):
            self.fig, self.ax = plt.subplots()

        # grid
        self.visualised_grid: np.ndarray = None
        self.wind_layers: np.ndarray = None     # list of wind_vector_index, each entry is unique
        self.wind_vectors: np.ndarray = np.array([
            [0, 1],     # N
            [1, 1],     # NE
            [1, 0],     # E
            [1, -1],    # SE
            [0, -1],    # S
            [-1, -1],   # SW
            [-1, 0],    # W
            [-1, 1],    # NW
            [0, 0],     # Stationary
        ])
        self.wind_vector_display: np.ndarray = np.array([
            [0.5, 0.25, 0.0, 0.353, 1.],        # N
            [0.25, 0.25, 0.25, 0.25, 1.],       # NE
            [0.25, 0.5, 0.353, 0.0, 1.],         # E
            [0.25, 0.75, 0.2475, -0.2475, 1.],  # SE
            [0.5, 0.75, 0.0, -0.353, 1.],  # S
            [0.75, 0.25, -0.353, 0.353, 1.],  # SW
            [0.75, 0.55, -0.353, 0.0, 1.],  # W
            [0.75, 0.25, -0.353, 0.353, 1.],  # NW
            [0., 0., 0., 0., 0.],               # Stationary
        ])
        self.agents: List[Agent] = None
        return

    def reset(self):
        self.visualised_grid = np.zeros(shape=self.size[:-1]).astype(dtype=float)  # 0 = not visited, 1 = visited
        self.agents = [Agent(position=
                             np.array([random.randint(0, self.size[2] - 1), random.randint(0, self.size[1] - 1), random.randint(0, self.size[0] - 1)]) if self.random_spawn else np.array([0, 0])
                             ) for _ in range(self.num_agents)
                       ]

        # windfield generation
        # 9 different wind-fields
        # random layers, random start location.
        wind_layers = np.zeros(shape=(9,)).astype(dtype=int) - 1
        for wind_vector_index in range(9):
            chosen: bool = False
            while not chosen:
                random_layer = np.random.randint(0, 9)  # [0,8]
                if wind_layers[random_layer] == -1:  # no wind chosen here
                    wind_layers[random_layer] = wind_vector_index
                    chosen = True
        self.wind_layers = wind_layers

        self.agent_visual_update()

        if self.visualise:
            self.plot()

    def agent_visual_update(self):
        for x in range(self.visualised_grid.shape[0]): # y axis    (row)
            for y in range(self.visualised_grid.shape[1]): # x axis    (column)
                for agent in self.agents:
                    if np.linalg.norm(agent.position[:2] - np.array([y, x])) < self.agent_camera_visualisation_radius:
                        self.visualised_grid[x, y] = 1
                    # if np.linalg.norm(agent.position[1:] - np.array([y, x])) < self.agent_camera_visualisation_radius:
                    #     self.visualised_grid[x, y] = 1

    def step(self, move_altitude: int, agent_id: int):
        # perform action
        agent = self.agents[agent_id]
        agent.position[2] = move_altitude

        # update the position based on the windfield
        wind_vector: np.ndarray = self.wind_vectors[self.wind_layers[move_altitude]]
        agent.position[0:2] += wind_vector

        # make sure the agent is inside the bounds of the environment

        agent.position[:2] = np.clip(agent.position[:2], np.zeros(shape=(2,)), np.array([self.size[1], self.size[0]]) - 1)

        # visualise update
        self.agent_visual_update()

        if self.visualise:
            self.plot()

    def plot(self):
        self.ax.clear()
        plt.cla()

        self.ax.imshow(self.visualised_grid, cmap='gray', vmin=0, vmax=1, extent=[0, self.size[1], 0, self.size[0]], origin='lower')
        vec_index = self.wind_layers[self.agents[0].position[2]]
        #altitude = self.agents[0].position[2]
        for x in range(self.visualised_grid.shape[0] - 1):
            for y in range(self.visualised_grid.shape[1] + 1):
                color = 'black' if self.visualised_grid[y, x] else 'white'
                if self.wind_vector_display[vec_index][4]:
                    self.ax.arrow(self.wind_vector_display[vec_index][0] + x, self.wind_vector_display[vec_index][1] + y, *tuple(self.wind_vector_display[vec_index][2:4]), head_width=0.1, head_length=0.15, fc=color, ec=color)

        # todo: need to check this,  draw grid lines - test with x > y and y > x to ensure the line is correct
        for x in range(self.size[1]):
            self.ax.axvline(x=x, color='black', linestyle='-')
        for y in range(self.size[0]):
            self.ax.axhline(y=y, color='black', linestyle='-')

        # Set x and y ticks to match grid size and align with the center of each cell
        self.ax.set_xticks(np.arange(self.size[1]) + 0.5)
        self.ax.set_yticks(np.arange(self.size[0]) + 0.5)
        # Set tick labels to integer values
        self.ax.set_xticklabels(range(self.size[1]))
        self.ax.set_yticklabels(range(self.size[0]))

        # plot the agents (blue)
        for agent in self.agents:
            # 0.5 offset for plotting, since everything is shifted.
            circle = Circle((agent.position[0] + 0.5, agent.position[1] + 0.5), 0.4, color='blue', zorder=2)  # Circle centered at (5.5, 5.5) with a radius of 0.4
            self.ax.add_patch(circle)

        plt.draw()
        plt.pause(0.001)


if __name__ == "__main__":
    windGridWorld = WindGridWorld(size = (9, 8, 3), num_target_points = 0, num_agents=1, agent_camera_visualisation_radius = 1, random_spawn=True, visualise=True)
    windGridWorld.reset()

    while True:
        windGridWorld.plot()
        windGridWorld.step(move_altitude=np.random.randint(0, 9), agent_id=0)
        plt.pause(0.25)
