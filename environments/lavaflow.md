# Lavaflow Environment Specification

A 2D gridworld environment consisting of an $n \times n$ maze.
Lava spreads to adjacent squares each timestep. The agent needs to place blocks in order to protect the
largest amount of area from lava.

<img src="environment_images/lavaflowenvironmentexample.png" alt="Lavaflow Environment Example" width="300"/>

| Parameter        | Type                                  | Description                                                                     |
|------------------|---------------------------------------|---------------------------------------------------------------------------------|
| Grid Size: $n$   | $\mathbb{N}$                          | Width/height of the maze.                                                       |
| Maze Layout: $M$ | $\left\{0, 1, 2\right\}^{n \times n}$ | The initial layout of the maze,<br/> describes where the blocks and lava start. |

| Property                | Upper Bound                                                                                                               |
|-------------------------|---------------------------------------------------------------------------------------------------------------------------|
| $\vert\mathcal{S}\vert$ | $2\left(3^{{\left(k - 1\right)}^{2}} + k\right)$ <br/>Where $k$ is the number of empty squares in the initial grid layout |
| $\vert\mathcal{A}\vert$ | $9$                                                                                                                       |

| Feature       | Value     |
|---------------|-----------|
| Deterministic | Yes       |
| Directed      | Partially |
| Continual     | No        |

## State Space

**Type:** $\left\{0, 1, 2, 3\right\}^{2}$

**Upper Bound:** $2\left(3^{{\left(k - 1\right)}^{2}} + k\right)$
<br/>Where $k$ is the number of empty squares in the initial grid layout

A state is a matrix where the $(i + 1, j + 1)$ index shows the value of the $(i, j)$ square in the grid.
Whether the square is empty, a block, lava, or holds the agent. The $(0, 0)$ entry shows a $1$ if the state is terminal and a $1$ otherwise
