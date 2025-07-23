# Tinytown Environment Specification

A discrete environment based on an $n \times m$ grid.
Each timestep the agent places resources onto empty grid squares,
if these resources are in the correct patterns, they can be turned into buildings.
The environment alternates between a _resource phase_ where the agent places a resource on an empty square,
and a _building phase_ where the agent converts groups of respurces into a single building.

Based on the board game [Tiny Towns](https://www.petermcpherson.com/games/tiny-towns) by Peter McPherson.

<img src="environment_images/tinytown.png" alt="Tinytown Environment Example" width="300"/>

| Parameter        | Type         | Description              |
|------------------|--------------|--------------------------|
| Grid Height: $n$ | $\mathbb{N}$ | Height of the town grid. |
| Grid Length: $m$ | $\mathbb{N}$ | Length of the town grid. |

| Property                | Value                                             | Upper Bound             |
|-------------------------|---------------------------------------------------|-------------------------|
| $\vert\mathcal{S}\vert$ | ~                                                 | $2\times{{5}^{nm}} - 1$ |
| $\vert\mathcal{A}\vert$ | $2{\left(nm\right)}^{2} + 2{\left(nm\right)} + 1$ | ~                       |

| Feature                            | Value |
|------------------------------------|-------|
| Deterministic                      | Yes   |
| Directed                           | Yes   |
| Continual                          | No    |
| All Actions Possible in all States | No    |

## State Space

**Type:** Matrix of size $\left(n + 1\right) \times \left(m + 1\right)$
with values from $\left\lbrace0, 1, 2, 3, 4\right\rbrace$.
More formally: ${\left\lbrace0, 1, 2, 3, 4\right\rbrace}^{m + 1 \times n + 1}$.

**Upper bound:** $2\times{{5}^{nm}} - 1$.

A state is a $n + 1 \times m + 1$ matrix where the $\left(i, j\right)$ index shows the
resource or building contained in the $\left(i, j\right)$ square in the town.
The value is $0$ if the grid square is empty. The value at index $\left(n, m\right)$ is $0$ 
if the state is in the _resource phase_, or $1$ if the state is in the _building phase_.

## Action Space

**Type:** $\mathbb{N}$.

Actions correspond to either placing a specific resource at a grid location,
converting a group of resources centered on a location into a building at another location,
and ending the building phase.

| Action Range               | Description                                                          | Conditions                                       |
|----------------------------|----------------------------------------------------------------------|--------------------------------------------------|
| $\left[0, nm\right]$       | For action $ij$, place a _brick_ at grid square $\left(i, j\right)$. | The square at $\left(i, j\right)$ must be empty. |
| $\left[nm + 1, 2nm\right]$ | For action $2ij$                                                     |                                                  |

