# Tinytown Environment Specification

A discrete environment based on an $n \times m$ grid.
Each timestep the agent places a resource onto an empty grid square.
The environment alternates between a _resource phase_ where the agent places a resource on an empty square,
and a _building phase_ where the agent converts groups of resources into a single building.

Based on the board game [Tiny Towns](https://www.petermcpherson.com/games/tiny-towns) by Peter McPherson.

<img src="environment_images/tinytown.png" alt="Tinytown Environment Example" width="300"/>

| Parameter          | Type         | Description                                                                                                                  |
|--------------------|--------------|------------------------------------------------------------------------------------------------------------------------------|
| Grid Height: $n$   | $\mathbb{N}$ | Height of the town grid.                                                                                                     |
| Grid Length: $m$   | $\mathbb{N}$ | Length of the town grid.                                                                                                     |
| Deterministic: $k$ | Boolean      | If true, each timestep the agent chooses which resource to place.<br/>If false, the resource is determined uniform randomly. |

| Property                | Value                                             | Upper Bound                                                                        |
|-------------------------|---------------------------------------------------|------------------------------------------------------------------------------------|
| $\vert\mathcal{S}\vert$ | ~                                                 | If deterministic: $2\times{{5}^{nm}} - 1$.<br/>Otherwise: $3\times{{5}^{nm}} - 1$. |
| $\vert\mathcal{A}\vert$ | $2{\left(nm\right)}^{2} + 2{\left(nm\right)} + 1$ | ~                                                                                  |

| Feature                            | Value  |
|------------------------------------|--------|
| Deterministic                      | Yes/No |
| Directed                           | Yes    |
| Continual                          | No     |
| All Actions Possible in all States | No     |

## State Space

**Type:** Matrix of size $\left(n + 1\right) \times \left(m + 1\right)$
containing values from $\left\lbrace0, 1, 2, 3, 4\right\rbrace$.
More formally: ${\left\lbrace0, 1, 2, 3, 4\right\rbrace}^{m + 1 \times n + 1}$.

**Upper bound:** If deterministic: $2\times{{5}^{nm}} - 1$, otherwise: $3\times{{5}^{nm}} - 1$.

A state is a $n + 1 \times m + 1$ matrix where the $\left(i, j\right)$ index shows the
resource or building contained in the $\left(i, j\right)$ square.
The value is $0$ if the grid square is empty. The value at index $\left(n, m\right)$ is $0$
if the state is in the _resource phase_, or $1$ if the state is in the _building phase_.

## Action Space

**Type:** A vector of length $6$ containing natural numbers ($\mathbb{N}^{6}$).

Actions correspond to either placing a specific resource at a grid location,
converting a group of resources centered on a location into a building at another location,
or ending the building phase.

| Action                                              | Description                                                                                                                                                                         | Conditions                                                                                                                                                                                                                                 |
|-----------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| $\left(i, j, 0, \cdot, \cdot, 0\right)$             | Where $i < n$ and $j < m$.<br/> Place a _brick_ at grid square $\left(i, j\right)$.                                                                                                 | The square at $\left(i, j\right)$ must be empty<br/>and the state is in the resource phase.<br/>If not deterministic, the next resource to place must be a brick.                                                                          |
| $\left(i, j, 1, \cdot, \cdot, 0\right)$             | Where $i < n$ and $j < m$.<br/> Place _glass_ at grid square $\left(i, j\right)$.                                                                                                   | The square at $\left(i, j\right)$ must be empty<br/>and the state is in the resource phase.<br/>If not deterministic, the next resource to place must be glass.                                                                            |
| $\left(i, j, 0, \tilde{i}, \tilde{j}, 1\right)$     | Where ${i, \tilde{i} < n}$ and ${j, \tilde{j} < m}$.<br/>Use the resource pattern centered at $\left(i, j\right)$<br/>to place a cottage at $\left(\tilde{i}, \tilde{j}\right)$.    | The squares from $\left(i, j\right)$ to $\left(i + 1, j + 1\right)$<br/>contain the cottage resource pattern,<br/>${i - 1 \leq \tilde{i}\leq i + 1}$,<br/>$j - 1 \leq \tilde{j}\leq j + 1$,<br/>and the state is in the building phase.    |
| $\left(i, j, 1, \tilde{i}, \tilde{j}, 1\right)$     | Where ${i, \tilde{i} < n}$ and ${j, \tilde{j} < m}$.<br/>Use the resource pattern centered at $\left(i, j\right)$<br/>to place a greenhouse at $\left(\tilde{i}, \tilde{j}\right)$. | The squares from $\left(i, j\right)$ to $\left(i + 1, j + 1\right)$<br/>contain the greenhouse resource pattern,<br/>${i - 1 \leq \tilde{i}\leq i + 1}$,<br/>$j - 1 \leq \tilde{j}\leq j + 1$,<br/>and the state is in the building phase. |
| $\left(\cdot, \cdot, \cdot, \cdot, \cdot, 2\right)$ | End the building phase.                                                                                                                                                             | The state must be in the building phase.                                                                                                                                                                                                   |

## Transition Dynamics
In tinytown there are two resource types and two building types. Each building has a corresponding resource pattern.

| Index | Resource | Image                                                                                                 |
|-------|----------|-------------------------------------------------------------------------------------------------------|
| 0     | Brick    | <img src="environment_images/tinytown-brick-resource.png" alt="Tinytown Brick Resource" width="25"/>  |
| 1     | Glass    | <img src="environment_images/tinytown-glass-resource.png" alt="Tinytown Glass Resource" width="25"/>. |

| Index | Building   | Image                                                                                        | Resource Pattern                                                                                             |
|-------|------------|----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| 0     | Cottage    | <img src="environment_images/tinytown-cottage.png" alt="Tinytown Cottage" width="75"/>       | <img src="environment_images/tinytown-cottage-pattern.png" alt="Tinytown Cottage Pattern" width="75"/>       |
| 1     | Greenhouse | <img src="environment_images/tinytown-greenhouse.png" alt="Tinytown Greenhouse" width="75"/> | <img src="environment_images/tinytown-greenhouse-pattern.png" alt="Tinytown Greenhouse Pattern" width="75"/> |

The environment start state is an entirely empty grid in the resource phase.
During the resource phase, the agent places one resource in an empty grid square.
If the environment is deterministic, the agent can place any resource on any empty square in the grid.
Otherwise, the state specifies which resource is to be placed and the agent just chooses where to place it.
The resource to be placed is chosen uniform randomly.
After placing a resource, the environment transitions to the building phase.

In the building phase, the agent can place buildings on the grid. If the matching building pattern is on the grid,
the agent can take an action to remove all resources from this pattern and place the corresponding building in one of
the squares of the pattern. The agent can keep taking actions to place buildings (if there are corresponding patterns).
Alternatively, the agent can take the end building phase action to end the building phase and return to the resource phase.

The environment terminates when the agent takes the end building phase action in a state where every square is filled.

## Reward
No reward is given until the agent transitions to a terminal state.
When transitioning to a terminal state, the agent receives a reward of
$3\min{\left\lbrace c, 4g \right\rbrace} - nm + c + g$, 
where $c$ and $g$ are the number of cottages and greenhouses in the terminal state respectively.

This encourages the agent to construct as many buildings as possible,
building a greenhouse for every $4$ cottages.
