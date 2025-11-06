import copy
import json
import networkx as nx
import numpy as np
import random as rand
from scipy import sparse, stats
from typing import Dict, List, Tuple, Type

from learning_agents.agentbehaviour import AgentBehaviour
from learning_agents.optionsagent import Option, OptionsAgent
from progressbar import print_progress_bar

class PreparednessIncrementalOption(Option):

    def __init__(
            self,
            actions: List['PreparednessIncrementalOption'] | List[int],
            start_state: None | np.ndarray,
            end_states: List[np.ndarray],
            end_nodes: List[str],
            option_level: int,
            alpha: float,
            epsilon: float,
            gamma: float,
            state_transition_graph: nx.MultiDiGraph,
    ):
        self.actions: List['PreparednessIncrementalOption'] | List[int] = actions
        self.start_state: None|np.ndarray = start_state
        self.end_states: List[np.ndarray] = end_states
        self.end_nodes: List[str] = end_nodes
        self.option_level: int = option_level
        self.alpha: float = alpha
        self.epsilon: float = epsilon
        self.gamma: float = gamma
        self.state_transition_graph: nx.MultiDiGraph = state_transition_graph

        self.primitive_option: bool = self.option_level > 0
        self.option_lookup: Dict[str, Option] = {}
        if self.option_level > 1:
            self.option_lookup = {str(option): option for option in self.actions}

        self.q_values: Dict[str, float] = {}

        self.current_option_step: int = 0
        self.current_option: Option|None = None
        self.last_possible_actions: List[int] = []
        return

    def can_initiate(
            self,
            state: np.ndarray
    ) -> bool:
        if self.primitive_option:
            return True

        if self.start_state is not None:
            return np.array_equal(state, self.start_state)

        return self.initiated(state)

    def choose_action(
            self,
            state: np.ndarray,
            optimal_choice: bool=True,
            possible_actions: None|List[int] = None,
    ) -> int:
        if self.primitive_option:
            return self.actions[0]

        if possible_actions is None:
            possible_actions = self.actions

        if self.option_level == 1:
            if (not optimal_choice) and (rand.uniform(0, 1) <= self.epsilon):
                return rand.choice(possible_actions)

            action_values = self.get_action_values(state)
            max_action_value: float = -np.inf
            chosen_actions: List[int] = []

            for action in possible_actions:
                if action_values[str(action)] > max_action_value:
                    max_action_value = action_values[str(action)]
                    chosen_actions = [action]
                elif action_values[str(action)] == max_action_value:
                    chosen_actions.append(action)
            return rand.choice(chosen_actions)


        if self.current_option is None:
            self.current_option_step = 1
            self.current_option = self.choose_option(state, optimal_choice, possible_actions)

        action = self.current_option.choose_action(state, optimal_choice, possible_actions)
        return action

    def choose_option(
            self,
            state: np.ndarray,
            optimal_choice: bool=True,
            possible_actions: None|List[int] = None,
    ) -> Option:
        if possible_actions is None:
            possible_actions = self.actions

        option_values = self.get_option_values(state)
        chosen_option_str: str
        if (not optimal_choice) and (rand.uniform(0, 1) < self.epsilon):
            chosen_option_str = rand.choice(list(option_values.keys()))
        else:
            chosen_options: List[str] = []
            max_option_value: float = -np.inf
            for option in option_values:
                if option_values[option] > max_option_value:
                    chosen_options = [option]
                    max_option_value = option_values[option]
                elif option_values[option] == max_option_value:
                    chosen_options.append(option)
            chosen_option_str = rand.choice(chosen_options)

        chosen_option = self.option_lookup[chosen_option_str]
        return chosen_option

    def get_action_values(
            self,
            state: np.ndarray,
    ) -> Dict[str, float]:
        state = self.state_to_state_str(state)

        action_values: Dict[str, float] = {}
        state_action_tuple: str
        state_action_value: float
        for action in self.actions:
            state_action_tuple = str((state, str(action)))
            try:
                state_action_value = self.q_values[state_action_tuple]
            except KeyError:
                self.q_values[state_action_tuple] = 0.0
                state_action_value = 0.0
            action_values[str(action)] = state_action_value
        return action_values

    def get_option_values(
            self,
            state: np.ndarray,
    ) -> Dict[str, float]:
        possible_options: List[Option] = [
            option for option in self.actions if option.can_initiate(state)
        ]

        option_values: Dict[str, float] = {}
        state_option_tuple: str
        state_option_value: float
        for option in possible_options:
            state_option_tuple = str((state, str(option)))
            try:
                state_option_value = self.q_values[state_option_tuple]
            except KeyError:
                self.q_values[state_option_tuple] = 0.0
                state_option_value = 0.0
            option_values[str(option)] = state_option_value
        return option_values


    def initiated(
            self,
            state: np.ndarray
    ) -> bool:
        if self.primitive_option:
            return True

        for end_state in self.end_states:
            if np.array_equal(state, end_state):
                return False

        state_node: str
        try:
            state_node = self.state_to_node(state)
        except AttributeError:
            return False

        for end_node in self.end_nodes:
            if nx.has_path(self.state_transition_graph, state_node, end_node):
                return True

        return False

    def learn(
            self,
            state: np.ndarray,
            action: int,
            reward: float,
            next_state: np.ndarray,
            terminal: bool,
            next_state_possible_actions: None|List[int]=None
    ):
        if self.primitive_option:
            return

        # Increment current option step
        # Check if current option is terminated
        # Reset current option
        # CAN WORK OUT REWARD INTERNALLY
        pass

    def state_to_node(
            self,
            state: np.ndarray
    ) -> str:
        state_str = self.state_to_state_str(state)

        for node in self.state_transition_graph.nodes(data=True):
            node_state = self.state_transition_graph.nodes[node]['state']
            if node_state == state_str:
                return node

        raise AttributeError("Node not in STG")

    @staticmethod
    def state_to_state_str(
            state: np.ndarray,
    ) -> str:
        return np.array2string(state)

    def __str__(
            self
    ) -> str:
        if self.primitive_option:
            return str(self.actions[0])

        start_state_str: str = 'None'
        if self.start_state is not None:
            start_state_str = self.state_to_state_str(self.start_state)

        end_nodes_str: str = str(
            [end_node for end_node in self.end_nodes]
        )

        option_str: str = str(
            (self.option_level, start_state_str, end_nodes_str)
        )
        return option_str

    def terminated(
            self,
            state: np.ndarray
    ) -> bool:
        if self.primitive_option:
            return True

        return not self.initiated(state)

    def update_options(
            self,
            new_options: List[Option]
    ):
        self.actions = new_options
        self.option_lookup = {str(new_option): new_option for new_option in new_options}
        return

class PreparednessIncremental(OptionsAgent):

    skill_training_failure_reward: float = -1.0
    skill_training_step_reward: float = -0.01
    skill_training_success_reward: float = 1.0

    def __init__(
            self,
            actions: List[int],
            alpha: float,
            epsilon: float,
            gamma: float,
            max_hierarchy_height: int,
            state_dtype: Type,
            state_shape: Tuple[int, ...],
            graph_save_paths_prefix: str,
    ):
        self.actions: List[int] = actions
        self.alpha: float = alpha
        self.epsilon: float = epsilon
        self.gamma: float = gamma
        self.max_hierarchy_height: int = max_hierarchy_height
        self.state_dtype: Type = state_dtype
        self.state_shape: Tuple[int, ...] = state_shape
        self.graph_save_paths_prefix: str = graph_save_paths_prefix

        self.subgoals: Dict[int, List[str]] = {}
        self.subgoals_list: List[str] = []

        self.state_transition_graph: nx.DiGraph = nx.DiGraph()
        self.adjacency_matrix: None|sparse.spmatrix = None
        self.distance_matrix: None|sparse.csr_matrix = None
        self.num_nodes: int = 0
        self.stg_values: Dict[str, Dict[str, str|float]] = {}
        self.subgoal_graph: nx.DiGraph = nx.DiGraph()
        self.state_node_lookup: Dict[str, str] = {}
        self.node_state_lookup: Dict[str, str] = {}
        # node -> next_node -> num observations
        self.total_transitions: Dict[str, Dict[str, int]] = {}
        # {state, next_state, n} -> P(S_{t + n} = next_state | S_{t} = state) actions taken randomly
        self.random_transition_prob: Dict[str, float] = {}

        self.state_option_values: Dict[str, Dict[str, float]] = {}

        self.behaviour: AgentBehaviour = AgentBehaviour.EXPLORE
        return

    def assign_subgoals(
            self,
            unassigned_subgoals: Dict[int, List[int]],
            max_subgoal_height: int,
            key: str
    ) -> Dict[int, List[str]]:
        subgoal_height_found: bool
        subgoal_height_value: str
        height: int
        subgoals = {i: [] for i in range(1, max_subgoal_height + 1)}
        for node in self.state_transition_graph.nodes():
            subgoal_height_value = "None"

            subgoal_height_found = False
            height = max_subgoal_height
            while height > 0:
                if int(node) in unassigned_subgoals[height]:
                    subgoal_height_found = True
                    break
                height -= 1

            if subgoal_height_found:
                subgoals[height].append(node)
                subgoal_height_value = str(height)

            self.stg_values[node][key] = subgoal_height_value
        return subgoals

    def choose_action(
            self,
            state: np.ndarray,
            optimal_choice: bool=False,
            possible_actions: None | List[int] = None
    ) -> int:
        if possible_actions is None:
            possible_actions = self.actions

        if self.behaviour == AgentBehaviour.EXPLORE:
            action = rand.choice(possible_actions)
            return action
        pass

    def compute_preparedness(
            self,
            hops: int,
            use_existing_values: bool,
            verbose: bool=False
    ):
        frequency_entropy: None| float
        neighbourhood_entropy: None| float
        preparedness: None| float

        for node in self.stg_values:
            if verbose:
                print_progress_bar(
                    int(node),
                    self.adjacency_matrix.shape[0],
                    "Computing Preparedness node " + str(node),
                    " Complete"
                )

            frequency_entropy = None
            neighbourhood_entropy = None
            preparedness = None

            if use_existing_values:
                try:
                    frequency_entropy = self.stg_values[node][self.frequency_entropy_key(hops)]
                except KeyError:
                    ()
                try:
                    neighbourhood_entropy = self.stg_values[node][self.neighbourhood_entropy_key(hops)]
                except KeyError:
                    ()
                try:
                    preparedness = self.stg_values[node][self.preparedness_key(hops)]
                except KeyError:
                    ()

            if frequency_entropy is None:
                frequency_entropy = self.frequency_entropy(int(node), hops)
                self.stg_values[str(node)][self.frequency_entropy_key(hops)] = frequency_entropy
            if neighbourhood_entropy is None:
                neighbourhood_entropy = self.neighbourhood_entropy(int(node), hops)
                self.stg_values[str(node)][self.neighbourhood_entropy_key(hops)] = neighbourhood_entropy
            if preparedness is None:
                preparedness = frequency_entropy + neighbourhood_entropy
                self.stg_values[str(node)][self.preparedness_key(hops)] = preparedness

        return

    def copy_agent(
            self,
            copy_from: 'PreparednessIncremental'
    ):
        pass

    def create_options(
            self,
    ):
        subgoal_graph_distances: Dict[str, Dict[str, int]] = nx.floyd_warshall(self.subgoal_graph)

        max_option_level: float|int = -np.inf
        for start_node in self.subgoal_graph.nodes():
            for end_node in self.subgoal_graph.nodes():
                distance = subgoal_graph_distances[start_node][end_node]
                if distance >= np.inf:
                    continue
                if distance > max_option_level:
                    max_option_level = int(distance)

        new_options_between_subgoals: Dict[str, List[Option]] = {
            str(i): [] for i in range(1, max_option_level + 1)
        }

        # Options Between Subgoals


        pass

    def create_subgoal_graph(
            self
    ):
        h_low: int
        h_high: int
        connection_found: bool
        connecting_subgoals: List[str]

        max_height: int = max(self.subgoals.keys())
        self.subgoal_graph = nx.MultiDiGraph()

        for height in self.subgoals:
            for start_subgoal in self.subgoals[height]:
                connecting_subgoals = []
                if not self.subgoal_graph.has_node(start_subgoal):
                    self.subgoal_graph.add_node(start_subgoal)

                # Connections to subgoals of lower heights
                h_low = height - 1
                connection_found = False
                while (h_low >= 1) and (not connection_found):
                    for end_subgoal in self.subgoals[h_low]:
                        if nx.has_path(
                                self.state_transition_graph,
                                start_subgoal,
                                end_subgoal
                        ):
                            connection_found = True
                            connecting_subgoals.append(end_subgoal)
                    h_low -= 1

                # Connections to subgoals of higher heights
                h_high = height + 1
                connection_found = False
                while (h_high <= max_height) and (not connection_found):
                    for end_subgoal in self.subgoals[h_high]:
                        if nx.has_path(
                                self.state_transition_graph,
                                start_subgoal,
                                end_subgoal
                        ):
                            connection_found = True
                            connecting_subgoals.append(end_subgoal)
                    h_high += 1

                # Adding connections to subgoal graph
                for end_subgoal in connecting_subgoals:
                    if not self.subgoal_graph.has_node(end_subgoal):
                        self.subgoal_graph.add_node(end_subgoal)
                    self.subgoal_graph.add_edge(
                        start_subgoal,
                        end_subgoal
                    )

        nx.set_node_attributes(self.subgoal_graph, self.stg_values)
        return

    def find_local_maxima(
            self,
            key: str,
            hops: int
    ) -> List[int]:
        local_maxima: List[int] = []
        local_maxima_key: str = key + "-local-maxima"
        is_subgoal: str
        node_value: float

        for node in self.state_transition_graph.nodes():
            is_subgoal = 'True'

            out_neighbourhood = self.get_out_neighbourhood(int(node), hops)
            in_neighbourhood = self.get_in_neighbourhood(int(node), hops)

            if (len(out_neighbourhood) <= 0) or (len(in_neighbourhood) <= 0):
                self.stg_values[node][local_maxima_key] = 'False'
                continue

            node_value = self.stg_values[node][key]

            for neighbour in out_neighbourhood + in_neighbourhood:
                if int(node) == int(neighbour):
                    continue
                neighbour_value = self.stg_values[str(neighbour)][key]
                if node_value <= neighbour_value:
                    is_subgoal = 'False'
                    break

            self.stg_values[node][local_maxima_key] = is_subgoal

            if is_subgoal == 'True':
                local_maxima.append(int(node))

        return local_maxima

    def find_preparedness_subgoals(
            self,
            find_frequency_entropy_subgoals: bool=False,
            find_neighbourhood_entropy_subgoals: bool=False,
            use_existing_values: bool=False,
            verbose: bool=False
    ) -> bool:
        subgoals: Dict[int, List[int]] = {}
        frequency_entropy_subgoals: Dict[int, List[int]] = {}
        neighbourhood_entropy_subgoals: Dict[int, List[int]] = {}
        preparedness_subgoals_found: bool = False
        frequency_entropy_subgoals_found: bool = not find_frequency_entropy_subgoals
        neighbourhood_entropy_subgoals_found: bool = not find_neighbourhood_entropy_subgoals
        max_subgoal_height: int = self.max_hierarchy_height
        max_frequency_entropy_height: int = self.max_hierarchy_height
        max_neighbourhood_entropy_height: int = self.max_hierarchy_height

        for hops in range(1, self.max_hierarchy_height + 1):
            if verbose:
                print("Finding Subgoals for height " + str(hops))
                print("     Computing Preparedness at height " + str(hops))

            self.compute_preparedness(hops, use_existing_values, verbose)

            if not preparedness_subgoals_found:
                if verbose:
                    print("     Finding preparedness local maxima at height " + str(hops))
                subgoals[hops] = copy.copy(self.find_local_maxima(self.preparedness_key(hops), hops))

            if not frequency_entropy_subgoals_found:
                if verbose:
                    print("     Finding frequency entropy local maxima at height " + str(hops))
                frequency_entropy_subgoals[hops] = copy.copy(
                    self.find_local_maxima(self.frequency_entropy_key(hops), hops)
                )
            if not neighbourhood_entropy_subgoals_found:
                if verbose:
                    print("     Finding neighbourhood entropy local maxima at height " + str(hops))
                neighbourhood_entropy_subgoals[hops] = copy.copy(
                    self.find_local_maxima(self.neighbourhood_entropy_key(hops), hops)
                )

            if hops > 1:
                if not preparedness_subgoals_found:
                    if subgoals[hops] == subgoals[hops - 1]:
                        preparedness_subgoals_found = True
                        max_subgoal_height = hops
                    elif subgoals[hops] == []:
                        preparedness_subgoals_found = True
                        max_subgoal_height = hops - 1
                    if preparedness_subgoals_found:
                        if verbose:
                            print("Preparedness subgoals found")
                if not frequency_entropy_subgoals_found:
                    if frequency_entropy_subgoals[hops] == frequency_entropy_subgoals[hops - 1]:
                        frequency_entropy_subgoals_found = True
                        max_frequency_entropy_height = hops
                    elif frequency_entropy_subgoals[hops] == []:
                        frequency_entropy_subgoals_found = True
                        max_frequency_entropy_height = hops - 1
                    if frequency_entropy_subgoals_found:
                        if verbose:
                            print("Frequency entropy subgoals found")
                if not neighbourhood_entropy_subgoals_found:
                    if neighbourhood_entropy_subgoals[hops] == neighbourhood_entropy_subgoals[hops - 1]:
                        neighbourhood_entropy_subgoals_found = True
                        max_neighbourhood_entropy_height = hops
                    elif neighbourhood_entropy_subgoals[hops] == []:
                        neighbourhood_entropy_subgoals_found = True
                        max_neighbourhood_entropy_height = hops - 1
                    if neighbourhood_entropy_subgoals_found:
                        if verbose:
                            print("Neighbourhood entropy subgoals found")


            if (
                    preparedness_subgoals_found and
                    frequency_entropy_subgoals_found and
                    neighbourhood_entropy_subgoals_found
            ):
                break

        if not preparedness_subgoals_found and verbose:
            print("Preparedness Subgoal Height maxed-out")
        if not frequency_entropy_subgoals_found and verbose:
            print("Frequency Entropy Subgoal Height maxed-out")
        if not neighbourhood_entropy_subgoals_found and verbose:
            print("Neighbourhood Entropy Subgoal Height maxed-out")

        self.subgoals = self.assign_subgoals(
            subgoals,
            max_subgoal_height,
            'preparedness-subgoal-height'
        )
        self.subgoals_list = []
        for key in self.subgoals:
            self.subgoals_list += self.subgoals[key]

        if find_frequency_entropy_subgoals:
            _ = self.assign_subgoals(
                frequency_entropy_subgoals,
                max_frequency_entropy_height,
                'frequency-entropy-subgoal-height'
            )
        if find_neighbourhood_entropy_subgoals:
            _ = self.assign_subgoals(
                neighbourhood_entropy_subgoals,
                max_neighbourhood_entropy_height,
                'neighbourhood-entropy-subgoal-height'
            )

        nx.set_node_attributes(self.state_transition_graph, self.stg_values)

        return preparedness_subgoals_found

    def frequency_entropy(
            self,
            node: int,
            hops: int
    ) -> float:
        prob_values: List[float] = []
        out_neighbourhood = self.get_out_neighbourhood(node, hops)
        for neighbour in out_neighbourhood:
            prob_values.append(self.get_random_transition_prob(node, neighbour, hops))

        if sum(prob_values) <= 0:
            return 0.0
        frequency_entropy = stats.entropy(prob_values, base=2)
        return frequency_entropy

    @staticmethod
    def frequency_entropy_key(
            hops: int
    ) -> str:
        return str(hops) + "-frequency-entropy"

    def get_in_neighbourhood(
            self,
            node: int,
            hops: int
    ) -> List[int]:
        in_neighbourhood: List[int] = [
            j for j in range(self.distance_matrix.shape[0]) if 0 < self.distance_matrix[j, node] <= hops
        ]
        return in_neighbourhood

    def get_out_neighbourhood(
            self,
            node: int,
            hops: int
    ) -> List[int]:
        out_neighbourhood: List[int] = [
            j for j in range(self.distance_matrix.shape[0]) if 0 < self.distance_matrix[node, j] <= hops
        ]
        return out_neighbourhood

    def get_random_transition_prob(
            self,
            start_node: int,
            end_node: int,
            hops: int
    ) -> float:
        if hops <= 0:
            if start_node == end_node:
                return 1.0
            else:
                return 0.0
        elif hops == 1:
            return self.adjacency_matrix[start_node, end_node]

        try:
            trans_prob = self.random_transition_prob[str((start_node, end_node, hops))]
            return trans_prob
        except KeyError:
            ()

        out_neighbourhood = self.get_out_neighbourhood(start_node, 1)
        trans_prob = 0.0
        for out_neighbour in out_neighbourhood:
            trans_prob += (
                    self.adjacency_matrix[start_node, out_neighbour] *
                    self.get_random_transition_prob(out_neighbour, end_node, hops - 1)
            )

        self.random_transition_prob[str((start_node, end_node, hops))] = trans_prob
        return trans_prob

    def learn(
            self,
            state: np.ndarray,
            action: int,
            reward: float,
            next_state: np.ndarray,
            terminal: bool|None = None,
            next_state_possible_actions: List[int]|None = None
    ):
        if self.behaviour == AgentBehaviour.EXPLORE:
            self.update_state_transition_graph(state, next_state)
            return
        pass

    def load(
            self,
            save_path: str
    ):
        with open(save_path, 'r') as f:
            agent_data = json.load(f)

        self.num_nodes = agent_data['num_nodes']
        self.stg_values = agent_data['stg_values']
        self.state_node_lookup = agent_data['state_node_lookup']
        self.node_state_lookup = agent_data['node_state_lookup']
        self.total_transitions = agent_data['total_transitions']

        self.state_transition_graph = nx.read_gexf(agent_data['state_transition_graph_save_path'])
        self.subgoal_graph = nx.read_gexf(agent_data['subgoal_graph_save_path'])
        nx.set_node_attributes(self.state_transition_graph, self.stg_values)
        nx.set_node_attributes(self.subgoal_graph, self.stg_values)
        return

    @staticmethod
    def preparedness_key(
            hops: int
    ) -> str:
        return str(hops) + "-preparedness"

    def neighbourhood_entropy(
            self,
            node: int,
            hops: int
    ) -> float:
        out_neighbourhood = self.get_out_neighbourhood(node, hops)
        if node not in out_neighbourhood:
            out_neighbourhood.append(node)
        probs: List[float] = []

        for end_node in out_neighbourhood:
            transition_prob: float = 0.0
            for start_node in out_neighbourhood:
                transition_prob += self.get_random_transition_prob(start_node, end_node, hops)
            probs.append(transition_prob)

        prob_sum: float = sum(probs)
        if prob_sum <= 0.0:
            return 0.0

        probs = [prob / prob_sum for prob in probs]

        neighbourhood_entropy = stats.entropy(probs, base=2)
        return neighbourhood_entropy

    @staticmethod
    def neighbourhood_entropy_key(
            hops: int
    ) -> str:
        return str(hops) + "-neighbourhood-entropy"

    def reset_random_transition_probs(
            self
    ):
        self.random_transition_prob = {}
        return

    def save(
            self,
            save_path: str,
    ):
        nx.set_node_attributes(self.state_transition_graph, self.stg_values)
        nx.set_node_attributes(self.subgoal_graph, self.stg_values)

        stg_save_path: str = self.graph_save_paths_prefix + "_state_transition_graph.gexf"
        subgoal_graph_save_path: str = self.graph_save_paths_prefix + "_subgoal_graph.gexf"

        agent_save_file = {
            'state_transition_graph_save_path': stg_save_path,
            'num_nodes': self.num_nodes,
            'stg_values': self.stg_values,
            'subgoal_graph_save_path': subgoal_graph_save_path,
            'state_node_lookup': self.state_node_lookup,
            'node_state_lookup': self.node_state_lookup,
            'total_transitions': self.total_transitions,
            'random_transition_prob': self.random_transition_prob,
            'subgoals': self.subgoals,
        }

        with open(save_path, 'w') as f:
            json.dump(agent_save_file, f)

        nx.write_gexf(self.state_transition_graph, stg_save_path)
        nx.write_gexf(self.subgoal_graph, subgoal_graph_save_path)
        return

    def set_behaviour(
            self,
            behaviour: AgentBehaviour
    ):
        self.behaviour = behaviour

        if behaviour == AgentBehaviour.TRAIN_SKILLS:
            # Update distance matrix
            self.update_distance_matrix()
            # Reset Transition Probs
            self.reset_random_transition_probs()
            # Run Preparedness + Find Subgoals
            subgoals_found: bool = self.find_preparedness_subgoals(
                False,
                False,
                False,
                False
            )

            # If no Subgoals found, go back to exploring
            if not subgoals_found:
                self.behaviour = AgentBehaviour.EXPLORE
                return

            # Create Subgoal Graph
            self.create_subgoal_graph()

            # Find Skills

            # Setting behaviour back to exploring for testing
            self.behaviour = AgentBehaviour.EXPLORE

        return

    def state_to_node(
            self,
            state: np.ndarray
    ) -> str:
        state_str = self.state_to_state_str(state)
        try:
            state_node = self.state_node_lookup[state_str]
        except KeyError:
            state_node = str(self.num_nodes)
            self.state_node_lookup[state_str] = state_node
            self.num_nodes += 1
        return state_node

    def update_distance_matrix(
            self
    ):
        self.adjacency_matrix = nx.to_scipy_sparse_array(
            self.state_transition_graph
        )


        self.distance_matrix = sparse.csgraph.dijkstra(
            self.adjacency_matrix,
            True,
            unweighted=True,
            limit=self.max_hierarchy_height
        )
        return

    def update_state_transition_graph(
            self,
            state: np.ndarray,
            next_state: np.ndarray
    ):
        state_str: str = self.state_to_state_str(state)
        next_state_str: str = self.state_to_state_str(next_state)
        state_node: str = self.state_to_node(state)
        next_state_node: str = self.state_to_node(next_state)

        if not self.state_transition_graph.has_node(state_node):
            self.state_transition_graph.add_node(state_node, state=state_str)
            self.total_transitions[state_str] = {}
            self.stg_values[state_node] = {'state': state_str}

        if not self.state_transition_graph.has_node(next_state_node):
            self.state_transition_graph.add_node(next_state_node, state=next_state_str)
            self.total_transitions[next_state_str] = {}
            self.stg_values[next_state_node] = {'state': next_state_str}

        if not self.state_transition_graph.has_edge(state_node, next_state_node):
            self.state_transition_graph.add_edge(state_node, next_state_node, weight=1.0)
            self.total_transitions[state_str][next_state_str] = 1
        else:
            self.total_transitions[state_str][next_state_str] += 1

        self.update_state_transition_graph_weights(state_node)
        return

    def update_state_transition_graph_weights(
            self,
            node: str
    ):
        start_state_str: str = self.state_transition_graph.nodes[node]['state']
        total_out_transitions: int = sum(self.total_transitions[start_state_str].values())
        for end_node in self.state_transition_graph.neighbors(node):
            end_state_str: str = self.state_transition_graph.nodes[end_node]['state']
            weight: float = self.total_transitions[start_state_str][end_state_str] / total_out_transitions
            self.state_transition_graph.edges[node, end_node]['weight'] = weight
        return
