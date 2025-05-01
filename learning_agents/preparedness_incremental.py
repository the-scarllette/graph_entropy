import copy
import json
import networkx as nx
import numpy as np
import random as rand
import sys

from requests import options
from scipy import sparse
from typing import Callable, Dict, List, Tuple, Type

from environments.environment import Environment
from learning_agents.agentbehaviour import AgentBehaviour
from learning_agents.optionsagent import Option, OptionsAgent
from learning_agents.preparednessagent import PreparednessAgent, PreparednessOption
from learning_agents.qlearningagent import QLearningAgent
from learning_agents.rodagent import RODAgent
from progressbar import print_progress_bar

class PreparednessSkill(Option):

    def __init__(
            self,
            start_state: None|np.ndarray,
            end_state: None|np.ndarray,
            level: str,
            pathing_function: Callable[[np.ndarray, np.ndarray], bool],
            end_states: None|List[np.ndarray]=None
    ):
        self.start_state: None|np.ndarray = start_state
        self.end_state: None|np.ndarray = end_state
        self.level: str = level
        self.pathing_function: Callable[[np.ndarray, np.ndarray], bool] = pathing_function
        self.end_states: None|List[np.ndarray] = end_states

        self.current_skill: None|'PreparednessSKill' = None
        return

    def __eq__(
            self,
            other: 'PreparednessSkill'
    ) -> bool:
        if (self.start_state != other.start_state) and (self.level != other.level):
            return False
        if self.end_state is not None:
            return self.end_states == other.end_state
        for end_state in self.end_states:
            if end_state not in other.end_states:
                return False
        return True

    def initiated(
            self,
            state: np.ndarray
    ) -> bool:
        if self.start_state is not None:
            return np.array_equal(state, self.start_state)
        return not self.terminated(state)

    def reset_skill(
            self
    ):
        if self.current_skill is not None:
            self.current_skill.reset_skill()
            self.current_skill = None
        return

    def set_skill(
            self,
            skill: 'PreparednessSKill'
    ):
        if self.current_skill is not None:
            self.current_skill.reset_skill()
        self.current_skill = skill
        return

    def terminated(
            self,
            state: np.ndarray
    ) -> bool:
        end_states = self.end_states
        if self.end_state is not None:
            end_states = [self.end_state]

        for end_state in end_states:
            if np.array_equal(state, end_state):
                return True
            if self.pathing_function(state, end_state):
                return False
        return True

class SkillTowardsSubgoal(Option):

    def __init__(
            self,
            skill_options: List[Option]|List[int],
            end_state: np.ndarray,
            level: int,
            alpha: float,
            epsilon: float,
            gamma: float,
            state_dtype: Type,
            pathing_function: Callable[[np.ndarray, np.ndarray], bool]
    ):
        self.options = skill_options
        self.end_state = end_state
        self.level = level
        self.pathing_function = pathing_function

        if self.level <= 1:
            self.policy = QLearningAgent(
                self.options,
                alpha,
                epsilon,
                gamma
            )
        else:
            self.policy = OptionsAgent(
                alpha,
                epsilon,
                gamma,
                self.options,
                state_dtype=state_dtype
            )
        return

    def initiated(
            self,
            state: np.ndarray
    ) -> bool:
        if np.array_equal(state, self.end_state):
            return False
        return self.pathing_function(state, self.end_state)

    def terminated(
            self,
            state: np.ndarray
    ) -> bool:
        return not self.initiated(state)

# Skills are dataclasses:
#   Hold the behaviour of the skill
#   What current skill is being followed
# Agent stores skills
#   Onboarding skills
#   Skills between subgoals
#   Skills towards subgoals
# Agent holds Q tables for all skills
# Each skill chooses skills from the agent list of dictionaries
# When adding or removing skills, done so from master list of skills
# When agent is acting:
#   If exploring
#       Choose random action from primitives
#   if training skill:
#       If not following a skill:
#           choose a skill uniform randomly
#       act from skill
#   If training:
#       if not following a skill:
#           choose a skill or primitive action greedily
#       act from chosen skill or action
# When agent is acting from skill:
#       If skill acts over primitives:
#           use intetrnal Q table to find action
#       else:
#           If skill is not following an internal skill:
#               have skill choose an internal skill
#           act accordind to internal skill
class PreparednessIncremental(RODAgent):

    skill_training_failure_reward: float = -1.0
    skill_training_step_reward: float = -0.0001
    skill_training_success_reward: float = 1.0

    def __init__(
            self,
            actions: List[int],
            skill_training_window: int,
            alpha: float,
            epsilon: float,
            gamma: float,
            state_dtype: Type,
            state_shape: Tuple[int, ...],
            max_subgoal_height: int,
            option_onboarding: str,
            option_discovery_method: str
    ):
        self.actions: List[int] = actions
        self.skill_training_window: int = skill_training_window
        self.alpha: float = alpha
        self.epsilon: float = epsilon
        self.gamma: float = gamma
        self.state_dtype: Type = state_dtype
        self.state_shape: Tuple[int, ...] = state_shape

        self.max_subgoal_height: int = max_subgoal_height
        self.option_onboarding: str = option_onboarding
        self.option_discovery_method: str = option_discovery_method

        self.num_nodes: int = 0
        self.adjacency_matrix: None|sparse.SparseMatrix = None
        self.state_transition_graph: nx.DiGraph = nx.DiGraph()
        self.subgoal_graph: None|nx.DiGraph = None
        self.state_node_lookup: Dict[str, str] = {}
        self.node_state_lookup: Dict[str, str] = {}
        # node -> next_node -> num observations
        self.total_transitions: Dict[str, Dict[str, int]] = {}
        self.subgoals_list: None|List[List[str]] = None

        self.current_skill: None|PreparednessSkill = None

        # action: int
        # skill: (start_state, end_state, level)
        # skill -> state -> skill|action -> q-value
        self.skill_policies: Dict[Tuple[str, str, str], Dict[str, Dict[int|Tuple[str, str, str], float]]] = {}
        # state -> skill|action -> q-value
        self.q_values: Dict[str, Dict[Tuple[str, str, str]|int, float]] = {}

        # skill -> skill_object
        self.skill_lookup: Dict[Tuple[str, str, str], PreparednessSkill] = {}
        self.skills: List[PreparednessSkill] = []
        return

    def add_node_to_graph(
            self,
            state: np.ndarray,
    ):
        state_str = self.state_to_state_str(state)
        try:
            _ = self.state_node_lookup[state_str]
        except KeyError:
            node = str(self.num_nodes)
            self.state_node_lookup[state_str] = node
            self.node_state_lookup[node] = state_str
            self.num_nodes += 1
            self.state_transition_graph.add_node(node, attr={"state": state_str})
        return

    def add_start_state(
            self,
            state: np.ndarray
    ):
        self.add_node_to_graph(state)
        return

    def choose_action(self,
            state: np.ndarray,
            optimal_choice: bool = False,
            possible_actions: None | List[int] = None
        ) -> int:
        if possible_actions is None:
            possible_actions = self.actions

        if self.behaviour == AgentBehaviour.EXPLORE:
            return rand.choice(possible_actions)

        if self.behaviour == AgentBehaviour.TRAIN_SKILLS:
            if self.current_skill is None:
                self.current_skill = self.choose_training_skill(state)

            if self.current_skill is None:
                return rand.choice(possible_actions)

            return self.follow_current_skill(state, False, possible_actions)

        # Agent Behaviour is LEARN
        if self.current_skill is not None:
            return self.follow_current_skill(state, True, possible_actions)
        state_str = self.state_to_state_str(state)
        if possible_actions is None:
            possible_actions = self.actions

        try:
            state_values = self.q_values[state_str]
        except KeyError:
            possible_skills = []
            possible_skills += possible_actions
            for skill in self.skills:
                if skill.initiated(state):
                    possible_skills.append(self.get_skill_tuple(skill))
            self.q_values[state_str] = {
                possible_skill: 0.0 for possible_skill in possible_skills
            }
            state_values = self.q_values[state_str]

        if (not optimal_choice) and (rand.uniform(0, 1) <= self.epsilon):
            skill_tuple = rand.choice(list(state_values.keys()))
        else:
            chosen_skills = []
            max_value = -np.inf
            for skill in state_values:
                skill_value = state_values[skill]
                if skill_value > max_value:
                    max_value = skill_value
                    chosen_skills = [skill]
                elif skill_value == max_value:
                    chosen_skills.append(skill)
            skill_tuple = rand.choice(chosen_skills)

        if type(skill_tuple) == tuple:
            self.current_skill = self.skill_lookup[skill_tuple]
            return self.follow_current_skill(state, True, possible_actions)
        return int(skill_tuple)

    def choose_training_skill(
            self,
            state: np.ndarray
    ) -> None|PreparednessSkill:
        possible_skills = []
        for skill in self.skills:
            if skill.initiated(state):
                possible_skills.append(skill)

        if len(possible_skills) == 0:
            return None

        training_skill = rand.choice(possible_skills)
        return training_skill

    # TODO: Copy Agent
    def copy_agent(
            self,
            copy_from: 'PreparednessIncremental'
    ):
        pass

    def compute_graph_preparedness(self, hop: int):
        for node in self.state_transition_graph.nodes():
            self.preparedness(str(node), hop)
        return

    def create_skill_between_subgoals(
            self,
            start_state: np.ndarray,
            end_state: np.ndarray,
            level: int,
            skill_options: List[SkillBetweenSubgoals]|None
    ) -> 'SkillBetweenSubgoals':
        if skill_options is None:
            skill_options = self.actions
        return SkillBetweenSubgoals(
            skill_options,
            start_state,
            end_state,
            level,
            self.alpha,
            self.epsilon,
            self.gamma,
            self.state_dtype,
            self.has_path_to_state
        )

    def create_skill_towards_subgoal(
            self,
            end_state: np.ndarray,
            level: int,
            skill_options: List[Option] | None
    ) -> 'SkillTowardsSubgoal':
        if skill_options is None:
            skill_options = self.actions
        return SkillTowardsSubgoal(
            skill_options,
            end_state,
            level,
            self.alpha,
            self.epsilon,
            self.gamma,
            self.state_dtype,
            self.has_path_to_state
        )

    def create_skills_from_subgoal_graph(
            self,
            subgoal_graph: nx.MultiDiGraph,
            use_existing_skills: bool,
    ) -> (Dict[str, List[PreparednessOption]]|
          Tuple[List[PreparednessOption], Dict[str, List[PreparednessOption]], List[PreparednessOption]]):
        subgoal_graph_distances = nx.floyd_warshall(self.subgoal_graph)
        max_option_level = -np.inf

        for start_node in self.aggregate_graph.nodes(data=False):
            for end_node in self.aggregate_graph.nodes(data=False):
                distance = subgoal_graph_distances[start_node][end_node]
                if distance >= np.inf:
                    continue
                if distance > max_option_level:
                    max_option_level = distance
        max_option_level = int(max_option_level)

        skills_between_subgoals = {str(i): [] for i in range(1, max_option_level + 1)}
        options_for_option = []

        # Options Between Subgoals
        for k in range(1, max_option_level + 1):
            for start_node in self.subgoal_graph.nodes(data=False):
                start_state = self.node_to_state(start_node)
                for end_node in self.subgoal_graph.nodes(data=False):
                    if k != subgoal_graph_distances[start_node][end_node]:
                        continue

                    end_state = self.node_to_state(end_node)

                    skill = self.create_skill_between_subgoals(
                        start_state,
                        end_state,
                        k,
                        options_for_option
                    )

                    skills_between_subgoals[str(k)].append(skill)

            options_for_option += skills_between_subgoals[str(k)]

        # Generic Onboarding
        generic_onboarding_skill = Option(
            policy=OptionsAgent(
                self.alpha,
                self.epsilon,
                self.gamma,
                self.actions,
                state_dtype=self.state_dtype
            ),
            initiation_func=self.generic_onboarding_initiation,
            terminating_func=self.generic_onboarding_termination
        )

        # Specific Onboarding
        specific_onboarding_skills = []
        for node in self.subgoal_graph.nodes(data=False):
            if len(self.subgoal_graph.in_egdes(node)) <= 0
                specific_onboarding_skills.append(
                    self.create_skill_towards_subgoal(
                        self.node_to_state(node),
                        1,
                        None
                    )
                )

        # Options to Subgoals

        return onboarding_skills, skills_between_subgoals, skill_to_subgoals

    def create_subgoal_graph(
            self
    ):
        self.subgoal_graph = nx.MultiDiGraph()

        max_subgoal_level = len(self.subgoals_list)

        for level in range(max_subgoal_level):
            for subgoal in self.subgoals_list[level]:
                self.subgoal_graph.add_node(subgoal, attr={'preparedness subgoal level': level})

                increasing_paths_found = False
                increasing_level = level - 1
                while (not increasing_paths_found) and (increasing_level > 0):
                    for subgoal_hat in self.subgoals_list[increasing_level]:
                        if nx.has_path(self.state_transition_graph, subgoal, subgoal_hat):
                            increasing_paths_found = True
                            self.subgoal_graph.add_node(
                                subgoal_hat,
                                attr={'preparedness subgoal level': increasing_level}
                            )
                            self.subgoal_graph.add_edge(subgoal, subgoal_hat)
                    increasing_level -= 1

                decreasing_paths_found = False
                decreasing_level = level + 1
                while (not decreasing_paths_found) and (decreasing_level < max_subgoal_level):
                    for subgoal_hat in self.subgoals_list[decreasing_level]:
                        if nx.has_path(self.state_transition_graph, subgoal, subgoal_hat):
                            decreasing_paths_found = True
                            self.subgoal_graph.add_node(
                                subgoal_hat,
                                attr={'preparedness subgoal level': decreasing_level}
                            )
                            self.subgoal_graph.add_edge(subgoal, subgoal_hat)
                    decreasing_level += 1

        return

    def discover_skills(
            self
    ):
        # REPLACE:
        # Find subgoals
        # Make subgoal graph
        # generate set of skills
        # skip over any skills that already exist

        # UPDATE:
        # Find subgoals
        # add subgoals onto existing subgoal graph
        # generate new set of skills
        # add set of skills to existing set

        if self.option_discovery_method == 'replace':
            self.subgoals_list = self.find_subgoals()
            if self.subgoals_list is None:
                return
            self.create_subgoal_graph()
        elif self.option_discovery_method == 'update':
            new_subgoals = self.find_subgoals()
            new_skills = False

            if self.subgoal_list is None:
                self.subgoals_list = new_subgoals
                new_skills = True
            else:
                current_height = len(self.subgoals_list)
                new_subgoals_height = len(new_subgoals)
                for subgoal_level in range(new_subgoals_height):
                    for new_subgoal in new_subgoals[subgoal_level]:
                        subgoal_found = False
                        i = 0
                        while (i < current_height) and (not subgoal_found):
                            if new_subgoal in self.subgoals_list[i]:
                                subgoal_found = True
                            i += 1
                        if not subgoal_found:
                            new_skills = True
                            try:
                                self.subgoals_list[subgoal_level].append(new_subgoal)
                            except IndexError:
                                self.subgoals_list.append([new_subgoal])

            if new_skills:
                self.create_subgoal_graph()

        # FIND skills
        # If skill already exists - keep, if not add to new list
        # Update skill policy and agent policy tables to account for new skills
        new_skills = []
        max_level = -np.inf
        for subgoal in list(self.subgoal_graph.nodes()):
            for subgoal_hat in self.subgoals_list[subgoal]:
                try:
                    level = nx.shortest_path_length(
                        self.subgoal_graph,
                        subgoal,
                        subgoal_hat
                    )
                except nx.NetworkXNoPath:
                    continue

                if level > max_level:
                    max_level = level

                new_skills.append(
                    PreparednessSkill(
                        self.node_to_state(subgoal),
                        self.node_to_state(subgoal_hat),
                        str(level),
                        self.has_path_to_state
                    )
                )

        # Skills towards subgoals
        if self.option_onboarding in ['generic', 'specific']:
            for subgoal in list(self.subgoal_graph.nodes()):
                new_skills.append(
                    PreparednessSkill(
                        None,
                        self.node_state_lookop[subgoal],
                        str(max_level),
                        self.has_path_to_state
                    )
                )


        # Onboarding skills
        if self.option_onboarding == 'specific':
            for subgoal in list(self.subgoal_graph.nodes()):
                num_in_edges = len(nx.get_in_edges(self.subgoal_graph, subgoal))
                if num_in_edges <= 0:
                    new_skills.append(
                        PreparednessSkill(
                            None,
                            self.node_state_lookop[subgoal],
                            '1',
                            self.has_path_to_state
                        )
                    )
        elif self.option_onboarding == 'generic':
            new_skills.append(
                PreparednessSkill(
                    None,
                    None,
                    '1',
                    self.has_path_to_state,
                    list(self.subgoal_graph.nodes())
                )
            )

        # REPLACE: all new skills are added, only keep a skill if it appears in the set of new skills
        # UPDATE: all current skills are kept, only add a new skill if it does not appear in current skills
        if self.option_discovery_method == 'replace':
            self.skills = new_skills
            for new_skill in self.skills:
                self.skill_lookup[self.get_skill_tuple(new_skill)] = new_skill
        elif self.option_discovery_method == 'update':
            # Checking which skills already exist, adding the ones that do not
            for new_skill in new_skills:
                skill_exists = False
                for existing_skill in self.skills:
                    if existing_skill == new_skill:
                        skill_exists = True
                        break

                if skill_exists:
                    continue
                self.skills.append(new_skill)
                self.skill_lookup[self.get_skill_tuple(new_skill)] = new_skill

        self.update_available_skills()
        return

    @staticmethod
    def entropy(
            distribution: List[float],
            log_base: float=2
    ) -> float:
        entropy = 0
        for prob in distribution:
            if prob <= 0:
                continue
            entropy -= prob * np.emath.logn(log_base, prob)
        return entropy

    def find_subgoals(
            self
    ) -> None|List[List[str]]:
        # Look for subgoals up to the max height
        self.adjacency_matrix = nx.to_scipy_sparse_array(self.state_transition_graph)

        subgoals = {}
        subgoals_complete = False
        hop = 1
        while (hop <= self.max_subgoal_height) and not subgoals_complete:
            hop_subgoals = []

            self.compute_graph_preparedness(hop)

            distances = sparse.csgraph.dijkstra(
                self.adjacency_matrix,
                True,
                unweighted=True,
                limit=hop + 1
            )

            for node in self.state_transition_graph.nodes():
                is_subgoal = True
                in_neighbours = np.where(
                    (distances[:, int(node)] <= hop) & (0 < distances[: int(node)])[0]
                )
                if in_neighbours.size <= 0:
                    is_subgoal = False
                else:
                    out_neighbours = np.where(distances[int(node), :] <= hop)[0]
                    value = self.preparedness_values[node][self.preparedness_key(hop)]
                    for neighbour in np.append(in_neighbours, out_neighbours):
                        neighbour_str = str(neighbour)
                        if neighbour_str == node:
                            continue
                        neighbour_value = self.preparedness_values[neighbour_str][self.preparedness_key(hop)]
                        if neighbour_value >= value:
                            is_subgoal = False
                            continue
                is_subgoal_str = "False"
                if is_subgoal:
                    is_subgoal_str = "True"
                self.preparedness_values[node][self.subgoal_key(hop)] = is_subgoal_str

                if is_subgoal:
                    hop_subgoals.append(node)

            subgoals[hop] = hop_subgoals.copy()

            if (hop <= 2) and (subgoals[hop] == subgoals[hop - 1]):
                subgoals_complete = True

            hop += 1

        if not subgoals_complete:
            return None

        max_subgoal_hop = hop - 1
        pruned_subgoals = {hop: [] for hop in range(1, max_subgoal_hop + 1)}
        found_subgoals = []
        for hop in range(1, max_subgoal_hop + 1):
            for subgoal in subgoals[hop]:
                if subgoal in found_subgoals:
                    continue

                subgoal_pruned = False
                for i in range(max_subgoal_hop, hop, -1):
                    if subgoal in subgoals[i]:
                        pruned_subgoals[i].append(subgoal)
                        found_subgoals.append(subgoal)
                        subgoal_pruned = True
                        break
                if not subgoal_pruned:
                    pruned_subgoals[hop].append(subgoal)

        subgoals_no_empty = []
        for i in range(1, max_subgoal_hop + 1):
            if not pruned_subgoals[i]:
                continue
            subgoals_no_empty.append(pruned_subgoals[i])

        return subgoals_no_empty

    def follow_current_skill(
            self,
            state: np.ndarray,
            optimal_choice: bool=False,
            possible_actions: None|List[int]=None
    ) -> int:
        if self.current_skill is None:
            raise AttributeError("Must have a current skill to follow")

        level = int(self.current_skill.level)
        skill = self.current_skill
        while level > 1:
            if skill.current_skill is not None:
                skill = skill.current_skill
            else:
                next_skill = self.lookup_skill_policy(skill, state, optimal_choice, possible_actions)
                skill.set_skill(next_skill)
                skill = next_skill

            level = int(skill.level)

        chosen_action = self.lookup_skill_policy(skill, state, optimal_choice, possible_actions)
        return chosen_action

    def frequency_entropy(
            self,
            node: str,
            hops: int,
            neighbours: np.ndarray,
            accuracy: int=4
    ) -> float:
        if hops > 0 and neighbours.size == 1:
            try:
                frequency_entropy = self.preparedness_values[node][self.frequency_entropy_key(hops)]
                return frequency_entropy
            except KeyError:
                ()

        # P(S_t+n | s_t)
        def prob(start_node, goal_node, hops_away):
            p = 0
            W_start_node = 0.0
            for j in neighbours:
                W_start_node += self.adjacency_matrix[int(start_node), int(j)]
            if (W_start_node <= 0) and (start_node == goal_node):
                return 1.0
            if hops_away == 1:
                if W_start_node <= 0:
                    return 0
                p = self.adjacency_matrix[int(start_node), int(goal_node)] / W_start_node
                return p

            for j in neighbours:
                w_start_node_j = self.adjacency_matrix[int(start_node), int(j)]
                if w_start_node_j <= 0:
                    continue
                p += (w_start_node_j / W_start_node) * prob(j, goal_node, hops_away - 1)

            return p

        neighbour_probabilities = [prob(node, neighbour, hops) for neighbour in neighbours]
        return round(self.entropy(neighbour_probabilities), accuracy)

    @staticmethod
    def frequency_entropy_key(
            hops: int
    ) -> str:
        return 'frequency_entropy ' + str(hops) + ' hops'

    def generic_onboarding_initiation(
            self,
            state: np.ndarray
    ) -> bool:
        for level in self.subgoals_list:
            for subgoal in level:
                subgoal_state = self.node_to_state(subgoal)
                if np.array_equal(state, subgoal_state):
                    return False
        for subgoal_list in self.subgoals_list:
            for subgoal in subgoal_list:
                if self.has_path_to_node(state, subgoal):
                    return True
        return False

    def generic_onboarding_termination(
            self,
            state: np.ndarray
    ) -> bool:
        return not self.generic_onboarding_initiation(state)

    def get_skill_action_values(
            self,
            skill: PreparednessSkill,
            state: np.ndarray,
            possible_actions: List[int]|None=None
    ) -> Dict[int|Tuple[str, str, str], float]:
        if possible_actions is None:
            possible_actions = self.actions
        skill_tuple = self.get_skill_tuple(skill)
        state_str = self.state_to_state_str(state)

        try:
            skill_values = self.skill_policies[skill_tuple]
        except KeyError:
            self.skill_policies[skill_tuple] = {}
            skill_values = {}

        try:
            state_values = skill_values[state_str]
        except KeyError:
            skills_for_skill = self.get_skills_for_skill(skill, possible_actions)
            self.skill_policies[skill_tuple][state_str] = {
                state_str: {
                    skill_for_skill: 0.0 for skill_for_skill in skills_for_skill
                }
            }
            state_values = self.skill_policies[skill_tuple][state_str]

        return state_values

    def get_skill_tuple(
            self,
            skill: PreparednessSkill
    ) -> Tuple[str, str, str]:
        skill_start_state = str(None)
        if skill.start_state is not None:
            skill_start_state = self.state_to_state_str(skill.start_state)

        skill_end_state = str(None)
        if skill.end_state is not None:
            skill_end_state = self.state_to_state_str(skill.end_state)

        return skill_start_state, skill_end_state, skill.level

    def get_skills_for_skill(
            self,
            skill: PreparednessSkill,
            possible_actions: None | List[int] = None
    ) -> List[Tuple[str, str, str]|int]:
        # Skills Between Subgoals:
        #   If level 1:
        #       primitive actions
        #   Else:
        #       all skills between subgoals of < level
        # Onboarding Skills:
        #       Primitive actions
        # Skills Towards Subgoals:
        #       Onboarding skill + all skills between subgoals

        skill_level = int(skill.level)
        if possible_actions is None:
            possible_actions = self.actions

        if skill.start_state is not None and skill.end_state is not None:
            if skill_level <= 1:
                return possible_actions
            skills = [possible_skill for possible_skill in self.skills if int(possible_skill.level) < skill_level]
            return skills

        if skill_level <= 1:
            return self.actions

        skills = [
            self.get_skill_tuple(possible_skill)
            for possible_skill in self.skills
            if int(possible_skill.level) < skill_level
               and skill.start_state is not None
               and skill.end_state is not None
        ]

        if skill.start_state is not None and skill.end_state is not None:
            return skills

        if skill.start_state is None:
            for possible_skill in self.skills:
                if int(possible_skill.level) == 1 and possible_skill.start_state is None:
                    skills.append(self.get_skill_tuple(possible_skill))

        return skills

    def has_path_to_state(
            self,
            start_state: np.ndarray,
            end_state: np.ndarray
    ) -> bool:
        try:
            end_node = self.state_to_node(end_state)
        except ValueError:
            return False
        return self.has_path_to_node(start_state, end_node)

    def has_path_to_node(
            self,
            state: np.ndarray,
            node: str
    ) -> bool:
        try:
            state_node = self.state_to_node(state)
        except ValueError:
            return False
        return nx.has_path(self.state_transition_graph, state_node, node)

    def lookup_skill_policy(
            self,
            skill: PreparednessSkill,
            state: np.ndarray,
            optimal_choice: bool=False,
            possible_actions: None|List[int]=None
    ) -> int|PreparednessSkill:
        state_values = self.get_skill_action_values(skill, state, possible_actions)

        if int(skill.level) > 1:
            available_skills = []
            for possible_skill_tuple in state_values:
                possible_skill = self.skill_lookup[possible_skill_tuple]
                if possible_skill.initiated(state):
                    available_skills.append(possible_skill_tuple)
        else:
            available_skills = list(state_values.keys())

        if (not optimal_choice) and (rand.uniform(0, 1) <= self.epsilon):
            chosen_skill_tuple = rand.choice(available_skills)
        else:
            possible_skills = []
            max_value = -np.inf
            for possible_skill in available_skills:
                value = state_values[possible_skill]
                if value > max_value:
                    possible_skills = [possible_skill]
                    max_value = value
                elif value == max_value:
                    possible_skills.append(possible_skill)

            chosen_skill_tuple = rand.choice(possible_skills)

        if type(chosen_skill_tuple) == tuple:
            chosen_skill = self.skill_lookup[chosen_skill_tuple]
        else:
            chosen_skill = int(chosen_skill_tuple)
        return chosen_skill

    def neighbourhood_entropy(
            self,
            node: str,
            hops: int,
            neighbours: np.ndarray,
            accuracy: int=4
    ) -> float:

        # W_n_i_j
        def weights_out(start_node, hops_away):
            W = 0.0
            if hops_away == 1:
                for j in neighbours:
                    W += self.adjacency_matrix[int(start_node), int(j)]
                return W

            for j in neighbours:
                w_start_node_j = self.adjacency_matrix[int(start_node), int(j)]
                if w_start_node_j <= 0:
                    continue
                W += (w_start_node_j * weights_out(j, hops_away - 1))
            return W

        T = 0
        for neighbour in neighbours:
            T += weights_out(neighbour, hops)

            # Compute Hops to each neighbourhood
            def weights_to_node(start_node, goal_node, hops_away):
                if hops_away == 1:
                    return self.adjacency_matrix[int(start_node), int(goal_node)]

                P_hat = 0.0
                if hops_away == 2:
                    for j in neighbours:
                        P_hat += (self.adjacency_matrix[int(start_node), int(j)] * self.adjacency_matrix[int(j), int(goal_node)])
                    return P_hat

                for j in neighbours:
                    w_start_node_j = self.adjacency_matrix[int(start_node), int(j)]
                    if w_start_node_j <= 0:
                        continue
                    P_hat += (w_start_node_j * weights_to_node(j, goal_node, hops_away - 1))
                return P_hat

            neighbour_probabilities = []
            if T == 0:
                return 0.0
            for goal_neighbour in neighbours:
                P = 0
                for start_neighbour in neighbours:
                    P += weights_to_node(start_neighbour, goal_neighbour, hops)
                neighbour_probabilities.append(P / T)

            # Compute Entropy
            return round(self.entropy(neighbour_probabilities), accuracy)

    @staticmethod
    def neighbourhood_entropy_key(
            hops: int
    ) -> str:
        return "neighbourhood_entropy " + str(hops) + ' hops'

    def node_to_state(
            self,
            node: str
    ) -> np.ndarray:
        return self.state_str_to_state(self.node_to_state_str(node))

    def policy_learn(
            self,
            state: np.ndarray,
            action: int,
            reward: float,
            next_state: np.ndarray,
            terminal: bool | None = None,
            next_state_possible_actions: List[int] | None = None
    ):
        pass

    def preparedness(self, node: str, hop: int) -> float:
        distances = sparse.csgraph.dijkstra(
            self.adjacency_matrix, directed=True, indicies=int(node), unweighted=True, limit=hop+1
        )
        neighbours = np.where((0 < distances) & (distances <= hop))[0]

        frequency_entropy = self.frequency_entropy(
            node,
            hop,
            neighbours
        )
        neighbourhood_entropy = self.neighbourhood_entropy(
            node,
            hop,
            neighbours
        )
        node_preparedness = frequency_entropy + neighbourhood_entropy

        try:
            existing_values = self.preparedness_values[node]
        except KeyError:
            existing_values = {}
        existing_values[self.frequency_entropy_key(hop)] = frequency_entropy
        existing_values[self.neighbourhood_entropy_key(hop)] = neighbourhood_entropy
        existing_values[self.preparedness_key(hop)] = node_preparedness

        return node_preparedness

    @staticmethod
    def preparedness_key(
            hops: int
    ) -> str:
        return "preparedness " + str(hops) + ' hops'

    def save_representation(
            self,
            save_path: str
    ):
        pass

    @staticmethod
    def skill_successful(
            skill: PreparednessSkill,
            state: np.ndarray
    ) -> bool:
        end_states = skill.end_states
        if skill.end_state is not None:
            end_states = [skill.end_state]

        for end_state in end_states:
            if np.array_equal(state, end_state):
                return True
        return False

    def state_to_node(
            self,
            state: np.ndarray
    ) -> str:
        state_str = self.state_to_state_str(state)
        try:
            node = self.state_node_lookup[state_str]
            return node
        except KeyError:
            raise ValueError("State has not been explored yet")

    @staticmethod
    def subgoal_key(
            hop: int
    ) -> str:
        return 'preparedness subgoal ' + str(hop) + " hops"

    # TODO
    def train_skill(
            self,
            skill: PreparednessSkill,
            state: np.ndarray,
            action: int,
            reward: float,
            next_state: np.ndarray,
            terminal: bool | None = None,
            next_state_possible_actions: List[int] | None = None
    ):
        # Checks if skill is terminal:
        #   if so sets current skill as none
        # Finds reward
        #   Reaching goal - positive
        #   terminating and not at goal - negative
        #   neither: small negative
        # Trains skill policy using Q-learning for primitive and macro Q for hierarchical

        current_skill_tuple = self.get_skill_tuple(self.current_skill)
        if next_state_possible_actions is None:
            next_state_possible_actions = self.actions
        skill_reward = self.skill_training_step_reward
        skill_terminated = False
        state_str = self.state_to_state_str(state)

        if self.skill_successful(self.current_skill, next_state):
            skill_reward = self.skill_training_success_reward
            skill_terminated = True
        elif self.current_skill.terminated(next_state) or terminal:
            skill_reward = self.skill_training_failure_reward
            skill_terminated = True

        if int(self.current_skill.level) <= 1:
            action_values = self.get_skill_action_values(skill, state)
            next_state_action_values = self.get_skill_action_values(skill, next_state, next_state_possible_actions)
            action_value = action_values[action]
            action_value += self.alpha * (
                    skill_reward + (self.gamma * max(next_state_action_values.values()) - action_value)
            )
            self.skill_policies[current_skill_tuple][state_str][action] = action_value
        else:
            # TODO
            pass

        if skill_terminated:
            self.current_skill.reset_skill()
            self.current_skill = None

        return

    def update_available_skills(
            self
    ):
        # updating skill policies
        for skill_tuple in self.skill_policies:
            skill = self.skill_lookup[skill_tuple]
            if int(skill.level) <= 1:
                continue
            new_skills = self.get_skills_for_skill(skill)
            keys_to_add = []
            keys_to_remove = []
            state_str = list(self.skill_policies[skill_tuple].keys())[0]

            # REPLACE:
            # add: all new skills not in existing skills = new_skills \ existing_skills
            # remove: all existing skills not in new skills = existing_skills \ new_skills
            if self.option_discovery_method == 'replace':
                keys_to_add = new_skills.copy()
                keys_to_remove = list(self.skill_policies[skill_tuple][state_str].keys())
                for existing_skill_tuple in self.skill_policies[skill_tuple][state_str]:
                    if existing_skill_tuple in new_skills:
                        keys_to_add.remove(existing_skill_tuple)
                        keys_to_remove.remove(existing_skill_tuple)
            # UPDATE:
            # add: all new skills not in existing skills
            elif self.option_discovery_method == 'update':
                keys_to_add = []
                existing_skills = list(self.skill_policies[skill_tuple][state_str].keys())
                for new_skill_tuple in new_skills:
                    if new_skill_tuple not in existing_skills:
                        keys_to_add.append(new_skill_tuple)

            # set skill policies so only skills remaining are new keys
            for state_str in self.skill_policies[skill_tuple]:
                for key_to_add in keys_to_add:
                    self.skill_policies[skill_tuple][state_str][key_to_add] = 0.0
                for key_to_remove in keys_to_remove:
                    del self.skill_policies[skill_tuple][state_str][key_to_remove]

        # updating agent policy
        for state_str in self.q_values:
            skills_to_add = []
            skills_to_remove = []

            # REPLACE:
            # add: any skills that can be initiated and not existing
            # remove: any existing skills no loger existing
            if self.option_discovery_method == 'replace':
                for new_skill in self.skills:
                    new_skill_tuple = self.get_skill_tuple(new_skill)
                    try:
                        self.q_values[state_str][new_skill_tuple]
                    except KeyError:
                        if new_skill.initiated(self.state_str_to_state(state_str)):
                            skills_to_add.append(new_skill_tuple)

                for skill_action in self.q_values[state_str]:
                    if type(skill_action) == int:
                        continue
                    try:
                        self.skill_lookup[skill_action]
                    except KeyError:
                        skills_to_remove.append(skill_action)
                        continue

            # UPDATE:
            # add: any skills that can now be initiated in the state and not existing
            elif self.option_discovery_method == 'update':
                for new_skill in self.skills:
                    new_skill_tuple = self.get_skill_tuple(new_skill)
                    try:
                        self.q_values[state_str][new_skill_tuple]
                    except KeyError:
                        if new_skill.initiated(self.state_str_to_state(state_str)):
                            skills_to_add.append(new_skill_tuple)

            for skill_to_add in skills_to_add:
                self.q_values[state_str][skill_to_add] = 0.0
            for skill_to_remove in skills_to_remove:
                del self.q_values[state_str][skill_to_remove]

        return

    def update_representation(
            self,
            state: np.ndarray,
            action: int,
            reward: float,
            next_state: np.ndarray,
            terminal: bool | None = None
    ):
        self.add_node_to_graph(state)
        self.add_node_to_graph(next_state)

        state_node = self.state_to_node(state)
        next_state_node = self.state_to_node(next_state)

        # update transition observations
        try:
            transitions = self.total_transitions[state_node]
            try:
                num_transitions = transitions[next_state_node]
            except KeyError:
                num_transitions = 0
        except KeyError:
            num_transitions = 0
            self.total_transitions[state_node] = {}
        self.total_transitions[state_node][next_state_node] = num_transitions + 1

        sum_transitions = sum(self.total_transitions[state_node].values())

        # Update edge weights
        new_edge_weights = [(state_node, v, self.total_transitions[state_node][v]/sum_transitions)
                           for v in self.total_transitions[state_node]]
        self.state_transition_graph.add_weighted_edges_from(new_edge_weights)
        pass
