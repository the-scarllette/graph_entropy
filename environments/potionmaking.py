# Simplified version of the Quacks of Quedlingburg Game

import random as rand
import numpy as np
from typing import Any

from environments.environment import Environment

'''
pull chips
find score, rive reward as score
buy chips
reset bag
repeat

State:
ingredients in bag
bombs in bag
ingredients not in bag
bombs not in bag
in making or buying phase
current score
current round
'''


class Ingredient:

    def __init__(self, cost=None, value=0, is_bomb=False):
        self.cost = cost
        self.value = value
        self.is_bomb = is_bomb
        return


class PotionMaking(Environment):

    default_bombs = [1]
    default_start_ingredients = [1, 1]
    default_explosion_trigger = 1

    default_costs = {1: 2,
                     2: 3,
                     4: 6}

    default_rounds = 3

    draw_ingredient_action = 0
    stop_drawing_action = 1
    stop_buying_action = 2

    def __init__(self, bombs=None, start_ingredients=None, explosion_trigger=None, ingredient_costs=None,
                 num_rounds=None):

        self.ingredient_costs = ingredient_costs
        if self.ingredient_costs is None:
            self.ingredient_costs = PotionMaking.default_costs

        if start_ingredients is None:
            start_ingredients = PotionMaking.default_start_ingredients
        self.start_ingredients = [self.make_ingredient(ingredient)
                                  for ingredient in start_ingredients]

        if bombs is None:
            bombs = PotionMaking.default_bombs
        self.num_bombs = len(set(bombs))

        self.bombs = [self.make_ingredient(bomb, True) for bomb in bombs]

        self.environment_name = 'Potion_Making'

        self.explosion_trigger = explosion_trigger
        if self.explosion_trigger is None:
            self.explosion_trigger = PotionMaking.default_explosion_trigger

        self.num_rounds = num_rounds
        if self.num_rounds is None:
            self.num_rounds = PotionMaking.default_rounds

        self.num_start_ingredients = len(self.start_ingredients) + len(self.bombs)

        self.bag = None
        self.drawn = None
        self.in_making_phase = False
        self.num_ingredients = len(self.ingredient_costs)
        self.in_making_phase_index = 2 * (self.num_bombs + self.num_ingredients)
        self.current_score_index = self.in_making_phase_index + 1
        self.current_round_index = self.current_score_index + 1
        self.num_ingredients_in_bag = 0
        self.score = 0
        self.state_size = 2 * (self.num_bombs + self.num_ingredients) + 3
        self.bombs_drawn = 0
        self.current_round = 0
        self.terminal = True

        # Actions: draw ingredient, stop drawing, stop buying, buy each kind of ingredient
        self.possible_actions = [self.draw_ingredient_action, self.stop_drawing_action, self.stop_buying_action]
        i = 1
        for _ in self.ingredient_costs:
            self.possible_actions.append(self.stop_buying_action + i)
            i += 1

        # Environment Name
        def count_array(a):
            counts = {}
            a.sort()
            for elm in a:
                try:
                    counts[elm] += 1
                except KeyError:
                    counts[elm] = 1
            return counts
        ingredient_counts = count_array(start_ingredients)
        bomb_counts = count_array(bombs)

        self.environment_name = "potion_making"
        for ingredient in ingredient_counts:
            self.environment_name += "_" + str(ingredient_counts[ingredient]) + "x" + str(ingredient)
        for bomb in bomb_counts:
            self.environment_name += "_" + str(bomb_counts[bomb]) + "x" + str(bomb) + "b"
        return

    def get_current_state(self):
        if self.terminal:
            raise AttributeError("Environment is terminal, cannot get current states")

        current_state = np.zeros((self.state_size,), dtype=int)
        bomb_to_index = {1: 3}
        ingredient_to_index = {1: 0,
                               2: 1,
                               4: 2}

        # Ingredients in bag
        for ingredient in self.bag:
            if ingredient.is_bomb:
                index = bomb_to_index[ingredient.value]
            else:
                index = ingredient_to_index[ingredient.value]
            current_state[index] += 1

        # Ingredients drawn from bag
        for ingredient in self.drawn:
            if ingredient.is_bomb:
                index = bomb_to_index[ingredient.value]
            else:
                index = ingredient_to_index[ingredient.value]
            current_state[index + self.num_bombs + len(self.ingredient_costs)] += 1

        # In Potion Making Phase
        current_state[self.in_making_phase_index] = int(self.in_making_phase)

        # Current Score
        current_state[self.current_score_index] = int(self.score)

        # Current Round
        current_state[self.current_round_index] = int(self.current_round)

        return current_state

    def get_possible_actions(self, state=None):
        if state is None:
            if self.terminal:
                return []
            state = self.get_current_state()

        # If in make phase draw ingredient or stop drawing
        if state[self.in_making_phase_index]:
            return [self.draw_ingredient_action, self.stop_drawing_action]

        # If in buy phase stop buying action, Check money, can buy all ingredients it can afford
        possible_actions = [self.stop_buying_action]
        i = 1
        for ingredient in self.ingredient_costs:
            if state[self.current_score_index] >= self.ingredient_costs[ingredient]:
                possible_actions.append(self.stop_buying_action + i)
            i += 1
        return possible_actions

    def get_start_states(self):
        start_state = np.zeros((self.state_size,), dtype=int)
        bomb_to_index = {1: 3}
        ingredient_to_index = {1: 0,
                               2: 1,
                               4: 2}

        for ingredient in (self.start_ingredients + self.bombs):
            if ingredient.is_bomb:
                index = bomb_to_index[ingredient.value]
            else:
                index = ingredient_to_index[ingredient.value]
            start_state[index] += 1

        start_state[self.in_making_phase_index] = 1

        start_state[self.current_score_index] = 0

        start_state[self.current_round_index] = 0

        return [start_state]

    def get_successor_states(self, state, probability_weights=False):
        bomb_to_index = {1: 3}
        ingredient_to_index = {1: 0,
                               2: 1,
                               4: 2}
        ingredients_bombs = list(ingredient_to_index.keys()) + list(bomb_to_index.keys())

        successor_states = []
        weights = []

        in_making_phase = bool(state[self.in_making_phase_index])
        current_round = state[self.current_round_index]
        current_score = state[self.current_score_index]

        if current_round >= self.num_rounds:
            return successor_states, weights

        def reset_state_bag(to_reset):
            for i in range(self.num_ingredients + self.num_bombs):
                to_reset[i] += to_reset[i + self.num_ingredients + self.num_bombs]
                to_reset[i + self.num_ingredients + self.num_bombs] = 0
            to_reset[self.in_making_phase_index] = 0
            to_reset[self.current_round_index] += 1
            return to_reset

        if in_making_phase:
            bag_size = sum(state[0: self.num_ingredients + self.num_bombs])
            bombs_drawn = sum(state[(2 * self.num_ingredients) + self.num_bombs:
                                     2 * (self.num_ingredients + self.num_bombs)])

            # Stop Drawing
            successor = reset_state_bag(state.copy())
            successor_states.append(successor)
            weight = 1.0
            if probability_weights:
                weight = 0.5
            weights.append(weight)

            # Draw when only 1 ingredient left in bag
            if bag_size <= 1:
                ingredient_left = None
                for i in range(self.num_ingredients):
                    if state[i] > 0:
                        ingredient_left = i
                        break
                if ingredient_left is None:
                    bombs_drawn += 1
                    for i in range(self.num_ingredients, self.num_ingredients + self.num_bombs):
                        if state[i] > 0:
                            ingredient_left = i
                            break
                successor = reset_state_bag(state.copy())

                new_score = successor[self.current_score_index] + ingredient_left
                if bombs_drawn >= self.explosion_trigger:
                    new_score = 0
                successor[self.current_score_index] = new_score

                successor_states.append(successor)
                weight = 1.0
                if probability_weights:
                    weight = 0.5
                weights.append(weight)
                return successor_states, weights

            # Check if chance of explosion
            start_index = self.num_ingredients
            end_index = self.num_ingredients + self.num_bombs
            explosion_chance = bombs_drawn >= self.explosion_trigger - 1
            if not explosion_chance:
                start_index = end_index

            # Draw, no explosion
            for i in range(start_index):
                num_in_bag = state[i]
                if num_in_bag > 0:
                    successor = state.copy()
                    successor[i] -= 1
                    successor[i + self.num_ingredients + self.num_bombs] += 1
                    successor[self.current_score_index] += ingredients_bombs[i]

                    weight = 1.0
                    if probability_weights:
                        weight = num_in_bag / (2 * bag_size)
                    successor_states.append(successor)
                    weights.append(weight)

            # Skip explosion chance if either no explosion possible or
            # current score <= 0 and we are not using edge weights as then drawing and exploding is equivalent to
            # stopping drawing
            if (not explosion_chance) or (current_score <= 0 and (not probability_weights)):
                return successor_states, weights

            # Draw, explosion
            weight = 1.0
            if probability_weights:
                weight = 0.0
                for i in range(start_index, end_index):
                    num_in_bag = state[i]
                    weight += num_in_bag * (1 / bag_size)
                weight *= 0.5

            if current_score > 0:
                successor = reset_state_bag(state.copy())
                successor[self.current_score_index] = 0
                successor_states.append(successor)
                weights.append(weight)
                return successor_states, weights

            weights[0] += weight
            return successor_states, weights

        # Buying phase
        # End Buying Phase
        successor = state.copy()
        successor[self.in_making_phase_index] = 1
        successor[self.current_score_index] = 0
        successor_states.append(successor)

        # Buy each possible chip
        num_possible_actions = 1
        for ingredient in self.ingredient_costs:
            cost = self.ingredient_costs[ingredient]
            new_score = current_score - cost
            if new_score < 0:
                continue
            num_possible_actions += 1

            successor = state.copy()
            successor[ingredient_to_index[ingredient]] += 1
            successor[self.current_score_index] = new_score
            successor[self.in_making_phase_index] = int(new_score == 0)
            successor_states.append(successor)

        weight = 1.0
        if probability_weights:
            weight = 1 / num_possible_actions
        weights = [weight] * num_possible_actions

        return successor_states, weights

    def make_ingredient(self, value=0, is_bomb=False):
        if is_bomb:
            return Ingredient(value=value, is_bomb=True)
        return Ingredient(cost=self.ingredient_costs[value], value=value, is_bomb=False)

    def reset(self) -> Any:
        self.bag = (self.start_ingredients + self.bombs).copy()
        self.drawn = []
        self.in_making_phase = True
        self.score = 0
        self.num_ingredients = self.num_start_ingredients
        self.num_ingredients_in_bag = self.num_ingredients
        self.bombs_drawn = 0
        self.current_round = 0
        self.terminal = False
        return self.get_current_state()

    def reset_bag(self):
        if self.terminal:
            raise AttributeError("Cannot reset bag if environment is terminal")

        for ingredient in self.drawn:
            self.bag.append(ingredient)
        self.drawn.clear()
        self.in_making_phase = False
        self.current_round += 1
        self.bombs_drawn = 0
        return

    def step(self, action) -> (Any, float, bool, Any):
        '''
        if in making phase
            draw an ingredient
            increment score
            update white chip amount
            update bag count
            If: (bag empty OR over white chip amount):
                Put everything back into bag
                if  at round maximum:
                    Score up and terminate
                Move to buy phase
        If in buying phase:
            purchase chip: remove score amount and perchase a chip
            no purchase chip or out of money or bount 2nd chip:
                End buying phase
        '''

        if self.terminal:
            raise AttributeError("Cannot step if environment is terminal")

        if self.in_making_phase:
            if action == self.draw_ingredient_action:
                bag_size = len(self.bag)
                drawn_ingredient_index = rand.randint(0, bag_size - 1)
                drawn_ingredient = self.bag.pop(drawn_ingredient_index)
                self.drawn.append(drawn_ingredient)

                self.score += drawn_ingredient.value
                if drawn_ingredient.is_bomb:
                    self.bombs_drawn += 1
                    if self.bombs_drawn >= self.explosion_trigger:
                        self.score = 0
                        self.reset_bag()
                elif bag_size <= 1:
                    self.reset_bag()
            elif action == self.stop_drawing_action:
                self.reset_bag()
            reward = 0
            if not self.in_making_phase:
                reward = self.score
            current_state = self.get_current_state()
            self.terminal = self.current_round >= self.num_rounds
            return current_state, reward, self.terminal, None

        min_cost = np.inf
        for ingredient in self.ingredient_costs:
            cost = self.ingredient_costs[ingredient]
            if cost < min_cost:
                min_cost = cost
        if (action == self.stop_buying_action) or (self.score < min_cost):
            self.in_making_phase = True
            self.score = 0
            return self.get_current_state(), 0.0, False, None

        to_buy_index = action - self.stop_buying_action
        i = 1
        buying_ingredient = None
        for ingredient in self.ingredient_costs:
            if i == to_buy_index:
                buying_ingredient = ingredient
                break
            i += 1

        if buying_ingredient is None:
            raise AttributeError(str(action) + " is an invalid action for this state")

        buying_ingredient = self.make_ingredient(ingredient)
        self.score -= buying_ingredient.cost
        self.bag.append(buying_ingredient)

        if self.score == 0:
            self.in_making_phase = True

        return self.get_current_state(), 0.0, False, None
