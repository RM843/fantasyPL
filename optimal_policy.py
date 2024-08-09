from itertools import combinations, product

from generic_code.value_iteration import ValueIteration
from helper_methods import timing_decorator


class Example:

    def __init__(self, all_options, rounds, scores, initial_selection_size):
        self.all_options = all_options
        self.rounds = rounds
        self.scores = scores
        self.initial_selection_size = initial_selection_size

    def is_legal_state(self, state):
        rnd, selection = state
        conds = [len([x for x in selection if x in ['A', 'E']]) < 2,
                 len(selection) == len(set(selection))]
        return all(conds)

    # @timing_decorator
    def transition_model(self, state, action):
        rnd, selection = state
        index_to_change, option = action

        selection = [option if i == index_to_change else
                     opt for i, opt in enumerate(selection)]
        selection.sort()
        next_state = (rnd + 1, tuple(selection))
        if not self.is_legal_state(next_state):
            return False
        return next_state
    # @timing_decorator
    def reward_function(self, state):
        rnd, selection = state

        score = 0
        for selct in selection:
            score += self.scores[selct][rnd - 1]
        return score

    # is_affordable()
    # max_teams()
    # positions_fit()


if __name__ == '__main__':
    # Example Data
    ALL_OPTIONS = ['A', 'B', 'C', 'D', "E"]
    ROUNDS = 5
    scores = {
        'A': {0: 2, 1: 6, 2: 5, 3: 6, 4: 40},
        'B': {0: 6, 1: 5, 2: 6, 3: 5, 4: 6},
        'C': {0: 1, 1: 8, 2: 1, 3: 8, 4: 9},
        'D': {0: 8, 1: 7, 2: 8, 3: 7, 4: 8},
        'E': {0: 4, 1: 7, 2: 5, 3: 33, 4: 4}
    }

    initial_selection_size = 2
    problem_obj = Example(ALL_OPTIONS, range(ROUNDS), scores, initial_selection_size)

    # initial_states = [x for x in list(combinations(ALL_OPTIONS, initial_selection_size))]
    #
    # states = list(product(range(ROUNDS), initial_states))
    Value_it = ValueIteration(problem_obj)
    v, policy, strat = Value_it.value_iteration()
    print(strat)
