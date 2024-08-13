from fantasyPL.generic_code.find_max_leaf_binary_tree import mcts_playout
from fantasyPL.generic_code.monte_carlo_pure import monte_carlo_policy_evaluation, random_policy, NewEnvironment
from fantasyPL.generic_code.policy_iteration import PolicyIteration
from fantasyPL.generic_code.reinforment_learning import Policy, PolicyOrValueIteration
from fantasyPL.generic_code.value_iteration import ValueIteration


class Example:

    def __init__(self, all_options, rounds, scores, initial_selection_size):
        self.all_options = all_options
        self.rounds = rounds
        self.scores = scores
        self.initial_selection_size = initial_selection_size

    def is_legal_selection(self, selection):
        # rnd, selection = state
        conds = [len([x for x in selection if x in ['A', 'E']]) < 2,
                 len(selection) == len(set(selection))]
        return all(conds)

    # @timing_decorator
    def transition_model(self, state, action,start_state):
        assert not state[0] ==self.rounds.stop
        if start_state:
            rnd = self.rounds.start -1
            selection = action
            option = None
        else:
            rnd, selection = state
            index_to_change, option = action
        if option is not None:

            selection = list(selection)
            selection[index_to_change] = option
            selection.sort()
            selection = tuple(selection)


        next_state = (rnd + 1, tuple(selection))
        if not self.is_legal_selection(selection):
            return False
        return next_state

    # @timing_decorator
    def reward_function(self, state):
        rnd, selection = state

        score = 0

        for selct in selection:
            # initial state selction = value of 1st round scores
            # if rnd ==-1:
            #     rnd = list(self.scores[selct].keys())[0]
            score += self.scores[selct][rnd]
        return score


def check_policy_it_equals_value_it():
    # Example Data
    # ALL_OPTIONS = ['A', 'B', 'C', 'D', "E"]
    # ROUNDS = 5
    scores = {
        "A": {1: 2, 2: 335, 3: 1, 4: 16, 5: 5},
        "B": {1: 9, 2: 5, 3: 6, 4: 5, 5: 6},
        "C": {1: 1, 2: 8, 3: 1, 4: 8, 5: 9},
        "D": {1: 8, 2: 7, 3: 8, 4: 7, 5: 8},
        "E": {1: 4, 2: 7, 3: 5, 4: 8, 5: 4}
    }
    simple_scores = {
        "A": {1: 2, 2: 8},
        "B": {1: 9, 2: 5},
        "C": {1: 5, 2: 7}
    }
    # scores = simple_scores

    ALL_OPTIONS = list(scores.keys())
    ROUNDS = len(scores[ALL_OPTIONS[0]].keys())
    initial_selection_size = 2
    problem_obj = Example(ALL_OPTIONS, range(1, ROUNDS), scores, initial_selection_size)

    pi = PolicyIteration(problem_obj)
    vi = ValueIteration(problem_obj)
    v, policy, strat, best_score = pi.run()
    v2, policy2, strat2, best_score2 = vi.run()
    assert strat == strat2
    assert v == v2
    assert policy.policy == policy2.policy
    assert best_score == best_score2

    env = NewEnvironment(problem_obj=problem_obj)
    # Define the number of episodes for MC evaluation
    num_episodes = 1000

    policy = Policy(get_allowed_actions_method =ValueIteration(problem_obj).get_allowed_actions,random=True)
    # Evaluate the policy
    v = monte_carlo_policy_evaluation(policy, env, num_episodes)


    print("The value table is:")
    print(v)

    # initial_selection = strat[0]["action"]
    # for explore in [200, 200,20,5,1]:
    #     print("Exploration weight = ",explore)
    #     mcts_playout( num_iter=500, num_rollout=10, exploration_weight=.2 ,
    #              problem_obj=problem_obj)


if __name__ == '__main__':
    check_policy_it_equals_value_it()
