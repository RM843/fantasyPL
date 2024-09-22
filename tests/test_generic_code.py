from fantasyPL.generic_code.envs import SelectorGameMini, CliffWalkingEnv
from fantasyPL.generic_code.find_max_leaf_binary_tree import mcts_playout
from fantasyPL.generic_code.monte_carlo_pure import monte_carlo_policy_evaluation
from fantasyPL.generic_code.policy_iteration import PolicyIteration
# from fantasyPL.generic_code.q_learning import QLearn
from fantasyPL.generic_code.reinforment_learning import Policy, INITIAL_STATE
# from fantasyPL.generic_code.sarsa import Sarsa
from fantasyPL.generic_code.value_iteration import ValueIteration
import numpy as np






class NewEnvironment:
    def __init__(self, problem_obj):
        self.problem_obj = problem_obj
        self.reset()

    def step(self, action):
        self.state = self.problem_obj.transition_model(self.state, action, start_state=self.state == INITIAL_STATE)
        reward = self.problem_obj.reward_function(self.state)

        return self.state, reward, self.state[0] == self.problem_obj.rounds.stop

    def reset(self):
        self.state = INITIAL_STATE
        return self.state


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
    problem_obj = SelectorGameMini(ALL_OPTIONS, range(1, ROUNDS), scores, initial_selection_size)

    pi = PolicyIteration(problem_obj)
    vi = ValueIteration(problem_obj)
    v, policy, strat, best_score = pi.run(gamma = 1.0, epsilon = 0.0001)
    v2, policy2, strat2, best_score2 = vi.run(gamma = 1.0, epsilon = 0.0001)
    assert strat == strat2
    assert v == v2
    assert policy.policy == policy2.policy
    assert best_score == best_score2

    env = NewEnvironment(problem_obj=problem_obj)
    # Define the number of episodes for MC evaluation
    num_episodes = 1000

    policy = Policy(get_allowed_actions_method=ValueIteration(problem_obj).get_allowed_actions, random=True)
    # Evaluate the policy
    v = monte_carlo_policy_evaluation(policy, env, num_episodes)

    print("The value table is:")
    print(v)
    env = CliffWalkingEnv()
    sarsa = QLearn(problem_obj=problem_obj,env=env)
    v3, policy3, strat3, best_score3 = sarsa.run( exploration_rate=0.05, lr=0.2, discount_factor=1)

    # initial_selection = strat[0]["action"]
    # for explore in [200, 200,20,5,1]:
    #     print("Exploration weight = ",explore)
    #     mcts_playout( num_iter=500, num_rollout=10, exploration_weight=.2 ,
    #              problem_obj=problem_obj)


if __name__ == '__main__':
    check_policy_it_equals_value_it()
