import numpy as np
import random

from matplotlib import pyplot as plt

from fantasyPL.generic_code.Q_and_Sarsa import GenericAgent
from fantasyPL.generic_code.envs import SelectorGameMini
from fantasyPL.generic_code.policy_iteration import PolicyIteration
from fantasyPL.generic_code.value_iteration import ValueIteration
from helper_methods import print_list_of_dicts


class QLearningAgent(GenericAgent):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)

    def learn(self, state, action, reward, next_state, done):
        """Update Q-values using the Q-learning formula."""
        super().learn(state, action, reward,
                     next_state, done, next_action=None, sarsa=False)

    def train(self,  max_steps=100, patience=10, min_delta=1.0):
        """Train the SARSA agent for a specified number of episodes."""
        super().train( max_steps, patience, min_delta, sarsa=False)


if __name__ == '__main__':
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
    initial_selection_size = 4
    env = SelectorGameMini(ALL_OPTIONS, range(1, ROUNDS), scores, initial_selection_size)

    pi = PolicyIteration(env)
    vi = ValueIteration(env)
    v, policy, strat, best_score = pi.run(gamma=1.0, epsilon=0.0001)
    v2, policy2, strat2, best_score2 = vi.run(gamma=1.0, epsilon=0.0001)
    assert strat == strat2
    assert v == v2
    assert policy.policy == policy2.policy
    assert best_score == best_score2





    # Example usage with the CliffWalkingEnv:
    # env = CliffWalkingEnv()
    agent = QLearningAgent(env,epsilon_min=0.1,episodes=400000)
    agent.train(
        patience=10000, min_delta=1)
    # agent.print_policy()
    q_strat,value = agent.run_policy(policy=agent.get_policy())
    print_list_of_dicts(q_strat)
    print(f"Total Reward = {value}")


