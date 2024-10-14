import numpy as np
import random

from matplotlib import pyplot as plt

from fantasyPL.generic_code.Q_and_Sarsa import GenericAgent
from fantasyPL.generic_code.envs import SelectorGameMini
from fantasyPL.generic_code.plotting import plot_line_dict
from fantasyPL.generic_code.policy_iteration import PolicyIteration
from fantasyPL.generic_code.value_iteration import ValueIteration
from helper_methods import print_list_of_dicts


class QLearningAgent(GenericAgent):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)

    def calculate_td_target(self, state, action, reward, next_state, done, next_action=None) -> float:
        """Q-learning specific TD target calculation."""
        # Get the best action for the next state using the Q-learning approach
        best_next_action = self.q_table.get_best_action(self.env, next_state)
        if best_next_action is None:
            next_afterstate = None
        else:
            next_afterstate = self.q_table.afterstate(self.env, next_state, best_next_action)

        # Q-learning formula for TD target
        return reward + self.discount_factor * self.q_table.get_q_value(next_afterstate) * (1 - done)

    def learn(self, state, action, reward, next_state, done):
        """Update Q-values using the Q-learning formula."""
        super().learn(state, action, reward,
                     next_state, done, next_action=None)

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
    agent = QLearningAgent(env,epsilon_min=0.1,episodes=4000)
    agent.train(
        patience=10000, min_delta=1)
    # agent.print_policy()
    agent1 = QLearningAgent(env, epsilon_min=0.01, episodes=4000)
    agent1.train(
        patience=10000, min_delta=1)
    lines_dict = {
        'Line 1':[agent.plotter.moving_avg_graph._x, agent.plotter.moving_avg_graph._y],
        'Line 2':  [agent1.plotter.moving_avg_graph._x, agent1.plotter.moving_avg_graph._y],

    }
    plot_line_dict(lines_dict)
    q_strat,value = agent.run_policy(policy=agent.get_policy())
    print_list_of_dicts(q_strat)
    print(f"Total Reward = {value}")


