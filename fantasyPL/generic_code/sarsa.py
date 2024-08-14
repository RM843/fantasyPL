import random
from itertools import product
from typing import List
import matplotlib.pyplot as plt

from tqdm import tqdm

from fantasyPL.generic_code.q_learning import QLearn
from fantasyPL.generic_code.reinforment_learning import PolicyOrValueIteration


class Sarsa(QLearn):
    def __init__(self, problem_obj, env):

        super().__init__( problem_obj, env)

    def algorithm(self, exploration_rate, lr, discount_factor=0.99):
        """

        """
        all_rewards = []
        while True:
            state = self.env.reset()
            action = self.get_epsilon_greedy_action( state, exploration_rate)
            ep_rewards = 0
            while True:
                self.env.render()
                next_state, reward, terminal = self.env.step(action)
                ep_rewards += reward
                if not self.env.is_terminal(next_state):
                    next_action = self.get_epsilon_greedy_action( next_state, exploration_rate)
                else:
                    next_action=""

                self.V[(state, action)] = (self.V.get((state, action),0) +
                                           lr * (reward + discount_factor*self.V.get((next_state, next_action),0) -
                                            self.V.get((state, action),0)))  # Update the state value with the best action value
                state, action = next_state, next_action
                if self.env.is_terminal(state):
                    break
            all_rewards.append(ep_rewards)
            self.plot_rewards( ep_rewards)
