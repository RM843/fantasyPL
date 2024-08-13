import random
from itertools import product

from tqdm import tqdm

from fantasyPL.generic_code.reinforment_learning import PolicyOrValueIteration


class Sarsa(PolicyOrValueIteration):
    def __init__(self, problem_obj, env):

        super().__init__(problem_obj)
        self.env = env

    def algorithm(self, exploration_rate, lr, discount_factor=0.99):
        """

        """
        # self.states_superset = self.get_states_superset()
        best_score_so_far = float("-inf")
        best_strat = None
        while True:
            state = self.env.reset()
            action = self.get_epsilon_greedy_action( state, exploration_rate)

            while True:
                next_state, reward, terminal = self.env.step(action)
                if not self.is_terminal(next_state):
                    next_action = self.get_epsilon_greedy_action( next_state, exploration_rate)
                else:
                    next_action=""

                self.V[(state, action)] = (self.V.get((state, action),0) +
                                           discount_factor * (reward + self.V.get((next_state, next_action),0) -
                                            self.V.get((state, action),0)))  # Update the state value with the best action value
                state, action = next_state, next_action
                if self.is_terminal(state):
                    break

            self.strat = self.get_strat()  # Derive the strategy from the optimal policy
            final_score = self.eval_strat()
            print(final_score)
            if final_score==413:
                return
    def get_epsilon_greedy_action(self,state,exploration_rate):
        # explore
        if random.random() < exploration_rate:
            next_action = random.choice(self.get_allowed_actions(state))
        # exploit
        else:
            next_action, _ = self.policy.get_action(state)
        return next_action

    def is_terminal(self,state):
        return state[0] == self.problem_obj.rounds.stop