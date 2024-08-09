from itertools import product

from tqdm import tqdm

from fantasyPL.generic_code.reinforment_learning import PolicyOrValueIteration


class ValueIteration(PolicyOrValueIteration):
    """
    Implements the Value Iteration algorithm for solving Markov Decision Processes (MDPs).

    The Value Iteration algorithm is a method for finding the optimal policy and value function
    in a Markov Decision Process. It iteratively updates the value function for each state
    based on the expected future rewards, until the value function converges to a stable set of values.
    Once the value function has converged, the optimal policy is extracted based on the computed values.
    """
    def algorithm(self, gamma: float = 1.0, epsilon: float = 0.0001):
        """
        Perform value iteration to find the optimal value function and policy.

        Args:
            gamma (float, optional): The discount factor for future rewards. Default is 1.0.
            epsilon (float, optional): A small threshold for determining convergence. Default is 0.0001.

        Returns:
            tuple: A tuple containing:
                - V (dict): The optimal value function mapping each state to its value.
                - policy (dict): The optimal policy mapping each state to its best action.
                - strat (list): The strategy derived from the optimal policy.
        """
        states = [x for x in self.get_selections_generator(self.problem_obj.all_options,
                                                           self.problem_obj.initial_selection_size)]
        states = [x for x in product(self.problem_obj.rounds, states) if self.problem_obj.is_legal_state(x)]
        while True:

            deltas = 0  # Total change in value across all states in this iteration
            for state in tqdm(states):
                # for rnd in self.problem_obj.rounds:

                def do_loop():

                    # state = (rnd, selection)

                    # if not self.problem_obj.is_legal_state(state):
                    #     return 0
                    self.states_superset.append(state)

                    delta = 0  # Change in value for the current state
                    v = self.V.get(state, 0)  # Current value of the state
                    best_action, best_action_value = self.get_best_action(state, gamma)
                    self.V[state] = best_action_value  # Update the state value with the best action value
                    # Calculate the maximum change in value for the current state
                    return max(delta, abs(v - self.V[state]))

                # Accumulate the changes across all states
                deltas += do_loop()

            print("Deltas:", deltas)

            # Check for convergence: if the total change in value is less than epsilon, stop iterating
            if deltas < epsilon:
                break

        # Ensure the number of computed states matches the expected number of states
        assert len(set(self.states_superset)) == len(self.V), \
            f"""{len(set(self.states_superset))} possible
                           states {len(self.V)} found"""

        print("Getting Policy")
        # Extract the optimal policy based on the final value function
        for s in tqdm(states):
            self.policy.policy[s] = self.get_best_action(s, gamma)