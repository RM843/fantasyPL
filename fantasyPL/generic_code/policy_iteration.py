from itertools import product

from tqdm import tqdm

from fantasyPL.generic_code.reinforment_learning import PolicyOrValueIteration


class PolicyIteration(PolicyOrValueIteration):
    """
    A class that implements the Policy Iteration algorithm for solving Markov Decision Processes (MDPs).
    This algorithm alternates between policy evaluation and policy improvement until convergence.

    Attributes:
        V (dict): A dictionary representing the value function, mapping states to their values.
        policy (Policy): A policy object that defines the actions to take in each state.
        problem_obj (Problem): An object that defines the MDP, including the state space, actions, transition model, and reward function.
    """

    def algorithm(self, gamma: float = 1.0, epsilon: float = 0.0001):
        """
        Executes the Policy Iteration algorithm.

        The algorithm alternates between policy evaluation and policy improvement until the policy becomes stable.
        Policy evaluation updates the value function for the current policy, while policy improvement updates the policy
        based on the updated value function.

        Args:
            gamma (float): The discount factor, a value between 0 and 1, that represents the importance of future rewards. Default is 1.0.
            epsilon (float): A small threshold used to determine when the value function has sufficiently converged. Default is 0.0001.
        """
        states = [x for x in self.get_selections_generator(self.problem_obj.all_options,
                                                           self.problem_obj.initial_selection_size)]
        states = [x for x in product(self.problem_obj.rounds, states) if self.problem_obj.is_legal_state(x)]
        while True:
            # Policy Evaluation
            delta = epsilon * 10
            while delta > epsilon:
                delta = 0

                for state in tqdm(states):
                    # for rnd in self.problem_obj.rounds:
                    #     state = (rnd, selection)

                    # if not self.problem_obj.is_legal_state(state):
                    #     continue

                    v = self.V.get(state, 0)  # Current value of the state
                    action, _ = self.policy.get_action(state)  # Get the action according to the current policy
                    action_value = self.get_action_value(state=state, action=action, gamma=gamma)
                    self.V[state] = action_value  # Update the state value with the computed action value

                    # Calculate the maximum change in value for the current state
                    delta = max(delta, abs(v - self.V[state]))
                        # print("Delta:", delta)

            # Policy Improvement
            policy_stable = True
            for state in tqdm(states):
                # for rnd in self.problem_obj.rounds:
                    # state = (rnd, selection)

                    # if not self.problem_obj.is_legal_state(state):
                    #     continue

                old_action, _ = self.policy.get_action(state)  # Get the current action for the state
                best_action, best_action_value = self.get_best_action(state, gamma)  # Find the best action
                self.policy.policy[state] = best_action, best_action_value  # Update the policy with the best action

                if old_action != best_action:
                    print("Policy not yet stable")
                    policy_stable = False
                    break# If any action changes, the policy is not stable yet

            # If the policy is stable, the algorithm has converged
            if policy_stable:
                return

