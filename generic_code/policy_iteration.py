from tqdm import tqdm

from generic_code.reinforment_learning import PolicyOrValueIteration


class PolicyIteration(PolicyOrValueIteration):
    def algorithm(self, gamma: float = 1.0, epsilon: float = 0.0001):

        while True:
            delta = epsilon * 10
            while delta > epsilon:
                delta = 0
                for selection in tqdm(self.get_selections_generator(self.problem_obj.all_options,
                                                                    self.problem_obj.initial_selection_size)):
                    for rnd in self.problem_obj.rounds:

                        state = (rnd, selection)

                        if not self.problem_obj.is_legal_state(state):
                            continue
                        # self.states_superset.append(state)

                        v = self.V.get(state, 0)  # Current value of the state
                        action,_ = self.policy.get_action(state)
                        action_value = self.get_action_value(state=state,
                                                             action=action,
                                                             gamma=gamma)
                        self.V[state] = action_value  # Update the state value with the best action value
                        # Calculate the maximum change in value for the current state
                        delta = max(delta, abs(v - self.V[state]))
                        print("Delta:", delta)

            policy_stable = True
            for selection in tqdm(self.get_selections_generator(self.problem_obj.all_options,
                                                                self.problem_obj.initial_selection_size)):
                for rnd in self.problem_obj.rounds:
                    state = (rnd, selection)
                    if not self.problem_obj.is_legal_state(state):
                        continue
                    old_action,_ = self.policy.get_action(state)
                    best_action, best_action_value = self.get_best_action(state, gamma)
                    self.policy.policy[state] = best_action, best_action_value
                    if old_action != best_action:
                        policy_stable = False

            if policy_stable:
                return
