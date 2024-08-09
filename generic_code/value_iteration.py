from tqdm import tqdm

from helper_methods import combination_generator, timing_decorator


class ValueIteration:
    """
    Implements the Value Iteration algorithm for solving Markov Decision Processes (MDPs).

    The Value Iteration algorithm is a method for finding the optimal policy and value function
    in a Markov Decision Process. It iteratively updates the value function for each state
    based on the expected future rewards, until the value function converges to a stable set of values.
    Once the value function has converged, the optimal policy is extracted based on the computed values.

    Attributes:
        problem_obj (object): An object representing the problem domain, which must provide
            methods for transition modeling, reward function, and legality checks.
        V (dict): A dictionary mapping each state to its computed value after convergence.
        policy (dict): A dictionary mapping each state to the best action according to the optimal policy.
        strat (list): A list of actions and state transitions that represent the optimal strategy derived from the value iteration.
        value_it_run (bool): A flag indicating whether the value iteration process has been completed.

    Methods:
        value_iteration(states, gamma=1.0, epsilon=0.0001):
            Executes the value iteration algorithm to compute the optimal value function and policy.
        get_best_action(state, gamma):
            Determines the best action for a given state based on the current value function.
        is_allowed_action(state, action):
            Checks if a specific action is valid in a given state.
        get_allowed_actions(state):
            Retrieves all actions that are permissible in a given state.
        is_final(state):
            Determines if a given state is a final state in the MDP.
        get_best_initial_state():
            Identifies the best initial state based on the highest value function score.
        get_strat():
            Constructs the optimal strategy from the computed value function and policy.
    """

    def __init__(self, problem_obj):
        self.problem_obj = problem_obj
        self.V = {}
        self.policy = {}
        self.strat = []
        self.value_it_run = False
        self.states_superset = []
        self.validate_problem_obj()  # Validate the problem_obj during initialization
        # self.selections_generator = self.get_selections_generator(self.problem_obj.all_options,
        #                                                 self.problem_obj.initial_selection_size)

    def validate_problem_obj(self):
        """Check if the problem_obj has the required methods and attributes."""
        required_methods = [
            'transition_model',  # Method to model transitions between states
            'reward_function',  # Method to calculate rewards
            'is_legal_state'  # Method to check if a selection is legal
        ]
        required_attributes = [
            'all_options',  # Attribute containing all possible options
            'rounds',  # Attribute specifying the number of rounds
            'initial_selection_size'
        ]

        # Check if all required methods exist
        for method in required_methods:
            if not callable(getattr(self.problem_obj, method, None)):
                raise TypeError(f"problem_obj is missing the required method: '{method}'")

        # Check if all required attributes exist
        for attr in required_attributes:
            if not hasattr(self.problem_obj, attr):
                raise TypeError(f"problem_obj is missing the required attribute: '{attr}'")

    def get_selections_generator(self, options, selection_size):

        return combination_generator(options, selection_size)
        # initial_states = [x for x in list(combinations(options, selection_size))]
        #
        # states = list(product(range(self.problem_obj.rounds), initial_states))
        # # if any states are invalid, remove them
        # return [s for s in states if self.problem_obj.is_legal_selection(s)]

    @timing_decorator
    def value_iteration(self, gamma: float = 1.0, epsilon: float = 0.0001) -> tuple:
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

        while True:
            deltas = 0  # Total change in value across all states in this iteration
            for selection in tqdm(self.get_selections_generator(self.problem_obj.all_options,
                                                           self.problem_obj.initial_selection_size)):
                for rnd in self.problem_obj.rounds:


                    @timing_decorator
                    def do_loop():

                        state = (rnd, selection)

                        if not self.problem_obj.is_legal_state(state):
                            return 0
                        self.states_superset.append(state)

                        delta = 0  # Change in value for the current state
                        v = self.V.get(state, 0)  # Current value of the state
                        best_action, best_action_value = self.get_best_action(state, gamma)
                        self.V[state] = best_action_value  # Update the state value with the best action value
                        # Calculate the maximum change in value for the current state
                        return max(delta, abs(v - self.V[state]))

                    # Accumulate the changes across all states
                    deltas +=do_loop()



            print("Deltas:", deltas)

            # Check for convergence: if the total change in value is less than epsilon, stop iterating
            if deltas < epsilon:
                break

        # Ensure the number of computed states matches the expected number of states
        assert len(set(self.states_superset)) == len(self.V), \
            f"""{len(set(self.states_superset))} possible
                   states {len(self.V)} found"""

        # Extract the optimal policy based on the final value function

        for s in self.states_superset:
            self.policy[s] = self.get_best_action(s, gamma)

        self.value_it_run = True
        self.get_strat()  # Derive the strategy from the optimal policy
        return self.V, self.policy, self.strat
    @timing_decorator
    def get_best_action(self, state: tuple, gamma: float) -> tuple:
        """
        Determine the best action to take in a given state based on the value iteration process.

        Args:
            state (tuple): The current state for which the best action is to be determined.
            gamma (float): The discount factor for future rewards.

        Returns:
            tuple: A tuple containing:
                - best_action (tuple): The action that yields the highest expected value.
                - best_action_value (float): The value associated with the best action.
        """

        # Get all available actions for the current state
        available_actions = self.get_allowed_actions(state)
        action_values = []

        # Evaluate the value of each available action
        for action in available_actions:
            next_state = self.problem_obj.transition_model(state, action)  # Determine the next state
            reward = self.problem_obj.reward_function(next_state)  # Calculate the reward for the next state
            next_state_value = self.V.get(next_state, 0)  # Get the value of the next state, defaulting to 0
            action_values.append(gamma * next_state_value + reward)  # Calculate the value of taking this action

        # Identify the best action and its corresponding value
        best_action_value = max(action_values)  # Maximum value among all evaluated actions
        best_action = available_actions[action_values.index(best_action_value)]  # Corresponding action

        return best_action, best_action_value

    def is_allowed_action(self, state: tuple, action: tuple) -> bool:
        """
        Check if a given action is allowed in a specific state.

        Args:
            state (tuple): The current state in which the action is being considered.
            action (tuple): The action to be evaluated.

        Returns:
            bool: True if the action is allowed (i.e., leads to a legal next state), False otherwise.
        """

        # returns False if not allowed, so we check if it is false
        return not not self.problem_obj.transition_model(state, action)
    @timing_decorator
    def get_allowed_actions(self, state: tuple) -> list:
        """
        Retrieve all allowed actions for a given state.

        Args:
            state (tuple): The current state for which allowed actions are being determined.

        Returns:
            list: A list of allowed actions. Each action is represented as a tuple.
        """

        actions = [(-1, None)]  # Include the "do nothing" option by default

        rnd, selection = state  # Unpack the current state into round number and selection

        # Iterate through each item in the selection
        for i, _ in enumerate(selection):
            # Iterate through each option available in the problem
            for opt in self.problem_obj.all_options:
                # Check if the action (i.e., selecting the option `opt` at position `i`) is allowed
                if self.is_allowed_action(state, (i, opt)):
                    actions.append((i, opt))  # If allowed, add the action to the list of actions

        return actions  # Return the list of allowed actions

    def is_final(self, state: tuple) -> bool:
        """
        Determine if a given state is a final state.

        Args:
            state (tuple): The state to be evaluated, typically consisting of a round number and a selection.

        Returns:
            bool: True if the state is a final state, False otherwise.
        """

        rnd, selection = state  # Unpack the state into round number and selection

        # Check if the current round number is greater than or equal to the total number of rounds
        return rnd >= len([x for x in self.problem_obj.rounds])

    def get_best_initial_state(self) -> tuple:
        """
        Retrieve the best initial state based on the highest value function score.

        Returns:
            tuple: The state with the highest value among the initial states (where round number is 0).

        Raises:
            AssertionError: If value iteration has not been run (`self.value_it_run` is False).
        """

        # Ensure that the value iteration process has been completed
        assert self.value_it_run, "Value iteration has not been run. Cannot determine the best initial state."

        # Filter out states where the round number is 0 (initial states) and get their values
        initial_state_scores = [(k, v) for k, v in self.V.items() if k[0] == 0]

        # Extract the values from the filtered states
        values = [v for (k, v) in initial_state_scores]

        # Find the index of the state with the maximum value
        max_value_index = values.index(max(values))

        # Return the state with the highest value
        return initial_state_scores[max_value_index][0]

    def get_strat(self) -> None:
        """
        Generate the strategy based on the optimal policy and store it in `self.strat`.

        This method constructs the strategy by starting from the best initial state and
        following the policy until a final state is reached. It updates `self.strat` with
        the sequence of states, actions, and values.

        Returns:
            None: This method updates the `self.strat` attribute and does not return any value.
        """

        # Retrieve the best initial state based on the value iteration results
        state = self.get_best_initial_state()

        self.strat = []  # Initialize the strategy list

        # Continue generating the strategy until reaching a final state
        while not self.is_final(state):
            rnd, selection = state  # Unpack the current state
            action, value = self.policy[state]  # Get the best action and its value for the current state

            # Append the current state, action, and value to the strategy if not in the initial round
            if rnd != 0:
                self.strat.append({"state": state, "action": action, "value": value})

            # Transition to the next state based on the chosen action
            state = self.problem_obj.transition_model(state, action)

            # If we were in the initial round, record this transition without an action
            if rnd == 0:
                self.strat.append({"state": (0, state[1]), "action": None, "value": value})
