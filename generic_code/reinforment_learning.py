import random
from abc import ABC, abstractmethod

from tqdm import tqdm

from helper_methods import combination_generator, timing_decorator


class Policy:
    def __init__(self, get_allowed_actions_method):
        self.policy = {}
        self.get_allowed_actions = get_allowed_actions_method

    def get_action(self, state):
        if state not in self.policy:
            self.policy[state] = random.choice(self.get_allowed_actions(state)),None
        return  self.policy[state]


class PolicyOrValueIteration(ABC):
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
        self.policy = Policy(get_allowed_actions_method=self.get_allowed_actions)
        self.strat = []
        self.algo_run = False
        self.states_superset = []
        self.validate_problem_obj()  # Validate the problem_obj during initialization
        # self.selections_generator = self.get_selections_generator(self.problem_obj.all_options,
        #                                                 self.problem_obj.initial_selection_size)
    @abstractmethod
    def algorithm(self,gamma,epsilon):
        ''''''
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

    @timing_decorator
    def run(self, gamma: float = 1.0, epsilon: float = 0.0001) -> tuple:
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
        self.algorithm(gamma=gamma,epsilon=epsilon)
        self.algo_run = True
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
            action_values.append(
                self.get_action_value(state, action, gamma))  # Calculate the value of taking this action

        # Identify the best action and its corresponding value
        best_action_value = max(action_values)  # Maximum value among all evaluated actions
        best_action = available_actions[action_values.index(best_action_value)]  # Corresponding action

        return best_action, best_action_value

    def get_action_value(self, state: any, action: any, gamma: float) -> float:
        """
        Calculate the value of taking a specific action in a given state.

        This function computes the expected value of performing an action in a
        given state, based on the transition model, reward function, and the
        current value function. The value is calculated as the sum of the
        immediate reward and the discounted value of the resulting next state.

        Args:
            state (any): The current state.
            action (any): The action to be taken in the current state.
            gamma (float): The discount factor, a value between 0 and 1, that
                           represents the importance of future rewards.

        Returns:
            float: The calculated value of taking the specified action in the
                   given state.
        """

        next_state = self.problem_obj.transition_model(state, action)  # Determine the next state
        reward = self.problem_obj.reward_function(next_state)  # Calculate the reward for the next state
        next_state_value = self.V.get(next_state, 0)  # Get the value of the next state, defaulting to 0
        return gamma * next_state_value + reward

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

    # @timing_decorator
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
        assert self.algo_run, "Algorithm has not been run. Cannot determine the best initial state."

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
            action, value = self.policy.policy[state]  # Get the best action and its value for the current state

            # Append the current state, action, and value to the strategy if not in the initial round
            if rnd != 0:
                self.strat.append({"state": state, "action": action, "value": value})

            # Transition to the next state based on the chosen action
            state = self.problem_obj.transition_model(state, action)

            # If we were in the initial round, record this transition without an action
            if rnd == 0:
                self.strat.append({"state": (0, state[1]), "action": None, "value": value})
