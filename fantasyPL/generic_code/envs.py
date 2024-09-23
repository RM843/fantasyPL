import numpy as np

from helper_methods import combination_generator


class CliffWalkingEnv:
    def __init__(self, width=12, height=4, start=(3, 0), goal=(3, 11)):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.cliff = [(3, i) for i in range(1, self.width - 1)]
        self.state_size = self.width * self.height  # Number of discrete states
        self.action_size = 4  # Four actions: Up, Right, Down, Left
        self.reset()

    def reset(self):
        """Reset the environment to the start state."""
        self.agent_position = self.start
        return self.agent_position

    def step(self, action):
        """Take an action in the environment.

        Actions:
        - 0: Up
        - 1: Right
        - 2: Down
        - 3: Left

        Returns:
        - next_state: The new agent position
        - reward: The reward after taking the action
        - done: Whether the episode has ended
        """
        x, y = self.agent_position

        # Define actions
        if action == 0:  # Up
            x = max(x - 1, 0)
        elif action == 1:  # Right
            y = min(y + 1, self.width - 1)
        elif action == 2:  # Down
            x = min(x + 1, self.height - 1)
        elif action == 3:  # Left
            y = max(y - 1, 0)

        # Update position
        self.agent_position = (x, y)

        # Determine reward and terminal state
        if self.agent_position in self.cliff:
            reward = -100
            done = True
            next_state = self.start
        elif self.agent_position == self.goal:
            reward = 0
            done = True
            next_state = self.goal
        else:
            reward = -1
            done = False
            next_state = self.agent_position

        return next_state, reward, done

    def get_allowed_actions(self, state):
        """Return a list of allowed actions for a given state.

        Actions:
        - 0: Up
        - 1: Right
        - 2: Down
        - 3: Left

        Parameters:
        - state: The current state index.

        Returns:
        - allowed_actions: A list of actions that can be taken from the given state.
        """
        x, y = state
        allowed_actions = []

        # Check if the movement is within bounds
        if x > 0:  # Can move up
            allowed_actions.append(0)
        if y < self.width - 1:  # Can move right
            allowed_actions.append(1)
        if x < self.height - 1:  # Can move down
            allowed_actions.append(2)
        if y > 0:  # Can move left
            allowed_actions.append(3)

        return allowed_actions

    def is_terminal(self, state):
        """Check if the given state is terminal.

        Parameters:
        - state: The current state index.

        Returns:
        - True if the state is terminal (goal or cliff), otherwise False.
        """
        position = state
        return position == self.goal or position in self.cliff

    def _state_to_index(self, state):
        """Convert (x, y) state to a single index."""
        x, y = state
        return x * self.width + y

    def _index_to_state(self, index):
        """Convert a single index to (x, y) state."""
        x = index // self.width
        y = index % self.width
        return (x, y)

    def render(self):
        """Render the current state of the grid."""
        grid = np.full((self.height, self.width), ' ')
        grid[3, 1:self.width - 1] = 'C'  # Cliff
        grid[self.goal] = 'G'
        grid[self.start] = 'S'
        x, y = self.agent_position
        grid[x, y] = 'A'

        print('\n'.join([''.join(row) for row in grid]))
        print()

class SelectorGameMini:

    def __init__(self, all_options, rounds, scores, initial_selection_size):
        self.all_options = all_options
        self.rounds = rounds
        self.scores = scores
        self.initial_selection_size = initial_selection_size
        self.selections_superset = None
        self._initial_state = "INITIAL"
        self._do_nothing_action = (-1, None)

        self.reset()

    @property
    def initial_state(self):
        """Read-only property."""
        return self._initial_state

    @property
    def do_nothing_action(self):
        """Read-only property."""
        return self._do_nothing_action


    def reset(self):
        self.state=self.initial_state
        return self.state
    def is_legal_selection(self, selection):
        # rnd, selection = state
        conds = [len([x for x in selection if x in ['A', 'E']]) < 2,
                 len(selection) == len(set(selection))]
        return all(conds)

    # @timing_decorator
    def transition_model(self, state, action, start_state=None):
        assert not state[0] == self.rounds.stop
        if self.initial_state==state:
            rnd = self.rounds.start - 1
            selection = action
            option = None
        else:
            rnd, selection = state
            index_to_change, option = action
        if option is not None:
            selection = list(selection)
            selection[index_to_change] = option
            selection.sort()
            selection = tuple(selection)

        next_state = (rnd + 1, tuple(selection))
        if not self.is_legal_selection(selection):
            return False
        return next_state
    def step(self,action):
        start_state = self.state == self.initial_state
        next_state = self.transition_model(self.state, action,
                              start_state=start_state)

        if start_state:
            reward = self.reward_function((self.rounds.start,action))
        else:
            reward = self.reward_function(next_state)
        self.state = next_state
        return  next_state, reward, self.is_terminal(next_state)
    def get_selections_generator(self, options, selection_size):

        return combination_generator(options, selection_size)

    def get_selections_superset(self, options_limit=None):
        if self.selections_superset is None:
            self.selections_superset = [x for x in self.get_selections_generator(self.all_options,
                                                                             self.initial_selection_size)
                                    if self.is_legal_selection(x)]
        return self.selections_superset
    # @timing_decorator
    def reward_function(self, state):
        rnd, selection = state

        score = 0

        for selct in selection:
            score += self.scores[selct][rnd]
        return score
    def is_terminal(self,state):
        return state[0] == self.rounds.stop
    def get_allowed_actions(self,state):
        if state == self.initial_state:
            return self.get_selections_superset()
        if self.is_terminal(state):
            return []

        actions = [self.do_nothing_action]  # Include the "do nothing" option by default

        rnd, selection = state  # Unpack the current state into round number and selection

        # Iterate through each item in the selection
        for i, _ in enumerate(selection):
            # Iterate through each option available in the problem
            for opt in self.all_options:
                # Check if the action (i.e., selecting the option `opt` at position `i`) is allowed
                if self.is_allowed_action(state, (i, opt)):
                    actions.append((i, opt))  # If allowed, add the action to the list of actions

        return actions  # Return the list of allowed actions

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
        next_state = self.transition_model(state, action,
                                           start_state=state == self.initial_state)
        if not next_state:
            return False
        # don't want duplicate do nothing actions
        if next_state == state and action != self.do_nothing_action:
            return False
        return True