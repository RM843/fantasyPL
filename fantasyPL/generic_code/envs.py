import numpy as np


class CliffWalkingEnv:
    def __init__(self, width=12, height=4, start=(3, 0), goal=(3, 11)):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.cliff = [(3, i) for i in range(1, self.width - 1)]
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

        if action == 0:  # Up
            x = max(x - 1, 0)
        elif action == 1:  # Right
            y = min(y + 1, self.width - 1)
        elif action == 2:  # Down
            x = min(x + 1, self.height - 1)
        elif action == 3:  # Left
            y = max(y - 1, 0)

        self.agent_position = (x, y)

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

    def get_allowed_actions(self, state):
        """Return a list of allowed actions for a given state.

        Actions:
        - 0: Up
        - 1: Right
        - 2: Down
        - 3: Left

        Parameters:
        - state: The current position (x, y) of the agent.

        Returns:
        - allowed_actions: A list of actions that can be taken from the given state.
        """
        x, y = state
        allowed_actions = []

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
        - state: The current position (x, y) of the agent.

        Returns:
        - True if the state is terminal (goal or cliff), otherwise False.
        """
        if state == self.goal or state in self.cliff:
            return True
        return False

class Example:

    def __init__(self, all_options, rounds, scores, initial_selection_size):
        self.all_options = all_options
        self.rounds = rounds
        self.scores = scores
        self.initial_selection_size = initial_selection_size

    def is_legal_selection(self, selection):
        # rnd, selection = state
        conds = [len([x for x in selection if x in ['A', 'E']]) < 2,
                 len(selection) == len(set(selection))]
        return all(conds)

    # @timing_decorator
    def transition_model(self, state, action, start_state):
        assert not state[0] == self.rounds.stop
        if start_state:
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

    # @timing_decorator
    def reward_function(self, state):
        rnd, selection = state

        score = 0

        for selct in selection:
            # initial state selction = value of 1st round scores
            # if rnd ==-1:
            #     rnd = list(self.scores[selct].keys())[0]
            score += self.scores[selct][rnd]
        return score
    def is_terminal(self,state):
        return state[0] == self.rounds.stop