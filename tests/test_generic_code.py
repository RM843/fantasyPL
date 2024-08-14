from fantasyPL.generic_code.find_max_leaf_binary_tree import mcts_playout
from fantasyPL.generic_code.monte_carlo_pure import monte_carlo_policy_evaluation
from fantasyPL.generic_code.policy_iteration import PolicyIteration
from fantasyPL.generic_code.q_learning import QLearn
from fantasyPL.generic_code.reinforment_learning import Policy, INITIAL_STATE
from fantasyPL.generic_code.sarsa import Sarsa
from fantasyPL.generic_code.value_iteration import ValueIteration
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




class NewEnvironment:
    def __init__(self, problem_obj):
        self.problem_obj = problem_obj
        self.reset()

    def step(self, action):
        self.state = self.problem_obj.transition_model(self.state, action, start_state=self.state == INITIAL_STATE)
        reward = self.problem_obj.reward_function(self.state)

        return self.state, reward, self.state[0] == self.problem_obj.rounds.stop

    def reset(self):
        self.state = INITIAL_STATE
        return self.state


def check_policy_it_equals_value_it():
    # Example Data
    # ALL_OPTIONS = ['A', 'B', 'C', 'D', "E"]
    # ROUNDS = 5
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
    initial_selection_size = 2
    problem_obj = Example(ALL_OPTIONS, range(1, ROUNDS), scores, initial_selection_size)

    pi = PolicyIteration(problem_obj)
    vi = ValueIteration(problem_obj)
    v, policy, strat, best_score = pi.run(gamma = 1.0, epsilon = 0.0001)
    v2, policy2, strat2, best_score2 = vi.run(gamma = 1.0, epsilon = 0.0001)
    assert strat == strat2
    assert v == v2
    assert policy.policy == policy2.policy
    assert best_score == best_score2

    env = NewEnvironment(problem_obj=problem_obj)
    # Define the number of episodes for MC evaluation
    num_episodes = 1000

    policy = Policy(get_allowed_actions_method=ValueIteration(problem_obj).get_allowed_actions, random=True)
    # Evaluate the policy
    v = monte_carlo_policy_evaluation(policy, env, num_episodes)

    print("The value table is:")
    print(v)
    env = CliffWalkingEnv()
    sarsa = QLearn(problem_obj=problem_obj,env=env)
    v3, policy3, strat3, best_score3 = sarsa.run( exploration_rate=0.05, lr=0.2, discount_factor=1)

    # initial_selection = strat[0]["action"]
    # for explore in [200, 200,20,5,1]:
    #     print("Exploration weight = ",explore)
    #     mcts_playout( num_iter=500, num_rollout=10, exploration_weight=.2 ,
    #              problem_obj=problem_obj)


if __name__ == '__main__':
    check_policy_it_equals_value_it()
