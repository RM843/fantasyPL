import random
import time
from collections import Counter
from typing import Dict, Any, Optional

import numpy as np

from fantasyPL.generic_code.Q_and_Sarsa import GenericAgent, REFRESH_RATE
from fantasyPL.generic_code.envs import SelectorGameMini
from fantasyPL.generic_code.plotting import plot_line_dict
from fantasyPL.generic_code.q_learning import QLearningAgent
from fantasyPL.generic_code.q_table import QTable
from helper_methods import print_list_of_dicts


import numpy as np
import random
from typing import Any, Dict, Optional, Tuple, List
from fantasyPL.generic_code.q_table import QTable


class DoubleQLearningAgent(GenericAgent):
    def __init__(self, env: Any, **kwargs):
        super().__init__(env, **kwargs)
        # Initialize two Q-tables for Double Q-Learning
        self.q_table1 = QTable()
        self.q_table2 = QTable()
        # To keep track of which Q-table to update
        self.update_table = 1  # Start with Q1

    def _best_action(self, state: Any, allowed_actions=None) -> Any:
        """Select the best action based on the sum of Q1 and Q2."""
        q_values = {}
        if self.env.is_terminal(state):
            return None
        if allowed_actions is None:
            allowed_actions = self.env.get_allowed_actions(state)


        for action in allowed_actions:
            afterstate = self.q_table1.afterstate(self.env, state, action)
            q1 = self.q_table1.get_q_value(afterstate)
            q2 = self.q_table2.get_q_value(afterstate)
            q_values[action] = q1 + q2  # Combine Q-values
        max_value = max(q_values.values())
        # In case multiple actions have the same max value, randomly choose among them
        best_actions = [action for action, value in q_values.items() if value == max_value]
        return random.choice(best_actions)

    # def get_q_values(self, state: Any) -> Dict[Any, float]:
    #     """
    #     Retrieve Q-values for all allowed actions in the current state
    #     from the single Q-table.
    #     """
    #
    #     q1 = self.q_table1.get_q_values(state=state, env=self.env)
    #     q2 = self.q_table2.get_q_values(state=state, env=self.env)
    #     return {key: q1.get(key, 0) + q2.get(key, 0) for key in set(q1) | set(q2)}
    def learn(
            self,
            state: Any,
            action: Any,
            reward: float,
            next_state: Any,
            done: bool,
            next_action: Optional[Any] = None,
            sarsa: bool = False
    ):
        """Generalized learn method with TD target calculation for Double Q-learning."""
        start_time = time.time()

        # Validate and initialize Q-values
        self._initialize(state, next_state, action)

        # Choose a random action for next_state from allowed actions
        allowed_actions = self.env.get_allowed_actions(next_state)
        next_action = random.choice(allowed_actions) if allowed_actions else None

        # Compute the TD target using both Q-tables
        td_target = self.calculate_td_target(state, action, reward, next_state, done, next_action)

        # Update the appropriate Q-table with the calculated TD target
        self._update(state, action, td_target, next_state)

        # Decay epsilon
        if done:
            current_episode = len(self.plotter.x)
            self.epsilon_decay.decay(done, current_episode)

        self.time_tracker.add_time('learning', time.time() - start_time)

    def _initialize(self, state: Any, next_state: Any, action: Any):
        """Initialize Q-values for both tables and validate the action."""
        self.q_table1.initialize_q_value(state)
        self.q_table1.initialize_q_value(next_state)
        self.q_table2.initialize_q_value(state)
        self.q_table2.initialize_q_value(next_state)
        self.validate_action(state, action)

    def calculate_td_target(
            self,
            state: Any,
            action: Any,
            reward: float,
            next_state: Any,
            done: bool,
            next_action: Optional[Any]
    ) -> float:
        """Compute the TD target for Double Q-learning."""

        next_action = self._best_action(next_state)
        if self.update_table == 1:
            q_table_primary = self.q_table1
            q_table_secondary = self.q_table2
        else:
            q_table_primary = self.q_table2
            q_table_secondary = self.q_table1

        current_afterstate = q_table_primary.afterstate(self.env, state, action)
        if current_afterstate is None:
            return 0.0

        if next_action is not None:
            next_afterstate = q_table_secondary.afterstate(self.env, next_state, next_action)
            max_q_secondary = q_table_secondary.get_q_value(next_afterstate)
        else:
            max_q_secondary = 0.0

        # Calculate TD target
        td_target = reward + self.discount_factor * max_q_secondary * (1 - done)
        return td_target

    def _update(self, state: Any, action: Any, td_target: float, next_state: Any):
        """Update the appropriate Q-table based on the calculated TD target."""
        if self.update_table == 1:
            self._update_q_table(self.q_table1, state, action, td_target)
            self.update_table = 2  # Switch to the other table next time
        else:
            self._update_q_table(self.q_table2, state, action, td_target)
            self.update_table = 1  # Switch back

    def _update_q_table(
            self,
            q_table: QTable,
            state: Any,
            action: Any,
            td_target: float
    ):
        """Update the Q-value in the specified Q-table based on the TD target."""
        current_afterstate = q_table.afterstate(self.env, state, action)
        if current_afterstate is None:
            return

        # Update Q-value based on TD error
        td_error = td_target - q_table.get_q_value(current_afterstate)
        new_q_value = q_table.get_q_value(current_afterstate) + self.learning_rate * td_error
        q_table.set_q_value(current_afterstate, new_q_value)


if __name__ == '__main__':
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
    initial_selection_size = 4
    env = SelectorGameMini(ALL_OPTIONS, range(1, ROUNDS), scores, initial_selection_size)



    # Example usage with the CliffWalkingEnv:
    # env = CliffWalkingEnv()
    agent = DoubleQLearningAgent(env, epsilon_min=0.01, episodes=40000)
    agent.train(
        patience=10000, min_delta=1)
    agent1 = QLearningAgent(env, epsilon_min=0.01, episodes=40000)
    agent1.train(
        patience=10000, min_delta=1)


    lines_dict = {
        'DoubleQLearningAgent': [agent.plotter.moving_avg_graph._x, agent.plotter.moving_avg_graph._y],
        'QLearningAgent': [agent1.plotter.moving_avg_graph._x, agent1.plotter.moving_avg_graph._y],

    }
    plot_line_dict(lines_dict)
    q_strat, value = agent.run_policy(policy=agent.get_policy())
    print_list_of_dicts(q_strat)
    print(f"Total Reward = {value}")