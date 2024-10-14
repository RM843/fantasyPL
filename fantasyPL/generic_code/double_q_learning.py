import random
import time
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

    def calculate_td_target(self):
        pass
    def _best_action(self, state: Any, allowed_actions: List[Any]) -> Any:
        """Select the best action based on the sum of Q1 and Q2."""
        q_values = {}
        for action in allowed_actions:
            q1 = self.q_table1.get_q_value(self.q_table1.afterstate(self.env, state, action))
            q2 = self.q_table2.get_q_value(self.q_table2.afterstate(self.env, state, action))
            q_values[action] = q1 + q2  # Combine Q-values
        max_value = max(q_values.values())
        # In case multiple actions have the same max value, randomly choose among them
        best_actions = [action for action, value in q_values.items() if value == max_value]
        return random.choice(best_actions)

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
        """Update Q-values using the Double Q-learning formula."""
        start_time = time.time()

        # Validate and initialize Q-values
        self.q_table1.initialize_q_value(state)
        self.q_table1.initialize_q_value(next_state)
        self.q_table2.initialize_q_value(state)
        self.q_table2.initialize_q_value(next_state)
        self.validate_action(state, action)

        # Determine which Q-table to update
        if self.update_table == 1:
            self._update_q_table(
                self.q_table1, self.q_table2, state, action, reward, next_state, done
            )
            self.update_table = 2  # Switch to the other table next time
        else:
            self._update_q_table(
                self.q_table2, self.q_table1, state, action, reward, next_state, done
            )
            self.update_table = 1  # Switch back

        # Decay epsilon
        if done:
            current_episode = len(self.plotter.x)
            self.epsilon_decay.decay(done, current_episode)

        self.time_tracker.add_time('learning', time.time() - start_time)

    def _update_q_table(
        self,
        q_table_primary: QTable,
        q_table_secondary: QTable,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: bool
    ):
        """Update the primary Q-table based on Double Q-learning."""
        current_afterstate = q_table_primary.afterstate(self.env, state, action)
        if current_afterstate is None:
            return

        # Choose a random action for next_state from allowed actions
        allowed_actions = self.env.get_allowed_actions(next_state)
        if allowed_actions:
            next_action = random.choice(allowed_actions)
            next_afterstate = q_table_secondary.afterstate(self.env, next_state, next_action)
            max_q_secondary = q_table_secondary.get_q_value(next_afterstate)
        else:
            max_q_secondary = 0.0

        # Calculate TD target
        td_target = reward + self.discount_factor * max_q_secondary * (1 - done)

        # Update Q-value
        td_error = td_target - q_table_primary.get_q_value(current_afterstate)
        new_q_value = q_table_primary.get_q_value(current_afterstate) + self.learning_rate * td_error
        q_table_primary.set_q_value(current_afterstate, new_q_value)


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