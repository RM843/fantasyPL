import random
from typing import Dict, Any, Optional


from fantasyPL.generic_code.Q_and_Sarsa import GenericAgent, REFRESH_RATE
from fantasyPL.generic_code.envs import SelectorGameMini, CliffWalkingEnv, MaximizationBiasEnv
from fantasyPL.generic_code.plotting import plot_line_dict
from fantasyPL.generic_code.q_learning import QLearningAgent
from fantasyPL.generic_code.q_table import QTable
from helper_methods import print_list_of_dicts


import numpy as np
import random
from typing import Any, Dict, Optional, Tuple, List
from fantasyPL.generic_code.q_table import QTable


class DoubleQLearningAgent(QLearningAgent):
    def __init__(self, env: Any, **kwargs):
        super().__init__(env, **kwargs)
        # Initialize two Q-tables for Double Q-Learning
        self.q_table1 = QTable()
        self.q_table2 = QTable()
        # To keep track of which Q-table to update
        self.update_table = 1  # Start with Q1


    def _initialize(self, state: Any, next_state: Any, action: Any):
        """Initialize Q-values for both tables and validate the action."""

        self.validate_action(state, action)
        state_rep_to_use = self._get_state(state, action)
        self.q_table1.initialize_q_value(state_rep_to_use)
        self.q_table2.initialize_q_value(state_rep_to_use)
        return self._get_state(state,action)
    def get_q_values(self, state: Any) -> Dict[Any, float]:
        """
        Retrieve Q-values for all allowed actions in the current state
        from the single Q-table.
        """

        q1 = self.q_table1.get_q_values(state=state, env=self.env)
        q2 = self.q_table2.get_q_values(state=state, env=self.env)
        return {key: q1.get(key, 0) + q2.get(key, 0) for key in set(q1) | set(q2)}
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

        # # Choose a random action for next_state from allowed actions
        allowed_actions = self.env.get_allowed_actions(next_state)
        next_action = random.choice(allowed_actions) if allowed_actions else None # best_next_action = self._best_action(next_state)

        if self.update_table == 1:
            q_table_primary = self.q_table1
            q_table_secondary = self.q_table2
        else:
            q_table_primary = self.q_table2
            q_table_secondary = self.q_table1

        current_state =  self._get_state(state,action,q_table=q_table_primary)
        if current_state is None:
            return 0.0

        if next_action is not None:

            state_rep_to_use = self._get_state(state=next_state, action=next_action)
            max_q_secondary = q_table_secondary.get_q_value(state_rep_to_use)
        else:
            max_q_secondary = 0.0

        # Calculate TD target
        td_target = reward + self.discount_factor * max_q_secondary * (1 - done)
        return td_target

    def _update(self,state_rep_to_use, td_target: float):
        """Update the appropriate Q-table based on the calculated TD target."""
        if self.update_table == 1:
            super()._update(state_rep_to_use,td_target,q_table =self.q_table1 )
            self.update_table = 2  # Switch to the other table next time
        else:
            super()._update(state_rep_to_use,td_target,q_table =self.q_table2 )
            self.update_table = 1  # Switch back



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
    env = MaximizationBiasEnv()
    # env= CliffWalkingEnv()
    agent1 = QLearningAgent(env=env, epsilon=0.1,epsilon_min=0.01, episodes=500 ,learning_rate=0.1,discount_factor=1,epsilon_strategy="fixed",verbose=True)
    agent1.train()
    agent = DoubleQLearningAgent(env=env,epsilon=0.1, epsilon_min=0.01, episodes=500,learning_rate=0.1,discount_factor=1,epsilon_strategy="fixed",verbose=True)
    agent.train()





    # lines_dict = {
    #     'DoubleQLearningAgent': [agent.plotter.moving_avg_graph.xData, agent.plotter.moving_avg_graph.yData],
    #     'QLearningAgent': [agent1.plotter.moving_avg_graph.xData, agent1.plotter.moving_avg_graph.yData],
    #
    # }
    # plot_line_dict(lines_dict)
    window_size=100
    left_moves_by_episode = [{k[-1]:v for k,v in x.items()}.get("left",0) for x in agent1.plotter.action_values.values()]
    left_moving_av =  [(n+window_size,sum(left_moves_by_episode[i:i + window_size]) / window_size) for n,i in
                      enumerate(range(len(left_moves_by_episode) - window_size + 1))]

    left_moves_by_episode2 = [{k[-1]: v for k, v in x.items()}.get("left", 0) for x in
                             agent.plotter.action_values.values()]
    left_moving_av2 = [(n + window_size, sum(left_moves_by_episode2[i:i + window_size]) / window_size) for n, i in
                      enumerate(range(len(left_moves_by_episode2) - window_size + 1))]

    right_moves_by_episode = [{k[-1]:v for k,v in x.items()}.get("right",0) for x in agent1.plotter.action_values.values()]
    right_moving_av = [(n+window_size,sum(right_moves_by_episode[i:i + window_size]) / window_size) for n,i in
                      enumerate(range(len(right_moves_by_episode) - window_size + 1))]
    rewards_dq = [(n+window_size,sum(agent.plotter.y[i:i + window_size]) / window_size) for n,i in
                      enumerate(range(len(agent.plotter.y) - window_size + 1))]

    rewards_q = [(n + window_size, sum(agent1.plotter.y[i:i + window_size]) / window_size) for n, i in
                  enumerate(range(len(agent1.plotter.y) - window_size + 1))]
   
    lines_dict = {
        'DoubleQLearningAgent': [[x[0] for x in left_moving_av2],[x[1] for x in left_moving_av2]],
        # 'DoubleQLearningAgentReward': [[x[0] for x in rewards_dq],[x[1] for x in rewards_dq]] ,
        'QLearningAgent':[[x[0] for x in left_moving_av],[x[1] for x in left_moving_av]],
        # 'QLearningAgentReward': [[x[0] for x in rewards_q],[x[1] for x in rewards_q]]

    }
    plot_line_dict(lines_dict)
    q_strat, value = agent.run_policy(policy=agent.get_policy())
    print_list_of_dicts(q_strat)
    print(f"Total Reward = {value}")