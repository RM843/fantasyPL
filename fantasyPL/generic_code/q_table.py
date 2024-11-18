import random
from typing import Any, Dict, List, Optional


class QTable:
    """Manages Q-values for each state."""
    def __init__(self):
        self.q_table: Dict[Any, float] = {}

    def initialize_q_value(self, state: Any):
        """Initialize Q-value for a given state if not already done."""
        if state is None:
            return
        if state not in self.q_table:
            self.q_table[state] = 0.0

    def get_q_value(self, state: Any, action =None) -> float:
        """Retrieve the Q-value for a given state."""
        if action is not None:
            to_use = (state,action)
        else:
            to_use = state
        return self.q_table.get(to_use, 0.0)

    def set_q_value(self, state: Any, value: float):
        """Set the Q-value for a given state."""
        self.q_table[state] = value

    def get_q_values(self, state: Any,env: Optional =None,allowed_actions:Optional  =None) -> Dict[Any, float]:
        """
        Retrieve Q-values for all allowed actions in the current state
        from the single Q-table.
        """
        if allowed_actions is None:
            allowed_actions = env.get_allowed_actions(state)

        q_values = {}
        for action in allowed_actions:
            if env.use_afterstates:
                state_rep_to_use = self.afterstate(env, state, action)
                q_value = self.get_q_value(state_rep_to_use)
            else:
                state_rep_to_use = state#
                q_value = self.get_q_value(state_rep_to_use, action)

            q_values[action] = q_value
        return q_values


    @staticmethod
    def afterstate(env: Any, state: Any, action: Any) -> Any:
        """Compute the afterstate given the current state and action."""
        if env.is_terminal(state):
            return None
        return env.transition_model(state, action)
