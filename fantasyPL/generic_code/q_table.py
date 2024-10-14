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

    def get_q_value(self, state: Any) -> float:
        """Retrieve the Q-value for a given state."""
        return self.q_table.get(state, 0.0)

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
            afterstate = self.afterstate(env, state, action)
            q_value = self.get_q_value(afterstate)
            q_values[action] = q_value
        return q_values


    def get_best_action(self, env: Any, state: Any) -> Optional[Any]:
        """Return the action with the highest Q-value for the given state."""
        allowed_actions = env.get_allowed_actions(state)
        if not allowed_actions:
            return None
        q_values =self.get_q_values(state,env=env,allowed_actions =allowed_actions)
        # Handle the case where all afterstates are None or have Q-value of 0.0
        if not any(q_values.values()):
            return random.choice(allowed_actions)  # Fallback to random action
        return max(q_values, key=q_values.get) if q_values else None

    @staticmethod
    def afterstate(env: Any, state: Any, action: Any) -> Any:
        """Compute the afterstate given the current state and action."""
        if env.is_terminal(state):
            return None
        return env.transition_model(state, action)