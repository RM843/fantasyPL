import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import random
import abc

from fantasyPL.generic_code.epsilion_decay import EpsilonDecay
from fantasyPL.generic_code.plotting import TrainingPlotter
from fantasyPL.generic_code.q_table import QTable
from fantasyPL.generic_code.timing import TimeTracker

# Constants
REFRESH_RATE = 5  # How often the plot updates (Seconds)


class GenericAgent(abc.ABC):
    def __init__(
        self,
        env: Any,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        moving_average_period: int = 100,
        episodes: int = 100
    ):
        self.env = env  # The environment object
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.episodes = episodes

        # Initialize EpsilonDecay object
        self.epsilon_decay = EpsilonDecay(epsilon, epsilon_decay, epsilon_min, episodes,strategy="inverse_sigmoid")

        self.moving_average_period = moving_average_period  # Period for moving average
        self.early_stopping = 0  # Tracks consecutive episodes without significant improvement

        # Initialize Q-table
        self.q_table = QTable()

        # Initialize Plotter
        self.plotter = TrainingPlotter(self.moving_average_period)

        # Initialize TimeTracker
        self.time_tracker = TimeTracker()

    def state_to_index(self, state: Any) -> Any:
        """Assumes states are already in index form, override if not."""
        return state

    def validate_action(self, state: Any, action: Any):
        """Ensure the action is valid for the given state."""
        if self.env.is_terminal(state):
            return
        allowed_actions = self.env.get_allowed_actions(state)
        if action not in allowed_actions:
            raise ValueError(
                f"Action {action} is not allowed in state {state}. Allowed actions: {allowed_actions}"
            )

    def get_q_values(self, state: Any) -> Dict[Any, float]:
        """
        Retrieve Q-values for all allowed actions in the current state
        from the single Q-table.
        """

        return self.q_table.get_q_values(state=state,env=self.env)

    def choose_action_based_on_policy(self, state: Any) -> Any:
        """
        Choose an action based on epsilon-greedy policy using the Q-values.
        This method is unified and shared across all agents.
        """
        start_time = time.time()

        allowed_actions = self.env.get_allowed_actions(state)
        if not allowed_actions:
            return None  # No actions available

        if np.random.rand() <= self.epsilon_decay.get_epsilon():
            action = random.choice(allowed_actions)
        else:
            action = self._best_action(state, allowed_actions)

        end_time = time.time()
        self.time_tracker.add_time('action_selection', end_time - start_time)
        return action


    def _best_action(self, state: Any, allowed_actions=None) -> Any:
        """
        Determine the best action based on the Q-values for the given state.
        If there are multiple actions with the same max Q-value, one is chosen randomly.
        """
        if self.env.is_terminal(state):
            return None
        q_values = self.get_q_values(state)
        max_q = max(q_values.values())
        # Handle multiple actions with the same max Q-value
        best_actions = [action for action, q in q_values.items() if q == max_q]
        if best_actions:
            return random.choice(best_actions)
        else:
            if allowed_actions is None:
                allowed_actions = self.env.get_allowed_actions(state)
            random.choice(allowed_actions)
    def plot_rewards(
        self,
        reward: float,
        episode: int,
        patience: int,
        show: bool,
        policy_score: float
    ):
        """Plot the list of rewards per episode with a moving average line, policy score, and early stopping info."""
        start_time = time.time()
        # Always add data
        self.plotter.add_data(
            reward=reward,
            epsilon=self.epsilon_decay.epsilon,
            policy_score=policy_score
        )

        # Refresh plot only if show is True
        if show:
            self.plotter.refresh_plot(
                episode=episode,
                early_stopping=self.early_stopping,
                patience=patience
            )

        self.time_tracker.add_time('plotting', time.time() - start_time)

    def check_early_stopping(self, patience: int = 10, min_delta: float = 1.0) -> bool:
        """Check if training should stop early based on moving average."""
        if len(self.plotter.moving_avg_y) < 2:
            return False  # Not enough data to decide

        recent_avg = self.plotter.moving_avg_y[-1]
        past_avg = self.plotter.moving_avg_y[-2]
        # Check if the improvement is less than min_delta
        if recent_avg - past_avg < min_delta:
            self.early_stopping += 1
        else:
            self.early_stopping = 0
        if self.early_stopping >= patience:
            return True
        return False

    def get_policy(self) -> Dict[Any, Any]:
        """Get the current policy from the Q-table."""
        policy = {}
        for state in self.q_table.q_table:
            best_action = self._best_action( state)
            if best_action is not None:
                policy[state] = best_action
        return policy

    def print_policy(self):
        """Print a human-readable policy with actions representing the best choices."""
        policy = self.get_policy()
        for state, action in policy.items():
            print(f"State {state}: Action {action}")

    def print_time_breakdown(self):
        """Print a breakdown of the proportion of time spent on each part of the training process."""
        self.time_tracker.print_time_breakdown()

    @abc.abstractmethod
    def calculate_td_target(self, state, action, reward, next_state, done, next_action=None) -> float:
        """Abstract method to be implemented by subclasses for calculating the TD target."""
        pass

    def learn(self, state: Any, action: Any, reward: float, next_state: Any, done: bool,
              next_action: Optional[Any] = None):
        """Generalized learn method with dependency injection for TD calculation."""
        start_time = time.time()

        # Initialize Q-values and validate actions
        current_afterstate = self._initialize(state=state, action=action, next_state=next_state)

        if current_afterstate is None:
            self.time_tracker.add_time('learning', time.time() - start_time)
            return

        # Use the agent-specific TD target calculation
        td_target = self.calculate_td_target(state, action, reward, next_state, done, next_action)

        # Update Q-value based on TD error
        self._update(current_afterstate, td_target)

        # Decay epsilon
        if done:
            current_episode = len(self.plotter.x)
            self.epsilon_decay.decay(done, current_episode)

        self.time_tracker.add_time('learning', time.time() - start_time)

    def _initialize(self, state: Any, action: Any, next_state: Any) -> Optional[Any]:
        """Initialize Q-values and validate actions."""
        self.q_table.initialize_q_value(state)
        self.q_table.initialize_q_value(next_state)
        self.validate_action(state, action)

        return self.q_table.afterstate(self.env, state, action)

    def _update(self, current_afterstate: Any, td_target: float, q_table=None):
        """Update the Q-value based on the TD target."""
        if q_table is None:
            q_table = self.q_table
        td_error = td_target - q_table.get_q_value(current_afterstate)
        new_q_value = q_table.get_q_value(current_afterstate) + self.learning_rate * td_error
        q_table.set_q_value(current_afterstate, new_q_value)
    def run_policy(self, policy: Dict[Any, Any]) -> Tuple[List[Dict[str, Any]], float]:
        """Run a policy and return the strategy and total reward."""
        state = self.env.reset()
        total_reward = 0.0
        strategy = []

        while True:
            if state not in policy:
                action = random.choice(self.env.get_allowed_actions(state))
            else:
                action = policy[state]

            start_time = time.time()
            next_state, reward, done = self.env.step(action)
            end_time = time.time()
            self.time_tracker.add_time('environment_step', end_time - start_time)

            strategy.append({"state": state, "action": action, "value": reward})
            total_reward += reward
            state = next_state

            if done:
                break

        return strategy, total_reward

    def train(
        self,
        max_steps: int = 100,
        patience: int = 10,
        min_delta: float = 1.0,
        sarsa: bool = False
    ): 
        """Train the agent for a specified number of episodes."""
        start_time = time.time()
        show = True
        for episode in range(self.episodes):
            state = self.env.reset()
            action = self.choose_action_based_on_policy(state) if sarsa else None  # Only for SARSA
            total_reward = 0.0

            for step in range(max_steps):
                if sarsa:
                    next_state, reward, done = self.env.step(action)
                    if not done:
                        next_action = self.choose_action_based_on_policy(next_state)
                    else:
                        next_action = None  # Assuming env has no actions after terminal state
                    self.learn(state=state, action=action, reward=reward, next_state=next_state, done=done, next_action=next_action)
                    state, action = next_state, next_action
                else:
                    action = self.choose_action_based_on_policy(state)
                    next_state, reward, done = self.env.step(action)
                    self.learn(state, action, reward, next_state, done)
                    state = next_state

                total_reward += reward

                if done:
                    break

            # Determine if it's time to show the plot
            if time.time() - start_time > REFRESH_RATE:
                show = True
                start_time = time.time()

            # Plot the performance after each episode
            policy_score = 0.0
            if show:
                policy = self.get_policy()
                _, policy_score = self.run_policy(policy)
                self.plot_rewards(
                    reward=total_reward,
                    episode=episode,
                    patience=patience,
                    show=True,
                    policy_score=policy_score
                )
                show = False
            else:
                # Use the last policy score if not showing
                last_policy_score = self.plotter.policy_scores[-1] if self.plotter.policy_scores else 0.0
                self.plot_rewards(
                    reward=total_reward,
                    episode=episode,
                    patience=patience,
                    show=False,
                    policy_score=last_policy_score
                )

            # Check for early stopping and print progress
            early_stop = self.check_early_stopping(patience, min_delta)

            if early_stop:
                print(f"Early stopping triggered at episode {episode + 1}.")
                break

        # Print time breakdown after training is complete
        self.print_time_breakdown()
        # Final plot update
        policy = self.get_policy()
        _, policy_score = self.run_policy(policy)
        self.plot_rewards(
            reward=total_reward,
            episode=episode,
            patience=patience,
            show=True,
            policy_score=policy_score
        )
        self.plotter.finalize_plot()
