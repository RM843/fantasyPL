import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import random
import abc

from fantasyPL.generic_code.epsilion_decay import EpsilonDecay

# Constants
REFRESH_RATE = 5  # How often the plot updates (Seconds)


class TimeTracker:
    """Tracks the execution time of various components in the training process."""
    def __init__(self):
        self.time_spent = {
            'action_selection': 0.0,
            'learning': 0.0,
            'plotting': 0.0,
            'environment_step': 0.0
        }

    def add_time(self, key: str, duration: float):
        """Add the duration to the specified key."""
        if key in self.time_spent:
            self.time_spent[key] += duration
        else:
            self.time_spent[key] = duration

    def print_time_breakdown(self):
        """Print a breakdown of the proportion of time spent on each part of the training process."""
        total_time = sum(self.time_spent.values())
        if total_time == 0:
            print("No time data to report.")
            return

        print("\nTraining Time Breakdown:")
        for key, value in self.time_spent.items():
            proportion = (value / total_time) * 100
            print(f"{key.capitalize()}: {proportion:.2f}% of training time")


class TrainingPlotter:
    """Handles all plotting functionalities for training performance."""
    def __init__(self, moving_average_period: int):
        self.x = [0]
        self.y = [0]
        self.epsilon_values = [0]
        self.policy_scores = [0]
        self.moving_avg_y = [0]
        self.moving_average_period = moving_average_period

        # Initialize plot
        plt.ion()  # Interactive mode on
        self.fig, self.ax1 = plt.subplots(figsize=(12, 8))
        self.ax2 = self.ax1.twinx()

        # Primary axis for rewards and policy scores
        self.graph, = self.ax1.plot(self.x, self.y, color='g', label='Rewards')
        self.moving_avg_graph, = self.ax1.plot(
            self.x, self.moving_avg_y, color='b', linestyle='--',
            label=f'Moving Average (Period: {self.moving_average_period})'
        )
        self.policy_score_graph, = self.ax1.plot(
            self.x, self.policy_scores, color='m', linestyle=':',
            label='Policy Score'
        )

        # Secondary axis for epsilon values
        self.epsilon_graph, = self.ax2.plot(
            self.x, self.epsilon_values, color='r', linestyle='-.', label='Epsilon'
        )

        # Annotations
        self.annotation = self.ax1.annotate(
            '', xy=(0, 0), xytext=(5, 5), textcoords='offset points', fontsize=10, color='blue'
        )
        self.policy_annotation = self.ax1.annotate(
            '', xy=(0, 0), xytext=(5, -10), textcoords='offset points', fontsize=10, color='magenta'
        )

        # Axis labels and title
        self.ax1.set_xlabel('Episodes')
        self.ax1.set_ylabel('Total Reward / Policy Score')
        self.ax2.set_ylabel('Epsilon')
        self.ax1.set_title('Training Performance')

        # Legends
        self.ax1.legend(loc='upper left')
        self.ax2.legend(loc='upper right')
        plt.show()

    def add_data(
        self,
        reward: float,
        epsilon: float,
        policy_score: float
    ):
        """Add new data to the plotter's data lists."""
        self.y.append(reward)
        self.x.append(self.x[-1] + 1)
        self.epsilon_values.append(epsilon)
        self.policy_scores.append(policy_score)

        # Compute moving average
        moving_avg = self.compute_moving_average(self.y, self.moving_average_period)
        self.moving_avg_y.append(moving_avg)

    def refresh_plot(
        self,
        episode: int,
        early_stopping: int,
        patience: int
    ):
        """Refresh the plot with the current data."""
        # Update plot data
        self.graph.set_data(self.x, self.y)
        self.moving_avg_graph.set_data(self.x, self.moving_avg_y)
        self.policy_score_graph.set_data(self.x, self.policy_scores)
        self.epsilon_graph.set_data(self.x, self.epsilon_values)

        # Adjust plot limits
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()

        # Update annotations
        if len(self.moving_avg_y) > 0:
            self.annotation.set_text(f'{self.moving_avg_y[-1]:.2f}')
            self.annotation.xy = (self.x[-1], self.moving_avg_y[-1])

        if len(self.policy_scores) > 0:
            self.policy_annotation.set_text(f'{self.policy_scores[-1]:.2f}')
            self.policy_annotation.xy = (self.x[-1], self.policy_scores[-1])

        # Update title with early stopping info
        self.ax1.set_title(
            f'Training Performance\nEpisode: {episode + 1}, Early Stopping: {early_stopping}/{patience}'
        )

        # Update legends
        self.ax1.legend(loc='upper left')
        self.ax2.legend(loc='upper right')

        # Redraw the plot
        plt.pause(0.001)

    @staticmethod
    def compute_moving_average(data: List[float], window_size: int) -> float:
        """Compute the moving average of the data."""
        if len(data) < window_size:
            return sum(data) / len(data)
        return sum(data[-window_size:]) / window_size

    def finalize_plot(self):
        """Finalize the plot after training."""
        plt.ioff()
        plt.show()


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

    def get_best_action(self, env: Any, state: Any, allowed_actions: List[Any]) -> Optional[Any]:
        """Return the action with the highest Q-value for the given state."""
        if not allowed_actions:
            return None
        q_values = {
            action: self.get_q_value(self.afterstate(env, state, action)) for action in allowed_actions
        }
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
        self.epsilon_decay = EpsilonDecay(epsilon, epsilon_decay, epsilon_min, episodes)

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

    def choose_action(self, state: Any) -> Any:
        """Choose an action based on epsilon-greedy policy."""
        start_time = time.time()

        if np.random.rand() <= self.epsilon_decay.get_epsilon():
            action = random.choice(self.env.get_allowed_actions(state))
        else:
            allowed_actions = self.env.get_allowed_actions(state)
            action = self.q_table.get_best_action(self.env, state, allowed_actions)
            if action is None:
                action = random.choice(allowed_actions)

        end_time = time.time()
        self.time_tracker.add_time('action_selection', end_time - start_time)
        return action

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
            allowed_actions = self.env.get_allowed_actions(state)
            best_action = self.q_table.get_best_action(self.env, state, allowed_actions)
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
        """Update Q-values using either the SARSA or Q-learning formula."""
        start_time = time.time()
        self.q_table.initialize_q_value(state)
        self.q_table.initialize_q_value(next_state)
        self.validate_action(state, action)

        current_afterstate = self.q_table.afterstate(self.env, state, action)
        if current_afterstate is None:
            self.time_tracker.add_time('learning', time.time() - start_time)
            return

        if sarsa and next_action is not None:
            next_afterstate = self.q_table.afterstate(self.env, next_state, next_action)
            td_target = reward + self.discount_factor * self.q_table.get_q_value(next_afterstate) * (1 - done)
        else:
            # Q-learning update
            allowed_actions = self.env.get_allowed_actions(next_state)
            best_next_action = self.q_table.get_best_action(self.env, next_state, allowed_actions)
            if best_next_action is None:
                next_afterstate = None
            else:
                next_afterstate = self.q_table.afterstate(self.env, next_state, best_next_action)
            td_target = reward + self.discount_factor * self.q_table.get_q_value(next_afterstate) * (1 - done)

        td_error = td_target - self.q_table.get_q_value(current_afterstate)
        new_q_value = self.q_table.get_q_value(current_afterstate) + self.learning_rate * td_error
        self.q_table.set_q_value(current_afterstate, new_q_value)

        # Decay epsilon
        if done:
            current_episode = len(self.plotter.x)
            self.epsilon_decay.decay(done, current_episode)
        self.time_tracker.add_time('learning', time.time() - start_time)

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
            action = self.choose_action(state) if sarsa else None  # Only for SARSA
            total_reward = 0.0

            for step in range(max_steps):
                if sarsa:
                    next_state, reward, done = self.env.step(action)
                    if not done:
                        next_action = self.choose_action(next_state)
                    else:
                        next_action = None  # Assuming env has no actions after terminal state
                    self.learn(state, action, reward, next_state, done, next_action=next_action, sarsa=True)
                    state, action = next_state, next_action
                else:
                    action = self.choose_action(state)
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
