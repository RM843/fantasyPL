import time
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import random
import abc

REFRESH_RATE = 5 # how often the plot updates (Seconds)
class GenericAgent(abc.ABC):
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995,
                 epsilon_min=0.01, moving_average_period=100):
        self.env = env  # The environment object
        # self.state_size = env.state_size  # Total number of states in the environment
        # self.action_size = env.action_size  # Total number of actions in the environment
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.moving_average_period = moving_average_period  # Period for moving average
        self.early_stopping = 0  # Tracks consecutive episodes without significant improvement


        self.q_table = {}  # Use a dictionary to store Q-values for each state


        # For plotting
        self.x = [0]
        self.y = [0]
        self.epsilon_values = [epsilon]
        self.policy_scores = [0]
        self.moving_avg_y = [0]
        self.setup_plot()

        # Time tracking variables
        self.time_spent = {
            'action_selection': 0,
            'learning': 0,
            'plotting': 0,
            'environment_step': 0
        }

    def setup_plot(self):
        """Initializes the plot with a secondary axis for epsilon values and policy scores."""
        plt.ion()  # Interactive mode on
        self.fig, self.ax1 = plt.subplots(figsize=(12, 8))  # Adjust the figure size here (width, height)
        self.ax2 = self.ax1.twinx()  # Create a secondary y-axis

        # Primary axis for rewards and policy scores
        self.graph, = self.ax1.plot(self.x, self.y, color='g', label='Rewards')
        self.moving_avg_graph, = self.ax1.plot(self.x, self.moving_avg_y, color='b', linestyle='--',
                                               label=f'Moving Average (Period: {self.moving_average_period})')
        self.policy_score_graph, = self.ax1.plot(self.x, self.policy_scores, color='m', linestyle=':',
                                                 label='Policy Score')

        # Secondary axis for epsilon values
        self.epsilon_graph, = self.ax2.plot(self.x, self.epsilon_values, color='r', linestyle='-.', label='Epsilon')

        self.annotation = self.ax1.annotate('', xy=(0, 0), xytext=(5, 5), textcoords='offset points')
        self.policy_annotation = self.ax1.annotate('', xy=(0, 0), xytext=(5, -10),
                                                   textcoords='offset points')  # New annotation for policy score

        # Axis labels and title
        self.ax1.set_xlabel('Episodes')
        self.ax1.set_ylabel('Total Reward / Policy Score')
        self.ax2.set_ylabel('Epsilon')
        self.ax1.set_title('Training Performance')

        # Legends for both axes
        self.ax1.legend(loc='upper left')
        self.ax2.legend(loc='upper right')
        plt.show()

    def state_to_index(self, state):
        """Assumes states are already in index form, override if not."""
        return state

    def initialize_q_value(self, state):
        """Initialize Q-values for the given state if not already done."""
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in self.env.get_allowed_actions(state)}

    def validate_action(self, state, action):
        """Ensure the action is valid for the given state."""
        if self.env.is_terminal(state):
            return
        allowed_actions = self.env.get_allowed_actions(state)
        if action not in allowed_actions:
            raise ValueError(f"Action {action} is not allowed in state {state}. Allowed actions: {allowed_actions}")
    def choose_action(self, state):
        """Choose an action based on epsilon-greedy policy."""
        start_time = time.time()

        if (np.random.rand() <= self.epsilon) or (state not in self.q_table):
            action = random.choice(self.env.get_allowed_actions(state))
        else:
            action = max(self.q_table[state], key=self.q_table[state].get)
        self.time_spent['action_selection'] += time.time() - start_time
        return action

    def compute_moving_average(self, data, window_size=3):
        """Compute the moving average of the data."""
        if len(data) < window_size:
            return sum(data) / len(data)
        return sum(data[-window_size:]) / window_size

    def plot_rewards(self, reward, episode, patience, show):
        """Plot the list of rewards per episode with a moving average line, policy score, and early stopping info."""
        start_time = time.time()
        # Update the data
        self.y.append(reward)
        self.x.append(self.x[-1] + 1)
        self.epsilon_values.append(self.epsilon)  # Append current epsilon value

        # Run the policy to get the policy score only if `show` is True
        if show:
            policy = self.get_policy()
            _,policy_score = self.run_policy(policy)
        else:
            policy_score = self.policy_scores[-1]  # Repeat the last policy score if not shown

        self.policy_scores.append(policy_score)  # Append the current policy score when `show` is True

        # Compute the moving average and update the moving average data
        moving_avg = self.compute_moving_average(self.y, window_size=self.moving_average_period)
        self.moving_avg_y.append(moving_avg)
        if not show:
            return

        # Remove the older graphs
        self.graph.remove()
        self.moving_avg_graph.remove()
        self.epsilon_graph.remove()
        self.policy_score_graph.remove()

        # Plot the updated graph and moving average
        self.graph, = self.ax1.plot(self.x, self.y, color='g', label='Rewards')
        self.moving_avg_graph, = self.ax1.plot(self.x, self.moving_avg_y, color='b', linestyle='--',
                                               label=f'Moving Average (Period: {self.moving_average_period})')
        self.policy_score_graph, = self.ax1.plot(self.x, self.policy_scores, color='m', linestyle=':', label='Policy Score')
        self.epsilon_graph, = self.ax2.plot(self.x, self.epsilon_values, color='r', linestyle='-.', label='Epsilon')

        # Annotate the last value of the moving average
        if self.annotation:
            self.annotation.remove()
        self.annotation = self.ax1.annotate(f'{moving_avg:.2f}',
                                            xy=(self.x[-1], self.moving_avg_y[-1]),
                                            xytext=(5, 5),
                                            textcoords='offset points',
                                            fontsize=10,
                                            color='blue')

        # Annotate the last value of the policy score
        if self.policy_annotation:
            self.policy_annotation.remove()
        self.policy_annotation = self.ax1.annotate(f'{policy_score:.2f}',
                                                   xy=(self.x[-1], self.policy_scores[-1]),
                                                   xytext=(5, -10),
                                                   textcoords='offset points',
                                                   fontsize=10,
                                                   color='magenta')

        # Update the title to include early stopping information
        self.ax1.set_title(
            f'Training Performance\nEpisode: {episode + 1}, Early Stopping: {self.early_stopping}/{patience}')

        # Update the legends to reflect changes
        self.ax1.legend(loc='upper left')
        self.ax2.legend(loc='upper right')
        if show:
            # Redraw the plot with updated data
            plt.pause(0.0001)
        self.time_spent['plotting'] += time.time() - start_time

    def check_early_stopping(self, patience=10, min_delta=1.0):
        """Check if training should stop early based on moving average."""
        recent_avg = self.moving_avg_y[-1]
        past_avg = self.moving_avg_y[-2]
        # Check if the improvement is less than min_delta
        if recent_avg - past_avg < min_delta:
            self.early_stopping += 1
        else:
            self.early_stopping = 0
        if self.early_stopping == patience:
            return True
        return False

    def get_policy(self):
        """Get the current policy from the Q-table."""
        policy = {}
        for state in self.q_table:
            policy[state] = self.get_best_action(state)
        return policy

    def print_policy(self):
        """Print a human-readable policy with actions representing the best choices."""
        policy = self.get_policy()
        for state, action in policy.items():
            print(f"State {state}: Action {action}")

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

    def learn(self, state, action, reward, next_state, done, next_action=None, sarsa=False):
        """Update Q-values using either the SARSA or Q-learning formula."""
        self.initialize_q_value(state)
        self.initialize_q_value(next_state)
        # Validate actions
        self.validate_action(state, action)
        if sarsa:
            pass
            # # if next_action is None:
            # #     raise ValueError("Next action must be provided for SARSA.")
            # # self.validate_action(next_state, next_action)
            # q_value = self.q_table[next_state][next_action] if next_action is not None else 0
            # # SARSA update
            # td_target = reward + self.discount_factor *  * (1 - done)
        else:  # Q-learning update
            next_action = self.get_best_action(next_state)
        q_value = self.q_table[next_state][next_action] if next_action is not None else 0
        td_target = reward + self.discount_factor * q_value * (1 - done)

        td_error = td_target - self.q_table[state][action]

        self.q_table[state][action] += self.learning_rate * td_error

        # Decay epsilon
        if done:
            decay_rate = (self.epsilon - self.epsilon_min) /( self.episodes/2)
            self.epsilon = max(self.epsilon_min, self.epsilon - decay_rate)
    def get_best_action(self,state):
        return max(self.q_table[state], key=self.q_table[state].get) if self.q_table[state] else None
    def train(self, episodes=1000, max_steps=100, patience=10, min_delta=1.0, sarsa=False):
        """Train the agent for a specified number of episodes."""
        self.episodes = episodes  # Set total number of episodes for the decay calculation
        start_time = time.time()
        show = True
        for episode in range(episodes):
            state = self.env.reset()
            action = self.choose_action(state) if sarsa else None  # Only for SARSA
            total_reward = 0

            for step in range(max_steps):
                if sarsa:
                    next_state, reward, done = self.env.step(action)
                    if not done:
                        next_action = self.choose_action(next_state)
                    else:
                        next_action = None # Assuming env has no actions after terminal state
                    self.learn(state, action, reward, next_state, next_action, done)
                    state, action = next_state, next_action
                else:
                    action = self.choose_action(state)
                    next_state, reward, done = self.env.step(action)
                    self.learn(state, action, reward, next_state, done)
                    state = next_state

                total_reward += reward

                if done:
                    break
            if time.time() - start_time > REFRESH_RATE:
                show = True
                start_time = time.time()



            # Plot the performance after each episode, but update policy score only if `show` is True
            self.plot_rewards(total_reward, episode, patience, show=show)
            show = False

            # Check for early stopping and print progress
            early_stop = self.check_early_stopping(patience, min_delta)

            if early_stop:
                print(f"Early stopping triggered at episode {episode + 1}.")
                break

        # Print time breakdown after training is complete
        self.print_time_breakdown()
        self.plot_rewards(total_reward, episode, patience, show=True)
        # Keep the plot open after training
        plt.ioff()
        plt.show()

    def run_policy(self,policy) :
        """

        """

        # Retrieve the best initial state based on the value iteration results
        state = self.env.reset()
        total_reward = 0
        strat = []  # Initialize the strategy list

        # Continue generating the strategy until reaching a final state
        while True:
            if state not in policy:
                action = random.choice(self.env.get_allowed_actions(state))
            else:

                action = policy[state]  # Get the best action and its value for the current state



            # Transition to the next state based on the chosen action
            next_state, reward, done = self.env.step(action)

            strat.append({"state": state, "action": action, "value": reward})
            total_reward +=reward
            state = next_state
            if done:
                break

        return strat ,total_reward