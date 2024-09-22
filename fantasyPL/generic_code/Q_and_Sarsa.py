import time
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import random
import abc

from helper_methods import print_list_of_dicts


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
        """Initializes the plot."""
        plt.ion()  # Interactive mode on
        self.fig, self.ax = plt.subplots()
        self.graph, = plt.plot(self.x, self.y, color='g', label='Rewards')
        self.moving_avg_graph, = plt.plot(self.x, self.moving_avg_y, color='b', linestyle='--',
                                          label=f'Moving Average (Period: {self.moving_average_period})')
        self.annotation = self.ax.annotate('', xy=(0, 0), xytext=(5, 5), textcoords='offset points')
        plt.xlabel('Episodes')
        plt.ylabel('Total Reward')
        plt.title('Training Performance')
        plt.legend()
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
        allowed_actions = self.env.get_allowed_actions(state)
        if action not in allowed_actions:
            raise ValueError(f"Action {action} is not allowed in state {state}. Allowed actions: {allowed_actions}")
    def choose_action(self, state):
        """Choose an action based on epsilon-greedy policy."""
        start_time = time.time()
        # if state not in self.q_table:
        #     self.q_table[state] = np.zeros(len(self.env.get_allowed_actions(state)))

        if (np.random.rand() <= self.epsilon) or (state not in self.q_table):
            action =  random.choice(self.env.get_allowed_actions(state))
        else:
            action = max(self.q_table[state], key=self.q_table[state].get)
        self.time_spent['action_selection'] += time.time() - start_time
        return action

    def compute_moving_average(self, data, window_size=3):
        """Compute the moving average of the data."""
        if len(data) < window_size:
            return sum(data) / len(data)
        return sum(data[-window_size:]) / window_size

    def plot_rewards(self, reward, episode, patience,show):
        """Plot the list of rewards per episode with a moving average line and early stopping info."""
        start_time = time.time()
        # Update the data
        self.y.append(reward)
        self.x.append(self.x[-1] + 1)

        # Compute the moving average and update the moving average data
        moving_avg = self.compute_moving_average(self.y, window_size=self.moving_average_period)
        self.moving_avg_y.append(moving_avg)

        # Remove the older graphs
        self.graph.remove()
        self.moving_avg_graph.remove()

        # Plot the updated graph and moving average
        self.graph, = plt.plot(self.x, self.y, color='g', label='Rewards')
        self.moving_avg_graph, = plt.plot(self.x, self.moving_avg_y, color='b', linestyle='--',
                                          label=f'Moving Average (Period: {self.moving_average_period})')

        # Annotate the last value of the moving average
        if self.annotation:
            self.annotation.remove()
        self.annotation = self.ax.annotate(f'{moving_avg:.2f}',
                                           xy=(self.x[-1], self.moving_avg_y[-1]),
                                           xytext=(5, 5),
                                           textcoords='offset points',
                                           fontsize=10,
                                           color='blue')

        # Update the title to include early stopping information
        self.ax.set_title(
            f'Training Performance\nEpisode: {episode + 1}, Early Stopping: {self.early_stopping}/{patience}')

        # Update the legend to reflect changes
        plt.legend()
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
            if next_action is None:
                raise ValueError("Next action must be provided for SARSA.")
            self.validate_action(next_state, next_action)

            # SARSA update
            td_target = reward + self.discount_factor * self.q_table[next_state][next_action] * (1 - done)
        else:  # Q-learning update
            best_next_action = self.get_best_action(next_state)
            q_value = self.q_table[next_state][best_next_action] if best_next_action is not None else 0
            td_target = reward + self.discount_factor * q_value * (1 - done)

        td_error = td_target - self.q_table[state][action]

        self.q_table[state][action] += self.learning_rate * td_error

        # Decay epsilon
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    def get_best_action(self,state):
        return max(self.q_table[state], key=self.q_table[state].get) if self.q_table[state] else None
    def train(self, episodes=1000, max_steps=100, patience=10, min_delta=1.0, sarsa=False):
        """Train the agent for a specified number of episodes."""
        for episode in range(episodes):
            state = self.env.reset()
            action = self.choose_action(state) if sarsa else None  # Only for SARSA
            total_reward = 0

            for step in range(max_steps):
                if sarsa:
                    next_state, reward, done = self.env.step(action)
                    next_action = self.choose_action(next_state)
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

            # Plot the performance after each episode
            self.plot_rewards(total_reward, episode, patience, show=episode % 100 == 0)

            # Check for early stopping and print progress
            early_stop = self.check_early_stopping(patience, min_delta)

            # Print training progress
            # print(
            #     f"Episode: {episode + 1}, Reward: {total_reward}, Epsilon: {self.epsilon:.4f}, Early Stopping: {self.early_stopping}/{patience}")

            if early_stop:
                print(f"Early stopping triggered at episode {episode + 1}.")
                break

        # Print time breakdown after training is complete
        self.print_time_breakdown()
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
                return None
            action = policy[state]  # Get the best action and its value for the current state



            # Transition to the next state based on the chosen action
            next_state, reward, done = self.env.step(action)

            strat.append({"state": state, "action": action, "value": reward})
            total_reward +=reward
            state = next_state
            if done:
                break
        print_list_of_dicts(strat)
        print(f"Total Reward = {total_reward}")
        return strat ,reward