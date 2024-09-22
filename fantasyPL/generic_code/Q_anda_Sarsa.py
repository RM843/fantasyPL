import numpy as np
import matplotlib.pyplot as plt
import random


class QandSARSAAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995,
                 epsilon_min=0.01, moving_average_period=100):
        self.env = env  # The environment object
        self.state_size = env.height * env.width  # Total number of states in the grid
        self.action_size = 4  # 4 possible actions: Up, Right, Down, Left
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.moving_average_period = moving_average_period  # Period for moving average
        self.early_stopping = 0  # Tracks consecutive episodes without significant improvement

        # Initialize Q-table with zeros, dimensions: [state_size, action_size]
        self.q_table = np.zeros((self.state_size, self.action_size))

        # For plotting
        self.x = [0]
        self.y = [0]
        self.moving_avg_y = [0]
        self.setup_plot()

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
        """Convert (x, y) state to a single index."""
        x, y = state
        return x * self.env.width + y

    def choose_action(self, state):
        """Choose an action based on epsilon-greedy policy."""
        state_index = self.state_to_index(state)
        if np.random.rand() <= self.epsilon:
            return random.choice(self.env.get_allowed_actions(state))
        else:
            return np.argmax(self.q_table[state_index])

    def compute_moving_average(self, data, window_size=3):
        """Compute the moving average of the data."""
        if len(data) < window_size:
            return sum(data) / len(data)
        return sum(data[-window_size:]) / window_size

    def plot_rewards(self, reward, episode, patience):
        """Plot the list of rewards per episode with a moving average line and early stopping info."""
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

        # Redraw the plot with updated data
        plt.pause(0.0001)

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