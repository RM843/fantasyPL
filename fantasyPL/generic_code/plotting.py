from typing import List

from matplotlib import pyplot as plt


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
