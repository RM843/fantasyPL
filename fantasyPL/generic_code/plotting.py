from collections import deque
from typing import List

import numpy as np
from matplotlib import pyplot as plt

from helper_methods import replace_nones_with_previous



import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


colours = ['r', 'g', 'b', 'y', 'c', 'm', 'w', 'k']
class TrainingPlotter2:
    def __init__(self, moving_average_period: int):
        self.x = []
        self.y = []
        self.epsilon_values = []
        self.policy_scores = []
        self.action_values = {}
        self.moving_avg_y = []
        self.moving_average_period = moving_average_period
        # self.ax3 =self.win.addPlot(title="Actions")
        self.actions_graph = {}


        # Create a PyQtGraph application
        self.app = QtWidgets.QApplication([])

        # Set up the main window and plot
        self.win = pg.GraphicsLayoutWidget(show=True)
        self.win.setWindowTitle("Training Performance")
        self.win.resize(1000, 800)
        # Create plots
        self.ax1 = self.win.addPlot(title="Total Reward / Policy Score")
        self.ax1.setLabels(left="Total Reward / Policy Score", bottom="Episodes")
        self.graph = self.ax1.plot(self.x, self.y, pen='g', name='Rewards')
        self.moving_avg_graph = self.ax1.plot(self.x, self.moving_avg_y, pen='b', name='Moving Average')
        self.policy_score_graph = self.ax1.plot(self.x, self.policy_scores, pen='m', name='Policy Score')

        # Secondary axis for epsilon values
        self.ax2 = self.win.addPlot(title="Epsilon")
        self.epsilon_graph = self.ax2.plot(self.x, self.epsilon_values, pen='r', name='Epsilon')



        # Legends
        self.ax1.addLegend()
        self.ax2.addLegend()
        # self.ax3.addLegend()

        # Timer to update the plot
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(lambda: self.refresh_plot(self.x[-1] if len(self.x)>0 else 0))  # Pass the episode dynamically
        self.timer.start(50)  # Update every 50 ms

        # Annotations
        self.annotation = pg.TextItem('', anchor=(1, 0))
        self.ax1.addItem(self.annotation)
        self.policy_annotation = pg.TextItem('', anchor=(1, 0))
        self.ax1.addItem(self.policy_annotation)

    def add_data(
            self,
            reward: float,
            epsilon: float,
            actions:dict,
            policy_score: float
    ):
        """Add new data to the plotter's data lists."""
        episode = len(self.epsilon_values)-1
        self.y.append(reward)
        self.x.append(len(self.x) + 1)
        self.epsilon_values.append(epsilon)
        self.policy_scores.append(policy_score)
        # all actions that have ever been taken plus any new ones from this episode
        # action_superset = set(list(actions.keys())+list(self.action_values.keys()))
        episode, episode_actions = actions
        assert episode not in self.action_values
        self.action_values[episode] =episode_actions
        # total_ep_actions = sum(list(actions.values()))
        # for action in action_superset:
        #     self.action_values[action] = self.action_values.get(action,[0]*(episode)) +[actions.get(action,0)/total_ep_actions]

        # Compute moving average
        moving_avg = self.compute_moving_average(self.y, self.moving_average_period)
        self.moving_avg_y.append(moving_avg)

    def refresh_plot(self, episode: int):
        """Refresh the plot with the current data."""
        # Update plot data
        # self.graph.setData(self.x, self.y)
        self.moving_avg_graph.setData(self.x, self.moving_avg_y)

        self.policy_score_graph.setData(self.x,replace_nones_with_previous(self.policy_scores))
        self.epsilon_graph.setData(self.x, self.epsilon_values)

        # for action in self.action_values:
        #     x,y = self.x,[0] * (len(self.x) - len(self.action_values[action]))+self.action_values[action]
        #     if action in self.actions_graph:
        #         self.actions_graph[action].setData(x,y)
        #     else:
        #         self.actions_graph[action] = self.ax3.plot(x,y, pen=colours[len(self.actions_graph)%len(colours)], name=action)

        # Update annotations
        if len(self.moving_avg_y) > 0:
            self.annotation.setText(f'Moving Avg: {self.moving_avg_y[-1]:.2f}')
            self.annotation.setPos(self.x[-1], self.moving_avg_y[-1])

        if len(self.policy_scores) > 0:
            self.policy_annotation.setText(f'Policy Score: {self.policy_scores[-1]:.2f}')
            self.policy_annotation.setPos(self.x[-1], self.policy_scores[-1])

        # Update title with current episode information
        self.ax1.setTitle(f'Training Performance\nEpisode: {episode}')

        # Process GUI events to ensure the application remains responsive
        self.app.processEvents()

    @staticmethod
    def compute_moving_average(data: List[float], window_size: int) -> float:
        """Compute the moving average of the data."""
        if len(data) < window_size:
            return sum(data) / len(data)
        return sum(data[-window_size:]) / window_size

    def finalize_plot(self):
        """Finalize the plot after training."""
        self.timer.stop()
        self.win.close()
        self.app.quit()

def plot_line_dict(line_dict):
    """
    Plots a dictionary of lists, where each key is a line name and each value
    is a list of two lists: the first for x values, the second for y values.

    Parameters:
    line_dict (dict): A dictionary where keys are line names (strings)
                      and values are lists of two lists: [x_values, y_values].
    """
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Set up the color cycle
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Loop through the dictionary and add each line to the plot
    for i, (name, (x_values, y_values)) in enumerate(line_dict.items()):
        # Get the color from the cycle
        color = color_cycle[i % len(color_cycle)]

        # Plot the line with auto-coloring
        ax.plot(x_values, y_values, label=name, color=color)

    # Add a legend to identify the lines
    ax.legend()

    # Show the plot
    plt.show()