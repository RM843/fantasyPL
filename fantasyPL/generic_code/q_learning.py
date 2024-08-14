import random
from itertools import product
from typing import List
import matplotlib.pyplot as plt

from tqdm import tqdm

from fantasyPL.generic_code.reinforment_learning import PolicyOrValueIteration


class QLearn(PolicyOrValueIteration):
    def __init__(self, problem_obj, env):

        super().__init__(problem_obj)
        self.env = env
        plt.ion()
        self.fig, self.ax = plt.subplots()
        # Enable interactive mode
        self.graph = plt.plot([], [])[0]
        # plt.ylim(0, 10)
        self.x, self.y = [0], [0]
        self.moving_avg_y = [0]  # List to store moving averages
        self.moving_avg_graph, = plt.plot(self.x, self.moving_avg_y, color='b', linestyle='--', label='Moving Average')
        self.ax.set_title('Rewards per Episode')
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Reward')
        self.ax.grid(True)
        self.ax.legend()
        self.annotation = None  # Placeholder for the annotation obje
    def algorithm(self, exploration_rate, lr, discount_factor=0.99):
        """

        """
        main_exploration_rate = exploration_rate
        ep_count = 0
        all_rewards = []
        while True:
            test_ep = ep_count%10 ==0 and ep_count!=0
            if test_ep:
                exploration_rate = 0
            else:
                exploration_rate = main_exploration_rate
            state = self.env.reset()
            # self.env.render()
            ep_rewards = 0
            steps_count = 0
            while True:

                action,rand = self.get_epsilon_greedy_action(state, exploration_rate)
                next_state, reward, terminal = self.env.step(action)
                steps_count +=1
                # self.env.render()
                ep_rewards += reward
                if reward ==-100 and test_ep :
                    pass
                _, best_next_state_action_value =  self.get_best_action(state,discount_factor)
                self.V[(state, action)] = (self.V.get((state, action), 0) +
                                           lr * (reward + discount_factor * best_next_state_action_value -
                                                 self.V.get((state, action),
                                                            0)))  # Update the state value with the best action value
                state =next_state
                if self.env.is_terminal(state):
                    ep_count +=1
                    # self.plot_q_values()
                    break
            if  test_ep:
                all_rewards.append(ep_rewards)
                self.plot_rewards(ep_rewards)

    def get_epsilon_greedy_action(self, state, exploration_rate):
        # explore
        if random.random() < exploration_rate:
            return random.choice(self.get_allowed_actions(state)) ,True
        # exploit
        else:
            next_action, _ = self.get_best_action(state, 1)
            return next_action,False

    def compute_moving_average(self, data, window_size=3):
        """Compute the moving average of the data."""
        if len(data) < window_size:
            return sum(data) / len(data)
        return sum(data[-window_size:]) / window_size
    def plot_rewards(self, reward):
        """Plot the list of rewards per episode with a moving average line."""
        # Update the data
        self.y.append(reward)
        self.x.append(self.x[-1] + 1)

        # Compute the moving average and update the moving average data
        moving_avg = self.compute_moving_average(self.y,window_size=100)
        self.moving_avg_y.append(moving_avg)

        # Remove the older graphs
        self.graph.remove()
        self.moving_avg_graph.remove()

        # Plot the updated graph and moving average
        self.graph, = plt.plot(self.x, self.y, color='g', label='Rewards')
        self.moving_avg_graph, = plt.plot(self.x, self.moving_avg_y, color='b', linestyle='--', label='Moving Average')

        # Annotate the last value of the moving average
        if self.annotation:
            self.annotation.remove()
        self.annotation = self.ax.annotate(f'{moving_avg:.2f}',
                                           xy=(self.x[-1], self.moving_avg_y[-1]),
                                           xytext=(5, 5),
                                           textcoords='offset points',
                                           fontsize=10,
                                           color='blue')

        # Redraw the plot with updated data
        plt.pause(0.0001)
    def get_action_value(self, state: any, action: any, gamma: float) -> float:
        if (state, action) in self.V:
            return self.V[(state, action)]
        else:
            return 0

    # def plot_q_values(self):
    #     """Visualize the Q-values on a grid."""
    #     fig2, ax2 = plt.subplots(self.env.width, self.env.height, figsize=(15, 5))
    #
    #     for i in range(self.env.width):
    #         for j in range( self.env.height):
    #             allowed_actions = self.get_allowed_actions((i,j))
    #             state_q_values = {}
    #             for action in allowed_actions:
    #                 state_q_values[action] = self.V.get(((i, j),action),0)
    #
    #             # Create an arrow plot to represent the action Q-values
    #             ax2[i, j].quiver([0, 0, 0, 0], [0, 0, 0, 0],
    #                             [0, 1, 2, 0],
    #                             [3, 0, 0,-4],
    #                             scale=1.5, scale_units='xy', angles='xy', color=['r', 'g', 'b', 'y'])
    #
    #             ax2[i, j].set_xticks([])
    #             ax2[i, j].set_yticks([])
    #             ax2[i, j].set_xlim(-0.5, 0.5)
    #             ax2[i, j].set_ylim(-0.5, 0.5)
    #             ax2[i, j].invert_yaxis()  # Make the y-axis point upwards to match the grid orientation
    #
    #     plt.tight_layout()
    #     plt.show()