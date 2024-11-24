import gc

import numpy as np

from fantasyPL.generic_code.double_q_learning import DoubleQLearningAgent
from fantasyPL.generic_code.envs import MaximizationBiasEnv
from fantasyPL.generic_code.plotting import plot_line_dict
from fantasyPL.generic_code.q_learning import QLearningAgent


def run_experiment(agents):
    """Run multiple instances of an agent and average the results."""
    rewards_list = []
    left_moves_list = []


    for agent in agents:

        agent.train()

        # Track rewards and left moves
        rewards = agent.plotter.y  # Total rewards per episode

        left_moves_by_episode = [
            {k[-1]: v for k, v in x.items()}.get("left", 0) for x in agent.plotter.action_values.values()
        ]
        rewards_list.append(rewards)
        left_moves_list.append(left_moves_by_episode)
    # Average the results across runs
    avg_rewards = np.mean(rewards_list, axis=0)
    avg_left_moves = np.mean(left_moves_list, axis=0)
    return avg_rewards, avg_left_moves


def calculate_moving_average(data, window_size):
    """Calculate moving average for smoother visualization."""
    return [
        (n + window_size, sum(data[i:i + window_size]) / window_size)
        for n, i in enumerate(range(len(data) - window_size + 1))
    ]

if __name__ == '__main__':

    # Parameters
    env = MaximizationBiasEnv()  # Or CliffWalkingEnv()
    num_runs = 10000  # Number of runs for averaging
    window_size = 1

    # Agent parameters
    agent_params = {
        "env":env,
        "epsilon": 0.1,
        "epsilon_min": 0.01,
        "episodes": 500,
        "learning_rate": 0.1,
        "discount_factor": 1,
        "epsilon_strategy": "fixed",
        "verbose": True,
        'live_plot':False
    }
    q_agents = [QLearningAgent(**agent_params)  for i in range(num_runs)]
    double_q_agents =   [DoubleQLearningAgent(**agent_params)  for i in range(num_runs)]
    # Run experiments for both agents
    avg_rewards_q, avg_left_moves_q = run_experiment( q_agents)
    avg_rewards_dq, avg_left_moves_dq = run_experiment(double_q_agents)

    # Calculate moving averages
    left_moving_av_q = calculate_moving_average(avg_left_moves_q, window_size)
    left_moving_av_dq = calculate_moving_average(avg_left_moves_dq, window_size)
    rewards_moving_av_q = calculate_moving_average(avg_rewards_q, window_size)
    rewards_moving_av_dq = calculate_moving_average(avg_rewards_dq, window_size)

    # Prepare data for plotting
    lines_dict = {
        'DoubleQLearningAgent - Left Moves': [[x[0] for x in left_moving_av_dq], [x[1] for x in left_moving_av_dq]],
        'QLearningAgent - Left Moves': [[x[0] for x in left_moving_av_q], [x[1] for x in left_moving_av_q]],
        'DoubleQLearningAgent - Rewards': [[x[0] for x in rewards_moving_av_dq], [x[1] for x in rewards_moving_av_dq]],
        'QLearningAgent - Rewards': [[x[0] for x in rewards_moving_av_q], [x[1] for x in rewards_moving_av_q]],
    }

    # Plot the results
    plot_line_dict(lines_dict)
    print("here")