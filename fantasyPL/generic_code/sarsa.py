import numpy as np
from matplotlib import pyplot as plt

from fantasyPL.generic_code.Q_anda_Sarsa import QandSARSAAgent
from fantasyPL.generic_code.envs import CliffWalkingEnv


class SARSAAgent(QandSARSAAgent):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)

    def learn(self, state, action, reward, next_state, next_action, done):
        """Update Q-values using the SARSA formula."""
        state_index = self.state_to_index(state)
        next_state_index = self.state_to_index(next_state)

        td_target = reward + self.discount_factor * self.q_table[next_state_index][next_action] * (1 - done)
        td_error = td_target - self.q_table[state_index][action]
        self.q_table[state_index][action] += self.learning_rate * td_error

        # Decay epsilon to reduce exploration over time
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, episodes=1000, max_steps=100, patience=10, min_delta=1.0):
        """Train the SARSA agent for a specified number of episodes."""
        for episode in range(episodes):
            state = self.env.reset()
            action = self.choose_action(state)
            total_reward = 0

            for step in range(max_steps):
                next_state, reward, done = self.env.step(action)
                next_action = self.choose_action(next_state)

                self.learn(state, action, reward, next_state, next_action, done)

                state = next_state
                action = next_action
                total_reward += reward

                if done:
                    break

            # Plot the performance after each episode
            self.plot_rewards(total_reward, episode, patience)

            # Check for early stopping and print progress
            early_stop = self.check_early_stopping(patience, min_delta)

            # Print training progress
            print(f"Episode: {episode + 1}, Reward: {total_reward}, Epsilon: {self.epsilon:.4f}, Early Stopping: {self.early_stopping}/{patience}")

            if early_stop:
                print(f"Early stopping triggered at episode {episode + 1}.")
                break

        # Keep the plot open after training
        plt.ioff()
        plt.show()

if __name__ == '__main__':

    # Example usage with the CliffWalkingEnv:
    env = CliffWalkingEnv()
    agent = SARSAAgent(env)
    agent.train(episodes=5000, patience=1000, min_delta=1)