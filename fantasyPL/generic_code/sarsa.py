from matplotlib import pyplot as plt

from fantasyPL.generic_code.Q_and_Sarsa import GenericAgent
from fantasyPL.generic_code.envs import CliffWalkingEnv


class SARSAAgent(GenericAgent):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)

    def learn(self, state, action, reward, next_state, next_action, done):
        """Update Q-values using the SARSA formula."""
        super().learn( state, action, reward,
                       next_state, done, next_action=next_action, sarsa=True)

    def train(self, episodes=1000, max_steps=100, patience=10, min_delta=1.0):
        """Train the SARSA agent for a specified number of episodes."""
        super().train(episodes, max_steps, patience, min_delta, sarsa=True)


if __name__ == '__main__':

    # Example usage with the CliffWalkingEnv:
    env = CliffWalkingEnv()
    env
    agent = SARSAAgent(env, epsilon_min=0.1)
    agent.train(episodes=2000, patience=1000, min_delta=1)
    agent.print_policy()