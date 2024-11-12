import numpy as np


class EpsilonDecay:
    def __init__(self, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, episodes=1000, strategy="linear"):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.episodes = episodes
        self.strategy = strategy  # Strategy: 'linear', 'exponential', 'inverse_sigmoid', 'adaptive'
        self.epsilon_history = []

    def decay(self, done, current_episode=None, success_rate=None):
        """Decays epsilon based on the chosen strategy."""
        if done:
            if self.strategy == "linear":
                self.linear_decay()
            elif self.strategy == "exponential":
                self.exponential_decay()
            elif self.strategy == "inverse_sigmoid" and current_episode is not None:
                self.inverse_sigmoid_decay(current_episode)
            elif self.strategy == "adaptive" and success_rate is not None:
                self.adaptive_decay(success_rate)
            elif self.strategy== "fixed":
                pass

        self.epsilon_history.append(self.epsilon)

    def linear_decay(self):
        """Linear decay - reduces epsilon linearly."""
        decay_rate = (self.epsilon - self.epsilon_min) / (self.episodes / 2)
        self.epsilon = max(self.epsilon_min, self.epsilon - decay_rate)

    def exponential_decay(self):
        """Exponential decay - reduces epsilon exponentially."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def inverse_sigmoid_decay(self, current_episode):
        """Inverse sigmoid decay - epsilon decreases rapidly at first, then slowly."""
        k = 10  # Controls how fast the decay happens
        self.epsilon = self.epsilon_min + (1 - self.epsilon_min) / (1 + np.exp(k * (current_episode / self.episodes - 0.5)))
        self.epsilon = max(self.epsilon_min, self.epsilon)

    def adaptive_decay(self, success_rate):
        """Adaptive decay - reduces epsilon based on success rate (performance-based)."""
        # If agent is doing well, decay faster; if not, decay slower
        if success_rate >= 0.8:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        elif success_rate < 0.5:
            self.epsilon = min(1.0, self.epsilon * 1.05)  # Increase exploration slightly

    def get_epsilon(self):
        """Returns the current epsilon value."""
        return self.epsilon