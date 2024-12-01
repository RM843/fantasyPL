import numpy as np


# Define a simple environment with deterministic transitions
# For simplicity, let's assume there are 5 states and
# moving from one state to the next gives a reward of 1, with state 4 being terminal


class SimpleEnvironment:
	def __init__(self, num_states=5):
		self.num_states = num_states

	def step(self, state):
		reward = 0
		terminal = False

		if state < self.num_states - 1:
			next_state = state + 1
			reward = 1
		else:
			next_state = state
			terminal = True

		return next_state, reward, terminal

	def reset(self):
		return 0 # Start from state 0


# Define a random policy for the sake of demonstration
def random_policy(state, num_actions=5):
	return np.random.choice(num_actions)


# Monte Carlo Policy Evaluation function
def monte_carlo_policy_evaluation(policy, env, num_episodes, gamma=1.0):
	value_table ={}
	returns = {}

	for _ in range(num_episodes):
		state = env.reset()
		episode = []
		# Generate an episode
		while True:
			action,_ = policy(state)
			next_state, reward, terminal = env.step(action)
			episode.append((next_state, reward))
			if terminal:
				break
			state = next_state

		# Calculate the return and update the value table
		G = 0
		for state, reward in reversed(episode):
			G = gamma * G + reward
			returns[state] = returns.get(state,[]) +[G]
			value_table[state] = np.mean(returns[state])

	return value_table

if __name__ == '__main__':

	# Define the number of episodes for MC evaluation
	num_episodes = 1000

	# Create a simple environment instance
	env = SimpleEnvironment(num_states=5)

	# Evaluate the policy
	v = monte_carlo_policy_evaluation(random_policy, env, num_episodes)

	print("The value table is:")
	print(v)
