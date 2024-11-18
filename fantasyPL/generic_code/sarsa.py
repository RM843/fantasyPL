from matplotlib import pyplot as plt

from fantasyPL.generic_code.Q_and_Sarsa import GenericAgent
from fantasyPL.generic_code.envs import CliffWalkingEnv, SelectorGameMini
from helper_methods import print_list_of_dicts


class SARSAAgent(GenericAgent):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)

    def calculate_td_target(self, state, action, reward, next_state, done, next_action=None) -> float:
        """SARSA-specific TD target calculation."""
        # Use the next action chosen using the SARSA policy for TD calculation
        state_rep_to_use = self._get_state(state=next_state, action=next_action)
        return reward + self.discount_factor * self.q_table.get_q_value(state_rep_to_use) * (1 - done)

    def learn_episode(self,max_steps):
        state = self.env.reset()
        action = self.choose_action_based_on_policy(state)
        total_reward = 0.0
        actions_count = {}
        actions_count[action] = actions_count.get(action,0)+1
        for step in range(max_steps):

            next_state, reward, done = self.env.step(action)
            actions_count[(state,action)] = actions_count.get(action, 0) + 1
            if not done:
                next_action = self.choose_action_based_on_policy(next_state)
            else:
                next_action = None  # Assuming env has no actions after terminal state
            self.learn(state=state, action=action, reward=reward, next_state=next_state, done=done,
                       next_action=next_action)
            state, action = next_state, next_action

            total_reward += reward

            if done:
                break
        return total_reward,actions_count
    def learn(self, state, action, reward, next_state, next_action, done):
        """Update Q-values using the SARSA formula."""
        super().learn( state, action, reward,
                       next_state, done, next_action=next_action)

    def train(self, episodes=1000, max_steps=100):
        """Train the SARSA agent for a specified number of episodes."""
        super().train( max_steps)


if __name__ == '__main__':
    scores = {
        "A": {1: 2, 2: 335, 3: 1, 4: 16, 5: 5},
        "B": {1: 9, 2: 5, 3: 6, 4: 5, 5: 6},
        "C": {1: 1, 2: 8, 3: 1, 4: 8, 5: 9},
        "D": {1: 8, 2: 7, 3: 8, 4: 7, 5: 8},
        "E": {1: 4, 2: 7, 3: 5, 4: 8, 5: 4}
    }
    simple_scores = {
        "A": {1: 2, 2: 8},
        "B": {1: 9, 2: 5},
        "C": {1: 5, 2: 7}
    }
    # scores = simple_scores

    ALL_OPTIONS = list(scores.keys())
    ROUNDS = len(scores[ALL_OPTIONS[0]].keys())
    initial_selection_size = 2
    env = SelectorGameMini(ALL_OPTIONS, range(1, ROUNDS), scores, initial_selection_size)



    # Example usage with the CliffWalkingEnv:
    env = CliffWalkingEnv()
    agent = SARSAAgent(env, epsilon_min=0.01,episodes=400000)
    agent.train()
    # agent.print_policy()
    q_strat, value = agent.run_policy(policy=agent.get_policy())
    print_list_of_dicts(q_strat)
    print(f"Total Reward = {value}")