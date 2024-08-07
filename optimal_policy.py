from itertools import combinations

# Example Data
ALL_OPTIONS = ['A', 'B', 'C', 'D']
ROUNDS = 5
scores = {
    'A': [2, 6, 5, 6, 40],
    'B': [6, 5, 6, 5, 6],
    'C': [7, 8, 7, 8, 9],
    'D': [8, 7, 8, 7, 8]
}
initial_selection_size = 2  # Size of the initial selection


def calculate_max_score(initial_selection, scores):
    """
    Calculates the maximum score achievable given an initial selection of options,
    the list of all options, their respective scores per round, and the total number of rounds.

    This function uses a dynamic programming approach to iteratively build up the maximum
    score for each possible selection of options at each round, considering the constraint
    that only one option can be changed between rounds.

    Parameters:
    - initial_selection: tuple, the starting set of options.
    - scores: dict, expected scores for each option per round.

    Returns:
    - max_score: int, the maximum score achievable with the given initial selection.

    Algorithm:
    1. Initialize the DP table.
    2. Set the initial selection's score in the DP table.
    3. For each round:
        a. For each selection in the current round's DP table:
            i. Calculate the current score.
            ii. Generate new selections by replacing each option with another option not in the current selection.
            iii. Calculate the new score for each new selection.
            iv. Update the DP table for the next round with the new scores.
    4. Extract the maximum score from the last round's DP table.


    Return the maximum score in dp[rounds]
    ```
    """
    # Initialize DP table
    dp = [{} for _ in range(ROUNDS )]
    dp[0][tuple(initial_selection)] = (sum(scores[opt][0] for opt in initial_selection),initial_selection)

    # DP Transition
    for round_index in range(ROUNDS-1):
        next_round_index = round_index+1
        for selection in dp[round_index]:
            current_score = dp[round_index][selection][0]
            for i in range(len(selection)):
                # score for no changes
                next_round_score = sum(scores[opt][next_round_index] for opt in selection)
                new_score = current_score + next_round_score
                dp[next_round_index][selection] = (new_score, selection)
                for new_option in ALL_OPTIONS:
                    if new_option not in selection:
                        new_selection = list(selection)
                        new_selection[i] = new_option
                        new_selection = tuple(new_selection)
                        # is_final_round= round_index == ROUNDS-1
                        # if not is_final_round:
                        next_round_score =  sum(scores[opt][next_round_index] for opt in new_selection)
                        new_score = current_score + next_round_score
                        # else:
                        #     new_score = current_score

                        if new_selection not in dp[next_round_index]:
                            dp[next_round_index][new_selection] = (new_score,new_selection)
                        else:
                            dp[next_round_index][new_selection] = max(dp[next_round_index][new_selection], (new_score, new_selection), key=lambda x: x[0])

        # Extract Result
    max_score = -float('inf')
    best_end_selection  = None
    for selection ,  (score, _) in dp[-1].items():
        if score > max_score:
            max_score = score
            best_end_selection  = selection 

    # Backtrack to find the sequence of selections
    best_sequence = []
    for j in range(ROUNDS-1, 0, -1):
        best_sequence.append(best_end_selection )
        best_end_selection = dp[j][best_end_selection][1]
    best_sequence.append(initial_selection)
    best_sequence.reverse()

    return max_score, best_sequence


#
# # Generate all possible initial selections
# all_initial_selections = list(combinations(ALL_OPTIONS, initial_selection_size))
#
# # Initialize variables to track the best selection and score
# best_initial_selection = None
# best_score = -float('inf')
# best_sequence = []
#
# # Evaluate each initial selection
# for initial_selection in all_initial_selections:
#     score, sequence = calculate_max_score(initial_selection, scores)
#     if score > best_score:
#         best_score = score
#         best_initial_selection = initial_selection
#         best_sequence = sequence
#
# # Output the best initial selection, the maximum score, and the sequence of selections
# print("Best initial selection:", best_initial_selection)
# print("Maximum score:", best_score)
# print("Sequence of selections per round:")
# for round_idx, selection in enumerate(best_sequence):
#     print(f"Round {round_idx}: {selection}")

def is_legal_team():

    is_affordable()
    max_teams()
    positions_fit()

def value_iteration(initial_states, get_allowed_actions, transition_model, reward_function, gamma=1, epsilon=0.0001):

    V = {}
    for initial_state in initial_states:
        state = initial_state
        while True:
            delta = 0

            # current value of state
            v = V.get(state,0)
            available_actions =get_allowed_actions(state)
            action_values = []
            for action in available_actions:
                reward = reward_function(state,action)
                next_state,_ = transition_model(state, action)
                next_state_value = V.get(next_state,0)
                action_values.append(gamma*next_state_value+reward)

            best_action_value = max(action_values)
            v[state] = best_action_value
            best_action = action_values.index(best_action_value)
            state, end = transition_model(state, best_action)
            delta = max(delta, abs(v - V[state]))
            if end:
                break
            # # # Check for convergence
            # if delta < epsilon:
            #     break

    # Extract optimal policy
    # policy = {}
    # for s in states:
    #     policy[s] = max(actions,
    #                     key=lambda a: sum(
    #                         transition_model(s, a, s_next) *
    #                         (reward_function(
    #                             s, a, s_next) + gamma * V[s_next])
    #                         for s_next in states))
    return V

inital_states

value_iteration(states, get_allowed_actions, transition_model, reward_function