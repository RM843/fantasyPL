from itertools import combinations, product

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




def is_legal_team():

    is_affordable()
    max_teams()
    positions_fit()

def get_best_action(state,V,gamma,get_allowed_actions, transition_model, reward_function):

    available_actions = get_allowed_actions(state)
    action_values = []
    for action in available_actions:

        next_state = transition_model(state, action)
        reward = reward_function(next_state)
        next_state_value = V.get(next_state, 0)
        action_values.append(gamma * next_state_value + reward)

    best_action_value = max(action_values)

    best_action = available_actions[action_values.index(best_action_value)]
    return best_action, best_action_value
def value_iteration(states, get_allowed_actions, transition_model, reward_function, gamma=1, epsilon=0.0001):

    V = {}
    while True:
        deltas=0
        for state in states:


            delta = 0
            # current value of state
            v = V.get(state, 0)
            best_action,best_action_value = get_best_action(state, V, gamma,get_allowed_actions, transition_model, reward_function)
            V[state] = best_action_value

            delta = max(delta, abs(v - V[state]))
            deltas+=delta


        print("Deltas:",deltas)
        # Check for convergence
        if deltas == 0:
            break

    assert len(initial_states)*ROUNDS==len(V),f"""{len(initial_states)*(ROUNDS-1)} possible
            states {len(V)} found"""

    # Extract optimal policy
    policy = {}
    for s in states:
        policy[s] = get_best_action(s,V,gamma,get_allowed_actions, transition_model, reward_function)
    initial_state_scores = [(k,v) for k,v in V.items() if k[0]==0]
    i = [v for (k,v) in initial_state_scores].index(max([v for (k,v) in initial_state_scores]))

    return V,policy,initial_state_scores[i][0]

def get_allowed_actions(state):
    actions =[(-1,None)] # do nothing option
    rnd, selection = state

    for i,selct in enumerate(selection):
        for opt in ALL_OPTIONS:
            if opt not in selection:
                actions.append((i,opt))
    return actions

def is_final(state):
    rnd, selection = state
    return rnd >= ROUNDS
def transition_model(state,action):
    rnd, selection = state
    index_to_change, option =action

    selection = [option if i==index_to_change else
                      opt for i, opt in enumerate(selection)]
    selection.sort()

    return (rnd+1,tuple(selection))

def reward_function(state):
    rnd, selection = state

    score = 0
    for selct in selection:

        score +=scores[selct][rnd-1]
    return score


initial_states = [x for x in list(combinations(ALL_OPTIONS, 2))]

states = list(product(range(ROUNDS), initial_states))
v,policy,best_initial_state = value_iteration(states=states,
                get_allowed_actions=get_allowed_actions,
                transition_model=transition_model,
                reward_function=reward_function)

state = best_initial_state
strat = []
while not is_final(state):
    rnd,selection =state
    action,value = policy[state]
    if not rnd==0:

        strat.append({"state":state,"action":action,"value":value})
    state = transition_model(state,action)
    if rnd==0:
        strat.append({"state": (0,state[1]), "action": None, "value": value})

print(v)