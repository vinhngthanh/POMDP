import numpy as np

def read_state_weights(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        state_weights = {}
        for line in lines[2:]:
            state, weight = line.strip().split(" ")
            state_weights[state] = int(weight)
    return state_weights

def read_state_action_state_weights(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        state_action_weights = {}
        for line in lines[2:]:
            state, action, next_state, weight = line.strip().split(" ")
            state_action_weights[(state, action, next_state)] = int(weight)
    return state_action_weights

def read_state_observation_weights(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        state_observation_weights = {}
        for line in lines[2:]:
            state, observation, weight = line.strip().split(" ")
            state_observation_weights[(state, observation)] = int(weight)
    return state_observation_weights

def read_observation_actions(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        observation_actions = []
        for line in lines[2:]:
            parts = line.strip().split(" ")
            if len(parts) == 2:
                observation, action = parts
                observation_actions.append((observation, action))
            elif len(parts) == 1:
                observation = parts[0]
                observation_actions.append((observation, None))
    return observation_actions

def normalize_state_weights(state_weights):
    total_weight = sum(state_weights.values())
    for state in state_weights:
        state_weights[state] /= total_weight
    return state_weights

def normalize_state_action_state_weights(state_action_weights):
    normalized_weights = {}
    actions_by_state = {}
    
    for (state, action, next_state), weight in state_action_weights.items():
        if (state, action) not in actions_by_state:
            actions_by_state[(state, action)] = 0
        actions_by_state[(state, action)] += weight
    
    for (state, action, next_state), weight in state_action_weights.items():
        normalized_weights[(state, action, next_state)] = weight / actions_by_state[(state, action)]
    
    return normalized_weights

def normalize_state_observation_weights(state_observation_weights):
    normalized_weights = {}
    observations_by_state = {}
    
    for (state, observation), weight in state_observation_weights.items():
        if state not in observations_by_state:
            observations_by_state[state] = 0
        observations_by_state[state] += weight
    
    for (state, observation), weight in state_observation_weights.items():
        normalized_weights[(state, observation)] = weight / observations_by_state[state]
    
    return normalized_weights

def viterbi_algorithm(observations, actions, state_weights, state_action_weights, state_observation_weights, states):
    num_states = len(states)
    num_steps = len(observations)
    
    dp = np.zeros((num_steps, num_states))
    backpointer = np.zeros((num_steps, num_states), dtype = int)

    for i, state in enumerate(states):
        dp[0, i] = state_weights[state] * state_observation_weights.get((state, observations[0]), 0)
    
    for t in range(1, num_steps):
        for j, state_j in enumerate(states):
            for i in range(num_states):
                prob = (dp[t-1, i] * 
                        state_action_weights.get((states[i], actions[t-1], state_j), 0) * 
                        state_observation_weights.get((state_j, observations[t]), 0))
                if prob > dp[t, j]:
                    dp[t, j] = prob
                    backpointer[t, j] = i

    best_path = np.zeros(num_steps, dtype=int)
    best_path[-1] = np.argmax(dp[-1, :])
    
    for t in range(num_steps - 2, -1, -1):
        best_path[t] = backpointer[t + 1, best_path[t + 1]]

    return [states[i] for i in best_path]

def write_state_sequence(filename, states):
    with open(filename, 'w') as file:
        file.write("states\n")
        n = len(states)
        file.write(f"{n}\n")
        for i, state in enumerate(states):
            if i < n - 1:
                file.write(f'"{state}"\n')
            else:
                file.write(f'"{state}"')

def main():
    state_weights = normalize_state_weights(read_state_weights('state_weights.txt'))
    state_action_weights = normalize_state_action_state_weights(read_state_action_state_weights('state_action_state_weights.txt'))
    state_observation_weights = normalize_state_observation_weights(read_state_observation_weights('state_observation_weights.txt'))
    observation_actions = read_observation_actions('observation_actions.txt')
    
    observations, actions = zip(*observation_actions)
    states = list(state_weights.keys())

    predicted_states = viterbi_algorithm(observations, actions, state_weights, state_action_weights, state_observation_weights, states)

    write_state_sequence('states.txt', predicted_states)

if __name__ == '__main__':
    main()
