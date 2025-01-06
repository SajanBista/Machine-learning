import gym
import numpy as np
import time, pickle, os
env = gym.make('FrozenLake-v1', render_mode='human')


epsilon = 0.9
min_epsilon = 0.1
max_epsilon = 1.0
decay_rate = 0.01

total_episodes = 10000
max_steps = 100

lr_rate = 0.81
gamma = 0.96

Q = np.zeros((env.observation_space.n, env.action_space.n))
    
def choose_action(state):
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action

def learn(state, state2, reward, action, action2):
    predict = Q[state, action]
    target = reward + gamma * Q[state2, action2]
    Q[state, action] = Q[state, action] + lr_rate * (target - predict)

# Start
rewards = 0

for episode in range(total_episodes):
    t = 0
    state, _ = env.reset()  # Get state and additional info
    action = choose_action(state)
    
    while t < max_steps:
        env.render()

        state2, reward, terminated, truncated, info = env.step(action)

        action2 = choose_action(state2)

        learn(state, state2, reward, action, action2)

        state = state2
        action = action2

        t += 1
        rewards += reward  # Add reward, not 1, to track score

        if terminated or truncated:
            break

    # Epsilon decay for exploration-exploitation trade-off
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    # Optional: Clear console
    # os.system('clear')
    time.sleep(0.1)

print("Score over time: ", rewards / total_episodes)
print(Q)

# Save the Q-table for future use
with open("frozenLake_qTable_sarsa.pkl", 'wb') as f:
    pickle.dump(Q, f)

import numpy as np
import gym

# Helper function: Epsilon-greedy policy
def epsilon_greedy_policy(state, Q, epsilon, action_space):
    if np.random.rand() < epsilon:
        return np.random.choice(action_space)  # Random action (exploration)
    else:
        return np.argmax(Q[state])  # Best action based on Q-values (exploitation)

# Initialize environment
env = gym.make('FrozenLake-v1', is_slippery=False)  # example environment

# Initialize Q-table with zeros (state x action)
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Parameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 0.1  # Exploration rate
num_episodes = 1000

"""
# SARSA algorithm
for episode in range(num_episodes):
    state = env.reset()  # Reset environment to start a new episode
    action = epsilon_greedy_policy(state, Q, epsilon, env.action_space.n)  # Choose action using epsilon-greedy
    
    done = False
    while not done:
        next_state, reward, done, info = env.step(action)  # Take action and observe outcome
        next_action = epsilon_greedy_policy(next_state, Q, epsilon, env.action_space.n)  # Choose next action
        
        # Update Q-value using SARSA formula
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])
        
        # Update state and action
        state = next_state
        action = next_action"""

