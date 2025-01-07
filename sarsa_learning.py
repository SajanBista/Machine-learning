#importing necessary liabaries
import gym
import numpy as np
import time, pickle, os 


env = gym.make('FrozenLake-v1', render_mode='human') #setting the environment

#initializing the parameters 
epsilon = 0.9#initial epsilon
min_epsilon = 0.1#minimum epsilon
max_epsilon = 1.0#maximum epsilon
decay_rate = 0.01 #decay rate

#setting the number of episodes and maximum steps
total_episodes = 10000 #number of episodes
max_steps = 100#maximum steps

lr_rate = 0.89 #learning rate
gamma = 0.96 #discount factor

#initializing the Q-table
Q = np.zeros((env.observation_space.n, env.action_space.n,3))
    
def choose_action(state):  #choosing the action
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample() #exploration because of epsilon
    else:
        action = np.argmax(Q[state, :]) #exploitation because of the Q-table
    return action

def learn(state, state2, reward, action, action2): #learning the Q-table using SARSA 
    predict = Q[state, action] #predicting the Q-value
    target = reward + gamma * Q[state2, action2] #target value
    Q[state, action] = Q[state, action] + lr_rate * (target - predict) #updating the Q-value


rewards = 0 #initializing the rewards

for episode in range(total_episodes): #iterating through the episodes
    t = 0 #initializing the steps
    state, _ = env.reset()  #resetting the environment
    action = choose_action(state) #choosing the action
    
    while t < max_steps:  #iterating through the steps
        env.render() #rendering the environment

        state2, reward, terminated, truncated, info = env.step(action) #taking the action and observing the outcome

        action2 = choose_action(state2)#choosing the next action

        learn(state, state2, reward, action, action2)#learning the Q-table

        state = state2 #updating the state
        action = action2 #updating the action

        t += 1 #incrementing the steps
        rewards += reward  #incrementing the rewards
        if terminated or truncated: # why the episode is terminated or truncated because of the maximum steps
            break

    
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode) #decaying the epsilon
    
    time.sleep(0.1) #sleeping for 0.1 seconds

print("Score over time: ", rewards / total_episodes) #printing the score
print(Q)


with open("frozenLake_qTable_sarsa.pkl", 'wb') as f:
    pickle.dump(Q, f) #saving the Q-table
    f.close()
    