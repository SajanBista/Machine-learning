"""import pickle
example_dict = {1: "Hello", 2: "World", 3: "!"}
pickle_out = open("dict.pickle", "wb")
pickle.dump(example_dict, pickle_out)
pickle_out.close()


pickle_in =open("dict.pickle", "rb")
example_dict=pickle.load(pickle_in)
print(example_dict)"""

import numpy as np
random_value = np.random.uniform(0,1)
print(random_value)

import gym

# Create environment
env = gym.make('CartPole-v1')

# Reset the environment to start a new episode
state = env.reset()

# Sample a random action from the action space
action = env.action_space.sample()

print(f"Random action chosen: {action}")



