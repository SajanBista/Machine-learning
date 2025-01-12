import numpy as np
import matplotlib.pyplot as plt

# Defining the grid world size
grid_size = 4
goal_state = (3, 3)
wall_states = [(1, 1), (2, 2)]  # walls/obstacles

# Defining Rewards values
rewards = np.full((grid_size, grid_size), -1)  # reward for moving in any state is -1
rewards[goal_state] = 10  # reward for reaching the goal is 10

# Define actions (Up, Down, Left, Right)
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # (dx, dy) for up(-1,0), down(1,0), left(0,-1), right(0,1)

# Value Iteration function
def value_iteration(grid_size, rewards, gamma=0.9, threshold=1e-6):
    V = np.zeros((grid_size, grid_size))  # Initialize value function to 0
    
    while True:
        delta = 0
        new_V = V.copy()  # Creating a copy of the value function to update simultaneously.

        for i in range(grid_size):
            for j in range(grid_size):
                if (i, j) == goal_state or (i, j) in wall_states:  # Skip walls or goal states
                    continue

                max_value = -float('inf')  # Initialize max value for the current state to negative infinity

                # Check each possible action (Up, Down, Left, Right)
                for action in actions:
                    new_i, new_j = i + action[0], j + action[1]

                    # Ensure new position is within bounds and not a wall
                    if 0 <= new_i < grid_size and 0 <= new_j < grid_size and (new_i, new_j) not in wall_states:
                        # Compute the expected value (reward + discounted future value)
                        value = rewards[new_i, new_j] + gamma * V[new_i, new_j]
                        max_value = max(max_value, value)

                new_V[i, j] = max_value  # Updating the value function for the current state
                delta = max(delta, abs(new_V[i, j] - V[i, j]))  # Update the maximum change in value function
        
        V = new_V  # Update value function
        
        if delta < threshold:  # If the maximum change in value function is less than the threshold, exit
            break

    return V

# Run value iteration
optimal_values = value_iteration(grid_size, rewards)
print("Optimal Values:")
print(optimal_values)

# Visualization of value function
def plot_value_function(V):
    plt.figure(figsize=(6,6))
    plt.imshow(V, cmap='viridis', interpolation='nearest')
    for i in range(grid_size):
        for j in range(grid_size):
            if (i, j) != goal_state and (i, j) not in wall_states:
                plt.text(j, i, f'{V[i, j]:.2f}', ha='center', va='center', color='white', fontsize=12)
    plt.colorbar(label='Value')
    plt.title("Optimal Value Function")
    plt.show()

# Visualization of optimal policy (actions)
def plot_policy(V):
    policy = np.full((grid_size, grid_size), None)
    
    for i in range(grid_size):
        for j in range(grid_size):
            if (i, j) == goal_state or (i, j) in wall_states:  # Skip goal and walls
                continue
            
            best_action = None
            max_value = -float('inf')

            # Check each possible action and pick the one with the maximum value
            for action in actions:
                new_i, new_j = i + action[0], j + action[1]
                
                if 0 <= new_i < grid_size and 0 <= new_j < grid_size and (new_i, new_j) not in wall_states:
                    value = rewards[new_i, new_j] + gamma * V[new_i, new_j]
                    if value > max_value:
                        max_value = value
                        best_action = action
            
            policy[i, j] = best_action

    # Plotting policy with arrows
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(np.zeros((grid_size, grid_size)), cmap='gray', interpolation='nearest')

    for i in range(grid_size):
        for j in range(grid_size):
            if (i, j) == goal_state:
                ax.text(j, i, 'G', ha='center', va='center', fontsize=15, color='green')
            elif (i, j) in wall_states:
                ax.text(j, i, 'X', ha='center', va='center', fontsize=15, color='red')
            else:
                # Plotting the optimal policy arrows
                action = policy[i, j]
                if action == (-1, 0):  # up
                    ax.arrow(j, i, 0, -0.3, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
                elif action == (1, 0):  # down
                    ax.arrow(j, i, 0, 0.3, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
                elif action == (0, -1):  # left
                    ax.arrow(j, i, -0.3, 0, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
                elif action == (0, 1):  # right
                    ax.arrow(j, i, 0.3, 0, head_width=0.1, head_length=0.1, fc='blue', ec='blue')

    plt.xlim(-0.5, grid_size - 0.5)
    plt.ylim(grid_size - 0.5, -0.5)
    plt.title("Optimal Policy with Actions (Arrows)")
    plt.show()

# Visualize the final result
plot_value_function(optimal_values)
plot_policy(optimal_values)
