import numpy as np

# Sigmoid function (used for predictions)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Compute cost function (logistic regression cost)
def compute_cost(X, y, theta):
    m = len(y)
    predictions = sigmoid(X.dot(theta))  # h_theta(x)
    cost = -1/m * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    return cost

# Compute gradient (partial derivatives of cost with respect to each theta)
def compute_gradient(X, y, theta):
    m = len(y)
    predictions = sigmoid(X.dot(theta))  # h_theta(x)
    gradient = 1/m * X.T.dot(predictions - y)
    return gradient

# Gradient descent to optimize the theta parameters
def gradient_descent(X, y, theta, alpha, num_iterations):
    cost_history = []
    for _ in range(num_iterations):
        gradient = compute_gradient(X, y, theta)  # Calculate gradient
        theta = theta - alpha * gradient  # Update the parameters
        cost_history.append(compute_cost(X, y, theta))  # Record the cost at each iteration
    return theta, cost_history

# Example dataset (X = features, y = labels)
# For simplicity, let's create a small 3-sample dataset with 2 features (and a bias term)
X = np.array([[1, 2],   # Feature 1 and Feature 2 for sample 1
              [1, 3],   # Feature 1 and Feature 2 for sample 2
              [1, 4]])  # Feature 1 and Feature 2 for sample 3

y = np.array([0, 1, 1])  # Target labels for the samples

# Add a bias column (column of ones) to X
X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)  # Add the bias term

# Initializing theta parameters (with zeros)
theta_initial = np.zeros(X.shape[1])  # theta will have 3 values (including bias)
alpha = 0.01  # Learning rate
num_iterations = 1000  # Number of iterations

# Running gradient descent
theta_final, cost_history = gradient_descent(X, y, theta_initial, alpha, num_iterations)

# Output the optimized theta and cost history
print("Optimized theta:", theta_final)
print("Cost history:", cost_history[-10:])  # Display the last 10 cost values for insight
