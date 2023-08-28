import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Preprocessing Input data
data = pd.read_csv('fires_thefts.csv')
X = data.iloc[:, 0]
Y = data.iloc[:, 1]
plt.scatter(X, Y)
plt.show()

a = 0
b = 0
L = 0.001  # learning rate
iteration_number = 25000  # The number of iterations to perform gradient descent
n = float(len(X))  # Number of elements in X


# Performing Gradient Descent
for i in range(iteration_number):
    Y_pred = a*X + b  # The current predicted value of Y
    D_a = (-2/n) * sum(X * (Y - Y_pred))  # Derivative wrt m
    D_b = (-2/n) * sum(Y - Y_pred)  # Derivative wrt c
    a = a - L * D_a  # Update m
    b = b - L * D_b  # Update c

print(a, b)

# Making predictions
Y_pred = a*X + b

plt.scatter(X, Y)
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')  # regression line
plt.show()