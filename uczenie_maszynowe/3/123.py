import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_cost():  # loss function
    n = 42
    Y_pred = a * X + b
    J = (1 / n) * (np.sum(Y - Y_pred) ** 2)
    print(f'J = {J}')
    return J  # return number

# Preprocessing Input data
data = pd.read_csv('fires_thefts.csv')
X = data.iloc[:, 0]
Y = data.iloc[:, 1]
plt.scatter(X, Y)
plt.show()

a = 0
b = 0
L = 0.001  # learning rate
eps= 0.01
iteration_number = 0  # The number of iterations to perform gradient descent
current_cost = compute_cost()
prev_cost = 5000000000000000000000000

while True:
    if abs(prev_cost - current_cost) <= eps:
        print(a, b)
        print(iteration_number)
        print(current_cost)
        Y_pred = a * X + b
        plt.scatter(X, Y)
        plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')  # regression line
        plt.show()
        break
    else:
        iteration_number += 1
        prev_cost = current_cost
        current_cost = compute_cost()
        Y_pred = a * X + b  # The current predicted value of Y
        D_a = (-2 / 42) * sum(X * (Y - Y_pred))  # Derivative wrt m
        D_b = (-2 / 42) * sum(Y - Y_pred)  # Derivative wrt c
        a = a - L * D_a  # Update m
        b = b - L * D_b  # Update c

# # Performing Gradient Descent
# for i in range(iteration_number):
#     Y_pred = a*X + b  # The current predicted value of Y
#     D_a = (-2/42) * sum(X * (Y - Y_pred))  # Derivative wrt m
#     D_b = (-2/42) * sum(Y - Y_pred)  # Derivative wrt c
#     a = a - L * D_a  # Update m
#     b = b - L * D_b  # Update c
#
# print(a, b)
#
# # Making predictions
# Y_pred = a*X + b
#
# plt.scatter(X, Y)
# plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')  # regression line
# plt.show()