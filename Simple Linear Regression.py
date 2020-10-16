import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd

# data = pd.read_csv('filename.type')

def estimate_values(x, y):
    n = np.size(x)
    m_x, m_y = np.mean(x), np.mean(y)  # calculating mean of x and y in a single line
    SS_xy = n * (np.sum(y * x)) - np.sum(x) * np.sum(y)  # sum can replaced with mean.. I guess..
    SS_xx = n * (np.sum(x * x)) - (np.sum(x)) ** 2
    b_1 = SS_xy / SS_xx
    b_0 = (np.sum(y) - b_1 * (np.sum(x))) / n
    return b_0, b_1


def plot_regression_line(x, y, b):
    plt.scatter(x, y, color="m", marker='o', s=30)
    # plotting a scatter graph
    y_pred = b[0] + b[1] * x
    plt.plot(x, y_pred, color="g")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


# driver

# x = data[data.columns[0:1]]  # x contains data present in the first columnn in data
# y = data[data.columns[-1]]

x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

# more functionality in a numpy array
# to calculate estimated value
b = estimate_values(x, y)  # a function which returns the data in an array format
print('val1:', b[0], 'val2:', b[1])  # printing the 'parameters' that are in array b
plot_regression_line(x, y, b)  # a function to plot the graph
