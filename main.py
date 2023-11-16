from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math

def choose_training_data(data):
    rows = len(data)
    indices = np.sort(np.random.choice(rows, math.floor(rows * 0.95), replace=False))
    training_data = np.column_stack((indices, data[indices]))
    return training_data

def choose_test_data(data, training_data):
    indices = np.setdiff1d(range(len(data)), training_data[:, 0])
    test_data = np.column_stack((indices, data[indices]))
    return test_data

def build_matrix(data, degree):
    rows = data.shape[0]
    res = np.ones((rows, degree + 1), dtype='float')
    for i in range(rows):
        for j in range(1, degree + 1):
            res[i, j] = data[i][0] ** j
    return res

def estimate_equation(data, degree):
    X = build_matrix(data, degree)
    Xt = np.transpose(X)
    v = data[:, 1]
    return np.dot(np.linalg.inv(np.dot(Xt, X)), np.dot(Xt, v))

def log_test_data(test_data, coefficients, mode):
    print(f"\n\n{mode} ESTIMATION RESULT =============================================")
    print("{:<20} {:<25} {:<25}".format('Real value', 'Estimated Value', 'Error'))
    for i in range(test_data.shape[0]):
        index = test_data[i][0]
        real_value = test_data[i][1]
        estimated_value = sum(coefficients[j] * (index ** j) for j in range(len(coefficients)))
        print("{:<20} {:<25} {:<25}".format(real_value, estimated_value, real_value - estimated_value))

if __name__ == "__main__":

    # Extract data
    df = pd.read_csv('covid_cases.csv')
    data = df[input("> Country's name: ")]
    rows = len(data)
    training_data = choose_training_data(data)
    test_data = choose_test_data(data, training_data)

    # Plot estimated linear/polynomial
    degree = 1  # for linear, change to 2 for quadratic, etc.
    coefficients = estimate_equation(training_data, degree)

    x = np.linspace(0, rows, rows)
    y = sum(coefficients[j] * (x ** j) for j in range(len(coefficients)))

    plt.plot(x, y, label=f'Estimated Degree-{degree} Polynomial')
    plt.scatter(test_data[:, 0], test_data[:, 1], marker='.', c='green', label='Test Data')
    plt.plot(data, linewidth='0.5', label='Actual Data')
    plt.legend()
    plt.show()

    # Log test data info
    log_test_data(test_data, coefficients, f"Degree-{degree}")
