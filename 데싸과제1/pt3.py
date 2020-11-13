import csv

matrix = []
matrix2 = []

f = open('data2.csv', 'rt')
reader = csv.reader(f)


for line in reader:
    if line[6] == 'ufdust':
        continue
    if line[11] == 'Sum':
        continue
    else:
        matrix.append(float(line[11]))
        matrix2.append(float(line[6]))


f.close()


import random
from typing import TypeVar, List, Tuple


X = TypeVar('X')  # generic type to represent a data point
Y = TypeVar('Y')  # generic type to represent output variables


def split_data(data: List[X], prob: float) -> Tuple[List[X], List[X]]:
    """Split data into fractions [prob, 1 - prob]"""
    data = data[:]                    # Make a shallow copy
    cut = int(len(data) * prob)       # Use prob to find a cutoff
    return data[:cut], data[cut:]     # and split the shuffled list there.


def train_test_split(xs: List[X],
                     ys: List[Y],
                     test_pct: float) -> Tuple[List[X], List[X], List[Y], List[Y]]:
    # Generate the indices and split them.
    idxs = [i for i in range(len(xs))]
    train_idxs, test_idxs = split_data(idxs, 1 - test_pct)

    return ([xs[i] for i in train_idxs],  # x_train
            [xs[i] for i in test_idxs],   # x_test
            [ys[i] for i in train_idxs],  # y_train
            [ys[i] for i in test_idxs])   # y_test


x_train, x_test, y_train, y_test = train_test_split(matrix, matrix2, 0.3)



from typing import Tuple
from scratch.linear_algebra import Vector
from scratch.statistics import correlation, standard_deviation, mean

def least_squares_fit(x: Vector, y: Vector) -> Tuple[float, float]:
    """
    Given two vectors x and y,
    find the least-squares values of alpha and beta
    """
    beta = correlation(x, y) * standard_deviation(y) / standard_deviation(x)
    alpha = mean(y) - beta * mean(x)
    return alpha, beta


alpha, beta = least_squares_fit(x_train, y_train);

print("alpha값 : ")
print(alpha)
print('beta값 : ')
print(beta)

# ---------------------------------------------------------------------
print("#--------------------------------------------")

from scratch.simple_linear_regression import total_sum_of_squares
from scratch.simple_linear_regression import sum_of_sqerrors

def r_squared(alpha: float, beta: float, x: Vector, y: Vector) -> float:
    """
    the fraction of variation in y captured by the model, which equals
    1 - the fraction of variation in y not captured by the model
    """
    return 1.0 - (sum_of_sqerrors(alpha, beta, x, y) /
                  total_sum_of_squares(y))


print("R^2값 : ")
print(r_squared(alpha, beta, x_test, y_test))


