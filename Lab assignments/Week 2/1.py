import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt

iris = datasets.load_iris()
# print(iris)

print("First five flowers: \n", iris.data[:5, :])
print("Their labels: ", iris.target[:5])
print("And the label names: ", iris.target_names)

last_five_flowers = iris.data[-5:, :]
third_feature_only = iris.data[:, 2]
def name_func(x): return iris.target_names[x]


first_ten_names = name_func(iris.target[:10])
# START ANSWER
# END ANSWER


def data_func(x): return iris.data[x]


setosa_flowers = data_func(np.where(iris.target == 0))
versicolor_flowers = data_func(np.where(iris.target == 1))
virginica_flowers = data_func(np.where(iris.target == 2))
# START ANSWER
# END ANSWER

print("Last five flowers: \n", last_five_flowers)
print("Only the third feature: ", third_feature_only)
print("All label names: ", first_ten_names)

print("Class: ", iris.target_names[0], "; Items: \n", setosa_flowers)

assert last_five_flowers.shape == (
    5, 4), "Expected a two dimensional array of shape (5,4)"
assert third_feature_only.shape == (150,), "Expected an array of shape (150,)"
assert first_ten_names.shape == (10,), "Expected an array of shape (10,)"

assert setosa_flowers.shape == (
    50, 4), "Expected a two dimensional array of shape (50,4)"
assert versicolor_flowers.shape == (
    50, 4), "Expected a two dimensional array of shape (50,4)"
assert virginica_flowers.shape == (
    50, 4), "Expected a two dimensional array of shape (50,4)"

# Create a scatterplot of the first two features, and use their labels as colour values.
plt.scatter(
    iris.data[:, 0],  # Sepal length
    iris.data[:, 1],  # Sepal width
    c=iris.target)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()

# Create a scatterplot of the third and fourth feature.
plt.scatter(
    iris.data[:, 2],  # Petal length
    iris.data[:, 3],  # Petal width
    c=iris.target)
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.show()
