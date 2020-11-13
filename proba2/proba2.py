from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier

wine = load_wine()

default = DecisionTreeClassifier(criterion='entropy', random_state=0)
default.fit(wine.data, wine.target)

print("[[class_0, class_1, class_2]] of probability : ", default.predict_proba([[0.4, 10.7, 20.3, 10.6, 12, 2.8, 3, 0.1, 2.5, 5.1, 1.0, 3.2, 123]]))
print("predict : ", default.predict([[0.4, 10.7, 20.3, 10.6, 12, 2.8, 3, 0.1, 2.5, 5.1, 1.0, 3.2, 123]]))
print("Leaf Count : ", default.get_n_leaves())
print("Depth : ", default.get_depth())
