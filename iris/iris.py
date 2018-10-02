from sklearn.datasets import load_iris
import numpy as np
iris = load_iris()
text_idx = [0,50,100]
from sklearn import tree

# training data
train_target = np.delete(iris.target, text_idx)
train_data = np.delete(iris.data, text_idx, axis=0)

# testing data

test_target = iris.target[text_idx]
test_data = iris.data[text_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data,train_target)

print(test_target)
print(clf.predict(test_data))

