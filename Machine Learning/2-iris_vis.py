import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
test_idx = [0, 50, 100]

#training data
training_target = np.delete(iris.target, test_idx)
training_data = np.delete(iris.data, test_idx, axis = 0)

#testing data
testing_target = iris.target[test_idx]
testing_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(training_data, training_target)

print testing_target
print clf.predict(testing_data)

#vis code
from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()  
tree.export_graphviz(clf, out_file=dot_data,  
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydot.graph_from_dot_data(dot_data.getvalue())  
graph.write_svg("iris.svg")

print testing_data[1], testing_target[1]
print iris.feature_names, iris.target_names
