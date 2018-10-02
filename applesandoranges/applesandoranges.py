from sklearn import tree
# 0 are bumps and 1 is smooth
features = [[140,0],[130,0],[150, 1],[170, 1]]
# 0 are orange 1 apple
labels= [0,0,1,1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print(clf.predict([[160,0]]))
