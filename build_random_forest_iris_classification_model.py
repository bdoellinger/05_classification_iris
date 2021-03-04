from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import pickle

# get data for features X and labels Y 
iris = datasets.load_iris()
X = iris.data
Y = iris.target

# use random forest classifier to generate model (using X and Y)
clf = RandomForestClassifier()
clf.fit(X,Y)

# mean accuracy
print(clf.score(X,Y))

# save generated model (such that we don't have to generate again)
pickle.dump(clf, open("iris_classifier.pkl","wb"))
