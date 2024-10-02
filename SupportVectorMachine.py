from sklearn import svm
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd

#Load the data
data = load_breast_cancer()
dataset = pd.DataFrame(data=data['data'], columns=data['feature_names'])
#print(dataset)

#Prepare data
X = dataset.copy()
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33)

#Create decision tree
svmachine = svm.SVC()
clf = svmachine.fit(X_train, y_train)
#print(clf.get_params())

#Print predictions
predictions = svmachine.predict(X_test)
#print(predictions)
#print(clf.predict_proba(X_test))

#Print results
matrix = confusion_matrix(y_test, predictions)
print(matrix)

print(classification_report(y_test, predictions))