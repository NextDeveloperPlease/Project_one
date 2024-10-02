from sklearn.neural_network import MLPClassifier
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
NN = MLPClassifier(hidden_layer_sizes=(100,))
NN = NN.fit(X_train, y_train)
#print(clf.get_params())

#Evaluate predictions
predictions = NN.predict(X_test)
#print(predictions)
#print(clf.predict_proba(X_test))

#Evaluate predictions (of training set)
training_prediction = NN.predict(X_train)
print(predictions)

#Print results
matrix = confusion_matrix(y_test, predictions)
print(matrix)

print(classification_report(y_test, predictions))

'''
How would you go about analyzing the data to make meaningful understandings of the errors and successes?
For example, if my model always guesses wrong on a specific situation, how would I go about finding that
in the data? What should stand out when analyzing predictions?

    Look at the tree (top of the tree has most impactful nodes)
    Look at coorelation in features
    Look at feature information (Maybe it is just completely random)
    Really deep dive into the features (Feature engineering)

Running the same program multiple times (not setting random_state) gives different precision/recall scores.
If I run the code once, getting 88%, and another getting 96%, is it likely the second run has a better model?
Would it be smart to run the same algorithm multiple times with different random_state values to get the model
with the highest score?

    Re-running the same program can be useful, but you don't know how well it will work on the test set
    This means you should look at hyperperameters more than anything.

Should I clean the data before or after I pull out my test set? Won't I see the test data before which could
subconsciously skew my hyperparameters?

    Technically no (Treat test set as if it is real world input (Create a pipeline (input all the way to prediction)
    that the training data is sent through that you can send your test set through as well))

What is the difference between Entropy and information gain? They use the same calculations (Info gain is the
weighted entropy or gini score)~~~~
Is info gain separate from Gini scores and Entropy? As in, you can use either as the score value for the info gain?

    Info gain is regardless of purity score 

How would I implement 10-fold cross validation? Is there a scikit-learn module that does this? Would you manually
create each subset?
Once you have the 10 models, do you pick the best one? What do you do with the models?

    Put in ensemble or choose the one with the best score
    Useful for examining the technique more than any specific model
    
    There is a scikit-learn to do this

Final projects: You might be able to use the same project in both classes 497 and 334

Idea: Train a model to look at datasets and determine the best algorithm to use on it.
'''
