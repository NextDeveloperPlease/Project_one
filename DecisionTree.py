from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd

#Load the data
#data = load_breast_cancer()
#dataset = pd.DataFrame(data=data['data'], columns=data['feature_names'])
#print(dataset)

data = pd.read_csv('android-malware-detection-dataset.zip')
#print(data.head())
#print(data.info())

if (True):
    #Prepare data
    #drop_col = ['Label', 'Flow ID', 'Timestamp', 'CWE Flag Count', 'Down/UP Ratio', 'Fwd Avg Bytes/Bulk'] # Try without dropping, then replace with this
    drop_col = ['Unnamed: 0','Flow ID',' Timestamp',' CWE Flag Count',' Down/Up Ratio','Fwd Avg Bytes/Bulk']
    
    X = data.drop(drop_col, axis=1)
    y = data['Label']
    cols = X.select_dtypes(include=['object']).columns
    
    #Data sanitize
    le = LabelEncoder()
    for i in X.columns:
        if X[i].dtype == 'object':
            X[i] = le.fit_transform(X[i])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=0)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=.20, random_state=0)

    #Create decision tree
    clf = DecisionTreeClassifier(random_state=0) # Maybe add a max depth?
    clf = clf.fit(X_train, y_train)
    #print(clf.get_params())

    #Print predictions on validation
    predictions = clf.predict(X_validation)

    #Print results
    matrix = confusion_matrix(y_validation, predictions)
    print(matrix)

    print(classification_report(y_validation, predictions))
    
    #Print predictions on test
    test_pred = clf.predict(X_test)

    #Print results
    matrix = confusion_matrix(y_test, test_pred)
    print(matrix)

    print(classification_report(y_test, test_pred))
    
    plot_tree(clf)
    plt.show()