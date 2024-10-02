import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

#print(network_data.head())
#print(network_data.info())

choice = 'network'
X = 0
y = 0

match (choice):
    case 'android':
        # Import android dataset
        android_data = pd.read_csv('android-malware-detection-dataset.zip')
        
        #Prepare data
        drop_col = ['Unnamed: 0','Flow ID',' Timestamp',' CWE Flag Count',' Down/Up Ratio','Fwd Avg Bytes/Bulk']
        
        X = android_data.drop(drop_col, axis=1)
        y = android_data['Label']
        
        #Data sanitize
        le = LabelEncoder()
        for i in X.columns:
            if X[i].dtype == 'object':
                X[i] = le.fit_transform(X[i])
    
    case 'network':
        # Import and combine network dataset
        network_datasets = {
            'Friday-DDOS': pd.read_csv('network-intrustion-dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'),
            'Friday-PortScan': pd.read_csv('network-intrustion-dataset/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv'),
            'Friday-Morning': pd.read_csv('network-intrustion-dataset/Friday-WorkingHours-Morning.pcap_ISCX.csv'),
            'Monday': pd.read_csv('network-intrustion-dataset/Monday-WorkingHours.pcap_ISCX.csv'),
            'Thursday-Infilteration': pd.read_csv('network-intrustion-dataset/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv'),
            'Thursday-WebAttacks': pd.read_csv('network-intrustion-dataset/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv'),
            'Tuesday': pd.read_csv('network-intrustion-dataset/Tuesday-WorkingHours.pcap_ISCX.csv'),
            'Wednesday': pd.read_csv('network-intrustion-dataset/Wednesday-WorkingHours.pcap_ISCX.csv')
        }
        
        combine_df = pd.DataFrame()
        
        for key,set in network_datasets.items():
            set.insert(0, 'Day', key)
            combine_df = pd.concat([combine_df, set])
        
        combine_df.rename(columns={' Label': 'Label'}, inplace=True)
        print(combine_df.head())
        print(combine_df.tail())
        print(combine_df.info())
        
        #Data sanitize
        combine_df.fillna(0)
        combine_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        combine_df.dropna(inplace=True)
        
        #Prepare data
        drop_col = ['Label']
        
        X = combine_df.drop(drop_col, axis=1)
        y = combine_df['Label']
        
        #Data encoding
        le = LabelEncoder()
        for i in X.columns:
            if X[i].dtype == 'object':
                X[i] = le.fit_transform(X[i])

    case 'webpage':
        # Import webpage dataset
        webpage_data = pd.read_csv('web-page-detection-dataset.zip')
        
        #Prepare data
        #drop_col = ['Unnamed: 0','Flow ID',' Timestamp',' CWE Flag Count',' Down/Up Ratio','Fwd Avg Bytes/Bulk']
        
        X = android_data.drop(drop_col, axis=1)
        y = android_data['Label']
        
        #Data sanitize
        le = LabelEncoder()
        for i in X.columns:
            if X[i].dtype == 'object':
                X[i] = le.fit_transform(X[i])
    
    case _:
        sys.exit()
        
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=0)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=.20, random_state=0)

#Create decision tree
clf = DecisionTreeClassifier(random_state=0) # Maybe add a max depth?
clf = clf.fit(X_train, y_train)

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