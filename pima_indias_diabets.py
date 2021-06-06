#%% Import Libraies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")

from collections import Counter

import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

#%% Read the Dataset

data = pd.read_csv("diabetes.csv")

#%% EDA (Expolarity Data Analysis)
data.columns
"""
['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
"""


data.info()

data.describe()

# In this dataset , outcome is a column which is target.I'm going to change the name of this column as a "Target"

data.rename({"Outcome":"Target"},axis = 1,inplace = True)


#%% To find missing values
data.columns[data.isnull().any()] # Which column has a missing value ?
data.isnull().sum()

# we don't need filling missing values operation because of fact that dataset has no any missing value.

#%% Categorical Variables = Pregnancies and Target

def bar_plot(variable):
    
    # get variable
    var = data[variable]
    
    # count number of the variables
    
    varValue = var.value_counts()
    
    # visualize
    
    plt.figure(figsize = (10,10))
    plt.bar(varValue.index,varValue)
    plt.xticks(varValue.index,varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    
categorical_variables = ["Pregnancies","Target"]   
for Q in categorical_variables:
    bar_plot(Q)

#%% Numerical Variables = Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age

def plot_hist(numerical_variable):
    plt.figure(figsize = (10,10))
    plt.hist(data[numerical_variable],bins = 150)
    plt.xlabel(numerical_variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist".format(numerical_variable))

numerical_variables = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
for T in numerical_variables:
    plot_hist(T)

#%% Outlier Detection

def detect_outliers(df,features):
    outlier_indices = []
    for c in features:
        # 1 st quartile
        Q1 = np.percentile(df[c],25)
        
        # 3 rd quartile
        Q3 = np.percentile(df[c],75)
        
        # IQR
        IQR = Q3 - Q1
        
        # Outlier step
        outlier_step = IQR * 1.5
   
        # detect outlier and their indeces
        outlier_list_col = df[(df[c] < Q1-outlier_step) | (df[c] > Q3 + outlier_step)].index
        
        # store indeces
        outlier_indices.extend(outlier_list_col)

    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)

    return multiple_outliers

data.loc[detect_outliers(data,['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'])]
# Dataset has no outlier values

#%% Correlation Matrix

f,ax = plt.subplots(figsize = (10,10))
sns.heatmap(data.corr(),annot = True,linewidths=.5,fmt = ".2f",ax = ax)
plt.title("Correlation Matrix")
plt.show()

#%% Get x and y coordinates

y = data.Target.values
x_data = data.drop(["Target"],axis = 1)

#%% Normalization Operation

x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data))

#%% Train-Test Split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 42)



#%% Random Forest Classification

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=10000,random_state = 1) # n_estimator , which means number of trees
rf.fit(x_train,y_train)
print("Accuracy of the Random Forest Classification model : % {}".format(rf.score(x_test,y_test)*100))

"""
Accuracy of the Random Forest Classification model : % 74.67532467532467
"""

#%% Naive Bayes Classification
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)

print("Accuracy of the Naive Bayes Classification model : % {}".format(nb.score(x_test,y_test)*100))


"""
Accuracy of the Naive Bayes Classification model : % 76.62337662337663
"""


#%% K-Nearst Neighbor Classification

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
print("Accuracy of the KNN model : % {}".format(knn.score(x_test,y_test)*100))

"""
Accuracy of the KNN model : % 69.48051948051948
"""
# Let's find best k value for the KNN

score_list = []

for each in range(1,100):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))

plt.plot(range(1,100),score_list)
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("K Value vs Accuracy")
plt.show()

# Graphs says 46 for best k value
from sklearn.neighbors import KNeighborsClassifier
knn3 = KNeighborsClassifier(n_neighbors=46)
knn3.fit(x_train,y_train)
print("Accuracy of the KNN model : % {}".format(knn3.score(x_test,y_test)*100))

"""
Accuracy of the KNN model : % 77.27272727272727
"""

#%% Support Vector Machines

from sklearn.svm import SVC

svm = SVC(random_state=1)
svm.fit(x_train,y_train)
print("Accuracy of the SVM model : % {}".format(svm.score(x_test,y_test)*100))

"""
Accuracy of the SVM model : % 74.67532467532467
"""

#%% K-Fold Cross Validation
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = knn, X = x_train,y = y_train,cv = 10)

print("average accuracy = ",np.mean(accuracies))
print("average std = ",np.std(accuracies))

"""
average accuracy =  0.73783712321523
average std =  0.042823746296891785
"""


#%% Confusion Matrix
y_pred = knn3.predict(x_test)
y_true = y_test

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true,y_pred)

# Confusion Matrix Visualize
f,ax = plt.subplots(figsize = (10,10))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax = ax)
plt.ylabel("y_pred")
plt.xlabel("y_true")
plt.title("Confusion Matrix")
plt.show()

#%% GridSearchCV
from sklearn.model_selection import GridSearchCV

grid = {"n_neighbors":np.arange(1,100)}
knn3 = KNeighborsClassifier()

knn_cv = GridSearchCV(knn3, grid, cv = 10) # GridSearchCV

knn_cv.fit(x,y)


#%% print hyperparameter KNN algoritması K değeri

print("tuned hyperparameter K: ",knn_cv.best_params_)
print("tuned parametreye göre en iyi accuracy (best score): ",knn_cv.best_score_)
