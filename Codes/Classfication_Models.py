# importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

""" --------------- Logistic Regression -----------------"""

from sklearn.datasets import load_iris
iris = load_iris()

dir(iris)

data = pd.DataFrame(iris.data , columns= iris.feature_names)
data.head()

data["target"] = iris.target
data.head(10)

data["flower_names"] = data.target.apply(lambda x : iris.target_names[x])
data.head()

X = data.iloc[: , :4].values
y = data.iloc[: , 4].values

# splitting the datset into training & testing dataset
from sklearn.model_selection import train_test_split
X_train , X_test , y_train,y_test = train_test_split(X , y , test_size= 0.2 , random_state= 0)

# LogisticRegression model

from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression(random_state= 0 )
log_model.fit(X_train , y_train)

y_pred = log_model.predict(X_test)
log_model.score(X_test , y_test)

# confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test , y_pred)
cm

# Confusion_Matrix using heatmap
plt.figure(figsize=(5 , 5) , dpi = 100 , facecolor= "yellow")
plt.title("Iris LogisticRegression Model")
sns.heatmap(cm , annot=True)
plt.xlabel("Truth")
plt.ylabel("Predicted")
plt.show()

"""" ---------------- K-Nearest Neighbors (K-NN) ------------------------ """
from sklearn.datasets import load_iris
iris = load_iris()
dir(iris)

data = pd.DataFrame(iris.data , columns=iris.feature_names)
data.head(10)
data["target"] = iris.target
data["flower_names"] = data.target.apply(lambda x: iris.target_names[x])
data.head(10)

# splitting the dataset into training & testing set
# here i have choose "Sepal" Charecteristics features
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(data.iloc[: , :2].values ,
                                                       data.iloc[: , 4].values ,
                                                       test_size= 0.25 ,
                                                       random_state= 0)

print(len(X_train),len(X_test))

# creating KNN model

from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5 , p = 2 , metric = "minkowski")
knn_model.fit(X_train , y_train)

y_pred = knn_model.predict(X_test)

# Confusion Matrix
from sklearn.metrics import accuracy_score , confusion_matrix
cm = confusion_matrix(y_test , y_pred)
knn_score = accuracy_score(y_test, y_pred)
knn_score

# heatmap for ConfusionMatrix
plt.figure(figsize=(8 , 5) , dpi = 100 , facecolor= "orange")
plt.title("Iris K-NN for Sepal charecters" , fontdict= {"fontsize": 25 , "color": "Red"})
sns.heatmap(cm , annot= True )
plt.xlabel("Truth" , fontdict= {"fontsize": 15 , "color": "blue"})
plt.ylabel("Predicted" , fontdict= {"fontsize": 15 , "color": "blue"})
plt.show()


""" ----------- SVM Classifier ------------"""
# 1. iris dataset
from sklearn.svm import SVC
svm_model = SVC(C = 1 , kernel = "linear" , random_state= 0)
svm_model.fit(X_train , y_train)
svm_model.score(X_test , y_test)

# 2. digits dataset

from sklearn.datasets import load_digits
digits = load_digits()
dir(digits)

data = pd.DataFrame(digits.data)
for i in range(5):
    plt.matshow(digits.images[i])
    
# splitting DataSet into training & testing

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(digits.data , digits.target , test_size= 0.25)

print(len(X_train) , len(X_test))
print(X_train.shape)
print(y_train.shape)

# creating SVC model
from sklearn.svm import SVC
svc_model = SVC(C = 1 , kernel= "linear" , random_state= 0)

svm_model.fit(X_train , y_train)
y_pred = svm_model.predict(X_test)

svm_model.predict(plt.matshow(digits.images[67]))

# confusion Matrix & accuracy of model

from sklearn.metrics import confusion_matrix , accuracy_score
cm = confusion_matrix(y_test , y_pred)
cm
accuracy_score(y_test , y_pred) # 0.9844444444444445

# heatmap
plt.figure(figsize=(8 , 5) , dpi = 100 , facecolor= "orange")
plt.title("Digits SVM model" , fontdict= {"fontsize": 25 , "color": "Red"})
sns.heatmap(cm , annot= True )
plt.xlabel("Truth" , fontdict= {"fontsize": 15 , "color": "blue"})
plt.ylabel("Predicted" , fontdict= {"fontsize": 15 , "color": "blue"})
plt.show()

""" ------------ Decision Tree Classifier -----------------"""
# 1. iris dataset
from sklearn.datasets import load_iris
iris = load_iris()

dir(iris)

data = pd.DataFrame(iris.data , columns= iris.feature_names)
data.head(10)

data["target"] = iris.target
data["flower_names"] = data.target.apply(lambda  x : iris.target_names[x])
data.head(10)

# splitting Datasets into Training /testing data
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(data.iloc[: , :4].values ,
                                                       data.iloc[: , 4].values ,
                                                       test_size= 0.2 , random_state= 0)

# DTC model
from sklearn.tree import DecisionTreeClassifier, plot_tree
dtc_model = DecisionTreeClassifier(criterion= "gini" , random_state= 0)
dtc_model.fit(X_train , y_train)
y_pred = dtc_model.predict(X_test)
# analysing the tree
plt.figure(dpi = 100,facecolor="orange")
iris_tree = plot_tree(dtc_model)

# confusion matrix
from sklearn.metrics import confusion_matrix , accuracy_score
cm = confusion_matrix(y_test , y_pred)
dtc_score = accuracy_score(y_test , y_pred)

# heatmap
plt.figure(figsize=(8 , 5) , dpi = 100 , facecolor= "orange")
plt.title("Iris DecisionTreeClassifier Model" , fontdict= {"fontsize": 20 , "color": "Red"})
sns.heatmap(cm , annot= True )
plt.xlabel("Truth" , fontdict= {"fontsize": 15 , "color": "blue"})
plt.ylabel("Predicted" , fontdict= {"fontsize": 15 , "color": "blue"})
plt.show()

# 2. digits

from sklearn.datasets import load_digits
digits = load_digits()

dir(digits)

# just to see what it looks like
for i in range(5):
    print(plt.matshow(digits.images[i]))
    
data.head()

# train/test dataset

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(digits.data ,
                                                       digits.target ,
                                                       test_size= 0.2 , random_state= 0 )
# DTC model
from sklearn.tree import DecisionTreeClassifier
digits_model = DecisionTreeClassifier(criterion= "entropy" , random_state= 0)
digits_model.fit(X_train , y_train)


# confusion matrix
from sklearn.metrics import accuracy_score , confusion_matrix
cm = confusion_matrix(y_test , digits_model.predict(X_test))
accuracy_score(y_test , digits_model.predict(X_test))

# heatmap
plt.figure(figsize=(8 , 5) , dpi = 100 , facecolor= "orange")
plt.title("Digits DecisionTreeClassifier Model" , fontdict= {"fontsize": 20 , "color": "Red"})
sns.heatmap(cm , annot= True )
plt.xlabel("Truth" , fontdict= {"fontsize": 15 , "color": "blue"})
plt.ylabel("Predicted" , fontdict= {"fontsize": 15 , "color": "blue"})
plt.show()

""" ------------------ RandomForestClassifier --------------------------"""
# 1. iris dataset

from sklearn.datasets import load_iris
iris = load_iris()

dir(iris)

data = pd.DataFrame(iris.data , columns= iris.feature_names)
data.head(10)

data["target"] = iris.target
data["flower_names"] = data.target.apply(lambda  x : iris.target_names[x])
data.head(10)

# splitting Datasets into Training /testing data
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(data.iloc[: , :4].values ,
                                                       data.iloc[: , 4].values ,
                                                       test_size= 0.3 , random_state= 0)

# RF model
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators= 5 , random_state= 0)

rf_model.fit(X_train , y_train)
y_pred = rf_model.predict(X_test)

rf_model.predict_proba(X_test)

# consfusion matrix
from sklearn.metrics import accuracy_score , confusion_matrix
cm = confusion_matrix(y_test , y_pred)
accuracy_score(y_test , y_pred) # 0.9555555555555556

# heatmap
plt.figure(figsize=(8 , 5) , dpi = 100 , facecolor= "orange")
plt.title("iris RandomForestClassifier Model" , fontdict= {"fontsize": 20 , "color": "Red"})
sns.heatmap(cm , annot= True )
plt.xlabel("Truth" , fontdict= {"fontsize": 15 , "color": "blue"})
plt.ylabel("Predicted" , fontdict= {"fontsize": 15 , "color": "blue"})
plt.show()

# 2. digits

from sklearn.datasets import load_digits

digits = load_digits()

dir(digits)

# just to see what it looks like
for i in range(5):
    print(plt.matshow(digits.images[i]))

# train/test dataset

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(digits.data,
                                                    digits.target,
                                                    test_size=0.3, random_state=0)
# model
from sklearn.ensemble import RandomForestClassifier
digits_model = RandomForestClassifier(n_estimators=5 , random_state= 0)
digits_model.fit(X_train , y_train)

# Confusion matrix
from sklearn.metrics import accuracy_score , confusion_matrix
cm = confusion_matrix(y_test , digits_model.predict(X_test))
accuracy_score(y_test , digits_model.predict(X_test)) # 0.8962962962962963

# heatmap
plt.figure(figsize=(8 , 5) , dpi = 100 , facecolor= "orange")
plt.title("Digits RandomForestClassifier Model" , fontdict= {"fontsize": 20 , "color": "Red"})
sns.heatmap(cm , annot= True )
plt.xlabel("Truth" , fontdict= {"fontsize": 15 , "color": "blue"})
plt.ylabel("Predicted" , fontdict= {"fontsize": 15 , "color": "blue"})
plt.show()