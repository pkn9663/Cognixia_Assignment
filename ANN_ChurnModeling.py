#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/pkn9663/Cognixia_Assignment/blob/master/ANN_ChurnModeling.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# [<img src= "https://keras.io/img/logo.png" width = 200></img>](https://keras.io/)

# [**Churn Modelling**](https://keras.io/api/layers/core_layers/dense/)
# 
# 1. Create three layers (Input, Hidden and Output)
# 2. Use Rectifier (RELU) activation function in input and hidden layer
# 3. Use sigmoid activation function at the output layer
# 4. Find the accuracy of the model using confusion matrix
# 
# 

# In[ ]:


# Importing libraries
import pandas as pd
import numpy as np


# In[ ]:


dataset = pd.read_csv("/content/drive/My Drive/Colab Notebooks/Python Basics/Churn_Modelling.csv")


# In[5]:


dataset.info()


# In[ ]:


X = dataset.iloc[: , 3:-1].values
y = dataset.iloc[: , 13:].values


# In[ ]:


# handelling categorical data in independent variable using OneHotEncoder
from sklearn.preprocessing import LabelEncoder , OneHotEncoder , StandardScaler
from sklearn.compose import ColumnTransformer


# In[14]:


X


# In[ ]:


le = LabelEncoder()
ohe = OneHotEncoder()
scalar = StandardScaler()


# In[ ]:


X[: , 1] = le.fit_transform(X[: , 1])
X[: , 2] = le.fit_transform(X[: , 2])


# In[16]:


X


# In[ ]:


ct = ColumnTransformer(transformers=[("encoder" ,  ohe , [1])], remainder= "passthrough")


# In[ ]:


X = np.array(ct.fit_transform(X))


# In[ ]:


X = X[: , 1:]


# In[34]:


X.shape


# In[ ]:


# splitting the dataset into training & testing dataset
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X ,y ,test_size = 0.25 , random_state = 0)


# In[27]:


# feature scalling the training & testing set
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)
print(len(X_train) , len(X_test))


# # **Deep Learning**

# In[31]:


import keras


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense


# In[ ]:


# creating a model
classifier = Sequential()


# In[38]:


# Adding input layer & 1st Hidden Layor
classifier.add(Dense(round(11/2) , input_shape = (11 , ) , activation= 'relu'))
print(classifier.output_shape , classifier.input_shape)


# In[ ]:


# now adding 2nd Hidden layer
classifier.add(Dense(round(11/2) , activation= 'relu'))


# In[ ]:


# now adding output layer
classifier.add(Dense(1 , activation= 'sigmoid'))


# [Loss Reference](https://colab.research.google.com/drive/1v9CVrJq9EzVO6ouWtomnueK2t2rR4n6B#scrollTo=_4S4bEpxLHLe&line=2&uniqifier=1)

# In[ ]:


# compiling the ANN this makes the ANN is ready to use
classifier.compile(optimizer='adam' , loss = 'binary_crossentropy' , metrics= ['accuracy'])


# In[44]:


# fitting the training models
classifier.fit(X_train , y_train , batch_size= 10 , epochs= 100)


# In[ ]:


# predicting
y_pred = classifier.predict(X_test) * 100


# In[49]:


print(y_pred)


# In[ ]:


# this is helpful to convert the y_pred into Binary for visualising
y_pred = (y_pred > 50)


# In[51]:


# Visualising the Result
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test , y_pred)
cm


# In[56]:


import pandas.util.testing as tm
import seaborn as sns
sns.heatmap(cm , annot = True)

