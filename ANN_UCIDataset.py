#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/pkn9663/Cognixia_Assignment/blob/master/ANN_UCIDataset.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


# importing libraries
import pandas as pd
import numpy as np


# In[2]:


import os


# In[3]:


os.getcwd()


# In[4]:


files = [file for file in os.listdir('C:\\Users\\prave\\OneDrive\\Documents\\GitHub\\Cognixia_Assignment\\Datasets')]


# In[5]:


files


# In[7]:


df = pd.DataFrame()
for file in files:
  df1 = pd.read_excel('C:\\Users\\prave\\OneDrive\\Documents\\GitHub\\Cognixia_Assignment\\Datasets\\'+file)
  data = pd.concat([df , df1])
data.head()


# In[8]:


data.count()


# In[9]:


from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(data.iloc[: , 0:4].values , 
                                                       data.iloc[: , 4:].values , 
                                                       test_size = 0.25 , 
                                                       random_state = 0)
print(len(X_train) , len(X_test))


# In[22]:


"""
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()

X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)
"""


# In[10]:


(data.iloc[: , 0:4].values).shape


# # ANN Part

# In[11]:


import keras
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()


# In[12]:


model.add(Dense(units = 6 , activation= 'relu'))
model.add(Dense(units = 6 , activation= 'relu'))


# In[13]:


model.add(Dense(units = 1 , activation= None))


# In[16]:


model.compile(optimizer= 'adam' , loss = 'mse' , metrics= ['mse'])


# In[17]:


model.fit(X_train , y_train , batch_size = 10 , epochs= 100)


# In[18]:


y_pred = model.predict(X_test)


# In[19]:


y_pred


# In[20]:


print(np.concatenate((y_pred , y_test),1))


# In[21]:


model.summary()

