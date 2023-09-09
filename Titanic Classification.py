#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as mlt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")


# In[2]:


titanic=pd.read_csv("train.csv")
titanic.head()


# In[3]:


titanic.shape


# In[4]:


titanic.info()


# In[5]:


titanic.isnull().sum()


# In[6]:


titanic=titanic.drop(columns='Cabin',axis=1)


# In[7]:


titanic['Age'].fillna(titanic['Age'].mean(),inplace=True)


# In[8]:


titanic['Embarked'].mode()


# In[9]:


titanic['Embarked'].fillna(titanic['Embarked'].mode()[0],inplace=True)


# In[10]:


titanic.isnull().sum()


# In[11]:


#Data Analysis

titanic.describe()


# In[12]:


titanic['Survived'].value_counts()


# In[13]:


#Data Visualisation

sns.set()


# In[14]:


sns.countplot(x='Survived',data=titanic)


# In[15]:


sns.countplot(x='Sex',data=titanic)


# In[16]:


sns.countplot(x='Survived',hue='Sex',data=titanic)


# In[17]:


sns.countplot(x='Pclass',data=titanic)


# In[18]:


sns.countplot(x='Pclass',hue='Survived',data=titanic)


# In[19]:


# Encoding

titanic['Survived'].value_counts()


# In[20]:


titanic['Sex'].value_counts()


# In[21]:


titanic['Embarked'].value_counts()


# In[22]:


titanic.replace({'Sex':{'male':0,'female':1},'Embarked':{'S':0,'C':1,'Q':2}},inplace=True)
titanic.head()


# In[23]:


x=titanic.drop(['PassengerId','Name','Survived','Ticket'],axis=1)
y=titanic['Survived']


# In[24]:


x


# In[25]:


y


# In[26]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[27]:


model = LogisticRegression()
model.fit(x_train,y_train)


# In[28]:


x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(y_train,x_train_prediction)
print("Accuracy on the training data is:",training_data_accuracy)


# In[29]:


x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(y_test,x_test_prediction)
print("Accuracy on the testing data is:",test_data_accuracy)


# In[30]:


input_data= (1,0,46,0,1,7.28,1)
#change the input array into numpy array
input_as_numpy =np.asarray(input_data)


# In[31]:


input_reshaped =input_as_numpy.reshape(1,-1)
prediction =model.predict(input_reshaped)


# In[32]:


print(prediction)
if (prediction[0]==0):
    print("THE PERSON WON'T BE SAVED FROM SINKING.")
else:
    print("THE PERSON WILL BE SAVED FROM SINKING.")


# In[ ]:




