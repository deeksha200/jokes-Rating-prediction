#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from mlxtend.regressor import StackingRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
pd.pandas.set_option("display.max_columns",None)
print("all necessary libraries are imported")


# In[2]:


jokes=pd.read_csv('C:\\Users\\Deeksha Rai\\Desktop\\projects\\train_MaefO4x\\jokes.csv')
train=pd.read_csv('C:\\Users\\Deeksha Rai\\Desktop\\projects\\train_MaefO4x\\train.csv')
test=pd.read_csv('C:\\Users\\Deeksha Rai\\Desktop\\projects\\test_MElQnvy\\test.csv')


# In[3]:


jokes.head()


# In[4]:


train.head()


# In[5]:


test.head()


# In[6]:


jokes.isnull().any()


# In[7]:


train.isnull().any()


# In[8]:


test.isnull().any()


# In[9]:


jokes.shape,train.shape,test.shape


# In[10]:


train['user_id'].nunique()


# In[11]:


test['user_id'].nunique()


# In[12]:


train.Rating.plot(kind='hist')


# In[13]:


y=train['Rating']


# In[14]:


y


# In[15]:


train.drop(['Rating'],axis=1,inplace=True)


# In[16]:


train.head()


# In[83]:


model=RandomForestRegressor(n_estimators=100,min_samples_leaf=100,oob_score = True,random_state=0,max_features='auto',n_jobs=-1,
                           min_samples_split=2)


# In[84]:


model


# In[85]:


x_train,x_test,y_train,y_test=train_test_split(train,y,test_size=0.5,random_state=0)


# In[86]:


model.fit(x_train,y_train)


# In[87]:


pred=model.predict(x_test)


# In[88]:


mean_absolute_error(pred,y_test)


# In[89]:


pred


# In[90]:


pred1=model.predict(test)


# In[91]:


submission=pd.read_csv("C:\\Users\\Deeksha Rai\\Desktop\\projects\\sample_submission_5ms57N3.csv")


# In[92]:


submission


# In[93]:


submission['id']=test['id']


# In[94]:


test['id'].shape


# In[95]:


pred1.shape


# In[96]:


submission['Rating']=pred1


# In[97]:


pd.DataFrame(submission,columns=['id','Rating']).to_csv('jokes.csv',index=False)


# In[98]:


pd.read_csv("jokes.csv")

