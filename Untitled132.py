#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 


# In[2]:


from sklearn.datasets import load_breast_cancer


# In[19]:


df=load_breast_cancer()
df


# In[21]:


cs=pd.DataFrame(df.data,columns=df.feature_names)
cs


# In[22]:


x=df.data 
y=df.target 


# In[30]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)


# In[31]:


x_train


# In[32]:


y_train


# In[33]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)


# In[34]:


y_pred=model.predict(x_test)


# In[35]:


from sklearn.metrics import accuracy_score,confusion_matrix
acc=accuracy_score(y_pred,y_test)
print("accuracy:",acc)
cf=confusion_matrix(y_pred,y_test)
print("confusion matrix:",cf)


# In[36]:


import matplotlib.pyplot as plt 
import seaborn as sns
cf=confusion_matrix(y_pred,y_test)
sns.heatmap(cf,annot=True)
plt.title("confusion matrix")
plt.xlabel("prediction")
plt.ylabel("target")
plt.show()


# In[ ]:




