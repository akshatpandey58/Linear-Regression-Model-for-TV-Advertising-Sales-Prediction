#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[2]:


df_tv=pd.read_csv(r"C:\Users\Acer\OneDrive\Documents\tvmarketing.csv")


# In[3]:


print("Number of records and features:", df_tv.shape)


# In[4]:


print("Feature names:", df_tv.columns)


# In[5]:


df_tv.info()


# In[6]:


print("\nNumerical description of the dataframe:")
print(df_tv.describe())


# In[7]:


print("\nMissing values in the dataframe:")
print(df_tv.isnull().sum())


# In[9]:


plt.figure(figsize=(10, 5))
sb.scatterplot(data=df_tv, x='TV', y='Sales')
plt.title('Scatter plot between TV advertising budget and Sales')
plt.xlabel('TV Advertising Budget ($)')
plt.ylabel('Sales')
plt.show()


# In[10]:


X = df_tv[['TV']]  
y = df_tv['Sales'] 


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[12]:


model = LinearRegression()


# In[13]:


model.fit(X_train, y_train)


# In[14]:


y_pred = model.predict(X_test)


# In[15]:


comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nComparison of actual and predicted values:")
print(comparison_df.head())


# In[16]:


plt.figure(figsize=(10, 5))
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.plot(X_train, model.predict(X_train), color='red', label='Fitted line')
plt.title('Training Data and Fitted Line')
plt.xlabel('TV Advertising Budget ($)')
plt.ylabel('Sales')
plt.legend()
plt.show()


# In[17]:


mse = mean_squared_error(y_test, y_pred)
print("\nMean Squared Error:", mse)


# In[18]:


plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, color='purple')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.title('Actual vs Predicted Sales')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.show()


# In[20]:


new_record = np.array([[150]])
predicted_sales = model.predict(new_record)
print("\nPredicted sales for a TV advertising budget of $150:", predicted_sales[0])


# In[ ]:




