#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


#Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# In[3]:


#Algorithms
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor


# In[4]:


#Tuning
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


# In[5]:


#Metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# In[6]:


df = pd.read_csv('Properties_links_olx.csv')


# In[7]:


df.info()


# In[8]:


df['Area'] = df['Area'].replace('Unknown', np.nan)
df['Bedrooms'] = df['Bedrooms'].replace('Unknown', np.nan)
df['Bathrooms'] = df['Bathrooms'].replace('Unknown', np.nan)
df['Price'] = df['Price'].replace('Unknown', np.nan)
df.dropna(inplace=True)


# In[10]:


df['Bedrooms'] = df['Bedrooms'].replace('10+',11)
df['Bathrooms'] = df['Bathrooms'].replace('10+',11)
df['Furnished'] = df['Furnished'].replace('Unknown','No')
df['Payment_Option'] = df['Payment_Option'].replace('Unknown','Cash')
df['Delivery_Term'] = df['Delivery_Term'].replace('Unknown ', 'Not Finished')


# In[11]:


df['Bedrooms'] = df['Bedrooms'].astype(int)
df['Bathrooms'] = df['Bathrooms'].astype(int)
df['Area'] = df['Area'].astype(float).astype(int)
df['Price'] = df['Price'].astype(int)


# In[12]:


print(df['Level'].unique())
print(df['Type'].unique())
print(df['Payment_Option'].unique())


# In[13]:


df=df.drop(df[(df['Level']=='Unknown')&(df['Type']=='Duplex')].index)
df=df.drop(df[(df['Level']=='Unknown')&(df['Type']=='Apartment')].index)
df=df.drop(df[(df['Level']=='Unknown')&(df['Type']=='Studio')].index)


# In[14]:


df.loc[(df['Level']=='10+'),'Level'] = 11
df.loc[(df['Level']=='Highest'),'Level'] = 12
df.loc[(df['Level']=='Ground'),'Level'] = 0

df.loc[(df['Type']=='Penthouse'),'Level'] = 12


# In[15]:


df['Level'] = df['Level'].astype(int)


# In[16]:


print(df['Level'].unique())


# In[17]:


df.reset_index(inplace=True)
df.drop(['index'],axis=1,inplace=True)


# In[18]:


df['Type'].value_counts()


# In[19]:


city_name = df['City'].value_counts(dropna=False).keys().tolist()
val = df['City'].value_counts(dropna=False).tolist()
value_dict = list(zip(city_name, val))


# In[20]:


Low_frequency_city = []
y = 'Less'
for city_name,val in value_dict:
    if val <= 5:
        Low_frequency_city.append(city_name)
    else :
        pass
def lcdlt(x):
    if x in Low_frequency_city:
        return y
    else :
        return x
df['City'] = df['City'].apply(lcdlt)
df=df.drop(df[(df['City']=='Less')].index)


# In[21]:


lcc = df['City'].unique()
for x in lcc:
    Q1= df[(df['City']==x)]['Price'].quantile(0.25)
    Q3= df[(df['City']==x)]['Price'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.2 * IQR
    lower_bound = Q1 - 1.2 * IQR
    df=df.drop(df[(df['City']==x)&(df['Price']>=upper_bound)].index)
    df=df.drop(df[(df['City']==x)&(df['Price']<=lower_bound)].index)


# In[22]:


df['City'].unique()


# In[23]:


df=df.drop(df[(df['Area']<=100)&(df['Bedrooms']>=4)].index)
df=df.drop(df[(df['Area']<=30)&(df['Type']!='Studio')].index)


# In[24]:


df.reset_index(inplace=True)
df.drop(['index'],axis=1,inplace=True)


# In[25]:


for col in df.columns:
    print(col,':',df[col].nunique())
    print(df[col].value_counts().nlargest(7))
    print('\n'+'*'*20+'\n')


# In[26]:


df = pd.get_dummies(df, columns = ['Type','Delivery_Term','Furnished','City' ,'Payment_Option'])
X = df.drop(columns = ['Price'])
y = df[['Price']]


# In[27]:


from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test = train_test_split(X,y,test_size = 0.20,shuffle = True ,random_state = 404)

X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[28]:


def performance(model,X_train,y_train,y_pred,y_test):
    print('Training Score:',model.score(X_train,y_train))
    print('Testing Score:',r2_score(y_test,y_pred))
    print('Other Metrics In Testing Data: ')
    print('MSE:',mean_squared_error(y_test,y_pred))
    print('MAE:',mean_absolute_error(y_test,y_pred))


# In[29]:


lr = LinearRegression()
lr.fit(X_train,y_train)

lr_pred = lr.predict(X_test)

performance(lr,X_train,y_train,lr_pred,y_test)


# In[30]:


ridge = Ridge(alpha = 1)
ridge.fit(X_train,y_train)

#The predicted data
ridge_pred = ridge.predict(X_test)


#The performance
performance(ridge,X_train,y_train,ridge_pred,y_test)


# In[31]:


dt = DecisionTreeRegressor()
dt.fit(X_train,y_train)

#The predicted data
dt_pred = dt.predict(X_test)


#The performance
performance(dt,X_train,y_train,dt_pred,y_test)


# In[32]:


rf = RandomForestRegressor()
rf.fit(X_train,y_train.values.ravel())

#The predicted data
rf_pred = rf.predict(X_test)

#The performance
performance(rf,X_train,y_train,rf_pred,y_test)


# In[33]:


xgb = XGBRegressor()
xgb.fit(X_train,y_train)

#The predicted data
xgb_pred = xgb.predict(X_test)

#The performance
performance(xgb,X_train,y_train,xgb_pred,y_test)


# In[34]:


params = [{'max_depth':list(range(5,20)),'min_samples_split':list(range(2,15)),"min_samples_leaf":[2,3,4]}]

grid_search = GridSearchCV(estimator=DecisionTreeRegressor(),param_grid=params,cv=10,n_jobs=-1)

grid_search.fit(X_train,y_train)

print('Best Estimator:',grid_search.best_estimator_)
print('Best Params:',grid_search.best_params_)

grid_pred = grid_search.predict(X_test)

performance(grid_search,X_train,y_train,grid_pred,y_test)


# In[35]:


params = [{'n_estimators':[100,200,3000,400,500,600],
           'max_depth':list(range(5,20)),'min_samples_split':list(range(2,15))
           ,"min_samples_leaf":[2,3,4,5]}]

rand_search = RandomizedSearchCV(RandomForestRegressor(),params,cv=10,n_jobs=-1)
rand_search.fit(X_train,y_train.values.ravel())

print('Best Estimator:',rand_search.best_estimator_)
print('Best Params:',rand_search.best_params_)

rand_pred = rand_search.predict(X_test)

performance(rand_search,X_train,y_train,rand_pred,y_test)


# In[36]:


params = {'max_depth': list(range(5,15)),'n_estimators': [300,400,500,600,700]
          ,'learning_rate': [0.01,0.1,0.2,0.9]}

rand_search = RandomizedSearchCV(XGBRegressor(),params,cv=10,n_jobs=-1)
rand_search.fit(X_train,y_train)

print('Best Estimator:',rand_search.best_estimator_)
print('Best Params:',rand_search.best_params_)

rand_pred = rand_search.predict(X_test)

performance(rand_search,X_train,y_train,rand_pred,y_test)

