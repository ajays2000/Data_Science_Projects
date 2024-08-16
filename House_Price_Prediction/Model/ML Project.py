#!/usr/bin/env python
# coding: utf-8

# # Real Estate Price Prediction Project

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


#To read the file
df1 =  pd.read_csv('Bengaluru_House_Data.csv')
df1.head()


# #### Data Cleaning

# In[4]:


df1.groupby('area_type')['area_type'].agg('count')


# In[5]:


#To extract the important column datas
df2 = df1.drop(['area_type', 'society', 'balcony', 'availability'], axis = 1)
df2.head()


# In[6]:


#To check the NaN values

df2.isna().sum()


# In[7]:


df3 = df2.dropna()
df3.isna().sum()


# In[8]:


#To create a new column for the room size

df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))


# In[9]:


df3.head()


# In[10]:


df3['bhk'].unique()


# In[11]:


df3[df3['bhk'] > 20]   #found an error as the sqft and bedroom numbers are not matching


# In[12]:


df3['total_sqft'].unique()


# In[13]:


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[14]:


df3[~df3['total_sqft'].apply(is_float)].head(10)


# In[15]:


def convert_sqft_to_num(x):
    tokens = x.split('-')
    
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2
    try:
        return float(x)
    except:
        return None


# In[16]:


df4 = df3.copy()


# In[17]:


df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)
df4.head()


# #### Feature Engineering and Dimensionality Techniques

# In[18]:


df5 = df4.copy()


# In[19]:


df5['price_per_sqft'] = df5['price'] * 100000 /  df5['total_sqft']
df5.head()


# In[20]:


len(df5['location'].unique())


# In[21]:


df5['location'] = df5['location'].apply(lambda x: x.strip()) #Remove any spaces at the end of the location name


# In[22]:


location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending = False)


# In[23]:


location_stats


# In[24]:


location_stats_less_than_10 = location_stats[location_stats <= 10]


# In[25]:


location_stats_less_than_10


# In[26]:


df5['location'] = df5['location'].apply(lambda x : 'Other' if x in location_stats_less_than_10 else x)


# In[27]:


df5


# #### Outlier Removal

# In[28]:


df6 = df5[~(df5['total_sqft'] / df5['bhk'] < 300)] 


# In[29]:


df6


# In[30]:


df6['price_per_sqft'].describe()


# In[31]:


def remove_pps_outliers(df):
    
    df_out = pd.DataFrame()
    
    for key, subdf in df.groupby('location'):
        
        m = np.mean(subdf['price_per_sqft'])
        st = np.std(subdf['price_per_sqft'])
        reduced_df = subdf[(subdf['price_per_sqft'] > (m - st)) & (subdf['price_per_sqft'] <= (m + st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index = True)
    
    return df_out

df7 = remove_pps_outliers(df6)
df7.shape


# In[32]:


def plot_scatter_chart(df, location):
    
    bhk2 = df[(df['location'] == location) & (df['bhk'] == 2)]
    bhk3 = df[(df['location'] == location) & (df['bhk'] == 3)]
    
    plt.scatter(bhk2['total_sqft'], bhk2['price'], color = 'blue', label = '2 BHK', s = 50)
    plt.scatter(bhk3['total_sqft'], bhk3['price'], marker = '+', color = 'red', label = '3 BHK', s = 50)
     
    plt.xlabel('Total Sqft Area')
    plt.ylabel('Price')
    plt.title(location)
    plt.legend()
    
    plt.show()


# In[33]:


plot_scatter_chart(df7, 'Hebbal')


# #### To remove thos 2 BHK apartments whose price per sqft is less than mean price per sqft of 1 BHK apartments

# In[34]:


def remove_bhk_outliers(df):
    
    exclude_indices = np.array([])
    
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                              'mean' : np.mean(bhk_df['price_per_sqft']),
                              'std' : np.std(bhk_df['price_per_sqft']),
                              'count' : bhk_df.shape[0]
                               }
            
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk - 1)
            
            if stats and stats['count'] > 5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df['price_per_sqft'] < (stats['mean'])].index.values)
    return df.drop(exclude_indices, axis = 'index')

df8 = remove_bhk_outliers(df7)
df8


# In[35]:


plot_scatter_chart(df8, 'Hebbal')


# In[36]:


#To plot the histogram for the price per sqft

plt.hist(df8['price_per_sqft'], rwidth = 0.8)
plt.xlabel('Price Per Sqft')
plt.ylabel('Count')
plt.show()


# In[37]:


df8['bath'].unique()


# In[38]:


df8[df8['bath'] > 10]


# In[39]:


plt.hist(df8['bath'], rwidth = 0.8)
plt.xlabel('Number of Bathrooms')
plt.ylabel('Count')
plt.show()


# In[40]:


df9  = df8[df8['bath'] < df8['bhk'] + 2]  #Condition
df9.shape


# In[41]:


#To make the data with only the required columns
df10 = df9.drop(['size', 'price_per_sqft'], axis = 1)


# In[42]:


df10


# #### To build the model

# In[43]:


dummies = pd.get_dummies(df10['location'])


# In[44]:


df11 = pd.concat([df10, dummies], axis =1)


# In[45]:


df11


# In[46]:


df12 = df11.drop(['location'], axis = 1)


# In[47]:


df12


# In[48]:


x = df12.drop(['price'], axis =1)


# In[49]:


y = df12['price']


# In[50]:


from sklearn.model_selection import train_test_split


# In[51]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 10)


# In[52]:


from sklearn.linear_model import LinearRegression


# In[53]:


model = LinearRegression()

model.fit(x_train, y_train)
model.predict(x_test)
model.score(x_test, y_test)


# In[54]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits = 5, test_size = 0.2, random_state = 0)
cross_val_score(LinearRegression(), x, y, cv = cv)


# #### Hyper Parameter Tuning

# In[55]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

import warnings
warnings.filterwarnings('ignore')

def find_best_model_gridsearchgcv(x, y):
    
    algos = {
             'linear_regreesion': {
                                   'model' : LinearRegression(),
                                   'params': {'fit_intercept': [True, False],
                                              'copy_X': [True, False]}
                                   },
             'lasso' : {
                        'model' : Lasso(),
                        'params': {'alpha' : [1,2],
                                   'selection': ['random', 'cyclic']}
                        },
             'decisio_tree' : {
                              'model' : DecisionTreeRegressor(),
                              'params':{'criterion' : ['mse', 'friedman_mse'],
                                        'splitter' : ['best', 'random']}
                               }
            }
    scores = []
    cv = ShuffleSplit(n_splits = 5, test_size = 0.2, random_state = 0)
    for algo_name, config in algos.items():
        
        gs = GridSearchCV(config['model'], config['params'], cv = cv, return_train_score = False)
        gs.fit(x,y)
        scores.append({
                       'model' : algo_name,
                       'best_score' : gs.best_score_,
                       'best_params' : gs.best_params_
                      })
    return pd.DataFrame(scores, columns = ['model', 'best_score', 'best_params'])


# In[56]:


find_best_model_gridsearchgcv(x, y)


# In[57]:


#To predict the price
x = df12.drop(['price'], axis =1)
def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(x.columns == location)[0][0]
    
    x1 = np.zeros(len(x.columns))
    x1[0] = sqft
    x1[1] = bath
    x1[2] = bhk
    
    if loc_index >= 0:
        x1[loc_index] =  1
        
    return model.predict([x1])[0]


# In[58]:


predict_price('1st Phase JP Nagar', 1000, 2, 2)


# In[59]:


predict_price('Indira Nagar', 1000, 2, 2)


# In[60]:


#To save it as pickle file

import pickle
with open('bangalore_home_prices_model.pickle', 'wb') as f:
    pickle.dump(model, f)


# In[61]:


# To save it as a json file

import json

columns = {
           'data_columns' : [col.lower() for col in x.columns]
           }
with open('columns.json', 'w') as f:
    f.write(json.dumps(columns))


# In[ ]:





# In[ ]:





# In[ ]:




