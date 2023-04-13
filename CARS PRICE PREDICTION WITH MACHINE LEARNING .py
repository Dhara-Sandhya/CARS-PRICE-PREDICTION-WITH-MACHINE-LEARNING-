#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Pandas
import pandas as pd
# Numpy
import numpy as np
# Libraries for plotting
import matplotlib.pyplot as plt
import seaborn as sns
# Train-test split
from sklearn.model_selection import train_test_split
# Min-max scling
from sklearn.preprocessing import MinMaxScaler
# Statsmodel 
import statsmodels.api as sm
# VIF 
from statsmodels.stats.outliers_influence import variance_inflation_factor
#R-squared
from sklearn.metrics import r2_score
# Label encoding
from sklearn.preprocessing import LabelEncoder
# Importing RFE
from sklearn.feature_selection import RFE
# Importing LinearRegression
from sklearn.linear_model import LinearRegression
# Supress warning
import warnings
warnings.filterwarnings('ignore')
# Libraries for cross validation 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor

from sklearn import datasets
from sklearn.model_selection import cross_val_score, cross_val_predict
pd.set_option('display.max_columns',None)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df=pd.read_csv('https://raw.githubusercontent.com/amankharwal/Website-data/master/CarPrice.csv')
df


# In[4]:


df.info()


# In[5]:


df.describe()


# ## Spliting the "Car company" from "CarName"
# 

# In[7]:


car_company = df["CarName"].str.split(" ", n = 1, expand = True)
df['CarCompany'] = car_company[0]

# Dropping 'CarName' column
df.drop('CarName',axis=1,inplace=True)
df.head()


# In[8]:


# Dropping car_ID column as it will not be used in our analysis
df.drop('car_ID',axis=1,inplace=True)


# In[9]:


#Replacing '4wd' with 'fwd' in 'drivewheel' column
df['drivewheel'] = df['drivewheel'].replace('4wd','fwd')
# Replacing 'maxda' with 'mazda' in 'CarCompany' column
df['CarCompany'] = df['CarCompany'].replace('maxda','mazda')
# Replacing 'porcshce' with 'porsche' in 'CarCompany' column
df['CarCompany'] = df['CarCompany'].replace('porcshce','porsche')
# Replacing 'toyouta' with 'toyota' in 'CarCompany' column
df['CarCompany'] = df['CarCompany'].replace('toyouta','toyota')
# Replacing 'vokswagen' with 'volkswagen' in 'CarCompany' column
df['CarCompany'] = df['CarCompany'].replace('vokswagen','volkswagen')
# Replacing 'Nisaan' with 'nissan' in 'CarCompany' column
df['CarCompany'] = df['CarCompany'].replace('Nissan','nissan')
# Replacing 'vw' with 'volkswagen' in 'CarCompany' column
df['CarCompany'] = df['CarCompany'].replace('vw','volkswagen')


# In[10]:


# Finding outliers in all the numerical columns with 1.5 IQR rule and removing the outlier records 
col_numeric = ['wheelbase','carlength','carwidth','carheight','curbweight',
                    'enginesize','boreratio','stroke','compressionratio','horsepower','peakrpm','citympg','highwaympg','price']

for col in col_numeric: 
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3-q1
    range_low  = q1-1.5*iqr
    range_high = q3+1.5*iqr
    df= df.loc[(df[col] > range_low) & (df[col] < range_high)]

df.shape


# #### We can see that there are (205-123)=82 records, which are outliers in the dataset.
# 
# 

# ### Check data imbalence
# 

# In[12]:


# Listing categorical columns for checking data imbalance and plotting them
col_category = ['symboling','fueltype','aspiration','doornumber','carbody','drivewheel','enginelocation','enginetype',
           'cylindernumber','fuelsystem','CarCompany']

k=0
plt.figure(figsize=(20,25))
for col in col_category:    
    k=k+1
    plt.subplot(4, 3,k)    
    df[col].value_counts().plot(kind='bar');
    plt.title(col)


# In[13]:


# Visualising the numerical variables
plt.figure(figsize=(12,12))
sns.pairplot(df[col_numeric])
plt.show()


# In[14]:


# Boxplot for all categorical variables except CarCompany
# As X labels are not clearly visible for CarCompany. It is plotted in the next cell with bigger figure size.
k=0
plt.figure(figsize=(20,18))
for col in range (len(col_category)-1):    
    k=k+1
    plt.subplot(4, 3, k)   
    ax = sns.boxplot(x = col_category[col], y = 'price', data = df)


# ### Converting categorical variables to two levels to binary variables
# 

# In[15]:


# fueltype
# Convert "gas" to 1 and "diesel" to 0
df['fueltype'] = df['fueltype'].map({'gas': 1, 'diesel': 0})
df.head()


# In[16]:


# aspiration
# Convert "std" to 1 and "turbo" to 0
df['aspiration'] = df['aspiration'].map({'std':1, 'turbo':0})
df.head()


# In[17]:


# doornumber
# Convert "four" to 1 and "two" to 0
df['doornumber'] = df['doornumber'].map({'four':1, 'two':0})
df.head()


# In[18]:


# enginelocation
# Convert "front" to 1 and "rear" to 0
df['enginelocation'] = df['enginelocation'].map({'front':1, 'rear':0})
df.head()


# In[19]:


# Creating dummy variables for 'symboling'
# Dropping the redundant dummy variable (-2)
symboling_status = pd.get_dummies(df['symboling'],drop_first=True)
symboling_status.head()


# In[20]:


# Renaming column names for better readability
symboling_status = symboling_status.rename(columns={-1:'symboling(-1)', 0:'symboling(0)', 1:'symboling(1)',2:'symboling(2)', 3:'symboling(3)'})
symboling_status.head()


# In[21]:


# Renaming column names for better readability
symboling_status = symboling_status.rename(columns={-1:'symboling(-1)', 0:'symboling(0)', 1:'symboling(1)',2:'symboling(2)', 3:'symboling(3)'})
symboling_status.head()


# In[22]:


# Dropping the 'symboling' column as we don't need it anymore
df= df.drop('symboling',axis=1)
df.head()


# In[24]:


# Creating dummy variables for 'carbody'
# Dropping the redundant dummy variable (convertible)
carbody_status = pd.get_dummies(df['carbody'],drop_first=True)
carbody_status.head()


# In[25]:


# Renaming column names for better readability
carbody_status = carbody_status.rename(columns={'hardtop':'carbody(hardtop)', 'hatchback':'carbody(hatchback)', 'sedan':'carbody(sedan)','wagon':'carbody(wagon)'})
carbody_status.head()


# In[26]:


# Concating the dummy dataframe with original dataframe
df= pd.concat([df,carbody_status], axis=1)
df.head()


# In[27]:


# Dropping the 'cylindernumber' column as we don't need it
df = df.drop('cylindernumber',axis=1)
df.head()


# In[28]:


# Creating dummy variables for 'fuelsystem'
# Dropping the redundant dummy variable (1bbl)
fuelsystem_status = pd.get_dummies(df['fuelsystem'], drop_first=True)
fuelsystem_status.head()


# In[29]:


# Renaming column name for better readability
fuelsystem_status = fuelsystem_status.rename(columns={'2bbl':'fuelsystem(2bbl)', '4bbl':'fuelsystem(4bbl)', 'idi':'fuelsystem(idi)', 
                                                      'mfi':'fuelsystem(mfi)','mpfi':'fuelsystem(mpfi)' ,'spdi':'fuelsystem(spdi)',
                                                             'spfi':'fuelsystem(spfi)'})
fuelsystem_status.head()


# In[30]:


# Concating the dummy dataframe with original dataframe
df = pd.concat([df,fuelsystem_status], axis=1)
df.head()


# In[31]:


# Dropping the 'fuelsystem' column as we don't need it
df = df.drop('fuelsystem',axis=1)
df.head()


# In[32]:


# Creating dummy variables for 'CarCompany'
# Dropping the redundant dummy variable (alfa-romero)
CarCompany_status = pd.get_dummies(df['CarCompany'], drop_first=True)
CarCompany_status.head()


# In[33]:


# Renaming column name for better readability
CarCompany_status = CarCompany_status.rename(columns={'audi':'CarCompany(audi)', 'bmw':'CarCompany(bmw)', 'buick':'CarCompany(buick)', 
                                                      'chevrolet':'CarCompany(chevrolet)','dodge':'CarCompany(dodge)' ,'honda':'CarCompany(honda)',
                                                      'isuzu':'CarCompany(isuzu)','jaguar':'CarCompany(jaguar)','mazda':'CarCompany(mazda)',
                                                      'mercury':'CarCompany(mercury)','mitsubishi':'CarCompany(mitsubishi)','nissan':'CarCompany(nissan)',
                                                      'peugeot':'CarCompany(peugeot)','plymouth':'CarCompany(plymouth)','porsche':'CarCompany(porsche)',
                                                      'renault':'CarCompany(renault)','saab':'CarCompany(saab)','subaru':'CarCompany(subaru)',
                                                      'toyota':'CarCompany(toyota)','volkswagen':'CarCompany(volkswagen)','volvo':'CarCompany(volvo)'})
                                                    
CarCompany_status.head()


# In[34]:


# Concating the dummy dataframe with original dataframe
df= pd.concat([df,CarCompany_status], axis=1)
df.head()


# In[35]:


# Dropping the 'CarCompany' column as we don't need it
df= df.drop('CarCompany',axis=1)
df.head()


# In[36]:


df.info()


# In[37]:


# Splitting train and test dataset into 70:30 percent ratio.
df_train, df_test = train_test_split(df, train_size=0.7, random_state=100)
print(df_train.shape)
print(df_test.shape)


# In[38]:


# Create a list of numeric variables. We don't need categorical variables because they are already scalled in 0 and 1.
num_vars = ['wheelbase','carlength','carwidth','carheight','curbweight','enginesize','boreratio','stroke',
            'compressionratio','horsepower','peakrpm','citympg','highwaympg','price']

# Instantiate an object
scaler = MinMaxScaler()

# Fit the data in the object
df_train[num_vars] = scaler.fit_transform(df_train[num_vars])
df_train.head()


# In[39]:


df_train.describe()


# In[40]:


#Let's check the correlation coefficients of all numerical variables except categorical variables to see which variables are highly correlated

plt.figure(figsize = (16, 8))
sns.heatmap(df_train[num_vars].corr(), annot = True, cmap="YlGnBu")
plt.show()


# In[41]:


# Popping out the 'price' column for y_train
y_train = df_train.pop('price') 
# Creating X_train
X_train = df_train


# In[42]:


y_train.head()


# In[44]:


# Popping out the 'price' column for y_test
y_test = df_test.pop('price')

# Creating X_test
X_test = df_test


# In[45]:


# Add constant
X_test_sm = sm.add_constant(X_test)
X_test_sm.head()


# In[ ]:




