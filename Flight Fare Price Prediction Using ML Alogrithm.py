


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
#import re 




##Loading dataset

train_data = pd.read_excel("Data_Train.xlsx")

pd.set_option('display.max_columns', None)

print(train_data.head(10))

print(train_data.info())

print(train_data.describe())

##Let's check missing values in dataset

print(train_data.isnull().sum())

##We drop null values from dataset
print(train_data.dropna(inplace = True))

print(train_data.isnull().sum())

## We can see that there is no missing values in datasets.

##We can see that duration features in hour and minutes

print(train_data['Duration'].value_counts())


#  - Now we do the EDA
#  - From the dataset we can see that, 'Date_of_Journey' is object data type, Therefore, we have to convert this datatype into timestamps data type, So as use this column properly for prediction.
#  - We can use to_datetime() function to convert object to timestampes data type.
#  - "".dt.day method will extract only day of that date, & "".dt.month method will extract only month of that date.

train_data['Journey_day'] = pd.to_datetime(train_data['Date_of_Journey'], format = "%d/%m/%Y").dt.day

train_data['Journey_month'] = pd.to_datetime(train_data['Date_of_Journey'], format = "%d/%m/%Y").dt.month

print(train_data.head()) 

#Since we have converted date_of_journey feature into integers, now we can drop as it is no of use.

train_data.drop('Date_of_Journey', inplace = True, axis = 1)

print(train_data.head())

##Departure Time is when a plane leaves the gate.
##Similar to DateofJourney we can extract values from Dep_Time.
##Extracting hours from Dep_time
train_data['Dep_hour'] = pd.to_datetime(train_data['Dep_Time']).dt.hour

##Extracting minutes from Deptime
train_data['Dep_minutes'] = pd.to_datetime(train_data['Dep_Time']).dt.minute

print(train_data.head())

#Now we can drop dep_time feature as it is no of use.
train_data.drop('Dep_Time', axis = 1, inplace = True)

print(train_data.head())

##Arrival time is when the plane pulls up to gate.
##Similar to Day of Journey we can exttract values from arrival_time.

##Extraccting Hour
train_data['Arrival_Hour'] = pd.to_datetime(train_data['Arrival_Time']).dt.hour

##Extracting Minutes
train_data['Arrvial_Minutes'] = pd.to_datetime(train_data['Arrival_Time']).dt.minute

##now we can drop the arrivale time feature as it is no of use.
train_data.drop(['Arrival_Time'], axis = 1, inplace = True)

## Time taken by plane to reach destination is called Duration.
## It is difference between Departure time and arrival time.

##assining and coverting duration column to lists
duration = list(train_data['Duration'])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:  ##check if duration contains only hour or minute
        if 'h' in duration[i]:
            duration[i] = duration[i].strip() + " 0m" #adds 0 minute
        else:
            duration[i] = "0h " + duration[i]  ##adds 0 hour
        
duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = 'h')[0]))  ##extract hours from duration
    duration_mins.append(int(duration[i].split(sep = 'm')[0].split()[-1]))  ##extract only minutes from duration


print(list(train_data['Duration'])) 

#Now we adding duration_hours and durations_mins list to train_data dataframe

train_data['Duration_hour'] = duration_hours
train_data['Duration_mins'] = duration_mins

##Now we drop duration featues as it is no of use.
print(train_data.drop('Duration', axis = 1, inplace = True))

print(train_data.head())


# #### Handling Categorical Data 
#  - We can handle many way categorical data, Some of them are:
#  1. "Nominal data" -> data are not in order : Here we used OneHotEncoder.
#  2. "Ordinal data" -> data are in order : Here we used LabelEncoder.

# In[28]:


train_data.columns


# In[29]:


train_data['Airline'].value_counts()


# In[30]:


#let's viusalize dataset
sns.catplot(x = 'Airline', y = 'Price', data = train_data.sort_values('Price', ascending = False),kind = 'boxen', height = 6, aspect = 3)


#  - From above plot the we can see that Jet Airways airlnie have highest Price.
#  

# In[31]:


##as Airline feature is Nominal Categorical data we do OneHotEncoding
Airline = train_data[['Airline']]
Airline = pd.get_dummies(Airline, drop_first = True)
Airline.head()


# In[32]:


train_data['Source'].value_counts()


# In[33]:


#let's viusalize dataset
sns.catplot(x = 'Source', y = 'Price', data = train_data.sort_values('Price', ascending = False),kind = 'boxen', height = 6, aspect = 3)


# In[34]:


##As Source feature is Nominal Categorical Data we will perform OneHotENcoding

Source = train_data[['Source']]
Source = pd.get_dummies(Source, drop_first = True)
Source.head()


# In[35]:


train_data['Destination'].value_counts()


# In[36]:


## As Destination is Nominal categorical data we will perform OneHotEncoding

Destination =  train_data[['Destination']]
Destination = pd.get_dummies(Destination, drop_first = True)
Destination.head()


# In[37]:


train_data.head()


# In[38]:


train_data['Route']


# In[39]:


## Additional_info contains almost 80% no_info 
## Route and Total_Stops are related to each other.


# In[40]:


train_data.drop(["Route", "Additional_Info"], axis = 1, inplace = True)


# In[41]:


train_data.shape


# In[42]:


train_data['Total_Stops'].value_counts()


# In[43]:


##It is case of Ordinal categorical variable we perform LabelEncoder
##Here we assigned values with corresponding keys.
train_data.replace({'non-stop':0, '1 stop': 1, '2 stops':2, '3 stops': 3, '4 stops': 4}, inplace = True)


# In[44]:


train_data.head()


# In[45]:


##Concatenate Dataframe --> train_data + Airline + Source + Destination

data_train = pd.concat([train_data, Airline, Source, Destination], axis = 1)


# In[46]:


data_train.head()


# In[47]:


data_train.drop(['Airline', 'Source', 'Destination'], axis = 1, inplace = True)


# In[48]:


data_train.head()


# In[49]:


data_train.shape


# ### Test Data

# In[50]:


test_data = pd.read_excel("C:\PythonCSV\CSV_file\Flight_price_prediction\Test_set.xlsx")


# In[51]:


test_data.head()


# In[52]:


test_data.shape


# ### Data Preprocessing

# In[53]:


test_data.columns


# In[54]:


test_data.isnull().sum()


# In[55]:


test_data['Journey_day'] = pd.to_datetime(test_data.Date_of_Journey, format = "%d/%m/%Y").dt.day


# In[56]:


test_data['Journey_month'] = pd.to_datetime(test_data.Date_of_Journey, format = '%d/%m/%Y').dt.month


# In[57]:


test_data.drop('Date_of_Journey', axis= 1, inplace = True)


# In[58]:


##Departure Time is when a plane leaves the gate.
##Similar to DateofJourney we can extract values from Dep_Time.
##Extracting hours from Dep_time
test_data['Dep_hour'] = pd.to_datetime(test_data['Dep_Time']).dt.hour

##Extracting minutes from Deptime
test_data['Dep_minutes'] = pd.to_datetime(test_data['Dep_Time']).dt.minute


# In[59]:


test_data.drop('Dep_Time', axis= 1, inplace = True)


# In[60]:


##Arrival time is when the plane pulls up to gate.
##Similar to Day of Journey we can exttract values from arrival_time.

##Extraccting Hour
test_data['Arrival_Hour'] = pd.to_datetime(test_data['Arrival_Time']).dt.hour

##Extracting Minutes
test_data['Arrvial_Minutes'] = pd.to_datetime(test_data['Arrival_Time']).dt.minute

##now we can drop the arrivale time feature as it is no of use.
test_data.drop(['Arrival_Time'], axis = 1, inplace = True)


# In[61]:


## Time taken by plane to reach destination is called Duration.
## It is difference between Departure time and arrival time.

##assining and coverting duration column to lists
duration = list(test_data['Duration'])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:  ##check if duration contains only hour or minute
        if 'h' in duration[i]:
            duration[i] = duration[i].strip() + " 0m" #adds 0 minute
        else:
            duration[i] = "0h " + duration[i]  ##adds 0 hour
        
duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = 'h')[0]))  ##extract hours from duration
    duration_mins.append(int(duration[i].split(sep = 'm')[0].split()[-1]))  ##extract only minutes from duration
    


# In[62]:


#now we adding duration_hours and durations_mins list to train_data dataframe

test_data['Duration_hour'] = duration_hours
test_data['Duration_mins'] = duration_mins


# In[63]:


test_data.head()


# In[64]:


##as Airline feature is Nominal Categorical data we do OneHotEncoding
Airline = test_data[['Airline']]
Airline = pd.get_dummies(Airline, drop_first = True)
Airline.head()


# In[65]:


##As Source feature is Nominal Categorical Data we will perform OneHotENcoding

Source = test_data[['Source']]
Source = pd.get_dummies(Source, drop_first = True)
Source.head()


# In[66]:


## As Destination is Nominal categorical data we will perform OneHotEncoding

Destination =  test_data[['Destination']]
Destination = pd.get_dummies(Destination, drop_first = True)
Destination.head()


# In[67]:


## Additional_info contains almost 80% no_info 
## Route and Total_Stops are related to each other.

test_data.drop(["Route", "Additional_Info"], axis = 1, inplace = True)


# In[68]:


##It is case of Ordinal categorical variable we perform LabelEncoder
##Here we assigned values with corresponding keys.
test_data.replace({'non-stop':0, '1 stop': 1, '2 stops':2, '3 stops': 3, '4 stops': 4}, inplace = True)


# In[69]:


test_data.shape


# In[70]:


##Concatenate Dataframe --> train_data + Airline + Source + Destination

data_test = pd.concat([test_data, Airline, Source, Destination], axis = 1)


# In[71]:


data_test.drop(['Airline', 'Source', 'Destination'], axis = 1, inplace = True)


# In[72]:


data_test.head()


# In[73]:


data_test.drop('Duration', axis = 1, inplace = True)


# In[74]:


data_test.head()


# In[75]:


test_data.shape


# In[76]:


data_test.shape


# ### Feature Selection
#  - Finding out best feature which will contribute and have good relation with target variable, following are some of the feature selection methos.
#  1. Heatmap
#  2. SelectKbest
#  3. feature_importance_

# In[77]:


data_train.shape


# In[78]:


data_train.columns


# In[79]:


X = data_train.drop('Price', axis = 1)


# In[80]:


X.head()


# In[81]:


y = data_train['Price']


# In[82]:


y.head()


# In[83]:


##we finde correlation between independent variable and dependent variable
plt.figure(figsize = (18,18))
sns.heatmap(train_data.corr(), annot = True, cmap = 'RdYlGn')


# In[84]:


train_data.head()


# In[ ]:





# In[85]:


##we see that feature importance using extratree regresssor
from sklearn.ensemble import ExtraTreesRegressor


# In[86]:


selection = ExtraTreesRegressor()


# In[87]:


selection.fit(X,y)


# In[88]:


print(selection.feature_importances_)


# - We plot graph of the feature importance for better visualization

# In[89]:


plt.figure(figsize = (12,8))
feat_importance = pd.Series(selection.feature_importances_, index = X.columns)
feat_importance.nlargest(20).plot(kind = 'barh')
plt.show()


#  - We can see that total_Stops feature has more importance follow by journey_day, jet_airways, durration_hour 

# In[90]:


#Now we split dataset into train_data & test_data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)


# In[91]:


print(x_train.shape, x_test.shape)


# In[92]:


print(y_train.shape, y_test.shape)


# In[93]:


##Here we are using RandomForestRegressor algorithm 
from sklearn.ensemble import RandomForestRegressor


# In[94]:


forest = RandomForestRegressor()


# In[95]:


##Now we train the model
forest.fit(x_train, y_train)


# In[96]:


##Now we predict model on test data
pred_data = forest.predict(x_test)


# In[97]:


pred_data


# In[98]:


forest.score(x_test, y_test)


# In[99]:


forest.score(x_train, y_train)


# In[100]:


sns.distplot(y_test - pred_data)


# In[101]:


plt.scatter(y_test, pred_data, alpha = 0.5)
plt.xlabel('Y-test')
plt.ylabel('Predicted Data')
plt.show()


# In[102]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# In[103]:


mse = mean_squared_error(y_test, pred_data)


# In[104]:


mse


# In[105]:


mae = mean_absolute_error(y_test, pred_data)
mae


# In[106]:


rmse = np.sqrt(mean_squared_error(y_test, pred_data))
rmse


# In[107]:


score_r2 = r2_score(y_test, pred_data)
score_r2


# #### Hyperparameter Tunning
#  - We do the hyperparameter tunning using RandomsizedSearchCV method.
#  - We assign hyperparameters in the form of dictionary
#  - Then fit the model
#  - Check the best parameters and best score

# In[108]:


from sklearn.model_selection import RandomizedSearchCV


# In[109]:


#Numeber of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 12)]
##Number of feature to consider at every split
max_features = ['auto', 'sqrt']

### Maximum level of trees
max_depth = [int(x) for x in np.linspace(start = 5, stop = 30,num = 6)]

###maximum number of samples required to split node
min_samples_split = [2,5, 10, 15]

##minimum number of samples requred at each leaf node
min_samples_leaf = [1,2,5,10]


# In[110]:


##create random grid as dictionary
random_grid = {"n_estimators": n_estimators,
               "max_features": max_features,
              "max_depth":max_depth,
              "min_samples_split": min_samples_split,
              "min_samples_leaf":min_samples_leaf}


# In[111]:


rf_random = RandomizedSearchCV(estimator= forest, param_distributions= random_grid, scoring = 'neg_mean_squared_error', n_iter= 10, cv = 5, verbose = 2, random_state= 42, n_jobs = 1)


# In[112]:


rf_random.fit(x_train, y_train)


# In[113]:


rf_random.best_params_


# In[114]:


prediction = rf_random.predict(x_test)


# In[115]:


sns.distplot(y_test - prediction)


# In[116]:


plt.scatter(y_test, prediction, alpha = 0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()


# In[117]:


print('MAE:', mean_absolute_error(y_test, prediction))
print('MSE:', mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(mean_squared_error(y_test, prediction)))


# #### Save the model to reuse it again

# In[118]:


import pickle


# In[120]:


##open a file, where we want to store
file = open("flight_rf.pkl", 'wb')


# In[121]:


##dump information to that file
pickle.dump(forest, file)


# In[122]:


model = open('flight_rf.pkl','rb')


# In[123]:


random = pickle.load(model)


# In[124]:


y_prediction = random.predict(x_test)


# In[125]:


r2_score(y_test, y_prediction)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




