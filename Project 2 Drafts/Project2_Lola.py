#!/usr/bin/env python
# coding: utf-8

# # DS-7331 Machine Learning Project 2
# ## Airbnb Price Data - Logistic and SVM
# ### Allen Miller, Ana Glaser, Jake Harrison, Lola Awodipe
# 
# https://nbviewer.jupyter.org/github/allenmiller17/SMU_7331_ML1_Project_1/blob/main/Mini_Project_Final.ipynb

# In[2]:


#loading libraries
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns

from sklearn.model_selection import train_test_split

import numpy as np
from scipy import stats

from sklearn import metrics as mt


# In[4]:


#setting path and loading data
pdata = pd.read_csv("airbnb.csv")


# 
# For our project, we decided to use AirBnb data from six major cities in the United States from kaggle.com. Our objective is to classify the type of property based on the data attributes like city, number of reviews, bathrooms, bedrooms and number of people it accommodates. To assess the effectiveness of our classification, we will look at the accuracy, precision, recall and evaluate the confusion matrix results.

# ### **Business Understanding**

# ### **Data Preparation**

# ##### Defining the Variables

# In[5]:


#importing the data and exploring the attributes
#pdata = pd.read_csv("airbnb.csv")
pdata.info()


# ##### Preparing Class Variables
# 
# Based on our understanding of the data structure and the assumptions required for this type of classification task, we prepared and transformed the data as follows.

# Due to feedback received from our last submission and implementing the CRISP methodology we decided to re-evaluate  the outliers and structure of the dataset.  Four features stood out to us as being heavily skewed, so we took a deeper dive into those data sets and evaluates how to approach their outliers. 
# 
# The number of reviews feature showed data that was heavily skewed to the left. Running the models with and without the outliers, we concluded that these outliers didn't provide a lot of predictive power, but the feature overall did. This led us to take all the outliers that fell below the 1st quartile, and bring them up ot the lowest whisker. Although it did not normally distribute the observations it did significantly reduce the level of skewness in the data from 3 to 1.
# 
# This process was repeated for the review scores rating, and beds features with a similar result in reducing the skewness of the data.  Since Logistic regression does not require the data to be normally distributed, we proceeded with the analysis.
# 
# Finally after exploring the bedrooms, accommodates and bathrooms outliers, we felt it best to leave these outliers untouched, since these attributes are more relevant when determining what type of property it is. We felt that they still held predictive power.
# 

# ##### Attributes and Assumptions

# Because normality is not an assumption for Logistic Regression we weren't worried about having a normal distribution for our feature observations, so we chose not to transform any of these variables.
# 
# The attribute of neighborhood made our data very sparse and it increased run time dramatically, when we tested the models with and without it, we only lost 3% accuracy points, so the cost benefit of run time vs model performance seemed like a fair trade-off.

# In[6]:


#removing outliers to reduce skewness of data 
z = pdata[pdata['number_of_reviews']> 100]
for i in list (z[z['number_of_reviews']> 100].index):
    pdata.loc[i,'number_of_reviews'] = 100


# In[7]:


y = pdata[pdata['review_scores_rating']< 80]
for i in list (y[y['review_scores_rating']< 80].index):
    pdata.loc[i,'review_scores_rating'] = 80


# In[8]:


x = pdata[pdata['beds']> 5]
for i in list (x[x['beds']> 5].index):
    pdata.loc[i,'beds'] = 5


# In[10]:


#evaluating data for skewness after outlier transformation
pdata.skew()


# In[11]:


#dropping records with excess blank values, still had over 64k records to evaluate
pdata_cls = pdata.dropna()


# In[12]:


#evaluating data for skewness after outlier transformation after splitting data based on task
pdata_cls.skew()


# In[ ]:





# ##### Data Transformation

# To assist in predicting the property type of an Airbnb we decided to make a couple of changes to the features that could increase the predictability. 
# 
# We first collapsed the property type feature to only contain two distinct values, making this a binary classification problem. A property type could only be classified as Apartment or Other. This eliminated the smaller sub-types of a property such as, Loft, Condo, House, etc. We chose Apartment due to the large amounts of observations that were present in the data set and saw an increase in our accuracy (found at the end of this report)
# 
# Second we decided to remove the longitude and latitude variables and replace them with a variable called region. This variable split the United States in half and classified the observation as either East or West. This increased the performance of our models and allowed us to reduce the number of predictors included in the models.

# In[13]:


#transforming the property type to a binary classification
value_list = ["Apartment"]
boolean_series = ~pdata_cls.property_type.isin(value_list)
filtered_df = pdata_cls[boolean_series]

filtered_df.head(100)

for i in list (filtered_df.index):
    pdata_cls.loc[i,'property_type'] = "other"

#transforming the longitude and latitude variables to East / West
pdata_cls["region"] = pd.cut(pdata_cls.longitude,[-200,-100,0],2,labels=["West","East"])


# In[12]:


#evaluating the data after transformation
pdata_cls.head()


# ##### Encoding binary variables

# To help our model we encoded all of the boolean features that were originally stored as character fields to reflect actual boolean type variables and reflected true values with a 1 and false values with a 0.
# 
# We also encoded our response variable to 1 vs 2 to reflect Apartment vs Other.

# In[14]:


#Encoding boolean and categorical variables
replaceStruct = {
                "cleaning_fee":     {True: 1, False: 0},
                "instant_bookable":     {"t": 1, "f": 0},
                "host_identity_verified":     {"t": 1, "f": 0},
                "property_type":     {"Apartment": 1, "other": 2},
                    }

pdata_cls=pdata_cls.replace(replaceStruct)
pdata_cls.head()


# ##### One-Hot Encoding Categorical Variables

# # We proceeded to one hot encode the categorical variables that we were going to leave in our models. This created a reference variable (0) and allows us to interpret the coefficients of the variables easier.
# 
# We also evaluated the number of unique values found in the categorical variables, since hot-encoding the neighborhood attribute, which seemed useful in predicting property type in some cases, had 590 distinct values.  This made the model run time very slow and only gained a modest amount of accuracy.
# 
# We then dropped all of the other columns that would not be used in the proceeding models, like property descriptions, and those that resulted in a 0.0 coefficient value, lacking predictive power.

# In[15]:


#evaluating categorical value count for one-hot-encoding
pdata_cls.nunique()


# In[17]:


#one hot encoding categorical variables and dropping columns that are not used
oneHotCols=["room_type","bed_type","city","cancellation_policy","region"]
pdata_cls.drop(['description','host_response_rate','first_review','host_since','last_review','zipcode','id','latitude','longitude','neighbourhood','cleaning_fee','host_has_profile_pic'], axis=1, inplace=True)
pdata_cls=pd.get_dummies(pdata_cls, columns=oneHotCols,drop_first=True)
pdata_cls.head(10)


# ##### Scaling

# We went with an 80:20 train:test split of the data providing 80% of the data into a training set for teaching the model and 20% of the data into a test set for testing how well the model performs at predicting the property type.
# 
# We also decided to scale the data using the training data set. This helped our model since we had attributes in a variety of scales in our dataset.

# In[18]:


# create variables we are more familiar with
X_cls = pdata_cls.drop('property_type',axis=1).values     
y_cls = pdata_cls['property_type'].values

yhat_cls = np.zeros(y_cls.shape) # we will fill this with predictions
scl_cls = StandardScaler()
X_scaled_cls = scl_cls.fit_transform(X_cls)


# In[ ]:





# ##### Describing the Final Dataset

# In[19]:


pdata_cls.info()


# ### **Modeling and Evaluation**

# ##### Evaluation Metrics Described (Classification / Regression)

# #### Task One: Classification

# In[18]:


#Random Forest - Me
#KNN
#Logistic Regression


# ##### Method for splitting Train and Test Data

# In[23]:


# create cross validation iterator
cv = StratifiedShuffleSplit(n_splits=10, test_size = 0.2, train_size = 0.8)

# now iterate through and get predictions, saved to the correct row in yhat
# NOTE: you can parallelize this using the cross_val_predict method
# fill in the training and testing data and save as separate variables
for trainidx, testidx in cv.split(X_scaled_cls,y_cls):
    # note that these are sparse matrices
    X_train_scaled_cls = X_scaled_cls[trainidx]
    X_test_scaled_cls = X_scaled_cls[testidx]
    y_train_cls = y_cls[trainidx]
    y_test_cls = y_cls[testidx]


# In[ ]:


from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline


#pipe = make_pipeline(('scale', StandardScaler()),  ('knn', KNeighborsClassifier()))

pipe = Pipeline([('scale', StandardScaler()),  
                         ('knn', KNeighborsClassifier())]) 

# Define a range of hyper parameters for grid search
parameters = { 'knn__n_neighbors': [1,5,10,15,20,25,30,35,40,45,50]
              , 'knn__algorithm' : ["auto",  "ball_tree", "kd_tree", "brute"]
             }

#Perform the grid search using accuracy as a metric during cross validation.
gridKnn = GridSearchCV(pipe, parameters, cv=cv, scoring='accuracy') # can try f1_micro, f1_maco accuracy....

#Use the best features from recursive feature elimination during the grid search
gridKnn.fit(X_train_scaled_cls, y_train_cls)

#display the best pipeline model identified during the grid search
gridKnn.best_estimator_


# In[24]:


#verifying the test vs train split
print("{0:0.2f}% data is in training set".format((len(X_train_scaled_cls)/len(pdata_cls.index)) * 100))
print("{0:0.2f}% data is in test set".format((len(X_test_scaled_cls)/len(pdata_cls.index)) * 100))


# In[21]:


from sklearn.ensemble import RandomForestClassifier

#Upload the Random Forest Classifier to the Grid Search
modelGRID_RFC = RandomForestClassifier()

parameters_RFC = {'n_estimators' : [5,10,15,20,25,30,50,75,100,500],
                 'max_features' : ['auto', 'sqrt', 'log2'],
                 'max_depth' : [5,10,15,20,25,30,35,'None'],
                 'min_samples_split' : [2,5,10],
                 'min_samples_leaf' : [1,2,4],
                 'bootstrap' : [True, False]}


# In[ ]:


from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(modelGRID_RFC, n_jobs = 25, param_grid=parameters_RFC,cv=10,scoring='accuracy')
gs.fit(X_train_scaled_cls, y_train_cls)


# In[ ]:


#Random Search
from sklearn.model_selection import RandomizedSearchCV
rnd


# In[ ]:


gs.best_params_


# In[ ]:


params = gs.cv_results_['params']
result = gs.cv_results_['mean_test_score']
for params,result in zip(params, result):
    print(params, 'has an accuracy of', round(result,ndigits=2))


# In[ ]:


from sklearn import metrics as mt
clf_rfc = RandomForestClassifier(bootstrap=False,
                                 max_depth=15,
                                 max_features='auto',
                                 min_samples_leaf= 2,
                                 min_samples_split= 2,
                                 n_estimators=500)
clf_rfc.fit(X_train_scaled_cls, y_train_cls)
y_predict_rfc = clf_rfc.predict(X_test_scaled_cls)
model_score_rfc = round(mt.accuracy_score(y_test_cls, y_predict_rfc), ndigits=3)
print(model_score_rfc)


# In[ ]:


from sklearn.tree import export_graphviz

tree_cls = clf_rfc.estimators[5]

export_graphviz(tree_cls, feature_names=X_train_scaled_cls.columns,filled=True,rounded=True)


# #### Task Two: Regression

# In[ ]:


#LASSO
#Random Forest
#Need another?


# In[ ]:


#imputing missing numerical data by using the median, removing records with missing categorical values
pdata_reg = pdata.fillna(pdata.median())
pdata_reg = pdata_reg.dropna()


# In[ ]:


pdata_reg.skew()


# In[ ]:


replaceStruct = {
                "cleaning_fee":     {True: 1, False: 0},
                "instant_bookable":     {"t": 1, "f": 0},
                "host_identity_verified":     {"t": 1, "f": 0},
                    }
pdata_reg=pdata_reg.replace(replaceStruct)
pdata_reg.head()


# In[ ]:


#one hot encoding categorical variables and dropping columns that are not used
oneHotCols=["room_type","bed_type","city","cancellation_policy", "property_type"]
pdata_reg.drop(['description','host_response_rate','first_review','host_since', 'zipcode','last_review','id','neighbourhood','cleaning_fee','host_has_profile_pic'], axis=1, inplace=True)
pdata_reg=pd.get_dummies(pdata_reg, columns=oneHotCols,drop_first=True)
pdata_reg.head(10)


# In[ ]:


# create regression x and y 
X_reg = pdata_reg.drop('log_price',axis=1).values     
y_reg = pdata_reg['log_price'].values

yhat_reg = np.zeros(y_reg.shape) # we will fill this with predictions
scl_reg = StandardScaler()
X_scaled_reg = scl_reg.fit_transform(X_reg)


# In[ ]:


pdata_reg.info()


# In[ ]:


# create cross validation iterator
from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=10, test_size = 0.2, train_size = 0.8)

# now iterate through and get predictions, saved to the correct row in yhat
# NOTE: you can parallelize this using the cross_val_predict method
# fill in the training and testing data and save as separate variables
for trainidx, testidx in cv.split(X_scaled_reg,y_reg):
    # note that these are sparse matrices
    X_train_scaled_reg = X_scaled_reg[trainidx]
    X_test_scaled_reg = X_scaled_reg[testidx]
    y_train_reg = y_reg[trainidx]
    y_test_reg = y_reg[testidx]


# In[ ]:


pdata_reg.info()


# ## Mini Lab Work 

# ##### Parameter Tuning

# We ran a grid search using a penalty of l2, using 10 fold cross-validation, and proceeded to test which type of solver, C value and max_iterations would be best suited for our data.
# 
# We found that a liblinear solver performed best with our specific data, using the accuracy metric and decided to use it in our Logistic Regression model.

# In[ ]:


#run a grid search to evaluate the parameters as part of hyperparameter tuning
modelGRID = LogisticRegression()

parameters = {'penalty': ['l2'],
             'C': [.001,.75,10],
             'solver': ['newton-cg','liblinear','sag','saga'],
              'max_iter':[100,500]
            }


# In[ ]:


#identify the best parameters as determined by our grid search accuracy results
from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(modelGRID,n_jobs=25,param_grid=parameters,cv=10,scoring='accuracy')
gs.fit(X_train_scaled, y_train)


# In[ ]:


#mean test score values associated with each grid search parameter combination 
params = gs.cv_results_['params']
result = gs.cv_results_['mean_test_score']
for params,result in zip(params, result):
    print(params, 'has an accuracy of', round(result,ndigits=2))


# In[ ]:


#parameter combination with the highest score
gs.best_params_


# ### **Model Evaluation**

# With our optimized parameters in place, we trained our model using the training data set.
# 
# We then predicted the property types we would expect to see using our test variables.
# 
# We finally compared our true response variables from the test data with our y predicted variables and calculated the accuracy of that prediction.
# 
# Our accuracy ended up being 73.6% overall

# In[ ]:


# Fit the model on train
modelFINAL = LogisticRegression(random_state=42,penalty='l2', C = 0.001,solver='liblinear', max_iter=100)
modelFINAL.fit(X_train_scaled, y_train)
#predict on test
y_predict = modelFINAL.predict(X_test_scaled)
model_score = round(mt.accuracy_score(y_test,y_predict),ndigits=3)
print(model_score)


# In[ ]:


#parameters used in the final Logistic Regression Model
modelFINAL.get_params()


# ### Logistic Regression Coefficients

# #### **Interpret Feature Importance**
# 
# The larger the weight, the higher importance the feature is to the model, therefore enhancing our predictability for the property_type. The reference variable encompasses the categorical features of Entire home/apt (room_type), Airbed (bed_type), Boston (city), flexible cancellation_policy, and West region.
# 
# Based on the coefficient values we see listed above, we can determine that room type = Private, with a coefficient of .41 has a singificant influence on the probability of a room being an apartment vs other.  Additionally, the number of bathrooms with a coefficient of .29 is also a significant factor in classifying the property type.
# 
# On the contrary, if they have a bed type = futon or a pull-out sofa, with a coefficient of 0.01, will not carry as much weight in predicting a property type.
# 
# The value of these coefficients make sense to us in the influence they carry in predicting the property type, as one would look at the city, room type and number of bathrooms as more indicative of what type of property it is.  Where bed type may not be strongly indicative of what type of property it is.

# In[ ]:


#coefficient output based on in-class example
weights = modelFINAL.coef_.T
variable_names = x_train.columns
for coef,name in zip(weights, variable_names):
    print(name, 'has weight of', round(coef[0],ndigits=2))


# #### Exceptional work, model performance interpretation
# 
# Looking at the accuracy alone, we would assume the model performs quite well.  However, if we look at the precision, recall and F1 score for the non-apartment or "other" property type, we see the model does not perform as well as expected.  The large difference in F1 score between the property type = apartment vs property type = other, signals we should consider rebalancing our lower representative property types to fully leverage their predictive attributes. 

# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

## function to get confusion matrix in a proper format
def draw_cm( actual, predicted ):
    cm = confusion_matrix( actual, predicted)
    sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels = [0,1] , yticklabels = [0,1] )
    plt.ylabel('Observed')
    plt.xlabel('Predicted')
    plt.show()

print("Training accuracy",round(modelFINAL.score(X_train_scaled,y_train),ndigits=3))
print()
print("Testing accuracy",round(mt.accuracy_score(y_test, y_predict),ndigits=3))
print()
print('Confusion Matrix')
print(draw_cm(y_test,y_predict))

print(classification_report(y_test,y_predict))


# The plot below is a visual representation of the logistic regression coefficient, as displayed on the chart having a value of NYC or being in the Eastern region of the united states reduces the odds of a property being an apartment vs other.

# In[ ]:


#coeffient plot

from matplotlib import pyplot as pyplot
get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use('ggplot')

weights = pd.Series(modelFINAL.coef_[0],index=x_train.columns)
weights.plot(kind='bar')
plt.show()


# ### Support Vector Machines

# ##### Hyperparameter Tuning

# For our support vector machine model we applied a grid search function with 10 fold cross validation, to determine which parameters would perform best given our data.
# 
# We found that using an alpha 0.01, using a hinge loss and providing a penalty of l2 is optimal for our classification task.
# 
# Additionally, the stochastic gradient descent model performed more efficiently with regards to run time since our data was initially quite sparse given the number of columns in our grid due to one hot encoding the neighborhood category.

# In[ ]:


#https://michael-fuchs-python.netlify.app/2019/11/11/introduction-to-sgd-classifier/

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
params = {
    "loss" : ["hinge", "log", "squared_hinge", "modified_huber", "perceptron"],
    "alpha" : [0.0001, 0.001, 0.01, 0.1],
    "penalty" : ["l2"],
}
clf = SGDClassifier()
grid = GridSearchCV(clf, param_grid=params, cv=10)
grid.fit(X_train_scaled, y_train)
print(grid.best_params_)


# 
# The plot below shows the perfomance of different loss metrics on our data and shows hinge perfoming best.

# In[ ]:


losses = ["hinge", "log", "modified_huber", "perceptron", "squared_hinge"]
scores = []
for loss in losses:
    clf = SGDClassifier(loss='hinge', penalty="l2", alpha=0.01, max_iter=1000)
    clf.fit(X_train_scaled, y_train)
    scores.append(clf.score(X_test_scaled, y_test))
plt.title("Effect of loss")
plt.xlabel("loss")
plt.ylabel("score")
x = np.arange(len(losses))
plt.xticks(x, losses)
plt.plot(x, scores)


# ##### Final Support Vector Machine Model

# After identifying the optimal parameters using the grid search function, we applied them to our final SVM Model.
# 
# The model accuracy for our classification task is 73%

# In[ ]:


svm_model = SGDClassifier(alpha= 0.01, loss= 'hinge', penalty='l2')
svm_model.fit(X_train_scaled, y_train)
grid_predictions = svm_model.predict(X_test_scaled)
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, grid_predictions)))


# In[ ]:


#coefficient list based on class example
weights = svm_model.coef_.T
variable_names = x_train.columns
for coef,name in zip(weights, variable_names):
    print(name, 'has weight of', round(coef[0],ndigits=2))


# ### Exceptional work, model performance interpretation SVM
# 
# Just like the accuracy in the Logistic Regression, we would assume the model performs quite well based on accuracy score alone. However, when we look at the precision, recall and F1 score for the non-apartment or "other" property type, we see the model does not perform as well as expected. 
# 
# Like the Logistic Regression model performance above, the large difference in F1 score between the property type = apartment vs property type = other, signals we should consider rebalancing our lower representative property types to fully leverage their predictive attributes.
# 
# This is surprising given the difference in coefficient weights, since the coefficient values differ in weight, we would expect a different performance metrics from the SVM model results. 

# In[ ]:


print(draw_cm(y_test, grid_predictions))
print(classification_report(y_test, grid_predictions))


# ### **Model Comparison and Model Advantages**

# To determine the best overall model, accuracy, precision, recall and their harmonized value of F1-score were considered as the core metrics. The Logistic Regression model had a slightly better performance based on the F1 score compared to Stochastic Gradient Descent Support Vector Machine model at 81% Vs 80% respectively.
# 
# We started modeling the full dataset and the time and efficiency to run each model was quite high. Additionally, the number of unique values in the neighborhood data, which we considered an important attribute at the time made our data structure quite sparse.
# 
# To have a better performance on both execution and training of the large amount of data, we moved from SVM to Stochastic Gradient Descent optimized SVM, which improved performance but still had quite a long run time.  We then removed neighborhood (removing the columns with too many values).This reduced the accuracy across all our models by 3%, reducing our accuracy from 76% to 73% but increased the speed and performance for both models.  Another approach would have been to sample the data but we would have to worry about balancing the classes as there are property types that are significantly under-represented in the data set.
# 
# Overall the models performed quite similarly in our classification task, we would choose the Logistic Regression model due to the simplicity of the model and our ability to calculate the odds ratio given any set of attributes, predicting the likelihood of a certain classification. We consider the execution speed and interpretability of the Logistic Regression model to be a major advantage over the SVM model.

# ## **Interpret Support Vectors**

# Looking at the separation of the original data and the sepaation of the support vectors we see that the original data provides greater separation of the two. Likely this is due to our support vectors containing both correct and incorrect classifications. Recall that we had an overall accuracy of 73% for support vectors, however the recall barely provided 50% accuracy when classifying the Other property type. These misclassified values shrink the margin that SVC would create.

# In[ ]:


from sklearn.model_selection import ShuffleSplit
X = X.values
Y = Y.values

num_cv_iterations = 3
num_instances = len(Y)
cv_object = ShuffleSplit(n_splits=num_cv_iterations,test_size=0.2)
print(cv_object)

for train_indices, test_indices in cv_object.split(X,Y):
    X_train = X[train_indices]
    y_train = Y[train_indices]
    
    X_test = X[test_indices]
    y_test = Y[test_indices] 


# In[ ]:


from sklearn.svm import SVC

svm_chart_model = SVC()
svm_chart_model.fit(X_train, y_train)


# In[ ]:


#X_train = pd.DataFrame(X_train)
df_tested_on = X_train.iloc[train_indices].copy() # saved from above, the indices chosen for training

# now get the support vectors from the trained model
df_support = df_tested_on.iloc[svm_chart_model.support_,:].copy()
df_tested_on.head()

variable_names = x_train.columns
df_support.columns = variable_names
df_support['property_type'] = y_train[svm_chart_model.support_] # add back in the 'Survived' Column to the pandas dataframe


# In[ ]:


#chart from class example
from pandas.plotting import boxplot

# group the original data and the support vectors
df_grouped_support = df_support.groupby(['property_type'])
df_grouped = pdata.groupby(['property_type'])

# plot KDE of Different variables
vars_to_plot = ['bedrooms','beds','bathrooms']

for v in vars_to_plot:
    plt.figure(figsize=(10,4))
    # plot support vector stats
    plt.subplot(1,2,1)
    ax = df_grouped_support[v].plot.kde() 
    plt.legend(['Apartment','Other'])
    plt.title(v+' (Instances chosen as Support Vectors)')
    
    # plot original distributions
    plt.subplot(1,2,2)
    ax = df_grouped[v].plot.kde() 
    plt.legend(['Apartment','Other'])
    plt.title(v+' (Original)')


# In[ ]:





# In[ ]:


get_ipython().run_line_magic('time', '')
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline


#pipe = make_pipeline(('scale', StandardScaler()),  ('knn', KNeighborsClassifier()))

pipe = Pipeline([('scale', StandardScaler()),  
                         ('knn', KNeighborsClassifier())]) 

# Define a range of hyper parameters for grid search
parameters = { 'knn__n_neighbors': [1,5,10,15,20,25,30,35,40,45,50]
              , 'knn__algorithm' : ["auto",  "ball_tree", "kd_tree", "brute"]
             }

#Perform the grid search using accuracy as a metric during cross validation.
gridKnn = GridSearchCV(pipe, parameters, cv=cv, scoring='accuracy') # can try f1_micro, f1_maco accuracy....

#Use the best features from recursive feature elimination during the grid search
gridKnn.fit(pdata, Y)

#display the best pipeline model identified during the grid search
gridKnn.best_estimator_


# In[ ]:




