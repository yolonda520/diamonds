# %%
# Check change in new branch -- predictive
# %% [markdown]
# # 1. Import Packages
# %%
# we import necessary packages as below.
import pandas as pd
import numpy as np
import seaborn as sns
import random as rand
import matplotlib.pyplot as plt
from IPython.display import display

from sklearn.model_selection import train_test_split  # train/test split
# linear regression (scikit-learn)
from sklearn.linear_model import LinearRegression
import sklearn.linear_model  # linear models
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix    # confusion matrix

from sklearn.neighbors import KNeighborsRegressor  # KNN for Regression
from sklearn.preprocessing import StandardScaler  # standard scaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
# %%
# setting pandas print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

diamonds = pd.read_csv('diamonds.csv')
display(diamonds.head())
display(diamonds.shape)
# %% [markdown]
# # 2.Clean Data
# ## 2.1 Check data info
# %%
# check the diamonds info
diamonds.info()
# %% [markdown]
# Insight:<br>
# 1.It looks no missing value here;<br>
# 2.From the diamonds info, we can see 'cut','color','clarity' are objects.We
# will see what it is inside of the objects to find some relationship.<br>
# 3.'unnamed: 0' is index can be removed;<br>

# %%
# check the diamonds' description
diamonds.describe().round(2)
# %% [markdown]
# Insight:<br>
# 1.The max carat weight is 5.01, far away from the mean 0.78. Is that some
# interesting thing that such a high weight?<br>
# 2.The min of x,y,z, are 0! Which doesn't make sense that either length,
#  height or width is zero! That would probably be missing value!<br>
# %% [markdown]
#  ## 2.2 Check data missing value and handle missing value
# %%
diamonds.isnull().sum()
# %% [markdown]
# As we mentioned before, it looks no missing value at all. However, let's
# check the x,y,z which is 0.

# %%
diamonds.loc[diamonds['x'] == 0].count()
# %%
diamonds.loc[diamonds['y'] == 0].count()
# %%
diamonds.loc[diamonds['z'] == 0].count()
# %%
# To simplify this data set a bit more, let's combine the 'x', 'y' and 'z' columns into a 'volume' column.
diamonds['volume'] = (diamonds['x'] * diamonds['y'] * diamonds['z']).round(1)
# %%
diamonds.loc[:, 'volume'][diamonds['volume'] == 0].count()
# %% [markdown]
# Insight:<br>
# We can see there are 20 observations with dimension '0', it is not a big data, so I will choose to drop them.
diamonds = diamonds.drop(diamonds.iloc[:, 0][diamonds['volume'] == 0], axis=0)
diamonds.shape

# %% [markdown]
# ## 2.3 Check 'cut','color','clarity' categorical variables and change to number
# %%
print(f"""
cut
-----------
{diamonds['cut'].value_counts().sort_values()}

color
-----------
{diamonds['color'].value_counts().sort_index()}

clarity
-----------
{diamonds['clarity'].value_counts().sort_index()}
""")
# %% [markdown]
# Insight:<br>
# 1.We can see from the categories, cut quality of the cut (Fair, Good, Very Good, Premium, Ideal)<br>
#   We can put (1,2,3,4,5) instead of  (Fair, Good, Very Good, Premium, Ideal), 1 means worst, 5 means best.<br>
# 2.color diamond colour, from J (worst) to D (best)<br>
#   we can put (7,6,5,4,3,2,1) instead of (D,E,F,G,H,I,J),
#   7 is the best, 1 is the worst<br>
# 3.clarity a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))<br>
#   we can put (1,2,3,4,5,6,7,8) instead of (I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF)
#   I1 is worst, IF is best.<br>

# %%
diamond = diamonds.copy()
# Replace the categorical variables to number.
diamond.replace({'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5,
                 'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D': 7,
                 'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8}, inplace=True)
diamond.head()

# %% [markdown]
# ## 2.4 Drop undesired  columns
# %%
# carat weight can totally explain the volume,x,y,z, so I prefer to drop x,y,z first.
diamond = diamond.drop(['Unnamed: 0', 'x', 'y', 'z'], axis=1)

diamond.shape
# %%
diamond.head()
# %% [markdown]
# ## 2.5 Outliers Analysis
# %%
fig, ax = plt.subplots(nrows=2,
                       ncols=2,
                       figsize=(12, 12))
sns.boxplot(y=diamond['carat'],
            ax=ax[0, 0])
sns.boxplot(y=diamond['depth'],
            ax=ax[0, 1])
sns.boxplot(y=diamond['table'],
            ax=ax[1, 0])
sns.boxplot(y=diamond['price'],
            ax=ax[1, 1])
plt.show()

# %% [markdown]
# Insight:<br>
# 1. we can see there are a lot outliers in carat weight, it looks if carat > 2 means outliers;<br>
# 2. The depth should between 58 and 65 percentage, out of this range are outliers too;<br>
# 3. The table should between 51 and 64, and there are outliers out of this range;<br>
# 4. The price greater than 12500 are outliers.<br>


# %% [markdown]
# # 3. Analysis Data
# ## 3.1 Develop Pearson correlation matrix with data.
# %%
df_corr = diamond.corr().round(2)
print(df_corr.loc[:, 'price'].sort_values(ascending=False))
# %% [markdown]
# Insight:
# We can see that carat ,volume have high correlation with price.

# %% [markdown]
# ## 3.2 Plot data
# ### 3.2.1 Distplot
# %%
# Let's see the trend of variables.
fig, ax = plt.subplots(nrows=2,
                       ncols=2,
                       figsize=(12, 12))
sns.distplot(diamond['carat'],
             bins=20,
             color='r',
             ax=ax[0, 0])
sns.distplot(diamond['depth'],
             bins=20,
             color='g',
             ax=ax[0, 1])

sns.distplot(diamond['table'],
             bins=20,
             color='y',
             ax=ax[1, 0])

sns.distplot(diamond['price'],
             bins=20,
             color='b',
             ax=ax[1, 1])
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(nrows=2,
                       ncols=2,
                       figsize=(12, 12))
sns.distplot(diamond['cut'],
             bins=20,
             color='r',
             ax=ax[0, 0])

sns.distplot(diamond['clarity'],
             bins=20,
             color='g',
             ax=ax[0, 1])

sns.distplot(diamond['color'],
             bins=20,
             color='y',
             ax=ax[1, 0])

sns.distplot(diamond['volume'],
             bins=20,
             color='b',
             ax=ax[1, 1])
plt.tight_layout()
plt.show()

# %% [markdown]
# Insight:<br>
# 1. The carat weight have 5 peaks, Probably there are popular weight like 1, 2.<br>
# 2. The cut high level always more popular than low levels;<br>
# 3. The clarity 3,4 are more popular than others,which 'SI1': 3, 'VS2': 4 are more popular;<br>
# 4. color 4 which is G more popular than others.<br>

# %% [markdown]
# ### 3.2.2 Developing trend changes - Scatter plot
# %%
fig, ax = plt.subplots(nrows=2,
                       ncols=2,
                       figsize=(12, 12))

sns.scatterplot(x=diamond['carat'],
                y=diamond['price'],
                color='r',
                ax=ax[0, 0])

sns.scatterplot(x=diamond['volume'],
                y=diamond['price'],
                color='y',
                ax=ax[0, 1])

sns.scatterplot(x=diamond['table'],
                y=diamond['price'],
                color='g',
                ax=ax[1, 0])

sns.scatterplot(x=diamond['depth'],
                y=diamond['price'],
                color='b',
                ax=ax[1, 1])
plt.tight_layout()
plt.show()
####################################
fig, ax = plt.subplots(nrows=1,
                       ncols=3,
                       figsize=(12, 6))

sns.scatterplot(x=diamond['cut'],
                y=diamond['price'],
                color='r',
                ax=ax[0])

sns.scatterplot(x=diamond['color'],
                y=diamond['price'],
                color='y',
                ax=ax[1])

sns.scatterplot(x=diamond['clarity'],
                y=diamond['price'],
                color='g',
                ax=ax[2])
plt.tight_layout()
plt.show()
# %% [markdown]
# ### 3.2.3 Bar plot
# %%
fig, ax = plt.subplots(nrows=1,
                       ncols=3,
                       figsize=(12, 6))

sns.barplot(x=diamond['cut'],
            y=diamond['price'],
            color='r',
            ax=ax[0])

sns.barplot(x=diamond['color'],
            y=diamond['price'],
            color='y',
            ax=ax[1])
sns.barplot(x=diamond['clarity'],
            y=diamond['price'],
            color='g',
            ax=ax[2])

# %% [markdown]
# Insight:<br>
# 1. cut at 4 level is best deal price;<br>
# 2. color at 1 is best average price, which is weird, because the worst color has the best price.<br>
#    Probably most customers are hard to tell which is the better color;<br>
# 3. clarity 2 is the best price, which is weird too, as the clarity 2 is not a good level.

# %%
print(diamond['price'][diamond['color'] == 5].mean())
print(diamond['carat'][diamond['color'] == 5].mean())
# %%
print(diamond['price'][diamond['color'] == 1].mean())
print(diamond['carat'][diamond['color'] == 1].mean())

# %% [markdown]
# Insights:<br>
# 1.It looks worst color has the higher weight, so the color 1 much more expensive than color 5,which is make sense.<br>
# 2.We can see that the better cut,color, clarity diamonds always have light carat weight, base on carat weight is high related to price, so the price of high quality of cut,color, clarity is under the average.<br>
#  %% [markdown]
# # 4.Feature Engineering <br>
# ## 4.1 One-hot encoding for categorial features <br>
# %%
a = pd.get_dummies(diamonds)
a.head()
# %% [markdown]
# ## 4.2 Drop useless variables to decrease noise of data
# %%
# To get a better result for machine learning, I will drop some columns that can be known by other columns.
diamonds = a.drop(['Unnamed: 0', 'volume', 'cut_Very Good',
                   'color_G', 'clarity_VVS2'], axis=1)
diamonds.head()
# %%
diamonds.shape
# %% [markdown]
# ## 4.3
# %% [markdown]
# # 5. Predictive Models Preparation
# ## 5.1 Target Data and explanatory Data
# %%
diamonds_explanatory = diamonds.drop(['price'], axis=1)
diamonds_target = diamonds.loc[:, 'price']

# %% [markdown]
# ## 5.2 Distance Standardization <br>
# Transform the explanatory variables of a dataset so that they
# are standardized, or put into a form where each feature's variance is
#  measured on the same scale. In general, distance-based algorithms (i.e. KNN)
#  perform much better after standardization. <br><br>
# Standard Scaler:<br>
# Instantiate<br>
# Fit<br>
# Transform<br>
# Convert<br>
# %%
# INSTANTIATING a StandardScaler() object
scaler = StandardScaler()

# FITTING the scaler with housing_data
scaler.fit(diamonds_explanatory)

# TRANSFORMING our data after fit
X_scaled = scaler.transform(diamonds_explanatory)


# converting scaled data into a DataFrame
X_scaled_df = pd.DataFrame(X_scaled)


# checking the results
X_scaled_df.describe().round(2)
# %%
X_scaled_df.columns = diamonds_explanatory.columns
#  Checking pre- and post-scaling of the data
print(f"""
Dataset BEFORE Scaling
----------------------
{pd.np.var(diamonds_explanatory)}


Dataset AFTER Scaling
----------------------
{pd.np.var(X_scaled_df)}
""")
# %% [markdown]
# ## 5.3 Training and testing data split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df,
    diamonds_target,
    test_size=0.25,  # always use 0.25
    random_state=222)

# Training set
print(X_train.shape)
print(y_train.shape)

# Testing set
print(X_test.shape)
print(y_test.shape)

# %% [markdown]
# # 6. Model Building
# %%
# creating an empty DataFrame
model_performance = pd.DataFrame(columns=['Model',
                                          'Training Accuracy',
                                          'Testing Accuracy',
                                          'AUC Value'
                                          ])
model_performance
# %% [markdown]
# ## 6.1 KNN Model
# %%
# Use a loop and visually inspect the optimal value for k
# creating lists for training set accuracy and test set accuracy
training_accuracy = []
test_accuracy = []

# building a visualization of 1 to 50 neighbors
neighbors_settings = range(1, 15)


for n_neighbors in neighbors_settings:
    # Building the model
    clf = KNeighborsRegressor(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)

    # Recording the training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))

    # Recording the generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))


# plotting the visualization
fig, ax = plt.subplots(figsize=(12, 8))
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

# finding the optimal number of neighbors
opt_neighbors = test_accuracy.index(
    max(test_accuracy)) + 1           # +1因为index start 0
print(f"""The optimal number of neighbors is {opt_neighbors}""")

# %%
# INSTANTIATING a model with the optimal number of neighbors
knn_opt = KNeighborsRegressor(n_neighbors=4)


# FITTING the model based on the training data
knn_opt.fit(X_train, y_train)


# PREDITCING on new data
knn_opt_pred = knn_opt.predict(X_test).astype(int)


# SCORING the results
print('Training Score:', knn_opt.score(X_train, y_train).round(4))
print('Testing Score:',  knn_opt.score(X_test, y_test).round(4))


# saving scoring data for future use
knn_opt_score_train = knn_opt.score(X_train, y_train).round(4)
knn_opt_score_test = knn_opt.score(X_test, y_test).round(4)
#### problem below
# knn_opt_auc = roc_auc_score(y_true=np.array(y_test.astype(int)),
#                             y_score=np.array(knn_opt_pred.astype(int)),
#                             multi_class='ovr').round(4)
# %%
# appending to model_performance
model_performance = model_performance.append(
    {'Model': 'Opt KNN',
     'Training Accuracy': knn_opt_score_train,
     'Testing Accuracy': knn_opt_score_test,
     'AUC Value': knn_opt_auc},
    ignore_index=True)


# checking the results
model_performance
# %% [markdown]
# 6.2 Linear Regression Model
# %%
# creating a hyperparameter grid
param_grid = {
    'fit_intercept': [True, False],
    'normalize': [True, False],
    'copy_X': [True, False],
    'n_jobs': [None, -1]
}

# INSTANTIATING the model object without hyperparameters
lnreg = LinearRegression()

# GridSearchCV object
lnreg_cv = GridSearchCV(lnreg,
                        param_grid,
                        cv=10)

# FITTING to the data set (due to cross-validation)
lnreg_cv.fit(X_train, y_train)

lnreg_cv

# %%
# printing the optimal parameters and best score
print("Tuned Parameters  :", lnreg_cv.best_params_)
print("Tuned CV AUC      :", lnreg_cv.best_score_.round(2))

# %%
# applying modelin scikit-learn

# INSTANTIATING a model object
lr = LinearRegression(**lnreg_cv.best_params_)  # INSTANTIATE


# FITTING to the training data
lr_fit = lr.fit(X_train, y_train)


# PREDICTING on new data
lr_pred = lr_fit.predict(X_test)


# SCORING the results
print('LinearRegression Training Score:', lr.score(X_train, y_train).round(4))
print('LinearRegression Testing Score:',  lr.score(X_test, y_test).round(4))
lr_train_score = lr.score(X_train, y_train)
lr_test_score = lr.score(X_test, y_test)

# %%
# %% [markdown]
# 6.3 Ridge Regression Model
# %%
# INSTANTIATING a model object
ridge_model = sklearn.linear_model.Ridge()

# FITTING the training data
ridge_fit = ridge_model.fit(X_train, y_train)


# PREDICTING on new data
ridge_pred = ridge_fit.predict(X_test)

# SCORING the results
print('Ridge Regression Training Score:',
      ridge_model.score(X_train, y_train).round(4))
print('Ridge Regression Testing Score:',
      ridge_model.score(X_test, y_test).round(4))


# saving scoring data for future use
ridge_train_score = ridge_model.score(X_train, y_train).round(4)
ridge_test_score = ridge_model.score(X_test, y_test).round(4)

# %% [markdown]
# 6.4 Lasso Regression Model
# INSTANTIATING a model object
lasso_model = sklearn.linear_model.Lasso()

# FITTING the training data
lasso_fit = lasso_model.fit(X_train, y_train)


# PREDICTING on new data
lasso_pred = lasso_fit.predict(X_test)

print('Lasso Regression Training Score:',
      lasso_model.score(X_train, y_train).round(4))
print('Lasso Regression Testing Score:',
      lasso_model.score(X_test, y_test).round(4))

# saving scoring data for future use
lasso_train_score = lasso_model.score(X_train, y_train).round(4)
lasso_test_score = lasso_model.score(X_test, y_test).round(4)

# %%
help(RandomForestRegressor)
# %% [markdown]
# 6.5 Random Forest Regression Model
# %%
########################################
# GridSearchCV
########################################

# declaring a hyperparameter space (give GridSearch some values to loop over)
C_space = pd.np.arange(10, 200, 10)
warm_start_space = [True, False]
bootstrap_space = [True, False]

# creating a hyperparameter grid
param_grid = {'n_estimators': C_space,  # inputting C values to loop over
              'warm_start': warm_start_space,
              'bootstrap': bootstrap_space
              }   # inputting warm start values to loop over

# INSTANTIATING the model object without hyperparameters
rfr_tuned = RandomForestRegressor(criterion='mse',
                                  random_state=222)


# GridSearchCV object
rfr_tuned_cv = GridSearchCV(estimator=rfr_tuned,
                            param_grid=param_grid,   # where are the values for the hyperparameters
                            cv=3)  # objective metric

# FITTING to the FULL DATASET (due to cross-validation)
rfr_tuned_cv.fit(X_scaled_df, diamonds_target)

# %%
# PREDICT step is not needed

# printing the optimal parameters and best score
print("Random Forest Regression Tuned Parameters  :", rfr_tuned_cv.best_params_)
print("Random Forest Regression Tuned CV AUC      :",
      rfr_tuned_cv.best_score_.round(4))


# %%
rfr = RandomForestRegressor(bootstrap=True, ccp_alpha=0.0,
                            criterion='mse', max_depth=None,
                            max_features='auto',
                            max_leaf_nodes=None,
                            max_samples=None,
                            min_impurity_decrease=0.0,
                            min_impurity_split=None,
                            min_samples_leaf=1,
                            min_samples_split=2,
                            min_weight_fraction_leaf=0.0,
                            n_estimators=100, n_jobs=None,
                            oob_score=False, random_state=222,
                            verbose=0, warm_start=False)
rfr.fit(X_train, y_train)
rfr_pred = rfr.predict(X_test)
print('Random Forest Regression Training Score:',
      rfr.score(X_train, y_train).round(4))
print('Random Forest Regression Testing Score:',
      rfr.score(X_test, y_test).round(4))
rfr_train_score = rfr.score(X_train, y_train)
rfr_test_score = rfr.score(X_test, y_test)

# %% [markdown]
# # 6.6 GBDT Regression Model
# %%
gbdt = GradientBoostingRegressor(max_depth=2,
                                 subsample=0.9,
                                 min_samples_leaf=0.009,
                                 max_features=0.9,
                                 n_estimators=65,
                                 random_state=222)
gbdt.fit(X_train, y_train)
gbdt_pred = gbdt.predict(X_test)
# SCORING the results
print('Training Score:', gbdt.score(X_train, y_train).round(4))
print('Testing Score:',  gbdt.score(X_test, y_test).round(4))
gbdt_train_score = gbdt.score(X_train, y_train)
gbdt_test_score = gbdt.score(X_test, y_test)


# %%
