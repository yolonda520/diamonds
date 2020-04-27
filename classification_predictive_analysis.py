# %% [markdown]
# #  This report is going to build model that can predict the Ideal cut diamonds.

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

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.linear_model 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix    # confusion matrix

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler  # standard scaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
# %%
# setting pandas print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

diamonds = pd.read_csv('diamonds.csv')
display(diamonds.head())
display(diamonds.shape)

# %% [markdown]
# # 2. User-Defined Functions
# %%
########################################
# optimal_neighbors
########################################

def optimal_neighbors(X_data,
                      y_data,
                      standardize=True,
                      pct_test=0.25,
                      seed=222,
                      response_type='class',
                      max_neighbors=20,
                      show_viz=True):
    """
    Compute training and testing results for KNN.
    PARAMETERS
    ----------
    X_data        : explanatory variable data
    y_data        : response variable
    standardize   : whether or not to standardize the X data, default True
    pct_test      : test size for training and validation from (0,1), default 0.25
    seed          : random seed to be used in algorithm, default 802
    response_type : type of neighbors algorithm to use, default 'class'
        Use 'reg' for regression (KNeighborsRegressor)
        Use 'class' for classification (KNeighborsClassifier)
    max_neighbors : maximum number of neighbors in exhaustive search, default 100
    show_viz      : display or surpress k-neigbors visualization, default True

    """
    if standardize == True:
        scaler = StandardScaler()
        scaler.fit(X_data)
        X_scaled = scaler.transform(X_data)
        X_scaled_df = pd.DataFrame(X_scaled)
        X_data = X_scaled_df

    X_train, X_test, y_train, y_test = train_test_split(X_data,
                                                        y_data,
                                                        test_size=pct_test,
                                                        random_state=seed)

    training_accuracy = []
    test_accuracy = []

    neighbors_settings = range(1, max_neighbors + 1)

    for i_neighbors in neighbors_settings:
        if response_type == 'reg':
            clf = KNeighborsRegressor(n_neighbors=i_beighbors)
            clf.fit(X_train, y_train)
        elif response_type == 'class':
            clf = KNeighborsClassifier(n_neighbors=i_neighbors)
            clf.fit(X_train, y_train)
        else:
            print("Error: response_type must be 'reg' or 'class'")

        training_accuracy.append(clf.score(X_train, y_train))
        test_accuracy.append(clf.score(X_test, y_test))

    if show_viz == True:
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.plot(neighbors_settings, training_accuracy,
                 label="training accuracy")
        plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("n_neighbors")
        plt.legend()
        plt.show()

    print(
        f"The optimal number of neighbors is: {test_accuracy.index(max(test_accuracy))+1}")
    return test_accuracy.index(max(test_accuracy))+1


# %% [markdown]
# # 3.Features Engineering
# ## 3.1 Check data info
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
#  ## 3.2 Check data missing value and handle missing value
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
# ## 3.3 Check 'cut','color','clarity' categorical variables and change to number
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


#  %% [markdown]
# ## 3.4 One-hot encoding for categorial features <br>
# %%
a = pd.get_dummies(diamonds)
a.head()
# %% [markdown]
# ## 3.5 Drop useless variables to decrease noise of data
# %%
# To get a better result for machine learning, I will drop some columns that can be known by other columns.
diamonds = a.drop(['Unnamed: 0', 'volume', 'cut_Very Good',
                   'color_G', 'clarity_VVS2'], axis=1)
diamonds.head()
# %%
diamonds.shape
# %% [markdown]
# # 4. Predictive Models Preparation
# ## 4.1 Target Data and explanatory Data
# %%
diamonds_explanatory = diamonds.drop(['cut_Ideal'], axis=1)
diamonds_target = diamonds.loc[:, 'cut_Ideal']

# %% [markdown]
# ## 4.2 Distance Standardization <br>
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
# #  Checking pre- and post-scaling of the data
# print(f"""
# Dataset BEFORE Scaling
# ----------------------
# {pd.np.var(diamonds_explanatory)}


# Dataset AFTER Scaling
# ----------------------
# {pd.np.var(X_scaled_df)}
# """)
# %% [markdown]
# ## 4.3 Training and testing data split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df,
    diamonds_target,
    test_size=0.25,  
    random_state=222)

# Training set
print(X_train.shape)
print(y_train.shape)

# Testing set
print(X_test.shape)
print(y_test.shape)

# %% [markdown]
# # 5. Model Building
# %%
# creating an empty DataFrame
model_performance = pd.DataFrame(columns=['Model',
                                          'Training Accuracy',
                                          'Testing Accuracy',
                                          'AUC Value'
                                          ])
model_performance

# %% [markdown]
# ## 5.1 KNN Classification Model
# %%
opt_neighbors = optimal_neighbors(diamonds_explanatory,
                                  diamonds_target,
                                  standardize=True,
                                  pct_test=0.25,
                                  seed=222,
                                  response_type='class',
                                  max_neighbors=10,
                                  show_viz=True)
# %%
# INSTANTIATING a model with the optimal number of neighbors
knn_opt = KNeighborsClassifier(n_neighbors=opt_neighbors)


# FITTING the model based on the training data
knn_fit = knn_opt.fit(X_train, y_train)


# PREDITCING on new data
knn_opt_pred = knn_fit.predict(X_test)


# saving scoring data for future use
knn_opt_score_train = knn_opt.score(X_train, y_train).round(4)
knn_opt_score_test = knn_opt.score(X_test, y_test).round(4)
knn_auc = roc_auc_score(y_true=y_test,
                        y_score=knn_opt_pred).round(4)
# SCORING the results
print('Training Score:', knn_opt_score_train)
print('Testing Score :', knn_opt_score_test)
print('AUC Score     :', knn_auc)

# display(y_test)
# display(knn_opt_pred.reshape(-1, 1))
# %%
# appending to model_performance
model_performance = model_performance.append(
    {'Model': 'Opt KNN Classifier',
     'Training Accuracy': knn_opt_score_train,
     'Testing Accuracy': knn_opt_score_test,
     'AUC Value': knn_auc
     },
    ignore_index=True)


# checking the results
model_performance
# %% [markdown]
# ## 5.2 Logistic Regression on Hyperparameter Tuning with GridSearchCV

# %%
# # GridSearchCV
# I commented the process as it took a long time to run, I just record the results.   
# ########################################

# # declaring a hyperparameter space (give GridSearch some values to loop over)
# C_space = pd.np.arange(0.1, 3.0, 0.1)
# warm_start_space = [True, False]


# # creating a hyperparameter grid
# param_grid = {'C': C_space,  # inputting C values to loop over
#               'warm_start': warm_start_space}   # inputting warm start values to loop over


# # INSTANTIATING the model object without hyperparameters
# lr_tuned = LogisticRegression(solver='lbfgs',
#                               max_iter=1000,  # more iterations so model converges
#                               random_state=222)


# # GridSearchCV object
# lr_tuned_cv = GridSearchCV(estimator=lr_tuned,   # which model type to use (i.e. estimator)
#                            param_grid=param_grid,   # where are the values for the hyperparameters
#                            cv=3,             # how many test should GridSearch do?
#                            scoring=make_scorer(roc_auc_score,
#                                                needs_threshold=False))  # objective metric


# # FITTING to the FULL DATASET (due to cross-validation)
# lr_tuned_cv.fit(diamonds_explanatory, diamonds_target)

# # printing the optimal parameters and best score
# print("diamonds Tuned Parameters  :", lr_tuned_cv.best_params_)
# print("diamonds Tuned CV AUC      :", lr_tuned_cv.best_score_.round(4))
# %%
# Record the result of LogisticRegression Model best parameters
print(f"""
/--------------------------\\
|LogisticRegression Model |
\\--------------------------/

Diamonds best parameters and best AUC score:
-----------
Tuned Parameters  : 
'C': 1.4000000000000001, 'warm_start': True

Tuned CV AUC      : 
0.8464

""")
# %%
# building a model based on hyperparameter tuning results

# INSTANTIATING a logistic regression model with tuned values
lr_tuned = LogisticRegression(solver='lbfgs',
                              C=1.4000000000000001,
                              warm_start=True,
                              max_iter=1000,  # more iterations so model converges
                              random_state=222)


# FIT step is not needed if we did GridSearchCV already fit, but I will do here as I change the GridSearchCV every model and only use record.
lr_tuned.fit(X_train, y_train)

# PREDICTING based on the testing set
lr_tuned_pred = lr_tuned.predict(X_test)

# declaring model performance objects
lr_train_acc = lr_tuned.score(X_train, y_train).round(4)
lr_test_acc = lr_tuned.score(X_test, y_test).round(4)
lr_auc = roc_auc_score(y_true=y_test,
                       y_score=lr_tuned_pred).round(4)

# SCORING the results
print('LogisticRegression Training ACCURACY:', lr_train_acc)
print('LogisticRegression Testing  ACCURACY:', lr_test_acc)
print('LogisticRegression AUC Score        :', lr_auc)

# %%
# appending to model_performance
model_performance = model_performance.append(
    {'Model': 'Tuned LogisticRegression',
     'Training Accuracy': lr_train_acc,
     'Testing Accuracy': lr_test_acc,
     'AUC Value': lr_auc},
    ignore_index=True)


# checking the results
model_performance

# %% [markdown]
# ## 5.3 CART on Hyperparameter Tuning with GridSearchCV
# %%
# ########################################
# # GridSearchCV
# ########################################

# # declaring a hyperparameter space
# criterion_space = ['gini', 'entropy']
# splitter_space = ['best', 'random']
# depth_space = pd.np.arange(1, 25)
# leaf_space = pd.np.arange(1, 100)


# # creating a hyperparameter grid
# param_grid = {'criterion': criterion_space,
#               'splitter': splitter_space,
#               'max_depth': depth_space,
#               'min_samples_leaf': leaf_space}


# # INSTANTIATING the model object without hyperparameters
# tuned_tree = DecisionTreeClassifier(random_state=222)


# # GridSearchCV object
# tuned_tree_cv = GridSearchCV(estimator=tuned_tree,
#                              param_grid=param_grid,
#                              cv=3,
#                              scoring=make_scorer(roc_auc_score,
#                                                  needs_threshold=False))


# # FITTING to the FULL DATASET (due to cross-validation)
# #tuned_tree_cv.fit(chef_full, chef_target)
# tuned_tree_cv.fit(diamonds_explanatory, diamonds_target)

# # PREDICT step is not needed


# # printing the optimal parameters and best score
# print("Tuned Parameters  :", tuned_tree_cv.best_params_)
# print("Tuned Tree_cv AUC:", tuned_tree_cv.best_score_.round(4))
# %%
# Record the result of LogisticRegression Model best parameters
print(f"""
/--------------------------\\
|CART Model |
\\--------------------------/

Best parameters and best AUC score:
-----------
Tuned Parameters  : 
'criterion': 'gini', 'max_depth': 12, 'min_samples_leaf': 22, 'splitter': 'random'

Tuned CV AUC      : 
0.8931

""")
# %%
# building a model based on hyperparameter tuning results

# INSTANTIATING a logistic regression model with tuned values
tree_tuned = DecisionTreeClassifier(criterion='gini',
                                    random_state=222,
                                    max_depth=12,
                                    min_samples_leaf=22,
                                    splitter='random')


# FIT step is not needed if we did GridSearchCV already fit, but I will do here as I change the GridSearchCV every model and only use record.
tree_tuned.fit(X_train, y_train)

# PREDICTING based on the testing set
tree_tuned_pred = tree_tuned.predict(X_test)

# declaring model performance objects
tree_train_acc = tree_tuned.score(X_train, y_train).round(4)
tree_test_acc = tree_tuned.score(X_test, y_test).round(4)
tree_auc = roc_auc_score(y_true=y_test,
                         y_score=tree_tuned_pred).round(4)
# SCORING the results
print('Training ACCURACY:', tree_train_acc)
print('Testing  ACCURACY:', tree_test_acc)
print('AUC Score        :', tree_auc)
# %%
# appending to model_performance
model_performance = model_performance.append(
    {'Model': 'Tuned Tree(CART)',
     'Training Accuracy': tree_train_acc,
     'Testing Accuracy': tree_test_acc,
     'AUC Value': tree_auc},
    ignore_index=True)


# checking the results
model_performance

# %% [markdown]
# ## 5.4 Random Forest Classification on Hyperparameter Tuning with GridSearchCV
# %%
# ########################################
# # GridSearchCV
# ########################################

# declaring a hyperparameter space
estimator_space = pd.np.arange(10, 500, 20)
leaf_space = pd.np.arange(1, 31, 10)
criterion_space = ['gini', 'entropy']
bootstrap_space = [True, False]
warm_start_space = [True, False]


# creating a hyperparameter grid
param_grid = {'n_estimators': estimator_space,
              'min_samples_leaf': leaf_space,
              'criterion': criterion_space,
              'bootstrap': bootstrap_space,
              'warm_start': warm_start_space}


# INSTANTIATING the model object without hyperparameters
forest_grid = RandomForestClassifier(random_state=222)


# GridSearchCV object
forest_cv = GridSearchCV(estimator=forest_grid,
                         param_grid=param_grid,
                         cv=3,
                         scoring=make_scorer(roc_auc_score,
                                             needs_threshold=False))


# FITTING to the FULL DATASET (due to cross-validation)
forest_cv.fit(diamonds_explanatory, diamonds_target)

# printing the optimal parameters and best score
print("Tuned Parameters  :", full_forest_cv.best_params_)
print("Tuned Training AUC:", full_forest_cv.best_score_.round(4))
# %%
# Record the result of LogisticRegression Model best parameters
print(f"""
/--------------------------\\
|RandomForestClassifier Model |
\\--------------------------/

Best parameters and best AUC score:
-----------
Tuned Parameters  : 
'bootstrap': False, 'criterion': 'gini', 'min_samples_leaf': 1, 'n_estimators': 100, 'warm_start': True

chef_full Tuned AUC      : 
 0.6363

""")
# %%
# INSTANTIATING the model object with tuned values
rf_tuned = RandomForestClassifier(bootstrap=False,
                                  criterion='gini',
                                  min_samples_leaf=1,
                                  n_estimators=100,
                                  warm_start=True,
                                  random_state=222)


# FIT step is needed as we are not using .best_estimator
rf_tuned_fit = rf_tuned.fit(X_train, y_train)


# PREDICTING based on the testing set
rf_tuned_pred = rf_tuned_fit.predict(X_test)

rf_train_acc = rf_tuned_fit.score(X_train, y_train).round(4)
rf_test_acc = rf_tuned_fit.score(X_test, y_test).round(4)
rf_auc = roc_auc_score(y_true=y_test,
                       y_score=rf_tuned_pred).round(4)

# SCORING the results
print('rf Training ACCURACY:', rf_train_acc)
print('rf Testing  ACCURACY:', rf_test_acc)
print('rf AUC Score        :', rf_auc)
# %%
# appending to model_performance
model_performance = model_performance.append(
    {'Model': 'Tuned Random Forest',
     'Training Accuracy': rf_train_acc,
     'Testing Accuracy': rf_test_acc,
     'AUC Value': rf_auc},
    ignore_index=True)


# checking the results
model_performance
# %% [markdown]
# ## 5.5 Gradient Boosted Machines Classification on Hyperparameter Tuning with GridSearchCV
# %%
########################################
# GridSearchCV
########################################

# declaring a hyperparameter space
learn_space = pd.np.arange(0.1, 1.1, 0.1)
estimator_space = pd.np.arange(50, 150, 50)
depth_space = pd.np.arange(1, 10)
subsample_space = pd.np.arange(0.2, 1, 0.1)

# creating a hyperparameter grid
param_grid = {'learning_rate': learn_space,
              'max_depth': depth_space,
              'n_estimators': estimator_space,
              'subsample': subsample_space}


# INSTANTIATING the model object without hyperparameters
gbm_grid = GradientBoostingClassifier(random_state=222)


# GridSearchCV object
gbm_cv = GridSearchCV(estimator=gbm_grid,
                      param_grid=param_grid,
                      cv=3,
                      scoring=make_scorer(roc_auc_score,
                                          needs_threshold=False))


# FITTING to the FULL DATASET (due to cross-validation)
gbm_cv.fit(diamonds_explanatory, diamonds_target)

# printing the optimal parameters and best score
print("Tuned Parameters  :", full_gbm_cv.best_params_)
print("Tuned Training AUC:", full_gbm_cv.best_score_.round(4))
# %%
# Record the result of LogisticRegression Model best parameters
print("""
/--------------------------\\
|Gradient Boosted Machines Model |
\\--------------------------/

Best parameters and best AUC score:
-----------
Tuned Parameters  : 
'learning_rate': 1.0000000000000002, 'max_depth': 1, 'n_estimators': 50, 'subsample': 0.5

Tuned AUC      : 
0.655

""")
# %%
# INSTANTIATING the model object without hyperparameters
gbm_tuned = GradientBoostingClassifier(learning_rate=0.2,
                                       max_depth=1,
                                       n_estimators=50,
                                       subsample=0.5,
                                       random_state=222)


# FIT step is needed as we are not using .best_estimator
gbm_tuned_fit = gbm_tuned.fit(X_train, y_train)


# PREDICTING based on the testing set
gbm_tuned_pred = gbm_tuned_fit.predict(X_test)

gbm_train_acc = gbm_tuned_fit.score(X_train, y_train).round(4)
gbm_test_acc = gbm_tuned_fit.score(X_test, y_test).round(4)
gbm_auc = roc_auc_score(y_true=y_test,
                        y_score=gbm_tuned_pred).round(4)
# SCORING the results
print('Training ACCURACY:', gbm_train_acc)
print('Testing  ACCURACY:', gbm_test_acc)
print('AUC Score        :', gbm_auc)
# %%
# appending to model_performance
model_performance = model_performance.append(
    {'Model': 'Tuned GBM',
     'Training Accuracy': gbm_train_acc,
     'Testing Accuracy': gbm_test_acc,
     'AUC Value': gbm_auc},
    ignore_index=True)


# checking the results
model_performance
# %% [markdown]
# ## 5.6 Support Vector Machine Classification on Hyperparameter Tuning with GridSearchCV
# %%
########################################
# GridSearchCV
########################################

# declaring a hyperparameter space
C_space = [1e-3, 1e-2, 1e-1, 0.8, 1, 10, 100]  # pd.np.arange(0.1,10,0.1)
gamma_space = [0.5, 0.1, 0.01, 0.001, 0.0001]
kernel_space = ['linear', 'rbf']
# depth_space     = pd.np.arange(1, 10)
# subsample_space = pd.np.arange(0.5,1,0.1)

# creating a hyperparameter grid
param_grid = {'C': C_space,
              'gamma': gamma_space,
              'kernel': kernel_space}


# INSTANTIATING the model object without hyperparameters
svc_grid = SVC(random_state=222)


# GridSearchCV object
svc_cv = GridSearchCV(estimator=svc_grid,
                      param_grid=param_grid,
                      cv=3,
                      n_jobs=8,
                      verbose=1,
                      scoring=make_scorer(roc_auc_score,
                                          needs_threshold=False))


# FITTING to the FULL DATASET (due to cross-validation)
svc_cv.fit(diamonds_explanatory, diamonds_target)

# printing the optimal parameters and best score
print("Tuned Parameters  :", svc_cv.best_params_)
print("Tuned Training AUC:", svc_cv.best_score_.round(4))
# %%
# Record the result of LogisticRegression Model best parameters
print(f"""
/--------------------------\\
|Support Vector Machine Model |
\\--------------------------/

Full explanatory variables best parameters and best AUC score:
-----------
 Tuned Parameters  : 
'C': 100, 'gamma': 0.01, 'kernel': 'rbf'

Tuned AUC      : 
 0.6279

""")
# %%
# INSTANTIATING the model object with tuned values
svc_tuned = SVC(C=0.01,
                gamma=1,
                kernel='linear',
                random_state=222)


# FIT step is needed as we are not using .best_estimator
svc_tuned_fit = svc_tuned.fit(X_train, y_train)


# PREDICTING based on the testing set
svc_tuned_pred = svc_tuned_fit.predict(X_test)

svc_train_acc = svc_tuned_fit.score(X_train, y_train).round(4)
svc_test_acc = svc_tuned_fit.score(X_test, y_test).round(4)
svc_auc = roc_auc_score(y_true=y_test,
                        y_score=svc_tuned_pred).round(4)

# SCORING the results
print('Training ACCURACY:', svc_train_acc)
print('Testing  ACCURACY:', svc_test_acc)
print('AUC Score        :', svc_auc)
# %%
# appending to model_performance
model_performance = model_performance.append(
    {'Model': 'Tuned SVM',
     'Training Accuracy': svc_train_acc,
     'Testing Accuracy': svc_test_acc,
     'AUC Value': svc_auc},
    ignore_index=True)


# checking the results
model_performance.round(3)

# %% [markdown]
# # 7. Compare Results
# %%
# comparing results

model_performance.round(3)
