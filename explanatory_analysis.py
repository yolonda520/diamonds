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
# Insights:
# It looks worst color has the higher weight, so the color 1 much more expensive than color 5,which is make sense.
# We can see that the better cut,color, clarity diamonds always have light carat weight, base on carat weight is high related to price, so the price of high quality of cut,color, clarity is under the average.

