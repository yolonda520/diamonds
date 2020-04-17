# %% [markdown]
# # 1. Import Packages
# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
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
diamonds.info()
# %% 
diamonds.describe()
# %% [markdown]
#  ## 2.2 Check data missing value
# %%
diamonds.isnull().sum()
# %% [markdown]
# It looks no missing value at all. However, let's check the x,y,z
# %%
diamonds.loc[:, 'x'][diamonds['x'] == 0]
# %%
diamonds.loc[diamonds['y']==0]
# %%
diamonds.loc[diamonds['z']==0]
# %%
# To simplify this data set a bit more, let's combine the 'x', 'y' and 'z' columns into a 'volume' column.
diamonds['volume'] = diamonds['x']*diamonds['y']*diamonds['z']
# %%
fill = diamonds['volume'].median()
diamonds = diamonds.replace(to_replace=0,value = fill)
diamonds.shape
# %%
diamonds.loc[diamonds['volume']==0]
# %% [markdown]
# ##2.3 One hot encoding categorical variables
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
# %%
diamond = diamonds.copy()
diamond.shape
# df_cut = pd.DataFrame[diamond['cut'].value_counts()]
# df_cut
df_cut = pd.DataFrame(diamond['cut'].value_counts())
df_cut = df_cut.reset_index()
df_cut

df_color = pd.DataFrame(diamond['color'].value_counts())
df_color = df_color.reset_index()
df_color

df_clarity = pd.DataFrame(diamond['clarity'].value_counts())
df_clarity = df_clarity.reset_index()
df_clarity
# %%
fig,axes = plt.subplots(nrows   =1,
                        ncols   =3,
                        figsize =(12,6))
df_cut.plot(kind = 'bar',
            x     = 'index', 
            y     = 'cut', 
            color = 'r',
            ax    = axes[0])
df_color.plot(kind = 'bar',
            x     = 'index', 
            y     = 'color', 
            color = 'y',
            ax    = axes[1])
df_clarity.plot(kind = 'bar',
            x     = 'index', 
            y     = 'clarity', 
            color = 'g',
            ax    = axes[2])
plt.show()
# %% [markdown]
# Insights: 
# 1.From the plot we can see the most popular type of cut is 'ideal';
# 2.the most popular color are:'G','E','F'
# 3.the most popular clarity are : ( SI2, SI1, VS2, VS1)

# %%
# one hot encoding categorical variables
one_hot_cut     = pd.get_dummies(diamonds['cut'])
one_hot_color   = pd.get_dummies(diamonds['color'])
one_hot_clarity = pd.get_dummies(diamonds['clarity'])
# %%
# dropping categorical variables after they've been encoded
diamonds = diamonds.drop('cut',axis =1)
diamonds = diamonds.drop('color',axis =1)
diamonds = diamonds.drop('clarity',axis =1)

# %%
# joining one hot encoding categorical variables together
diamonds = diamonds.join([one_hot_cut, one_hot_color, one_hot_clarity])

diamonds.head()

# %% [markdown]
# ##2.4 Develop Pearson correlation matrix with data.
# %%
df_corr = diamonds.corr().round(2)
print(df_corr.loc[:,'price'].sort_values(ascending = False))
# %% [markdown]
# # We can see that carat , x , y, z have high correlation with price.
#  %%
# diamonds_explanatory = diamonds.drop(['price','Unnamed: 0'],axis = 1)
# diamonds_target      = diamonds.loc[:,'price']
# # %% [markdown]
# # # Training and testing data split

# X_train, X_test, y_train, y_test = train_test_split(
#             diamonds_explanatory,
#             diamonds_target,
#             test_size = 0.25,          #always use 0.25
#             random_state = 222)

# # %%
# # Training set 
# print(X_train.shape)
# print(y_train.shape)

# # Testing set
# print(X_test.shape)
# print(y_test.shape)

# %% [markdown]
# ##2.5 Plot data
# %%
diamonds.columns
fig,ax = plt.subplots(nrows   =2,
                      ncols   =2,
                      figsize =(12,12))
sns.distplot(diamonds['carat'],
             bins  =  20,
             color = 'r',
             ax    = ax[0,0])
sns.distplot(diamonds['depth'],
             bins  =  20,
             color = 'g',
             ax    = ax[0,1])

sns.distplot(diamonds['table'],
             bins  =  20,
             color = 'y',
             ax    = ax[1,0])

sns.distplot(diamonds['price'],
             bins  =  20,
             color = 'b',
             ax    = ax[1,1])
plt.tight_layout()
plt.show()

fig,ax = plt.subplots(2,2)
sns.distplot(diamonds['x'],
             bins  =  20,
             color = 'r',
             ax    = ax[0,0])

sns.distplot(diamonds['y'],
             bins  =  20,
             color = 'g',
             ax    = ax[0,1])

sns.distplot(diamonds['z'],
             bins  =  20,
             color = 'y',
             ax    = ax[1,0])

sns.distplot(diamonds['volume'],
             bins  =  20,
             color = 'b',
             ax    = ax[1,1])
plt.tight_layout()
plt.show()

# %% [markdown]
# Insights:
# It looks like there are four peaks in carat plot: 0.5, 1, 1.5, 2
# The price less than 2000 are popular, however, there is a peak at 5000
