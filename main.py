# %% Imports
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import numpy as np

# %% Read the Dataset
pth = r'data/WineQT.csv'
df = pd.read_csv(pth, sep=',', header=0)

# %% Plot & Analyze the Dataset
# Check if DataFrame is read in correctly by looking at the first 10 rows
print(df.head(10))

describe = df.describe()
# Count -> no empty values

df.info()

#%%
# Dataset Balanced?
df['quality'].value_counts().plot.bar(rot=0)
plt.show()
# -> Not balanced
# Fix it?


# PCA
fig, ax = plt.subplots(6, 2, figsize = (10, 20), tight_layout=True)
df.hist(column= ["fixed acidity"], ax=ax[0][0], color='green')
df.hist(column= ['volatile acidity'], ax=ax[1][0], color='lightgreen')
df.hist(column= ["citric acid"], ax=ax[2][0], color='cornflowerblue')
df.hist(column= ['residual sugar'], ax=ax[3][0], color='blue')
df.hist(column= ['chlorides'], ax=ax[4][0], color='limegreen')
df.hist(column= ['free sulfur dioxide'], ax=ax[5][0], color='turquoise')
df.hist(column= ['total sulfur dioxide'], ax=ax[0][1], color='yellowgreen')
df.hist(column= ['density'], ax=ax[1][1], color='royalblue')
df.hist(column= ['pH'], ax=ax[2][1], color='mediumblue')
df.hist(column= ['sulphates'], ax=ax[3][1], color='skyblue')
df.hist(column= ['alcohol'], ax=ax[4][1], color='steelblue')
df.hist(column= ['quality'], ax=ax[5][1], color='teal')
plt.show()

# Plot outliers
fig, ax = plt.subplots(6, 2, figsize = (10, 30), tight_layout=True)
df.boxplot(column= ["fixed acidity"], ax=ax[0][0]).set_facecolor('ivory')
df.boxplot(column= ['volatile acidity'], ax=ax[1][0]).set_facecolor('ivory')
df.boxplot(column= ["citric acid"], ax=ax[2][0]).set_facecolor('ivory')
df.boxplot(column= ['residual sugar'], ax=ax[3][0]).set_facecolor('ivory')
df.boxplot(column= ['chlorides'], ax=ax[4][0]).set_facecolor('ivory')
df.boxplot(column= ['free sulfur dioxide'], ax=ax[5][0]).set_facecolor('ivory')
df.boxplot(column= ['total sulfur dioxide'], ax=ax[0][1]).set_facecolor('ivory')
df.boxplot(column= ['density'], ax=ax[1][1]).set_facecolor('ivory')
df.boxplot(column= ['pH'], ax=ax[2][1]).set_facecolor('ivory')
df.boxplot(column= ['sulphates'], ax=ax[3][1]).set_facecolor('ivory')
df.boxplot(column= ['alcohol'], ax=ax[4][1]).set_facecolor('ivory')
df.boxplot(column= ['quality'], ax=ax[5][1]).set_facecolor('ivory')

plt.show()

# Handle Outliers

# -> Viele gerinige values in total sulphate adfs -> viele weine rotweine
# rotwein + rose mostly


# %% Prepare the Dataset

# Split in train - test set
X = df.drop(['Id', 'quality'], axis=1)
y = df['quality']

+
# Standardize data
X_scaled = MinMaxScaler().fit_transform(X)



# Train Test Split



