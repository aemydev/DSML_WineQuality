# %% Imports
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn import tree

# %% Read the Dataset
pth = r'data/WineQT.csv'
df = pd.read_csv(pth, sep=',', header=0)

# Plot & Analyze the Dataset
# Check if DataFrame is read in correctly by looking at the first 10 rows
print(df.head(10))

describe = df.describe()
# Count -> no empty values

df.info()

# Dataset Balanced?
df['quality'].value_counts().plot.bar(rot=0)
plt.show()
# -> Not balanced

# Barplot by feature
fig, ax = plt.subplots(6, 2, figsize=(10, 20), tight_layout=True)
df.hist(column=["fixed acidity"], ax=ax[0][0], color='green')
df.hist(column=['volatile acidity'], ax=ax[1][0], color='lightgreen')
df.hist(column=["citric acid"], ax=ax[2][0], color='cornflowerblue')
df.hist(column=['residual sugar'], ax=ax[3][0], color='blue')
df.hist(column=['chlorides'], ax=ax[4][0], color='limegreen')
df.hist(column=['free sulfur dioxide'], ax=ax[5][0], color='turquoise')
df.hist(column=['total sulfur dioxide'], ax=ax[0][1], color='yellowgreen')
df.hist(column=['density'], ax=ax[1][1], color='royalblue')
df.hist(column=['pH'], ax=ax[2][1], color='mediumblue')
df.hist(column=['sulphates'], ax=ax[3][1], color='skyblue')
df.hist(column=['alcohol'], ax=ax[4][1], color='steelblue')
df.hist(column=['quality'], ax=ax[5][1], color='teal')
plt.show()

# Plot outliers -> Boxplot
fig, ax = plt.subplots(6, 2, figsize=(10, 30), tight_layout=True)
df.boxplot(column=["fixed acidity"], ax=ax[0][0]).set_facecolor('ivory')
df.boxplot(column=['volatile acidity'], ax=ax[1][0]).set_facecolor('ivory')
df.boxplot(column=["citric acid"], ax=ax[2][0]).set_facecolor('ivory')
df.boxplot(column=['residual sugar'], ax=ax[3][0]).set_facecolor('ivory')
df.boxplot(column=['chlorides'], ax=ax[4][0]).set_facecolor('ivory')
df.boxplot(column=['free sulfur dioxide'], ax=ax[5][0]).set_facecolor('ivory')
df.boxplot(column=['total sulfur dioxide'], ax=ax[0][1]).set_facecolor('ivory')
df.boxplot(column=['density'], ax=ax[1][1]).set_facecolor('ivory')
df.boxplot(column=['pH'], ax=ax[2][1]).set_facecolor('ivory')
df.boxplot(column=['sulphates'], ax=ax[3][1]).set_facecolor('ivory')
df.boxplot(column=['alcohol'], ax=ax[4][1]).set_facecolor('ivory')
df.boxplot(column=['quality'], ax=ax[5][1]).set_facecolor('ivory')
plt.show()

# %% Approach 1: Binary Classification (Ines)
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

# Read the dataset
pth = r'data/WineQT.csv'
wine_data = pd.read_csv(pth, sep=',', header=0)
print(wine_data.head())
print(wine_data.tail())

# Drop Id column -> not needed
wine_data = wine_data.drop(columns=["Id"])

# Show that dataset is not balanced
wine_data['quality'].value_counts().plot.bar(rot=0)
plt.show()

print(df['quality'].value_counts())

# Balance dataset
wine_data["quality"] = wine_data["quality"].where(wine_data["quality"] > 5, 0)  # <= 5 -> 0 -> bad_wine (3, 4, 5)
wine_data["quality"] = wine_data["quality"].where(wine_data["quality"] < 6, 1)  # >=6 -> 1 -> good_wine (6, 7, 8)
print(wine_data.head())
print(wine_data.tail())

# Plot again
wine_data['quality'].value_counts().plot.bar(rot=0)
plt.show()

# Train test split
y = wine_data["quality"]
X = wine_data.drop(columns=["quality"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

print(f"Training target statistics: {Counter(y_train)}")
print(f"Testing target statistics: {Counter(y_test)}")


def build_test_model(_x_train, _x_test, _y_train, _y_test, class_weight=None):
    if class_weight:
        clf = tree.DecisionTreeClassifier(class_weight=class_weight)
    else:
        clf = tree.DecisionTreeClassifier()

    clf.fit(_x_train, _y_train)

    y_pred = clf.predict(_x_test)
    print(classification_report(_y_test, y_pred))

    # plot the decision tree -> do better -> not readable
    tree.plot_tree(clf)
    plt.show()


# ======== Test the model ========
# 1. Nothing -> Imbalanced Dataset
build_test_model(X_train, X_test, y_train, y_test)


# 2. Oversample Y_train, X_train using SMOTE
# we don't want to oversample the test data!
# Increase the number of samples of the smaller class up to the size of the biggest class

build_test_model(X_train, X_test, y_train, y_test)

# 3. Undersampling Y_train, X_train using ?
# Decrease the number of samples of the bigger class down to the size of the biggest class


build_test_model(X_train, X_test, y_train, y_test)


# Tune decision tree?
# Evaluate the Decision Tree