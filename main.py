# %% Imports
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn import tree
from sklearn import datasets, metrics, model_selection, svm
import imblearn
import graphviz

print(imblearn.__version__)

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


def export_tree(clf):
    dot_data = tree.export_graphviz(clf,
                                    out_file=None,
                                    feature_names=wine_data.columns[0:-1],
                                    class_names=["bad", "good"],
                                    filled=True, rounded=True,
                                    special_characters=True)

    graph = graphviz.Source(dot_data)
    print(graph)
    graph.render('test.gv', view=True)


# class weight? what does it do?
def build_test_model(_x_train, _x_test, _y_train, _y_test, max_depth=3):
    clf = tree.DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(_x_train, _y_train)

    # Evaluate the model
    y_pred = clf.predict(_x_test)
    print(classification_report(_y_test, y_pred))

    print("Test Accurracy: ", accuracy_score(_y_test, y_pred))
    print("Train Accurracy: ", accuracy_score(_y_train, clf.predict(_x_train))) # -> 1

    # roc curve
    y_score = clf.predict_proba(_x_test)
    fpr, tpr, thresholds = roc_curve(_y_test, y_score[:, 1])

    # area under curve
    roc_auc = auc(fpr, tpr)

    # Plot metrics
    metrics.plot_roc_curve(clf, _x_test, _y_test)
    plt.show()


    # plot the decision tree
    tree.plot_tree(clf)
    plt.show()

    # export_tree(clf)


# class weight? what does it do?
def build_test_model_gini(_x_train, _x_test, _y_train, _y_test):
    dt_gini = tree.DecisionTreeClassifier(criterion='gini')
    dt_gini.fit(_x_train, _y_train)

    # Evaluate the model
    y_pred = dt_gini.predict(_x_test)

    # Confusion Matrix
    cm = confusion_matrix(_y_test, y_pred)
    print(cm)

    # Classification Report
    print(classification_report(_y_test, y_pred))

    # roc curve
    y_score = dt_gini.predict_proba(_x_test)
    fpr, tpr, thresholds = roc_curve(_y_test, y_score[:, 1])
    roc_auc = auc(fpr, tpr)

    # plot roc
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # plot and export the decision tree
    tree.plot_tree(dt_gini)
    plt.show()
    export_tree(dt_gini)



#%%
# ======== Test the model ========
# 1. Nothing -> Imbalanced Dataset
build_test_model_gini(X_train, X_test, y_train, y_test)

#%%
# 2. Oversample Y_train, X_train using SMOTE
# we don't want to oversample the test data!
# Increase the number of samples of the smaller class up to the size of the biggest class
oversample = RandomOverSampler(sampling_strategy='minority')
X_over, y_over = oversample.fit_resample(X, y)
print(Counter(y_over))
#build_test_model(X_over, X_test, y_over, y_test, 9)
build_test_model_gini(X_over, X_test, y_over, y_test)

#%%
# 3. Undersampling Y_train, X_train using ?
# Decrease the number of samples of the bigger class down to the size of the biggest class
undersample = RandomUnderSampler(sampling_strategy='majority')
X_under, y_under = undersample.fit_resample(X, y)
print(Counter(y_under))
build_test_model(X_under, X_test, y_under, y_test,10)




# Prune the tree?
# Evaluate the Decision Tree (PCA, ROC, AUC?)
# https://imbalanced-learn.org/stable/over_sampling.html

# Does tree overfit?


# Gini as criterion