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
from sklearn.tree import export_text
from sklearn.model_selection import GridSearchCV

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


# export dt as pdf
def export_tree(clf):
    dot_data = tree.export_graphviz(clf,
                                    out_file=None,
                                    feature_names=wine_data.columns[0:-1],
                                    class_names=["bad", "good"], # labels correct? pls check
                                    filled=True, rounded=True,
                                    special_characters=True)

    graph = graphviz.Source(dot_data)
    #print(graph)
    graph.render('test.gv', view=True)


# train and evaluate the model
def build_test_model(x_train, x_test, y_train, _y_test, max_depth, min_samples_split):
    clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
    clf.fit(x_train, y_train)

    # plot the tree
    plt.figure(figsize=(30,10), facecolor='white')
    a = tree.plot_tree(clf,
                       feature_names=feature_names,
                       class_names=labels,
                       rounded=True,
                       filled=True,
                       fontsize=14)
    plt.show()

    # tree as text
    tree_rules = export_text(clf,
                             feature_names=list(feature_names))
    print(tree_rules)

    # make prediction
    y_pred = clf.predict(x_test)

    # confusion matrix
    import seaborn as sns
    confusion_matrix = pd.DataFrame(metrics.confusion_matrix(y_test, y_pred))

    ax = plt.axes()
    sns.set(font_scale=1.3)
    plt.figure(figsize=(10,7))
    sns.heatmap(confusion_matrix, annot=True, fmt="g", ax=ax, cmap="magma")

    ax.set_title('Confusion Matrix - Decision Tree')
    ax.set_xlabel("Predicted label", fontsize=15)
    ax.set_xticklabels(list(labels))
    ax.set_ylabel("True label", fontsize=15)
    ax.set_yticklabels(list(labels), rotation=0)

    plt.show()

    # performance metrics
    print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred))

    # precision:
    # recall:
    # f1-score:
    # support:

    # roc curve (plot)


    # feature importance
    importance = pd.DataFrame({'feature': X_train.columns,
                              'importance': np.round(clf.feature_importances_, 3)})
    importance.sort_values('importance', ascending=False, inplace=True)
    print(importance)

    # roc curve
    y_score = clf.predict_proba(x_test)
    fpr, tpr, thresholds = roc_curve(_y_test, y_score[:, 1])
    roc_auc = auc(fpr, tpr)

    ## plot roc
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


# find the best parameters
def improve_model(x_train, y_train):
    tuned_parameters = [{'max_depth': [1, 2, 3, 4, 5, 6, 7, 8],
                         'min_samples_split': [2, 4, 6, 8, 10, 12, 14, 16]}]
    scores = ['recall']

    for score in scores:
        print()
        print(f"Tuning hyperparameters for {score}")
        print()
        clf = GridSearchCV(
            tree.DecisionTreeClassifier(random_state=42), tuned_parameters, scoring=f'{score}_macro'
        )
        clf.fit(x_train, y_train)

        print("Best parameters set found on dev set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid  scores on dev set:")
        means = clf.cv_results_["mean_test_score"]
        stds = clf.cv_results_["std_test_score"]
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print(f"{mean:0.3f} (+/- {std*2:0.03f}) for {params}")


#%% ====== Data Preprocessing ======
# (Re)import the dataset
pth = r'data/WineQT.csv'
wine_data = pd.read_csv(pth, sep=',', header=0)
# print(wine_data.head())
# print(wine_data.tail())

# drop "Id" column -> not needed
wine_data = wine_data.drop(columns=["Id"])

# show that dataset is not balanced
wine_data['quality'].value_counts().plot.bar(rot=0)
plt.show()
print(wine_data['quality'].value_counts())

# balance dataset -> split into "good" and "bad" wine
wine_data["quality"] = wine_data["quality"].where(wine_data["quality"] > 5, 0)  # <= 5 -> 0 -> bad_wine (3, 4, 5)
wine_data["quality"] = wine_data["quality"].where(wine_data["quality"] < 6, 1)  # >=6 -> 1 -> good_wine (6, 7, 8)
# print(wine_data.head())
# print(wine_data.tail())

# plot again
wine_data['quality'].value_counts().plot.bar(rot=0)
plt.show()

# train test split
y = wine_data["quality"]
X = wine_data.drop(columns=["quality"])
feature_names = X.columns
# labels = y.unique()
# labels = labels.astype(str)
labels = ["bad", "good"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Training target statistics: {Counter(y_train)}")
print(f"Testing target statistics: {Counter(y_test)}")

#%% ====== Try different approaches to deal with not balanced dataset =====
# 1. Nothing -> Imbalanced Dataset
improve_model(X_train, y_train)
# -> {'max_depth': 5, 'min_samples_split': 2}
build_test_model(X_train, X_test, y_train, y_test, 5, 2)

#%%
# 2. Oversample Y_train, X_train
# we don't want to oversample the test data!
# Increase the number of samples of the smaller class up to the size of the biggest class
oversample = RandomOverSampler(sampling_strategy='minority')
X_over, y_over = oversample.fit_resample(X_train, y_train)
print(Counter(y_over))
improve_model(X_over, y_over)
build_test_model(X_over, X_test, y_over, y_test, 8, 2)

#%%
# 3. Undersampling Y_train, X_train using ?
# Decrease the number of samples of the bigger class down to the size of the biggest class
undersample = RandomUnderSampler(sampling_strategy='majority')
X_under, y_under = undersample.fit_resample(X_train, y_train)
print(Counter(y_under))
improve_model(X_over, y_over)
# -> {'max_depth': 4, 'min_samples_split': 2}
build_test_model(X_under, X_test, y_under, y_test, 4, 2)