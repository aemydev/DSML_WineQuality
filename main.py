# %% Imports
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, confusion_matrix, make_scorer
import numpy as np
from sklearn.model_selection import train_test_split, ParameterGrid
from collections import Counter
from sklearn import tree
from sklearn import datasets, metrics, model_selection, svm
import imblearn
from sklearn.preprocessing import StandardScaler
from sklearn.tree import export_text
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, confusion_matrix, make_scorer
import numpy as np
from sklearn.model_selection import train_test_split, ParameterGrid
from collections import Counter
from sklearn import tree
from sklearn import datasets, metrics, model_selection, svm
import imblearn
#import graphviz
from sklearn.tree import export_text
from sklearn.model_selection import GridSearchCV
import seaborn as sns


# %% ====== Explore the Dataset ======
# read the dataset
pth = r'data/WineQT.csv'
df = pd.read_csv(pth, sep=',', header=0)

# plot & analyze the dataset
print(df.head(10))

describe = df.describe()

df.info()

# is dataset balanced?
df['quality'].value_counts().plot.bar(rot=0)
plt.show()
# -> Not balanced

# barplot by feature
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

# plot outliers -> Boxplot
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


def plot_confusion_matrix(cm, title=""):
    ax = plt.axes()
    sns.set(font_scale=1.3)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="g", ax=ax, cmap="magma")

    ax.set_title(f'Confusion Matrix ({title})')
    ax.set_xlabel("Predicted label", fontsize=15)
    ax.set_xticklabels(list(labels))
    ax.set_ylabel("True label", fontsize=15)
    ax.set_yticklabels(list(labels), rotation=0)
    plt.show()


# train and evaluate the model
def build_evaluate_model(x_train, x_test, y_train, y_test, max_depth=None):
    clf = tree.DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(x_train, y_train)

    # plot the tree
    plt.figure(figsize=(30, 10), facecolor='white')
    tree.plot_tree(clf,
                   feature_names=feature_names,
                   class_names=labels,
                   rounded=True,
                   filled=True,
                   fontsize=14)
    plt.show()

    # tree as text
    # tree_rules = export_text(clf, feature_names=list(feature_names))
    # print(tree_rules)

    # confusion matrix
    plot_confusion_matrix(pd.DataFrame(metrics.confusion_matrix(y_train, clf.predict(x_train))), "train set")
    plot_confusion_matrix(pd.DataFrame(metrics.confusion_matrix(y_test, clf.predict(x_test))), "test set")

    # performance metrics
    print("train metrics")
    print(metrics.classification_report(y_train, clf.predict(x_train)))

    print("test metrics")
    print(metrics.classification_report(y_test, clf.predict(x_test)))

    # accuracy:
    # precision:
    # recall:
    # f1-score:
    # support:

    # feature importance
    importance = pd.DataFrame({'feature': X_train.columns,
                              'importance': np.round(clf.feature_importances_, 3)})
    importance.sort_values('importance', ascending=False, inplace=True)
    print(importance)

    # roc curve
    y_score = clf.predict_proba(x_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_score[:, 1])
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

    return clf


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
max_depth = build_evaluate_model(X_train, X_test, y_train, y_test).get_depth()

# Pre-prune -> tune hyperparameter max_depth
max_depth_grid_search = GridSearchCV(
        estimator=tree.DecisionTreeClassifier(random_state=42),
        scoring='f1',
        param_grid=ParameterGrid(
            {"max_depth": [[max_depth_] for max_depth_ in range(1, max_depth+1)]}))

max_depth_grid_search.fit(X_train, y_train)
print(max_depth_grid_search.best_params_['max_depth'])

build_evaluate_model(X_train, X_test, y_train, y_test, max_depth_grid_search.best_params_['max_depth'])

# %%
# 2. Undersample
undersample = RandomUnderSampler(sampling_strategy='majority')
X_train, y_train = undersample.fit_resample(X_train, y_train)
print(Counter(y_train))

max_depth = build_evaluate_model(X_train, X_test, y_train, y_test).get_depth()

# Pre-prune -> tune hyperparameter max_depth
max_depth_grid_search = GridSearchCV(
        estimator=tree.DecisionTreeClassifier(random_state=42),
        scoring='f1',
        param_grid=ParameterGrid(
            {"max_depth": [[max_depth_] for max_depth_ in range(1, max_depth+1)]}))

max_depth_grid_search.fit(X_train, y_train)
print(max_depth_grid_search.best_params_['max_depth'])

build_evaluate_model(X_train, X_test, y_train, y_test, max_depth_grid_search.best_params_['max_depth'])

#%%
# Oversample
oversample = RandomOverSampler(sampling_strategy='minority')
X_train, y_train = oversample.fit_resample(X_train, y_train)
print(Counter(y_train))

max_depth = build_evaluate_model(X_train, X_test, y_train, y_test).get_depth()

# Pre-prune -> tune hyperparameter max_depth
max_depth_grid_search = GridSearchCV(
        estimator=tree.DecisionTreeClassifier(random_state=42),
        scoring='f1', # other score?
        param_grid=ParameterGrid(
            {"max_depth": [[max_depth_] for max_depth_ in range(1, max_depth+1)]}))

max_depth_grid_search.fit(X_train, y_train)
print(max_depth_grid_search.best_params_['max_depth'])

build_evaluate_model(X_train, X_test, y_train, y_test, max_depth_grid_search.best_params_['max_depth'])


#%% Post-Pruning
# Post pruning
full_tree = build_evaluate_model(X_train, X_test, y_train, y_test)
ccp_alphas = full_tree.cost_complexity_pruning_path(X_train, y_train)["ccp_alphas"]
#ccp_alphas, impurities = path.ccp_alphas, path.impurities
print(ccp_alphas)

clfs = []
for ccp_alpha in ccp_alphas:
    clf = tree.DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)

clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]
node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
plt.scatter(ccp_alphas, node_counts)
plt.scatter(ccp_alphas, depth)
plt.plot(ccp_alphas, node_counts, label='no of nodes', drawstyle="steps-post")
plt.plot(ccp_alphas, depth, label="depth", drawstyle="steps-post")
plt.legend()
plt.show()

train_acc = []
test_acc = []

for c in clfs:
    y_train_pred = c.predict(X_train)
    y_test_pred = c.predict(X_test)
    train_acc.append(accuracy_score(y_train_pred, y_train))
    test_acc.append(accuracy_score(y_test_pred, y_test))

plt.scatter(ccp_alphas, train_acc)
plt.scatter(ccp_alphas, test_acc)
plt.plot(ccp_alphas, train_acc, label="train_accuracy", drawstyle="steps-post")
plt.plot(ccp_alphas, test_acc, label="train_accuracy", drawstyle="steps-post")
plt.legend()
plt.title("Accuaracy vs. alpha")
plt.show()

clf_ = tree.DecisionTreeClassifier(random_state=42, ccp_alpha=0.010)
clf_.fit(X_train, y_train)
y_train_pred = clf_.predict(X_train)
y_test_pred = clf_.predict(X_test)

print(f'Train score {accuracy_score(y_train_pred, y_train)}')
print(f'Test score {accuracy_score(y_test_pred, y_test)}')

plot_confusion_matrix(pd.DataFrame(metrics.confusion_matrix(y_train, y_train_pred)), "train set")
plot_confusion_matrix(pd.DataFrame(metrics.confusion_matrix(y_test, y_test_pred)), "test set")

# Conclusion: Model does not work very well
# We get many false positives and false negatives when making predictions on the training set
# Maybe splitting up the features 5 - 6 was not a good idea, since 5 and 6 are more similar than for e.g. 3 and 5
# PCA to prove this point:

#%%
from sklearn.decomposition import PCA

pth = r'data/WineQT.csv'
wine_data = pd.read_csv(pth, sep=',', header=0)
wine_data = wine_data.drop(columns=["Id"])
X = wine_data.drop(columns=["quality"])

X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X_scaled)
principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])

print(pca.explained_variance_ratio_)
scree_plot = pd.DataFrame(pca.explained_variance_ratio_).plot(kind="bar")
plt.show()

information_kept = sum(pca.explained_variance_ratio_)
print(information_kept.round(3))

# Add Targets back to Df
finalDf = pd.concat([principalDf, pd.DataFrame(y, columns=['target'])], axis=1)

# STEP 3 // plot in 2d
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('PC1', fontsize=12)
ax.set_ylabel('PC2', fontsize=12)

targets = wine_data["quality"].unique()
colors = ['yellow', 'orange', 'red', 'pink', 'blue', 'violet']

for target, color in zip(targets, colors):
    indices_to_keep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indices_to_keep, 'PC1'],
               finalDf.loc[indices_to_keep, 'PC2'],
               c=color,
               s=50)

ax.legend(targets)
plt.show()



