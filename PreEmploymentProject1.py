import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier

# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.ensemble import GradientBoostingClassifier

# from fancyimpute import MICE
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# from sklearn.decomposition import KernelPCA

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
# from sklearn.ensemble.partial_dependence import plot_partial_dependence


# Data Processing
train_file = "exercise_03_train.csv"
test_file = "exercise_03_test.csv"
train_data = pd.read_csv(train_file, header=0)
test_data = pd.read_csv(test_file, header=0)

X_train = train_data.drop(['y'], axis=1)
X_test = test_data
X_train.x41 = X_train.x41.str.strip('$').astype('float')
X_train.x45 = X_train.x45.str.strip('%').astype('float')
X_test.x41 = X_test.x41.str.strip('$').astype('float')
X_test.x45 = X_test.x45.str.strip('%').astype('float')
Y_train = train_data.y


def build_estimator(estimator, data=X_train):
    numeric_features = data.select_dtypes(exclude=['object']).columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        # ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_features = data.select_dtypes(include=['object']).columns
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        # ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        # ("KernelPCA", KernelPCA()),
        ("PCA", PCA()),
        # ("estimator", estimator)
    ])
    return pipeline


# Model Evaluation and Testing
number_of_cv = 5
n_jobs = 4                      # This is set to 32 for the HPC cluster I used to run the code
print()

print("Fitting RandomForestClassifier...")
my_RandomForestClassifier = build_estimator(
    RandomForestClassifier(n_estimators=100, criterion='entropy', n_jobs=n_jobs))
RandomForestClassifier_score = cross_val_score(my_RandomForestClassifier,
                                               X_train, Y_train, cv=number_of_cv, n_jobs=n_jobs)
print("RandomForestClassificationReport:\n",
      classification_report(Y_train,
                            my_RandomForestClassifier.fit(X_train, Y_train).predict(X_train),
                            digits=16))
print("RandomForestClassifierMean:\t", RandomForestClassifier_score.mean(),
      "\nRandomForestClassifierStd:\t", RandomForestClassifier_score.std())

print("Fitting MLPEnsembleClassifier...")
my_MLPEnsembleClassifier = build_estimator(
    BaggingClassifier(
        MLPClassifier(alpha=1e-1, hidden_layer_sizes=(100, 10, 10, 2), max_iter=200)))
MLPEnsembleClassifier_score = cross_val_score(my_MLPEnsembleClassifier,
                                              X_train, Y_train, cv=number_of_cv, n_jobs=n_jobs)
print("MLPEnsembleClassifierReport:\n",
      classification_report(Y_train,
                            my_MLPEnsembleClassifier.fit(X_train, Y_train).predict(X_train),
                            digits=16))
print("MLPEnsembleClassifierMean:\t", MLPEnsembleClassifier_score.mean(),
      "\nMLPEnsembleClassifierStd:\t", MLPEnsembleClassifier_score.std())


# Predictions
model1_RandomForest = build_estimator(
    RandomForestClassifier(n_estimators=100, criterion='entropy', n_jobs=n_jobs),
    data=X_test
)
fitted_RandomForest = model1_RandomForest.fit(X_train, Y_train)
RandomForest_TestProb = fitted_RandomForest.predict_proba(X_test)
RandomForest_Predictions = fitted_RandomForest.predict(X_test)
np.savetxt("results1.csv", RandomForest_TestProb[:, 1].reshape(-1, 1), delimiter=",")

model2_MLP = build_estimator(
    BaggingClassifier(
        MLPClassifier(alpha=1e-1, hidden_layer_sizes=(100, 10, 10, 2), max_iter=200)),
    data=X_test
)
fitted_MLP = model2_MLP.fit(X_train, Y_train)
MLP_TestProb = fitted_MLP.predict_proba(X_test)
MLP_Predictions = fitted_MLP.predict(X_test)
np.savetxt("results2.csv", MLP_TestProb[:, 1].reshape(-1, 1), delimiter=",")


# Plotting (these plots didnt make it into the report due to time restrictions)
def create_plots(estimator, param_name, param_range, title):
    train_scores, test_scores = validation_curve(estimator, X_train, Y_train,
                                                 param_name,
                                                 param_range,
                                                 cv=5, n_jobs=n_jobs)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.figure()
    plt.title("Validation Curve")
    plt.xlabel("max_features")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()
    plt.savefig(title + "_fig1")

    ylim = (0.7, 1.01)
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    plt.figure()
    plt.title("Learning Curves")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X_train, Y_train, cv=cv, n_jobs=n_jobs, train_sizes=np.linspace(.1, 1.0, 5))
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    plt.show()
    plt.savefig(title + "_fig2")


# param_name = "estimator__max_features"
# param_range = np.linspace(2, 100, 1, endpoint=False, dtype=np.int8)
# create_plots(model1_RandomForest, param_name, param_range, title="model1")

# param_name = "estimator__alpha"
# param_range = np.logspace(1e-4, 100, 1, endpoint=False, dtype=np.int8)
# create_plots(model2_MLP, param_name, param_range, title="model2")


# Unused Models
# print("Fitting MLPClassifier...")
# my_MLPClassifier = build_estimator(
#     MLPClassifier(alpha=1e-4, hidden_layer_sizes=(100, 10, 10, 2), max_iter=200))
# MLPClassifier_score = cross_val_score(my_MLPClassifier,
#                                       X_train, Y_train, cv=number_of_cv, n_jobs=n_jobs)
# print("MLPClassificationReport:\n",
#       classification_report(Y_train, my_MLPClassifier.fit(X_train, Y_train).predict(X_train),
#                             digits=16))
# print("MLPClassifierMean:\t", MLPClassifier_score.mean(),
#       "\nMLPClassifierStd:\t", MLPClassifier_score.std())

# print("Fitting ExtraTreesClassifier...")
# my_ExtraTreesClassifier = ExtraTreesClassifier(n_estimators=100)
# ExtraTreesClassifier_score = cross_val_score(my_ExtraTreesClassifier,
#                                              X_train, Y_train, cv=number_of_cv)

# print("Fitting DecisionTreeClassifier...")
# my_DecisionTreeClassifier = build_estimator(
#     BaggingClassifier())
# DecisionTreeClassifier_score = cross_val_score(my_DecisionTreeClassifier,
#                                                X_train, Y_train, cv=number_of_cv)
# print("DecisionTreeClassifierMean:\t", DecisionTreeClassifier_score.mean(),
#       "\nDecisionTreeClassifierStd:\t", DecisionTreeClassifier_score.std())

# print("Fitting LogisticRegressionClassifier...")
# my_LogisiticRegressionClassifier = build_estimator(
#     LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0, n_jobs=n_jobs))
# my_LogisiticRegressionClassifier.fit(X_train, Y_train)
# LogisticRegressionClassifier_score = cross_val_score(my_LogisiticRegressionClassifier,
#                                                      X_train, Y_train, cv=number_of_cv,
#                                                      n_jobs=n_jobs)
# print("LogisticRegressionClassificationReport:\n",
#       classification_report(Y_train,
#                             my_LogisiticRegressionClassifier.fit(X_train, Y_train).predict(X_train),
#                             digits=16))
# print("LogisticRegressionMean:\t", LogisticRegressionClassifier_score.mean(),
#       "\nLogisticRegressionStd:\t", LogisticRegressionClassifier_score.std())

# print("Fitting LogisiticAdaBoostClassifier...")
# my_LogisiticAdaBoostClassifier = build_estimator(
#     AdaBoostClassifier(LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0, n_jobs=n_jobs)))
# LogisticAdaBoostClassifier_score = cross_val_score(my_LogisiticAdaBoostClassifier,
#                                                    X_train, Y_train, cv=number_of_cv,
#                                                    n_jobs=n_jobs)
# print("LogisticAdaBoostClassificationReport:\n",
#       classification_report(Y_train,
#                             my_LogisiticAdaBoostClassifier.fit(X_train, Y_train).predict(X_train),
#                             digits=16))
# print("LogisticAdaBoostClassifierMean:\t", LogisticAdaBoostClassifier_score.mean(),
#       "\nLogisticAdaBoostClassifierStd:\t", LogisticAdaBoostClassifier_score.std())
