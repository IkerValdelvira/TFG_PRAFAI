import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def logistic_regression(train_dev, file_name, path):

    y_train = train_dev["sigue_fa"]
    X_train = train_dev.drop("sigue_fa", 1)

    importance = [0] * len(X_train.columns)
    for i in range(10):
        # define the model
        model = LogisticRegression(max_iter=10000)
        # fit the model
        model.fit(X_train, y_train)
        # get importance (sum)
        importance = [x + y for x, y in zip(importance, model.coef_[0])]
    # get importance (average)
    importance = [x / 10 for x in importance]
    # get feature names
    feature_names = np.array(X_train.columns)
    """# summarize feature importance
    for i, v in enumerate(importance):
        print('Feature: %s, Score: %.5f' % (feature_names[i], v))"""
    # figure size
    figure(figsize=(20, 10))
    # plot feature importance
    plt.bar(feature_names, importance)
    # x labels rotation
    plt.xticks(rotation=90)
    # set title
    plt.title(file_name + " --> Logistic Regression Feature Importance")
    # set axis labels
    plt.xlabel('Features')
    plt.ylabel('Importance')
    # tight plot to layout
    plt.tight_layout()
    # save plot
    plt.savefig(os.path.join(path, "LogisticRegression_FI_" + file_name + ".png"))
    print("LogisticRegression feature importance plot saved in: " + os.path.join(path, "LogisticRegression_FI_" + file_name + ".png"))
    plt.close()


def cart_classification(train_dev, file_name, path):

    y_train = train_dev["sigue_fa"]
    X_train = train_dev.drop("sigue_fa", 1)

    importance = [0] * len(X_train.columns)
    for i in range(10):
        # define the model
        model = DecisionTreeClassifier()
        # fit the model
        model.fit(X_train, y_train)
        # get importance (sum)
        importance = [x + y for x, y in zip(importance, model.feature_importances_)]
    # get importance (average)
    importance = [x / 10 for x in importance]
    # get feature names
    feature_names = np.array(X_train.columns)
    """# summarize feature importance
    for i, v in enumerate(importance):
        print('Feature: %s, Score: %.5f' % (feature_names[i], v))"""
    # figure size
    figure(figsize=(20, 10))
    # plot feature importance
    plt.bar(feature_names, importance)
    # x labels rotation
    plt.xticks(rotation=90)
    # set title
    plt.title(file_name + " --> CART Classification Feature Importance")
    # set axis labels
    plt.xlabel('Features')
    plt.ylabel('Importance')
    # tight plot to layout
    plt.tight_layout()
    # save plot
    plt.savefig(os.path.join(path, "CART_FI_" + file_name + ".png"))
    print("CART feature importance plot saved in: " + os.path.join(path, "CART_FI_" + file_name + ".png"))
    plt.close()


def random_forest(train_dev, file_name, path):

    y_train = train_dev["sigue_fa"]
    X_train = train_dev.drop("sigue_fa", 1)

    importance = [0] * len(X_train.columns)
    for i in range(10):
        # define the model
        model = RandomForestClassifier(random_state=0)
        # fit the model
        model.fit(X_train, y_train)
        # get importance (sum)
        importance = [x + y for x, y in zip(importance, model.feature_importances_)]
    # get importance (average)
    importance = [x / 10 for x in importance]
    # get feature names
    feature_names = np.array(X_train.columns)
    """# summarize feature importance
    for i, v in enumerate(importance):
        print('Feature: %s, Score: %.5f' % (feature_names[i], v))"""
    # figure size
    figure(figsize=(20, 10))
    # plot feature importance
    plt.bar(feature_names, importance)
    # x labels rotation
    plt.xticks(rotation=90)
    # set title
    plt.title(file_name + " --> Random Forest Classification Feature Importance")
    # set axis labels
    plt.xlabel('Features')
    plt.ylabel('Importance')
    # tight plot to layout
    plt.tight_layout()
    # save plot
    plt.savefig(os.path.join(path, "RandomForest_FI_" + file_name + ".png"))
    print("RandomForest feature importance plot saved in: " + os.path.join(path, "RandomForest_FI_" + file_name + ".png"))
    plt.close()


def xgboost(train_dev, file_name, path):

    y_train = train_dev["sigue_fa"]
    X_train = train_dev.drop("sigue_fa", 1)

    importance = [0] * len(X_train.columns)
    for i in range(10):
        # define the model
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        # fit the model
        model.fit(X_train, y_train)
        # get importance (sum)
        importance = [x + y for x, y in zip(importance, model.feature_importances_)]
    # get importance (average)
    importance = [x / 10 for x in importance]
    # get feature names
    feature_names = np.array(X_train.columns)
    """# summarize feature importance
    for i, v in enumerate(importance):
        print('Feature: %s, Score: %.5f' % (feature_names[i], v))"""
    # figure size
    figure(figsize=(20, 10))
    # plot feature importance
    plt.bar(feature_names, importance)
    # x labels rotation
    plt.xticks(rotation=90)
    # set title
    plt.title(file_name + " --> XGBoost Classification Feature Importance")
    # set axis labels
    plt.xlabel('Features')
    plt.ylabel('Importance')
    # tight plot to layout
    plt.tight_layout()
    # save plot
    plt.savefig(os.path.join(path, "XGBoost_FI_" + file_name + ".png"))
    print("XGBoost feature importance plot saved in: " + os.path.join(path, "XGBoost_FI_" + file_name + ".png"))
    plt.close()


def permutation_knn(train_dev, file_name, path):

    y_train = train_dev["sigue_fa"]
    X_train = train_dev.drop("sigue_fa", 1)

    importance = [0] * len(X_train.columns)
    for i in range(10):
        # define the model
        model = KNeighborsClassifier()
        # fit the model
        model.fit(X_train, y_train)
        # perform permutation importance
        results = permutation_importance(model, X_train, y_train, scoring='accuracy')
        # get importance (sum)
        importance = [x + y for x, y in zip(importance, results.importances_mean)]
    # get importance (average)
    importance = [x / 10 for x in importance]
    # get feature names
    feature_names = np.array(X_train.columns)
    """# summarize feature importance
    for i, v in enumerate(importance):
        print('Feature: %s, Score: %.5f' % (feature_names[i], v))"""
    # figure size
    figure(figsize=(20, 10))
    # plot feature importance
    plt.bar(feature_names, importance)
    # x labels rotation
    plt.xticks(rotation=90)
    # set title
    plt.title(file_name + " --> KNN Permutation Feature Importance (Accuracy)")
    # set axis labels
    plt.xlabel('Features')
    plt.ylabel('Importance')
    # tight plot to layout
    plt.tight_layout()
    # save plot
    plt.savefig(os.path.join(path, "PermutationKNN(Accuracy)_FI_" + file_name + ".png"))
    print("PermutationKNN(Accuracy) feature importance plot saved in: " + os.path.join(path, "PermutationKNN(Accuracy)_FI_" + file_name + ".png"))
    plt.close()

    importance = [0] * len(X_train.columns)
    for i in range(10):
        # define the model
        model = KNeighborsClassifier()
        # fit the model
        model.fit(X_train, y_train)
        # perform permutation importance
        results = permutation_importance(model, X_train, y_train, scoring='roc_auc')
        # get importance (sum)
        importance = [x + y for x, y in zip(importance, results.importances_mean)]
    # get importance (average)
    importance = [x / 10 for x in importance]
    # get feature names
    feature_names = np.array(X_train.columns)
    """# summarize feature importance
    for i, v in enumerate(importance):
        print('Feature: %s, Score: %.5f' % (feature_names[i], v))"""
    # figure size
    figure(figsize=(20, 10))
    # plot feature importance
    plt.bar(feature_names, importance)
    # x labels rotation
    plt.xticks(rotation=90)
    # set title
    plt.title(file_name + " --> KNN Permutation Feature Importance (AUC)")
    # set axis labels
    plt.xlabel('Features')
    plt.ylabel('Importance')
    # tight plot to layout
    plt.tight_layout()
    # save plot
    plt.savefig(os.path.join(path, "PermutationAUC(Accuracy)_FI_" + file_name + ".png"))
    print("PermutationKNN(AUC) feature importance plot saved in: " + os.path.join(path, "PermutationKNN(AUC)_FI_" + file_name + ".png"))
    plt.close()


def permutation_random_forest(train_dev, file_name, path):

    y_train = train_dev["sigue_fa"]
    X_train = train_dev.drop("sigue_fa", 1)

    importance = [0] * len(X_train.columns)
    for i in range(10):
        # define the model
        model = RandomForestClassifier(random_state=0)
        # fit the model
        model.fit(X_train, y_train)
        # perform permutation importance
        results = permutation_importance(model, X_train, y_train, scoring='accuracy')
        # get importance (sum)
        importance = [x + y for x, y in zip(importance, results.importances_mean)]
    # get importance (average)
    importance = [x / 10 for x in importance]
    # get feature names
    feature_names = np.array(X_train.columns)
    """# summarize feature importance
    for i, v in enumerate(importance):
        print('Feature: %s, Score: %.5f' % (feature_names[i], v))"""
    # figure size
    figure(figsize=(20, 10))
    # plot feature importance
    plt.bar(feature_names, importance)
    # x labels rotation
    plt.xticks(rotation=90)
    # set title
    plt.title(file_name + " --> RandomForest Permutation Feature Importance (Accuracy)")
    # set axis labels
    plt.xlabel('Features')
    plt.ylabel('Importance')
    # tight plot to layout
    plt.tight_layout()
    # save plot
    plt.savefig(os.path.join(path, "PermutationRandomForest(Accuracy)_FI_" + file_name + ".png"))
    print("PermutationRandomForest(Accuracy) feature importance plot saved in: " + os.path.join(path, "PermutationRandomForest(Accuracy)_FI_" + file_name + ".png"))
    plt.close()

    importance = [0] * len(X_train.columns)
    for i in range(10):
        # define the model
        model = RandomForestClassifier(random_state=0)
        # fit the model
        model.fit(X_train, y_train)
        # perform permutation importance
        results = permutation_importance(model, X_train, y_train, scoring='roc_auc')
        # get importance (sum)
        importance = [x + y for x, y in zip(importance, results.importances_mean)]
    # get importance (average)
    importance = [x / 10 for x in importance]
    # get feature names
    feature_names = np.array(X_train.columns)
    """# summarize feature importance
    for i, v in enumerate(importance):
        print('Feature: %s, Score: %.5f' % (feature_names[i], v))"""
    # figure size
    figure(figsize=(20, 10))
    # plot feature importance
    plt.bar(feature_names, importance)
    # x labels rotation
    plt.xticks(rotation=90)
    # set title
    plt.title(file_name + " --> RandomForest Permutation Feature Importance (AUC)")
    # set axis labels
    plt.xlabel('Features')
    plt.ylabel('Importance')
    # tight plot to layout
    plt.tight_layout()
    # save plot
    plt.savefig(os.path.join(path, "PermutationRandomForest(AUC)_FI_" + file_name + ".png"))
    print("PermutationRandomForest(AUC) feature importance plot saved in: " + os.path.join(path, "PermutationRandomForest(AUC)_FI_" + file_name + ".png"))
    plt.close()


def heatmap(train_dev, file_name, path):
    # get correlations of each features in dataset
    corrmat = train_dev.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(55, 55))
    # plot heat map
    sns.heatmap(train_dev[top_corr_features].corr(), annot=True, cmap="RdYlGn", linewidths=.5, fmt= '.2f')
    plt.title(file_name + " --> Features Correlation Matrix Heatmap")
    plt.savefig(os.path.join(path, "Heatmap_" + file_name + ".png"))
    print("Heatmap of feature correlation matrix saved in: " + os.path.join(path,"Heatmap_" + file_name + ".png"))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to get feature importance of training set. Different techniques and classifiers are available for getting feature importance: LogisticRegression, CART, RandomForest, XGBoost, Permutation_RandomForest and Permutation_KNN. Heatmaps of feature correlation matrices are also available.')
    parser.add_argument("input_train", help="Path to specific training set (Train) or folder with multiple training sets.")
    parser.add_argument("-fi", "--fi_techniques", nargs='+', help="Feature importance techniques to use: LogisticRegression [LR], CART [CART], RandomForest [RF], XGBoost [XGB], Permutation_RandomForest [PRF] and/or Permutation_KNN [PKNN]. Multiple options are accepted. The [all] option will apply all techniques. Default option: [LR].", default=['LR'])
    parser.add_argument("-hm", "--heatmap", dest='heatmap', action='store_true', help="Use this flag if you want to create heatmap correlation matrices on training set.")
    parser.add_argument("-o", "--output_dir", help="Path to directory for created plots explaining feature importance. Default option: current directory.", default=os.getcwd())
    args = vars(parser.parse_args())
    input_train = args['input_train']
    fi_techniques = args['fi_techniques']
    output_dir = args['output_dir']
    for val in fi_techniques:
        if val not in ['all', 'LR', 'CART', 'RF', 'XGB', 'PRF', 'PKNN']:
            parser.error(
                "'--fi_technique' values must be [LR], [CART], [RF], [XGB], [PRF], [PKNN] and/or [all]. Multiple options are accepted.")

    trains = {}
    paths = {}

    if (os.path.isdir(input_train)):
        for file in os.listdir(input_train):
            if file.endswith(".csv"):
                train_name = os.path.splitext(file)[0]
                print('Reading training set from: ' + os.path.join(input_train, file))
                trains[train_name] = pd.read_csv(os.path.join(input_train, file), delimiter=";")

    elif (not os.path.isdir(input_train)):
        train_name = os.path.splitext(input_train)[0]
        train_name = Path(train_name).stem
        print('Reading training set from: ' + str(input_train))
        trains[train_name] = pd.read_csv(input_train, delimiter=";")

    else:
        parser.error("'input_train' and 'input_test' values must be paths to two files or two folders.")

    for i in range(len(trains.keys())):
        train_name = list(trains.keys())[i]
        if ('mv' in train_name):
            Path(os.path.join(output_dir, "MV")).mkdir(parents=True, exist_ok=True)
            if ('StandardScaler' in train_name):
                Path(os.path.join(output_dir, "MV", "StandardScaler")).mkdir(parents=True, exist_ok=True)
                paths[train_name] = os.path.join(output_dir, "MV", "StandardScaler")
            elif ('MinMaxScaler' in train_name):
                Path(os.path.join(output_dir, "MV", "MinMaxScaler")).mkdir(parents=True, exist_ok=True)
                paths[train_name] = os.path.join(output_dir, "MV", "MinMaxScaler")
            elif ('MaxAbsScaler' in train_name):
                Path(os.path.join(output_dir, "MV", "MaxAbsScaler")).mkdir(parents=True, exist_ok=True)
                paths[train_name] = os.path.join(output_dir, "MV", "MaxAbsScaler")
            elif ('QuantileTransformer' in train_name):
                Path(os.path.join(output_dir, "MV", "QuantileTransformer")).mkdir(parents=True, exist_ok=True)
                paths[train_name] = os.path.join(output_dir, "MV", "QuantileTransformer")
            elif ('Normalizer' in train_name):
                Path(os.path.join(output_dir, "MV", "Normalizer")).mkdir(parents=True, exist_ok=True)
                paths[train_name] = os.path.join(output_dir, "MV", "Normalizer")
            else:
                paths[train_name] = os.path.join(output_dir, "MV")
        elif('mean' in train_name):
            Path(os.path.join(output_dir, "MV_Mean")).mkdir(parents=True, exist_ok=True)
            if('StandardScaler' in train_name):
                Path(os.path.join(output_dir, "MV_Mean", "StandardScaler")).mkdir(parents=True, exist_ok=True)
                paths[train_name] = os.path.join(output_dir, "MV_Mean", "StandardScaler")
            elif ('MinMaxScaler' in train_name):
                Path(os.path.join(output_dir, "MV_Mean", "MinMaxScaler")).mkdir(parents=True, exist_ok=True)
                paths[train_name] = os.path.join(output_dir, "MV_Mean", "MinMaxScaler")
            elif ('MaxAbsScaler' in train_name):
                Path(os.path.join(output_dir, "MV_Mean", "MaxAbsScaler")).mkdir(parents=True, exist_ok=True)
                paths[train_name] = os.path.join(output_dir, "MV_Mean", "MaxAbsScaler")
            elif ('QuantileTransformer' in train_name):
                Path(os.path.join(output_dir, "MV_Mean", "QuantileTransformer")).mkdir(parents=True, exist_ok=True)
                paths[train_name] = os.path.join(output_dir, "MV_Mean", "QuantileTransformer")
            elif ('Normalizer' in train_name):
                Path(os.path.join(output_dir, "MV_Mean", "Normalizer")).mkdir(parents=True, exist_ok=True)
                paths[train_name] = os.path.join(output_dir, "MV_Mean", "Normalizer")
            else:
                paths[train_name] = os.path.join(output_dir, "MV_Mean")
        elif ('med' in train_name):
            Path(os.path.join(output_dir, "MV_Med")).mkdir(parents=True, exist_ok=True)
            if ('StandardScaler' in train_name):
                Path(os.path.join(output_dir, "MV_Med", "StandardScaler")).mkdir(parents=True, exist_ok=True)
                paths[train_name] = os.path.join(output_dir, "MV_Med", "StandardScaler")
            elif ('MinMaxScaler' in train_name):
                Path(os.path.join(output_dir, "MV_Med", "MinMaxScaler")).mkdir(parents=True, exist_ok=True)
                paths[train_name] = os.path.join(output_dir, "MV_Med", "MinMaxScaler")
            elif ('MaxAbsScaler' in train_name):
                Path(os.path.join(output_dir, "MV_Med", "MaxAbsScaler")).mkdir(parents=True, exist_ok=True)
                paths[train_name] = os.path.join(output_dir, "MV_Med", "MaxAbsScaler")
            elif ('QuantileTransformer' in train_name):
                Path(os.path.join(output_dir, "MV_Med", "QuantileTransformer")).mkdir(parents=True, exist_ok=True)
                paths[train_name] = os.path.join(output_dir, "MV_Med", "QuantileTransformer")
            elif ('Normalizer' in train_name):
                Path(os.path.join(output_dir, "MV_Med", "Normalizer")).mkdir(parents=True, exist_ok=True)
                paths[train_name] = os.path.join(output_dir, "MV_Med", "Normalizer")
            else:
                paths[train_name] = os.path.join(output_dir, "MV_Med")
        elif ('pred' in train_name):
            Path(os.path.join(output_dir, "MV_Pred")).mkdir(parents=True, exist_ok=True)
            if ('StandardScaler' in train_name):
                Path(os.path.join(output_dir, "MV_Pred", "StandardScaler")).mkdir(parents=True, exist_ok=True)
                paths[train_name] = os.path.join(output_dir, "MV_Pred", "StandardScaler")
            elif ('MinMaxScaler' in train_name):
                Path(os.path.join(output_dir, "MV_Pred", "MinMaxScaler")).mkdir(parents=True, exist_ok=True)
                paths[train_name] = os.path.join(output_dir, "MV_Pred", "MinMaxScaler")
            elif ('MaxAbsScaler' in train_name):
                Path(os.path.join(output_dir, "MV_Pred", "MaxAbsScaler")).mkdir(parents=True, exist_ok=True)
                paths[train_name] = os.path.join(output_dir, "MV_Pred", "MaxAbsScaler")
            elif ('QuantileTransformer' in train_name):
                Path(os.path.join(output_dir, "MV_Pred", "QuantileTransformer")).mkdir(parents=True, exist_ok=True)
                paths[train_name] = os.path.join(output_dir, "MV_Pred", "QuantileTransformer")
            elif ('Normalizer' in train_name):
                Path(os.path.join(output_dir, "MV_Pred", "Normalizer")).mkdir(parents=True, exist_ok=True)
                paths[train_name] = os.path.join(output_dir, "MV_Pred", "Normalizer")
            else:
                paths[train_name] = os.path.join(output_dir, "MV_Pred")
        elif ('diff' in train_name):
            Path(os.path.join(output_dir, "MV_Diff")).mkdir(parents=True, exist_ok=True)
            if ('StandardScaler' in train_name):
                Path(os.path.join(output_dir, "MV_Diff", "StandardScaler")).mkdir(parents=True, exist_ok=True)
                paths[train_name] = os.path.join(output_dir, "MV_Diff", "StandardScaler")
            elif ('MinMaxScaler' in train_name):
                Path(os.path.join(output_dir, "MV_Diff", "MinMaxScaler")).mkdir(parents=True, exist_ok=True)
                paths[train_name] = os.path.join(output_dir, "MV_Diff", "MinMaxScaler")
            elif ('MaxAbsScaler' in train_name):
                Path(os.path.join(output_dir, "MV_Diff", "MaxAbsScaler")).mkdir(parents=True, exist_ok=True)
                paths[train_name] = os.path.join(output_dir, "MV_Diff", "MaxAbsScaler")
            elif ('QuantileTransformer' in train_name):
                Path(os.path.join(output_dir, "MV_Diff", "QuantileTransformer")).mkdir(parents=True, exist_ok=True)
                paths[train_name] = os.path.join(output_dir, "MV_Diff", "QuantileTransformer")
            elif ('Normalizer' in train_name):
                Path(os.path.join(output_dir, "MV_Diff", "Normalizer")).mkdir(parents=True, exist_ok=True)
                paths[train_name] = os.path.join(output_dir, "MV_Diff", "Normalizer")
            else:
                paths[train_name] = os.path.join(output_dir, "MV_Diff")

        if 'all' in fi_techniques:
            if (not trains[train_name].isnull().values.any()):
                print('\n[' + train_name + '] Performing LogisticRegression feature importance...')
                logistic_regression(trains[train_name], train_name, paths[train_name])
                print('\n[' + train_name + '] Performing CART feature importance...')
                cart_classification(trains[train_name], train_name, paths[train_name])
                print('\n[' + train_name + '] Performing RandomForest feature importance...')
                random_forest(trains[train_name], train_name, paths[train_name])
                print('\n[' + train_name + '] Performing XGBoost feature importance...')
                xgboost(trains[train_name], train_name, paths[train_name])
                print('\n[' + train_name + '] Performing Permutation_RandomForest feature importance...')
                permutation_random_forest(trains[train_name], train_name, paths[train_name])
                print('\n[' + train_name + '] Performing Permutation_KNN feature importance...')
                permutation_knn(trains[train_name], train_name, paths[train_name])
            else:
                print('\n[' + train_name + "] LogisticRegression feature importance cannot be applied to dataset because it contains missing values.")
                print('\n[' + train_name + "] CART feature importance cannot be applied to dataset because it contains missing values.")
                print('\n[' + train_name + "] RandomForest feature importance cannot be applied to dataset because it contains missing values.")
                print('\n[' + train_name + "] Permutation_RandomForest feature importance cannot be applied to dataset because it contains missing values.")
                print('\n[' + train_name + "] Permutation_KNN feature importance cannot be applied to dataset because it contains missing values.")
                print('\n[' + train_name + '] Performing XGBoost feature importance...')
                xgboost(trains[train_name], train_name, paths[train_name])

        else:
            for val in fi_techniques:
                if (val == 'LR'):
                    if (not trains[train_name].isnull().values.any()):
                        print('\n[' + train_name + '] Performing LogisticRegression feature importance...')
                        logistic_regression(trains[train_name], train_name, paths[train_name])
                    else:
                        print('\n[' + train_name + "] LogisticRegression feature importance cannot be applied to dataset because it contains missing values.")
                elif (val == 'CART'):
                    if (not trains[train_name].isnull().values.any()):
                        print('\n[' + train_name + '] Performing CART feature importance...')
                        cart_classification(trains[train_name], train_name, paths[train_name])
                    else:
                        print('\n[' + train_name + "] CART feature importance cannot be applied to dataset because it contains missing values.")
                elif (val == 'RF'):
                    if (not trains[train_name].isnull().values.any()):
                        print('\n[' + train_name + '] Performing RandomForest feature importance...')
                        random_forest(trains[train_name], train_name, paths[train_name])
                    else:
                        print('\n[' + train_name + "] RandomForest feature importance cannot be applied to dataset because it contains missing values.")
                elif (val == 'XGB'):
                    print('\n[' + train_name + '] Performing XGBoost feature importance...')
                    xgboost(trains[train_name], train_name, paths[train_name])
                elif (val == 'PRF'):
                    if (not trains[train_name].isnull().values.any()):
                        print('\n[' + train_name + '] Performing Permutation_RandomForest feature importance...')
                        permutation_random_forest(trains[train_name], train_name, paths[train_name])
                    else:
                        print('\n[' + train_name + "] Permutation_RandomForest feature importance cannot be applied to dataset because it contains missing values.")
                elif (val == 'PKNN'):
                    if (not trains[train_name].isnull().values.any()):
                        print('\n[' + train_name + '] Performing Permutation_KNN feature importance...')
                        permutation_knn(trains[train_name], train_name, paths[train_name])
                    else:
                        print('\n[' + train_name + "] Permutation_KNN feature importance cannot be applied to dataset because it contains missing values.")

        if (args['heatmap']):
            print('\n[' + train_name + '] Creating heatmap of feature correlation matrix...')
            heatmap(trains[train_name], train_name, paths[train_name])