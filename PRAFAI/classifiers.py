import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from PRAFAI.Classifiers.adaboost import AdaBoost
from PRAFAI.Classifiers.bagging import Bagging
from PRAFAI.Classifiers.decisiontree import DecisionTree
from PRAFAI.Classifiers.knn import KNN
from PRAFAI.Classifiers.lightgbm_classifier import LightGBM
from PRAFAI.Classifiers.mlp import MLP
from PRAFAI.Classifiers.naivebayes import NaiveBayes
from PRAFAI.Classifiers.perceptron import Perceptron
from PRAFAI.Classifiers.randomforest import RandomForest
from PRAFAI.Classifiers.svm import SVM
from PRAFAI.Classifiers.xgboost_classifier import XGBoost

import argparse
import os
from pathlib import Path
import pandas as pd

from PRAFAI.Classifiers.logistic import Logistic


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to train a model and evaluate it on validation and test sets. Different machine learning algorithms are available for model training: DecisionTree, LogisticRegression, KNN, NaiveBayes, Perceptron, MultilayerPerceptron, SVM, RandomForest, XGBoost, LightGBM, Bagging and AdaBoost.')
    parser.add_argument("input_train", help="Path to specific training set (Train) or folder with multiple training sets.")
    parser.add_argument("input_test", help="Path to specific testing set (Test) or folder with the corresponding test sets.")
    parser.add_argument("-c", "--classifiers", nargs='+', help="Classifiers to use: DecisionTree [DT], LogisticRegression [LR], KNN [KNN], NaiveBayes [NB], Perceptron [P], MultilayerPerceptron [MLP], SVM [SVM], RandomForest [RF], XGBoost [XGB], LightGBM [LGBM], Bagging [B] and/or AdaBoost [AB]. Multiple options are accepted. The [all] option will apply all techniques. Default option: [SVM].", default=['SVM'])
    parser.add_argument("-o", "--output_dir", help="Path to the output directory with the training report, trained model and evaluation results. Default option: current directory.", default=os.getcwd())
    args = vars(parser.parse_args())
    input_train = args['input_train']
    input_test = args['input_test']
    classifiers = args['classifiers']
    output_dir = args['output_dir']
    for val in classifiers:
        if val not in ['all', 'DT', 'LR', 'KNN', 'NB', 'P', 'MLP', 'SVM', 'RF', 'XGB', 'LGBM', 'B', 'AB']:
            parser.error("'--classifiers' values must be [DT], [LR], [KNN], [NB], [P], [MLP], [SVM], [RF], [XGB], [LGBM], [B], [AB] and/or [all]. Multiple options are accepted.")

    trains = {}
    tests = {}

    if (os.path.isdir(input_train) and os.path.isdir(input_test)):
        for file in os.listdir(input_train):
            if file.endswith(".csv"):
                train_name = os.path.splitext(file)[0].replace('train', '')
                print('Reading training set from: ' + str(input_train + file))
                trains[train_name] = pd.read_csv(os.path.join(input_train, file), delimiter=";")
        for file in os.listdir(input_test):
            if file.endswith(".csv"):
                test_name = os.path.splitext(file)[0].replace('test', '')
                print('Reading test set from: ' + str(input_test + file))
                tests[test_name] = pd.read_csv(os.path.join(input_test, file), delimiter=";")

    elif (not os.path.isdir(input_train) and not os.path.isdir(input_test)):
        train_name = os.path.splitext(input_train)[0].replace('train', '')
        train_name = Path(train_name).stem
        print('Reading training set from: ' + str(input_train))
        trains[train_name] = pd.read_csv(input_train, delimiter=";")
        test_name = os.path.splitext(input_test)[0].replace('test', '')
        test_name = Path(test_name).stem
        print('Reading test set from: ' + str(input_test))
        tests[test_name] = pd.read_csv(input_test, delimiter=";")

    else:
        parser.error("'input_train' and 'input_test' values must be paths to two files or two folders.")




    if 'all' in classifiers:
        # Decision Tree (DT)
        results = {}
        for i in range(len(trains.keys())):
            train_name = list(trains.keys())[i]
            test_name = list(tests.keys())[i]
            if (not trains[train_name].isnull().values.any()):
                print('\n[train' + train_name + '] Training DecisionTree model...')
                Path(os.path.join(output_dir, "DecisionTree")).mkdir(parents=True, exist_ok=True)
                path_results = os.path.join(output_dir, "DecisionTree")
                if 'StandardScaler' in train_name:
                    Path(os.path.join(path_results, "StandardScaler")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "StandardScaler")
                elif 'MinMaxScaler' in train_name:
                    Path(os.path.join(path_results, "MinMaxScaler")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "MinMaxScaler")
                elif 'MaxAbsScaler' in train_name:
                    Path(os.path.join(path_results, "MaxAbsScaler")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "MaxAbsScaler")
                elif 'QuantileTransformer' in train_name:
                    Path(os.path.join(path_results, "QuantileTransformer")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "QuantileTransformer")
                elif 'Normalizer' in train_name:
                    Path(os.path.join(path_results, "Normalizer")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "Normalizer")
                y_train, fpr, tpr, auc, classifier, X_test, y_test = DecisionTree.train(trains[train_name],
                                                                                        tests[test_name],
                                                                                        'train' + train_name,
                                                                                        path_results)
                results[train_name] = [y_train, fpr, tpr, auc, classifier, X_test, y_test]
            else:
                print("\nModel training cannot be performed to '" + str(
                    'train' + train_name) + "' dataset because it contains missing values.")
        if (len(trains.keys()) > 1):
            DecisionTree.evaluation_comparison(results, os.path.join(output_dir, "DecisionTree"))

        # Logistic Regression (LR)
        results = {}
        for i in range(len(trains.keys())):
            train_name = list(trains.keys())[i]
            test_name = list(tests.keys())[i]
            if (not trains[train_name].isnull().values.any()):
                print('\n[train' + train_name + '] Training LogisticRegression model...')
                Path(os.path.join(output_dir, "LogisticRegression")).mkdir(parents=True, exist_ok=True)
                path_results = os.path.join(output_dir, "LogisticRegression")
                if 'StandardScaler' in train_name:
                    Path(os.path.join(path_results, "StandardScaler")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "StandardScaler")
                elif 'MinMaxScaler' in train_name:
                    Path(os.path.join(path_results, "MinMaxScaler")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "MinMaxScaler")
                elif 'MaxAbsScaler' in train_name:
                    Path(os.path.join(path_results, "MaxAbsScaler")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "MaxAbsScaler")
                elif 'QuantileTransformer' in train_name:
                    Path(os.path.join(path_results, "QuantileTransformer")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "QuantileTransformer")
                elif 'Normalizer' in train_name:
                    Path(os.path.join(path_results, "Normalizer")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "Normalizer")
                y_train, fpr, tpr, auc, classifier, X_test, y_test = Logistic.train(trains[train_name],
                                                                                    tests[test_name],
                                                                                    'train' + train_name, path_results)
                results[train_name] = [y_train, fpr, tpr, auc, classifier, X_test, y_test]
            else:
                print("\nModel training cannot be performed to '" + str(
                    'train' + train_name) + "' dataset because it contains missing values.")
        if (len(trains.keys()) > 1):
            Logistic.evaluation_comparison(results, os.path.join(output_dir, "LogisticRegression"))

        # KNN
        results = {}
        for i in range(len(trains.keys())):
            train_name = list(trains.keys())[i]
            test_name = list(tests.keys())[i]
            if (not trains[train_name].isnull().values.any()):
                print('\n[train' + train_name + '] Training KNN model...')
                Path(os.path.join(output_dir, "KNN")).mkdir(parents=True, exist_ok=True)
                path_results = os.path.join(output_dir, "KNN")
                if 'StandardScaler' in train_name:
                    Path(os.path.join(path_results, "StandardScaler")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "StandardScaler")
                elif 'MinMaxScaler' in train_name:
                    Path(os.path.join(path_results, "MinMaxScaler")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "MinMaxScaler")
                elif 'MaxAbsScaler' in train_name:
                    Path(os.path.join(path_results, "MaxAbsScaler")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "MaxAbsScaler")
                elif 'QuantileTransformer' in train_name:
                    Path(os.path.join(path_results, "QuantileTransformer")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "QuantileTransformer")
                elif 'Normalizer' in train_name:
                    Path(os.path.join(path_results, "Normalizer")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "Normalizer")
                y_train, fpr, tpr, auc, classifier, X_test, y_test = KNN.train(trains[train_name], tests[test_name],
                                                                               'train' + train_name, path_results)
                results[train_name] = [y_train, fpr, tpr, auc, classifier, X_test, y_test]
            else:
                print("\nModel training cannot be performed to '" + str(
                    'train' + train_name) + "' dataset because it contains missing values.")
        if (len(trains.keys()) > 1):
            KNN.evaluation_comparison(results, os.path.join(output_dir, "KNN"))

        # Naive Bayes (NB)
        results = {}
        for i in range(len(trains.keys())):
            train_name = list(trains.keys())[i]
            test_name = list(tests.keys())[i]
            if (not trains[train_name].isnull().values.any()):
                print('\n[train' + train_name + '] Training NaiveBayes model...')
                Path(os.path.join(output_dir, "NaiveBayes")).mkdir(parents=True, exist_ok=True)
                path_results = os.path.join(output_dir, "NaiveBayes")
                if 'StandardScaler' in train_name:
                    Path(os.path.join(path_results, "StandardScaler")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "StandardScaler")
                elif 'MinMaxScaler' in train_name:
                    Path(os.path.join(path_results, "MinMaxScaler")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "MinMaxScaler")
                elif 'MaxAbsScaler' in train_name:
                    Path(os.path.join(path_results, "MaxAbsScaler")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "MaxAbsScaler")
                elif 'QuantileTransformer' in train_name:
                    Path(os.path.join(path_results, "QuantileTransformer")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "QuantileTransformer")
                elif 'Normalizer' in train_name:
                    Path(os.path.join(path_results, "Normalizer")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "Normalizer")
                y_train, fpr, tpr, auc, classifier, X_test, y_test = NaiveBayes.train(trains[train_name],
                                                                                      tests[test_name],
                                                                                      'train' + train_name,
                                                                                      path_results)
                results[train_name] = [y_train, fpr, tpr, auc, classifier, X_test, y_test]
            else:
                print("\nModel training cannot be performed to '" + str(
                    'train' + train_name) + "' dataset because it contains missing values.")
        if (len(trains.keys()) > 1):
            NaiveBayes.evaluation_comparison(results, os.path.join(output_dir, "NaiveBayes"))

        # Perceptron (P)
        results = {}
        for i in range(len(trains.keys())):
            train_name = list(trains.keys())[i]
            test_name = list(tests.keys())[i]
            if (not trains[train_name].isnull().values.any()):
                print('\n[train' + train_name + '] Training Perceptron model...')
                Path(os.path.join(output_dir, "Perceptron")).mkdir(parents=True, exist_ok=True)
                path_results = os.path.join(output_dir, "Perceptron")
                if 'StandardScaler' in train_name:
                    Path(os.path.join(path_results, "StandardScaler")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "StandardScaler")
                elif 'MinMaxScaler' in train_name:
                    Path(os.path.join(path_results, "MinMaxScaler")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "MinMaxScaler")
                elif 'MaxAbsScaler' in train_name:
                    Path(os.path.join(path_results, "MaxAbsScaler")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "MaxAbsScaler")
                elif 'QuantileTransformer' in train_name:
                    Path(os.path.join(path_results, "QuantileTransformer")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "QuantileTransformer")
                elif 'Normalizer' in train_name:
                    Path(os.path.join(path_results, "Normalizer")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "Normalizer")
                X_train, y_train, mean_fpr, mean_tpr, mean_auc, classifier, X_test, y_test = Perceptron.train(
                    trains[train_name], tests[test_name], 'train' + train_name, path_results)
                results[train_name] = [X_train, y_train, mean_fpr, mean_tpr, mean_auc, classifier, X_test, y_test]
            else:
                print("\nModel training cannot be performed to '" + str(
                    'train' + train_name) + "' dataset because it contains missing values.")
        if (len(trains.keys()) > 1):
            Perceptron.evaluation_comparison(results, os.path.join(output_dir, "Perceptron"))

        # MLP
        results = {}
        for i in range(len(trains.keys())):
            train_name = list(trains.keys())[i]
            test_name = list(tests.keys())[i]
            if (not trains[train_name].isnull().values.any()):
                print('\n[train' + train_name + '] Training MultilayerPerceptron model...')
                Path(os.path.join(output_dir, "MLP")).mkdir(parents=True, exist_ok=True)
                path_results = os.path.join(output_dir, "MLP")
                if 'StandardScaler' in train_name:
                    Path(os.path.join(path_results, "StandardScaler")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "StandardScaler")
                elif 'MinMaxScaler' in train_name:
                    Path(os.path.join(path_results, "MinMaxScaler")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "MinMaxScaler")
                elif 'MaxAbsScaler' in train_name:
                    Path(os.path.join(path_results, "MaxAbsScaler")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "MaxAbsScaler")
                elif 'QuantileTransformer' in train_name:
                    Path(os.path.join(path_results, "QuantileTransformer")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "QuantileTransformer")
                elif 'Normalizer' in train_name:
                    Path(os.path.join(path_results, "Normalizer")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "Normalizer")
                y_train, fpr, tpr, auc, classifier, X_test, y_test = MLP.train(trains[train_name], tests[test_name],
                                                                               'train' + train_name, path_results)
                results[train_name] = [y_train, fpr, tpr, auc, classifier, X_test, y_test]
            else:
                print("\nModel training cannot be performed to '" + str(
                    'train' + train_name) + "' dataset because it contains missing values.")
        if (len(trains.keys()) > 1):
            MLP.evaluation_comparison(results, os.path.join(output_dir, "MLP"))

        # Random Forest (RF)
        results = {}
        for i in range(len(trains.keys())):
            train_name = list(trains.keys())[i]
            test_name = list(tests.keys())[i]
            if (not trains[train_name].isnull().values.any()):
                print('\n[train' + train_name + '] Training RandomForest model...')
                Path(os.path.join(output_dir, "RandomForest")).mkdir(parents=True, exist_ok=True)
                path_results = os.path.join(output_dir, "RandomForest")
                if 'StandardScaler' in train_name:
                    Path(os.path.join(path_results, "StandardScaler")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "StandardScaler")
                elif 'MinMaxScaler' in train_name:
                    Path(os.path.join(path_results, "MinMaxScaler")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "MinMaxScaler")
                elif 'MaxAbsScaler' in train_name:
                    Path(os.path.join(path_results, "MaxAbsScaler")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "MaxAbsScaler")
                elif 'QuantileTransformer' in train_name:
                    Path(os.path.join(path_results, "QuantileTransformer")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "QuantileTransformer")
                elif 'Normalizer' in train_name:
                    Path(os.path.join(path_results, "Normalizer")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "Normalizer")
                y_train, fpr, tpr, auc, classifier, X_test, y_test = RandomForest.train(trains[train_name],
                                                                                        tests[test_name],
                                                                                        'train' + train_name,
                                                                                        path_results)
                results[train_name] = [y_train, fpr, tpr, auc, classifier, X_test, y_test]
            else:
                print("\nModel training cannot be performed to '" + str(
                    'train' + train_name) + "' dataset because it contains missing values.")
        if (len(trains.keys()) > 1):
            RandomForest.evaluation_comparison(results, os.path.join(output_dir, "RandomForest"))

        # SVM
        results = {}
        for i in range(len(trains.keys())):
            train_name = list(trains.keys())[i]
            test_name = list(tests.keys())[i]
            if (not trains[train_name].isnull().values.any()):
                print('\n[train' + train_name + '] Training SVM model...')
                Path(os.path.join(output_dir, "SVM")).mkdir(parents=True, exist_ok=True)
                path_results = os.path.join(output_dir, "SVM")
                if 'StandardScaler' in train_name:
                    Path(os.path.join(path_results, "StandardScaler")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "StandardScaler")
                elif 'MinMaxScaler' in train_name:
                    Path(os.path.join(path_results, "MinMaxScaler")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "MinMaxScaler")
                elif 'MaxAbsScaler' in train_name:
                    Path(os.path.join(path_results, "MaxAbsScaler")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "MaxAbsScaler")
                elif 'QuantileTransformer' in train_name:
                    Path(os.path.join(path_results, "QuantileTransformer")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "QuantileTransformer")
                elif 'Normalizer' in train_name:
                    Path(os.path.join(path_results, "Normalizer")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "Normalizer")
                y_train, fpr, tpr, auc, classifier, X_test, y_test = SVM.train(trains[train_name], tests[test_name],
                                                                               'train' + train_name, path_results)
                results[train_name] = [y_train, fpr, tpr, auc, classifier, X_test, y_test]
            else:
                print("\nModel training cannot be performed to '" + str(
                    'train' + train_name) + "' dataset because it contains missing values.")
        if (len(trains.keys()) > 1):
            SVM.evaluation_comparison(results, os.path.join(output_dir, "SVM"))

        # XGBoost (XGB)
        results = {}
        for i in range(len(trains.keys())):
            train_name = list(trains.keys())[i]
            test_name = list(tests.keys())[i]
            print('\n[train' + train_name + '] Training XGBoost model...')
            Path(os.path.join(output_dir, "XGBoost")).mkdir(parents=True, exist_ok=True)
            path_results = os.path.join(output_dir, "XGBoost")
            if 'StandardScaler' in train_name:
                Path(os.path.join(path_results, "StandardScaler")).mkdir(parents=True, exist_ok=True)
                path_results = os.path.join(path_results, "StandardScaler")
            elif 'MinMaxScaler' in train_name:
                Path(os.path.join(path_results, "MinMaxScaler")).mkdir(parents=True, exist_ok=True)
                path_results = os.path.join(path_results, "MinMaxScaler")
            elif 'MaxAbsScaler' in train_name:
                Path(os.path.join(path_results, "MaxAbsScaler")).mkdir(parents=True, exist_ok=True)
                path_results = os.path.join(path_results, "MaxAbsScaler")
            elif 'QuantileTransformer' in train_name:
                Path(os.path.join(path_results, "QuantileTransformer")).mkdir(parents=True, exist_ok=True)
                path_results = os.path.join(path_results, "QuantileTransformer")
            elif 'Normalizer' in train_name:
                Path(os.path.join(path_results, "Normalizer")).mkdir(parents=True, exist_ok=True)
                path_results = os.path.join(path_results, "Normalizer")
            y_train, fpr, tpr, auc, classifier, X_test, y_test = XGBoost.train(trains[train_name], tests[test_name],
                                                                               'train' + train_name, path_results)
            results[train_name] = [y_train, fpr, tpr, auc, classifier, X_test, y_test]
        if (len(trains.keys()) > 1):
            XGBoost.evaluation_comparison(results, os.path.join(output_dir, "XGBoost"))

        # LightGBM (LGBM)
        results = {}
        for i in range(len(trains.keys())):
            train_name = list(trains.keys())[i]
            test_name = list(tests.keys())[i]
            print('\n[train' + train_name + '] Training LightGBM model...')
            Path(os.path.join(output_dir, "LightGBM")).mkdir(parents=True, exist_ok=True)
            path_results = os.path.join(output_dir, "LightGBM")
            if 'StandardScaler' in train_name:
                Path(os.path.join(path_results, "StandardScaler")).mkdir(parents=True, exist_ok=True)
                path_results = os.path.join(path_results, "StandardScaler")
            elif 'MinMaxScaler' in train_name:
                Path(os.path.join(path_results, "MinMaxScaler")).mkdir(parents=True, exist_ok=True)
                path_results = os.path.join(path_results, "MinMaxScaler")
            elif 'MaxAbsScaler' in train_name:
                Path(os.path.join(path_results, "MaxAbsScaler")).mkdir(parents=True, exist_ok=True)
                path_results = os.path.join(path_results, "MaxAbsScaler")
            elif 'QuantileTransformer' in train_name:
                Path(os.path.join(path_results, "QuantileTransformer")).mkdir(parents=True, exist_ok=True)
                path_results = os.path.join(path_results, "QuantileTransformer")
            elif 'Normalizer' in train_name:
                Path(os.path.join(path_results, "Normalizer")).mkdir(parents=True, exist_ok=True)
                path_results = os.path.join(path_results, "Normalizer")
            y_train, fpr, tpr, auc, classifier, X_test, y_test = LightGBM.train(trains[train_name], tests[test_name],
                                                                                'train' + train_name, path_results)
            results[train_name] = [y_train, fpr, tpr, auc, classifier, X_test, y_test]
        if (len(trains.keys()) > 1):
            LightGBM.evaluation_comparison(results, os.path.join(output_dir, "LightGBM"))

        # Bagging (B)
        results = {}
        for i in range(len(trains.keys())):
            train_name = list(trains.keys())[i]
            test_name = list(tests.keys())[i]
            if (not trains[train_name].isnull().values.any()):
                print('\n[train' + train_name + '] Training Bagging model...')
                Path(os.path.join(output_dir, "Bagging")).mkdir(parents=True, exist_ok=True)
                path_results = os.path.join(output_dir, "Bagging")
                if 'StandardScaler' in train_name:
                    Path(os.path.join(path_results, "StandardScaler")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "StandardScaler")
                elif 'MinMaxScaler' in train_name:
                    Path(os.path.join(path_results, "MinMaxScaler")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "MinMaxScaler")
                elif 'MaxAbsScaler' in train_name:
                    Path(os.path.join(path_results, "MaxAbsScaler")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "MaxAbsScaler")
                elif 'QuantileTransformer' in train_name:
                    Path(os.path.join(path_results, "QuantileTransformer")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "QuantileTransformer")
                elif 'Normalizer' in train_name:
                    Path(os.path.join(path_results, "Normalizer")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "Normalizer")
                y_train, fpr, tpr, auc, classifier, X_test, y_test = Bagging.train(trains[train_name], tests[test_name],
                                                                                   'train' + train_name, path_results)
                results[train_name] = [y_train, fpr, tpr, auc, classifier, X_test, y_test]
            else:
                print("\nModel training cannot be performed to '" + str(
                    'train' + train_name) + "' dataset because it contains missing values.")
        if (len(trains.keys()) > 1):
            Bagging.evaluation_comparison(results, os.path.join(output_dir, "Bagging"))

        # AdaBoost (AB)
        results = {}
        for i in range(len(trains.keys())):
            train_name = list(trains.keys())[i]
            test_name = list(tests.keys())[i]
            if (not trains[train_name].isnull().values.any()):
                print('\n[train' + train_name + '] Training AdaBoost model...')
                Path(os.path.join(output_dir, "AdaBoost")).mkdir(parents=True, exist_ok=True)
                path_results = os.path.join(output_dir, "AdaBoost")
                if 'StandardScaler' in train_name:
                    Path(os.path.join(path_results, "StandardScaler")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "StandardScaler")
                elif 'MinMaxScaler' in train_name:
                    Path(os.path.join(path_results, "MinMaxScaler")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "MinMaxScaler")
                elif 'MaxAbsScaler' in train_name:
                    Path(os.path.join(path_results, "MaxAbsScaler")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "MaxAbsScaler")
                elif 'QuantileTransformer' in train_name:
                    Path(os.path.join(path_results, "QuantileTransformer")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "QuantileTransformer")
                elif 'Normalizer' in train_name:
                    Path(os.path.join(path_results, "Normalizer")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(path_results, "Normalizer")
                y_train, fpr, tpr, auc, classifier, X_test, y_test = AdaBoost.train(trains[train_name],
                                                                                    tests[test_name],
                                                                                    'train' + train_name, path_results)
                results[train_name] = [y_train, fpr, tpr, auc, classifier, X_test, y_test]
            else:
                print("\nModel training cannot be performed to '" + str(
                    'train' + train_name) + "' dataset because it contains missing values.")
        if (len(trains.keys()) > 1):
            AdaBoost.evaluation_comparison(results, os.path.join(output_dir, "AdaBoost"))

    else:
        for val in classifiers:
            if (val == 'DT'):
                results = {}
                for i in range(len(trains.keys())):
                    train_name = list(trains.keys())[i]
                    test_name = list(tests.keys())[i]
                    if (not trains[train_name].isnull().values.any()):
                        print('\n[train' + train_name + '] Training DecisionTree model...')
                        Path(os.path.join(output_dir, "DecisionTree")).mkdir(parents=True, exist_ok=True)
                        path_results = os.path.join(output_dir, "DecisionTree")
                        if 'StandardScaler' in train_name:
                            Path(os.path.join(path_results, "StandardScaler")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "StandardScaler")
                        elif 'MinMaxScaler' in train_name:
                            Path(os.path.join(path_results, "MinMaxScaler")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "MinMaxScaler")
                        elif 'MaxAbsScaler' in train_name:
                            Path(os.path.join(path_results, "MaxAbsScaler")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "MaxAbsScaler")
                        elif 'QuantileTransformer' in train_name:
                            Path(os.path.join(path_results, "QuantileTransformer")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "QuantileTransformer")
                        elif 'Normalizer' in train_name:
                            Path(os.path.join(path_results, "Normalizer")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "Normalizer")
                        y_train, fpr, tpr, auc, classifier, X_test, y_test = DecisionTree.train(trains[train_name], tests[test_name], 'train' + train_name, path_results)
                        results[train_name] = [y_train, fpr, tpr, auc, classifier, X_test, y_test]
                    else:
                        print("\nModel training cannot be performed to '" + str('train' + train_name) + "' dataset because it contains missing values.")
                if (len(trains.keys()) > 1):
                    DecisionTree.evaluation_comparison(results, os.path.join(output_dir, "DecisionTree"))

            elif (val == 'LR'):
                results = {}
                for i in range(len(trains.keys())):
                    train_name = list(trains.keys())[i]
                    test_name = list(tests.keys())[i]
                    if (not trains[train_name].isnull().values.any()):
                        print('\n[train' + train_name + '] Training LogisticRegression model...')
                        Path(os.path.join(output_dir, "LogisticRegression")).mkdir(parents=True, exist_ok=True)
                        path_results = os.path.join(output_dir, "LogisticRegression")
                        if 'StandardScaler' in train_name:
                            Path(os.path.join(path_results, "StandardScaler")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "StandardScaler")
                        elif 'MinMaxScaler' in train_name:
                            Path(os.path.join(path_results, "MinMaxScaler")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "MinMaxScaler")
                        elif 'MaxAbsScaler' in train_name:
                            Path(os.path.join(path_results, "MaxAbsScaler")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "MaxAbsScaler")
                        elif 'QuantileTransformer' in train_name:
                            Path(os.path.join(path_results, "QuantileTransformer")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "QuantileTransformer")
                        elif 'Normalizer' in train_name:
                            Path(os.path.join(path_results, "Normalizer")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "Normalizer")
                        y_train, fpr, tpr, auc, classifier, X_test, y_test = Logistic.train(trains[train_name], tests[test_name], 'train' + train_name, path_results)
                        results[train_name] = [y_train, fpr, tpr, auc, classifier, X_test, y_test]
                    else:
                        print("\nModel training cannot be performed to '" + str('train' + train_name) + "' dataset because it contains missing values.")
                if(len(trains.keys()) > 1):
                    Logistic.evaluation_comparison(results, os.path.join(output_dir, "LogisticRegression"))

            elif (val == 'KNN'):
                results = {}
                for i in range(len(trains.keys())):
                    train_name = list(trains.keys())[i]
                    test_name = list(tests.keys())[i]
                    if (not trains[train_name].isnull().values.any()):
                        print('\n[train' + train_name + '] Training KNN model...')
                        Path(os.path.join(output_dir, "KNN")).mkdir(parents=True, exist_ok=True)
                        path_results = os.path.join(output_dir, "KNN")
                        if 'StandardScaler' in train_name:
                            Path(os.path.join(path_results, "StandardScaler")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "StandardScaler")
                        elif 'MinMaxScaler' in train_name:
                            Path(os.path.join(path_results, "MinMaxScaler")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "MinMaxScaler")
                        elif 'MaxAbsScaler' in train_name:
                            Path(os.path.join(path_results, "MaxAbsScaler")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "MaxAbsScaler")
                        elif 'QuantileTransformer' in train_name:
                            Path(os.path.join(path_results, "QuantileTransformer")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "QuantileTransformer")
                        elif 'Normalizer' in train_name:
                            Path(os.path.join(path_results, "Normalizer")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "Normalizer")
                        y_train, fpr, tpr, auc, classifier, X_test, y_test = KNN.train(trains[train_name], tests[test_name], 'train' + train_name, path_results)
                        results[train_name] = [y_train, fpr, tpr, auc, classifier, X_test, y_test]
                    else:
                        print("\nModel training cannot be performed to '" + str('train' + train_name) + "' dataset because it contains missing values.")
                if(len(trains.keys()) > 1):
                    KNN.evaluation_comparison(results, os.path.join(output_dir, "KNN"))

            elif (val == 'NB'):
                results = {}
                for i in range(len(trains.keys())):
                    train_name = list(trains.keys())[i]
                    test_name = list(tests.keys())[i]
                    if (not trains[train_name].isnull().values.any()):
                        print('\n[train' + train_name + '] Training NaiveBayes model...')
                        Path(os.path.join(output_dir, "NaiveBayes")).mkdir(parents=True, exist_ok=True)
                        path_results = os.path.join(output_dir, "NaiveBayes")
                        if 'StandardScaler' in train_name:
                            Path(os.path.join(path_results, "StandardScaler")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "StandardScaler")
                        elif 'MinMaxScaler' in train_name:
                            Path(os.path.join(path_results, "MinMaxScaler")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "MinMaxScaler")
                        elif 'MaxAbsScaler' in train_name:
                            Path(os.path.join(path_results, "MaxAbsScaler")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "MaxAbsScaler")
                        elif 'QuantileTransformer' in train_name:
                            Path(os.path.join(path_results, "QuantileTransformer")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "QuantileTransformer")
                        elif 'Normalizer' in train_name:
                            Path(os.path.join(path_results, "Normalizer")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "Normalizer")
                        y_train, fpr, tpr, auc, classifier, X_test, y_test = NaiveBayes.train(trains[train_name], tests[test_name], 'train' + train_name, path_results)
                        results[train_name] = [y_train, fpr, tpr, auc, classifier, X_test, y_test]
                    else:
                        print("\nModel training cannot be performed to '" + str('train' + train_name) + "' dataset because it contains missing values.")
                if(len(trains.keys()) > 1):
                    NaiveBayes.evaluation_comparison(results, os.path.join(output_dir, "NaiveBayes"))

            elif (val == 'P'):
                results = {}
                for i in range(len(trains.keys())):
                    train_name = list(trains.keys())[i]
                    test_name = list(tests.keys())[i]
                    if (not trains[train_name].isnull().values.any()):
                        print('\n[train' + train_name + '] Training Perceptron model...')
                        Path(os.path.join(output_dir, "Perceptron")).mkdir(parents=True, exist_ok=True)
                        path_results = os.path.join(output_dir, "Perceptron")
                        if 'StandardScaler' in train_name:
                            Path(os.path.join(path_results, "StandardScaler")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "StandardScaler")
                        elif 'MinMaxScaler' in train_name:
                            Path(os.path.join(path_results, "MinMaxScaler")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "MinMaxScaler")
                        elif 'MaxAbsScaler' in train_name:
                            Path(os.path.join(path_results, "MaxAbsScaler")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "MaxAbsScaler")
                        elif 'QuantileTransformer' in train_name:
                            Path(os.path.join(path_results, "QuantileTransformer")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "QuantileTransformer")
                        elif 'Normalizer' in train_name:
                            Path(os.path.join(path_results, "Normalizer")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "Normalizer")
                        X_train, y_train, mean_fpr, mean_tpr, mean_auc, classifier, X_test, y_test = Perceptron.train(trains[train_name], tests[test_name], 'train' + train_name, path_results)
                        results[train_name] = [X_train, y_train, mean_fpr, mean_tpr, mean_auc, classifier, X_test, y_test]
                    else:
                        print("\nModel training cannot be performed to '" + str('train' + train_name) + "' dataset because it contains missing values.")
                if(len(trains.keys()) > 1):
                    Perceptron.evaluation_comparison(results, os.path.join(output_dir, "Perceptron"))

            elif (val == 'MLP'):
                results = {}
                for i in range(len(trains.keys())):
                    train_name = list(trains.keys())[i]
                    test_name = list(tests.keys())[i]
                    if (not trains[train_name].isnull().values.any()):
                        print('\n[train' + train_name + '] Training MultilayerPerceptron model...')
                        Path(os.path.join(output_dir, "MLP")).mkdir(parents=True, exist_ok=True)
                        path_results = os.path.join(output_dir, "MLP")
                        if 'StandardScaler' in train_name:
                            Path(os.path.join(path_results, "StandardScaler")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "StandardScaler")
                        elif 'MinMaxScaler' in train_name:
                            Path(os.path.join(path_results, "MinMaxScaler")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "MinMaxScaler")
                        elif 'MaxAbsScaler' in train_name:
                            Path(os.path.join(path_results, "MaxAbsScaler")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "MaxAbsScaler")
                        elif 'QuantileTransformer' in train_name:
                            Path(os.path.join(path_results, "QuantileTransformer")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "QuantileTransformer")
                        elif 'Normalizer' in train_name:
                            Path(os.path.join(path_results, "Normalizer")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "Normalizer")
                        y_train, fpr, tpr, auc, classifier, X_test, y_test = MLP.train(trains[train_name], tests[test_name], 'train' + train_name, path_results)
                        results[train_name] = [y_train, fpr, tpr, auc, classifier, X_test, y_test]
                    else:
                        print("\nModel training cannot be performed to '" + str('train' + train_name) + "' dataset because it contains missing values.")
                if(len(trains.keys()) > 1):
                    MLP.evaluation_comparison(results, os.path.join(output_dir, "MLP"))

            elif (val == 'RF'):
                results = {}
                for i in range(len(trains.keys())):
                    train_name = list(trains.keys())[i]
                    test_name = list(tests.keys())[i]
                    if (not trains[train_name].isnull().values.any()):
                        print('\n[train' + train_name + '] Training RandomForest model...')
                        Path(os.path.join(output_dir, "RandomForest")).mkdir(parents=True, exist_ok=True)
                        path_results = os.path.join(output_dir, "RandomForest")
                        if 'StandardScaler' in train_name:
                            Path(os.path.join(path_results, "StandardScaler")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "StandardScaler")
                        elif 'MinMaxScaler' in train_name:
                            Path(os.path.join(path_results, "MinMaxScaler")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "MinMaxScaler")
                        elif 'MaxAbsScaler' in train_name:
                            Path(os.path.join(path_results, "MaxAbsScaler")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "MaxAbsScaler")
                        elif 'QuantileTransformer' in train_name:
                            Path(os.path.join(path_results, "QuantileTransformer")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "QuantileTransformer")
                        elif 'Normalizer' in train_name:
                            Path(os.path.join(path_results, "Normalizer")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "Normalizer")
                        y_train, fpr, tpr, auc, classifier, X_test, y_test = RandomForest.train(trains[train_name], tests[test_name], 'train' + train_name, path_results)
                        results[train_name] = [y_train, fpr, tpr, auc, classifier, X_test, y_test]
                    else:
                        print("\nModel training cannot be performed to '" + str('train' + train_name) + "' dataset because it contains missing values.")
                if(len(trains.keys()) > 1):
                    RandomForest.evaluation_comparison(results, os.path.join(output_dir, "RandomForest"))

            elif (val == 'SVM'):
                results = {}
                for i in range(len(trains.keys())):
                    train_name = list(trains.keys())[i]
                    test_name = list(tests.keys())[i]
                    if (not trains[train_name].isnull().values.any()):
                        print('\n[train' + train_name + '] Training SVM model...')
                        Path(os.path.join(output_dir, "SVM")).mkdir(parents=True, exist_ok=True)
                        path_results = os.path.join(output_dir, "SVM")
                        if 'StandardScaler' in train_name:
                            Path(os.path.join(path_results, "StandardScaler")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "StandardScaler")
                        elif 'MinMaxScaler' in train_name:
                            Path(os.path.join(path_results, "MinMaxScaler")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "MinMaxScaler")
                        elif 'MaxAbsScaler' in train_name:
                            Path(os.path.join(path_results, "MaxAbsScaler")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "MaxAbsScaler")
                        elif 'QuantileTransformer' in train_name:
                            Path(os.path.join(path_results, "QuantileTransformer")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "QuantileTransformer")
                        elif 'Normalizer' in train_name:
                            Path(os.path.join(path_results, "Normalizer")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "Normalizer")
                        y_train, fpr, tpr, auc, classifier, X_test, y_test = SVM.train(trains[train_name], tests[test_name], 'train' + train_name, path_results)
                        results[train_name] = [y_train, fpr, tpr, auc, classifier, X_test, y_test]
                    else:
                        print("\nModel training cannot be performed to '" + str('train' + train_name) + "' dataset because it contains missing values.")
                if(len(trains.keys()) > 1):
                    SVM.evaluation_comparison(results, os.path.join(output_dir, "SVM"))

            elif (val == 'XGB'):
                results = {}
                for i in range(len(trains.keys())):
                    train_name = list(trains.keys())[i]
                    test_name = list(tests.keys())[i]
                    print('\n[train' + train_name + '] Training XGBoost model...')
                    Path(os.path.join(output_dir, "XGBoost")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(output_dir, "XGBoost")
                    if 'StandardScaler' in train_name:
                        Path(os.path.join(path_results, "StandardScaler")).mkdir(parents=True, exist_ok=True)
                        path_results = os.path.join(path_results, "StandardScaler")
                    elif 'MinMaxScaler' in train_name:
                        Path(os.path.join(path_results, "MinMaxScaler")).mkdir(parents=True, exist_ok=True)
                        path_results = os.path.join(path_results, "MinMaxScaler")
                    elif 'MaxAbsScaler' in train_name:
                        Path(os.path.join(path_results, "MaxAbsScaler")).mkdir(parents=True, exist_ok=True)
                        path_results = os.path.join(path_results, "MaxAbsScaler")
                    elif 'QuantileTransformer' in train_name:
                        Path(os.path.join(path_results, "QuantileTransformer")).mkdir(parents=True, exist_ok=True)
                        path_results = os.path.join(path_results, "QuantileTransformer")
                    elif 'Normalizer' in train_name:
                        Path(os.path.join(path_results, "Normalizer")).mkdir(parents=True, exist_ok=True)
                        path_results = os.path.join(path_results, "Normalizer")
                    y_train, fpr, tpr, auc, classifier, X_test, y_test = XGBoost.train(trains[train_name], tests[test_name], 'train' + train_name, path_results)
                    results[train_name] = [y_train, fpr, tpr, auc, classifier, X_test, y_test]
                if(len(trains.keys()) > 1):
                    XGBoost.evaluation_comparison(results, os.path.join(output_dir, "XGBoost"))

            elif (val == 'LGBM'):
                results = {}
                for i in range(len(trains.keys())):
                    train_name = list(trains.keys())[i]
                    test_name = list(tests.keys())[i]
                    print('\n[train' + train_name + '] Training LightGBM model...')
                    Path(os.path.join(output_dir, "LightGBM")).mkdir(parents=True, exist_ok=True)
                    path_results = os.path.join(output_dir, "LightGBM")
                    if 'StandardScaler' in train_name:
                        Path(os.path.join(path_results, "StandardScaler")).mkdir(parents=True, exist_ok=True)
                        path_results = os.path.join(path_results, "StandardScaler")
                    elif 'MinMaxScaler' in train_name:
                        Path(os.path.join(path_results, "MinMaxScaler")).mkdir(parents=True, exist_ok=True)
                        path_results = os.path.join(path_results, "MinMaxScaler")
                    elif 'MaxAbsScaler' in train_name:
                        Path(os.path.join(path_results, "MaxAbsScaler")).mkdir(parents=True, exist_ok=True)
                        path_results = os.path.join(path_results, "MaxAbsScaler")
                    elif 'QuantileTransformer' in train_name:
                        Path(os.path.join(path_results, "QuantileTransformer")).mkdir(parents=True, exist_ok=True)
                        path_results = os.path.join(path_results, "QuantileTransformer")
                    elif 'Normalizer' in train_name:
                        Path(os.path.join(path_results, "Normalizer")).mkdir(parents=True, exist_ok=True)
                        path_results = os.path.join(path_results, "Normalizer")
                    y_train, fpr, tpr, auc, classifier, X_test, y_test = LightGBM.train(trains[train_name], tests[test_name], 'train' + train_name, path_results)
                    results[train_name] = [y_train, fpr, tpr, auc, classifier, X_test, y_test]
                if(len(trains.keys()) > 1):
                    LightGBM.evaluation_comparison(results, os.path.join(output_dir, "LightGBM"))

            elif (val == 'B'):
                results = {}
                for i in range(len(trains.keys())):
                    train_name = list(trains.keys())[i]
                    test_name = list(tests.keys())[i]
                    if (not trains[train_name].isnull().values.any()):
                        print('\n[train' + train_name + '] Training Bagging model...')
                        Path(os.path.join(output_dir, "Bagging")).mkdir(parents=True, exist_ok=True)
                        path_results = os.path.join(output_dir, "Bagging")
                        if 'StandardScaler' in train_name:
                            Path(os.path.join(path_results, "StandardScaler")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "StandardScaler")
                        elif 'MinMaxScaler' in train_name:
                            Path(os.path.join(path_results, "MinMaxScaler")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "MinMaxScaler")
                        elif 'MaxAbsScaler' in train_name:
                            Path(os.path.join(path_results, "MaxAbsScaler")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "MaxAbsScaler")
                        elif 'QuantileTransformer' in train_name:
                            Path(os.path.join(path_results, "QuantileTransformer")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "QuantileTransformer")
                        elif 'Normalizer' in train_name:
                            Path(os.path.join(path_results, "Normalizer")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "Normalizer")
                        y_train, fpr, tpr, auc, classifier, X_test, y_test = Bagging.train(trains[train_name], tests[test_name], 'train' + train_name, path_results)
                        results[train_name] = [y_train, fpr, tpr, auc, classifier, X_test, y_test]
                    else:
                        print("\nModel training cannot be performed to '" + str('train' + train_name) + "' dataset because it contains missing values.")
                if(len(trains.keys()) > 1):
                    Bagging.evaluation_comparison(results, os.path.join(output_dir, "Bagging"))

            elif (val == 'AB'):
                results = {}
                for i in range(len(trains.keys())):
                    train_name = list(trains.keys())[i]
                    test_name = list(tests.keys())[i]
                    if (not trains[train_name].isnull().values.any()):
                        print('\n[train' + train_name + '] Training AdaBoost model...')
                        Path(os.path.join(output_dir, "AdaBoost")).mkdir(parents=True, exist_ok=True)
                        path_results = os.path.join(output_dir, "AdaBoost")
                        if 'StandardScaler' in train_name:
                            Path(os.path.join(path_results, "StandardScaler")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "StandardScaler")
                        elif 'MinMaxScaler' in train_name:
                            Path(os.path.join(path_results, "MinMaxScaler")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "MinMaxScaler")
                        elif 'MaxAbsScaler' in train_name:
                            Path(os.path.join(path_results, "MaxAbsScaler")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "MaxAbsScaler")
                        elif 'QuantileTransformer' in train_name:
                            Path(os.path.join(path_results, "QuantileTransformer")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "QuantileTransformer")
                        elif 'Normalizer' in train_name:
                            Path(os.path.join(path_results, "Normalizer")).mkdir(parents=True, exist_ok=True)
                            path_results = os.path.join(path_results, "Normalizer")
                        y_train, fpr, tpr, auc, classifier, X_test, y_test = AdaBoost.train(trains[train_name], tests[test_name], 'train' + train_name, path_results)
                        results[train_name] = [y_train, fpr, tpr, auc, classifier, X_test, y_test]
                    else:
                        print("\nModel training cannot be performed to '" + str('train' + train_name) + "' dataset because it contains missing values.")
                if(len(trains.keys()) > 1):
                    AdaBoost.evaluation_comparison(results, os.path.join(output_dir, "AdaBoost"))