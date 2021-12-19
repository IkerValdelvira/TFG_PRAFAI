import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, RFECV, SelectFromModel, SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt


output_dir = ""

def rfe(train_dev, test, file_name):
    file = open(os.path.join(output_dir, 'RFE', 'Train', 'Reports', 'REPORT_RFE_' + file_name + '.txt'),"w")
    file.write("Dataset: " + file_name + "\n")

    # Borrar variables con varianza = 0
    selector = VarianceThreshold(threshold=(0))
    selector.fit(train_dev)
    all_columns = train_dev.columns
    train_dev = train_dev[train_dev.columns[selector.get_support(indices=True)]]
    current_columns = train_dev.columns
    deleted_columns = list(set(all_columns) - set(current_columns))
    file.write('\nDeleted features with variance to zero: {}'.format(len(deleted_columns)))
    file.write('\n' + str(deleted_columns))

    #RFE
    y_train = train_dev["sigue_fa"]
    X_train = train_dev.drop("sigue_fa", 1)

    # Create the RFE object and compute a cross-validated score.
    svc = SVC(kernel="linear")

    min_features_to_select = 1  # Minimum number of features to consider
    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(10),
                  scoring='roc_auc',
                  min_features_to_select=min_features_to_select)
    rfecv.fit(X_train, y_train)

    train_dev = X_train[X_train.columns[rfecv.get_support(indices=True)]]
    train_dev['sigue_fa'] = y_train
    feature_names = np.array(X_train.columns)

    file.write('\n\nNumber of current features: {}'.format((X_train.shape[1])))
    file.write('\nNumber of selected features: {}'.format(len(feature_names[rfecv.get_support()])))
    file.write('\nNumber of deleted features: {}'.format(len(list(set(feature_names) - set(feature_names[rfecv.get_support()])))))
    file.write('\n\nFeatures selected by RFE:')
    file.write('\n' + str(feature_names[rfecv.get_support()]))


    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.figtext(0.52, 0.15, "Optimal number of features: %d" % rfecv.n_features_, wrap=True, horizontalalignment='center', fontsize=10)
    plt.title("RFE Feature Selection --> " + file_name)
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (AUC)")
    plt.plot(range(min_features_to_select, len(rfecv.grid_scores_) + min_features_to_select), rfecv.grid_scores_)
    plt.savefig(os.path.join(output_dir, "RFE", "Train", 'Reports', 'RFE_' + file_name + '.png'))
    #plt.show()
    plt.close()

    # Borrar variables del Test
    removed_features = list(set(test.columns) - set(train_dev.columns))
    test = test.drop(removed_features, axis=1)

    return train_dev, test


def lasso_sfm(train_dev, test, file_name):
    file = open(os.path.join(output_dir, 'Lasso_SFM', 'Train', 'Reports', 'REPORT_Lasso_SelectFromModel_' + file_name + '.txt'),"w")
    file.write("Dataset: " + file_name + "\n")

    # Borrar variables con varianza = 0
    selector = VarianceThreshold(threshold=(0))
    selector.fit(train_dev)
    all_columns = train_dev.columns
    train_dev = train_dev[train_dev.columns[selector.get_support(indices=True)]]
    current_columns = train_dev.columns
    deleted_columns = list(set(all_columns) - set(current_columns))
    file.write('\nDeleted features with variance to zero: {}'.format(len(deleted_columns)))
    file.write('\n' + str(deleted_columns))

    y_train = train_dev["sigue_fa"]
    X_train = train_dev.drop("sigue_fa", 1)

    # Lasso
    lasso = LogisticRegression(C=1, penalty='l1', solver='liblinear').fit(X_train, y_train)
    importance = np.abs(lasso.coef_)
    importance = np.concatenate(importance, axis=0)
    feature_names = np.array(X_train.columns)
    plt.figure(figsize=(25, 15))
    plt.barh(width=importance, y=feature_names)
    plt.title("Feature importances via coefficients --> " + file_name)
    plt.savefig(os.path.join(output_dir, "Lasso_SFM", "Train", 'Reports', 'Lasso_SelectFromModel_' + file_name + '.png'))
    #plt.show()
    plt.close()

    # SelectFromModel
    sfm = SelectFromModel(lasso).fit(X_train, y_train)
    train_dev = X_train[X_train.columns[sfm.get_support()]]
    train_dev['sigue_fa'] = y_train

    selected_feat = X_train.columns[(sfm.get_support())].tolist()
    selected_feat_indices = sfm.get_support(indices=True)
    file.write('\n\nNumber of current features: {}'.format((X_train.shape[1])))
    file.write('\nNumber of selected features: {}'.format(len(selected_feat)))
    file.write('\nNumber of features with coefficients shrank to zero: {}'.format(np.sum(sfm.estimator_.coef_ == 0)))
    file.write('\n\nFeatures selected by SelectFromModel:')
    i=0
    for ind in selected_feat_indices:
        file.write('\n\t' + str(selected_feat[i]) + ": " + str(importance[ind]))
        i += 1

    # Borrar variables del Test
    removed_features = list(set(test.columns) - set(train_dev.columns))
    test = test.drop(removed_features, axis=1)

    return train_dev, test


def sfs(train_dev, test, file_name):
    file = open(os.path.join(output_dir, 'SFS', 'Train', 'Reports', 'REPORT_SFS_' + file_name + '.txt'), "w")
    file.write("Dataset: " + file_name + "\n")

    # Borrar variables con varianza = 0
    selector = VarianceThreshold(threshold=(0))
    selector.fit(train_dev)
    all_columns = train_dev.columns
    train_dev = train_dev[train_dev.columns[selector.get_support(indices=True)]]
    current_columns = train_dev.columns
    deleted_columns = list(set(all_columns) - set(current_columns))
    file.write('\nDeleted features with variance to zero: {}'.format(len(deleted_columns)))
    file.write('\n' + str(deleted_columns))

    y_train = train_dev["sigue_fa"]
    X_train = train_dev.drop("sigue_fa", 1)

    # Lasso
    lasso = LogisticRegression(C=1, penalty='l1', solver='liblinear', max_iter=1500).fit(X_train, y_train)
    importance = np.abs(lasso.coef_)
    importance = np.concatenate(importance, axis=0)
    feature_names = np.array(X_train.columns)

    # SequentialFeatureSelection
    sfs_forward = SequentialFeatureSelector(lasso, direction='forward').fit(X_train, y_train)
    train_dev = X_train[X_train.columns[sfs_forward.get_support()]]
    train_dev['sigue_fa'] = y_train

    selected_feat = X_train.columns[(sfs_forward.get_support())].tolist()
    selected_feat_indices = sfs_forward.get_support(indices=True)
    file.write('\n\nNumber of current features: {}'.format((X_train.shape[1])))
    file.write('\nNumber of selected features: {}'.format(len(feature_names[sfs_forward.get_support()])))
    file.write('\nNumber of deleted features: {}'.format(len(list(set(feature_names) - set(feature_names[sfs_forward.get_support()])))))
    file.write('\n\nFeatures selected by Forward SequentialFeatureSelection:')
    i=0
    for ind in selected_feat_indices:
        file.write('\n\t' + str(selected_feat[i]) + ": " + str(importance[ind]))
        i += 1

    # Borrar variables del Test
    removed_features = list(set(test.columns) - set(train_dev.columns))
    test = test.drop(removed_features, axis=1)

    return train_dev, test


def selectkbest(train_dev, test, file_name):
    file = open(os.path.join(output_dir, 'SelectKBest', 'Train', 'Reports', 'REPORT_SelectKBest_' + file_name + '.txt'), "w")
    file.write("Dataset: " + file_name + "\n")

    # Borrar variables con varianza = 0
    selector = VarianceThreshold(threshold=(0))
    selector.fit(train_dev)
    all_columns = train_dev.columns
    train_dev = train_dev[train_dev.columns[selector.get_support(indices=True)]]
    current_columns = train_dev.columns
    deleted_columns = list(set(all_columns) - set(current_columns))
    file.write('\nDeleted features with variance to zero: {}'.format(len(deleted_columns)))
    file.write('\n' + str(deleted_columns))

    y_train = train_dev["sigue_fa"]
    X_train = train_dev.drop("sigue_fa", 1)

    # SelectKBest Chi2
    skb = SelectKBest(chi2, k=25).fit(X_train, y_train)
    pvalues = skb.scores_
    feature_names = np.array(X_train.columns)
    plt.figure(figsize=(25, 15))
    plt.barh(width=pvalues, y=feature_names)
    plt.title("Feature importances via scores --> " + file_name)
    plt.savefig(os.path.join(output_dir, "SelectKBest", "Train", 'Reports', 'SelectKBest_' + file_name + '.png'))
    #plt.show()
    plt.close()

    train_dev = X_train[X_train.columns[skb.get_support()]]
    train_dev['sigue_fa'] = y_train

    selected_feat = X_train.columns[(skb.get_support())].tolist()
    selected_feat_indices = skb.get_support(indices=True)
    file.write('\n\nNumber of current features: {}'.format(X_train.shape[1]))
    file.write('\nNumber of selected features: {}'.format(len(selected_feat)))
    file.write('\nNumber of deleted features: {}'.format(X_train.shape[1] - len(selected_feat)))
    file.write('\n\nFeatures selected by SelectKBest:')
    i=0
    for ind in selected_feat_indices:
        file.write('\n\t' + str(selected_feat[i]) + ": " + str(pvalues[ind]))
        i += 1

    # Borrar variables del Test
    removed_features = list(set(test.columns) - set(train_dev.columns))
    test = test.drop(removed_features, axis=1)

    return train_dev, test


def trees_sfm(train_dev, test, file_name):
    file = open(os.path.join(output_dir, 'Trees_SFM', 'Train', 'Reports', 'REPORT_Trees_SelectFromModel_' + file_name + '.txt'), "w")
    file.write("Dataset: " + file_name + "\n")

    # Borrar variables con varianza = 0
    selector = VarianceThreshold(threshold=(0))
    selector.fit(train_dev)
    all_columns = train_dev.columns
    train_dev = train_dev[train_dev.columns[selector.get_support(indices=True)]]
    current_columns = train_dev.columns
    deleted_columns = list(set(all_columns) - set(current_columns))
    file.write('\nDeleted features with variance to zero: {}'.format(len(deleted_columns)))
    file.write('\n' + str(deleted_columns))

    y_train = train_dev["sigue_fa"]
    X_train = train_dev.drop("sigue_fa", 1)

    # ExtraTreesClassifier
    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(X_train, y_train)
    importance = clf.feature_importances_
    feature_names = np.array(X_train.columns)
    plt.figure(figsize=(25, 15))
    plt.barh(width=importance, y=feature_names)
    plt.title("Feature importances via Gini importance --> " + file_name)
    plt.savefig(os.path.join(output_dir, "Trees_SFM", "Train", 'Reports', 'Trees_SelectFromModel_' + file_name + '.png'))
    #plt.show()
    plt.close()

    # SelectFromModel
    sfm = SelectFromModel(clf).fit(X_train, y_train)
    train_dev = X_train[X_train.columns[sfm.get_support(indices=True)]]
    train_dev['sigue_fa'] = y_train
    feature_names = np.array(X_train.columns)

    selected_feat = X_train.columns[(sfm.get_support())].tolist()
    selected_feat_indices = sfm.get_support(indices=True)
    file.write('\n\nNumber of current features: {}'.format((X_train.shape[1])))
    file.write('\nNumber of selected features: {}'.format(len(feature_names[sfm.get_support()])))
    file.write('\nNumber of deleted features: {}'.format(len(list(set(feature_names) - set(feature_names[sfm.get_support()])))))
    file.write('\n\nFeatures selected by SelectFromModel:')
    i = 0
    for ind in selected_feat_indices:
        file.write('\n\t' + str(selected_feat[i]) + ": " + str(importance[ind]))
        i += 1

    # Borrar variables del Test
    removed_features = list(set(test.columns) - set(train_dev.columns))
    test = test.drop(removed_features, axis=1)

    return train_dev, test


def pca(train_dev, test, file_name):
    y_train = train_dev["sigue_fa"]
    y_test = test["sigue_fa"]
    X_train = train_dev.drop("sigue_fa", 1)
    X_test = test.drop("sigue_fa", 1)

    pca= PCA(n_components=0.99)
    pca.fit(X_train)

    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    train_dev = pd.DataFrame(X_train_pca)
    test = pd.DataFrame(X_test_pca)

    train_dev['sigue_fa'] = y_train
    test['sigue_fa'] = y_test

    # Plot
    pca_plot = PCA(n_components=len(X_train.columns))
    pca_plot.fit(X_train)

    var = np.cumsum(np.round(pca_plot.explained_variance_ratio_, decimals=3)*100)
    plt.ylabel('% Variance Explained')
    plt.xlabel('No. of Features')
    plt.figtext(0.52, 0.15, "Optimal number of components (99% variance): " + str(len(train_dev.columns)-1) , wrap=True, horizontalalignment='center', fontsize=10)
    plt.title('PCA Analysis --> ' + file_name)
    plt.ylim(20,110)
    plt.xlim(0,100)
    plt.plot(var)
    plt.savefig(os.path.join(output_dir, 'PCA', 'Train', 'Reports', 'PCA_' + file_name + '.png'))
    #plt.show()
    plt.close()

    return train_dev, test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to perform feature selection or dimensionality reduction on training set and apply it to the test set. Different techniques are available for feature selection: RFE, Lasso_SelectFromModel, Trees_SelectFromModel, SFS and SelectKBest. The available technique for dimensionality reduction is PCA.')
    parser.add_argument("input_train", help="Path to specific training set (Train) or folder with multiple training sets.")
    parser.add_argument("input_test", help="Path to specific testing set (Test) or folder with the corresponding test sets.")
    parser.add_argument("-fs", "--fs_techniques", nargs='+', help="Feature selection and/or dimensionality reduction techniques to use: RFE [RFE], Lasso_SelectFromModel [LSFM], Trees_SelectFromModel [TSFM], SFS [SFS], SelectKBest [SKB] and/or PCA [PCA]. Multiple options are accepted. The [all] option will apply all techniques. Default option: [RFE].", default=['RFE'])
    parser.add_argument("-o", "--output_dir", help="Path to directory for created Train/Test sets with feature selection and/or dimensionality reduction. Default option: current directory.", default=os.getcwd())
    args = vars(parser.parse_args())
    input_train = args['input_train']
    input_test = args['input_test']
    fs_techniques = args['fs_techniques']
    output_dir = args['output_dir']
    for val in fs_techniques:
        if val not in ['all', 'RFE', 'LSFM', 'TSFM', 'SFS', 'SKB', 'PCA']:
            parser.error("'--fs_technique' values must be [RFE], [LSFM], [TSFM], [SFS], [SKB], [PCA] and/or [all]. Multiple options are accepted.")

    trains = {}
    tests = {}

    if(os.path.isdir(input_train) and os.path.isdir(input_test)):
        for file in os.listdir(input_train):
            if file.endswith(".csv"):
                train_name = os.path.splitext(file)[0].replace('train','')
                print('Reading training set from: ' + str(input_train + file))
                trains[train_name] = pd.read_csv(os.path.join(input_train,file), delimiter=";")
        for file in os.listdir(input_test):
            if file.endswith(".csv"):
                test_name = os.path.splitext(file)[0].replace('test','')
                print('Reading test set from: ' + str(input_test + file))
                tests[test_name] = pd.read_csv(os.path.join(input_test,file), delimiter=";")

    elif(not os.path.isdir(input_train) and not os.path.isdir(input_test)):
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

    for i in range(len(trains.keys())):
        train_name = list(trains.keys())[i]
        test_name = list(tests.keys())[i]
        if (not trains[train_name].isnull().values.any()):

            if 'all' in fs_techniques:
                print('\n[train' + train_name + '] Performing RFE feature selection...')
                Path(os.path.join(output_dir, "RFE")).mkdir(parents=True, exist_ok=True)
                Path(os.path.join(output_dir, "RFE", "Train")).mkdir(parents=True, exist_ok=True)
                Path(os.path.join(output_dir, "RFE", "Test")).mkdir(parents=True, exist_ok=True)
                Path(os.path.join(output_dir, "RFE", "Train", "Reports")).mkdir(parents=True, exist_ok=True)
                train_RFE, test_RFE = rfe(trains[train_name], tests[test_name], 'train' + train_name)
                print('RFE feature selection report saved in: ' + os.path.join(output_dir, "RFE", "Train", 'Reports'))
                print('Saving ' + str(os.path.join(output_dir, "RFE", "Train", 'train' + train_name + "_RFE.csv")))
                train_RFE.to_csv(os.path.join(output_dir, "RFE", "Train", 'train' + train_name + "_RFE.csv"), index=False, sep=';')
                print('Saving ' + str(os.path.join(output_dir, "RFE", "Test", 'test' + test_name + "_RFE.csv")))
                test_RFE.to_csv(os.path.join(output_dir, "RFE", "Test", 'test' + test_name + "_RFE.csv"), index=False, sep=';')

                print('\n[train' + train_name + '] Performing Lasso_SFM feature selection...')
                Path(os.path.join(output_dir, "Lasso_SFM")).mkdir(parents=True, exist_ok=True)
                Path(os.path.join(output_dir, "Lasso_SFM", "Train")).mkdir(parents=True, exist_ok=True)
                Path(os.path.join(output_dir, "Lasso_SFM", "Test")).mkdir(parents=True, exist_ok=True)
                Path(os.path.join(output_dir, "Lasso_SFM", "Train", "Reports")).mkdir(parents=True, exist_ok=True)
                train_LSFM, test_LSFM = lasso_sfm(trains[train_name], tests[test_name], 'train' + train_name)
                print('Lasso_SFM feature selection report saved in: ' + os.path.join(output_dir, "Lasso_SFM", "Train", 'Reports'))
                print('Saving ' + str(os.path.join(output_dir, "Lasso_SFM", "Train", 'train' + train_name + "_LSFM.csv")))
                train_LSFM.to_csv(os.path.join(output_dir, "Lasso_SFM", "Train", 'train' + train_name + "_LSFM.csv"), index=False, sep=';')
                print('Saving ' + str(os.path.join(output_dir, "Lasso_SFM", "Test", 'test' + test_name + "_LSFM.csv")))
                test_LSFM.to_csv(os.path.join(output_dir, "Lasso_SFM", "Test", 'test' + test_name + "_LSFM.csv"), index=False, sep=';')

                print('\n[train' + train_name + '] Performing Trees_SFM feature selection...')
                Path(os.path.join(output_dir, "Trees_SFM")).mkdir(parents=True, exist_ok=True)
                Path(os.path.join(output_dir, "Trees_SFM", "Train")).mkdir(parents=True, exist_ok=True)
                Path(os.path.join(output_dir, "Trees_SFM", "Test")).mkdir(parents=True, exist_ok=True)
                Path(os.path.join(output_dir, "Trees_SFM", "Train", "Reports")).mkdir(parents=True, exist_ok=True)
                train_TSFM, test_TSFM = trees_sfm(trains[train_name], tests[test_name], 'train' + train_name)
                print('Trees_SFM feature selection report saved in: ' + os.path.join(output_dir, "Trees_SFM", "Train", 'Reports'))
                print('Saving ' + str(os.path.join(output_dir, "Trees_SFM", "Train", 'train' + train_name + "_TSFM.csv")))
                train_TSFM.to_csv(os.path.join(output_dir, "Trees_SFM", "Train", 'train' + train_name + "_TSFM.csv"), index=False, sep=';')
                print('Saving ' + str(os.path.join(output_dir, "Trees_SFM", "Test", 'test' + test_name + "_TSFM.csv")))
                test_TSFM.to_csv(os.path.join(output_dir, "Trees_SFM", "Test", 'test' + test_name + "_TSFM.csv"), index=False, sep=';')

                print('\n[train' + train_name + '] Performing SFS feature selection...')
                Path(os.path.join(output_dir, "SFS")).mkdir(parents=True, exist_ok=True)
                Path(os.path.join(output_dir, "SFS", "Train")).mkdir(parents=True, exist_ok=True)
                Path(os.path.join(output_dir, "SFS", "Test")).mkdir(parents=True, exist_ok=True)
                Path(os.path.join(output_dir, "SFS", "Train", "Reports")).mkdir(parents=True, exist_ok=True)
                train_SFS, test_SFS = sfs(trains[train_name], tests[test_name], 'train' + train_name)
                print('SFS feature selection report saved in: ' + os.path.join(output_dir, "SFS", "Train", 'Reports'))
                print('Saving ' + str(os.path.join(output_dir, "SFS", "Train", 'train' + train_name + "_SFS.csv")))
                train_SFS.to_csv(os.path.join(output_dir, "SFS", "Train", 'train' + train_name + "_SFS.csv"), index=False, sep=';')
                print('Saving ' + str(os.path.join(output_dir, "SFS", "Test", 'test' + test_name + "_SFS.csv")))
                test_SFS.to_csv(os.path.join(output_dir, "SFS", "Test", 'test' + test_name + "_SFS.csv"), index=False, sep=';')

                # Con los datasets con imputación de MV diff, el IMC puede tener valor -1 (negativo), por lo que se sustituye provisionalmente por 8 (positivo) para aplicar SelectKBest
                train_copy = trains[train_name].copy()
                test_copy = tests[test_name].copy()
                train_copy['imc'] = train_copy['imc'].replace(-1, 8)
                test_copy['imc'] = test_copy['imc'].replace(-1, 8)
                if (not (train_copy.values < 0).any()):
                    print('\n[train' + train_name + '] Performing SelectKBest feature selection...')
                    Path(os.path.join(output_dir, "SelectKBest")).mkdir(parents=True, exist_ok=True)
                    Path(os.path.join(output_dir, "SelectKBest", "Train")).mkdir(parents=True, exist_ok=True)
                    Path(os.path.join(output_dir, "SelectKBest", "Test")).mkdir(parents=True, exist_ok=True)
                    Path(os.path.join(output_dir, "SelectKBest", "Train", "Reports")).mkdir(parents=True, exist_ok=True)
                    train_SKB, test_SKB = selectkbest(train_copy, test_copy, 'train' + train_name)
                    if ("imc" in train_SKB.columns):
                        train_SKB['imc'] = train_SKB['imc'].replace(8, -1)
                        test_SKB['imc'] = test_SKB['imc'].replace(8, -1)
                    print('SelectKBest feature selection report saved in: ' + os.path.join(output_dir, "SelectKBest", "Train", 'Reports'))
                    print('Saving ' + str(os.path.join(output_dir, "SelectKBest", "Train", 'train' + train_name + "_SKB.csv")))
                    train_SKB.to_csv(os.path.join(output_dir, "SelectKBest", "Train", 'train' + train_name + "_SKB.csv"), index=False, sep=';')
                    print('Saving ' + str(os.path.join(output_dir, "SelectKBest", "Test", 'test' + test_name + "_SKB.csv")))
                    test_SKB.to_csv(os.path.join(output_dir, "SelectKBest", "Test", 'test' + test_name + "_SKB.csv"), index=False, sep=';')
                else:
                    print("\nSelectKBest cannot be applied to '" + 'train' + str(train_name) + "' and '" + str(
                        'test' + test_name) + "' datasets because they contain negative values after standardization/normalization.")

                print('\n[train' + train_name + '] Performing PCA dimensionality reduction...')
                Path(os.path.join(output_dir, "PCA")).mkdir(parents=True, exist_ok=True)
                Path(os.path.join(output_dir, "PCA", "Train")).mkdir(parents=True, exist_ok=True)
                Path(os.path.join(output_dir, "PCA", "Test")).mkdir(parents=True, exist_ok=True)
                Path(os.path.join(output_dir, "PCA", "Train", "Reports")).mkdir(parents=True, exist_ok=True)
                train_PCA, test_PCA = pca(trains[train_name], tests[test_name], 'train' + train_name)
                print('PCA dimensionality reduction report saved in: ' + os.path.join(output_dir, "PCA", "Train", 'Reports'))
                print('Saving ' + str(os.path.join(output_dir, "PCA", "Train", 'train' + train_name + "_PCA.csv")))
                train_PCA.to_csv(os.path.join(output_dir, "PCA", "Train", 'train' + train_name + "_PCA.csv"), index=False, sep=';')
                print('Saving ' + str(os.path.join(output_dir, "PCA", "Test", 'test' + test_name + "_PCA.csv")))
                test_PCA.to_csv(os.path.join(output_dir, "PCA", "Test", 'test' + test_name + "_PCA.csv"), index=False, sep=';')

            else:
                for val in fs_techniques:
                    if (val == 'RFE'):
                        print('\n[train' + train_name + '] Performing RFE feature selection...')
                        Path(os.path.join(output_dir, "RFE")).mkdir(parents=True, exist_ok=True)
                        Path(os.path.join(output_dir, "RFE", "Train")).mkdir(parents=True, exist_ok=True)
                        Path(os.path.join(output_dir, "RFE", "Test")).mkdir(parents=True, exist_ok=True)
                        Path(os.path.join(output_dir, "RFE", "Train", "Reports")).mkdir(parents=True, exist_ok=True)
                        train_RFE, test_RFE = rfe(trains[train_name], tests[test_name], 'train' + train_name)
                        print('RFE feature selection report saved in: ' + os.path.join(output_dir, "RFE", "Train", 'Reports'))
                        print('Saving ' + str(os.path.join(output_dir, "RFE", "Train", 'train' + train_name + "_RFE.csv")))
                        train_RFE.to_csv(os.path.join(output_dir, "RFE", "Train", 'train' + train_name + "_RFE.csv"), index=False, sep=';')
                        print('Saving ' + str(os.path.join(output_dir, "RFE", "Test", 'test' + test_name + "_RFE.csv")))
                        test_RFE.to_csv(os.path.join(output_dir, "RFE", "Test", 'test' + test_name + "_RFE.csv"), index=False, sep=';')
                    elif (val == 'LSFM'):
                        print('\n[train' + train_name + '] Performing Lasso_SFM feature selection...')
                        Path(os.path.join(output_dir, "Lasso_SFM")).mkdir(parents=True, exist_ok=True)
                        Path(os.path.join(output_dir, "Lasso_SFM", "Train")).mkdir(parents=True, exist_ok=True)
                        Path(os.path.join(output_dir, "Lasso_SFM", "Test")).mkdir(parents=True, exist_ok=True)
                        Path(os.path.join(output_dir, "Lasso_SFM", "Train", "Reports")).mkdir(parents=True, exist_ok=True)
                        train_LSFM, test_LSFM = lasso_sfm(trains[train_name], tests[test_name], 'train' + train_name)
                        print('Lasso_SFM feature selection report saved in: ' + os.path.join(output_dir, "Lasso_SFM", "Train", 'Reports'))
                        print('Saving ' + str(os.path.join(output_dir, "Lasso_SFM", "Train", 'train' + train_name + "_LSFM.csv")))
                        train_LSFM.to_csv(os.path.join(output_dir, "Lasso_SFM", "Train", 'train' + train_name + "_LSFM.csv"), index=False, sep=';')
                        print('Saving ' + str(os.path.join(output_dir, "Lasso_SFM", "Test", 'test' + test_name + "_LSFM.csv")))
                        test_LSFM.to_csv(os.path.join(output_dir, "Lasso_SFM", "Test", 'test' + test_name + "_LSFM.csv"), index=False, sep=';')
                    elif (val == 'TSFM'):
                        print('\n[train' + train_name + '] Performing Trees_SFM feature selection...')
                        Path(os.path.join(output_dir, "Trees_SFM")).mkdir(parents=True, exist_ok=True)
                        Path(os.path.join(output_dir, "Trees_SFM", "Train")).mkdir(parents=True, exist_ok=True)
                        Path(os.path.join(output_dir, "Trees_SFM", "Test")).mkdir(parents=True, exist_ok=True)
                        Path(os.path.join(output_dir, "Trees_SFM", "Train", "Reports")).mkdir(parents=True, exist_ok=True)
                        train_TSFM, test_TSFM = trees_sfm(trains[train_name], tests[test_name], 'train' + train_name)
                        print('Trees_SFM feature selection report saved in: ' + os.path.join(output_dir, "Trees_SFM", "Train", 'Reports'))
                        print('Saving ' + str(os.path.join(output_dir, "Trees_SFM", "Train", 'train' + train_name + "_TSFM.csv")))
                        train_TSFM.to_csv(os.path.join(output_dir, "Trees_SFM", "Train", 'train' + train_name + "_TSFM.csv"), index=False, sep=';')
                        print('Saving ' + str(os.path.join(output_dir, "Trees_SFM", "Test", 'test' + test_name + "_TSFM.csv")))
                        test_TSFM.to_csv(os.path.join(output_dir, "Trees_SFM", "Test", 'test' + test_name + "_TSFM.csv"), index=False, sep=';')
                    elif (val == 'SFS'):
                        print('\n[train' + train_name + '] Performing SFS feature selection...')
                        Path(os.path.join(output_dir, "SFS")).mkdir(parents=True, exist_ok=True)
                        Path(os.path.join(output_dir, "SFS", "Train")).mkdir(parents=True, exist_ok=True)
                        Path(os.path.join(output_dir, "SFS", "Test")).mkdir(parents=True, exist_ok=True)
                        Path(os.path.join(output_dir, "SFS", "Train", "Reports")).mkdir(parents=True, exist_ok=True)
                        train_SFS, test_SFS = sfs(trains[train_name], tests[test_name], 'train' + train_name)
                        print('SFS feature selection report saved in: ' + os.path.join(output_dir, "SFS", "Train", 'Reports'))
                        print('Saving ' + str(os.path.join(output_dir, "SFS", "Train", 'train' + train_name + "_SFS.csv")))
                        train_SFS.to_csv(os.path.join(output_dir, "SFS", "Train", 'train' + train_name + "_SFS.csv"), index=False, sep=';')
                        print('Saving ' + str(os.path.join(output_dir, "SFS", "Test", 'test' + test_name + "_SFS.csv")))
                        test_SFS.to_csv(os.path.join(output_dir, "SFS", "Test", 'test' + test_name + "_SFS.csv"), index=False, sep=';')
                    elif (val == 'SKB'):
                        # Con los datasets con imputación de MV diff, el IMC puede tener valor -1 (negativo), por lo que se sustituye provisionalmente por 8 (positivo) para aplicar SelectKBest
                        train_copy = trains[train_name].copy()
                        test_copy = tests[test_name].copy()
                        train_copy['imc'] = train_copy['imc'].replace(-1, 8)
                        test_copy['imc'] = test_copy['imc'].replace(-1, 8)
                        if (not (train_copy.values < 0).any()):
                            print('\n[train' + train_name + '] Performing SelectKBest feature selection...')
                            Path(os.path.join(output_dir, "SelectKBest")).mkdir(parents=True, exist_ok=True)
                            Path(os.path.join(output_dir, "SelectKBest", "Train")).mkdir(parents=True, exist_ok=True)
                            Path(os.path.join(output_dir, "SelectKBest", "Test")).mkdir(parents=True, exist_ok=True)
                            Path(os.path.join(output_dir, "SelectKBest", "Train", "Reports")).mkdir(parents=True, exist_ok=True)
                            train_SKB, test_SKB = selectkbest(train_copy, test_copy, 'train' + train_name)
                            if ("imc" in train_SKB.columns):
                                train_SKB['imc'] = train_SKB['imc'].replace(8, -1)
                                test_SKB['imc'] = test_SKB['imc'].replace(8, -1)
                            print('SelectKBest feature selection report saved in: ' + os.path.join(output_dir, "SelectKBest", "Train", 'Reports'))
                            print('Saving ' + str(os.path.join(output_dir, "SelectKBest", "Train", 'train' + train_name + "_SKB.csv")))
                            train_SKB.to_csv(os.path.join(output_dir, "SelectKBest", "Train", 'train' + train_name + "_SKB.csv"), index=False, sep=';')
                            print('Saving ' + str(os.path.join(output_dir, "SelectKBest", "Test", 'test' + test_name + "_SKB.csv")))
                            test_SKB.to_csv(os.path.join(output_dir, "SelectKBest", "Test", 'test' + test_name + "_SKB.csv"), index=False, sep=';')
                        else:
                            print("\nSelectKBest cannot be applied to '" + 'train' + str(train_name) + "' and '" + str('test' + test_name) + "' datasets because they contain negative values after standardization/normalization.")
                    elif (val == 'PCA'):
                        print('\n[train' + train_name + '] Performing PCA dimensionality reduction...')
                        Path(os.path.join(output_dir, "PCA")).mkdir(parents=True, exist_ok=True)
                        Path(os.path.join(output_dir, "PCA", "Train")).mkdir(parents=True, exist_ok=True)
                        Path(os.path.join(output_dir, "PCA", "Test")).mkdir(parents=True, exist_ok=True)
                        Path(os.path.join(output_dir, "PCA", "Train", "Reports")).mkdir(parents=True, exist_ok=True)
                        train_PCA, test_PCA = pca(trains[train_name], tests[test_name], 'train' + train_name)
                        print('PCA dimensionality reduction report saved in: ' + os.path.join(output_dir, "PCA", "Train", 'Reports'))
                        print('Saving ' + str(os.path.join(output_dir, "PCA", "Train", 'train' + train_name + "_PCA.csv")))
                        train_PCA.to_csv(os.path.join(output_dir, "PCA", "Train", 'train' + train_name + "_PCA.csv"), index=False, sep=';')
                        print('Saving ' + str(os.path.join(output_dir, "PCA", "Test", 'test' + test_name + "_PCA.csv")))
                        test_PCA.to_csv(os.path.join(output_dir, "PCA", "Test", 'test' + test_name + "_PCA.csv"), index=False, sep=';')

        else:
            print("\nFeature selection / dimensionality reduction cannot be performed to '" + str(
                'train' + train_name) + "' and '" + str(
                'test' + test_name) + "' datasets because they contain missing values.")
