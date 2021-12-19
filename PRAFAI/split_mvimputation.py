import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def class_and_split(dataset):
    dataset['fecha'] = pd.to_datetime(dataset['fecha'])
    dataset['fecha'] = dataset['fecha'].astype(np.int64)

    # Only instances with class value (with missing values)
    dataset_con_clase = dataset[dataset['sigue_fa'].notnull()]

    # Train/Dev/Test split
    labels = dataset_con_clase['sigue_fa']
    data = dataset_con_clase.iloc[:, 1:len(dataset_con_clase.columns) - 1]
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, shuffle=True, stratify=labels)
    train = X_train
    train['sigue_fa'] = y_train
    test = X_test
    test['sigue_fa'] = y_test

    # Borrar instancias del dataset completo que están en el test
    test_indices = test.index.tolist()
    dataset_train = dataset.drop(test_indices)

    return train, test, dataset_train


def impute_mean(train, test, dataset_train):
    train_copy = train.copy()
    test_copy = test.copy()

    mean_dict = {}
    deleted_dict = set()
    for column in train_copy:
        if(column != "sigue_fa"):
            if (train_copy[column].isnull().all()):       # Si todos los ejemplos tienen missing value, se elimina la columna
                del train_copy[column]
                print("\tFeature '" + str(column) + "' was deleted because all values in training set were missing.")
                deleted_dict.add(column)
            elif(train_copy[column].dtype == np.float64):     # Se imputa la media de la columna si es una variable numérica
                column_mean = round(dataset_train[column].mean(),3)
                train_copy[column] = train_copy[column].fillna(column_mean)
                mean_dict[column] = column_mean
    train_copy['imc'] = train_copy['imc'].round()

    # Cambios en test
    for column in deleted_dict:
        del test_copy[column]
    for column in mean_dict:
        test_copy[column] = test_copy[column].fillna(mean_dict[column])
    test_copy['imc'] = test_copy['imc'].round()

    return train_copy, test_copy


def impute_median(train, test, dataset_train):
    train_copy = train.copy()
    test_copy = test.copy()

    median_dict = {}
    deleted_dict = set()
    for column in train_copy:
        if (column != "sigue_fa"):
            if (train_copy[column].isnull().all()):  # Si todos los ejemplos tienen missing value, se elimina la columna
                del train_copy[column]
                print("\tFeature '" + str(column) + "' was deleted because all values in training set were missing.")
                deleted_dict.add(column)
            elif (train_copy[column].dtype == np.float64):  # Se imputa la media de la columna si es una variable numérica
                column_median = round(dataset_train[column].median(), 3)
                train_copy[column] = train_copy[column].fillna(column_median)
                median_dict[column] = column_median
    train_copy['imc'] = train_copy['imc'].round()

    # Cambios en test
    for column in deleted_dict:
        del test_copy[column]
    for column in median_dict:
        test_copy[column] = test_copy[column].fillna(median_dict[column])
    test_copy['imc'] = test_copy['imc'].round()

    return train_copy, test_copy


def impute_prediction_linear_regression(train, test, dataset_train):
    Y_train = train['sigue_fa']
    Y_test = test['sigue_fa']
    dataset_train_copy = dataset_train.copy()
    train_copy = train.copy()
    test_copy = test.copy()
    del dataset_train_copy['id']
    del dataset_train_copy['sigue_fa']
    del train_copy['sigue_fa']
    del test_copy['sigue_fa']

    imputed_columns = []        # Se va a guardar las columnas en las que se van imputando missing values
    for column in train_copy:
        if(column != "sigue_fa"):
            if (train_copy[column].isnull().all()):       # Si todos los ejemplos tienen missing value, se elimina la columna
                del dataset_train_copy[column]
                del train_copy[column]
                del test_copy[column]
                print("\tFeature '" + str(column) + "' was deleted because all values in training set were missing.")
            elif((train_copy[column].dtype == np.float64) & (train_copy[column].isnull().any())):       # Si es numerica se van a imputar los missing values
                # Se escogen solo las columnas sin missing values y la actual (la que se quiere predecir)
                train_column = dataset_train_copy.drop(imputed_columns, axis=1)
                train_column = train_column.loc[:, (-train_column.isnull().any()) | (train_column.columns.isin([column]))]
                train_column_TRAIN = train_copy.drop(imputed_columns, axis=1)
                train_column_TRAIN = train_column_TRAIN.loc[:,(-train_column_TRAIN.isnull().any()) | (train_column_TRAIN.columns.isin([column]))]
                train_column_TEST = test_copy.drop(imputed_columns, axis=1)
                train_column_TEST = train_column_TEST.loc[:,(-train_column_TEST.isnull().any()) | (train_column_TEST.columns.isin([column]))]

                # Se guardan los ejemplos que tienen missing values en esa columna
                test_data = train_column[train_column[column].isnull()]
                test_data_TRAIN = train_column_TRAIN[train_column_TRAIN[column].isnull()]
                test_data_TEST = train_column_TEST[train_column_TEST[column].isnull()]
                # Se eliminan todos los ejemplos que tengan missing values en esa columna
                train_column_no_missing_values = train_column.dropna()

                # Valores de esa columna de los ejemplos SIN missing values (clase del entrenamiento)
                y_train = train_column_no_missing_values[column]
                # Ejemplos que NO tienen missing values en esa columna sin el valor de esa columna (conjunto de entrenamiento)
                X_train = train_column_no_missing_values.drop(column, axis=1)
                # Ejemplos que SI tienen missing values en esa columna sin el valor de esa columna (conjunto de test)
                X_test = test_data.drop(column, axis=1)
                X_test_TRAIN = test_data_TRAIN.drop(column, axis=1)
                X_test_TEST = test_data_TEST.drop(column, axis=1)

                # Regresion lineal
                model = LinearRegression()
                model.fit(X_train, y_train)

                # Array con las predicciones de los ejemplos que tenian missing values en esa columna
                y_pred = model.predict(X_test)
                y_pred = np.round(y_pred, 3)
                y_pred_TRAIN = model.predict(X_test_TRAIN)
                y_pred_TRAIN = np.round(y_pred_TRAIN, 3)
                y_pred_TEST = model.predict(X_test_TEST)
                y_pred_TEST = np.round(y_pred_TEST, 3)

                # Los valores que se han predicho como negativos se corrigen a 0
                y_pred[y_pred < 0] = 0
                y_pred_TRAIN[y_pred_TRAIN < 0] = 0
                y_pred_TEST[y_pred_TEST < 0] = 0

                # Establecer predicciones a la columna con el conjunto de datos completo
                m = train_copy[column].isna()
                train_copy.loc[m, column] = y_pred_TRAIN
                m = test_copy[column].isna()
                test_copy.loc[m, column] = y_pred_TEST

                imputed_columns.append(column)

    train_copy['imc'] = train_copy['imc'].round()
    test_copy['imc'] = test_copy['imc'].round()

    train_copy['sigue_fa'] = Y_train
    test_copy['sigue_fa'] = Y_test

    return train_copy, test_copy


def impute_different_value(train, test):
    train_copy = train.copy()
    test_copy = test.copy()

    diff_dict = set()
    deleted_dict = set()
    for column in train_copy:
        if (column != "sigue_fa"):
            if (train_copy[column].isnull().all()):  # Si todos los ejemplos tienen missing value, se elimina la columna
                del train_copy[column]
                print("\tFeature '" + str(column) + "' was deleted because all values in training set were missing.")
                deleted_dict.add(column)
            else:
                train_copy[column] = train_copy[column].fillna(-1)
                diff_dict.add(column)

    # Cambios en test
    for column in deleted_dict:
        del test_copy[column]
    for column in diff_dict:
        test_copy[column] = test_copy[column].fillna(-1)

    return train_copy, test_copy


def not_impute(train, test):
    train_copy = train.copy()
    test_copy = test.copy()

    for column in train_copy:
        if (column != "sigue_fa"):
            if (train_copy[column].isnull().all()):  # Si todos los ejemplos tienen missing value, se elimina la columna
                del train_copy[column]
                del test_copy[column]

    return train_copy, test_copy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to divide the dataset into Train/Test subsets (80%/20%) and imputation of missing values. Only labeled items will be taken. Different techniques are available for the imputation of missing values: arithmetic mean, median, prediction by linear regression or different value.')
    parser.add_argument("input_dataset", help="Path to PRAFAI dataset.")
    parser.add_argument("-mv", "--mv_techniques", nargs='+', help="Missing values imputation techniques to use: arithmetic mean [mean], median [med], prediction [pred] and/or different value [diff]. Multiple options are accepted. The [all] option will apply all techniques. Default option: missing values are preserved.", default=[])
    parser.add_argument("-o", "--output_dir", help="Path to directory for the created Train/Test sets. Default option: current directory.", default=os.getcwd())
    args = vars(parser.parse_args())
    input_dataset = args['input_dataset']
    mv_techniques = args['mv_techniques']
    output_dir = args['output_dir']
    for val in mv_techniques:
        if val not in ['all', 'mean', 'med', 'pred', 'diff']:
            parser.error("'--mv_technique' values must be [mean], [med], [pred], [diff], [all] and/or empty. Multiple options are accepted.")


    print('Reading PRAFAI dataset from: ' + str(input_dataset))
    dataset = pd.read_csv(input_dataset, delimiter=";")

    print('\nCreating Train/Test subsets...')
    train, test, dataset_train = class_and_split(dataset)

    if 'all' in mv_techniques:
        print('\nImputing arithmetic mean...')
        train_mean, test_mean = impute_mean(train, test, dataset_train)
        print('Saving ' + str(os.path.join(output_dir,"train_mean.csv")))
        train_mean.to_csv(os.path.join(output_dir,"train_mean.csv"),index=False, sep=';')
        print('Saving ' + str(os.path.join(output_dir,"test_mean.csv")))
        test_mean.to_csv(os.path.join(output_dir,"test_mean.csv"),index=False, sep=';')

        print('\nImputing median...')
        train_median, test_median = impute_median(train, test, dataset_train)
        print('Saving ' + str(os.path.join(output_dir,"train_med.csv")))
        train_median.to_csv(os.path.join(output_dir,"train_med.csv"),index=False, sep=';')
        print('Saving ' + str(os.path.join(output_dir,"test_med.csv")))
        test_median.to_csv(os.path.join(output_dir,"test_med.csv"),index=False, sep=';')

        print('\nImputing predicted values by linear regression...')
        train_pred, test_pred = impute_prediction_linear_regression(train, test, dataset_train)
        print('Saving ' + str(os.path.join(output_dir,"train_pred.csv")))
        train_pred.to_csv(os.path.join(output_dir,"train_pred.csv"),index=False, sep=';')
        print('Saving ' + str(os.path.join(output_dir,"test_pred.csv")))
        test_pred.to_csv(os.path.join(output_dir,"test_pred.csv"),index=False, sep=';')

        print('\nImputing different value...')
        train_diff, test_diff = impute_different_value(train, test)
        print('Saving ' + str(os.path.join(output_dir,"train_diff.csv")))
        train_diff.to_csv(os.path.join(output_dir,"train_diff.csv"),index=False, sep=';')
        print('Saving ' + str(os.path.join(output_dir,"test_diff.csv")))
        test_diff.to_csv(os.path.join(output_dir,"test_diff.csv"),index=False, sep=';')

        train_mv, test_mv = not_impute(train, test)
        train_mv.to_csv(os.path.join(output_dir,"train_mv.csv"),index=False, sep=';')
        test_mv.to_csv(os.path.join(output_dir,"test_mv.csv"),index=False, sep=';')

    elif (len(mv_techniques) == 0):
        train_mv, test_mv = not_impute(train, test)
        train_mv.to_csv(os.path.join(output_dir,"train_mv.csv"),index=False, sep=';')
        test_mv.to_csv(os.path.join(output_dir,"test_mv.csv"),index=False, sep=';')

    else:
        for val in mv_techniques:
            if(val == 'mean'):
                print('\nImputing arithmetic mean...')
                train_mean, test_mean = impute_mean(train, test, dataset_train)
                print('Saving ' + str(os.path.join(output_dir, "train_mean.csv")))
                train_mean.to_csv(os.path.join(output_dir, "train_mean.csv"), index=False, sep=';')
                print('Saving ' + str(os.path.join(output_dir, "test_mean.csv")))
                test_mean.to_csv(os.path.join(output_dir, "test_mean.csv"), index=False, sep=';')
            elif (val == 'med'):
                print('\nImputing median...')
                train_median, test_median = impute_median(train, test, dataset_train)
                print('Saving ' + str(os.path.join(output_dir, "train_med.csv")))
                train_median.to_csv(os.path.join(output_dir, "train_med.csv"), index=False, sep=';')
                print('Saving ' + str(os.path.join(output_dir, "test_med.csv")))
                test_median.to_csv(os.path.join(output_dir, "test_med.csv"), index=False, sep=';')
            elif (val == 'pred'):
                print('\nImputing predicted values by linear regression...')
                train_pred, test_pred = impute_prediction_linear_regression(train, test, dataset_train)
                print('Saving ' + str(os.path.join(output_dir, "train_pred.csv")))
                train_pred.to_csv(os.path.join(output_dir, "train_pred.csv"), index=False, sep=';')
                print('Saving ' + str(os.path.join(output_dir, "test_pred.csv")))
                test_pred.to_csv(os.path.join(output_dir, "test_pred.csv"), index=False, sep=';')
            elif (val == 'diff'):
                print('\nImputing different value...')
                train_diff, test_diff = impute_different_value(train, test)
                print('Saving ' + str(os.path.join(output_dir, "train_diff.csv")))
                train_diff.to_csv(os.path.join(output_dir, "train_diff.csv"), index=False, sep=';')
                print('Saving ' + str(os.path.join(output_dir, "test_diff.csv")))
                test_diff.to_csv(os.path.join(output_dir, "test_diff.csv"), index=False, sep=';')



