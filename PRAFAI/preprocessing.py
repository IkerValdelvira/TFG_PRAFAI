import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

from pathlib import Path
import argparse
import os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, QuantileTransformer, Normalizer


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


def standardization(train, test, type):
    # type:
    #   StandardScaler
    #   MinMaxScaler [0,1]
    #   MaxAbsScaler [-1,1]
    #   QuantileTransformer(random_state=0)
    #   Normalizer

    y_train = train["sigue_fa"]
    X_train = train.drop("sigue_fa", 1)
    y_test = test["sigue_fa"]
    X_test = test.drop("sigue_fa", 1)

    columns = train.columns.tolist()
    cols_standardize_aux = ['fecha', 'urea', 'creatinina', 'albumina', 'glucosa', 'hba1c', 'potasio', 'calcio', 'hdl', 'no_hdl', 'ldl', 'colesterol', 'ntprobnp', 'bnp', 'troponina_tni', 'troponina_tnt', 'dimero_d', 'fibrinogeno', 'aldosterona', 'leucocitos', 'pct', 'tsh', 't4l', 'sodio', 'vsg', 'tl3', 'fevi', 'diametro_ai', 'area_ai', 'numero_ingresos', 'numero_dias_desde_ingreso_hasta_evento', 'numero_dias_ingresado', 'edad', 'talla', 'peso']
    cols_standardize = [var for var in cols_standardize_aux if var in list(columns)]
    cols_not_standardize_aux = ['tipo_fa', 'ecocardiograma', 'ecocardiograma_contraste', 'electrocardiograma', 'cardioversion', 'ablacion', 'marcapasos_dai', 'etilogia_cardiologica', 'depresion', 'alcohol', 'drogodependencia', 'ansiedad', 'demencia', 'insuficiencia_renal', 'menopausia', 'osteoporosis', 'diabetes_tipo1', 'diabetes_tipo2', 'dislipidemia', 'hipercolesterolemia', 'fibrilacion_palpitacion', 'flutter', 'insuficiencia_cardiaca', 'fumador', 'sahos', 'hipertiroidismo', 'sindrome_metabolico', 'hipertension_arterial', 'cardiopatia_isquemica', 'ictus', 'miocardiopatia', 'otras_arritmias_psicogena_ritmo_bigeminal', 'bloqueos_rama', 'bloqueo_auriculoventricular', 'bradicardia', 'contracciones_prematuras_ectopia_extrasistolica', 'posoperatoria', 'sinusal_coronaria', 'valvula_mitral_reumaticas', 'otras_valvulopatias', 'valvulopatia_mitral_congenita', 'arteriopatia_periferica', 'epoc', 'genero', 'pensionista', 'residenciado', 'imc', 'b01a', 'n02ba01', 'a02bc', 'c03', 'g03a', 'a10', 'n06a', 'n05a', 'n05b', 'c01', 'c01b', 'c02', 'c04', 'c07', 'c08', 'c09', 'c10', 'polimedicacion']
    cols_not_standardize = [var for var in cols_not_standardize_aux if var in list(columns)]
    if(type == "StandardScaler"):
        ct = ColumnTransformer([('columntransformer', StandardScaler(), cols_standardize)], remainder='passthrough')
    elif(type == "MinMaxScaler"):
        ct = ColumnTransformer([('columntransformer', MinMaxScaler(), cols_standardize)], remainder='passthrough')
    elif (type == "MaxAbsScaler"):
        ct = ColumnTransformer([('columntransformer', MaxAbsScaler(), cols_standardize)], remainder='passthrough')
    elif (type == "QuantileTransformer"):
        ct = ColumnTransformer([('columntransformer', QuantileTransformer(random_state=0), cols_standardize)], remainder='passthrough')
    elif (type == "Normalizer"):
        ct = ColumnTransformer([('columntransformer', Normalizer(), cols_standardize)], remainder='passthrough')
    ct.fit(X_train)
    X_train_array = ct.transform(X_train)
    X_test_array = ct.transform(X_test)

    array_columns = cols_standardize + cols_not_standardize
    train = pd.DataFrame(X_train_array, index=X_train.index, columns=array_columns)
    test = pd.DataFrame(X_test_array, index=X_test.index, columns=array_columns)

    train['sigue_fa'] = y_train
    test['sigue_fa'] = y_test

    return train, test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to divide the dataset into Train/Test subsets (80%/20%), imputate missing values in numeric features and standardize/normalize numeric features of the training set (Train), and apply the same imputation and rescaling to the testing set (Test). Only labeled items will be taken. Different techniques are available for the imputation of missing values: arithmetic mean, median, prediction by linear regression or different value. Different techniques are available for standardization/normalization: StandardScaler, MinMaxScaler, MaxAbsScaler, QuantileTransformer or Normalizer.')
    parser.add_argument("input_dataset", help="Path to PRAFAI dataset.")
    parser.add_argument("-mv", "--mv_techniques", nargs='+', help="Missing values imputation techniques to use: arithmetic mean [mean], median [med], prediction [pred] and/or different value [diff]. Multiple options are accepted. The [all] option will apply all techniques. Default option: no imputation is applied.", default=[])
    parser.add_argument("-std", "--std_techniques", nargs='+', help="Standardization/Normalization techniques to use: StandardScaler [SS], MinMaxScaler [MMS], MaxAbsScaler [MAS], QuantileTransformer [QT] and/or Normalizer [N]. Multiple options are accepted. The [all] option will apply all techniques. Default option: no standardization/normalization is applied.", default=[])
    parser.add_argument("-o", "--output_dir", help="Path to directory for the created Train/Test sets. Default option: current directory.", default=os.getcwd())
    args = vars(parser.parse_args())
    input_dataset = args['input_dataset']
    mv_techniques = args['mv_techniques']
    std_techniques = args['std_techniques']
    output_dir = args['output_dir']
    Path(os.path.join(output_dir, "Train")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_dir, "Test")).mkdir(parents=True, exist_ok=True)
    for val in mv_techniques:
        if val not in ['all', 'mean', 'med', 'pred', 'diff']:
            parser.error("'--mv_technique' values must be [mean], [med], [pred], [diff], [all] and/or empty. Multiple options are accepted.")
    for val in std_techniques:
        if val not in ['all', 'SS', 'MMS', 'MAS', 'QT', 'N']:
            parser.error(
                "'--std_technique' values must be [SS], [MMS], [MAS], [QT], [N] and/or [all]. Multiple options are accepted.")


    print('Reading PRAFAI dataset from: ' + str(input_dataset))
    dataset = pd.read_csv(input_dataset, delimiter=";")

    print('\nCreating Train/Test subsets...')
    train, test, dataset_train = class_and_split(dataset)

    trains = {}
    tests = {}

    if 'all' in mv_techniques:
        print('\nImputing arithmetic mean...')
        train_mean, test_mean = impute_mean(train, test, dataset_train)
        print("Creating 'train_mean' set...")
        trains['train_mean'] = train_mean
        print("Creating 'test_mean' set...")
        tests['test_mean'] = test_mean

        print('\nImputing median...')
        train_median, test_median = impute_median(train, test, dataset_train)
        print("Creating 'train_med' set...")
        trains['train_med'] = train_median
        print("Creating 'test_med' set...")
        tests['test_med'] = test_median

        print('\nImputing predicted values by linear regression...')
        train_pred, test_pred = impute_prediction_linear_regression(train, test, dataset_train)
        print("Creating 'train_pred' set...")
        trains['train_pred'] = train_pred
        print("Creating 'test_pred' set...")
        tests['test_pred'] = test_pred

        print('\nImputing different value...')
        train_diff, test_diff = impute_different_value(train, test)
        print("Creating 'train_diff' set...")
        trains['train_diff'] = train_diff
        print("Creating 'test_diff' set...")
        tests['test_diff'] = test_diff

        print('\nNot imputing missing values...')
        train_mv, test_mv = not_impute(train, test)
        print("Creating 'train_mv' set...")
        trains['train_mv'] = train_mv
        print("Creating 'test_mv' set...")
        tests['test_mv'] = test_mv

    elif (len(mv_techniques) == 0):
        print('\nNot imputing missing values...')
        train_mv, test_mv = not_impute(train, test)
        print("Creating 'train_mv' set...")
        trains['train_mv'] = train_mv
        print("Creating 'test_mv' set...")
        tests['test_mv'] = test_mv

    else:
        for val in mv_techniques:
            if(val == 'mean'):
                print('\nImputing arithmetic mean...')
                train_mean, test_mean = impute_mean(train, test, dataset_train)
                print("Creating 'train_mean' set...")
                trains['train_mean'] = train_mean
                print("Creating 'test_mean' set...")
                tests['test_mean'] = test_mean
            elif (val == 'med'):
                print('\nImputing median...')
                train_median, test_median = impute_median(train, test, dataset_train)
                print("Creating 'train_med' set...")
                trains['train_med'] = train_median
                print("Creating 'test_med' set...")
                tests['test_med'] = test_median
            elif (val == 'pred'):
                print('\nImputing predicted values by linear regression...')
                train_pred, test_pred = impute_prediction_linear_regression(train, test, dataset_train)
                print("Creating 'train_pred' set...")
                trains['train_pred'] = train_pred
                print("Creating 'test_pred' set...")
                tests['test_pred'] = test_pred
            elif (val == 'diff'):
                print('\nImputing different value...')
                train_diff, test_diff = impute_different_value(train, test)
                print("Creating 'train_diff' set...")
                trains['train_diff'] = train_diff
                print("Creating 'test_diff' set...")
                tests['test_diff'] = test_diff

    for i in range(len(trains.keys())):
        train_name = list(trains.keys())[i]
        test_name = list(tests.keys())[i]

        if 'all' in std_techniques:
            print("\nPreprocessing '" + str(train_name) + "' and '" + str(test_name) + "' with StandardScaler...")
            train_StandardScaler, test_StandardScaler = standardization(trains[train_name], tests[test_name], "StandardScaler")
            print('Saving ' + str(os.path.join(output_dir, "Train", str(train_name)+"_StandardScaler.csv")))
            train_StandardScaler.to_csv(os.path.join(output_dir, "Train", str(train_name)+"_StandardScaler.csv"), index=False, sep=';')
            print('Saving ' + str(os.path.join(output_dir, "Test", str(test_name)+"_StandardScaler.csv")))
            test_StandardScaler.to_csv(os.path.join(output_dir, "Test", str(test_name)+"_StandardScaler.csv"), index=False, sep=';')

            print("\nPreprocessing '" + str(train_name) + "' and '" + str(test_name) + "' with MinMaxScaler...")
            train_MinMaxScaler, test_MinMaxScaler = standardization(trains[train_name], tests[test_name], "MinMaxScaler")
            print('Saving ' + str(os.path.join(output_dir, "Train", str(train_name)+"_MinMaxScaler.csv")))
            train_MinMaxScaler.to_csv(os.path.join(output_dir, "Train", str(train_name)+"_MinMaxScaler.csv"), index=False, sep=';')
            print('Saving ' + str(os.path.join(output_dir, "Test", str(test_name)+"_MinMaxScaler.csv")))
            test_MinMaxScaler.to_csv(os.path.join(output_dir, "Test", str(test_name)+"_MinMaxScaler.csv"), index=False, sep=';')

            print("\nPreprocessing '" + str(train_name) + "' and '" + str(test_name) + "' with MaxAbsScaler...")
            train_MaxAbsScaler, test_MaxAbsScaler = standardization(trains[train_name], tests[test_name], "MaxAbsScaler")
            print('Saving ' + str(os.path.join(output_dir, "Train", str(train_name)+"_MaxAbsScaler.csv")))
            train_MaxAbsScaler.to_csv(os.path.join(output_dir, "Train", str(train_name)+"_MaxAbsScaler.csv"), index=False, sep=';')
            print('Saving ' + str(os.path.join(output_dir, "Test", str(test_name)+"_MaxAbsScaler.csv")))
            test_MaxAbsScaler.to_csv(os.path.join(output_dir, "Test", str(test_name)+"_MaxAbsScaler.csv"), index=False, sep=';')

            print("\nPreprocessing '" + str(train_name) + "' and '" + str(test_name) + "' with QuantileTransformer...")
            train_QuantileTransformer, test_QuantileTransformer = standardization(trains[train_name], tests[test_name], "QuantileTransformer")
            print('Saving ' + str(os.path.join(output_dir, "Train", str(train_name)+"_QuantileTransformer.csv")))
            train_QuantileTransformer.to_csv(os.path.join(output_dir, "Train", str(train_name)+"_QuantileTransformer.csv"), index=False, sep=';')
            print('Saving ' + str(os.path.join(output_dir, "Test", str(test_name)+"_QuantileTransformer.csv")))
            test_QuantileTransformer.to_csv(os.path.join(output_dir, "Test", str(test_name)+"_QuantileTransformer.csv"), index=False, sep=';')

            if(not trains[train_name].isnull().values.any()):
                print("\nPreprocessing '" + str(train_name) + "' and '" + str(test_name) + "' with Normalizer...")
                train_Normalizer, test_Normalizer = standardization(trains[train_name], tests[test_name], "Normalizer")
                print('Saving ' + str(os.path.join(output_dir, "Train", str(train_name)+"_Normalizer.csv")))
                train_Normalizer.to_csv(os.path.join(output_dir, "Train", str(train_name)+"_Normalizer.csv"), index=False, sep=';')
                print('Saving ' + str(os.path.join(output_dir, "Test", str(test_name)+"_Normalizer.csv")))
                test_Normalizer.to_csv(os.path.join(output_dir, "Test", str(test_name)+"_Normalizer.csv"), index=False, sep=';')
            else:
                print("\nNormalizer cannot be applied to '" + str(train_name) + "' and '" + str(test_name) + "' datasets because they contain missing values.")

        elif (len(std_techniques) == 0):
            print('\nNot applying standardization/normalization techniques...')
            print('Saving ' + str(os.path.join(output_dir, "Train", str(train_name) + ".csv")))
            trains[train_name].to_csv(os.path.join(output_dir, "Train", str(train_name) + ".csv"), index=False, sep=';')
            print('Saving ' + str(os.path.join(output_dir, "Test", str(test_name) + ".csv")))
            tests[test_name].to_csv(os.path.join(output_dir, "Test", str(test_name) + ".csv"), index=False, sep=';')

        else:
            for val in std_techniques:
                if(val == 'SS'):
                    print("\nPreprocessing '" + str(train_name) + "' and '" + str(test_name) + "' with StandardScaler...")
                    train_StandardScaler, test_StandardScaler = standardization(trains[train_name], tests[test_name], "StandardScaler")
                    print('Saving ' + str(os.path.join(output_dir, "Train", str(train_name) + "_StandardScaler.csv")))
                    train_StandardScaler.to_csv(os.path.join(output_dir, "Train", str(train_name) + "_StandardScaler.csv"), index=False, sep=';')
                    print('Saving ' + str(os.path.join(output_dir, "Test", str(test_name) + "_StandardScaler.csv")))
                    test_StandardScaler.to_csv(os.path.join(output_dir, "Test", str(test_name) + "_StandardScaler.csv"), index=False, sep=';')
                elif(val == 'MMS'):
                    print("\nPreprocessing '" + str(train_name) + "' and '" + str(test_name) + "' with MinMaxScaler...")
                    train_MinMaxScaler, test_MinMaxScaler = standardization(trains[train_name], tests[test_name], "MinMaxScaler")
                    print('Saving ' + str(os.path.join(output_dir, "Train", str(train_name) + "_MinMaxScaler.csv")))
                    train_MinMaxScaler.to_csv(os.path.join(output_dir, "Train", str(train_name) + "_MinMaxScaler.csv"), index=False, sep=';')
                    print('Saving ' + str(os.path.join(output_dir, "Test", str(test_name) + "_MinMaxScaler.csv")))
                    test_MinMaxScaler.to_csv(os.path.join(output_dir, "Test", str(test_name) + "_MinMaxScaler.csv"), index=False, sep=';')
                elif(val == 'MAS'):
                    print("\nPreprocessing '" + str(train_name) + "' and '" + str(test_name) + "' with MaxAbsScaler...")
                    train_MaxAbsScaler, test_MaxAbsScaler = standardization(trains[train_name], tests[test_name], "MaxAbsScaler")
                    print('Saving ' + str(os.path.join(output_dir, "Train", str(train_name) + "_MaxAbsScaler.csv")))
                    train_MaxAbsScaler.to_csv(os.path.join(output_dir, "Train", str(train_name) + "_MaxAbsScaler.csv"), index=False, sep=';')
                    print('Saving ' + str(os.path.join(output_dir, "Test", str(test_name) + "_MaxAbsScaler.csv")))
                    test_MaxAbsScaler.to_csv(os.path.join(output_dir, "Test", str(test_name) + "_MaxAbsScaler.csv"), index=False, sep=';')
                elif(val == 'QT'):
                    print("\nPreprocessing '" + str(train_name) + "' and '" + str(test_name) + "' with QuantileTransformer...")
                    train_QuantileTransformer, test_QuantileTransformer = standardization(trains[train_name], tests[test_name], "QuantileTransformer")
                    print('Saving ' + str(os.path.join(output_dir, "Train", str(train_name) + "_QuantileTransformer.csv")))
                    train_QuantileTransformer.to_csv( os.path.join(output_dir, "Train", str(train_name) + "_QuantileTransformer.csv"), index=False, sep=';')
                    print('Saving ' + str(os.path.join(output_dir, "Test", str(test_name) + "_QuantileTransformer.csv")))
                    test_QuantileTransformer.to_csv( os.path.join(output_dir, "Test", str(test_name) + "_QuantileTransformer.csv"), index=False, sep=';')
                elif (val == 'N'):
                    if (not trains[train_name].isnull().values.any()):
                        print("\nPreprocessing '" + str(train_name) + "' and '" + str(test_name) + "' with Normalizer...")
                        train_Normalizer, test_Normalizer = standardization(trains[train_name], tests[test_name], "Normalizer")
                        print('Saving ' + str(os.path.join(output_dir, "Train", str(train_name) + "_Normalizer.csv")))
                        train_Normalizer.to_csv(os.path.join(output_dir, "Train", str(train_name) + "_Normalizer.csv"), index=False, sep=';')
                        print('Saving ' + str(os.path.join(output_dir, "Test", str(test_name) + "_Normalizer.csv")))
                        test_Normalizer.to_csv(os.path.join(output_dir, "Test", str(test_name) + "_Normalizer.csv"), index=False, sep=';')
                    else:
                        print("\nNormalizer cannot be applied to '" + str(train_name) + "' and '" + str(test_name) + "' datasets because they contain missing values.")

