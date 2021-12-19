import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, QuantileTransformer, Normalizer

def preprocess(train, test, type):
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Script to standardize/normalize numerical features of the training set (Train) and apply the same rescaling to the testing set (Test). Different techniques are available for standardization/normalization: StandardScaler, MinMaxScaler, MaxAbsScaler, QuantileTransformer or Normalizer.')
    parser.add_argument("input_train", help="Path to training set (Train).")
    parser.add_argument("input_test", help="Path to testing set (Test).")
    parser.add_argument("-std", "--std_techniques", nargs='+', help="Standardization/Normalization techniques to use: StandardScaler [SS], MinMaxScaler [MMS], MaxAbsScaler [MAS], QuantileTransformer [QT] and/or Normalizer [N]. Multiple options are accepted. The [all] option will apply all techniques. Default option: StandardScaler [SS].", default=['SS'])
    parser.add_argument("-o", "--output_dir", help="Path to directory for the preprocessed Train/Test sets. Default option: current directory.", default=os.getcwd())
    args = vars(parser.parse_args())
    input_train = args['input_train']
    input_test = args['input_test']
    std_techniques = args['std_techniques']
    output_dir = args['output_dir']
    for val in std_techniques:
        if val not in ['all', 'SS', 'MMS', 'MAS', 'QT', 'N']:
            parser.error(
                "'--std_technique' values must be [SS], [MMS], [MAS], [QT], [N] and/or [all]. Multiple options are accepted.")


    print('Reading Train set from: ' + str(input_train))
    train = pd.read_csv(input_train, delimiter=";")
    print('Reading Test set from: ' + str(input_test))
    test = pd.read_csv(input_test, delimiter=";")

    train_name = Path(input_train).stem
    test_name = Path(input_test).stem

    if 'all' in std_techniques:
        print('\nPreprocessing with StandardScaler...')
        train_StandardScaler, test_StandardScaler = preprocess(train, test, "StandardScaler")
        print('Saving ' + str(os.path.join(output_dir,str(train_name)+"_StandardScaler.csv")))
        train_StandardScaler.to_csv(os.path.join(output_dir,str(train_name)+"_StandardScaler.csv"), index=False, sep=';')
        print('Saving ' + str(os.path.join(output_dir,str(test_name)+"_StandardScaler.csv")))
        test_StandardScaler.to_csv(os.path.join(output_dir,str(test_name)+"_StandardScaler.csv"), index=False, sep=';')

        print('\nPreprocessing with MinMaxScaler...')
        train_MinMaxScaler, test_MinMaxScaler = preprocess(train, test, "MinMaxScaler")
        print('Saving ' + str(os.path.join(output_dir,str(train_name)+"_MinMaxScaler.csv")))
        train_MinMaxScaler.to_csv(os.path.join(output_dir,str(train_name)+"_MinMaxScaler.csv"), index=False, sep=';')
        print('Saving ' + str(os.path.join(output_dir,str(test_name)+"_MinMaxScaler.csv")))
        test_MinMaxScaler.to_csv(os.path.join(output_dir,str(test_name)+"_MinMaxScaler.csv"), index=False, sep=';')

        print('\nPreprocessing with MaxAbsScaler...')
        train_MaxAbsScaler, test_MaxAbsScaler = preprocess(train, test, "MaxAbsScaler")
        print('Saving ' + str(os.path.join(output_dir,str(train_name)+"_MaxAbsScaler.csv")))
        train_MaxAbsScaler.to_csv(os.path.join(output_dir,str(train_name)+"_MaxAbsScaler.csv"), index=False, sep=';')
        print('Saving ' + str(os.path.join(output_dir,str(test_name)+"_MaxAbsScaler.csv")))
        test_MaxAbsScaler.to_csv(os.path.join(output_dir,str(test_name)+"_MaxAbsScaler.csv"), index=False, sep=';')

        print('\nPreprocessing with QuantileTransformer...')
        train_QuantileTransformer, test_QuantileTransformer = preprocess(train, test, "QuantileTransformer")
        print('Saving ' + str(os.path.join(output_dir,str(train_name)+"_QuantileTransformer.csv")))
        train_QuantileTransformer.to_csv(os.path.join(output_dir,str(train_name)+"_QuantileTransformer.csv"), index=False, sep=';')
        print('Saving ' + str(os.path.join(output_dir,str(test_name)+"_QuantileTransformer.csv")))
        test_QuantileTransformer.to_csv(os.path.join(output_dir,str(test_name)+"_QuantileTransformer.csv"), index=False, sep=';')

        if(not train.isnull().values.any()):
            print('\nPreprocessing with Normalizer...')
            train_Normalizer, test_Normalizer = preprocess(train, test, "Normalizer")
            print('Saving ' + str(os.path.join(output_dir,str(train_name)+"_Normalizer.csv")))
            train_Normalizer.to_csv(os.path.join(output_dir,str(train_name)+"_Normalizer.csv"), index=False, sep=';')
            print('Saving ' + str(os.path.join(output_dir,str(test_name)+"_Normalizer.csv")))
            test_Normalizer.to_csv(os.path.join(output_dir,str(test_name)+"_Normalizer.csv"), index=False, sep=';')
        else:
            print('\nNormalizer cannot be applied to this dataset because it contains missing values.')

    else:
        for val in std_techniques:
            if(val == 'SS'):
                print('\nPreprocessing with StandardScaler...')
                train_StandardScaler, test_StandardScaler = preprocess(train, test, "StandardScaler")
                print('Saving ' + str(os.path.join(output_dir, str(train_name) + "_StandardScaler.csv")))
                train_StandardScaler.to_csv(os.path.join(output_dir, str(train_name) + "_StandardScaler.csv"), index=False, sep=';')
                print('Saving ' + str(os.path.join(output_dir, str(test_name) + "_StandardScaler.csv")))
                test_StandardScaler.to_csv(os.path.join(output_dir, str(test_name) + "_StandardScaler.csv"), index=False, sep=';')
            elif(val == 'MMS'):
                print('\nPreprocessing with MinMaxScaler...')
                train_MinMaxScaler, test_MinMaxScaler = preprocess(train, test, "MinMaxScaler")
                print('Saving ' + str(os.path.join(output_dir, str(train_name) + "_MinMaxScaler.csv")))
                train_MinMaxScaler.to_csv(os.path.join(output_dir, str(train_name) + "_MinMaxScaler.csv"), index=False, sep=';')
                print('Saving ' + str(os.path.join(output_dir, str(test_name) + "_MinMaxScaler.csv")))
                test_MinMaxScaler.to_csv(os.path.join(output_dir, str(test_name) + "_MinMaxScaler.csv"), index=False, sep=';')
            elif(val == 'MAS'):
                print('\nPreprocessing with MaxAbsScaler...')
                train_MaxAbsScaler, test_MaxAbsScaler = preprocess(train, test, "MaxAbsScaler")
                print('Saving ' + str(os.path.join(output_dir, str(train_name) + "_MaxAbsScaler.csv")))
                train_MaxAbsScaler.to_csv(os.path.join(output_dir, str(train_name) + "_MaxAbsScaler.csv"), index=False, sep=';')
                print('Saving ' + str(os.path.join(output_dir, str(test_name) + "_MaxAbsScaler.csv")))
                test_MaxAbsScaler.to_csv(os.path.join(output_dir, str(test_name) + "_MaxAbsScaler.csv"), index=False, sep=';')
            elif(val == 'QT'):
                print('\nPreprocessing with QuantileTransformer...')
                train_QuantileTransformer, test_QuantileTransformer = preprocess(train, test, "QuantileTransformer")
                print('Saving ' + str(os.path.join(output_dir, str(train_name) + "_QuantileTransformer.csv")))
                train_QuantileTransformer.to_csv(os.path.join(output_dir, str(train_name) + "_QuantileTransformer.csv"), index=False, sep=';')
                print('Saving ' + str(os.path.join(output_dir, str(test_name) + "_QuantileTransformer.csv")))
                test_QuantileTransformer.to_csv(os.path.join(output_dir, str(test_name) + "_QuantileTransformer.csv"), index=False, sep=';')
            elif (val == 'N'):
                if (not train.isnull().values.any()):
                    print('\nPreprocessing with Normalizer...')
                    train_Normalizer, test_Normalizer = preprocess(train, test, "Normalizer")
                    print('Saving ' + str(os.path.join(output_dir, str(train_name) + "_Normalizer.csv")))
                    train_Normalizer.to_csv(os.path.join(output_dir, str(train_name) + "_Normalizer.csv"), index=False, sep=';')
                    print('Saving ' + str(os.path.join(output_dir, str(test_name) + "_Normalizer.csv")))
                    test_Normalizer.to_csv(os.path.join(output_dir, str(test_name) + "_Normalizer.csv"), index=False, sep=';')
                else:
                    print('\nNormalizer cannot be applied to this dataset because it contains missing values.')

