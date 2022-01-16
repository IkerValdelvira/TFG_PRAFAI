import json
import warnings
import joblib
from pathlib import Path
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold, RFECV
from sklearn.metrics import classification_report, confusion_matrix, plot_roc_curve, auc, roc_curve
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tabulate import tabulate

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'


def make_compatible(new_items, mv_imputation_dict, standard_scaler):

    # Imputación de missing values con la mediana en valores numéricos (incluidos edad y genero) y con 0 en valores binarios
    for column in new_items.columns:
        if(column in mv_imputation_dict.keys()):
            new_items[column] = new_items[column].fillna(mv_imputation_dict[column])
        else:
            new_items[column] = new_items[column].fillna(0)

    # Estandarización mediante StandardScaler
    dataset_columns = new_items.columns
    all_columns = ['fecha', 'tipo_fa', 'urea', 'creatinina', 'glucosa', 'potasio', 'calcio', 'hdl', 'no_hdl', 'ldl', 'colesterol', 'ntprobnp', 'tsh', 't4l', 'sodio', 'vsg', 'ecocardiograma', 'ecocardiograma_contraste', 'electrocardiograma', 'fevi', 'diametro_ai', 'area_ai', 'cardioversion', 'ablacion', 'marcapasos_dai', 'numero_ingresos', 'etilogia_cardiologica', 'numero_dias_desde_ingreso_hasta_evento', 'numero_dias_ingresado', 'depresion', 'alcohol', 'drogodependencia', 'ansiedad', 'demencia', 'insuficiencia_renal', 'menopausia', 'osteoporosis', 'diabetes_tipo1', 'diabetes_tipo2', 'dislipidemia', 'hipercolesterolemia', 'fibrilacion_palpitacion', 'flutter', 'insuficiencia_cardiaca', 'fumador', 'sahos', 'hipertiroidismo', 'sindrome_metabolico', 'hipertension_arterial', 'cardiopatia_isquemica', 'ictus', 'miocardiopatia', 'otras_arritmias_psicogena_ritmo_bigeminal', 'bloqueos_rama', 'bloqueo_auriculoventricular', 'bradicardia', 'contracciones_prematuras_ectopia_extrasistolica', 'posoperatoria', 'sinusal_coronaria', 'valvula_mitral_reumaticas', 'otras_valvulopatias', 'valvulopatia_mitral_congenita', 'arteriopatia_periferica', 'epoc', 'genero', 'edad', 'pensionista', 'residenciado', 'talla', 'imc', 'peso', 'b01a', 'n02ba01', 'a02bc', 'c03', 'g03a', 'a10', 'n06a', 'n05a', 'n05b', 'c01', 'c01b', 'c02', 'c04', 'c07', 'c08', 'c09', 'c10', 'polimedicacion']
    for column in all_columns:
        if (column not in new_items.columns):
            new_items[column] = [0] * new_items.shape[0]
    new_items_reorder = new_items[all_columns]
    new_items_reorder_array = standard_scaler.transform(new_items_reorder)
    columns_order = ['fecha', 'urea', 'creatinina', 'glucosa', 'potasio', 'calcio', 'hdl', 'no_hdl', 'ldl', 'colesterol', 'ntprobnp', 'tsh', 't4l', 'sodio', 'vsg', 'fevi', 'diametro_ai', 'area_ai', 'numero_ingresos', 'numero_dias_desde_ingreso_hasta_evento', 'numero_dias_ingresado', 'edad', 'talla', 'peso', 'tipo_fa', 'ecocardiograma', 'ecocardiograma_contraste', 'electrocardiograma', 'cardioversion', 'ablacion', 'marcapasos_dai', 'etilogia_cardiologica', 'depresion', 'alcohol', 'drogodependencia', 'ansiedad', 'demencia', 'insuficiencia_renal', 'menopausia', 'osteoporosis', 'diabetes_tipo1', 'diabetes_tipo2', 'dislipidemia', 'hipercolesterolemia', 'fibrilacion_palpitacion', 'flutter', 'insuficiencia_cardiaca', 'fumador', 'sahos', 'hipertiroidismo', 'sindrome_metabolico', 'hipertension_arterial', 'cardiopatia_isquemica', 'ictus', 'miocardiopatia', 'otras_arritmias_psicogena_ritmo_bigeminal', 'bloqueos_rama', 'bloqueo_auriculoventricular', 'bradicardia', 'contracciones_prematuras_ectopia_extrasistolica', 'posoperatoria', 'sinusal_coronaria', 'valvula_mitral_reumaticas', 'otras_valvulopatias', 'valvulopatia_mitral_congenita', 'arteriopatia_periferica', 'epoc', 'genero', 'pensionista', 'residenciado', 'imc', 'b01a', 'n02ba01', 'a02bc', 'c03', 'g03a', 'a10', 'n06a', 'n05a', 'n05b', 'c01', 'c01b', 'c02', 'c04', 'c07', 'c08', 'c09', 'c10', 'polimedicacion']
    new_items = pd.DataFrame(new_items_reorder_array, index=new_items_reorder.index, columns=columns_order)
    new_items = new_items[dataset_columns]

    return new_items


def make_predictions(new_items, model):
    file = open(os.path.join(output_dir, 'PREDICTIONS.txt'), "w")
    file.write("\nPredictions on new items:\n")

    pred = model.predict(new_items)
    pred_prob = model.predict_proba(new_items)
    i = 0
    results = []
    for index in new_items.index:
        pred_class = "no AF recurrence"
        if (pred[i] == 1.0):
            pred_class = "AF recurrence"
            prob = str(round((pred_prob[i][1] * 100), 2)) + "%"
        else:
            prob = str(round((pred_prob[i][0] * 100), 2)) + "%"

        results.append(["Item: " + str(index),
                        "Predicted: " + str(pred[i]) + "(" + pred_class + ")",
                        "Probability: " + prob])
        i += 1

    file.write(tabulate(results))
    print('Predictions on new items saved in: ' + str(os.path.join(output_dir, 'PREDICTIONS.txt')))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to make predictions on new input items using PRAFAI predictive model.')
    parser.add_argument("new_items", help="CSV file which contains new items to predict.")
    parser.add_argument("model_folder",
                        help="Path to folder where final PRAFAI predictive model and all necessary files are saved.")
    parser.add_argument("-o", "--output_dir",
                        help="Path to the output directory for creating the file with new items predictions.",
                        default=os.getcwd())
    args = vars(parser.parse_args())
    new_items_path = args['new_items']
    model_folder = args['model_folder']
    output_dir = args['output_dir']

    # Read dataset, model and necessary files
    print('\nReading CSV file which contains new items to predict from: ' + str(new_items_path))
    new_items = pd.read_csv(new_items_path, index_col=0, delimiter=";")
    print('Reading PRAFAI predictive model from: ' + str(os.path.join(model_folder, "SVM_FINAL_MODEL.joblib")))
    model = joblib.load(os.path.join(model_folder, "SVM_FINAL_MODEL.joblib"))
    print('Reading missing values imputation file from: ' + str(os.path.join(model_folder, "med_imputation_dict.json")))
    with open(os.path.join(model_folder, "med_imputation_dict.json")) as json_file:
        mv_imputation_dict = json.load(json_file)
    print('Reading data standardization model from: ' + str(os.path.join(model_folder, 'std_scaler.bin')))
    standard_scaler = joblib.load(os.path.join(model_folder, 'std_scaler.bin'))

    print('\nMaking predictions on new input items...')
    new_items_compatible = make_compatible(new_items, mv_imputation_dict, standard_scaler)
    make_predictions(new_items_compatible, model)

