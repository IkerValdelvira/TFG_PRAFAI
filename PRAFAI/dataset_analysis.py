import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from pathlib import Path
import argparse
import os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from matplotlib import pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
from sklearn.model_selection import train_test_split


output_dir = ""

def histogram_class(dataset, feature_name):
    labels = ['no AF recurrence', 'AF recurrence']
    classes = pd.value_counts(dataset[feature_name], sort=False)
    barlist = plt.bar(classes.index, classes.values, alpha=0.5)
    for i, j in enumerate(classes):
        plt.text(i, 10, classes[i], ha='center', fontsize=15)
        if (i == 0):
            barlist[i].set_color('g')
        else:
            barlist[i].set_color('r')
    plt.title("AF recurrence class distribution")
    plt.xticks(range(2), labels)
    plt.xlabel("AF recurrence")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_dir, "DistribucionClase_" + feature_name + ".png"))
    print("\nHistogram of class distribution '" + feature_name + "' saved in: " + os.path.join(output_dir, "DistribucionClase_" + feature_name + ".png"))
    plt.close()


def histogram_binary_categorical(dataset, feature_name):
    column = dataset[feature_name].map(str)
    sns.histplot(column[dataset['sigue_fa'] == 1], color='red', label='AF recurrence')
    sns.histplot(column[dataset['sigue_fa'] == 0], color='green', label='no AF recurrence')
    plt.xlabel(feature_name)
    plt.legend()
    plt.title('Histogram of feature: ' + feature_name)
    plt.savefig(os.path.join(output_dir, "Distribucion_" + feature_name + ".png"))
    print("\nHistogram of feature '" + feature_name + "' saved in: " + os.path.join(output_dir, "Distribucion_" + feature_name + ".png"))
    plt.close()


def histogram_numeric(dataset, feature_name):
    sns.histplot(dataset[feature_name][dataset['sigue_fa'] == 1], color='red', kde=True, label='AF recurrence')
    sns.histplot(dataset[feature_name][dataset['sigue_fa'] == 0], color='green', kde=True, label='no AF recurrence')
    plt.xlabel(feature_name)
    plt.legend()
    plt.title('Histogram of feature: ' + feature_name)
    plt.savefig(os.path.join(output_dir, "Distribucion_" + feature_name + ".png"))
    print("\nHistogram of feature '" + feature_name + "' saved in: " + os.path.join(output_dir, "Distribucion_" + feature_name + ".png"))
    plt.close()


def get_train(dataset):
    # Train/Dev/Test split
    labels = dataset['sigue_fa']
    data = dataset.iloc[:, 1:len(dataset.columns) - 1]
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, shuffle=True, stratify=labels)
    train = X_train
    train['sigue_fa'] = y_train

    return train


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Script to get distribution histograms of each feature with in regard to the class.')
    parser.add_argument("input_dataset", help="Path to PRAFAI dataset.")
    parser.add_argument("-train", "--train", dest='train', action='store_true', help="Use this flag if you want to perform the analysis only on training set.")
    parser.add_argument("-o", "--output_dir", help="Path to directory for created plots with features distribution histograms. Default option: current directory.", default=os.getcwd())
    args = vars(parser.parse_args())
    input_dataset = args['input_dataset']
    output_dir = args['output_dir']

    print('Reading PRAFAI dataset from: ' + str(input_dataset))
    dataset = pd.read_csv(input_dataset, delimiter=";")

    dataset['fecha'] = pd.to_datetime(dataset['fecha'])
    dataset['fecha'] = dataset['fecha'].astype(np.int64)

    # Only instances with class value (with missing values)
    dataset = dataset[dataset['sigue_fa'].notnull()]

    if(args['train']):
        Path(os.path.join(output_dir, "Train")).mkdir(parents=True, exist_ok=True)
        output_dir = os.path.join(output_dir, "Train")
        dataset = get_train(dataset)

    print("\nStarting feature analysis...")
    histogram_class(dataset, "sigue_fa")
    histogram_numeric(dataset, "fecha")
    histogram_binary_categorical(dataset, "tipo_fa")
    histogram_numeric(dataset, "urea")
    histogram_numeric(dataset, "creatinina")
    histogram_numeric(dataset, "albumina")
    histogram_numeric(dataset, "glucosa")
    histogram_numeric(dataset, "hba1c")
    histogram_numeric(dataset, "potasio")
    histogram_numeric(dataset, "calcio")
    histogram_numeric(dataset, "hdl")
    histogram_numeric(dataset, "no_hdl")
    histogram_numeric(dataset, "ldl")
    histogram_numeric(dataset, "colesterol")
    histogram_numeric(dataset, "ntprobnp")
    histogram_numeric(dataset, "bnp")
    histogram_numeric(dataset, "troponina_tni")
    histogram_numeric(dataset, "troponina_tnt")
    histogram_numeric(dataset, "dimero_d")
    histogram_numeric(dataset, "fibrinogeno")
    histogram_numeric(dataset, "aldosterona")
    histogram_numeric(dataset, "leucocitos")
    histogram_numeric(dataset, "pct")
    histogram_numeric(dataset, "tsh")
    histogram_numeric(dataset, "t4l")
    histogram_numeric(dataset, "sodio")
    histogram_numeric(dataset, "vsg")
    histogram_numeric(dataset, "tl3")
    histogram_binary_categorical(dataset, "ecocardiograma")
    histogram_binary_categorical(dataset, "ecocardiograma_contraste")
    histogram_binary_categorical(dataset, "electrocardiograma")
    histogram_numeric(dataset, "fevi")
    histogram_numeric(dataset, "diametro_ai")
    histogram_numeric(dataset, "area_ai")
    histogram_binary_categorical(dataset, "cardioversion")
    histogram_binary_categorical(dataset, "ablacion")
    histogram_binary_categorical(dataset, "marcapasos_dai")
    histogram_numeric(dataset, "numero_ingresos")
    histogram_binary_categorical(dataset, "etilogia_cardiologica")
    histogram_numeric(dataset, "numero_dias_desde_ingreso_hasta_evento")
    histogram_numeric(dataset, "numero_dias_ingresado")
    histogram_binary_categorical(dataset, "depresion")
    histogram_binary_categorical(dataset, "alcohol")
    histogram_binary_categorical(dataset, "drogodependencia")
    histogram_binary_categorical(dataset, "ansiedad")
    histogram_binary_categorical(dataset, "demencia")
    histogram_binary_categorical(dataset, "insuficiencia_renal")
    histogram_binary_categorical(dataset, "menopausia")
    histogram_binary_categorical(dataset, "osteoporosis")
    histogram_binary_categorical(dataset, "diabetes_tipo1")
    histogram_binary_categorical(dataset, "diabetes_tipo2")
    histogram_binary_categorical(dataset, "dislipidemia")
    histogram_binary_categorical(dataset, "hipercolesterolemia")
    histogram_binary_categorical(dataset, "fibrilacion_palpitacion")
    histogram_binary_categorical(dataset, "flutter")
    histogram_binary_categorical(dataset, "insuficiencia_cardiaca")
    histogram_binary_categorical(dataset, "fumador")
    histogram_binary_categorical(dataset, "sahos")
    histogram_binary_categorical(dataset, "hipertiroidismo")
    histogram_binary_categorical(dataset, "sindrome_metabolico")
    histogram_binary_categorical(dataset, "hipertension_arterial")
    histogram_binary_categorical(dataset, "cardiopatia_isquemica")
    histogram_binary_categorical(dataset, "miocardiopatia")
    histogram_binary_categorical(dataset, "otras_arritmias_psicogena_ritmo_bigeminal")
    histogram_binary_categorical(dataset, "bloqueos_rama")
    histogram_binary_categorical(dataset, "bloqueo_auriculoventricular")
    histogram_binary_categorical(dataset, "bradicardia")
    histogram_binary_categorical(dataset, "contracciones_prematuras_ectopia_extrasistolica")
    histogram_binary_categorical(dataset, "posoperatoria")
    histogram_binary_categorical(dataset, "sinusal_coronaria")
    histogram_binary_categorical(dataset, "valvula_mitral_reumaticas")
    histogram_binary_categorical(dataset, "otras_valvulopatias")
    histogram_binary_categorical(dataset, "valvulopatia_mitral_congenita")
    histogram_binary_categorical(dataset, "arteriopatia_periferica")
    histogram_binary_categorical(dataset, "epoc")
    histogram_binary_categorical(dataset, "genero")
    histogram_numeric(dataset, "edad")
    histogram_binary_categorical(dataset, "pensionista")
    histogram_binary_categorical(dataset, "residenciado")
    histogram_numeric(dataset, "talla")
    histogram_binary_categorical(dataset, "imc")
    histogram_numeric(dataset, "peso")
    histogram_binary_categorical(dataset, "b01a")
    histogram_binary_categorical(dataset, "n02ba01")
    histogram_binary_categorical(dataset, "a02bc")
    histogram_binary_categorical(dataset, "c03")
    histogram_binary_categorical(dataset, "g03a")
    histogram_binary_categorical(dataset, "a10")
    histogram_binary_categorical(dataset, "n06a")
    histogram_binary_categorical(dataset, "n05a")
    histogram_binary_categorical(dataset, "n05b")
    histogram_binary_categorical(dataset, "c01")
    histogram_binary_categorical(dataset, "c01b")
    histogram_binary_categorical(dataset, "c02")
    histogram_binary_categorical(dataset, "c04")
    histogram_binary_categorical(dataset, "c07")
    histogram_binary_categorical(dataset, "c08")
    histogram_binary_categorical(dataset, "c09")
    histogram_binary_categorical(dataset, "c10")
    histogram_binary_categorical(dataset, "polimedicacion")
