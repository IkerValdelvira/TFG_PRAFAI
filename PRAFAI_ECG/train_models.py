import argparse
import os
import warnings
import pandas as pd

from Classifiers.cnn_Encoder import ENC
from Classifiers.cnn_Encoder_MLP import ENC_MLP
from Classifiers.cnn_FCN import FCN
from Classifiers.cnn_FCN_Encoder import FCN_ENC
from Classifiers.cnn_FCN_Encoder_MLP import FCN_ENC_MLP
from Classifiers.cnn_FCN_MLP import FCN_MLP
from Classifiers.rnn_LSTM import LSTM
from Classifiers.xgboost_classifier import XGBoostClassifier

pd.options.mode.chained_assignment = None  # default='warn'
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


def load_data(input_path):
    diagnostics = pd.read_excel(os.path.join(input_path, "Diagnostics.xlsx"))
    ecg_dict = {}       # Diccionario con Dataframes con los datos de los 12-lead ECGs
    ages = []           # Edad de los pacientes
    sexes = []          # Sexo de los pacientes
    labels = []         # Labels de los ECGs
    i_0 = 0
    i_1 = 0
    data_dir = os.path.join(input_path, "ECGDataDenoised")
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            # Obtener la clase
            diagnostic = diagnostics.loc[diagnostics['FileName'] == filename[:-4]]
            rhythm = diagnostic['Rhythm'].iloc[0]
            label = -1
            if (rhythm == 'SR'):
            #if (rhythm == 'SR') and (i_0 < 200):
                label = 0
                i_0 += 1
            elif (rhythm == 'AFIB'):
            #elif (rhythm == 'AFIB') and (i_1 < 200):
                label = 1
                i_1 += 1
            # Cargar ECG
            if (label == 0 or label == 1):
                labels.append(label)
                ecg = pd.read_csv(os.path.join(data_dir, filename), header=None, names=leads)
                if (ecg.isnull().values.any() == True):  # Si algun ECG tiene un NaN value, se imputa la media de su Lead
                    ecg = ecg.fillna(ecg.mean())
                ecg_dict[filename] = ecg
                # Obtener la edad y el sexo
                age = diagnostic['PatientAge'].iloc[0]
                sex = diagnostic['Gender'].iloc[0]
                if (sex == 'FEMALE'):
                    sex = 0  # Mujer --> 0
                else:
                    sex = 1  # Hombre --> 1
                ages.append(age)
                sexes.append(sex)

    return ecg_dict, ages, sexes, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser( description='Script to divide the dataset into Train/Test subsets (70%/30%) and imputation of missing values.')
    parser.add_argument("input_path", help="Path directory with input data (12-lead ECGs and diagnostics).")
    parser.add_argument("-c", "--classifier",
                        help="Classifier to use: XGBoost [XGB], FCN [FCN], FCN+MLP(age,sex) [FCN_MLP], Encoder [ENC], Encoder+MLP(age,sex) [ENC_MLP], FCN+Encoder [FCN_ENC], FCN+Encoder+MLP(age,sex) [FCN_ENC_MLP] or LSTM [LSTM]. Default option: [FCN_MLP].",
                        default='FCN_MLP')
    parser.add_argument("-xgb", "--xgb_features",
                        help="Features to use when XGBoost classifiers is selected: age and sex [AS], root mean squared of 12 leads [RMS], different model for each of 12 leads [Leads], root mean squared of 12 leads + age and sex [RMS_AS] or different model for each of 12 leads + ages and sex [Leads_AS]. This option is only used when XGBoost classifier is selected. Default: [Leads].",
                        default='Leads')
    parser.add_argument("-o", "--output_dir",
                        help="Path to directory for the created Train/Test sets. Default option: current directory.",
                        default=os.getcwd())
    args = vars(parser.parse_args())
    input_path = args['input_path']
    classifier = args['classifier']
    xgb_features = args['xgb_features']
    output_dir = args['output_dir']

    if classifier not in ['XGB', 'FCN', 'FCN_MLP', 'ENC', 'ENC_MLP', 'FCN_ENC', 'FCN_ENC_MLP', 'LSTM']:
        parser.error("'classifier' value must be [XGB], [FCN], [FCN_MLP], [ENC], [ENC_MLP], [FCN_ENC], [FCN_ENC_MLP] or [LSTM].")

    if xgb_features not in ['AS', 'RMS', 'Leads', 'RMS_AS', 'Leads_AS', '']:
        parser.error("'xgb_features' value must be [AS], [RMS], [Leads], [RMS_AS], [Leads_AS] or empty.")

    print("\nLoading 12-lead ECGs data and patients information from: " + input_path)
    ecg_dict, ages, sexes, labels = load_data(input_path)

    print("\nClassifier to use: " + classifier)
    if (classifier == 'XGB'):
        XGBoostClassifier.model(ecg_dict, ages, sexes, labels, xgb_features, output_dir)
    elif (classifier == 'FCN'):
        FCN.model(ecg_dict, labels, output_dir)
    elif (classifier == 'FCN_MLP'):
        FCN_MLP.model(ecg_dict, ages, sexes, labels, output_dir)
    elif (classifier == 'ENC'):
        ENC.model(ecg_dict, labels, output_dir)
    elif (classifier == 'ENC_MLP'):
        ENC_MLP.model(ecg_dict, ages, sexes, labels, output_dir)
    elif (classifier == 'FCN_ENC'):
        FCN_ENC.model(ecg_dict, labels, output_dir)
    elif (classifier == 'FCN_ENC_MLP'):
        FCN_ENC_MLP.model(ecg_dict, ages, sexes, labels, output_dir)
    elif (classifier == 'LSTM'):
        LSTM.model(ecg_dict, labels, output_dir)