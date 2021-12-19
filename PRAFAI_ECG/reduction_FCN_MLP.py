import argparse
import os
import statistics
import warnings
from pathlib import Path
import random
import pandas as pd
import numpy as np
from keras.callbacks import CSVLogger
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import ast
import matplotlib.pyplot as plt
import seaborn as sn

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
            #if (rhythm == 'SR') and (i_0 < 100):
                label = 0
                i_0 += 1
            elif (rhythm == 'AFIB'):
            #elif (rhythm == 'AFIB') and (i_1 < 100):
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


def reduce_dataset(ecgs, demo, labels, reduction):
    indices_0 = [i for i, e in enumerate(labels) if e == 0]
    indices_1 = [i for i, e in enumerate(labels) if e == 1]

    if(isinstance(reduction, int)):
        reduced_indices_0 = random.sample(indices_0, round(reduction * len(indices_0) / len(labels)))
        reduced_indices_1 = random.sample(indices_1, round(reduction * len(indices_1) / len(labels)))

    elif(isinstance(reduction, float)):
        reduced_indices_0 = random.sample(indices_0, int(len(indices_0) * (1 - reduction)))
        reduced_indices_1 = random.sample(indices_1, int(len(indices_1) * (1 - reduction)))

    reduced_indices = reduced_indices_0 + reduced_indices_1

    reduced_ecgs = [ecgs[i] for i in reduced_indices]
    reduced_demo = [demo[i] for i in reduced_indices]
    reduced_labels = [labels[i] for i in reduced_indices]

    return reduced_ecgs, reduced_demo, reduced_labels

def model(ecg_dict, ages, sexes, labels, reduction, output_dir):
    ecg_array = []
    for ecg_id in ecg_dict:
        ecg_array.append(ecg_dict[ecg_id].to_numpy())
    demo = []
    for i in range(len(ages)):
        new = [ages[i], sexes[i]]
        demo.append(new)

    # Train/Dev/Test split
    X_train_dev, X_test, demo_train_dev, demo_test, y_train_dev, y_test = train_test_split(np.array(ecg_array), np.array(demo), np.array(labels), test_size=0.15, random_state=42, shuffle=True, stratify=labels)
    X_train, X_dev, demo_train, demo_dev, y_train, y_dev = train_test_split(X_train_dev, demo_train_dev, y_train_dev, test_size=(len(y_test) / len(y_train_dev)), random_state=42, shuffle=True, stratify=y_train_dev)

    train_accuracies = []
    val_accuracies = []
    dev_accuracies = []
    test_accuracies = []
    train_aucs = []
    val_aucs = []
    dev_aucs = []
    test_aucs = []
    train_losses = []
    val_losses = []
    dev_losses = []

    n_items = []
    # Para reducir por PORCENTAJE
    if (isinstance(reduction, float)):
        reduction_num = 0
        while((len(y_train) * (1-reduction)) >= 50):
            Path(os.path.join(output_dir, "REDUCTION_" + str(reduction_num))).mkdir(parents=True, exist_ok=True)
            if (reduction_num > 1):
                print("\n[REDUCTION_" + str(reduction_num) + "] Reduction by fraction of " + str(reduction) + " will be applied")
                X_train, demo_train, y_train = reduce_dataset(X_train, demo_train, y_train, reduction)
            n_items.append(len(y_train))
            print("[REDUCTION_" + str(reduction_num) + "] The training dataset consists of " + str(len(y_train)) + " items")
            results = train_eval(np.array(X_train), np.array(demo_train), np.array(y_train), np.array(X_dev), np.array(demo_dev), np.array(y_dev), np.array(X_test), np.array(demo_test), np.array(y_test), reduction_num, os.path.join(output_dir, "REDUCTION_" + str(reduction_num)))
            train_accuracies.append(results[0])
            val_accuracies.append(results[1])
            dev_accuracies.append(results[2])
            test_accuracies.append(results[3])
            train_aucs.append(results[4])
            val_aucs.append(results[5])
            dev_aucs.append(results[6])
            test_aucs.append(results[7])
            train_losses.append(results[8])
            val_losses.append(results[9])
            dev_losses.append(results[10])
            reduction_num += 1

    # Para reducir por VALOR
    elif (isinstance(reduction, list)):
        reduction_num = 0
        Path(os.path.join(output_dir, "REDUCTION_" + str(reduction_num))).mkdir(parents=True, exist_ok=True)
        n_items.append(len(y_train))
        print("\n[REDUCTION_" + str(reduction_num) + "] The training dataset consists of " + str(len(y_train)) + " items")
        results = train_eval(np.array(X_train), np.array(demo_train), np.array(y_train), np.array(X_dev), np.array(demo_dev), np.array(y_dev), np.array(X_test), np.array(demo_test), np.array(y_test), reduction_num, os.path.join(output_dir, "REDUCTION_" + str(reduction_num)))
        train_accuracies.append(results[0])
        val_accuracies.append(results[1])
        dev_accuracies.append(results[2])
        test_accuracies.append(results[3])
        train_aucs.append(results[4])
        val_aucs.append(results[5])
        dev_aucs.append(results[6])
        test_aucs.append(results[7])
        train_losses.append(results[8])
        val_losses.append(results[9])
        dev_losses.append(results[10])
        reduction_num += 1
        for val in reduction:
            if (isinstance(val, int) and val <= len(y_train)):
                Path(os.path.join(output_dir, "REDUCTION_" + str(reduction_num))).mkdir(parents=True, exist_ok=True)
                n_items.append(val)
                X_train_aux, demo_train_aux, y_train_aux = reduce_dataset(X_train, demo_train, y_train, val)
                print("\n[REDUCTION_" + str(reduction_num) + "] The training dataset consists of " + str(len(y_train_aux)) + " items")
                results = train_eval(np.array(X_train_aux), np.array(demo_train_aux), np.array(y_train_aux), np.array(X_dev), np.array(demo_dev), np.array(y_dev), np.array(X_test), np.array(demo_test), np.array(y_test), reduction_num, os.path.join(output_dir, "REDUCTION_" + str(reduction_num)))
                train_accuracies.append(results[0])
                val_accuracies.append(results[1])
                dev_accuracies.append(results[2])
                test_accuracies.append(results[3])
                train_aucs.append(results[4])
                val_aucs.append(results[5])
                dev_aucs.append(results[6])
                test_aucs.append(results[7])
                train_losses.append(results[8])
                val_losses.append(results[9])
                dev_losses.append(results[10])
                reduction_num += 1
            elif (isinstance(val, int) and val > len(y_train)):
                print("\n[REDUCTION_" + str(reduction_num) + "] " + str(val) + " is higher than total size of the training original dataset, no reduction is applied.")
            else:
                print("\n[REDUCTION_" + str(reduction_num) + "] " + str(val) + " is not an integer, no reduction is applied.")

    # Resultados teniendo en cuenta todos los conjuntos de datos reducidos
    # Accuracy history plot
    plt.plot(n_items, train_accuracies, color='blue', marker='o', label='Training accuracy')
    plt.plot(n_items, val_accuracies, color='red', marker='o', label='Validation accuracy')
    plt.plot(n_items, dev_accuracies, color='green', linestyle='dashed', marker='o', label='Dev accuracy')
    plt.title('Training, validation and dev accuracy')
    plt.xlabel("Nº of training items")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "Reduction_Accuracy.png"))
    print("\nAccuracy history throughout reductions saved in: " + os.path.join(output_dir, "Reduction_Accuracy.png"))
    plt.close()

    # AUC history plot
    plt.plot(n_items, train_aucs, color='blue', marker='o', label='Training AUC')
    plt.plot(n_items, val_aucs, color='red', marker='o', label='Validation AUC')
    plt.plot(n_items, dev_aucs, color='green', linestyle='dashed', marker='o', label='Dev AUC')
    plt.title('Training, validation and dev AUC')
    plt.xlabel("Nº of training items")
    plt.ylabel("AUC")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "Reduction_AUC.png"))
    print("AUC history throughout reductions saved in: " + os.path.join(output_dir, "Reduction_Accuracy.png"))
    plt.close()

    # Loss history plot
    plt.plot(n_items, train_losses, color='blue', marker='o', label='Training loss')
    plt.plot(n_items, val_losses, color='red', marker='o', label='Validation loss')
    plt.plot(n_items, dev_losses, color='green', linestyle='dashed', marker='o', label='Dev loss')
    plt.title('Training, validation and dev loss')
    plt.xlabel("Nº of training items")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "Reduction_Loss.png"))
    print("Loss history throughout reductions saved in: " + os.path.join(output_dir, "Reduction_Accuracy.png"))
    plt.close()

    # Test reduction values plot
    # Loss history plot
    plt.plot(n_items, test_accuracies, color='blue', marker='o', label='Test accuracy')
    plt.plot(n_items, test_aucs, color='red', marker='o', label='Test AUC')
    plt.title('Test set accuracy and AUC')
    plt.xlabel("Nº of training items")
    plt.ylabel("Metric")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "Test_metrics.png"))
    print("\nTest metrics throughout reductions saved in: " + os.path.join(output_dir, "Test_metrics.png"))
    plt.close()


def train_eval(X_train, demo_train, y_train, X_dev, demo_dev, y_dev, X_test, demo_test, y_test, reduction_num, output_dir):

    X_train, X_val, demo_train, demo_val, y_train, y_val = train_test_split(X_train, demo_train, y_train, test_size=0.2, random_state=42, shuffle=True, stratify=y_train)

    # Standardization
    print("\n[REDUCTION_" + str(reduction_num) + "] Standardization of datasets...")
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
    X_train = scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_dev = scaler.transform(X_dev.reshape(-1, X_dev.shape[-1])).reshape(X_dev.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    scaler.fit(demo_train)
    demo_train = scaler.transform(demo_train)
    demo_val = scaler.transform(demo_val)
    demo_dev = scaler.transform(demo_dev)
    demo_test = scaler.transform(demo_test)

    batch_size = 30
    epochs = 20

    inputA = keras.layers.Input(shape=(5000, 12))
    inputB = keras.layers.Input(shape=(2,))

    conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, input_shape=(5000, 12), padding='same')(inputA)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation(activation='relu')(conv1)

    conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)

    conv3 = keras.layers.Conv1D(128, kernel_size=3, padding='same')(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)

    gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)
    model1 = keras.Model(inputs=inputA, outputs=gap_layer)

    mod3 = keras.layers.Dense(50, activation="relu")(inputB)
    mod3 = keras.layers.Dense(2, activation="sigmoid")(mod3)
    model3 = keras.Model(inputs=inputB, outputs=mod3)

    combined = keras.layers.concatenate([model1.output, model3.output])
    final_layer = keras.layers.Dense(1, activation="sigmoid")(combined)
    fcn_mlp_model = keras.models.Model(inputs=[inputA, inputB], outputs=final_layer)

    fcn_mlp_model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(),
                          metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                                   tf.keras.metrics.Recall(name='recall'),
                                   tf.keras.metrics.Precision(name='precision'),
                                   tf.keras.metrics.AUC(name="AUC")])

    file = open(os.path.join(output_dir, "REPORT_FCN_MLP.txt"), "w")
    file.write("FCN + MLP (age, sex) model: 12-Lead ECG classification\n\n")
    fcn_mlp_model.summary(print_fn=lambda x: file.write(x + '\n'))

    csv_logger = CSVLogger(os.path.join(output_dir, 'train_log.csv'), append=True, separator=';')

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_AUC', factor=0.1, patience=1, verbose=1, mode='max',
        min_delta=0.0001, cooldown=0, min_lr=0
    )

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_AUC', mode='max', verbose=1, patience=2)

    print("\n[REDUCTION_" + str(reduction_num) + "] Training FCN_MLP model...")
    train = fcn_mlp_model.fit([X_train, demo_train], y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=([X_val, demo_val], y_val), callbacks=[csv_logger,reduce_lr,early_stop])

    fcn_mlp_model.save(os.path.join(output_dir, "fcn_mlp_model.h5"))
    print("[REDUCTION_" + str(reduction_num) + "] Trained model saved in: " + os.path.join(output_dir, "fcn_mlp_model.h5"))

    # DEV evaluation
    print("\n[REDUCTION_" + str(reduction_num) + "] Evaluating model on Dev set...")
    dev_eval = fcn_mlp_model.evaluate([X_dev, demo_dev], y_dev, verbose=1)
    file.write("\n\nDev dataset evaluation:\n")
    file.write('\tDev loss: ' + str(round(dev_eval[0],4)) + "\n")
    file.write('\tDev accuracy: ' + str(round(dev_eval[1],4)) + "\n")
    file.write('\tDev recall: ' + str(round(dev_eval[2],4)) + "\n")
    file.write('\tDev precision: ' + str(round(dev_eval[3],4)) + "\n")
    file.write('\tDev AUC: ' + str(round(dev_eval[4],4)) + "\n\n")

    dev_accuracy = dev_eval[1]
    dev_auc = dev_eval[4]
    dev_loss = dev_eval[0]

    predicted_classes_dev = np.where(fcn_mlp_model.predict([X_dev, demo_dev]) > 0.5, 1, 0)
    file.write(classification_report(y_dev, predicted_classes_dev))

    # Confusion matrix (Dev)
    conf_mat_dev = confusion_matrix(y_dev, predicted_classes_dev)
    tn, fp, fn, tp = conf_mat_dev.ravel()
    specificity_test = tn / (tn + fp)
    sensitivity_test = tp / (tp + fn)
    file.write("\nspecificity\t\t" + str(round(specificity_test, 2)))
    file.write("\nsensitivity\t\t" + str(round(sensitivity_test, 2)) + "\n")
    df_cm_dev = pd.DataFrame(conf_mat_dev, index=['Sinus Rhythm', 'Atrial Fibrillation'], columns=['Sinus Rhythm', 'Atrial Fibrillation'])
    sn.heatmap(df_cm_dev, annot=True, fmt='g', cmap=plt.cm.Blues)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion matrix (Dev)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ConfusionMatrix_FCN_MLP.png"))
    print("[REDUCTION_" + str(reduction_num) + "] Confusion matrix saved in: " + os.path.join(output_dir, "ConfusionMatrix_FCN_MLP.png"))
    plt.close()

    # Accuracy history plot
    train_accuracy_history = train.history['accuracy']
    train_accuracy = train_accuracy_history[-1]
    val_accuracy_history = train.history['val_accuracy']
    val_accuracy = val_accuracy_history[-1]
    epochs = range(len(train_accuracy_history))
    plt.plot(epochs, train_accuracy_history, color='blue', marker='o', label='Training accuracy')
    plt.plot(epochs, val_accuracy_history, color='red', marker='o', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    print("[REDUCTION_" + str(reduction_num) + "] Model accuracy history plot saved in: " + os.path.join(output_dir, "Training_Validation_Accuracy.png"))
    plt.savefig(os.path.join(output_dir, "Training_Validation_Accuracy.png"))
    plt.close()

    # AUC history plot
    train_auc_history = train.history['AUC']
    train_auc = train_auc_history[-1]
    val_auc_history = train.history['val_AUC']
    val_auc = val_auc_history[-1]
    plt.plot(epochs, train_auc_history, color='blue', marker='o', label='Training AUC')
    plt.plot(epochs, val_auc_history, color='red', marker='o', label='Validation AUC')
    plt.title('Training and validation AUC')
    plt.xlabel("Epochs")
    plt.ylabel("AUC")
    plt.legend()
    print("[REDUCTION_" + str(reduction_num) + "] Model AUC history plot saved in: " + os.path.join(output_dir, "Training_Validation_AUC.png"))
    plt.savefig(os.path.join(output_dir, "Training_Validation_AUC.png"))
    plt.close()

    # Loss history plot
    train_loss_history = train.history['loss']
    train_loss = train_loss_history[-1]
    val_loss_history = train.history['val_loss']
    val_loss = val_loss_history[-1]
    plt.plot(epochs, train_loss_history, color='blue', marker='o', label='Training loss')
    plt.plot(epochs, val_loss_history, color='red', marker='o', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    print("[REDUCTION_" + str(reduction_num) + "] Model loss history plot saved in: " + os.path.join(output_dir, "Training_Validation_Loss.png"))
    plt.savefig(os.path.join(output_dir, "Training_Validation_Loss.png"))
    plt.close()

    # ROC curve and AUC (Dev)
    # roc curve for models
    y_pred_keras = fcn_mlp_model.predict([X_dev, demo_dev]).ravel()
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_dev, y_pred_keras)
    auc_keras = auc(fpr_keras, tpr_keras)
    # roc curve for tpr = fpr
    random_probs = [0 for i in range(len(y_dev))]
    p_fpr, p_tpr, _ = roc_curve(y_dev, random_probs, pos_label=1)
    # plot roc curves
    plt.plot(fpr_keras, tpr_keras, color='red',
             label='FCN+MLP (AUC = %0.4f)' % auc_keras)
    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
    # title
    plt.title("ROC curve (Dev)")
    # x label
    plt.xlabel('False Positive Rate')
    # y label
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.savefig(os.path.join(output_dir, "ROCcurve(Dev)_FCN_MLP.png"))
    print("[REDUCTION_" + str(reduction_num) + "] Dev ROC curve and AUC saved in: " + os.path.join(output_dir, "ROCcurve(Dev)_FCN_MLP.png"))
    plt.close()

    print("[REDUCTION_" + str(reduction_num) + "] Model evaluation report saved in: " + os.path.join(output_dir, "REPORT_FCN_MLP.txt"))

    # Test evaluation
    test_eval = fcn_mlp_model.evaluate([X_test, demo_test], y_test, verbose=0)
    test_accuracy = test_eval[1]
    test_auc = test_eval[4]

    return [train_accuracy, val_accuracy, dev_accuracy, test_accuracy, train_auc, val_auc, dev_auc, test_auc, train_loss, val_loss, dev_loss]


if __name__ == '__main__':
    parser = argparse.ArgumentParser( description='Script to divide the dataset into Train/Test subsets (70%/30%) and imputation of missing values.')
    parser.add_argument("input_path", help="Path directory with input data (12-lead ECGs and diagnostics).")
    parser.add_argument("-r", "--reduction",
                        help="Reduction to apply to datasets. If float (fraction), the data set will be reduced by that fraction for each iteration until it does not exceed 50 items. If list of integers, the number of items indicated will be selected for each iteration. Default: 0.5",
                        default=0.5)
    parser.add_argument("-o", "--output_dir",
                        help="Path to directory for the created Train/Test sets. Default option: current directory.",
                        default=os.getcwd())
    args = vars(parser.parse_args())
    input_path = args['input_path']
    reduction = args['reduction']
    output_dir = args['output_dir']

    try:
        reduction = float(reduction)
        if(reduction >= 1):
            parser.error("'reduction' value must be a float (fraction) or a list of integers.")
    except ValueError:
        try:
            reduction = ast.literal_eval(reduction)
        except ValueError:
            parser.error("'reduction' value must be a float (fraction) or a list of integers.")

    print("\nLoading 12-lead ECGs data and patients information from: " + input_path)
    ecg_dict, ages, sexes, labels = load_data(input_path)

    print("\nItem reduction --> Classifier to use: FCN + MLP (age, sex)")
    model(ecg_dict, ages, sexes, labels, reduction, output_dir)
