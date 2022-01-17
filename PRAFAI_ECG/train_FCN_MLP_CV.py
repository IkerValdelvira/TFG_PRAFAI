import argparse
import os
import statistics
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
from keras.callbacks import CSVLogger
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

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


def model(ecg_dict, ages, sexes, labels, output_dir):
    ecg_array = []
    for ecg_id in ecg_dict:
        ecg_array.append(ecg_dict[ecg_id].to_numpy())
    demo = []
    for i in range(len(ages)):
        new = [ages[i], sexes[i]]
        demo.append(new)
    train_eval(np.array(ecg_array), np.array(demo), np.array(labels), output_dir)


def train_eval(features, demo, labels, output_dir):
    indices = np.array(range(len(features)))

    X_train_dev, X_test, demo_train_dev, demo_test, y_train_dev, y_test, indices_train_dev, indices_test = train_test_split(features, demo, labels, indices, test_size=0.15, random_state=42, shuffle=True, stratify=labels)

    train_accuracy_folds = []
    val_accuracy_folds = []
    dev_accuracy_folds = []
    train_auc_folds = []
    val_auc_folds = []
    dev_auc_folds = []
    train_loss_folds = []
    val_loss_folds = []
    dev_loss_folds = []

    skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    fold_var = 0
    for train_index, dev_index in skf.split(np.zeros(len(y_train_dev)), y_train_dev):
        Path(os.path.join(output_dir, "FOLD_" + str(fold_var))).mkdir(parents=True, exist_ok=True)
        print("\n\nFOLD: " + str(fold_var))

        X_train_fold = X_train_dev[train_index]
        X_dev_fold = X_train_dev[dev_index]
        demo_train_fold = demo_train_dev[train_index]
        demo_dev_fold = demo_train_dev[dev_index]
        y_train_fold = y_train_dev[train_index]
        y_dev_fold = y_train_dev[dev_index]

        X_train_fold, X_val_fold, demo_train_fold, demo_val_fold, y_train_fold, y_val_fold = train_test_split(X_train_fold, demo_train_fold, y_train_fold, test_size=0.2, random_state=42, shuffle=True, stratify=y_train_fold)

        # Standardization
        print("\n[FOLD_" + str(fold_var) + "] Standardization of datasets...")
        scaler = StandardScaler()
        scaler.fit(X_train_fold.reshape(-1, X_train_fold.shape[-1]))
        X_train_fold = scaler.transform(X_train_fold.reshape(-1, X_train_fold.shape[-1])).reshape(X_train_fold.shape)
        X_val_fold = scaler.transform(X_val_fold.reshape(-1, X_val_fold.shape[-1])).reshape(X_val_fold.shape)
        X_dev_fold = scaler.transform(X_dev_fold.reshape(-1, X_dev_fold.shape[-1])).reshape(X_dev_fold.shape)
        scaler.fit(demo_train_fold)
        demo_train_fold = scaler.transform(demo_train_fold)
        demo_val_fold = scaler.transform(demo_val_fold)
        demo_dev_fold = scaler.transform(demo_dev_fold)

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

        file = open(os.path.join(output_dir, "FOLD_" + str(fold_var), "REPORT_FCN_MLP.txt"), "w")
        file.write("FCN + MLP (age, sex) model: 12-Lead ECG classification\n\n")
        fcn_mlp_model.summary(print_fn=lambda x: file.write(x + '\n'))

        csv_logger = CSVLogger(os.path.join(output_dir, "FOLD_" + str(fold_var), 'train_log.csv'), append=True, separator=';')

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_AUC', factor=0.1, patience=1, verbose=1, mode='max',
            min_delta=0.0001, cooldown=0, min_lr=0
        )

        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_AUC', mode='max', verbose=1, patience=2)

        print("\n[FOLD_" + str(fold_var) + "] Training FCN_MLP model...")
        train = fcn_mlp_model.fit([X_train_fold, demo_train_fold], y_train_fold, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=([X_val_fold, demo_val_fold], y_val_fold), callbacks=[csv_logger,reduce_lr,early_stop])

        fcn_mlp_model.save(os.path.join(output_dir, "FOLD_" + str(fold_var), "fcn_mlp_model.h5"))
        print("[FOLD_" + str(fold_var) + "] Trained model saved in: " + os.path.join(output_dir, "FOLD_" + str(fold_var), "fcn_mlp_model.h5"))

        # DEV evaluation
        print("\n[FOLD_" + str(fold_var) + "] Evaluating model on Dev set...")
        dev_eval = fcn_mlp_model.evaluate([X_dev_fold, demo_dev_fold], y_dev_fold, verbose=1)
        file.write("\n\nDev dataset evaluation:\n")
        file.write('\tDev loss: ' + str(round(dev_eval[0],4)) + "\n")
        file.write('\tDev accuracy: ' + str(round(dev_eval[1],4)) + "\n")
        file.write('\tDev recall: ' + str(round(dev_eval[2],4)) + "\n")
        file.write('\tDev precision: ' + str(round(dev_eval[3],4)) + "\n")
        file.write('\tDev AUC: ' + str(round(dev_eval[4],4)) + "\n\n")

        dev_accuracy_folds.append(dev_eval[1])
        dev_auc_folds.append(dev_eval[4])
        dev_loss_folds.append(dev_eval[0])

        predicted_classes_dev = np.where(fcn_mlp_model.predict([X_dev_fold, demo_dev_fold]) > 0.5, 1, 0)
        file.write(classification_report(y_dev_fold, predicted_classes_dev))

        # Confusion matrix (Dev)
        conf_mat_dev = confusion_matrix(y_dev_fold, predicted_classes_dev)
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
        plt.savefig(os.path.join(output_dir, "FOLD_" + str(fold_var), "ConfusionMatrix_FCN_MLP.png"))
        print("[FOLD_" + str(fold_var) + "] Confusion matrix saved in: " + os.path.join(output_dir, "FOLD_" + str(fold_var), "ConfusionMatrix_FCN_MLP.png"))
        plt.close()

        # Accuracy history plot
        train_accuracy = train.history['accuracy']
        train_accuracy_folds.append(train_accuracy[-1])
        val_accuracy = train.history['val_accuracy']
        val_accuracy_folds.append(val_accuracy[-1])
        epochs = range(len(train_accuracy))
        plt.plot(epochs, train_accuracy, color='blue', marker='o', label='Training accuracy')
        plt.plot(epochs, val_accuracy, color='red', marker='o', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        print("[FOLD_" + str(fold_var) + "] Model accuracy history plot saved in: " + os.path.join(output_dir, "FOLD_" + str(fold_var), "Training_Validation_Accuracy.png"))
        plt.savefig(os.path.join(output_dir, "FOLD_" + str(fold_var), "Training_Validation_Accuracy.png"))
        plt.close()

        # AUC history plot
        train_auc = train.history['AUC']
        train_auc_folds.append(train_auc[-1])
        val_auc = train.history['val_AUC']
        val_auc_folds.append(val_auc[-1])
        plt.plot(epochs, train_auc, color='blue', marker='o', label='Training AUC')
        plt.plot(epochs, val_auc, color='red', marker='o', label='Validation AUC')
        plt.title('Training and validation AUC')
        plt.xlabel("Epochs")
        plt.ylabel("AUC")
        plt.legend()
        print("[FOLD_" + str(fold_var) + "] Model AUC history plot saved in: " + os.path.join(output_dir, "FOLD_" + str(fold_var), "Training_Validation_AUC.png"))
        plt.savefig(os.path.join(output_dir, "FOLD_" + str(fold_var), "Training_Validation_AUC.png"))
        plt.close()

        # Loss history plot
        train_loss = train.history['loss']
        train_loss_folds.append(train_loss[-1])
        val_loss = train.history['val_loss']
        val_loss_folds.append(val_loss[-1])
        plt.plot(epochs, train_loss, color='blue', marker='o', label='Training loss')
        plt.plot(epochs, val_loss, color='red', marker='o', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        print("[FOLD_" + str(fold_var) + "] Model loss history plot saved in: " + os.path.join(output_dir, "FOLD_" + str(fold_var), "Training_Validation_Loss.png"))
        plt.savefig(os.path.join(output_dir, "FOLD_" + str(fold_var), "Training_Validation_Loss.png"))
        plt.close()

        # ROC curve and AUC (Dev)
        # roc curve for models
        y_pred_keras = fcn_mlp_model.predict([X_dev_fold, demo_dev_fold]).ravel()
        fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_dev_fold, y_pred_keras)
        auc_keras = auc(fpr_keras, tpr_keras)
        # roc curve for tpr = fpr
        random_probs = [0 for i in range(len(y_dev_fold))]
        p_fpr, p_tpr, _ = roc_curve(y_dev_fold, random_probs, pos_label=1)
        # plot roc curves
        plt.plot(fpr_keras, tpr_keras, color='red',
                 label='FCN (AUC = %0.4f)' % auc_keras)
        plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
        # title
        plt.title("ROC curve (Dev)")
        # x label
        plt.xlabel('False Positive Rate')
        # y label
        plt.ylabel('True Positive rate')
        plt.legend(loc='best')
        plt.savefig(os.path.join(output_dir, "FOLD_" + str(fold_var), "ROCcurve(Dev)_FCN_MLP.png"))
        print("[FOLD_" + str(fold_var) + "] Dev ROC curve and AUC saved in: " + os.path.join(output_dir, "FOLD_" + str(fold_var), "ROCcurve(Dev)_FCN_MLP.png"))
        plt.close()

        print("[FOLD_" + str(fold_var) + "] Model evaluation report saved in: " + os.path.join(output_dir, "FOLD_" + str(fold_var), "REPORT_FCN_MLP.txt"))

        fold_var += 1


    # Media de resultados teniendo en cuenta todas las folds
    # Accuracy history plot
    mean_train_accuracy = statistics.mean(train_accuracy_folds)
    mean_val_accuracy = statistics.mean(val_accuracy_folds)
    mean_dev_accuracy = statistics.mean(dev_accuracy_folds)
    folds = range(len(train_accuracy_folds))
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 4)
    plt.plot(folds, train_accuracy_folds, color='blue', marker='o', label='Training accuracy')
    plt.plot(folds, val_accuracy_folds, color='red', marker='o', label='Validation accuracy')
    plt.plot(folds, dev_accuracy_folds, color='green', linestyle='dashed', marker='o', label='Dev accuracy')
    textstr = '\n'.join((
        r'Mean training accuracy: %.4f' % mean_train_accuracy,
        r'Mean validation accuracy: %.4f' % mean_val_accuracy,
        r'Mean dev accuracy: %.4f' % mean_dev_accuracy))
    props = dict(boxstyle='round', facecolor='lightcyan')
    plt.text(1.03, 0.99, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='left', bbox=props)
    plt.title('Training, validation and dev accuracy')
    plt.xlabel("Folds")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Training_Validation_Dev_Accuracy.png"))
    print("\nModel accuracy plot in each fold saved in: " + os.path.join(output_dir, "Training_Validation_Dev_Accuracy.png"))
    plt.close()

    # AUC history plot
    mean_train_auc = statistics.mean(train_auc_folds)
    mean_val_auc = statistics.mean(val_auc_folds)
    mean_dev_auc = statistics.mean(dev_auc_folds)
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 4)
    plt.plot(folds, train_auc_folds, color='blue', marker='o', label='Training AUC')
    plt.plot(folds, val_auc_folds, color='red', marker='o', label='Validation AUC')
    plt.plot(folds, dev_auc_folds, color='green', linestyle='dashed', marker='o', label='Dev AUC')
    textstr = '\n'.join((
        r'Mean training AUC: %.4f' % mean_train_auc,
        r'Mean validation AUC: %.4f' % mean_val_auc,
        r'Mean dev AUC: %.4f' % mean_dev_auc))
    props = dict(boxstyle='round', facecolor='lightcyan')
    plt.text(1.03, 0.99, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='left', bbox=props)
    plt.title('Training, validation and dev AUC')
    plt.xlabel("Folds")
    plt.ylabel("AUC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Training_Validation_Dev_AUC.png"))
    print("Model AUC plot in each fold saved in: " + os.path.join(output_dir, "Training_Validation_Dev_AUC.png"))
    plt.close()

    # Loss history plot
    mean_train_loss = statistics.mean(train_loss_folds)
    mean_val_loss = statistics.mean(val_loss_folds)
    mean_dev_loss = statistics.mean(dev_loss_folds)
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 4)
    plt.plot(folds, train_loss_folds, color='blue', marker='o', label='Training loss')
    plt.plot(folds, val_loss_folds, color='red', marker='o', label='Validation loss')
    plt.plot(folds, dev_loss_folds, color='green', linestyle='dashed', marker='o', label='Test loss')
    textstr = '\n'.join((
        r'Mean training loss: %.4f' % mean_train_loss,
        r'Mean validation loss: %.4f' % mean_val_loss,
        r'Mean dev loss: %.4f' % mean_dev_loss))
    props = dict(boxstyle='round', facecolor='lightcyan')
    plt.text(1.03, 0.99, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='left', bbox=props)
    plt.title('Training, validation and dev loss')
    plt.xlabel("Folds")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Training_Validation_Dev_Loss.png"))
    print("Model loss plot in each fold saved in: " + os.path.join(output_dir, "Training_Validation_Dev_Loss.png"))
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser( description='Script to train FCN+MLP(age,sex) AF classification model based on 12-lead ECGs and evaluate it via 10-fold Cross Validation.')
    parser.add_argument("input_path", help="Path directory with input data (12-lead ECGs and diagnostics).")
    parser.add_argument("-o", "--output_dir",
                        help="Path to the output directory for creating model and evaluation files. Default option: current directory.",
                        default=os.getcwd())
    args = vars(parser.parse_args())
    input_path = args['input_path']
    output_dir = args['output_dir']

    print("\nLoading 12-lead ECGs data and patients information from: " + input_path)
    ecg_dict, ages, sexes, labels = load_data(input_path)

    print("\nClassifier to use with 10-fold Cross Validation: FCN + MLP (age, sex)")
    model(ecg_dict, ages, sexes, labels, output_dir)
