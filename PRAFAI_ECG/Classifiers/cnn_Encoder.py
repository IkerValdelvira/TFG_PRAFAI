import os

import joblib
import numpy as np
import pandas as pd
from keras.callbacks import CSVLogger
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sn

class ENC(object):

    @staticmethod
    def model(ecg_dict, labels, output_dir):
        ids = list(ecg_dict.keys())
        ecg_array = []
        for ecg_id in ecg_dict:
            ecg_array.append(ecg_dict[ecg_id].to_numpy())
        ENC.train_eval(object, ids, np.array(ecg_array), np.array(labels), output_dir)


    def train_eval(self, ids, features, labels, output_dir):
        indices = np.array(range(len(features)))

        X_train_dev, X_test, y_train_dev, y_test, indices_train_dev, indices_test = train_test_split(features, labels, indices, test_size=0.15, random_state=42, shuffle=True, stratify=labels)
        X_train, X_dev, y_train, y_dev, indices_train, indices_dev = train_test_split(X_train_dev, y_train_dev, indices_train_dev, test_size=(len(y_test) / len(y_train_dev)), random_state=42, shuffle=True, stratify=y_train_dev)
        X_train, X_val, y_train, y_val, indices_train, indices_val = train_test_split(X_train, y_train, indices_train, test_size=0.2, random_state=42, shuffle=True, stratify=y_train)

        ids_test = [ids[x] for x in indices_test]

        # Standardization
        print("\nStandardization of datasets...")
        scaler = StandardScaler()
        scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
        X_train = scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        X_dev = scaler.transform(X_dev.reshape(-1, X_dev.shape[-1])).reshape(X_dev.shape)
        X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        joblib.dump(scaler, os.path.join(output_dir, 'std_scaler_Encoder.bin'), compress=True)
        print("Standard scaler saved in: " + str(os.path.join(output_dir, 'std_scaler_Encoder.bin')))

        batch_size = 30
        epochs = 20

        input_layer = keras.layers.Input(shape=(5000, 12))

        # conv block -1
        conv1 = keras.layers.Conv1D(filters=128, kernel_size=5, strides=1, padding='same')(input_layer)
        conv1 = tfa.layers.InstanceNormalization()(conv1)
        conv1 = keras.layers.PReLU(shared_axes=[1])(conv1)
        conv1 = keras.layers.Dropout(rate=0.2)(conv1)
        conv1 = keras.layers.MaxPooling1D(pool_size=2)(conv1)
        # conv block -2
        conv2 = keras.layers.Conv1D(filters=256, kernel_size=11, strides=1, padding='same')(conv1)
        conv2 = tfa.layers.InstanceNormalization()(conv2)
        conv2 = keras.layers.PReLU(shared_axes=[1])(conv2)
        conv2 = keras.layers.Dropout(rate=0.2)(conv2)
        conv2 = keras.layers.MaxPooling1D(pool_size=2)(conv2)
        # conv block -3
        conv3 = keras.layers.Conv1D(filters=512, kernel_size=21, strides=1, padding='same')(conv2)
        conv3 = tfa.layers.InstanceNormalization()(conv3)
        conv3 = keras.layers.PReLU(shared_axes=[1])(conv3)
        conv3 = keras.layers.Dropout(rate=0.2)(conv3)
        # split for attention
        attention_data = keras.layers.Lambda(lambda x: x[:, :, :256])(conv3)
        attention_softmax = keras.layers.Lambda(lambda x: x[:, :, 256:])(conv3)
        # attention mechanism
        attention_softmax = keras.layers.Softmax()(attention_softmax)
        multiply_layer = keras.layers.Multiply()([attention_softmax, attention_data])
        # last layer
        dense_layer = keras.layers.Dense(units=256, activation='sigmoid')(multiply_layer)
        dense_layer = tfa.layers.InstanceNormalization()(dense_layer)
        # output layer
        flatten_layer = keras.layers.Flatten()(dense_layer)
        output_layer = keras.layers.Dense(units=1, activation='sigmoid')(flatten_layer)

        encoder_model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        encoder_model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                          metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                                   tf.keras.metrics.Recall(name='recall'),
                                   tf.keras.metrics.Precision(name='precision'),
                                   tf.keras.metrics.AUC(name="AUC")])

        file = open(os.path.join(output_dir, "REPORT_Encoder" + ".txt"), "w")
        file.write("Encoder model: 12-Lead ECG classification\n\n")
        encoder_model.summary(print_fn=lambda x: file.write(x + '\n'))

        csv_logger = CSVLogger(os.path.join(output_dir, 'train_log_Encoder.csv'), append=True, separator=';')

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_AUC', factor=0.1, patience=1, verbose=1, mode='max',
            min_delta=0.0001, cooldown=0, min_lr=0
        )

        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_AUC', mode='max', verbose=1, patience=2)

        print("\nTraining Encoder model...")
        train = encoder_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_val, y_val), callbacks=[csv_logger,reduce_lr,early_stop])

        encoder_model.save(os.path.join(output_dir, "encoder_model.h5"))
        print('Trained model saved in: ' + os.path.join(output_dir, "encoder_model.h5"))

        print('\nEvaluating model...')

        # DEV evaluation
        print('Evaluating on Dev set...')
        dev_eval = encoder_model.evaluate(X_dev, y_dev, verbose=1)
        file.write("\n\nDev dataset evaluation:\n")
        file.write('\tDev loss: ' + str(round(dev_eval[0],4)) + "\n")
        file.write('\tDev accuracy: ' + str(round(dev_eval[1],4)) + "\n")
        file.write('\tDev recall: ' + str(round(dev_eval[2],4)) + "\n")
        file.write('\tDev precision: ' + str(round(dev_eval[3],4)) + "\n")
        file.write('\tDev AUC: ' + str(round(dev_eval[4],4)) + "\n\n")

        # Classification report (Dev)
        predicted_classes_dev = np.where(encoder_model.predict(X_dev) > 0.5, 1, 0)
        file.write("Classification Report (Dev):\n" + classification_report(y_dev, predicted_classes_dev))

        # Confusion matrix (Dev)
        conf_mat_train = confusion_matrix(y_dev, predicted_classes_dev)
        tn, fp, fn, tp = conf_mat_train.ravel()
        specificity_train = tn / (tn + fp)
        sensitivity_train = tp / (tp + fn)
        file.write("\nspecificity\t\t" + str(round(specificity_train, 2)))
        file.write("\nsensitivity\t\t" + str(round(sensitivity_train, 2)) + "\n\n")
        df_cm_dev = pd.DataFrame(conf_mat_train, index=['Sinus Rhythm', 'Atrial Fibrillation'],
                                   columns=['Sinus Rhythm', 'Atrial Fibrillation'])

        # TEST evaluation
        print('Evaluating on Test set...')
        test_eval = encoder_model.evaluate(X_test, y_test, verbose=1)
        file.write("\n\nTest dataset evaluation:\n")
        file.write('\tTest loss: ' + str(round(test_eval[0],4)) + "\n")
        file.write('\tTest accuracy: ' + str(round(test_eval[1],4)) + "\n")
        file.write('\tTest recall: ' + str(round(test_eval[2],4)) + "\n")
        file.write('\tTest precision: ' + str(round(test_eval[3],4)) + "\n")
        file.write('\tTest AUC: ' + str(round(test_eval[4],4)) + "\n\n")

        # Classification report (Test)
        predicted_classes_test = np.where(encoder_model.predict(X_test) > 0.5, 1, 0)
        file.write("Classification Report (Dev):\n" + classification_report(y_test, predicted_classes_test))

        # Confusion matrix (Test)
        conf_mat_test = confusion_matrix(y_test, predicted_classes_test)
        tn, fp, fn, tp = conf_mat_test.ravel()
        specificity_test = tn / (tn + fp)
        sensitivity_test = tp / (tp + fn)
        file.write("\nspecificity\t\t" + str(round(specificity_test, 2)))
        file.write("\nsensitivity\t\t" + str(round(sensitivity_test, 2)) + "\n")
        df_cm_test = pd.DataFrame(conf_mat_test, index=['Sinus Rhythm', 'Atrial Fibrillation'],
                                  columns=['Sinus Rhythm', 'Atrial Fibrillation'])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5))
        fig.tight_layout(pad=5.0)
        ax1.title.set_text("Train/Dev")
        ax2.title.set_text("Test")

        sn.heatmap(df_cm_dev, annot=True, fmt='g', cmap=plt.cm.Blues, ax=ax1)
        sn.heatmap(df_cm_test, annot=True, fmt='g', cmap=plt.cm.Blues, ax=ax2)

        ax1.set(xlabel="Predicted label", ylabel="True label")
        ax2.set(xlabel="Predicted label", ylabel="True label")

        plt.suptitle("Confusion matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "ConfusionMatrix_Encoder.png"))
        print('Confusion matrices saved in: ' + os.path.join(output_dir, "ConfusionMatrix_Encoder.png"))
        plt.close()

        # Test predictions probabilities
        y_pred = np.where(encoder_model.predict(X_test) > 0.5, 1, 0)
        i = 0
        results = []
        for y in y_test:
            real_class = "Sinus Rhythm"
            pred_class = "Sinus Rhythm"
            success = "YES"
            if (y == 1):
                real_class = "Atrial Fibrillation"
            if (y_pred[i][0] == 1):
                pred_class = "Atrial Fibrillation"
            if (y != y_pred[i][0] == 1.0):
                success = "NO"

            results.append(["Instance: " + str(ids_test[i]), "Class: " + str(y) + "(" + real_class + ")",
                            "Predicted: " + str(y_pred[i][0]) + "(" + pred_class + ")",
                            "Success: " + success])
            i += 1

        file.write("\n\nPredictions on Test set:\n")
        file.write(tabulate(results))

        # Accuracy history plot
        accuracy = train.history['accuracy']
        val_accuracy = train.history['val_accuracy']
        epochs = range(len(accuracy))
        plt.plot(epochs, accuracy, color='blue', marker='o', label='Training accuracy')
        plt.plot(epochs, val_accuracy, color='red', marker='o', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        print("Model accuracy history plot saved in: " + os.path.join(output_dir, "Training_Validation_Accuracy.png"))
        plt.savefig(os.path.join(output_dir, "Training_Validation_Accuracy.png"))
        plt.close()

        # AUC history plot
        train_auc = train.history['AUC']
        val_auc = train.history['val_AUC']
        plt.plot(epochs, train_auc, color='blue', marker='o', label='Training AUC')
        plt.plot(epochs, val_auc, color='red', marker='o', label='Validation AUC')
        plt.title('Training and validation AUC')
        plt.xlabel("Epochs")
        plt.ylabel("AUC")
        plt.legend()
        print("Model AUC history plot saved in: " + os.path.join(output_dir, "Training_Validation_AUC.png"))
        plt.savefig(os.path.join(output_dir, "Training_Validation_AUC.png"))
        plt.close()

        # Loss history plot
        loss = train.history['loss']
        val_loss = train.history['val_loss']
        plt.plot(epochs, loss, color='blue', marker='o', label='Training loss')
        plt.plot(epochs, val_loss, color='red', marker='o', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        print("Model loss history plot saved in: " + os.path.join(output_dir, "Training_Validation_Loss.png"))
        plt.savefig(os.path.join(output_dir, "Training_Validation_Loss.png"))
        plt.close()

        # ROC curve and AUC (Dev)
        # roc curve for models
        y_pred_keras = encoder_model.predict(X_dev).ravel()
        fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_dev, y_pred_keras)
        auc_keras = auc(fpr_keras, tpr_keras)
        # roc curve for tpr = fpr
        random_probs = [0 for i in range(len(y_dev))]
        p_fpr, p_tpr, _ = roc_curve(y_dev, random_probs, pos_label=1)
        # plot roc curves
        plt.plot(fpr_keras, tpr_keras, color='red',
                 label='Encoder (AUC = %0.4f)' % auc_keras)
        plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
        # title
        plt.title("ROC curve (Dev)")
        # x label
        plt.xlabel('False Positive Rate')
        # y label
        plt.ylabel('True Positive rate')
        plt.legend(loc='best')
        plt.savefig(os.path.join(output_dir, "ROCcurve(Dev)_Encoder.png"))
        print('Dev ROC curve and AUC saved in: ' + os.path.join(output_dir, "ROCcurve(Dev)_Encoder.png"))
        plt.close()

        # ROC curve and AUC (Test)
        # roc curve for models
        y_pred_keras = encoder_model.predict(X_test).ravel()
        fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
        auc_keras = auc(fpr_keras, tpr_keras)
        # roc curve for tpr = fpr
        random_probs = [0 for i in range(len(y_test))]
        p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)
        # plot roc curves
        plt.plot(fpr_keras, tpr_keras, color='red',
                 label='Encoder (AUC = %0.4f)' % auc_keras)
        plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
        # title
        plt.title("ROC curve (Test)")
        # x label
        plt.xlabel('False Positive Rate')
        # y label
        plt.ylabel('True Positive rate')
        plt.legend(loc='best')
        plt.savefig(os.path.join(output_dir, "ROCcurve(Test)_Encoder.png"))
        print('Test ROC curve and AUC saved in: ' + os.path.join(output_dir, "ROCcurve(Test)_Encoder.png"))
        plt.close()

        print('Model evaluation report saved in: ' + os.path.join(output_dir, "REPORT_Encoder.txt"))

