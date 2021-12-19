import math
import os
from pathlib import Path
import seaborn as sn
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, plot_roc_curve, auc, roc_curve
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
from xgboost import XGBClassifier


class XGBoostClassifier(object):

    @staticmethod
    def model(ecg_dict, ages, sexes, labels, xgb_features, output_dir):
        ids = list(ecg_dict.keys())

        if(xgb_features == 'AS'):    # Crear un modelo con la edad y el sexo
            print("[Model AS] Features to use: AGE and SEX")
            data = pd.DataFrame(list(zip(ages, sexes, labels)), columns=['Age', 'Sex', 'Labels'])
            XGBoostClassifier.train_eval(object, ids, data.drop('Labels', 1), data['Labels'], xgb_features, output_dir)

        elif(xgb_features == 'Leads'):  # Crear un modelo por cada Lead
            print("Creating a model for each ECG lead...")
            leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
            for lead in leads:
                print("\n[MODEL " + lead + "] Features to use: " + lead + " lead data")
                data = XGBoostClassifier.extract_features(object, ecg_dict, lead)
                data['Labels'] = labels
                XGBoostClassifier.train_eval(object, ids, data.drop('Labels', 1), data['Labels'], lead, output_dir)

        elif(xgb_features == 'RMS'):  # Crear un solo modelo con la media cuadrática de cada Lead
            print("[Model RMS] Features to use: root mean squared for each of 12 leads data")
            data = XGBoostClassifier.extract_features(object, ecg_dict, 'RMS')
            data['Labels'] = labels
            XGBoostClassifier.train_eval(object, ids, data.drop('Labels', 1), data['Labels'], xgb_features, output_dir)


        elif (xgb_features == 'Leads_AS'):  # Crear un modelo por cada Lead + Edad y Sexo
            print("Creating a model for each ECG lead...")
            leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
            for lead in leads:
                print("\n[MODEL " + lead + "_AS] Features to use: " + lead + " lead data + AGE and SEX")
                data = XGBoostClassifier.extract_features(object, ecg_dict, lead)
                data['Age'] = ages
                data['Sex'] = sexes
                data['Labels'] = labels
                XGBoostClassifier.train_eval(object, ids, data.drop('Labels', 1), data['Labels'], lead + '-AS', output_dir)

        elif (xgb_features == 'RMS_AS'):  # Crear un solo modelo con la media cuadrática de cada Lead + Edad y Sexo
            print("[Model RMS_AS] Features to use: root mean squared for each of 12 leads data + AGE and SEX")
            data = XGBoostClassifier.extract_features(object, ecg_dict, 'RMS')
            data['Age'] = ages
            data['Sex'] = sexes
            data['Labels'] = labels
            XGBoostClassifier.train_eval(object, ids, data.drop('Labels', 1), data['Labels'], xgb_features, output_dir)


    def extract_features(self, ecg_dict, input_type):
        new_ecg_dict = {}

        # Extraer el Lead correspondiente (vector de 5000 features)
        if input_type in ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']:
            lead = input_type
            for ecg_id in ecg_dict:
                new_ecg_dict[ecg_id] = ecg_dict[ecg_id][lead].to_numpy()
            features = pd.DataFrame.from_dict(new_ecg_dict, orient='index', columns=[str(x) for x in list(range(0, 5000))])
            features.reset_index(drop=True, inplace=True)
            return features

        # Media cuadrática de cada Lead (vector de 12 features)
        elif (input_type == 'RMS'):
            for ecg_id in ecg_dict:
                ecg = ecg_dict[ecg_id]
                rms_array = []
                for lead in ecg.columns:
                    lead_array = ecg[lead].to_numpy()
                    rms_lead = round(math.sqrt(sum([x ** 2 for x in lead_array]) / len(lead_array)), 2)
                    rms_array.append(rms_lead)
                new_ecg_dict[ecg_id] = rms_array
            features = pd.DataFrame.from_dict(new_ecg_dict, orient='index', columns=[str(x) for x in list(range(0, 12))])
            features.reset_index(drop=True, inplace=True)
            return features


    def train_eval(self, ids, features, labels, input_type, output_dir):
        Path(os.path.join(output_dir, input_type)).mkdir(parents=True, exist_ok=True)
        output_dir = os.path.join(output_dir, input_type)

        file = open(os.path.join(output_dir, "REPORT_XGBoost_" + input_type + ".txt"), "w")
        file.write("XGBoost model, input features: " + input_type + "\n\n")

        # Train(70%)/Dev(15%)/Test(15%) split
        indices = np.array(range(len(features)))
        X_train_dev, X_test, y_train_dev, y_test, indices_train_dev, indices_test = train_test_split(features, labels, indices, test_size=0.15, random_state=42, shuffle=True, stratify=labels)
        X_train, X_dev, y_train, y_dev, indices_train, indices_dev = train_test_split(X_train_dev, y_train_dev, indices_train_dev, test_size=(len(y_test) / len(y_train_dev)), random_state=42, shuffle=True, stratify=y_train_dev)

        ids_test = [ids[x] for x in indices_test]

        # Standardization
        print("\nStandardization of datasets...")
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_train_dev = scaler.transform(X_train_dev)
        X_test = scaler.transform(X_test)
        joblib.dump(scaler, os.path.join(output_dir, 'std_scaler_xgb_' + input_type + '.bin'), compress=True)
        print("Standard scaler saved in: " + str(os.path.join(output_dir, 'std_scaler_xgb_' + input_type + '.bin')))

        # Model and predictions
        print("\nTraining XGBoost model...")
        classifier = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        classifier.fit(X_train, y_train)
        y_pred_train_dev = cross_val_predict(classifier, X_train_dev, y_train_dev, cv=10)
        y_pred_test = classifier.predict(X_test)

        # Save the model
        joblib.dump(classifier, os.path.join(output_dir, "XGBoost_MODEL_" + input_type + ".joblib"), compress=3)
        print('Trained model saved in: ' + os.path.join(output_dir, "XGBoost_MODEL_" + input_type + ".joblib"))

        print('\nEvaluating model...')

        # Classification report
        print('Evaluating on Dev set...')
        file.write("\nClassification Report (Train/Dev):\n" + classification_report(y_train_dev, y_pred_train_dev))

        # Confusion matrix (Train/Dev)
        conf_mat_train = confusion_matrix(y_train_dev, y_pred_train_dev)
        tn, fp, fn, tp = conf_mat_train.ravel()
        specificity_train = tn / (tn + fp)
        sensitivity_train = tp / (tp + fn)
        file.write("\nspecificity\t\t" + str(round(specificity_train, 2)))
        file.write("\nsensitivity\t\t" + str(round(sensitivity_train, 2)) + "\n\n")
        df_cm_train = pd.DataFrame(conf_mat_train, index=['Sinus Rhythm', 'Atrial Fibrillation'],
                                   columns=['Sinus Rhythm', 'Atrial Fibrillation'])

        # Classification report (Test)
        print('Evaluating on Test set...')
        file.write("Classification Report (Test):\n" + classification_report(y_test, y_pred_test))

        # Confusion matrix (Test)
        conf_mat_test = confusion_matrix(y_test, y_pred_test)
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

        sn.heatmap(df_cm_train, annot=True, fmt='g', cmap=plt.cm.Blues, ax=ax1)
        sn.heatmap(df_cm_test, annot=True, fmt='g', cmap=plt.cm.Blues, ax=ax2)

        ax1.set(xlabel="Predicted label", ylabel="True label")
        ax2.set(xlabel="Predicted label", ylabel="True label")

        plt.suptitle("Confusion matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "ConfusionMatrix_XGBoost_" + input_type + ".png"))
        print('Confusion matrices saved in: ' + os.path.join(output_dir, "ConfusionMatrix_XGBoost_" + input_type + ".png"))
        plt.close()

        # ROC curve (Train/Dev)
        cv = StratifiedKFold(n_splits=10)
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        fig, ax = plt.subplots()
        classifier_roc = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        for i, (train, test) in enumerate(cv.split(X_train_dev, y_train_dev)):
            classifier_roc.fit(X_train_dev[train, :], y_train_dev.iloc[train])
            viz = plot_roc_curve(classifier_roc, X_train_dev[test, :], y_train_dev.iloc[test],
                                 name='ROC fold {}'.format(i),
                                 alpha=0.3, lw=1, ax=ax)
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                label='Chance', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color='b',
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')

        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
               title="ROC curve (Train/Dev)")
        ax.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, "ROCcurve(Train_Dev)_XGBoost_" + input_type + ".png"))
        print('Dev ROC curve and AUC saved in: ' + os.path.join(output_dir, "ROCcurve(Train_Dev)_XGBoost_" + input_type + ".png"))
        plt.close()

        # ROC curve and AUC (Test)
        plt.style.use('seaborn')
        # roc curve for models
        y_pred_prob = classifier.predict_proba(X_test)
        fpr, tpr, thresh = roc_curve(y_test, y_pred_prob[:, 1], pos_label=1)
        auc_median = auc(fpr, tpr)
        # roc curve for tpr = fpr
        random_probs = [0 for i in range(len(y_test))]
        p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)
        # plot roc curves
        plt.plot(fpr, tpr, color='red',
                 label='XGBoost, ' + input_type + ' (AUC = %0.2f)' % auc_median)
        plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
        # title
        plt.title('ROC curve (Test)')
        # x label
        plt.xlabel('False Positive Rate')
        # y label
        plt.ylabel('True Positive rate')
        plt.legend(loc='best')
        plt.savefig(os.path.join(output_dir, "ROCcurve(Test)_XGBoost_" + input_type + ".png"))
        print('Dev ROC curve and AUC saved in: ' + os.path.join(output_dir, "ROCcurve(Train_Dev)_XGBoost_" + input_type + ".png"))
        plt.close()

        # Test predictions probabilities
        i = 0
        results = []
        for index in y_test.index:
            real_class = "Sinus Rhythm"
            pred_class = "Sinus Rhythm"
            success = "YES"
            if (y_test[index] == 1.0):
                real_class = "Atrial Fibrillation"
            if (y_pred_test[i] == 1.0):
                pred_class = "Atrial Fibrillation"
                prob = str(round((y_pred_prob[i][1] * 100), 2)) + "%"
            else:
                prob = str(round((y_pred_prob[i][0] * 100), 2)) + "%"
            if (y_test[index] != y_pred_test[i] == 1.0):
                success = "NO"

            results.append(["Instance: " + str(ids_test[i]), "Class: " + str(y_test[index]) + "(" + real_class + ")",
                            "Predicted: " + str(y_pred_test[i]) + "(" + pred_class + ")", "Probability: " + prob,
                            "Success: " + success])
            i += 1

        file.write("\n\nPredictions on Test set:\n")
        file.write(tabulate(results))

        print('Model evaluation report saved in: ' + os.path.join(output_dir, "REPORT_XGBoost_" + input_type + ".txt"))

