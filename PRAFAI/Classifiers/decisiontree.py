import pickle
import warnings
from pathlib import Path

import joblib

warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, plot_roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import seaborn as sn


class DecisionTree(object):

    @staticmethod
    def train(train_dev, test, file_name, path):
        file = open(os.path.join(path, "REPORT_DecisionTree_" + file_name + ".txt"), "w")
        file.write(file_name + ":\n\n")

        y_train = train_dev["sigue_fa"]
        y_test = test["sigue_fa"]
        X_train = train_dev.drop("sigue_fa", 1)
        X_test = test.drop("sigue_fa", 1)

        # Model and predictions
        classifier = DecisionTreeClassifier()
        classifier.fit(X_train, y_train)
        y_pred_train = cross_val_predict(classifier, X_train, y_train, cv=10)
        y_pred_test = classifier.predict(X_test)

        # Save the model
        joblib.dump(classifier, os.path.join(path, "DT_MODEL_" + file_name + ".joblib"), compress=3)
        print('Trained model saved in: ' + os.path.join(path, os.path.join(path, "DT_MODEL_" + file_name + ".joblib")))

        print('Evaluating model...')

        # Classification report
        file.write("Classification Report (Train/Dev):\n" + classification_report(y_train, y_pred_train))

        # Confusion matrix (Train/Dev)
        conf_mat_train = confusion_matrix(y_train, y_pred_train)
        tn, fp, fn, tp = conf_mat_train.ravel()
        specificity_train = tn / (tn + fp)
        sensitivity_train = tp / (tp + fn)
        file.write("\nspecificity\t\t" + str(specificity_train))
        file.write("\nsensitivity\t\t" + str(sensitivity_train) + "\n\n")
        df_cm_train = pd.DataFrame(conf_mat_train, index=['no AF recurrence', 'AF recurrence'],
                                   columns=['no AF recurrence', 'AF recurrence'])

        # Classification report (Test)
        file.write("Classification Report (Test):\n" + classification_report(y_test, y_pred_test))

        # Confusion matrix (Test)
        conf_mat_test = confusion_matrix(y_test, y_pred_test)
        tn, fp, fn, tp = conf_mat_test.ravel()
        specificity_test = tn / (tn + fp)
        sensitivity_test = tp / (tp + fn)
        file.write("\nspecificity\t\t" + str(specificity_test))
        file.write("\nsensitivity\t\t" + str(sensitivity_test) + "\n")
        df_cm_test = pd.DataFrame(conf_mat_test, index=['no AF recurrence', 'AF recurrence'],
                                  columns=['no AF recurrence', 'AF recurrence'])

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
        plt.savefig(os.path.join(path, "ConfusionMatrix_" + file_name + ".png"))
        #plt.figure(figsize=(6, 10))
        plt.close()
        print('Confusion matrixes saved in: ' + os.path.join(path, "ConfusionMatrix_" + file_name + ".png"))


        # ROC curve (Train/Dev)
        cv = StratifiedKFold(n_splits=10)
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        fig, ax = plt.subplots()
        classifier_roc = DecisionTreeClassifier()
        for i, (train, test) in enumerate(cv.split(X_train, y_train)):
            classifier_roc.fit(X_train.iloc[train], y_train.iloc[train])
            viz = plot_roc_curve(classifier_roc, X_train.iloc[test], y_train.iloc[test],
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
               title="ROC curve (Train/Dev), " + file_name)
        ax.legend(loc="lower right")
        plt.savefig(os.path.join(path, "ROCcurve(Train_Dev)_" + file_name + ".png"))
        print('Validation ROC curve and AUC saved in: ' + os.path.join(path, "ROCcurve(Train_Dev)_" + file_name + ".png"))
        plt.close()

        # ROC curve and AUC (Test)
        # roc curve for models
        y_pred_prob = classifier.predict_proba(X_test)
        fpr, tpr, thresh = roc_curve(y_test, y_pred_prob[:, 1], pos_label=1)
        auc_metric = auc(fpr, tpr)
        # roc curve for tpr = fpr
        random_probs = [0 for i in range(len(y_test))]
        p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)
        # plot roc curves
        plt.plot(fpr, tpr, color='red',
                 label='Decision Tree (AUC = %0.2f)' % auc_metric)
        plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
        # title
        plt.title('ROC curve (Test), ' + file_name)
        # x label
        plt.xlabel('False Positive Rate')
        # y label
        plt.ylabel('True Positive rate')
        plt.legend(loc='best')
        plt.savefig(os.path.join(path, "ROCcurve(Test)_" + file_name + ".png"))
        print('Test ROC curve and AUC saved in: ' + os.path.join(path, "ROCcurve(Test)_" + file_name + ".png"))
        plt.close()

        print('Model evaluation report saved in: ' + os.path.join(path, "REPORT_DecisionTree_" + file_name + ".txt"))

        return y_train, mean_fpr, mean_tpr, mean_auc, classifier, X_test, y_test

    @staticmethod
    def evaluation_comparison(results, path):  # Por cada elemento en results obtenemos un array: [y_train, fpr, tpr, auc, classifier, X_test, y_test]

        algorithm_name = 'Decision Tree'

        # StandardScaler
        if (len([val for val in results.keys() if 'StandardScaler' in val]) > 0):
            Path(os.path.join(path, "StandardScaler", "Comparisons")).mkdir(parents=True, exist_ok=True)
            result_mean_SS = []
            result_med_SS = []
            result_pred_SS = []
            result_diff_SS = []
            for val in results.keys():
                if '_mean_StandardScaler' in val:
                    result_mean_SS = results[val]
            for val in results.keys():
                if '_med_StandardScaler' in val:
                    result_med_SS = results[val]
            for val in results.keys():
                if '_pred_StandardScaler' in val:
                    result_pred_SS = results[val]
            for val in results.keys():
                if '_diff_StandardScaler' in val:
                    result_diff_SS = results[val]

            # ROC curve and AUC (Train/Dev) --> StandardScaler
            plt.style.use('seaborn')

            # roc curve for tpr = fpr
            random_probs = [0 for i in range(len(result_mean_SS[0]))]
            p_fpr, p_tpr, _ = roc_curve(result_mean_SS[0], random_probs, pos_label=1)

            # plot roc curves
            if (len(result_mean_SS) != 0):
                plt.plot(result_mean_SS[1], result_mean_SS[2], color='orange',
                         label=algorithm_name + ', MV mean (AUC = %0.2f)' % result_mean_SS[3])
            if (len(result_med_SS) != 0):
                plt.plot(result_med_SS[1], result_med_SS[2], color='red',
                         label=algorithm_name + ', MV median (AUC = %0.2f)' % result_med_SS[3])
            if (len(result_pred_SS) != 0):
                plt.plot(result_pred_SS[1], result_pred_SS[2], color='green',
                         label=algorithm_name + ', MV predicted (AUC = %0.2f)' % result_pred_SS[3])
            if (len(result_diff_SS) != 0):
                plt.plot(result_diff_SS[1], result_diff_SS[2], color='purple',
                         label=algorithm_name + ', MV different value (AUC = %0.2f)' % result_diff_SS[3])
            plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
            # title
            plt.title('ROC curve (Train/Dev) --> StandardScaler')
            # x label
            plt.xlabel('False Positive Rate')
            # y label
            plt.ylabel('True Positive rate')

            plt.legend(loc='best')
            plt.savefig(os.path.join(path, "StandardScaler", "Comparisons",
                                     "ROCcurve(Train_Dev)_Comparison_StandardScaler.png"))
            print("\n[StandardScaler] Validation set ROC curve comparison saved in: " + os.path.join(path,
                                                                                                     "StandardScaler",
                                                                                                     "ROCcurve(Train_Dev)_Comparison_StandardScaler.png"))
            plt.close()

            # ROC curve and AUC (Test) --> StandardScaler
            plt.style.use('seaborn')

            # roc curve for models
            if (len(result_mean_SS) != 0):
                y_pred_prob_mean = result_mean_SS[4].predict_proba(result_mean_SS[5])
                fpr_mean, tpr_mean, thresh_mean = roc_curve(result_mean_SS[6], y_pred_prob_mean[:, 1], pos_label=1)
                auc_mean = auc(fpr_mean, tpr_mean)
            if (len(result_med_SS) != 0):
                y_pred_prob_median = result_med_SS[4].predict_proba(result_med_SS[5])
                fpr_median, tpr_median, thresh_median = roc_curve(result_pred_SS[6], y_pred_prob_median[:, 1],
                                                                  pos_label=1)
                auc_median = auc(fpr_median, tpr_median)
            if (len(result_pred_SS) != 0):
                y_pred_prob_predicted = result_pred_SS[4].predict_proba(result_pred_SS[5])
                fpr_predicted, tpr_predicted, thresh_predicted = roc_curve(result_pred_SS[6],
                                                                           y_pred_prob_predicted[:, 1], pos_label=1)
                auc_predicted = auc(fpr_predicted, tpr_predicted)
            if (len(result_diff_SS) != 0):
                y_pred_prob_diff = result_diff_SS[4].predict_proba(result_diff_SS[5])
                fpr_diff, tpr_diff, thresh_diff = roc_curve(result_diff_SS[6], y_pred_prob_diff[:, 1], pos_label=1)
                auc_diff = auc(fpr_diff, tpr_diff)

            # roc curve for tpr = fpr
            random_probs = [0 for i in range(len(result_mean_SS[6]))]
            p_fpr, p_tpr, _ = roc_curve(result_mean_SS[6], random_probs, pos_label=1)

            # plot roc curves
            if (len(result_mean_SS) != 0):
                plt.plot(fpr_mean, tpr_mean, color='orange',
                         label=algorithm_name + ', MV mean (AUC = %0.2f)' % auc_mean)
            if (len(result_med_SS) != 0):
                plt.plot(fpr_median, tpr_median, color='red',
                         label=algorithm_name + ', MV median (AUC = %0.2f)' % auc_median)
            if (len(result_pred_SS) != 0):
                plt.plot(fpr_predicted, tpr_predicted, color='green',
                         label=algorithm_name + ', MV predicted (AUC = %0.2f)' % auc_predicted)
            if (len(result_diff_SS) != 0):
                plt.plot(fpr_diff, tpr_diff, color='purple',
                         label=algorithm_name + ', MV different value (AUC = %0.2f)' % auc_diff)
            plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
            # title
            plt.title('ROC curve (Test) --> StandardScaler')
            # x label
            plt.xlabel('False Positive Rate')
            # y label
            plt.ylabel('True Positive rate')

            plt.legend(loc='best')
            plt.savefig(
                os.path.join(path, "StandardScaler", "Comparisons", "ROCcurve(Test)_Comparison_StandardScaler.png"))
            print("\n[StandardScaler] Test set ROC curve comparison saved in: " + os.path.join(path, "StandardScaler",
                                                                                               "Comparisons",
                                                                                               "ROCcurve(Test)_Comparison_StandardScaler.png"))
            plt.close()

        # MinMaxScaler
        if (len([val for val in results.keys() if 'MinMaxScaler' in val]) > 0):
            Path(os.path.join(path, "MinMaxScaler", "Comparisons")).mkdir(parents=True, exist_ok=True)
            result_mean_MMS = []
            result_med_MMS = []
            result_pred_MMS = []
            result_diff_MMS = []
            for val in results.keys():
                if '_mean_MinMaxScaler' in val:
                    result_mean_MMS = results[val]
            for val in results.keys():
                if '_med_MinMaxScaler' in val:
                    result_med_MMS = results[val]
            for val in results.keys():
                if '_pred_MinMaxScaler' in val:
                    result_pred_MMS = results[val]
            for val in results.keys():
                if '_diff_MinMaxScaler' in val:
                    result_diff_MMS = results[val]

            # ROC curve and AUC (Train/Dev) --> MinMaxScaler
            plt.style.use('seaborn')

            # roc curve for tpr = fpr
            random_probs = [0 for i in range(len(result_mean_MMS[0]))]
            p_fpr, p_tpr, _ = roc_curve(result_mean_MMS[0], random_probs, pos_label=1)

            # plot roc curves
            if (len(result_mean_MMS) != 0):
                plt.plot(result_mean_MMS[1], result_mean_MMS[2], color='orange',
                         label=algorithm_name + ', MV mean (AUC = %0.2f)' % result_mean_MMS[3])
            if (len(result_med_MMS) != 0):
                plt.plot(result_med_MMS[1], result_med_MMS[2], color='red',
                         label=algorithm_name + ', MV median (AUC = %0.2f)' % result_med_MMS[3])
            if (len(result_pred_MMS) != 0):
                plt.plot(result_pred_MMS[1], result_pred_MMS[2], color='green',
                         label=algorithm_name + ', MV predicted (AUC = %0.2f)' % result_pred_MMS[3])
            if (len(result_diff_MMS) != 0):
                plt.plot(result_diff_MMS[1], result_diff_MMS[2], color='purple',
                         label=algorithm_name + ', MV different value (AUC = %0.2f)' % result_diff_MMS[3])
            plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
            # title
            plt.title('ROC curve (Train/Dev) --> MinMaxScaler')
            # x label
            plt.xlabel('False Positive Rate')
            # y label
            plt.ylabel('True Positive rate')

            plt.legend(loc='best')
            plt.savefig(
                os.path.join(path, "MinMaxScaler", "Comparisons", "ROCcurve(Train_Dev)_Comparison_MinMaxScaler.png"))
            print("\n[MinMaxScaler] Validation set ROC curve comparison saved in: " + os.path.join(path, "MinMaxScaler",
                                                                                                   "Comparisons",
                                                                                                   "ROCcurve(Train_Dev)_Comparison_MinMaxScaler.png"))
            plt.close()

            # ROC curve and AUC (Test) --> MinMaxScaler
            plt.style.use('seaborn')

            # roc curve for models
            if (len(result_mean_MMS) != 0):
                y_pred_prob_mean = result_mean_MMS[4].predict_proba(result_mean_MMS[5])
                fpr_mean, tpr_mean, thresh_mean = roc_curve(result_mean_MMS[6], y_pred_prob_mean[:, 1], pos_label=1)
                auc_mean = auc(fpr_mean, tpr_mean)
            if (len(result_med_MMS) != 0):
                y_pred_prob_median = result_med_MMS[4].predict_proba(result_med_MMS[5])
                fpr_median, tpr_median, thresh_median = roc_curve(result_med_MMS[6], y_pred_prob_median[:, 1],
                                                                  pos_label=1)
                auc_median = auc(fpr_median, tpr_median)
            if (len(result_pred_MMS) != 0):
                y_pred_prob_predicted = result_pred_MMS[4].predict_proba(result_pred_MMS[5])
                fpr_predicted, tpr_predicted, thresh_predicted = roc_curve(result_pred_MMS[6],
                                                                           y_pred_prob_predicted[:, 1], pos_label=1)
                auc_predicted = auc(fpr_predicted, tpr_predicted)
            if (len(result_diff_MMS) != 0):
                y_pred_prob_diff = result_diff_MMS[4].predict_proba(result_diff_MMS[5])
                fpr_diff, tpr_diff, thresh_diff = roc_curve(result_diff_MMS[6], y_pred_prob_diff[:, 1], pos_label=1)
                auc_diff = auc(fpr_diff, tpr_diff)

            # roc curve for tpr = fpr
            random_probs = [0 for i in range(len(result_mean_MMS[6]))]
            p_fpr, p_tpr, _ = roc_curve(result_mean_MMS[6], random_probs, pos_label=1)

            # plot roc curves
            if (len(result_mean_MMS) != 0):
                plt.plot(fpr_mean, tpr_mean, color='orange',
                         label=algorithm_name + ', MV mean (AUC = %0.2f)' % auc_mean)
            if (len(result_med_MMS) != 0):
                plt.plot(fpr_median, tpr_median, color='red',
                         label=algorithm_name + ', MV median (AUC = %0.2f)' % auc_median)
            if (len(result_pred_MMS) != 0):
                plt.plot(fpr_predicted, tpr_predicted, color='green',
                         label=algorithm_name + ', MV predicted (AUC = %0.2f)' % auc_predicted)
            if (len(result_diff_MMS) != 0):
                plt.plot(fpr_diff, tpr_diff, color='purple',
                         label=algorithm_name + ', MV different value (AUC = %0.2f)' % auc_diff)
            plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
            # title
            plt.title('ROC curve (Test) --> MinMaxScaler')
            # x label
            plt.xlabel('False Positive Rate')
            # y label
            plt.ylabel('True Positive rate')

            plt.legend(loc='best')
            plt.savefig(os.path.join(path, "MinMaxScaler", "Comparisons", "ROCcurve(Test)_Comparison_MinMaxScaler.png"))
            print("\n[MinMaxScaler] Test set ROC curve comparison saved in: " + os.path.join(path, "MinMaxScaler",
                                                                                             "Comparisons",
                                                                                             "ROCcurve(Test)_Comparison_MinMaxScaler.png"))
            plt.close()

        # MaxAbsScaler
        if (len([val for val in results.keys() if 'MaxAbsScaler' in val]) > 0):
            Path(os.path.join(path, "MaxAbsScaler", "Comparisons")).mkdir(parents=True, exist_ok=True)
            result_mean_MAS = []
            result_med_MAS = []
            result_pred_MAS = []
            result_diff_MAS = []
            for val in results.keys():
                if '_mean_MaxAbsScaler' in val:
                    result_mean_MAS = results[val]
            for val in results.keys():
                if '_med_MaxAbsScaler' in val:
                    result_med_MAS = results[val]
            for val in results.keys():
                if '_pred_MaxAbsScaler' in val:
                    result_pred_MAS = results[val]
            for val in results.keys():
                if '_diff_MaxAbsScaler' in val:
                    result_diff_MAS = results[val]

            # ROC curve and AUC (Train/Dev) --> MaxAbsScaler
            plt.style.use('seaborn')

            # roc curve for tpr = fpr
            random_probs = [0 for i in range(len(result_mean_MAS[0]))]
            p_fpr, p_tpr, _ = roc_curve(result_mean_MAS[0], random_probs, pos_label=1)

            # plot roc curves
            if (len(result_mean_MAS) != 0):
                plt.plot(result_mean_MAS[1], result_mean_MAS[2], color='orange',
                         label=algorithm_name + ', MV mean (AUC = %0.2f)' % result_mean_MAS[3])
            if (len(result_med_MAS) != 0):
                plt.plot(result_med_MAS[1], result_med_MAS[2], color='red',
                         label=algorithm_name + ', MV median (AUC = %0.2f)' % result_med_MAS[3])
            if (len(result_pred_MAS) != 0):
                plt.plot(result_pred_MAS[1], result_pred_MAS[2], color='green',
                         label=algorithm_name + ', MV predicted (AUC = %0.2f)' % result_pred_MAS[3])
            if (len(result_diff_MAS) != 0):
                plt.plot(result_diff_MAS[1], result_diff_MAS[2], color='purple',
                         label=algorithm_name + ', MV different value (AUC = %0.2f)' % result_diff_MAS[3])
            plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
            # title
            plt.title('ROC curve (Train/Dev) --> MaxAbsScaler')
            # x label
            plt.xlabel('False Positive Rate')
            # y label
            plt.ylabel('True Positive rate')

            plt.legend(loc='best')
            plt.savefig(
                os.path.join(path, "MaxAbsScaler", "Comparisons", "ROCcurve(Train_Dev)_Comparison_MaxAbsScaler.png"))
            print("\n[MaxAbsScaler] Validation set ROC curve comparison saved in: " + os.path.join(path, "MaxAbsScaler",
                                                                                                   "Comparisons",
                                                                                                   "ROCcurve(Train_Dev)_Comparison_MaxAbsScaler.png"))
            plt.close()

            # ROC curve and AUC (Test) --> MaxAbsScaler
            plt.style.use('seaborn')

            # roc curve for models
            if (len(result_mean_MAS) != 0):
                y_pred_prob_mean = result_mean_MAS[4].predict_proba(result_mean_MAS[5])
                fpr_mean, tpr_mean, thresh_mean = roc_curve(result_mean_MAS[6], y_pred_prob_mean[:, 1], pos_label=1)
                auc_mean = auc(fpr_mean, tpr_mean)
            if (len(result_med_MAS) != 0):
                y_pred_prob_median = result_med_MAS[4].predict_proba(result_med_MAS[5])
                fpr_median, tpr_median, thresh_median = roc_curve(result_med_MAS[6], y_pred_prob_median[:, 1],
                                                                  pos_label=1)
                auc_median = auc(fpr_median, tpr_median)
            if (len(result_pred_MAS) != 0):
                y_pred_prob_predicted = result_pred_MAS[4].predict_proba(result_pred_MAS[5])
                fpr_predicted, tpr_predicted, thresh_predicted = roc_curve(result_pred_MAS[6],
                                                                           y_pred_prob_predicted[:, 1], pos_label=1)
                auc_predicted = auc(fpr_predicted, tpr_predicted)
            if (len(result_diff_MAS) != 0):
                y_pred_prob_diff = result_diff_MAS[4].predict_proba(result_diff_MAS[5])
                fpr_diff, tpr_diff, thresh_diff = roc_curve(result_diff_MAS[6], y_pred_prob_diff[:, 1], pos_label=1)
                auc_diff = auc(fpr_diff, tpr_diff)

            # roc curve for tpr = fpr
            random_probs = [0 for i in range(len(result_mean_MAS[6]))]
            p_fpr, p_tpr, _ = roc_curve(result_mean_MAS[6], random_probs, pos_label=1)

            # plot roc curves
            if (len(result_mean_MAS) != 0):
                plt.plot(fpr_mean, tpr_mean, color='orange',
                         label=algorithm_name + ', MV mean (AUC = %0.2f)' % auc_mean)
            if (len(result_med_MAS) != 0):
                plt.plot(fpr_median, tpr_median, color='red',
                         label=algorithm_name + ', MV median (AUC = %0.2f)' % auc_median)
            if (len(result_pred_MAS) != 0):
                plt.plot(fpr_predicted, tpr_predicted, color='green',
                         label=algorithm_name + ', MV predicted (AUC = %0.2f)' % auc_predicted)
            if (len(result_diff_MAS) != 0):
                plt.plot(fpr_diff, tpr_diff, color='purple',
                         label=algorithm_name + ', MV different value (AUC = %0.2f)' % auc_diff)
            plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
            # title
            plt.title('ROC curve (Test) --> MaxAbsScaler')
            # x label
            plt.xlabel('False Positive Rate')
            # y label
            plt.ylabel('True Positive rate')

            plt.legend(loc='best')
            plt.savefig(os.path.join(path, "MaxAbsScaler", "Comparisons", "ROCcurve(Test)_Comparison_MaxAbsScaler.png"))
            print("\n[MaxAbsScaler] Test set ROC curve comparison saved in: " + os.path.join(path, "MaxAbsScaler",
                                                                                             "Comparisons",
                                                                                             "ROCcurve(Test)_Comparison_MaxAbsScaler.png"))
            plt.close()

        # QuantileTransformer
        if (len([val for val in results.keys() if 'QuantileTransformer' in val]) > 0):
            Path(os.path.join(path, "QuantileTransformer", "Comparisons")).mkdir(parents=True, exist_ok=True)
            result_mean_QT = []
            result_med_QT = []
            result_pred_QT = []
            result_diff_QT = []
            for val in results.keys():
                if '_mean_QuantileTransformer' in val:
                    result_mean_QT = results[val]
            for val in results.keys():
                if '_med_QuantileTransformer' in val:
                    result_med_QT = results[val]
            for val in results.keys():
                if '_pred_QuantileTransformer' in val:
                    result_pred_QT = results[val]
            for val in results.keys():
                if '_diff_QuantileTransformer' in val:
                    result_diff_QT = results[val]

            # ROC curve and AUC (Train/Dev) --> QuantileTransformer
            plt.style.use('seaborn')

            # roc curve for tpr = fpr
            random_probs = [0 for i in range(len(result_mean_QT[0]))]
            p_fpr, p_tpr, _ = roc_curve(result_mean_QT[0], random_probs, pos_label=1)

            # plot roc curves
            if (len(result_mean_QT) != 0):
                plt.plot(result_mean_QT[1], result_mean_QT[2], color='orange',
                         label=algorithm_name + ', MV mean (AUC = %0.2f)' % result_mean_QT[3])
            if (len(result_med_QT) != 0):
                plt.plot(result_med_QT[1], result_med_QT[2], color='red',
                         label=algorithm_name + ', MV median (AUC = %0.2f)' % result_med_QT[3])
            if (len(result_pred_QT) != 0):
                plt.plot(result_pred_QT[1], result_pred_QT[2], color='green',
                         label=algorithm_name + ', MV predicted (AUC = %0.2f)' % result_pred_QT[3])
            if (len(result_diff_QT) != 0):
                plt.plot(result_diff_QT[1], result_diff_QT[2], color='purple',
                         label=algorithm_name + ', MV different value (AUC = %0.2f)' % result_diff_QT[3])
            plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
            # title
            plt.title('ROC curve (Train/Dev) --> QuantileTransformer')
            # x label
            plt.xlabel('False Positive Rate')
            # y label
            plt.ylabel('True Positive rate')

            plt.legend(loc='best')
            plt.savefig(os.path.join(path, "QuantileTransformer", "Comparisons",
                                     "ROCcurve(Train_Dev)_Comparison_QuantileTransformer.png"))
            print("\n[QuantileTransformer] Validation set ROC curve comparison saved in: " + os.path.join(path,
                                                                                                          "QuantileTransformer",
                                                                                                          "Comparisons",
                                                                                                          "ROCcurve(Train_Dev)_Comparison_QuantileTransformer.png"))
            plt.close()

            # ROC curve and AUC (Test) --> QuantileTransformer
            plt.style.use('seaborn')

            # roc curve for models
            if (len(result_mean_QT) != 0):
                y_pred_prob_mean = result_mean_QT[4].predict_proba(result_mean_QT[5])
                fpr_mean, tpr_mean, thresh_mean = roc_curve(result_mean_QT[6], y_pred_prob_mean[:, 1], pos_label=1)
                auc_mean = auc(fpr_mean, tpr_mean)
            if (len(result_med_QT) != 0):
                y_pred_prob_median = result_med_QT[4].predict_proba(result_med_QT[5])
                fpr_median, tpr_median, thresh_median = roc_curve(result_med_QT[6], y_pred_prob_median[:, 1],
                                                                  pos_label=1)
                auc_median = auc(fpr_median, tpr_median)
            if (len(result_pred_QT) != 0):
                y_pred_prob_predicted = result_pred_QT[4].predict_proba(result_pred_QT[5])
                fpr_predicted, tpr_predicted, thresh_predicted = roc_curve(result_pred_QT[6],
                                                                           y_pred_prob_predicted[:, 1], pos_label=1)
                auc_predicted = auc(fpr_predicted, tpr_predicted)
            if (len(result_diff_QT) != 0):
                y_pred_prob_diff = result_diff_QT[4].predict_proba(result_diff_QT[5])
                fpr_diff, tpr_diff, thresh_diff = roc_curve(result_diff_QT[6], y_pred_prob_diff[:, 1], pos_label=1)
                auc_diff = auc(fpr_diff, tpr_diff)

            # roc curve for tpr = fpr
            random_probs = [0 for i in range(len(result_mean_QT[6]))]
            p_fpr, p_tpr, _ = roc_curve(result_mean_QT[6], random_probs, pos_label=1)

            # plot roc curves
            if (len(result_mean_QT) != 0):
                plt.plot(fpr_mean, tpr_mean, color='orange',
                         label=algorithm_name + ', MV mean (AUC = %0.2f)' % auc_mean)
            if (len(result_med_QT) != 0):
                plt.plot(fpr_median, tpr_median, color='red',
                         label=algorithm_name + ', MV median (AUC = %0.2f)' % auc_median)
            if (len(result_pred_QT) != 0):
                plt.plot(fpr_predicted, tpr_predicted, color='green',
                         label=algorithm_name + ', MV predicted (AUC = %0.2f)' % auc_predicted)
            if (len(result_diff_QT) != 0):
                plt.plot(fpr_diff, tpr_diff, color='purple',
                         label=algorithm_name + ', MV different value (AUC = %0.2f)' % auc_diff)
            plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
            # title
            plt.title('ROC curve (Test) --> QuantileTransformer')
            # x label
            plt.xlabel('False Positive Rate')
            # y label
            plt.ylabel('True Positive rate')

            plt.legend(loc='best')
            plt.savefig(os.path.join(path, "QuantileTransformer", "Comparisons",
                                     "ROCcurve(Test)_Comparison_QuantileTransformer.png"))
            print("\n[QuantileTransformer] Test set ROC curve comparison saved in: " + os.path.join(path,
                                                                                                    "QuantileTransformer",
                                                                                                    "Comparisons",
                                                                                                    "ROCcurve(Test)_Comparison_QuantileTransformer.png"))
            plt.close()

        # Normalizer
        if (len([val for val in results.keys() if 'Normalizer' in val]) > 0):
            Path(os.path.join(path, "Normalizer", "Comparisons")).mkdir(parents=True, exist_ok=True)
            result_mean_N = []
            result_med_N = []
            result_pred_N = []
            result_diff_N = []
            for val in results.keys():
                if '_mean_Normalizer' in val:
                    result_mean_N = results[val]
            for val in results.keys():
                if '_med_Normalizer' in val:
                    result_med_N = results[val]
            for val in results.keys():
                if '_pred_Normalizer' in val:
                    result_pred_N = results[val]
            for val in results.keys():
                if '_diff_Normalizer' in val:
                    result_diff_N = results[val]

            # ROC curve and AUC (Train/Dev) --> Normalizer
            plt.style.use('seaborn')

            # roc curve for tpr = fpr
            random_probs = [0 for i in range(len(result_mean_N[0]))]
            p_fpr, p_tpr, _ = roc_curve(result_mean_N[0], random_probs, pos_label=1)

            # plot roc curves
            if (len(result_mean_N) != 0):
                plt.plot(result_mean_N[1], result_mean_N[2], color='orange',
                         label=algorithm_name + ', MV mean (AUC = %0.2f)' % result_mean_N[3])
            if (len(result_med_N) != 0):
                plt.plot(result_med_N[1], result_med_N[2], color='red',
                         label=algorithm_name + ', MV median (AUC = %0.2f)' % result_med_N[3])
            if (len(result_pred_N) != 0):
                plt.plot(result_pred_N[1], result_pred_N[2], color='green',
                         label=algorithm_name + ', MV predicted (AUC = %0.2f)' % result_pred_N[3])
            if (len(result_diff_N) != 0):
                plt.plot(result_diff_N[1], result_diff_N[2], color='purple',
                         label=algorithm_name + ', MV different value (AUC = %0.2f)' % result_diff_N[3])
            plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
            # title
            plt.title('ROC curve (Train/Dev) --> Normalizer')
            # x label
            plt.xlabel('False Positive Rate')
            # y label
            plt.ylabel('True Positive rate')

            plt.legend(loc='best')
            plt.savefig(
                os.path.join(path, "Normalizer", "Comparisons", "ROCcurve(Train_Dev)_Comparison_Normalizer.png"))
            print("\n[Normalizer] Validation set ROC curve comparison saved in: " + os.path.join(path, "Normalizer",
                                                                                                 "Comparisons",
                                                                                                 "ROCcurve(Train_Dev)_Comparison_Normalizer.png"))
            plt.close()

            # ROC curve and AUC (Test) --> Normalizer
            plt.style.use('seaborn')

            # roc curve for models
            if (len(result_mean_N) != 0):
                y_pred_prob_mean = result_mean_N[4].predict_proba(result_mean_N[5])
                fpr_mean, tpr_mean, thresh_mean = roc_curve(result_mean_N[6], y_pred_prob_mean[:, 1], pos_label=1)
                auc_mean = auc(fpr_mean, tpr_mean)
            if (len(result_med_N) != 0):
                y_pred_prob_median = result_med_N[4].predict_proba(result_med_N[5])
                fpr_median, tpr_median, thresh_median = roc_curve(result_med_N[6], y_pred_prob_median[:, 1],
                                                                  pos_label=1)
                auc_median = auc(fpr_median, tpr_median)
            if (len(result_pred_N) != 0):
                y_pred_prob_predicted = result_pred_N[4].predict_proba(result_pred_N[5])
                fpr_predicted, tpr_predicted, thresh_predicted = roc_curve(result_pred_N[6],
                                                                           y_pred_prob_predicted[:, 1], pos_label=1)
                auc_predicted = auc(fpr_predicted, tpr_predicted)
            if (len(result_diff_N) != 0):
                y_pred_prob_diff = result_diff_N[4].predict_proba(result_diff_N[5])
                fpr_diff, tpr_diff, thresh_diff = roc_curve(result_diff_N[6], y_pred_prob_diff[:, 1], pos_label=1)
                auc_diff = auc(fpr_diff, tpr_diff)

            # roc curve for tpr = fpr
            random_probs = [0 for i in range(len(result_mean_N[6]))]
            p_fpr, p_tpr, _ = roc_curve(result_mean_N[6], random_probs, pos_label=1)

            # plot roc curves
            if (len(result_mean_N) != 0):
                plt.plot(fpr_mean, tpr_mean, color='orange',
                         label=algorithm_name + ', MV mean (AUC = %0.2f)' % auc_mean)
            if (len(result_med_N) != 0):
                plt.plot(fpr_median, tpr_median, color='red',
                         label=algorithm_name + ', MV median (AUC = %0.2f)' % auc_median)
            if (len(result_pred_N) != 0):
                plt.plot(fpr_predicted, tpr_predicted, color='green',
                         label=algorithm_name + ', MV predicted (AUC = %0.2f)' % auc_predicted)
            if (len(result_diff_N) != 0):
                plt.plot(fpr_diff, tpr_diff, color='purple',
                         label=algorithm_name + ', MV different value (AUC = %0.2f)' % auc_diff)
            plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
            # title
            plt.title('ROC curve (Test) --> Normalizer')
            # x label
            plt.xlabel('False Positive Rate')
            # y label
            plt.ylabel('True Positive rate')

            plt.legend(loc='best')
            plt.savefig(os.path.join(path, "Normalizer", "Comparisons", "ROCcurve(Test)_Comparison_Normalizer.png"))
            print("\n[Normalizer] Test set ROC curve comparison saved in: " + os.path.join(path, "Normalizer",
                                                                                           "Comparisons",
                                                                                           "ROCcurve(Test)_Comparison_Normalizer.png"))
            plt.close()
