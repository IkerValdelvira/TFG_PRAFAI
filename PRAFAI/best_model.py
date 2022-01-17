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

def impute_median_85_mv(train, test, dataset_train):
    train_copy = train.copy()
    test_copy = test.copy()

    median_dict = {}
    deleted_dict = set()
    for column in train_copy:
        if (column != "sigue_fa"):
            if (train_copy[column].isnull().sum() >= len(train_copy.index) * 0.85):  # Si más del 85% de los items tienen missing value, se elimina la columna                del train_copy[column]
                del train_copy[column]
                print("\tFeature '" + str(column) + "' was deleted because at least 85% of items contain missing values.")
                deleted_dict.add(column)
            elif (train_copy[column].dtype == np.float64):  # Se imputa la mediana de la columna si es una variable numérica
                column_median = round(dataset_train[column].median(), 3)
                train_copy[column] = train_copy[column].fillna(column_median)
                median_dict[column] = column_median
            elif (column in ["edad", "genero"]):    # Se guarda la mediana de la edad y el género para posibles imputaciones de nuevos items en un futuro
                column_median = dataset_train[column].median()
                median_dict[column] = column_median
    train_copy['imc'] = train_copy['imc'].round()

    # Cambios en test
    for column in deleted_dict:
        del test_copy[column]
    for column in median_dict:
        test_copy[column] = test_copy[column].fillna(median_dict[column])
    test_copy['imc'] = test_copy['imc'].round()

    # Guardar diccionario de imputaciones
    with open(os.path.join(output_dir, "FinalModel", "med_imputation_dict.json"), 'w') as file:
        json.dump(median_dict, file)

    return train_copy, test_copy


def standardization(train, test):
    y_train = train["sigue_fa"]
    X_train = train.drop("sigue_fa", 1)
    y_test = test["sigue_fa"]
    X_test = test.drop("sigue_fa", 1)

    columns = train.columns.tolist()
    cols_standardize_aux = ['fecha', 'urea', 'creatinina', 'albumina', 'glucosa', 'hba1c', 'potasio', 'calcio', 'hdl', 'no_hdl', 'ldl', 'colesterol', 'ntprobnp', 'bnp', 'troponina_tni', 'troponina_tnt', 'dimero_d', 'fibrinogeno', 'aldosterona', 'leucocitos', 'pct', 'tsh', 't4l', 'sodio', 'vsg', 'tl3', 'fevi', 'diametro_ai', 'area_ai', 'numero_ingresos', 'numero_dias_desde_ingreso_hasta_evento', 'numero_dias_ingresado', 'edad', 'talla', 'peso']
    cols_standardize = [var for var in cols_standardize_aux if var in list(columns)]
    cols_not_standardize_aux = ['tipo_fa', 'ecocardiograma', 'ecocardiograma_contraste', 'electrocardiograma', 'cardioversion', 'ablacion', 'marcapasos_dai', 'etilogia_cardiologica', 'depresion', 'alcohol', 'drogodependencia', 'ansiedad', 'demencia', 'insuficiencia_renal', 'menopausia', 'osteoporosis', 'diabetes_tipo1', 'diabetes_tipo2', 'dislipidemia', 'hipercolesterolemia', 'fibrilacion_palpitacion', 'flutter', 'insuficiencia_cardiaca', 'fumador', 'sahos', 'hipertiroidismo', 'sindrome_metabolico', 'hipertension_arterial', 'cardiopatia_isquemica', 'ictus', 'miocardiopatia', 'otras_arritmias_psicogena_ritmo_bigeminal', 'bloqueos_rama', 'bloqueo_auriculoventricular', 'bradicardia', 'contracciones_prematuras_ectopia_extrasistolica', 'posoperatoria', 'sinusal_coronaria', 'valvula_mitral_reumaticas', 'otras_valvulopatias', 'valvulopatia_mitral_congenita', 'arteriopatia_periferica', 'epoc', 'genero', 'pensionista', 'residenciado', 'imc', 'b01a', 'n02ba01', 'a02bc', 'c03', 'g03a', 'a10', 'n06a', 'n05a', 'n05b', 'c01', 'c01b', 'c02', 'c04', 'c07', 'c08', 'c09', 'c10', 'polimedicacion']
    cols_not_standardize = [var for var in cols_not_standardize_aux if var in list(columns)]
    ct = ColumnTransformer([('columntransformer', StandardScaler(), cols_standardize)], remainder='passthrough')
    ct.fit(X_train)
    X_train_array = ct.transform(X_train)
    X_test_array = ct.transform(X_test)

    array_columns = cols_standardize + cols_not_standardize
    train = pd.DataFrame(X_train_array, index=X_train.index, columns=array_columns)
    test = pd.DataFrame(X_test_array, index=X_test.index, columns=array_columns)

    train['sigue_fa'] = y_train
    test['sigue_fa'] = y_test

    # Guardar StandardScaler
    joblib.dump(ct, os.path.join(output_dir, "FinalModel", 'std_scaler.bin'), compress=True)

    return train, test


def rfe(train_dev, test, file_name):
    file = open(os.path.join(output_dir, "Preprocess", 'REPORT_RFE_' + file_name + '.txt'),"w")
    file.write("Dataset: " + file_name + "\n")

    # Borrar variables con varianza = 0
    selector = VarianceThreshold(threshold=(0))
    selector.fit(train_dev)
    all_columns = train_dev.columns
    train_dev = train_dev[train_dev.columns[selector.get_support(indices=True)]]
    current_columns = train_dev.columns
    deleted_columns = list(set(all_columns) - set(current_columns))
    file.write('\nDeleted features with variance to zero: {}'.format(len(deleted_columns)))
    file.write('\n' + str(deleted_columns))

    #RFE
    y_train = train_dev["sigue_fa"]
    X_train = train_dev.drop("sigue_fa", 1)

    # Create the RFE object and compute a cross-validated score.
    svc = SVC(kernel="linear")

    min_features_to_select = 1  # Minimum number of features to consider
    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(10),
                  scoring='roc_auc',
                  min_features_to_select=min_features_to_select)
    rfecv.fit(X_train, y_train)

    train_dev = X_train[X_train.columns[rfecv.get_support(indices=True)]]
    train_dev['sigue_fa'] = y_train
    feature_names = np.array(X_train.columns)

    file.write('\n\nNumber of current features: {}'.format((X_train.shape[1])))
    file.write('\nNumber of selected features: {}'.format(len(feature_names[rfecv.get_support()])))
    file.write('\nNumber of deleted features: {}'.format(len(list(set(feature_names) - set(feature_names[rfecv.get_support()])))))
    file.write('\n\nFeatures selected by RFE:')
    file.write('\n' + str(feature_names[rfecv.get_support()]))


    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.figtext(0.52, 0.15, "Optimal number of features: %d" % rfecv.n_features_, wrap=True, horizontalalignment='center', fontsize=10)
    plt.title("RFE Feature Selection --> " + file_name)
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (AUC)")
    plt.plot(range(min_features_to_select, len(rfecv.grid_scores_) + min_features_to_select), rfecv.grid_scores_)
    plt.savefig(os.path.join(output_dir, "Preprocess", 'RFE_' + file_name + '.png'))
    plt.close()

    # Borrar variables del Test
    removed_features = list(set(test.columns) - set(train_dev.columns))
    test = test.drop(removed_features, axis=1)

    return train_dev, test


def svm_model_evaluation(train, test, file_name):
    file = open(os.path.join(output_dir, "ModelEvaluation", 'REPORT_SVM.txt'), "w")
    file.write("Dataset: " + file_name + "\n")

    # Model Training --> SVM
    file.write("\nAlgorithm: SVM\n")
    file.write("Hyperparameters:\n")
    file.write("\tC: 1\n")
    file.write("\tgamma: 0.1\n")
    file.write("\tkernel: 'poly'\n")
    file.write("\tdegree: 2\n\n")
    y_train = train["sigue_fa"]
    y_test = test["sigue_fa"]
    X_train = train.drop("sigue_fa", 1)
    X_test = test.drop("sigue_fa", 1)
    feature_names = X_train.columns.tolist()

    # Model and predictions
    svm = SVC(probability=True, C=1, gamma=0.1, kernel='poly', degree=2)
    y_pred_train = cross_val_predict(svm, X_train, y_train, cv=10)
    svm.fit(X_train, y_train)
    y_pred_test = svm.predict(X_test)

    # Save the model
    joblib.dump(svm, os.path.join(output_dir, "ModelEvaluation", "SVM_MODEL.joblib"), compress=3)
    print('Trained model saved in: ' + os.path.join(output_dir, "ModelEvaluation", "SVM_MODEL.joblib"))

    print('Evaluating model...')

    # Classification report (Train/Dev)
    file.write("\nClassification Report (Train/Dev):\n" + classification_report(y_train, y_pred_train))

    # Confusion matrix (Train/Dev)
    conf_mat_train = confusion_matrix(y_train, y_pred_train)
    tn, fp, fn, tp = conf_mat_train.ravel()
    specificity_train = tn / (tn + fp)
    sensitivity_train = tp / (tp + fn)
    file.write("\nspecificity\t\t" + str(round(specificity_train,2)))
    file.write("\nsensitivity\t\t" + str(round(sensitivity_train,2)) + "\n\n")
    df_cm_train = pd.DataFrame(conf_mat_train, index=['no AF recurrence', 'AF recurrence'], columns=['no AF recurrence', 'AF recurrence'])

    # Classification report (Test)
    file.write("\nClassification Report (Test):\n" + classification_report(y_test, y_pred_test))

    # Confusion matrix (Test)
    conf_mat_test = confusion_matrix(y_test, y_pred_test)
    tn, fp, fn, tp = conf_mat_test.ravel()
    specificity_test = tn / (tn + fp)
    sensitivity_test = tp / (tp + fn)
    file.write("\nspecificity\t\t" + str(round(specificity_test,2)))
    file.write("\nsensitivity\t\t" + str(round(sensitivity_test,2)) + "\n")
    df_cm_test = pd.DataFrame(conf_mat_test, index=['no AF recurrence', 'AF recurrence'], columns=['no AF recurrence', 'AF recurrence'])
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
    plt.savefig(os.path.join(output_dir, "ModelEvaluation", "ConfusionMatrix_" + file_name + ".png"))
    # plt.figure(figsize=(6, 10))
    plt.close()
    print('Confusion matrixes saved in: ' + os.path.join(output_dir, "ModelEvaluation", "ConfusionMatrix.png"))

    #ROC curve (Train/Dev)
    cv = StratifiedKFold(n_splits=10)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()
    svm_roc = SVC(probability=True, C=1, gamma=0.1, kernel='poly', degree=2)
    for i, (train, test) in enumerate(cv.split(X_train, y_train)):
        svm_roc.fit(X_train.iloc[train], y_train.iloc[train])
        viz = plot_roc_curve(svm_roc, X_train.iloc[test], y_train.iloc[test], name='ROC fold {}'.format(i), alpha=0.3, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="ROC curve (Train/Dev), " + file_name)
    ax.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, "ModelEvaluation", "ROCcurve(Train_Dev).png"))
    print('Validation ROC curve and AUC saved in: ' + os.path.join(output_dir, "ModelEvaluation", "ROCcurve(Train_Dev).png"))
    plt.close()

    # ROC curve and AUC (Test)
    # roc curve for models
    y_pred_prob = svm.predict_proba(X_test)
    fpr, tpr, thresh = roc_curve(y_test, y_pred_prob[:, 1], pos_label=1)
    auc_metric = auc(fpr, tpr)
    # roc curve for tpr = fpr
    random_probs = [0 for i in range(len(y_test))]
    p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)
    # plot roc curves
    plt.plot(fpr, tpr, color='red',
             label='SVM (AUC = %0.2f)' % auc_metric)
    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
    # title
    plt.title('ROC curve (Test), ' + file_name)
    # x label
    plt.xlabel('False Positive Rate')
    # y label
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.savefig(os.path.join(output_dir, "ModelEvaluation", "ROCcurve(Test).png"))
    print('Test ROC curve and AUC saved in: ' + os.path.join(output_dir, "ModelEvaluation", "ROCcurve(Test).png"))
    plt.close()

    # Test predictions probabilities
    y_pred = svm.predict(X_test)
    y_pred_prob = svm.predict_proba(X_test)
    i = 0
    results = []
    for index in y_test.index:
        real_class = "no AF recurrence"
        pred_class = "no AF recurrence"
        success = "YES"
        if(y_test[index] == 1.0):
            real_class = "AF recurrence"
        if(y_pred[i] == 1.0):
            pred_class = "AF recurrence"
            prob = str(round((y_pred_prob[i][1] * 100), 2)) + "%"
        else:
            prob = str(round((y_pred_prob[i][0] * 100), 2)) + "%"
        if(y_test[index] != y_pred[i] == 1.0):
            success = "NO"

        results.append(["Instance: " + str(index), "Class: " + str(y_test[index]) + "(" + real_class + ")", "Predicted: " + str(y_pred[i]) + "(" + pred_class + ")", "Probability: " + prob, "Success: " + success])
        i += 1

    file.write("\n\nPredictions on Test set:\n")
    file.write(tabulate(results))

    print('Model evaluation report saved in: ' + os.path.join(output_dir, "ModelEvaluation", "REPORT_SVM_" + file_name + ".txt"))

    # Feature importances using SVM with linear kernel
    print('Getting model feature importance when trained using a SVM with linear kernel...')
    svm_linear = SVC(probability=True, C=1, gamma=0.1, kernel='linear', degree=2)
    svm_linear.fit(X_train, y_train)
    coef = svm_linear.coef_.ravel()
    coefficients = np.argsort(coef)
    # create plot
    plt.figure(figsize=(15, 10))
    colors = ['red' if c < 0 else 'blue' for c in coef[coefficients]]
    plt.bar(np.arange(len(coefficients)), coef[coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(len(coefficients)), feature_names[coefficients], rotation=60, ha='right')
    labels = ['no AF recurrence', 'AF recurrence']
    colors = {'no AF recurrence': 'red', 'AF recurrence': 'blue'}
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
    plt.legend(handles, labels)
    plt.title('Feature importances on linear SVM')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ModelEvaluation", "FeatureImportances_linearSVM.png"))
    print('Feature importances on linear SVM saved in: ' + os.path.join(output_dir, "ModelEvaluation", "FeatureImportances_linearSVM.png"))
    plt.close()

    # Feature absolute importances using SVM with linear kernel
    imp = abs(coef)
    imp, names = zip(*sorted(zip(imp, feature_names)))
    plt.figure(figsize=(10, 7))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.title('Feature absolute importances on linear SVM')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ModelEvaluation", "FeatureAbsoluteImportances_linearSVM.png"))
    print('Feature absolute importances on linear SVM saved in: ' + os.path.join(output_dir, "ModelEvaluation", "FeatureAbsoluteImportances_linearSVM.png"))
    plt.close()


def final_svm_model(dataset):
    # Model Training --> SVM
    y = dataset["sigue_fa"]
    X = dataset.drop("sigue_fa", 1)

    # Model
    svm = SVC(probability=True, C=1, gamma=0.1, kernel='poly', degree=2)
    svm.fit(X, y)

    # Save the model
    joblib.dump(svm, os.path.join(output_dir, "FinalModel", "SVM_FINAL_MODEL.joblib"), compress=3)
    print('Trained final model saved in: ' + os.path.join(output_dir, "FinalModel", "SVM_FINAL_MODEL.joblib"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to achieve the optimal AF recurrence predictive model on PRAFAI dataset.\nFollowing preprocessing techniques are appied to training set: elimination of features with at least 85% of missing values, imputation of missing values by median in numeric features, standardization of numeric features by StandardScaler and feature selection by RFE method. The predictive model is trained using SVM algorithm with a second-degree polynomial kernel. Finally, model evaluation is carried out on validation and test sets, as well as predictions made by the model on test items never seen before. A final model merging Train and Test sets is trained and saved for new predictions.')
    parser.add_argument("input_dataset", help="Path to PRAFAI dataset.")
    parser.add_argument("-o", "--output_dir", help="Path to the output directory for creating following files: middle-procces training and test sets, feature selection report, predictive model, evaluation report and feature importances, and final model for new predictions.", default=os.getcwd())
    args = vars(parser.parse_args())
    input_dataset = args['input_dataset']
    output_dir = args['output_dir']

    Path(os.path.join(output_dir, "Preprocess")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_dir, "ModelEvaluation")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_dir, "FinalModel")).mkdir(parents=True, exist_ok=True)

    # Read dataset
    print('\nReading PRAFAI dataset from: ' + str(input_dataset))
    dataset = pd.read_csv(input_dataset, delimiter=";")

    # Train/Test split
    print('\nCreating Train/Test subsets...')
    train, test, dataset_train = class_and_split(dataset)
    print('Saving ' + str(os.path.join(output_dir, "Preprocess", "train.csv ...")))
    train.to_csv(os.path.join(output_dir, "Preprocess", "train.csv"), index=False, sep=';')
    print('Saving ' + str(os.path.join(output_dir, "Preprocess", "test.csv ...")))
    test.to_csv(os.path.join(output_dir, "Preprocess", "test.csv"), index=False, sep=';')

    # Missing values imputation by median
    print("\nImputing median to missing values in 'train' and 'test' sets...")
    train_median, test_median = impute_median_85_mv(train, test, dataset_train)
    print('Saving ' + str(os.path.join(output_dir, "Preprocess", "train_med.csv ...")))
    train_median.to_csv(os.path.join(output_dir, "Preprocess", "train_med.csv"), index=False, sep=';')
    print('Saving ' + str(os.path.join(output_dir, "Preprocess", "test_med.csv ...")))
    test_median.to_csv(os.path.join(output_dir, "Preprocess", "test_med.csv"), index=False, sep=';')

    # Standardization by StandardScaler
    print("\nPreprocessing 'train_med' and 'test_med' sets with StandardScaler...")
    train_StandardScaler, test_StandardScaler = standardization(train_median, test_median)
    print('Saving ' + str(os.path.join(output_dir, "Preprocess", "train_med_StandardScaler.csv")))
    train_StandardScaler.to_csv(os.path.join(output_dir, "Preprocess", "train_med_StandardScaler.csv"), index=False, sep=';')
    print('Saving ' + str(os.path.join(output_dir, "Preprocess", "test_med_StandardScaler.csv")))
    test_StandardScaler.to_csv(os.path.join(output_dir, "Preprocess", "test_med_StandardScaler.csv"), index=False, sep=';')

    # Feature selection by RFE
    print("\nPerforming RFE feature selection on 'train_med_StandardScaler' and 'test_med_StandardScaler' sets...")
    print('Saving RFE feature selection report...')
    train_RFE, test_RFE = rfe(train_StandardScaler, test_StandardScaler, 'train_med_StandardScaler')
    print('Saving ' + str(os.path.join(output_dir, "Preprocess", 'train_med_StandardScaler_RFE.csv')))
    train_RFE.to_csv(os.path.join(output_dir, "Preprocess", "train_med_StandardScaler_RFE.csv"), index=False, sep=';')
    print('Saving ' + str(os.path.join(output_dir, "Preprocess", "test_med_StandardScaler_RFE.csv")))
    test_RFE.to_csv(os.path.join(output_dir, "Preprocess", "test_med_StandardScaler_RFE.csv"), index=False, sep=';')

    # Model Evaluation
    print("\nTraining SVM model with 'train_med_StandardScaler_RFE'...")
    svm_model_evaluation(train_RFE, test_RFE, 'train_med_StandardScaler_RFE')

    # Final Model (Train + Test)
    print("\nTraining and saving SVM final model (TRAIN + TEST)...")
    dataset = pd.concat([train_RFE, test_RFE])
    final_svm_model(dataset)




