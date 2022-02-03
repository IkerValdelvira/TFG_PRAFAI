# TFG-PRAFAI

Software developed to carry out the End-of-Degree Project ***PRAFAI (Prediction of Recurrence of Atrial Fibrillation using Artificial Intelligence)***. Here you will find all the scripts developed during this research work with the following objectives: the creation of a dataset joining the necessary information of each patient of the involved HMO (Health Maintenance Organization) hospital with a first atrial fibrillation episode beetwen 2015 and 2018, and Artificial Intelligence and Machine Learning tasks for developing the predictive tool of AF recurrence like data pre-processing, choice of supervised learning algorithms, implementation and generation of predictive models, results evaluation and most important predictive variables analysis.

The software used in the secondary objective to develop a predictive model of AF recurrence using 12-lead ECGs is also available. More specifically, there is the software used in experiments to classify ECGs with AF or normal sinus rhythm on the [*ECG recording from Chapman University and Shaoxing People’s Hospital*](https://www.nature.com/articles/s41597-020-0386-x) database.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Software description and usage](#software-description-and-usage)
<br />2.1. [PRAFAI package scripts](#prafai-package-scripts)
<br />2.2. [PRAFAI_ECG package scripts](#prafai_ecg-package-scripts)
3. [TUTORIAL: Getting PRAFAI predictive model and making predictions on new items](#tutorial-getting-prafai-predictive-model-and-making-predictions-on-new-items)
4. [Project documentation](#project-documentation)
5. [Author and contact](#author-and-contact)

## Prerequisites:

It is recommended to use the application in an environment with **Python 3.8** and the installation of the following packages is required:

* python-dateutil==2.8.2
* scikit-learn==0.24.1
* joblib==1.1.0
* keras==2.7.0
* lightgbm==3.3.0
* matplotlib==3.4.3
* numpy==1.21.3
* pandas==1.3.4
* seaborn==0.11.2
* tabulate==0.8.9
* tensorflow==2.7.0
* tensorflow_addons==0.15.0
* xgboost==1.5.0

To install the necessary packages with the corresponding versions automatically, execute the following command:

```
$ pip install -r requirements.txt
```

## Software description and usage:

There is a help option that shows the usage of each script in the application, e.g. in *dataset_creation.py* script:

```
$ python PRAFAI/dataset_creation.py -h
```
  
### PRAFAI package scripts:

* ***dataset_creation.py***: Script to create the PRAFAI dataset. Input folder with necessary files is needed.

* ***dataset_analysis.py***: Script to get distribution histograms of each feature with in regard to the class.

* ***split_mvimputation.py***: Script to divide the dataset into Train/Test subsets (80%/20%) and imputation of missing values in numeric features. Only labeled items will be taken. Different techniques are available for the imputation of missing values: arithmetic mean, median, prediction by linear regression or different value.

* ***standardization_normalization.py***: Script to standardize/normalize numeric features of the training set (Train) and apply the same rescaling to the testing set (Test). Different techniques are available for standardization/normalization: StandardScaler, MinMaxScaler, MaxAbsScaler, QuantileTransformer or Normalizer.

* ***preprocessing.py***: Script to divide the dataset into Train/Test subsets (80%/20%), imputate missing values in numeric features and standardize/normalize numeric features of the training set (Train), and apply the same imputation and rescaling to the testing set (Test). Only labeled items will be taken. Different techniques are available for the imputation of missing values: arithmetic mean, median, prediction by linear regression or different value. Different techniques are available for standardization/normalization: StandardScaler, MinMaxScaler, MaxAbsScaler, QuantileTransformer or Normalizer.

* ***feature_selection.py***: Script to perform feature selection or dimensionality reduction on training set and apply it to the test set. Different techniques are available for feature selection: RFE, Lasso_SelectFromModel, Trees_SelectFromModel, SFS and SelectKBest. The available technique for dimensionality reduction is PCA.

* ***feature_importance.py***: Script to get feature importance of training set. Different techniques and classifiers are available for getting feature importance: LogisticRegression, CART, RandomForest, XGBoost, Permutation_RandomForest and Permutation_KNN. Heatmaps of feature correlation matrices are also available.

* ***classifiers.py***: Script to train a model and evaluate it on validation and test sets. Different machine learning algorithms are available for model training: DecisionTree, LogisticRegression, KNN, NaiveBayes, Perceptron, MultilayerPerceptron, SVM, RandomForest, XGBoost, LightGBM, Bagging and AdaBoost.

* ***best_model.py***: Script to achieve the optimal AF recurrence predictive model on PRAFAI dataset. Following preprocessing techniques are appied to training set: elimination of features with at least 85% of missing values, imputation of missing values by median in numeric features, standardization of numeric features by StandardScaler and feature selection by RFE method. The predictive model is trained using SVM algorithm with a second-degree polynomial kernel. Finally, model evaluation is carried out on validation and test sets, as well as predictions made by the model on test items never seen before. A final model merging Train and Test sets is trained and saved for new predictions.

* ***make_predictions.py***: Script to make predictions on new input items using PRAFAI predictive model.


### PRAFAI_ECG package scripts:

* ***musexmlex.py***: Script to extract an 12-lead ECG rhythm strip from a MUSE(R) XML file. It converts MUSE-XML files to CSV files. Credits to [***PROJECT: musexmlexport***](https://github.com/rickead/musexmlexport).

* ***train_models.py***: Script to train and evaluate different AF classification models based on 12-lead ECGs: XGBoost, FCN, FCN+MLP(age,sex), Encoder, Encoder+MLP(age,sex), FCN+Encoder, FCN+Encoder+MLP(age,sex) or LSTM.

* ***train_FCN_MLP_CV.py***: Script to train FCN+MLP(age,sex) AF classification model based on 12-lead ECGs and evaluate it via 10-fold Cross Validation.

* ***reductionFCN_MLP.py***: Script in which the AF classification model development experiment is performed by reducing the number of training ECGs. FCN+MLP(age,sex) classification algorithm is used.


## TUTORIAL: Getting PRAFAI predictive model and making predictions on new items:

This section explains how to obtain the final predictive model and make predictions on new data.

**1. CREATE PRAFAI DATASET**

```
$ python PRAFAI/dataset_creation.py INPUT_DIR -o OUTPUT_DIR
```

**2. TRAIN AND GET PRAFAI PREDICTIVE MODEL**

```
$ python PRAFAI/best_model.py PATH_TO/dataset.csv -o OUTPUT_DIR
```
A folder called ***FinalModel*** will be created, among others, which contains the model and necessary files to make new predictions.


**3. MAKE PREDICTIONS ON NEW ITEMS**

To make a prediction on a new item, the PRAFAI model needs the values of the following 30 features:

* Numeric features:<br />**'potasio'**, **'no_hdl'**, **'colesterol'**, **'ntprobnp'**, **'vsg'**, **'fevi'**, **'diametro_ai'**, **'area_ai'**, **'numero_dias_desde_ingreso_hasta_evento'**, **'numero_dias_ingresado'** and **'edad'**.

* Binary features:<br />**'ablacion'**, **'ansiedad'**, **'demencia'**, **'sahos'**, **'hipertiroidismo'**, **'cardiopatia_isquemica'**, **'valvula_mitral_reumaticas'**, **'genero'**, **'pensionista'**, **'residenciado'**, **'n05a'**, **'n05b'**, **'c01'**, **'c01b'**, **'c02'**, **'c04'**, **'c09'**, **'c10'** and **'polimedicacion'**.

\* In **'genero'** (genre) feature **female** is set as **0** and **male** is set as **1**.
<br />\* The new items that are introduced can have missing values in any of the features. These missing values will be automatically handled to make the predictions, but the fewer missing values there are, the more reliable the predictions will be.

The new items to be predicted must be introduced in a **CSV file** delimited by comma (**,**). The first column (index) must be the ID of the new item. A template and example of this CSV file is available at: [new_items_template.csv](https://github.com/IkerValdelvira/TFG_PRAFAI/blob/master/templates/new_items_template.csv) and [new_items_example.csv](https://github.com/IkerValdelvira/TFG_PRAFAI/blob/master/templates/new_items_example.csv).

Following image shows the structure of a CSV file with some new items to be predicted ([new_items_example.csv](https://github.com/IkerValdelvira/TFG_PRAFAI/blob/master/templates/new_items_example.csv)):

![alt text](https://github.com/IkerValdelvira/TFG_PRAFAI/blob/master/example_images/new_items_example.png?raw=true)

To make the predictions, the following script must be executed:
```
$ python PRAFAI/make_predictions.py PATH_TO/new_items_example.csv PATH_TO/FinalModel -o OUTPUT_DIR
```

A file called ***PREDICTIONS.txt*** will be created, which contains the predictions made by the model on new input items. In this file appears the ID (index) of each new item introduced together with the outcome of the model (prediction) and its probability. Following image shows the output *PREDICTIONS.txt* file after having introduced [new_items_example.csv](https://github.com/IkerValdelvira/TFG_PRAFAI/blob/master/templates/new_items_example.csv):

![alt text](https://github.com/IkerValdelvira/TFG_PRAFAI/blob/master/example_images/predictions_example.png?raw=true)


## Project documentation:

The documentation describing the work in this project can be found here: [TFG_PRAFAI_Memoria_IkerValdelvira.pdf](https://github.com/IkerValdelvira/TFG_PRAFAI/blob/master/documentation/TFG_PRAFAI_Memoria_IkerValdelvira.pdf)


## Author and contact:

Iker Valdelvira ([ivaldelvira001@ikasle.ehu.eus](mailto:ivaldelvira001@ikasle.ehu.eus))
