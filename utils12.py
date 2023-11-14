from typing import Tuple, Union, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]


def get_model_parameters(model: LogisticRegression) -> LogRegParams:
    """Returns the paramters of a sklearn LogisticRegression model."""
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [
            model.coef_,
        ]
    return params


def set_model_params(
    model: LogisticRegression, params: LogRegParams
) -> LogisticRegression:
    """Sets the parameters of a sklean LogisticRegression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model: LogisticRegression):
    """Sets initial parameters as zeros Required since model params are uninitialized
    until model.fit is called.

    But server asks for initial parameters from clients at launch. Refer to
    sklearn.linear_model.LogisticRegression documentation for more information.
    """
    n_classes = 9
    n_features = 39
    model.classes_ = np.array([i for i in range(n_classes)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))


def shuffle(X: np.ndarray, y: np.ndarray) -> XY:
    """Shuffle X and y."""
    rng = np.random.default_rng()
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
    )


def preprocess(df_train):
    df_train.drop("Patient Id",axis=1,inplace=True)
    df_train.drop("Family Name",axis=1,inplace=True)
    df_train.drop("Patient First Name",axis=1,inplace=True)
    df_train.drop("Father's name",axis=1,inplace=True)
    df_train.drop("Institute Name",axis=1,inplace=True)
    df_train.drop("Location of Institute",axis=1,inplace=True)
    df_train.drop("Place of birth",axis=1,inplace=True)


    df_train["Patient Age"].fillna(str(df_train["Patient Age"].mode().values[0]),inplace=True)
    df_train["Inherited from father"].fillna(str(df_train["Inherited from father"].mode().values[0]),inplace=True)
    df_train["Maternal gene"].fillna(str(df_train["Maternal gene"].mode().values[0]),inplace=True)
    df_train["Mother's age"].fillna(str(df_train["Mother's age"].mode().values[0]),inplace=True)
    df_train["Father's age"].fillna(str(df_train["Father's age"].mode().values[0]),inplace=True)
    df_train["Respiratory Rate (breaths/min)"].fillna(str(df_train["Respiratory Rate (breaths/min)"].mode().values[0]),inplace=True)
    df_train["Heart Rate (rates/min"].fillna(str(df_train["Heart Rate (rates/min"].mode().values[0]),inplace=True)
    df_train["Test 1"].fillna(str(df_train["Test 1"].mode().values[0]),inplace=True)
    df_train["Test 2"].fillna(str(df_train["Test 2"].mode().values[0]),inplace=True)
    df_train["Test 3"].fillna(str(df_train["Test 3"].mode().values[0]),inplace=True)
    df_train["Test 4"].fillna(str(df_train["Test 4"].mode().values[0]),inplace=True)
    df_train["Test 5"].fillna(str(df_train["Test 5"].mode().values[0]),inplace=True)
    df_train["Parental consent"].fillna(str(df_train["Parental consent"].mode().values[0]),inplace=True)
    df_train["Follow-up"].fillna(str(df_train["Follow-up"].mode().values[0]),inplace=True)
    df_train["Gender"].fillna(str(df_train["Gender"].mode().values[0]),inplace=True)
    df_train["Birth asphyxia"].fillna(str(df_train["Birth asphyxia"].mode().values[0]),inplace=True)
    df_train["Autopsy shows birth defect (if applicable)"].fillna(str(df_train["Autopsy shows birth defect (if applicable)"].mode().values[0]),inplace=True)
    df_train["Folic acid details (peri-conceptional)"].fillna(str(df_train["Folic acid details (peri-conceptional)"].mode().values[0]),inplace=True)
    df_train["H/O serious maternal illness"].fillna(str(df_train["H/O serious maternal illness"].mode().values[0]),inplace=True)
    df_train["H/O radiation exposure (x-ray)"].fillna(str(df_train["H/O radiation exposure (x-ray)"].mode().values[0]),inplace=True)
    df_train["H/O substance abuse"].fillna(str(df_train["H/O substance abuse"].mode().values[0]),inplace=True)
    df_train["Assisted conception IVF/ART"].fillna(str(df_train["Assisted conception IVF/ART"].mode().values[0]),inplace=True)
    df_train["History of anomalies in previous pregnancies"].fillna(str(df_train["History of anomalies in previous pregnancies"].mode().values[0]),inplace=True)
    df_train["No. of previous abortion"].fillna(str(df_train["No. of previous abortion"].mode().values[0]),inplace=True)
    df_train["Birth defects"].fillna(str(df_train["Birth defects"].mode().values[0]),inplace=True)
    df_train["White Blood cell count (thousand per microliter)"].fillna(str(df_train["White Blood cell count (thousand per microliter)"].mode().values[0]),inplace=True)
    df_train["Blood test result"].fillna(str(df_train["Blood test result"].mode().values[0]),inplace=True)
    df_train["Symptom 1"].fillna(str(df_train["Symptom 1"].mode().values[0]),inplace=True)
    df_train["Symptom 2"].fillna(str(df_train["Symptom 2"].mode().values[0]),inplace=True)
    df_train["Symptom 3"].fillna(str(df_train["Symptom 3"].mode().values[0]),inplace=True)
    df_train["Symptom 4"].fillna(str(df_train["Symptom 4"].mode().values[0]),inplace=True)
    df_train["Symptom 5"].fillna(str(df_train["Symptom 5"].mode().values[0]),inplace=True)
    df_train["Genetic Disorder"].fillna(str(df_train["Genetic Disorder"].mode().values[0]),inplace=True)
    df_train["Disorder Subclass"].fillna(str(df_train["Disorder Subclass"].mode().values[0]),inplace=True)

    df_train["Genes in mother's side"]=[1 if i.strip()== "Yes" else 0 for i in df_train["Genes in mother's side"]]
    df_train["Inherited from father"]=[1 if i.strip()== "Yes" else 0 for i in df_train["Inherited from father"]]
    df_train["Maternal gene"]=[1 if i.strip()== "Yes" else 0 for i in df_train["Maternal gene"]]
    df_train["Paternal gene"]=[1 if i.strip()== "Yes" else 0 for i in df_train["Paternal gene"]]
    df_train["Parental consent"]=[1 if i.strip()== "Yes" else 0 for i in df_train["Parental consent"]]
    df_train["Birth asphyxia"]=[1 if i.strip()== "Yes" else 0 for i in df_train["Birth asphyxia"]]
    df_train["Folic acid details (peri-conceptional)"]=[1 if i.strip()== "Yes" else 0 for i in df_train["Folic acid details (peri-conceptional)"]]
    df_train["H/O radiation exposure (x-ray)"]=[1 if i.strip()== "Yes" else 0 for i in df_train["H/O radiation exposure (x-ray)"]]
    df_train["H/O substance abuse"]=[1 if i.strip()== "Yes" else 0 for i in df_train["H/O substance abuse"]]
    df_train["Assisted conception IVF/ART"]=[1 if i.strip()== "Yes" else 0 for i in df_train["Assisted conception IVF/ART"]]
    df_train["History of anomalies in previous pregnancies"]=[1 if i.strip()== "Yes" else 0 for i in df_train["History of anomalies in previous pregnancies"]]
    df_train["H/O serious maternal illness"]=[1 if i.strip()=="Yes" else 0 for i in df_train["H/O serious maternal illness"]]

    #Alive':1 'Deceased:0'
    df_train["Status"]=[1 if i.strip()== "Alive" else 0 for i in df_train["Status"]]
    #Normal (30-60):1' 'Tachypnea:0
    df_train["Respiratory Rate (breaths/min)"]=[1 if i.strip()== "Normal (30-60)" else 0 for i in df_train["Respiratory Rate (breaths/min)"]]
    #Normal:1' 'Tachycardia:0
    df_train["Heart Rate (rates/min"]=[1 if i.strip()== "Normal" else 0 for i in df_train["Heart Rate (rates/min"]]
    #High:1, Low:0
    df_train["Follow-up"]=[1 if i.strip()== "High" else 0 for i in df_train["Follow-up"]]
    #['Singular' 'Multiple']
    df_train["Birth defects"]=[1 if i.strip()== "Singular" else 0 for i in df_train["Birth defects"]]

    #1: male 0: female 2: ambiguous
    df_train["Gender"]=[1 if i.strip()== "Male" else 0 if i.strip() == "Female" else 2 for i in df_train["Gender"]]

    #Not applicable:3' 'None:2' 'No:0' 'Yes:1'
    df_train["Autopsy shows birth defect (if applicable)"]=[1 if i.strip()== "Yes" else 0 if i.strip() == "No" else 2 if i.strip()=="None" else 3 for i in df_train["Autopsy shows birth defect (if applicable)"]]

    #'slightly abnormal':1, 'normal':0, 'inconclusive':2 'abnormal:3']
    df_train["Blood test result"]=[1 if i.strip()== "slightly abnormal" else 0 if i.strip() == "normal" else 2 if i.strip()=="inconclusive" else 3 for i in df_train["Blood test result"]]

    #'Mitochondrial genetic inheritance disorders':1,'Multifactorial genetic inheritance disorders':0'Single-gene inheritance diseases:2'
    df_train["Genetic Disorder"]=[1 if i.strip()== "Mitochondrial genetic inheritance disorders" else 0 if i.strip() == "Multifactorial genetic inheritance disorders" else 2 for i in df_train["Genetic Disorder"]]

    df_train["Disorder Subclass"]=[1 if i.strip()== "Leber's hereditary optic neuropathy"
                                    else 0 if i.strip() == "Cystic fibrosis"
                                else 2 if i.strip()=="Diabetes"
                                else 3 if i.strip()=="Leigh syndrome"
                                else 4 if i.strip()=="Cancer"
                                else 5 if i.strip()=="Tay-Sachs"
                                else 6 if i.strip()=="Hemochromatosis"
                                else 7 if i.strip()=="Mitochondrial myopathy"
                                else 8 for i in df_train["Disorder Subclass"]]



    df_train = df_train.apply(pd.to_numeric,downcast="float")

    df_train["sum of Mother's and fathers age avg"]=(df_train["Mother's age"]+df_train["Father's age"]) / 2

    #total symptom
    df_train["total symptom"]=(df_train["Symptom 1"]+df_train["Symptom 2"]+df_train["Symptom 3"]+df_train["Symptom 4"]+df_train["Symptom 5"]) / 5


    """**Splitting data into Training and Splitting**"""
    # df_train.drop("Genetic Disorder",axis=1,inplace=True)
    X,y = df_train.loc[:,df_train.columns != 'Disorder Subclass'], df_train.loc[:,'Disorder Subclass']
    return X, y