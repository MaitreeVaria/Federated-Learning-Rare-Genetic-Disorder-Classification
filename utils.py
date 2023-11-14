import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow import keras
import keras.utils
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

def preprocess(df_train):
    df = df_train.copy()

    le = LabelEncoder()
    cat_data = df[["Genes in mother's side",'Inherited from father','Maternal gene','Paternal gene','Status',
                'Respiratory Rate (breaths/min)','Heart Rate (rates/min','Follow-up','Gender',
                'Folic acid details (peri-conceptional)','H/O serious maternal illness','Assisted conception IVF/ART',
                'History of anomalies in previous pregnancies','Birth defects','Blood test result',
                'Symptom 1','Symptom 2','Symptom 3','Symptom 4','Symptom 5', 'Disorder Subclass', 'Genetic Disorder']]
    num_data = df[['Patient Age','Blood cell count (mcL)',"Mother's age","Father's age",'No. of previous abortion',
                'White Blood cell count (thousand per microliter)']]
    for i in cat_data:
        cat_data[i] = le.fit_transform(cat_data[i])
    df_encoded = pd.concat([num_data, cat_data], axis=1)
    df_max = df_encoded.iloc[:,0:-2].max()
    df_encoded.iloc[:,0:-2] = df_encoded.iloc[:,0:-2].divide(df_max)
    df_subclass = df_encoded.drop(columns=['Genetic Disorder'])
    x_subclass = df_subclass.iloc[:,0:-1]
    y_subclass = df_subclass.iloc[:,-1]
    over = SMOTE()
    x_subclass, y_subclass = over.fit_resample(x_subclass, y_subclass)
    y_subclass = le.fit_transform(y_subclass)
    y_subclass = tf.keras.utils.to_categorical(y_subclass)
    # X_train, X_test, y_train, y_test = train_test_split(x_subclass, y_subclass)
    return x_subclass, y_subclass