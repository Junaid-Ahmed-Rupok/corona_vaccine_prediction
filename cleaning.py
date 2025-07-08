import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder


def preprocessing(read_csv):
    df = pd.read_csv(read_csv)  # converting the csv file into a dataframe

    for cols in df.columns:
        if df[cols].isna().sum() > df.shape[0] * 0.6:
            df = df.drop(columns=[cols])  # dropping the unnecessary columns

    df = df.drop(columns=['Date_Reported'])

    if df.isna().sum().sum() > 0:
        df = df.apply(
            lambda cols: cols.fillna(cols.mean()) if np.issubdtype(cols.dtype, np.number)
            else cols.fillna(cols.mode()[0])
        )  # dealing with null values

    if df.duplicated().sum():
        df = df.drop_duplicates()  # dealing with duplicates

    # 'Lab_Results'
    # ['AstraZeneca' 'Sinopharm' 'Sinovac' 'Pfizer' 'Moderna']
    df['Vaccine_Type'] = df['Vaccine_Type'].apply(
        lambda x: 0 if x == 'AstraZeneca' else 1 if x == 'Sinopharm' else 2 if x == 'Sinovac'
        else 3 if x == 'Pfizer' else 4
    )  # encoding the categorical label

    encoder = OrdinalEncoder()
    df_cats = df.select_dtypes(include=['object']).columns.tolist()  # selecting the categorical features
    df[df_cats] = encoder.fit_transform(df[df_cats])  # encoding the features

    joblib.dump(encoder, 'encoder.pkl')  # dumping the encoder

    return df


cleaned_dataframe = preprocessing(r"C:\Users\JUNAID AHMED\Hello\DATASETS\unbiased_covid_dataset.csv")
joblib.dump(cleaned_dataframe, 'cleaned_df.pkl')

