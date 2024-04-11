from sklearn.model_selection import train_test_split
import pandas as pd


class DataUtils:
    @staticmethod
    def get_variant_data(data_csv) -> pd.DataFrame:
        data = pd.read_csv('dataset/'+data_csv)
        return data

    @staticmethod
    def get_text_data(data_csv) -> pd.DataFrame:
        data = pd.read_csv('dataset/'+data_csv, sep="\|\|", engine="python", names=[
                           "ID", "TEXT"], skiprows=1)
        return data

    @staticmethod
    def merge_data(df_variants, df_text) -> pd.DataFrame:
        data = pd.merge(df_variants, df_text, on='ID', how='left')
        return data


class DatasetDevelopment:
    def __init__(self, df):
        self.df = df

    def divide_your_data(self,test_size):
        print("Dividing the data:- ")

        X = self.df.drop(["Class"], axis=1)
        y = self.df[["Class"]].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=test_size, random_state=1
        )
        return X_train, X_test, y_train, y_test
