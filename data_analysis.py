import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from wordcloud import WordCloud
from nltk.corpus import stopwords
import seaborn as sns


class DataAnalysis:
    def __init__(self, df) -> None:
        self.df = df

    def explore_data(self, vis=False):
        print('Number of data points:', self.df.shape[0])
        print('Number of features:', self.df.shape[1])
        print('Features:', self.df.columns.values)
        print('Head:', self.df.head())
        print('Describe:', self.df.describe())
        print('Info:', self.df.info())
        if vis:
            self.explore_data_visualization()

    def check_null(self):
        print(self.df.isnull().sum())

    def fill_null_values(self):
        self.df.loc[self.df['TEXT'].isnull(), 'TEXT'] = self.df['Gene'] + \
            ' '+self.df['Variation']

    def explore_data_visualization(self):
        sns.countplot(self.df.Variation)

        unique_gene = self.df['Gene'].value_counts()
        s = sum(unique_gene.values)
        h = unique_gene.values/s
        plt.plot(h, label="Histogram of Genes")
        plt.xlabel("Gene")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()
