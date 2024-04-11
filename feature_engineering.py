import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import joblib
from numpy import asarray
import numpy as np
from sklearn.decomposition import TruncatedSVD
from imblearn.over_sampling import SMOTE


class FeatureEngineering:

    def __init__(self, df):
        self.df = df

    def process(self, x):
        text = nltk.word_tokenize(x)
        return nltk.pos_tag(text)

    def findtags(self, tagged_text):
        cfd = nltk.ConditionalFreqDist((tag, word) for (
            word, tag) in tagged_text if tag == "NN")
        return len(list(cfd['NN'].keys()))

    def add_more_features(self):
        df_copy = self.df.copy()

        df_copy["Overall_text_length"] = df_copy["Processed_text"].apply(
            lambda x: len(x))
        df_copy["Number_of_sentences"] = df_copy["TEXT"].apply(
            lambda x: len(nltk.sent_tokenize(x))
        )
        df_copy["Number_of_words"] = df_copy["Processed_text"].apply(
            lambda x: len(nltk.word_tokenize(x))
        )
        df_copy["Number_of_unique_words"] = df_copy["Processed_text"].apply(
            lambda x: len(set(nltk.word_tokenize(x)))
        )
        df_copy["Number_of_characters"] = df_copy["Processed_text"].apply(
            lambda x: len(x)
        )
        df_copy["Number_of_characters_per_word"] = (
            df_copy["Number_of_characters"] / df_copy["Number_of_words"]
        )
        # df_copy["Number_of_words_containing_numbers"] = df_copy[
        #     "Processed_text"
        # ].apply(
        #     lambda x: len(
        #         [w for w in nltk.word_tokenize(x) if any(
        #             char.isdigit() for char in w)]
        #     )
        # )
        df_copy['pos'] = df_copy.Processed_text.apply(
            lambda x: self.process(x))
        df_copy['no. of nouns'] = df_copy["pos"].apply(
            lambda x: self.findtags(x))
        df_copy.drop(["pos"], axis=1)
        return df_copy

    def vectorize_features(self):
        # ====================================================
        vectorizer = TfidfVectorizer()
        bag_of_words = vectorizer.fit_transform(
            self.df["Processed_text"]).toarray()

        #vocab = vectorizer.vocabulary_
        #mapping = vectorizer.get_feature_names()
        #keys = list(vocab.keys())

        # save the vectorizer to disk
        joblib.dump(vectorizer, "vectorizer.pkl")
        return bag_of_words

    def extract_features_temp(self, train):
        # ====================================================
        vectorizer = TfidfVectorizer()
        bag_of_words = vectorizer.fit_transform(
            train["Processed_text"] + train.Variation + train.Gene).toarray()

        #vocab = vectorizer.vocabulary_
        #mapping = vectorizer.get_feature_names()
        #keys = list(vocab.keys())

        # save the vectorizer to disk
        # joblib.dump(vectorizer, "vectorizer.pkl")
        return bag_of_words

    def convert_cat_feat(self):
        count_vector = CountVectorizer()
        bag_of_cat_feat = count_vector.fit_transform(
            self.df.Variation+" "+self.df.Gene).toarray()
        joblib.dump(count_vector, 'count_vector.pkl')  # save the model

        return bag_of_cat_feat

    def reduce_features(self, df):
        svd = TruncatedSVD(20)
        truncated_bag_of_words = svd.fit_transform(df)
        joblib.dump(svd, 'svd.joblib')

        return truncated_bag_of_words

    def imbalanced_handling(self, x1, y1):
        oversample = SMOTE()
        X1, Y1 = oversample.fit_resample(x1, y1)

        return X1, Y1
