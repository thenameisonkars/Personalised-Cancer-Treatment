from run_prepare_data import DatasetDevelopment
from run_prepare_data import DataUtils
from data_analysis import DataAnalysis
from text_preprocessing import TextPreprocessor
from feature_engineering import FeatureEngineering
from model_training import ModelTraining
from evaluate import EvaluateModel
import joblib
import pandas as pd
import numpy as np


def main():
    training_text = DataUtils.get_text_data("training_text")
    # test_text = DataUtils.get_text_data("test_text")
    training_variants = DataUtils.get_variant_data("training_variants")
    # test_variants = DataUtils.get_variant_data("test_variants")

    training = DataUtils.merge_data(training_variants, training_text)
    # test = DataUtils.merge_data(test_variants, test_text)
    # test = test.dropna()
    training = training.dropna()

    tr = DatasetDevelopment(training)
    x_train, _, y_train, _ = tr.divide_your_data(.6)
    x_train["Class"] = y_train.flatten()

    for i in [x_train]:
        data_analysis = DataAnalysis(i)
        data_analysis.explore_data()
        data_analysis.explore_data_visualization()

    txtp = TextPreprocessor(x_train)
    df = txtp._preprocess_text()
    df.to_csv('df.csv')
    df.info()
    df = pd.read_csv("df.csv")

    feature_engineering = FeatureEngineering(df)
    df_new_added_features = feature_engineering.add_more_features()

    # df_new_added_features.info()
    # df_new_added_features.to_csv('df_new_added_features.csv')
    df_new_added_features = pd.read_csv("df_new_added_features.csv")

    df = pd.read_csv("df.csv")
    feature_engineering = FeatureEngineering(df)
    df_new_added_features = pd.read_csv("df_new_added_features.csv")

    data1 = feature_engineering.vectorize_features()
    data2 = feature_engineering.convert_cat_feat()
    data3 = feature_engineering.reduce_features(
        np.concatenate((data1, data2, df_new_added_features[['no. of nouns', 'Number_of_characters',
                                                            'Number_of_unique_words', 'Number_of_words', 'Number_of_sentences',
                                                             'Overall_text_length']]), axis=1)
    )

    x, y = feature_engineering.imbalanced_handling(
        data3, df_new_added_features.Class.values)

    # x = np.load('x.npy')
    # y = np.load('y.npy')
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25)
    # np.save('x.npy', x)
    # np.save('y.npy', y)

    model_train = ModelTraining(x_train, y_train)

    # ============================================ XGboost Model =========================================
    XGboostModel = model_train.Xgboost_model(fine_tuning=True)

    filename = 'xgb_model_finetuned.sav'  # 81
    filename = 'xgb_model_not_finetuned.sav'  # 79
    joblib.dump(XGboostModel, filename)
    XGboostModel = joblib.load(filename)
    y_pred = XGboostModel.predict(x_test)
    y_proba = XGboostModel.predict_proba(x_test)

    # check the unique values in predict
    evaluate = EvaluateModel(y_test, y_pred, y_proba, XGboostModel)
    evaluate.evaluate()  # 81 #
    evaluate.plot_confusion_matrix()
    evaluate.plot_roc_curve()
    # ================== end ====================== #
    LRModel = model_train.logistic_regression_model()

    # filename = 'lr_model.sav' # 29
    # # joblib.dump(LRModel, filename)
    # LRModel = joblib.load(filename)
    y_pred = LRModel.predict(x_test)
    y_proba = LRModel.predict_proba(x_test)

    # check the unique values in predict
    evaluate = EvaluateModel(y_test, y_pred, y_proba, LRModel)
    evaluate.evaluate()
    evaluate.plot_confusion_matrix()
    evaluate.plot_roc_curve()

# ====================================================================
    svm_model = model_train.svm_model(fine_tuning=True)

    filename = 'svm_model.sav'
    # joblib.dump(svm_model, filename)
    svm_model = joblib.load(filename)
    y_pred = svm_model.predict(x_test)
    y_proba = svm_model.predict_proba(x_test)  # p .79 a .39

    # check the unique values in predict
    evaluate = EvaluateModel(y_test, y_pred, y_proba, svm_model)
    evaluate.evaluate()
    evaluate.plot_confusion_matrix()
    evaluate.plot_roc_curve()
# =====================================================================
    random_forest_model = model_train.random_forest_model(finetuning=False)
    # random_forest_model.get_params()
    # filename = 'random_forest_model_fine.sav'# 73
    # filename = 'random_forest_model_nofine.sav' # 80
    # joblib.dump(random_forest_model, filename)
    # random_forest_model = joblib.load(filename)
    y_pred = random_forest_model.predict(x_test)
    y_proba = random_forest_model.predict_proba(x_test)

    # check the unique values in predict
    evaluate = EvaluateModel(y_test, y_pred, y_proba, random_forest_model)
    evaluate.evaluate()
    evaluate.plot_confusion_matrix()
    evaluate.plot_roc_curve()

# ============================================================================

    adaboost = model_train.adaboost_model()

    filename = 'adaboost.sav'
    # joblib.dump(adaboost, filename)
    adaboost = joblib.load(filename)
    y_pred = adaboost.predict(x_test)
    y_proba = adaboost.predict_proba(x_test)

    # check the unique values in predict
    evaluate = EvaluateModel(y_test, y_pred, y_proba, adaboost)
    evaluate.evaluate()  # 30 %
    evaluate.plot_confusion_matrix()
    evaluate.plot_roc_curve()

    # ===============================================
    stacking_model = model_train.stacking_model()

    filename = 'stacking_model.sav'
    # joblib.dump(stacking_model, filename)
    stacking_model = joblib.load(filename)
    y_pred = stacking_model.predict(x_test)
    y_proba = stacking_model.predict_proba(x_test)

    # check the unique values in predict
    evaluate = EvaluateModel(y_test, y_pred, y_proba, stacking_model)
    evaluate.evaluate()  # 78 %
    evaluate.plot_confusion_matrix()
    evaluate.plot_roc_curve()

    # ==================================================================
    LGBM_model = model_train.LGBM_model()

    # filename = 'LGBM_model.sav'
    # joblib.dump(LGBM_model, filename)
    #LGBM_model = joblib.load(filename)
    y_pred = LGBM_model.predict(x_test)
    y_proba = LGBM_model.predict_proba(x_test)

    # check the unique values in predict
    evaluate = EvaluateModel(y_test, y_pred, y_proba, LGBM_model)
    evaluate.evaluate()  # 81 %
    evaluate.plot_confusion_matrix()
    evaluate.plot_roc_curve()


def predict(msg):
    # text = DataUtils.get_text_data("training_text")
    # variants = DataUtils.get_variant_data("training_variants")
    # msg = DataUtils.merge_data(variants, text)
    # msg = msg.loc[[0]]

    txtp = TextPreprocessor(msg)
    df = txtp._preprocess_text()

    feature_engineering = FeatureEngineering(df)
    df_new_added_features = feature_engineering.add_more_features()

    vectorizer = joblib.load("tools/vectorizer.pkl")
    test_vector = vectorizer.transform(
        df_new_added_features.Processed_text).toarray()

    count_vector = joblib.load('tools/count_vector.pkl')
    bag_of_cat_feat = count_vector.transform(
        df_new_added_features.Variation+" "+df_new_added_features.Gene).toarray()

    svd = joblib.load('tools/svd.joblib')
    truncated_bag_of_words = svd.transform(np.concatenate((test_vector,
                                                           bag_of_cat_feat,
                                                           df_new_added_features[['no. of nouns', 'Number_of_characters',
                                                                                  'Number_of_unique_words', 'Number_of_words',
                                                                                  'Number_of_sentences',
                                                                                  'Overall_text_length']]), axis=1))

    model = joblib.load("saved_models/LGBM_model.sav")
    output = model.predict(truncated_bag_of_words)
    return output


if __name__ == "__main__":
    prediction = predict(
        ""
    )

    print(prediction)
