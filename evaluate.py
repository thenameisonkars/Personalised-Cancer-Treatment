from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_curve,
    roc_auc_score
)
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scikitplot
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class EvaluateModel:
    def __init__(self, y_test, y_pred, y_proba, model):
        self.y_pred = y_pred
        self.y_test = y_test
        self.model = model
        self.y_proba = y_proba

    def evaluate(self):
        print(classification_report(self.y_test, self.y_pred))
        # skplt.plot_confusion_matrix(self.y_test, self.y_pred)
        print("Completed evaluating the model")

    def plot_confusion_matrix(self):

        confusion = confusion_matrix(self.y_test, self.y_pred)

        Recall = ((confusion.T) / (confusion.sum(axis=1))).T
        # divide each element of the confusion matrix with the sum of elements in that column

        Precision = confusion / confusion.sum(axis=0)
        # divide each element of the confusion matrix with the sum of elements in that row

        plt.figure(figsize=(20, 4))

        labels = [0, 1]
        cmap = sns.light_palette("blue")
        plt.subplot(1, 3, 1)
        sns.heatmap(
            confusion,
            annot=True,
            cmap=cmap,
            fmt=".3f",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.xlabel("Predicted Class")
        plt.ylabel("Original Class")
        plt.title("Confusion matrix")

        plt.subplot(1, 3, 2)
        sns.heatmap(
            Precision,
            annot=True,
            cmap=cmap,
            fmt=".3f",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.xlabel("Predicted Class")
        plt.ylabel("Original Class")
        plt.title("Precision matrix")

        plt.subplot(1, 3, 3)
        sns.heatmap(
            Recall,
            annot=True,
            cmap=cmap,
            fmt=".3f",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.xlabel("Predicted Class")
        plt.ylabel("Original Class")
        plt.title("Recall matrix")

        plt.show()

    def plot_roc_curve(self):
        auroc = roc_auc_score(self.y_test.reshape(-1, 1),
                              self.y_proba, multi_class='ovr')
        print("AUROC Score:- ", auroc)

        fpr = {}
        tpr = {}
        thresh = {}

        n_class = 9

        for i in range(n_class):
            fpr[i], tpr[i], thresh[i] = roc_curve(
                self.y_test, self.y_proba[:, i], pos_label=i)
            plt.plot(fpr[i], tpr[i], label=f'Class {i+1} vs Rest')

        # plotting
        plt.title('Multiclass ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive rate')
        plt.legend(loc='best')

        plt.show()
