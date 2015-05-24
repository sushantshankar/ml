__author__ = 'sushant'

import numpy as np
import sklearn.cross_validation as cross_validation
import sklearn.linear_model as linear_model
import sklearn.svm as svm
import sklearn.multiclass as multiclass
import sklearn.metrics as metrics
import pandas as pd


def extract_labels_features(dataset_fname):
    """
    Take in a dataset and extract labels and features

    :param dataset_fname:
    :return:
    labels
    features
    """
    # Read in data as a pandas data-set.
    train_df = pd.read_csv(dataset_fname)

    # Extract the labels and features
    labels = train_df["hand"].values
    feature_columns = set(train_df.keys()) - set(['hand'])
    features = train_df[list(feature_columns)].values

    return features, labels


def train_model(features, labels):
    # Set up SVM model. Try linear kernel first.
    estimator = svm.SVC(kernel='linear')
    #estimator = linear_model.LogisticRegression()

    # Set up multi-class classifier
    #multi_label_classifier = multiclass.OneVsRestClassifier(estimator)
    multi_label_classifier = multiclass.OneVsOneClassifier(estimator)

    X_train, X_test, y_train, y_test = \
        cross_validation.train_test_split(features, labels, test_size=0.7, random_state=0)

    predictions = multi_label_classifier.fit(X_train, y_train).predict(X_test)
    score = metrics.accuracy_score(y_test, predictions)
    print 'Classification accuracy on training data: %f' % score

    print(metrics.classification_report(y_test, predictions))

    """
    # Get cross validation score.
    cross_validation.cross_val_score(estimator=multi_label_classifier, X=features, y=labels)
    cross_val_score = cross_validation.cross_val_score(estimator=multi_label_classifier, X=features, y=labels).mean()
    print cross_val_score
    """

def main():
    # Extract labels and features
    features, labels = extract_labels_features("data/train_1k.csv")

    # Train model
    train_model(features, labels)

if __name__=="__main__":
    main()