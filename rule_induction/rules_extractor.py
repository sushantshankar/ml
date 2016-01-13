__author__ = 'sushant'

import numpy as np
import sklearn.cross_validation as cross_validation
import sklearn.linear_model as linear_model
import sklearn.svm as svm
import sklearn.multiclass as multiclass
import sklearn.metrics as metrics
import pandas as pd


def extract_labels_features(train_df):
    """
    Take in a Pandas dataset and extract labels and features

    :param dataset_fname:
    :return:
    features
    labels
    train_df
    """
    # Extract the labels and features
    labels = train_df["hand"].values
    feature_columns = set(train_df.keys()) - set(['hand'])
    features = train_df[list(feature_columns)].values

    return features, labels, train_df


def feature_transformer(df, feature_transformer):
    """
    Given features (Pandas' data frame), transform the data-set
    with the given feature transformation method
    """
    df_new = pd.DataFrame([feature_transformer(s1,c1,s2,c2,s3,c3,s4,c4,s5,c5) for s1,c1,s2,c2,s3,c3,s4,c4,s5,c5 in zip(df[["S1", "C1", "S2", "C2", "S3", "C3", "S4", "C4", "S5", "C5"]])])
    #df["deck"] = df.apply(feature_transformer, axis=1)
    return df_new



def feature_transformer_to_deck(data_point, *args):
    """
    Given a data point of 5 cards, transform this into a deck.
    The data point could be:
    a) An numpy array.
    b) args - list of these data points

    >>> # Note that the last column is the label.
    >>> data_point = np.array([4,9,2,1,2,2,4,7,2,8,1])
    >>> deck_activation = feature_transformer_to_deck(data_point)
    >>> # Check to see where the deck is activated.
    >>> np.where(deck_activation == 1)[0]
    array([13, 14, 20, 45, 47])
    """


    deck_activation = np.zeros(52)

    # Pair each data point together
    for i in range(0, len(data_point) - 1, 2):
        data_act = data_point[i:i+2]
        suit, rank = data_act.tolist()
        deck_activation[suit_and_rank_to_idx(suit, rank)] = 1

    return deck_activation


def suit_and_rank_to_idx(suit, rank):
    """
    Given a suit and rank, turn this into an index from 0 - 51.
    >>> # Hearts - Ace
    >>> suit_and_rank_to_idx(1, 1)
    0
    >>> # Clubs - King
    >>> suit_and_rank_to_idx(4, 13)
    51
    >>> suit_and_rank_to_idx(4, 9)
    47
    >>> suit_and_rank_to_idx(2, 1)
    13
    >>> suit_and_rank_to_idx(2, 2)
    14
    >>> suit_and_rank_to_idx(4, 7)
    45
    >>> suit_and_rank_to_idx(2, 8)
    20
    >>> suit_and_rank_to_idx(2, 15)
    Traceback (most recent call last):
        ...
    AssertionError: Rank needs to be between 1 and 13: 15
    >>> suit_and_rank_to_idx(5, 13)
    Traceback (most recent call last):
        ...
    AssertionError: Suit needs to be between 1 and 4: 5
    """
    assert 1 <= suit <= 4, "Suit needs to be between 1 and 4: %d" % suit
    assert 1 <= rank <= 13, "Rank needs to be between 1 and 13: %d" % rank
    return (suit - 1) * 13 + (rank - 1)


def train_model(features, labels):
    # Set up SVM model. Try linear kernel first.
    estimator = svm.SVC(kernel='linear')
    #estimator = linear_model.LogisticRegression()

    # Set up multi-class classifier
    multi_label_classifier = multiclass.OneVsRestClassifier(estimator)
    # multi_label_classifier = multiclass.OneVsOneClassifier(estimator)

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
    # Read in data as a pandas data-set.
    train_df = pd.read_csv("data/train_1K.csv")
    print train_df

    features, labels, train_df = extract_labels_features(train_df)

    # Transform features
    train_df = feature_transformer(train_df, feature_transformer_to_deck)

    print train_df

    # Extract labels and features
    features, labels = extract_labels_features(train_df)

    # Train model
    train_model(features, labels)

if __name__=="__main__":
    main()