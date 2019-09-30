import pandas as pd
import seaborn as sn
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import string
import csv
import os
import numpy as np


"""
    Description: implementing Random Forest classifier model
                 to an annotated corpora that contains 
                 both argumentative and none sentences
"""

FILE_PATH = os.path.abspath(os.path.dirname(__file__))  # path of this file


def load_dataset(path):
    """
    Loading dataset of a given path, and creating features based on the sentences given.
    The first feature is a counter of words included in each sentence,
    the second feature is a counter of uppercase characters, while
    the third feature is a counter of special characters (punction)
    :param path: full path of dataset
    :return: a list of sentences and their labels, and a set of features
    """

    # load & prepare data
    with open(path, 'r') as file:
        dataset = csv.reader(file, delimiter=",")
        df = pd.DataFrame(dataset)

    del df[2]  # column full of none
    labels = df[1]
    df[1] = [len(sentences.split()) for sentences in df[0]]  # Word Count'
    df[2] = [sum(char.isupper() for char in sentence) \
             for sentence in df[0]]  # 'Uppercase Char Count'

    df[3] = [sum(char in string.punctuation for char in sentence) \
             for sentence in df[0]]  # 'Special Char Count'
    df[4] = pd.factorize(labels)[0]  # switch False to 0 and True to 1

    return df


def tokenize_sentences(df):
    """
    Tokenizing raw sentences by using One-hot encoding
    into a format that a computer can understand

    :param df: the dataframe that includes 4 columns
                [sentences, word Counter, uppercase counter,
                special char counter, label]
    :return: tokenized sentences, labels
    """
    vectorizer = OneHotEncoder(handle_unknown='ignore')
    vectorizer.fit([[line.strip()] for line in df[0]])
    sentences = vectorizer.transform([[line.strip()] for line in df[0]]).toarray()

    # sentences = vectorizer.fit_transform(df[0])
    labels = df[4]
    return sentences, labels


def define_model(x_train, y_train):
    """
    This function defines and trains the Random Forest model

    :param x_train: the training sentences
    :param y_train: the sentences' labels
    :return: trained model
    """
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    return model


def testing_model(model, x_test, y_test):
    """
    This function evaluates the model on the test set

    :param model: trained model
    :param x_test: the testing set of sentences
    :param y_test: the sentences' labels
    :return: a matrix of the predicted and real values
    """
    score = model.score(x_test, y_test)
    print("Accuracy: %.2f%%" % (score * 100))

    y_predicted = model.predict(x_test)
    cm = confusion_matrix(y_test, y_predicted)
    print(cm)
    return cm


def plot_results(cm):
    """
    This function is showing a heatmap plot based on
    the values of the confusion matrix that contains
    the real an predicted values.
    :param cm: matrix of both real and predicted values
    """
    plt.figure(figsize=(10, 7))
    sn.heatmap(cm, annot=True)
    plt.xlabel('Prediction')
    plt.ylabel('Truth')
    plt.show()


if __name__ == '__main__':
    df = load_dataset('../Results/dataset.csv')
    sentences, labels = tokenize_sentences(df)

    #  using features instead of the sentences
    #
    features = np.asarray(df[df.columns[1:4]].values)
    # split dataset into test and train data
    x_train, x_test, y_train, y_test = \
        train_test_split(features, labels, test_size=0.33)

    model = define_model(x_train, y_train)
    cm = testing_model(model, x_test, y_test)
    plot_results(cm)

    #  using sentences without their features
    #
    # split dataset into test and train data
    x_train, x_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.33)

    model = define_model(x_train, y_train)
    cm = testing_model(model, x_test, y_test)
    plot_results(cm)
