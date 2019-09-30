import csv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import Dense, LSTM, Embedding
from tensorflow.python.keras.models import Sequential
from keras import backend

"""
    Description: implementing lstm model to an annotated corpora
                 that contains both argumentative and none sentences
"""

FILE_PATH = os.path.abspath(os.path.dirname(__file__))  # path of this file


def load_dataset(path):
    """
    Loading dataset of a given path
    :param path: full path of dataset
    :return: a list of sentences and their labels
    """

    # load & prepare data
    with open(path, 'r') as file:
        dataset = csv.reader(file, delimiter=",")
        df = pd.DataFrame(dataset)

    del df[2]  # column full of none

    df[1] = pd.factorize(df[1])[0]  # False switched to 0 and True to 1
    texts = list(df[0].values)
    labels = list(df[1])

    return texts, labels


def get_tokenized_text(max_words, texts_train):
    """
    Tokenizing raw sentences.
    :param max_words: considering only the top max_words of dataset (most frequent ones)
    :param texts_train: sentences to for tokenization
    :return: tokenizer, sequences, word_index
    """
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts_train)
    sequences = tokenizer.texts_to_sequences(texts_train)
    word_index = tokenizer.word_index

    print('Found %s unique tokens.' % len(word_index))
    return tokenizer, sequences, word_index


def pad_sentences(sequences, labels_train):
    """
    This function transforms a list of num_samples sequences
    (lists of integers) into a 2D Numpy array of shape (num_samples, num_timesteps).
    The num_timesteps is either the maxlen argument if provided, or the length of the
    longest sequence otherwise.
    Sequences that are shorter than num_timesteps are padded with value at the end.

    :param sequences: list of tokenized sentences
    :param labels_train: list of sequences' labels
    :return: a 2D Numpy array of sentences and labels
    """
    data = pad_sequences(sequences)
    labels_train = np.asarray(labels_train)
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels_train.shape)

    # Splits the data into a training set and a validation set, but first shuffles the data
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels_train = labels_train[indices]

    return data, labels_train


def word_embedding(glove_file, max_words, word_index):
    """
    Parsing the GloVe's word-embeddings file in order to build an index
    that maps words (as strings) to their vector representation (as number vectors),
    and then build an embedding matrix that will be loaded into an Embedding layer
    later on.

    :param glove_file: path of Glove's word-embedding
    :param max_words: considering only the top max_words of dataset (most frequent ones)
    :param word_index: index of words after tokenization
    :return: embedding_dim, embedding_matrix of shape (max_words, embedding_dim)
    """

    embeddings_index = {}
    f = open(os.path.join(FILE_PATH, glove_file))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))

    embedding_dim = 100
    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in word_index.items():
        if i < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # Words not found in the embedding index will be all zeros.
                embedding_matrix[i] = embedding_vector

    return embedding_dim, embedding_matrix


def lstm_model(max_words, embedding_dim, embedding_matrix):
    """
    This function defines the model and
    loads pretrained word embedding into the Embedding layer

    :param max_words: considering only the top max_words of dataset (most frequent ones)
    :param embedding_dim: number of embedding dimensions used
    :param embedding_matrix: a matrix of shape shape (max_words, embedding_dim)
    :return: the defined model
    """

    # define model
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())

    # Loading pretrained word embeddings into the Embedding layer
    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = False

    return model


def recall_m(y_true, y_pred):
    """
    Calculating recall metric

    :param y_true: the real value of a sentence
    :param y_pred: the estimated value of a sentence
    :return: recall metric
    """
    true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = backend.sum(backend.round(backend.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + backend.epsilon())
    return recall


def precision_m(y_true, y_pred):
    """
    Calculating precision metric

    :param y_true: the real value of a sentence
    :param y_pred: the estimated value of a sentence
    :return: precision metric
    """
    true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = backend.sum(backend.round(backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + backend.epsilon())
    return precision


def f1_m(y_true, y_pred):
    """
    Calculating F1 Score metric

    :param y_true: the real value of a sentence
    :param y_pred: the estimated value of a sentence
    :return: f1_score metric
    """
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + backend.epsilon()))


def training_model(x_train, y_train, x_val, y_val, model):
    """
    This function is responsible for compiling and
    training the model

    :param x_train: sentences for training model (200 sentences)
    :param y_train: labels of training sentences
    :param x_val: sentences for validating trained model (1000 sentences)
    :param y_val: labels of validation sentences
    :param model: defined model
    :return: historing of training, trained model
    """

    model.compile(loss='binary_crossentropy', optimizer='rmsprop',
                  metrics=['accuracy', f1_m, precision_m, recall_m])
    history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_val, y_val))
    model.save_weights('pre_trained_glove_model.h5')

    return history, model


def plotting_results(history):
    """
    This function is showing plots of the training and validation results
    based on accuracy and loss metrics.

    :param history: history of training procedure
    """
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def testing_model(texts_test, labels_test, tokenizer, model):
    """
    This function tokenizes the data of test set,
    and evaluates the model on the test set

    :param texts_test: sentences for testing model
    :param labels_test: labels of the testing sentences
    :param tokenizer: tokenizer
    :param model: trained model
    :return: accuracy of the created model
    """

    # Tokenizing the data of the test set
    sequences = tokenizer.texts_to_sequences(texts_test)
    x_test = pad_sequences(sequences)
    y_test = np.asarray(labels_test)

    # Evaluating the model on the test set
    model.load_weights('pre_trained_glove_model.h5')
    loss, accuracy, f1_score, precision, recall = model.evaluate(x_test, y_test)
    print("Accuracy: %.2f%%" % (accuracy * 100))
    print("Precision: %.2f%%" % (precision * 100))
    print("Recall: %.2f%%" % (recall * 100))
    print("F1 score: %.2f%%" % (f1_score * 100))


def main():
    texts, labels = load_dataset('../Results/dataset.csv')
    texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, test_size=0.2)

    max_words = 10000  # Considers only the top 10,000 words in the dataset
    training_samples = 200
    validation_samples = 1000

    tokenizer, sequences, word_index = get_tokenized_text(max_words, texts_train)
    data, labels_train = pad_sentences(sequences, labels_train)

    x_train = data[:training_samples]
    y_train = labels_train[:training_samples]
    x_val = data[training_samples: training_samples + validation_samples]
    y_val = labels_train[training_samples: training_samples + validation_samples]

    embedding_dim, embedding_matrix \
        = word_embedding('../Reading/glove.6B/glove.6B.100d.txt', max_words, word_index)

    model = lstm_model(max_words, embedding_dim, embedding_matrix)
    history, model = training_model(x_train, y_train, x_val, y_val, model)
    plotting_results(history)
    testing_model(texts_test, labels_test, tokenizer, model)


if __name__ == '__main__':
    main()
