import os
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib import pyplot as plt

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D,GRU
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from sklearn.metrics import accuracy_score ,confusion_matrix, classification_report, f1_score
import seaborn as sns


def model_base_define(maxlen, max_features, embed_size):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size)(inp)
    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(16, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    print(model.summary())
    return model


def model_embedding_define(maxlen, max_features, embed_size, embedding_matrix):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(16, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    print(model.summary())
    return model
  

def best_f1_search(val_y,pred_val_y):
    best_f1 = 0
    best_thresh = 0

    for thresh in np.arange(0.1, 0.501, 0.01):
        thresh = np.round(thresh, 2)
        f1 = metrics.f1_score(val_y, (pred_val_y>thresh).astype(int))
        print("F1 score at threshold {0} is {1}".format(thresh, f1))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    print('\nbest f1 score is {} at threshold {}'.format(best_f1,best_thresh))

    return best_f1,best_thresh

def model_matrix_eval(val_y,pred_val_y,best_thresh,EMBEDDING_NAME):
    best_pred_val_y = (pred_val_y>best_thresh).astype(int)
    accuracy = accuracy_score(val_y, best_pred_val_y)
    print(f"Val accuracy: {accuracy:.4f}")
    print(f'F1 score: {f1_score(val_y, best_pred_val_y):.4f}')
    confution_lg = confusion_matrix(val_y,best_pred_val_y) #confusion metrics
    sns.heatmap(confution_lg, linewidths=0.01, annot=True,fmt= '.1f', color='red') #heat map
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix of ' + EMBEDDING_NAME)
    plt.savefig('../outputs/confusion_matrix_' + EMBEDDING_NAME + '.png')
    plt.close()

def plot_training(history, model_name):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.savefig('../outputs/loss_'+ model_name +'.png')
    plt.close()
    # plot training and validation loss