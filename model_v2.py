import numpy as np 
import pandas as pd 
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from datetime import datetime
import gc

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Model
from src.preprocessing import *
from src.model_define import model_base_define,model_embedding_define
from src.model_train import best_f1_search,plot_training,model_matrix_eval,get_embedding

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)

tqdm.pandas()


embedding_files = {
    'glove': './input/embeddings/glove.840B.300d/glove.840B.300d.txt',
    'wiki': './input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec',
    'paragram': './input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
}

# config values 
embed_size = 300 # how big is each word vector
max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a question to use


def train_model(EMBEDDING_NAME,train_X,train_y,val_X,val_y,test_X,test_y,tokenizer):

    print('start embedding {}'.format(EMBEDDING_NAME))
    EMBEDDING_FILE = embedding_files[EMBEDDING_NAME]
    print('embedding file: {}'.format(EMBEDDING_FILE))
    embedding_matrix, embeddings_index = get_embedding(EMBEDDING_FILE,tokenizer,max_features,embedding_files)
    oov = embedding_eval(train_df,embeddings_index)

    print('start training {}'.format(EMBEDDING_NAME))
    model = model_embedding_define(maxlen, max_features, embed_size, embedding_matrix)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))
    plot_training(history,EMBEDDING_NAME)
    model.save('./outputs/' + EMBEDDING_NAME + '_model.h5')

    print('start evaluating {}'.format(EMBEDDING_NAME))
    pred_test_y = model.predict([test_X], batch_size=1024, verbose=1)

    best_f1, best_thresh = best_f1_search(test_y,pred_test_y)
    model_matrix_eval(test_y,pred_test_y,best_thresh,EMBEDDING_NAME)

    # print('start predicting {}'.format(EMBEDDING_NAME))
    # pred_test_y = model.predict([test_X], batch_size=1024, verbose=1)
    
    del embeddings_index, embedding_matrix, model
    gc.collect()

    return {'pred_test_y':pred_test_y,'best_f1':best_f1,'best_thresh':best_thresh}



if __name__ == '__main__':

    gc.collect()
    print('\nloading data...')
    train_df = pd.read_csv("./input/train.csv")
    test_df = pd.read_csv("./input/test.csv")
    print("Train shape : ",train_df.shape)
    print("Test shape : ",test_df.shape)

    ## clean the text
    print('\ncleaning text...')
    train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: clean_text(x))
    train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: clean_numbers(x))
    train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: replace_typical_misspell(x))

    # split to train and val
    print('\nsplitting train and val and test set...')
    train_X, val_X, test_X, train_y, val_y, test_y, tokenizer = train_val_split(train_df)

    print('\ntesting each embedding...')
    EMBEDDING_NAME = 'glove'
    glove_result = train_model('glove',train_X,train_y,val_X,val_y,test_X,test_y,tokenizer)

    EMBEDDING_NAME = 'wiki'
    wiki_result = train_model('wiki',train_X,train_y,val_X,val_y,test_X,test_y,tokenizer)

    EMBEDDING_NAME = 'paragram'
    paragram_result = train_model('paragram',train_X,train_y,val_X,val_y,test_X,test_y,tokenizer)

    ## concat result
    print('\nconcating results...')
    pred_test_y = 0.33*glove_result['pred_test_y'] + 0.33*wiki_result['pred_test_y'] + 0.34*paragram_result['pred_test_y']
    best_f1, best_thresh = best_f1_search(test_y,pred_test_y)
    model_matrix_eval(test_y,pred_test_y,best_thresh,'concat')

    # save result to txt file
    print('\nsaving result...')
    with open('./outputs/result.txt', 'a') as f:
        f.write('time: {}\n'.format(datetime.now()))
        f.write('best f1: {}\n'.format(best_f1))
        f.write('best thresh: {}\n'.format(best_thresh))
        f.write('glove f1: {}\n'.format(glove_result['best_f1']))
        f.write('wiki f1: {}\n'.format(wiki_result['best_f1']))
        f.write('paragram f1: {}\n'.format(paragram_result['best_f1']))
        f.write('\n')

    # # if want to combine all three embeddings to train model, uncomment the following code
    # # best f1 score is 0.6756995581737849 at threshold 0.37 Val accuracy: 0.9579 F1 score: 0.6757
    # EMBEDDING_NAME = 'glove_wiki_paragram'
    # embedding_matrix_glove, _ = get_embedding(embedding_files['glove'],tokenizer,max_features,embedding_files)
    # embedding_matrix_wiki, _ = get_embedding(embedding_files['wiki'],tokenizer,max_features,embedding_files) 
    # embedding_matrix_paragram, _ = get_embedding(embedding_files['paragram'],tokenizer,max_features,embedding_files)
    # embedding_matrix = np.mean([embedding_matrix_glove, embedding_matrix_wiki,embedding_matrix_paragram], axis=0)
    # del embedding_matrix_glove, embedding_matrix_wiki, embedding_matrix_paragram
    # print('start training {}'.format(EMBEDDING_NAME))
    # model = model_embedding_define(maxlen, max_features, embed_size, embedding_matrix)
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # history = model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))
    # plot_training(history,EMBEDDING_NAME)
    # model.save('./outputs/' + EMBEDDING_NAME + '_model.h5')
    # print('start evaluating {}'.format(EMBEDDING_NAME))
    # pred_test_y = model.predict([test_X], batch_size=1024, verbose=1)
    # best_f1, best_thresh = best_f1_search(test_y,pred_test_y)
    # model_matrix_eval(test_y,pred_test_y,best_thresh,EMBEDDING_NAME)
