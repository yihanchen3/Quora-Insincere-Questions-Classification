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
from sklearn.model_selection import StratifiedKFold
from keras.models import Model
from src.preprocessing import *
from src.model_define import *
from src.model_train import *
tqdm.pandas()


embedding_files = {
    'glove': './input/embeddings/glove.840B.300d/glove.840B.300d.txt',
    'wiki': './input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec',
    'paragram': './input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
}



def get_embedding(EMBEDDING_FILE):

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    if EMBEDDING_FILE == embedding_files['glove']:
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding='utf-8', errors='ignore'))
    elif EMBEDDING_FILE == embedding_files['paragram'] or EMBEDDING_FILE == embedding_files['wiki']:
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    
    del word_index, all_embs
    
    return embedding_matrix,embeddings_index




if __name__ == '__main__':
    gc.collect()
    ## load data
    print('\nloading data...')
    train_df = pd.read_csv("./input/train.csv")
    
    print("Train shape : ",train_df.shape)

    ## clean the text
    print('\ncleaning text...')
    train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: clean_text(x))
    train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: clean_numbers(x))
    train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: replace_typical_misspell(x))

    ## split to train and val
    print('\nsplitting train and test...')
    train,test = train_test_split(train_df, test_size=0.1, random_state=42)
    len_train,len_test = len(train), len(test)
    y_train = train['target'].values
    y_test = test['target'].values

    ## tokenize the sentences
    max_features = 120000
    maxlen = 72
    embed_size = 300
    tokenizer = Tokenizer(lower = True, filters='', num_words=max_features)
    full_text = list(train['question_text'].values) + list(test['question_text'].values)
    tokenizer.fit_on_texts(full_text)
    train_tokenized = tokenizer.texts_to_sequences(train['question_text'].fillna('_na_'))
    test_tokenized = tokenizer.texts_to_sequences(test['question_text'].fillna('_na_'))

    X_train = pad_sequences(train_tokenized, maxlen = maxlen)
    X_test = pad_sequences(test_tokenized, maxlen = maxlen)

    splits = list(StratifiedKFold(n_splits=4, shuffle=True, random_state=10).split(X_train, y_train))


    print('\ntesting each embedding...')
    embedding_matrix_glove, _ = get_embedding(embedding_files['glove'])
    embedding_matrix_wiki, _ = get_embedding(embedding_files['wiki']) 
    embedding_matrix_paragram, _ = get_embedding(embedding_files['paragram'])
    embedding_matrix = np.mean([embedding_matrix_glove, embedding_matrix_wiki,embedding_matrix_paragram], axis=0)
    del embedding_matrix_glove, embedding_matrix_wiki, embedding_matrix_paragram

    x_test_cuda = torch.tensor(X_test, dtype=torch.long).cuda()
    test = torch.utils.data.TensorDataset(x_test_cuda)
    batch_size = 512
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    seed = 42
    seed_everything(seed)

    train_preds = np.zeros(len_train)
    test_preds = np.zeros((len_test, len(splits)))
    n_epochs = 2
    for i, (train_idx, valid_idx) in enumerate(splits):    
        x_train_fold = torch.tensor(X_train[train_idx], dtype=torch.long).cuda()
        y_train_fold = torch.tensor(y_train[train_idx, np.newaxis], dtype=torch.float32).cuda()
        x_val_fold = torch.tensor(X_train[valid_idx], dtype=torch.long).cuda()
        y_val_fold = torch.tensor(y_train[valid_idx, np.newaxis], dtype=torch.float32).cuda()
        
        train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
        valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)
        
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)
        
        print(f'Fold {i + 1}')
        
        seed_everything(seed + i)
        model = NeuralNet(embedding_matrix,max_features,embed_size,maxlen)
        model.cuda()
        valid_preds_fold, test_preds_fold = train_final_model(model,x_train_fold,y_train_fold,x_val_fold, y_val_fold,test_loader, validate=False)

        train_preds[valid_idx] = valid_preds_fold
        test_preds[:, i] = test_preds_fold

    train_f1, train_thresh = best_f1_search(y_train, train_preds)
    test_f1, test_thresh = best_f1_search(y_test, test_preds.mean(axis=1))
    print('train f1: {:.4f}, thresh: {:.4f}'.format(train_f1, train_thresh))
    print('test f1: {:.4f}, thresh: {:.4f}'.format(test_f1, test_thresh))
    model_matrix_eval(y_test,test_preds.mean(axis=1),test_thresh,'final')



