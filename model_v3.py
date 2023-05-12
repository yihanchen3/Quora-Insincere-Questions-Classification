import numpy as np 
import pandas as pd 
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from datetime import datetime
import gc
import torch
import torch.nn as nn   

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

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train_final_model(model, x_train, y_train, x_val, y_val ,test_loader, batch_size=512, n_epochs=5):
    optimizer = torch.optim.Adam(model.parameters())
    train = torch.utils.data.TensorDataset(x_train, y_train)
    valid = torch.utils.data.TensorDataset(x_val, y_val)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean').cuda()
    for epoch in range(n_epochs):
        start_time = time.time()
        model.train()
        avg_loss = 0.
        for x_batch, y_batch in tqdm(train_loader, disable=True):
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(train_loader)
        model.eval()
        valid_preds = np.zeros((x_val.size(0)))
        elapsed_time = time.time() - start_time
        print('Epoch {}/{} \t loss={:.4f} \t time={:.2f}s'.format(
            epoch + 1, n_epochs, avg_loss, elapsed_time))
    valid_preds = np.zeros((x_val.size(0)))
    avg_val_loss = 0.
    for i, (x_batch, y_batch) in enumerate(valid_loader):
        y_pred = model(x_batch).detach()
        avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
        valid_preds[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]
    print('Validation loss: ', avg_val_loss)
    test_preds = np.zeros((len(test_loader.dataset)))
    for i, (x_batch,) in enumerate(test_loader):
        y_pred = model(x_batch).detach()
        test_preds[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]
    return valid_preds, test_preds


if __name__ == '__main__':
    gc.collect()
    # load data
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
    embedding_matrix_glove, _ = get_embedding(embedding_files['glove'],tokenizer, max_features, embedding_files)
    embedding_matrix_wiki, _ = get_embedding(embedding_files['wiki'],tokenizer, max_features, embedding_files) 
    embedding_matrix_paragram, _ = get_embedding(embedding_files['paragram'],tokenizer, max_features, embedding_files)
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
        valid_preds_fold, test_preds_fold = train_final_model(model,x_train_fold,y_train_fold,x_val_fold, y_val_fold,test_loader)

        train_preds[valid_idx] = valid_preds_fold
        test_preds[:, i] = test_preds_fold

    train_f1, train_thresh = best_f1_search(y_train, train_preds)
    test_f1, test_thresh = best_f1_search(y_test, test_preds.mean(axis=1))
    print('train f1: {:.4f}, thresh: {:.4f}'.format(train_f1, train_thresh))
    print('test f1: {:.4f}, thresh: {:.4f}'.format(test_f1, test_thresh))
    model_matrix_eval(y_test,test_preds.mean(axis=1),test_thresh,'final')



