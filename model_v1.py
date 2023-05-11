import re
import string
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score, 
    classification_report,
    accuracy_score
)
from tqdm import tqdm
tqdm.pandas()

def plot_traget_count(train):
    count = train['target'].value_counts()
    count.plot(kind='bar')
    plt.title('Target Count')
    plt.xlabel('Target')
    plt.ylabel('Count')
    plt.savefig('./outputs/target_count.png')

def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def remove_stopwords(text):
    # nltk.download('stopwords')
    stop_words = stopwords.words('english')
    text = ' '.join(word for word in text.split(' ') if word not in stop_words)
    return text

def stemm_text(text):
    stemmer = nltk.SnowballStemmer("english")
    text = ' '.join(stemmer.stem(word) for word in text.split(' '))
    return text

def evaluate_model(y_val, y_pred):
    # y_pred = (y_pred>0.5).astype(int)
    print(f'Accuracy score: {accuracy_score(y_val, y_pred):.4f}')
    print(f'F1 score: {f1_score(y_val, y_pred):.4f}')
    print(classification_report(y_val, y_pred))
    confution_lg = confusion_matrix(y_val, y_pred) #confusion metrics
    sns.heatmap(confution_lg, linewidths=0.01, annot=True,fmt= '.1f', color='red') #heat map

def plot_training(history, model_name):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.savefig('./outpus/loss_'+ model_name +'.png')
    # plot training and validation loss


if __name__ == '__main__':

    train = pd.read_csv("./input/train.csv")
    plot_traget_count(train)
    print('train data loaded, shape: ', train.shape)

    print('cleaning text...')
    train['question_text'] = train['question_text'].progress_apply(clean_text)

    print('removing stopwords...')
    train['question_text'] = train['question_text'].progress_apply(remove_stopwords)

    print('stemming text...')
    train['question_text'] = train['question_text'].progress_apply(stemm_text)
    print('train data cleaned')
    print(train.head())

    x_train, x_test, y_train, y_test = train_test_split(train['question_text'] ,train['target'], test_size= 0.2 ,random_state=42)
    print('x_train: ', x_train.shape)

    vectoriser = TfidfVectorizer(ngram_range=(1,2),max_features=3891472)
    vectoriser.fit(x_train)

    x_train = vectoriser.transform(x_train)
    x_test  = vectoriser.transform(x_test)

    # Create a Logistic Regression model
    lr = LogisticRegression()
    # Train the model
    lr.fit(x_train, y_train)
    # Evaluate the model on the test set
    print('Evaluating model...')
    y_pred_lr = lr.predict(x_test)
    evaluate_model(y_test, y_pred_lr)

    # Create a Multinomial Naive Bayes model
    print('Training model...')
    nb = MultinomialNB()
    # Train the model
    nb.fit(x_train, y_train)
    # Evaluate the model on the test set
    print('Evaluating model...')
    y_pred_nb = nb.predict(x_test)
    evaluate_model(y_test, y_pred_nb)


