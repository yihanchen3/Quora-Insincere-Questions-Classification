# Natural Language Processing Approaches on Quora Insincere Questions Classification Task

In this project, a Quora Insincere Questions Classification task is addressed by three version of models individually (https://www.kaggle.com/c/quora-insincere-questions-classification). The main target of this binary classification task is to identify whether the question text are insincere or sincere. 

To achieve the goal, the first model developed is the traditional base model of Logistic Regression and Naive Bayes Classifier. It is trained with the data after standard preprocessing and TF-IDF Vectorizer.

The second model uses bidirectional GRU as the main layer to consist the deep learning model. It is trained with the data after customized preprocessing and three different pre-traiend word embeddings.

The final model improves the neural network with bidirectioan LSTM+GRU and Attention mechnaism. It is trained with the data after customized preprocessing and combined pre-traiend word embeddings.

## Getting Started

### Prerequisites

* Python 3.x
* Tensorflow
* NLTK
* numpy
* sklearn
* Pytorch

### Installing

1. Clone the repository:  `https://github.com/yihanchen3/DLNLP_Project.git`
2. Install the required libraries:  `pip install -r requirements.txt`
3. Download the input file (`train.csv` and `embeddings` ) from the Kaggle website, and put them into the corresponding folder as showed in the below project structure.

### Training and testing the models

1. For the base model, run the command: `python model_v1.py`
2. For the BiGRU model, run the command: `python model_v2.py`
3. For the final model, run the command: `python model_v3.py`

## Structure of the project

```
│  model_v1.py
│  model_v2.py
│  model_v3.py
│  README.md
│  requirements.txt
│  
├─input
│  │  test.csv
│  │  train.csv
│  │  
│  └─embeddings
│      ├─glove.840B.300d
│      │      glove.840B.300d.txt
│      │  
│      ├─GoogleNews-vectors-negative300
│      │      GoogleNews-vectors-negative300.bin
│      │  
│      ├─paragram_300_sl999
│      │      paragram_300_sl999.txt
│      │  
│      └─wiki-news-300d-1M
│              wiki-news-300d-1M.vec
│        
├─outputs
│
│  
└─src
    │  model_define.py
    │  model_train.py
    │  preprocessing.py
    │  
    └─__pycache__
      
```


## Role of each file

- `model_v1.py`: This file defines and trains the base models. Standard texts preprocessing is also defined in thsi file. This file  is a individual script not relying on other supporting codes.
- `model_v2.py`: This file trains the BiGRU model with three pre-trained word embeddings.
- `model_v3.py`: This file trains the Final model with the combined word embeddings.
- `src/`: This folder contains the supporting codes for model_v2 and v3 to work.

  - `model_define.py`: This file defines the BiGRU model and final LSTM+GRU+Attention mdel.
  - `model_train.py`: This file contains functions to train and evaluate the models defined in the previous file.
  - `preprocessing.py`: This file contains various NLP text preprocessing functions which are implemented before the training.
- `input/`: This folder contains question texts file and word embeddings.

  - `train.csv `: This file contains the question texts and corresponding labels and is used as the input of all models.
  - `fembeddings/`: This folder contains three pre-trained word embeddings usd in the models.
- `outpust/`: This folder contains various outputs from the model training process, including figures, saved models, and result logs.
