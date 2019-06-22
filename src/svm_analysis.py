# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import TweetTokenizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score

from sklearn.model_selection import learning_curve


def tokenize(text):
  tokenizer = TweetTokenizer()
  return tokenizer.tokenize(text)


def stem(document):
  return (stemmer.stem(w) for w in analyzer(document))


def report_results(model, X, y):
  pred_proba = model.predict_proba(X)[:, 1]
  pred = model.predict(X)

  auc = roc_auc_score(y, pred_proba)
  acc = accuracy_score(y, pred)
  f1 = f1_score(y, pred)
  prec = precision_score(y, pred)
  rec = recall_score(y, pred)
  result = {'auc': auc, 'f1': f1, 'acc': acc, 'precision': prec, 'recall': rec}
  return result

def get_roc_curve(model, x, y):
  pred_proba = model.predict_proba(x)[:, 1]
  fpr, tpr, _ = roc_curve(y, pred_proba)
  return fpr, tpr


def plot_learning_curve(x, y, train_sizes, train_scores,
                        test_scores, title='', ylim=None, figsize=(14, 8)):
  plt.figure(figsize=figsize)
  plt.title(title)
  if ylim is not None:
      plt.ylim(*ylim)
  plt.xlabel("Training examples")
  plt.ylabel("Score")

  train_scores_mean = np.mean(train_scores, axis=1)
  train_scores_std = np.std(train_scores, axis=1)
  test_scores_mean = np.mean(test_scores, axis=1)
  test_scores_std = np.std(test_scores, axis=1)
  plt.grid()

  plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1,
                    color="r")
  plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color="g")
  plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
            label="Training score")
  plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
            label="Cross-validation score")

  plt.legend(loc="lower right")
  return plt


if __name__ == '__main__':
  names = ['polarity', 'tweet_id', 'date', 'query', 'username', 'text']
  raw_data = pd.read_csv('./data/training_data.csv', names=names, encoding='latin-1')

  clean_data = raw_data.loc[:, ['text', 'polarity']]
  # clean_data.describe()

  unused_data, mini_set = train_test_split(clean_data, test_size=0.05, random_state=1)
  mini_set.describe()

  mini_set.loc[:, 'sentiment'] = mini_set['polarity'].apply(lambda x: 1 if x == 4 else 0)
  mini_set = mini_set.loc[:, ['text', 'sentiment']]

  # mini_set.describe()

  train, test = train_test_split(mini_set, test_size=0.2, random_state=1)
  x_train = train['text'].values
  y_train = train['sentiment'].values
  x_test = test['text'].values
  y_test = test['sentiment']

  print(y_train)

  en_stopwords = set(stopwords.words('english'))

  vectorizer = CountVectorizer(
      analyzer='word',
      tokenizer=tokenize,
      lowercase=True,
      ngram_range=(1, 1),
      stop_words=en_stopwords)

  kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

  np.random.seed(1)

  pipeline_svm = make_pipeline(vectorizer, SVC(probability=True, kernel='linear',
                                                class_weight='balanced'))
  grid_svm = GridSearchCV(pipeline_svm,
                          param_grid={'svc__C': [0.01, 0.1, 1]},
                          cv=kfolds,
                          scoring="roc_auc",
                          verbose=1,
                          n_jobs=-1)

  grid_svm.fit(x_train, y_train)
  print(grid_svm.score(x_test, y_test))

  print(grid_svm.best_params_)
  print(grid_svm.best_score_)
  print(report_results(grid_svm.best_estimator_, x_test, y_test))

  roc_svm = get_roc_curve(grid_svm.best_estimator_, x_test, y_test)

  fpr, tpr = roc_svm
  plt.figure(figsize=(14, 8))
  plt.plot(fpr, tpr, color="red")
  plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Roc curve')
  plt.show()

  train_sizes, train_scores, test_scores = learning_curve(
      grid_svm.best_estimator_,
      x_train,
      y_train,
      cv=5,
      n_jobs=-1,
      scoring="roc_auc",
      train_sizes=np.linspace(.1, 1.0, 10),
      random_state=1)

  plot_learning_curve(x_train, y_train, train_sizes,
                      train_scores, test_scores, ylim=(0.7, 1.01), figsize=(14,6))
  plt.show()

  grid_svm.predict(x_test)
