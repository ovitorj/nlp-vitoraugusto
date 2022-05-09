from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import TweetTokenizer
import joblib
import pickle
from sklearn import *
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
nltk.download('stopwords')


app = Flask(__name__)


def RemoveStopWords(instancia):
    stopwords = set(nltk.corpus.stopwords.words('portuguese'))
    palavras = [i for i in instancia.split() if not i in stopwords]
    return (" ".join(palavras))


df_test_under = pd.read_excel('base_tratada.xlsx')
SEED = 42
tweets = df_test_under['tweet']
classes = df_test_under['classificacao']
tweets = [RemoveStopWords(i) for i in tweets]
tweet_tokenizer = TweetTokenizer()
vectorizer = CountVectorizer(
    analyzer="word", tokenizer=tweet_tokenizer.tokenize)
freq_tweets = vectorizer.fit_transform(tweets)
X = df_test_under['tweet']
y = df_test_under['classificacao']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, shuffle=True)
freq_train = vectorizer.transform(X_train)
freq_test = vectorizer.transform(X_test)
freq_total = vectorizer.transform(X)
np.random.seed(SEED)
final_model = LogisticRegression(solver='liblinear', random_state=1)
final_model.fit(freq_total, y)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = vectorizer.transform(data).toarray()
        my_prediction = final_model.predict(vect)
    return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)
