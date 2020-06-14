
# A very simple Flask Hello World app for you to get started with...

from flask import Flask, redirect, render_template, request, url_for
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config["DEBUG"] = True

# model imports
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
import pickle
import joblib

# NLP Imports
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

text = []
predictions = []

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("userForm.html", predictions=predictions)

    else:
        text.append(request.form["contents"])
        success()
        return redirect(url_for('index'))



def success():
    if request.method == "GET":
        f = request.files['file']
        f.save(f.filename)
        h5file =  "/home/suiSense/mysite/model2.h5"

        model = joblib.load(h5file)
        Class = prediction(model, text)
        if (Class == 1):
            predictions.append("According to our algorithm, the text has been classified as suicidal.")
            return f.filename + ": According to our algorithm, the text has been classified as suicidal."
        else:
            predictions.append("According to our algorithm, the text has been classified as depression, not suicidal.")
            return f.filename + ": According to our algorithm, the text has been classified as depression, not suicidal."

def prediction(model, text):
    text_array = pd.Series(text)
    processed_text = processing_text(text_array)

    processed_array = pd.Series(processed_text)

    tvec_optimised = TfidfVectorizer(max_features=70, ngram_range=(1, 3),stop_words = 'english')
    processed_text_tvec = tvec_optimised.fit_transform(processed_array).todense()

    prediction = model.predict(processed_text_tvec)

    return(prediction[0])

def processing_text(series_to_process):
    new_list = []
    tokenizer = RegexpTokenizer(r'(\w+)')
    lemmatizer = WordNetLemmatizer()

    for i in range(len(series_to_process)):
        # tokenized item in a new list
        dirty_string = (series_to_process)[i].lower()
        words_only = tokenizer.tokenize(dirty_string) # words_only is a list of only the words, no punctuation
        #Lemmatize the words_only
        words_only_lem = [lemmatizer.lemmatize(i) for i in words_only]
        # removing stop words
        words_without_stop = [i for i in words_only_lem if i not in stopwords.words("english")]
        # return seperated words
        long_string_clean = " ".join(word for word in words_without_stop)
        new_list.append(long_string_clean)
        return new_list
