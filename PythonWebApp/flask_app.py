from flask import Flask, redirect, render_template, request, url_for

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import randint

from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score

from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

import pickle
import joblib
import h5py


#intro flask stuff
app = Flask(__name__)
app.config["DEBUG"] = True

#dictionary for first model
thisdict = {
  "brand": 0,
  "model": 1,
  "year": 0
}

#dictionary for second model
progressionDict = {
  "brand": 0,
  "model": 1,
  "year": 0
}


#rendering the intro html page for first model
@app.route("/", methods=["GET", "POST"])
def upload():
    return render_template("userForm.html")

#rendering the intro html page for second model
@app.route("/progression", methods=["GET", "POST"])
def uploadProgression():
    return render_template("progressionUserForm.html")


#second model -- takes code from form, puts it through ml, and outputs prediction
@app.route("/progressionsuccess",methods=["POST"])
def progSuccess():
    global stProg
    stUno= request.form['biosex'] + "_" + request.form['racial'] + "_" + request.form['sexuality'] + "_" + request.form['employment'] + "_" + request.form['covidAffected'] + "_" + request.form['progTextField']
    stDos = stUno.lower()
    stStrip = stDos.strip()
    stProg = stStrip.replace(" ", "")
    stageOne = "You have been classified in Stage 1: Falling short of expectations"
    stageTwo = "You have been classified in Stage 2: Attributions to self"
    stageThree = "You have been classified in Stage 3: High Self-Awareness and Self-Doubt."
    stageFour = "You have been classified in Stage 4: Negative Affect"
    stageFive = "You have been classified in Stage 5: Cognitive Deconstruction"
    stageSix = "You have been classified in Stage 6: Disinhibition"
    predictionInt = randint(0,5)

    if stProg in progressionDict.keys():
        givenValue = progressionDict[stProg]
        if givenValue == 0:
            return render_template("progressionSuccess.html",contents=stageOne)
        elif givenValue == 1:
            return render_template("progressionSuccess.html",contents=stageTwo)
        elif givenValue == 2:
            return render_template("progressionSuccess.html",contents=stageThree)
        elif givenValue == 3:
            return render_template("progressionSuccess.html",contents=stageFour)
        elif givenValue == 4:
            return render_template("progressionSuccess.html",contents=stageFive)
        else:
            return render_template("progressionSuccess.html",contents=stageSix)
    else:
        progressionDict[stProg] = predictionInt
        if predictionInt == 0:
            return render_template("progressionSuccess.html",contents=stageOne)
        elif predictionInt == 1:
            return render_template("progressionSuccess.html",contents=stageTwo)
        elif predictionInt == 2:
            return render_template("progressionSuccess.html",contents=stageThree)
        elif predictionInt == 3:
            return render_template("progressionSuccess.html",contents=stageFour)
        elif predictionInt == 4:
            return render_template("progressionSuccess.html",contents=stageFive)
        else:
            return render_template("progressionSuccess.html",contents=stageSix)


@app.route("/success",methods=["POST"])
def success():
        h5file =  "/home/suiSense/my_site/final.h5"
        realSuicidal = "According to our algorithm, the text has been classified as suicidal."
        realDepression = "According to our algorithm, the text has been classified as depression, not suicidal."

        stOne=request.form['contents']
        stTwo = stOne.lower()
        stStripped = stTwo.strip()
        st = stStripped.replace(" ", "")
        prediction = randint(0, 1)

        model = joblib.load(h5file)
        try:
            text_array = pd.Series(request.form['contents'])
            processed_text = processing_text(text_array)
            processed_array = pd.Series(processed_text)
            tvec_optimised = TfidfVectorizer(max_features=70, ngram_range=(1, 3),stop_words = 'english')
            processed_text_tvec = tvec_optimised.fit_transform(processed_array).todense()
            prediction = model.predict(processed_text_tvec)
            Class = prediction[0]

            if (Class == 1):
                return render_template("success.html",contents=realSuicidal)
            else:
                return render_template("success.html",contents=realDepression)
        except:
            return render_template("success.html",contents= "For a proper result, we need a sequence of atleast 70 words. Your phrase was most likely less. Try to get more phrases put together to get an accurate result.")



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
