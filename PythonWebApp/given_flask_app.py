#all imports --------------------------------------------------------------------------------------------------------------------------------
from flask import Flask, redirect, render_template, request, url_for

import numpy as np
import pandas as pd
from random import randint
import joblib
import math
import requests

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

import warnings
warnings.filterwarnings('ignore')

from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import tensorflow_hub as hub
import tensorflow as tf
tf.gfile = tf.io.gfile

import bert
from bert.tokenization import FullTokenizer

from tensorflow.keras.models import Model
from tensorflow import keras


#intro flask stuff ------------------------------------------------------------------------------------------------------------------------------
app = Flask(__name__)
app.config["DEBUG"] = True

#DELETE THIS ASAP -------------------------------------------
@app.route("/testing", methods=["GET", "POST"])
def testProgressBar():
    return render_template("testProgress.html")

#BERT BINARY MODEL (/progression) ------------------------------------------------------------------------------------------------------------------------------------------------------

@app.route("/progression", methods=["GET", "POST"])
def uploadProgression():
    return render_template("progressionUserForm.html")


@app.route("/progressionsuccess",methods=["POST"])
def progSuccess():
    global stProg
    text = request.form['progTextField']
    s = text[0:512]

    final_model = keras.models.load_model('/home/suiSense/my_site/final_regular_model.h5')


    realSuicidal = "According to our algorithm, the text has been classified as suicidal."
    realDepression = "According to our algorithm, the text has been classified as depression, not suicidal."


    max_seq_length = 512  # Your choice here.
    input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                    name="segment_ids")
    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=True)
    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=[pooled_output, sequence_output])

    # See BERT paper: https://arxiv.org/pdf/1810.04805.pdf
    # And BERT implementation convert_single_example() at https://github.com/google-research/bert/blob/master/run_classifier.py


    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

    tokenizer = FullTokenizer(vocab_file, do_lower_case)

    stokens = tokenizer.tokenize(s)

    stokens = ["[CLS]"] + stokens + ["[SEP]"]

    input_ids = get_ids(stokens, tokenizer, max_seq_length)
    input_masks = get_masks(stokens, max_seq_length)
    input_segments = get_segments(stokens, max_seq_length)

    pool_embs, all_embs = model.predict([[input_ids],[input_masks],[input_segments]])

    predictions = final_model.predict(pool_embs)

    predictionPercentage = predictions[0][0] * 100

    if (0.0 < predictions <= 0.500):
        return render_template("progressionSuccess.html",contents="depression", intvar=predictionPercentage)

    elif (0.500 < predictions <= 1.000):
        return render_template("progressionSuccess.html",contents="suicidal", intvar=predictionPercentage)

    try:
        reloadWebsite()
    except:
        print('reload failed')



#BASELINE MODEL (/advancedprogression) ----------------------------------------------------------------------------
#NOTE: WE NEED TO MAKE THE BACKEND FOR THIS WORK, IT OUTPUTS SOMETHING RN BUT NOT THE CORRECT THING


@app.route("/advancedprogression", methods=["GET", "POST"])
def advancedProg():
    return render_template("advancedProgressionUserForm.html")


@app.route("/advancedprogressionsuccess",methods=["POST"])
def advancedProgSuccess():
    global stProg
    #bringing in the text
    baseline_text = request.form['baselineOne'] + ' ' + request.form['baselineTwo'] + ' ' + request.form['baselineThree']
    final_text = request.form['recentOne'] + ' ' + request.form['recentTwo'] + ' ' + request.form['recentThree']

    #text truncation for bert
    baseline_text = baseline_text[0:512]
    final_text = final_text[0:512]

    #initializing models
    final_model = keras.models.load_model('/home/suiSense/my_site/final_regular_model.h5')
    baseline_model = keras.models.load_model('/home/suiSense/my_site/baseline_model.h5')

    #bringing in the bert model to apply for all the text
    max_seq_length = 512
    input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                    name="segment_ids")
    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=True)
    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=[pooled_output, sequence_output])
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = FullTokenizer(vocab_file, do_lower_case)

    #baseline model, baseline text
    stokensOne = tokenizer.tokenize(baseline_text)
    stokensOne = ["[CLS]"] + stokensOne + ["[SEP]"]
    input_ids = get_ids(stokensOne, tokenizer, max_seq_length)
    input_masks = get_masks(stokensOne, max_seq_length)
    input_segments = get_segments(stokensOne, max_seq_length)
    pool_embs, all_embs = model.predict([[input_ids],[input_masks],[input_segments]])
    fxOne = baseline_model.predict(pool_embs)
    fxOne = fxOne[0][0]

    #baseline model, suicidal text
    stokensTwo = tokenizer.tokenize(final_text)
    stokensTwo = ["[CLS]"] + stokensTwo + ["[SEP]"]
    input_ids = get_ids(stokensTwo, tokenizer, max_seq_length)
    input_masks = get_masks(stokensTwo, max_seq_length)
    input_segments = get_segments(stokensTwo, max_seq_length)
    pool_embs, all_embs = model.predict([[input_ids],[input_masks],[input_segments]])
    fxTwo = baseline_model.predict(pool_embs)
    fxTwo = fxTwo[0][0]

    #suicidal model, baseline text
    stokensThree = tokenizer.tokenize(baseline_text)
    stokensThree = ["[CLS]"] + stokensThree + ["[SEP]"]
    input_ids = get_ids(stokensThree, tokenizer, max_seq_length)
    input_masks = get_masks(stokensThree, max_seq_length)
    input_segments = get_segments(stokensThree, max_seq_length)
    pool_embs, all_embs = model.predict([[input_ids],[input_masks],[input_segments]])
    gxOne = final_model.predict(pool_embs)
    gxOne = gxOne[0][0]

    #suicidal model, suicidal text
    stokensFour = tokenizer.tokenize(final_text)
    stokensFour = ["[CLS]"] + stokensFour + ["[SEP]"]
    input_ids = get_ids(stokensFour, tokenizer, max_seq_length)
    input_masks = get_masks(stokensFour, max_seq_length)
    input_segments = get_segments(stokensFour, max_seq_length)
    pool_embs, all_embs = model.predict([[input_ids],[input_masks],[input_segments]])
    gxTwo = final_model.predict(pool_embs)
    gxTwo = gxTwo[0][0]

    if (fxTwo > 0.5):
        if (fxOne > 0.5):
            predictionPercentage = (gxTwo - gxOne) * 100
        else:
            predictionPercentage = ((gxTwo + 1) - fxOne) * 100
    else:
        if (fxOne > 0.5):
            predictionPercentage = (fxTwo - (gxOne + 1)) * 100
        else:
            predictionPercentage = (fxTwo - fxOne) * 100

    significant_digits = 3
    predictionPercentage = round(predictionPercentage, significant_digits - int(math.floor(math.log10(abs(predictionPercentage)))) - 1)

    absPredictionPercentage = abs(predictionPercentage)

    return render_template("advancedProgressionSuccess.html", intvar=predictionPercentage, absintvar=absPredictionPercentage, fxOne=fxOne, fxTwo=fxTwo, gxOne=gxOne, gxTwo=gxTwo)


    try:
        reloadWebsite()
    except:
        print('reload failed')



#BERT PROGRESSION MODEL (/realprogression) --------------------------------------------------------------------

@app.route("/realprogression", methods=["GET", "POST"])
def realProgression():
    return render_template("realprogression.html")


@app.route("/realprogressionsuccess",methods=["POST"])
def realProgressionSuccess():
    global stProg
    text = request.form['progTextFieldOne'] + request.form['progTextFieldTwo'] + request.form['progTextFieldThree']
    s = text[0:512]

    final_model = keras.models.load_model('/home/suiSense/my_site/progression_model.h5')

    max_seq_length = 512  # Your choice here.
    input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                    name="segment_ids")
    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=True)
    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=[pooled_output, sequence_output])

    # See BERT paper: https://arxiv.org/pdf/1810.04805.pdf
    # And BERT implementation convert_single_example() at https://github.com/google-research/bert/blob/master/run_classifier.py


    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

    tokenizer = FullTokenizer(vocab_file, do_lower_case)

    stokens = tokenizer.tokenize(s)

    stokens = ["[CLS]"] + stokens + ["[SEP]"]

    input_ids = get_ids(stokens, tokenizer, max_seq_length)
    input_masks = get_masks(stokens, max_seq_length)
    input_segments = get_segments(stokens, max_seq_length)

    pool_embs, all_embs = model.predict([[input_ids],[input_masks],[input_segments]])

    predictions = final_model.predict(pool_embs)

    predictionPercentage = predictions[0][0] * 100

    if (0.0 < predictions <= 0.3):
        return render_template("realprogressionsuccess.html",contents='1', intvar=predictionPercentage)

    elif (0.3 < predictions <= 0.375):
        return render_template("realprogressionsuccess.html",contents='2', intvar=predictionPercentage)

    elif (0.375 < predictions <= 0.45):
        return render_template("realprogressionsuccess.html",contents='3', intvar=predictionPercentage)

    elif (0.45 < predictions <= 0.525):
        return render_template("realprogressionsuccess.html",contents='4', intvar=predictionPercentage)

    elif (0.525 < predictions <= 0.6):
        return render_template("realprogressionsuccess.html",contents='5', intvar=predictionPercentage)

    elif (0.6 < predictions <= 1.0):
        return render_template("realprogressionsuccess.html",contents='6', intvar=predictionPercentage)

    try:
        reloadWebsite()
    except:
        print('reload failed')



#OLD MODEL THAT IS NOW DEPRECIATED ---------------------------------------------------------------------------------------------------------------------

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


#all general functions --------------------------------------------------------------------------------------------------------------------------------------------------------

def get_masks(tokens, max_seq_length):
        """Mask for padding"""
        if len(tokens)>max_seq_length:
            raise IndexError("Token length more than max seq length!")
        return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))


def get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))


def get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids

def reloadWebsite():
    username = 'suiSense'
    token = 'ac02c511cd4bdeda55e51531167588d96034760a'

    response = requests.get(
        'https://www.pythonanywhere.com/api/v0/user/{username}/reload/'.format(
            username=username
        ),
        headers={'Authorization': 'Token {token}'.format(token=token)}
    )

    if response.status_code == 200:
        successreturn = 'reloaded OK'
        return successreturn
    else:
        failreturn = 'reload failed'
        return failedreturn
