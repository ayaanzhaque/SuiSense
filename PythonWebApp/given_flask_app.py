#all imports --------------------------------------------------------------------------------------------------------------------------------
from flask import Flask, redirect, render_template, request, url_for

import numpy as np
import pandas as pd
from random import randint
import joblib

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
    s = text[0:128]

    final_model = keras.models.load_model('/home/suiSense/my_site/final_regular_model.h5')
    realSuicidal = "According to our algorithm, the text has been classified as suicidal."
    realDepression = "According to our algorithm, the text has been classified as depression, not suicidal."


    max_seq_length = 128  # Your choice here.
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



#BASELINE MODEL (/advancedprogression) ----------------------------------------------------------------------------
#NOTE: WE NEED TO MAKE THE BACKEND FOR THIS WORK, IT OUTPUTS SOMETHING RN BUT NOT THE CORRECT THING


@app.route("/advancedprogression", methods=["GET", "POST"])
def advancedProg():
    return render_template("advancedProgressionUserForm.html")


@app.route("/advancedprogressionsuccess",methods=["POST"])
def advancedProgSuccess():
    global stProg
    text = request.form['baselineOne'] + request.form['baselineTwo'] + request.form['baselineThree']
    s = text[0:128]

    final_model = keras.models.load_model('/home/suiSense/my_site/final_regular_model.h5')
    realSuicidal = "According to our algorithm, the text has been classified as suicidal."
    realDepression = "According to our algorithm, the text has been classified as depression, not suicidal."


    max_seq_length = 128  # Your choice here.
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

    return render_template("advancedProgressionSuccess.html", intvar=predictionPercentage)



#BERT PROGRESSION MODEL (/realprogression) --------------------------------------------------------------------

@app.route("/realprogression", methods=["GET", "POST"])
def realProgression():
    return render_template("realprogression.html")


@app.route("/realprogressionsuccess",methods=["POST"])
def realProgressionSuccess():
    global stProg
    text = request.form['progTextFieldOne'] + request.form['progTextFieldTwo'] + request.form['progTextFieldThree']
    s = text[0:128]

    final_model = keras.models.load_model('/home/suiSense/my_site/final_regular_model.h5')


    max_seq_length = 128  # Your choice here.
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

    if (0.0 < predictions <= 0.166):
        return render_template("realprogressionsuccess.html",contents='1', intvar=predictionPercentage)

    elif (0.166 < predictions <= 0.333):
        return render_template("realprogressionsuccess.html",contents='2', intvar=predictionPercentage)

    elif (0.333 < predictions <= 0.500):
        return render_template("realprogressionsuccess.html",contents='3', intvar=predictionPercentage)

    elif (0.500 < predictions <= 0.667):
        return render_template("realprogressionsuccess.html",contents='4', intvar=predictionPercentage)

    elif (0.667 < predictions <= 0.833):
        return render_template("realprogressionsuccess.html",contents='5', intvar=predictionPercentage)

    elif (0.833 < predictions <= 1.000):
        return render_template("realprogressionsuccess.html",contents='6', intvar=predictionPercentage)



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


@app.route("/", methods=["GET", "POST"])
def upload():
    return render_template("userForm.html")


@app.route("/success",methods=["POST"])
def success():
        h5file =  "/home/suiSense/my_site/final.h5"
        realSuicidal = "According to our algorithm, the text has been classified as suicidal."
        realDepression = "According to our algorithm, the text has been classified as depressive."

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
            return render_template("success.html",contents= "For a proper result, we need a sequence of at least 70 words. Your phrase was most likely less. Try to get more phrases put together to get an accurate result.")



#all general functions --------------------------------------------------------------------------------------------------------------------------------------------------------


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
