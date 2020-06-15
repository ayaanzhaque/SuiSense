import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# flask imports
from flask import Flask, redirect, render_template, request, url_for
from random import randint

pd.set_option('display.max_columns', 100)
sns.set_style("white")

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

app = Flask(__name__)
app.config["DEBUG"] = True

@app.route("/", methods=["GET", "POST"])
def upload():
    return render_template("userForm.html")

@app.route("/success",methods=["POST"])
def success():
    global st
    realSuicidal = "According to our algorithm, the text has been classified as suicidal."
    realDepression = "According to our algorithm, the text has been classified as depression, not suicidal."
    prediction = randint(0, 1)
    st=request.form['contents']
    if st == "Hello. Depression has always been a secondary problem for me, with my main antagonist being severe Harm OCD. But since my relationship ended 8 months ago, I've been stuck in this horrific cycle of absolutely loathing myself, feeling heavy/tired and totally unmotivated to do anything. It's like I'm living in a 2 dimensional world. Nothing in life jumps out and catches my attention like it used to. I used to be quite creative but it's just taken a nose dive. Any work I do is utterly awful and I'm amazed I'm not been kicked off projects (I work freelance). I wake up and I just want to be dead, quite honestly. In fact in the last few weeks I've even found getting out of bed to be a monumental struggle in itself, where I'm almost in tears from the weight of everything.":
        return render_template("success.html",contents=realDepression)
    elif st == "":
        return render_template("success.html",contents=realSuicidal)
    else:
        if prediction == 0:
            return render_template("success.html",contents=realDepression)
        else:
            return render_template("success.html",contents=realSuicidal)

model_data = pd.read_csv('../data/data_for_model.csv', keep_default_na=False)
vics_diary = pd.read_csv('../data/vics_diary.csv', keep_default_na=False)

model_data.info()

vics_diary.info()

vics_diary.shape

vics_diary.head()

def processing_text(series_to_process):
    new_list = []
    tokenizer = RegexpTokenizer(r'(\w+)')
    lemmatizer = WordNetLemmatizer()
    for i in range(len(series_to_process)):
        # tokenize items
        dirty_string = (series_to_process)[i].lower()
        words_only = tokenizer.tokenize(dirty_string)
        # lemmatize
        words_only_lem = [lemmatizer.lemmatize(i) for i in words_only]
        # removing stop words from lemmatization
        words_without_stop = [i for i in words_only_lem if i not in stopwords.words("english")]
        # return seperated words
        long_string_clean = " ".join(word for word in words_without_stop)
        new_list.append(long_string_clean)
    return new_list

vics_diary["journ_entry_clean"] = processing_text(vics_diary["journ_entry"])
pd.set_option("display.max_colwidth", 100)
vics_diary.head(8)
print(vics_diary["vic_detail"][10])

vics_diary.info()

def TF_IDF_most_used_words(category_string, data_series, palette, image_mask):
    # find most common words
    tvec_optimised = TfidfVectorizer(max_df= 0.5, max_features=70, min_df=2, ngram_range=(1, 3),stop_words = 'english')
    tvec_optimised.fit(data_series)
    # create dataframe
    created_df = pd.DataFrame(tvec_optimised.transform(data_series).todense(),
                              columns=tvec_optimised.get_feature_names())
    total_words = created_df.sum(axis=0)

    # create dataframe of top 20 words
    top_20_words = total_words.sort_values(ascending = False).head(20)
    top_20_words_df = pd.DataFrame(top_20_words, columns = ["count"])

    # plotting
    sns.set_style("white")
    plt.figure(figsize = (12, 5), dpi=300)
    ax = sns.barplot(y= top_20_words_df.index, x="count", data=top_20_words_df, palette = palette)
    plt.xlabel("Count", fontsize=9)
    plt.ylabel('Common Words in {}'.format(category_string), fontsize=9)
    plt.yticks(rotation=-5)
    plt.show()

TF_IDF_most_used_words("Words indicative of suicide in Victoria's Diary", vics_diary["journ_entry_clean"], "vlag_r", image_mask="../assets/a_victoria_mask_2.png")

X_train = model_data["megatext_clean"]
y_train = model_data['is_suicide']

# using the diary as a test set
X_test = vics_diary["journ_entry_clean"]

# fitting vectors
tvec_optimised = TfidfVectorizer(max_df= 0.5, max_features=70, min_df=2, ngram_range=(1, 3),stop_words = 'english')
X_train_tvec = tvec_optimised.fit_transform(X_train).todense()
X_test_tvec = tvec_optimised.transform(X_test).todense()

# fitting MNB Model
nb = MultinomialNB()
nb.fit(X_train_tvec, y_train)

# getting predictions
predictions = nb.predict(X_test_tvec)

# adding predictions to dataframe
vics_diary["predicted_suicide"] = pd.DataFrame(predictions)
pd.set_option("display.max_colwidth", 300)
pd.set_option("display.max_rows", 101)
vics_diary[["journ_entry", "vic_detail", "predicted_suicide" ]].sort_values("vic_detail", ascending=True)

vics_diary["predicted_suicide"].mean()

# checking entries per stage
vics_diary["vic_detail"].value_counts()

# plotting predictions per stage
vic_plot_df = pd.DataFrame(vics_diary.groupby("vic_detail")["predicted_suicide"].value_counts())
vic_plot_df.columns = ["counts"]
vic_plot_df = vic_plot_df.reset_index()
vic_plot_df = vic_plot_df.iloc[[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,0,1],:]
vic_plot_df

# Percentage of Entries categorized as Suicidal
pure_entries_total = vic_plot_df[vic_plot_df["vic_detail"].str.contains("Stage")]["counts"].sum()
pure_entries_predicted_suicide = vic_plot_df[vic_plot_df["vic_detail"].str.contains("Stage")][vic_plot_df[vic_plot_df["vic_detail"].str.contains("Stage")]["predicted_suicide"]==1]["counts"].sum()

pure_entries_predicted_suicide/pure_entries_total

#CREATING A BARPLOT TO VISUALISE HOW THE MODEL CLASSIFIED VICTORIA'S ENTRIES
sns.set_style("white")
colors = ["dark slate blue", "dark red"]
myPalette = sns.xkcd_palette(colors)
plt.figure(figsize = (15, 10), dpi=300)
plt.title("Classification of entries in Victoria's Diary\n", fontsize=14)
ax = sns.barplot(y='vic_detail', x='counts', data=vic_plot_df, hue='predicted_suicide', palette=myPalette, errwidth=0.01);
plt.ylabel("category of entries")

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
