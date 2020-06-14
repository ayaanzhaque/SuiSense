#!/usr/bin/env python
# coding: utf-8

# # Progression Model
# This is our model for determining the progression of someone's suicidal tendencies and depression.

# ## Victoria's Diary
# This is a dataset that we will use to train our progression.
# 
# Victoria's family made the contents of her diary available to Jesse Bering, a research psychologist at the University of Otago in New Zealand. We scraped exerpts of the diary from Bering's published findings in his book Suicidal: Why We Kill Ourselves.
# 
# Applying our model to Victoria's writings allows us to see if our model -- trained on data from online communities -- would generalise well to an unseen test set. In this case, an individual's words.
# 
# Using social psychologist Roy Baumeister's theory, Bering mapped different parts of Victoria's diary to six different progressive stages from "falling short of expectations"(stage one) to "high self-awareness"(stage three) to the final stage of "disinhibition". We've matched Bering's findings to each diary exerpt in our dataset.

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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


# In[3]:


model_data = pd.read_csv('../data/data_for_model.csv', keep_default_na=False)
vics_diary = pd.read_csv('../data/vics_diary.csv', keep_default_na=False)


# In[4]:


model_data.info()


# In[5]:


vics_diary.info()


# In[6]:


vics_diary.shape


# In[7]:


vics_diary.head()


# In[8]:


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


# In[20]:


vics_diary["journ_entry_clean"] = processing_text(vics_diary["journ_entry"])
pd.set_option("display.max_colwidth", 100)
vics_diary.head(8)
print(vics_diary["vic_detail"][10])


# In[10]:


vics_diary.info()


# In[11]:


# Visualize Data
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


# In[12]:


TF_IDF_most_used_words("Words indicative of suicide in Victoria's Diary", vics_diary["journ_entry_clean"], "vlag_r", image_mask="../assets/a_victoria_mask_2.png")


# In[13]:


# using model for testing

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


# There are six stages that can be determined in Victoria's diary.
# 
# Stage 1: Falling Short of Expectations
# 
# Stage 2: Attributions to Self
# 
# Stage 3: High Self-Awareness
# 
# Stage 4: Negative Affect
# 
# Stage 5: Cognitive Deconstruction
# 
# Stage 6: Disinhibition

# In[14]:


# average predictions
vics_diary["predicted_suicide"].mean()


# In[15]:


# checking entries per stage
vics_diary["vic_detail"].value_counts()


# These are the 6 stages and the frequency specifically in Victoria's Diary. She made the most diary posts during Stage 5 and 6, right before she took her life.

# In[16]:


# plotting predictions per stage
vic_plot_df = pd.DataFrame(vics_diary.groupby("vic_detail")["predicted_suicide"].value_counts())
vic_plot_df.columns = ["counts"]
vic_plot_df = vic_plot_df.reset_index()
vic_plot_df = vic_plot_df.iloc[[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,0,1],:]
vic_plot_df


# In[17]:


# Percentage of Entries categorized as Suicidal 
pure_entries_total = vic_plot_df[vic_plot_df["vic_detail"].str.contains("Stage")]["counts"].sum()
pure_entries_predicted_suicide = vic_plot_df[vic_plot_df["vic_detail"].str.contains("Stage")][vic_plot_df[vic_plot_df["vic_detail"].str.contains("Stage")]["predicted_suicide"]==1]["counts"].sum()

pure_entries_predicted_suicide/pure_entries_total


# Only about 70 percent of her entries were classified as suicidal, proving how cryptic these messages can be.

# In[18]:


#CREATING A BARPLOT TO VISUALISE HOW THE MODEL CLASSIFIED VICTORIA'S ENTRIES
sns.set_style("white")
colors = ["dark slate blue", "dark red"]  
myPalette = sns.xkcd_palette(colors)
plt.figure(figsize = (15, 10), dpi=300)
plt.title("Classification of entries in Victoria's Diary\n", fontsize=14)
ax = sns.barplot(y='vic_detail', x='counts', data=vic_plot_df, hue='predicted_suicide', palette=myPalette, errwidth=0.01);
plt.ylabel("category of entries");

