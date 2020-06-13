from flask import *
from datetime import *
from pytz import *

def get_pst_time():
    date_format='%m/%d/%Y %H:%M:%S'
    date = datetime.now(tz=utc)
    date = date.astimezone(timezone('US/Pacific'))
    pstDateTime=date.strftime(date_format)
    return pstDateTime


import os

app = Flask(__name__)
app.config["DEBUG"] = True

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

# def get_filenames():
#     global path
#     path = r"test"
#     return os.listdir(path)

comments = []
predictions = []

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("userForm.html", comments=comments)

    comments.append(request.form["contents"])
    return redirect(url_for('index'))


def success():
    if request.method == "GET":
        f = request.files['file']
        f.save(f.filename)
        h5file =  "/home/suiSense/mysite/model.h5"
        with h5py.File(h5file,'r') as fid:
            model = load_model(fid)
            Class = prediction(model, f.filename)
            diagnoses.clear()
            if (Class == 0):
                today = str(get_pst_time())
                predictions.append(today + ": According to our algorithm, it is likely that you have glaucoma. Please contact a medical professional as soon as possible for advice.")
                return f.filename + ": It is likely that you have glaucoma. Please contact a medical professional as soon as possible for advice."
            else:
                today = str(get_pst_time())
                predictions.append(today + ": According to our algorithm, you do not have glaucoma! If you have further questions, please contact a medical professional.")
                return f.filename + ": According to our algorithm, you do not have glaucoma! If you have further questions, please contact a medical professional."

def prediction(m, file):
    # list_of_files = glob.glob('data/test/*')
    # latest_file = max(list_of_files, key=os.path.getctime)
    img = cv2.imread(file)
    img = autoroi(img)
    img = cv2.resize(img, (256, 256))
    img = np.reshape(img, [1, 256, 256, 3])

    Class = m.predict(img)
    Class = prob.argmax(axis=-1)

    return(Class)
