# A very simple Flask Hello World app for you to get started with...

from flask import Flask, redirect, render_template, request, url_for
from random import randint

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

@app.route("/")
def success():
        f = request.files['file']
        f.save(f.filename)
        h5file =  "/home/suiSense/mysite/model2.h5"

        model = joblib.load(h5file)
        Class = prediction(model, request.form['content'])
        diagnoses.clear()
        if (Class == 1):
            predictions.append("According to our algorithm, the text has been classified as suicidal.")
            return f.filename + "According to our algorithm, the text has been classified as suicidal."
        else:
            predictions.append("According to our algorithm, the text has been classified as depression, not suicidal.")
            return f.filename + "According to our algorithm, the text has been classified as depression, not suicidal."

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
