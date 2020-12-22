[![Netlify Status](https://api.netlify.com/api/v1/badges/113e244d-b901-4ac2-95c2-77977458bf9d/deploy-status)](https://app.netlify.com/sites/suisense/deploys)

# [SuiSense](https://suisense.space/)

Using Artificial Intelligence to distinguish between suicidal and depressive messages.

- **4th Place @ Congressional App Challenge 2020**

- **2nd Place Overall @ GeomHacks 2020**

- **Honorable Mention @ MLH Summer League SHDH 2020**

Demo: https://www.youtube.com/watch?v=QHpKJBVObhA

Medium Article: (https://codeburst.io/suisense-an-innovative-approach-to-suicide-prevention-19cbdf150575)

## Overview of our Project

SuiSense is a progressive web application that uses Artificial Intelligence (AI) and Natural Language Processing (NLP) to distinguish between depressive and suicidal phrases and help concerned friends and family determine whether their struggling loved one is on the path to suicide. 

SuiSense provides 4 key services to do so. The first, our Initial Screening model, determines
whether provided phrases are representative of depressive or suicidal tendencies. Friends and
family can input concerning texts and receive a classification on whether they are depressive or
suicidal. Especially during the pandemic, it is essential to utilize online messages to classify
patients. An algorithm classifying depression versus suicide is incredibly important, as treatment
methods differ significantly. Second, our Baseline Screening allows users to determine how much
someone has progressed towards suicidality. Users upload 3 messages before and after a change
was noticed, and our algorithm calculates the percent change towards suicidality based on the
texts, which allows for direct comparison between the messages. Thirdly, our Progression
Screening classifies depression and suicide based on psychologist Jesse Bering’s 6 stages of
depression, a standard psychology metric. Our model fills a vital gap; there is no research in stage-based depression analysis
despite its demand. Users upload concerning texts and discover which stage their loved one is on. Finally, our support page allows users to upload key information about their loved one so they can
be paired with relevant therapists.

Our algorithms are trained on thousands of social media posts, specifically from the subreddits r/SuicideWatch, r/Depression, and r/CasualConversations. Field research has proven that people are likely to express their feelings on anonymous platforms, making it a great place to access data. After cleaning and labeling, we input it into Google’s BERT transformer, an NLP neural network that outputs word embeddings with the context of sentences. We trained three different Keras neural networks on three datasets. Facebook’s algorithm achieves 72% accuracy, but due to our unique approach, all our models range from 81% to 90% accuracy, proving their viability and effectiveness.
