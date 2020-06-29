[![Netlify Status](https://api.netlify.com/api/v1/badges/113e244d-b901-4ac2-95c2-77977458bf9d/deploy-status)](https://app.netlify.com/sites/suisense/deploys)

# [SuiSense](https://suisense.space/)

**2nd place overall @ Geom Hacks**
**Honorable Mention @ MLH Summer League SHDH 2020**

Using NLP to distinguish suicidal messages and provide personalized support.

## Overview of our Project

SuiSense is a unique progressive web application that allows concerned family and friends to determine whether their struggling loved one is on the path to suicide. The core of our project is a Natural Language Processing model(sci-kit) that classifies a phrase someone says as representing suicidal tendencies, depression, or neither. Users can input text from their messaging conversations, memory of their in-person conversations, letters, diary entries, or screenshots. Classifying between suicide and depression is important because the implications and methods for support are completely different, but determining the difference is a precise task that is best dealt with through ML. 

After our model determines a primary classification, we then feed data into our progression model, which also uses NLP. Psychologist Jesse Bering discovered six stages on the path from depression to suicide, and our model accurately assesses what stage a patient is at and the severity of their depression. We trained this model on metadata specific to COVID, including employment status, health problems, how the person was affected by COVID, and more. During COVID, the progression of these stages happens at a much faster rate, meaning detecting which stage a patient is at is urgent. 

Next, our framework allows friends and family to find the best therapists to address their specific needs. Given that COVID has taken away in-person interactions and other forms of meeting therapists, our Javascript algorithm gives quick and easy recommendations to help struggling people find therapists. We also have local therapists and support centers displayed on a map.

Lastly, we have a support page that allows therapists and patients to post on a forum, similar to a blog, creating a safe community for those who feel lonely and isolated. Since COVID has forced people online, creating online forums is a redesigned way to cope with loneliness and provide a great opportunity for regular people to communicate with certified therapists about different problems. 

This has been SuiSense, and we hope to have a strong impact on mental health during the COVID Crisis. 

