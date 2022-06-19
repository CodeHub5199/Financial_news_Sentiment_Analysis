import streamlit as st
import pickle
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import string
import requests
import json
import plotly.graph_objects as go

ps = PorterStemmer()
nltk.download('punkt')
nltk.download('stopwords')

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return ' '.join(y)


tfidf = pickle.load(open('vectorizer_tfidf.pkl', 'rb'))
model = pickle.load(open('model_log.pkl', 'rb'))
r = requests.get('https://newsapi.org/v2/top-headlines?country=in&category=business&apiKey=7c053139fa444fd1b03371509b60b9b2')
# r = requests.get('https://newsapi.org/v2/top-headlines?country=in&apiKey=7c053139fa444fd1b03371509b60b9b2')

select = st.sidebar.radio('Select one: ',['Get Headlines with Sentiment', 'Custom News Analysis'])
st.header('Financial News Sentiment Analysis')
if select == 'Get Headlines with Sentiment':
    data = json.loads(r.content)
    positive = []
    negative = []

    for z in range(20):
        news = data['articles'][z]['title']
        news_input = news
        transformed_msg = transform_text(news_input)
        vector_input = tfidf.transform([transformed_msg])
        result = model.predict(vector_input)[0]
        st.write(z+1, '. ', news)
        if result == 0:
            st.write('Predicted Sentiment: Negative')
            negative.append(result)
        else:
            st.write('Predicted Sentiment: Positive')
            positive.append(result)

    sentiment_score = len(positive)
    percentage = (100*sentiment_score)/20
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        number = {'suffix': "%"},
        value = percentage,
        gauge={
            'axis':{'range':[0,100]},
            'bar':{'color':'lightsteelblue'},
            'steps':[
                {'range':[0,49],'color':'tomato'},
                {'range':[51,100],'color':'lime'},
                {'range':[49,51],'color':'white'}
            ]},
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Sentiment Indicator"}))
    st.plotly_chart(fig)
    st.write('0% - 49% : Negative Sentiment')
    st.write('49% - 51% : Neutral Sentiment')
    st.write('51% - 100% : Positive Sentiment')


if select == 'Custom News Analysis':
    news_input = st.text_area('Write a News: ')
    if st.button('Predict'):
        transformed_msg = transform_text(news_input)
        vector_input = tfidf.transform([transformed_msg])
        result = model.predict(vector_input)[0]
        if result == 0:
            st.write('Predicted Sentiment: Negative')
        else:
            st.write('Predicted Sentiment: Positive')
