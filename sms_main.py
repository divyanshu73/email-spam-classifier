import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms=st.text_area("Enter the Message")

def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    #print(text)
    y=[]
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            if i.isalnum():
                y.append(ps.stem(i))
                    
        
    return " ".join(y)

if st.button('predict'):
    transformed_sms=transform_text(input_sms)

    vector_input=tfidf.transform([transformed_sms])

    result=model.predict(vector_input)[0]

    if result==1:
        st.header('Spam')
    else:
        st.header('Not Spam')
