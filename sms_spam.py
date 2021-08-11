import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time
@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def load_model_token():    
    bar=st.progress(0)
    process=25
    tokenizer=pd.read_pickle(r"/app/sms_span_detection/tokenizer.pkl")
    bar.progress(process+25)
    model=tf.keras.models.load_model(r"/app/sms_span_detection/spam_model")
    bar.progress(process+75)
    time.sleep(0.01)
    bar.empty()
    return model,tokenizer

model,tokenizer=load_model_token()

max_length = 8
st.title("SMS Span Prediction")
st.markdown("************")
input=st.text_input("Enter the sms: ")
sms=[input]
sms_proc=tokenizer.texts_to_sequences(sms)
sms_proc = pad_sequences(sms_proc, maxlen=max_length, padding='post')
st.write("")
submit=st.button("Hit me")

if submit:
    pred = (model.predict(sms_proc) > 0.5).astype("int32").item()
    if pred==1:
        st.write("This is a Spam Message")
    else:
        st.write("This is Han Message")
