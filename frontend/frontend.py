import streamlit as st
import plotly.graph_objects as go
from keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
import pyttsx3
import time



# Laden der Imports für die Speech to Text Funktion
import speech_recognition as sr
from time import ctime # get time details
import time
import playsound
import os
import random
from gtts import gTTS

#from python_scripts.transcript_creation import modell_speak, record_audio

##############################################
# Laden der Imports für die Zusammenfassung
import re
import networkx as nx
import spacy
from summa import keywords
from summa.summarizer import summarize

# Laden des Spacy-Modells
import evaluate
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer,PreTrainedTokenizerFast
from collections import Counter
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from transformers import pipeline
import torch
from deepmultilingualpunctuation import PunctuationModel

from python_scripts.zusammenfassung import execute_text_gen
##############################################


############ Laden der Modelle ###############
model_text_korrigieren = PunctuationModel()
## Laden des Modells für die Klassifikation
model_classification = load_model("classification/neuro_net_1.h5")
vectorizer = joblib.load("classification/vectorizer_1.joblib")

## Laden des Modells für die Zusammenfassung
for model_name in ['NICFRU/bart-base-paraphrasing-science','NICFRU/bart-base-paraphrasing-news','NICFRU/bart-base-paraphrasing-story','NICFRU/bart-base-paraphrasing-review']:
    tokenizer_zusammenfassung = AutoTokenizer.from_pretrained(model_name)
    paraphrase_zusammenfassung = pipeline("text2text-generation", model=model_name)


##############################################

# Schriftgröße für Buttons anpassen
st.write('<style>div.row-widget button{font-size: 30px !important;}</style>', unsafe_allow_html=True)

# Schriftgröße für Text Area anpassen
st.write('<style>div.row-widget textarea{font-size: 35px !important;}</style>', unsafe_allow_html=True)

all_texts = "Syntex. Please input your text here or choose a file to upload. Activate either the classification or the summarization by checking the boxes, and if needed, choose a compression ratio of the text. Then click on the last button, to get your desired result."

# engine = pyttsx3.init()
# engine.setProperty('rate', 150)  # Anpassen der Sprechgeschwindigkeit (optional)

# def text_to_speech(text):
#     global engine
#     engine.say(text)
#     engine.runAndWait()  

def extract_text_from_element(element):
    # Extrahiere den Text aus einem Streamlit-Element
    if hasattr(element, 'text'):
        return element.text
    elif isinstance(element, (list, tuple)):
        return ' '.join([extract_text_from_element(item) for item in element])
    elif isinstance(element, dict):
        return ' '.join([extract_text_from_element(item) for item in element.values()])
    else:
        return str(element)
    
def extract_text_from_page(page):
    # Extrahiere den Text aus einer Streamlit-Seite
    text = ''
    for element in page:
        text += extract_text_from_element(element)
    return text

# # Texteingabe
# text = st.text_input("Geben Sie den Text ein:")

# # Button zum Auslösen der Sprachausgabe
# if st.button("Text vorlesen"):
#     text_to_speech(text)

def read_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in range(len(pdf.pages) ):
        text += pdf.pages[page].extract_text()
    return text

def read_docx(file):
    doc = Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def read_txt(file):
    with open(file.name, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def summarize_text(text, compression_rate,predicted_class):
    # Hier erfolgt die Zusammenfassung des Textes unter Berücksichtigung der Kompressionsrate
    # Implementiere hier deine Logik zur Zusammenfassung des Textes
    dic=execute_text_gen({'classification':predicted_class,'text':text,'compression':compression_rate*0.01})
    paraphrase_zusammenfassung = dic['text_rank_text_2']
    text_compression_rate=round(dic['ent_com_rate']*100,2)
    print('Text: ',paraphrase_zusammenfassung,'Kompression: ' ,text_compression_rate)
    return paraphrase_zusammenfassung, text_compression_rate

    #return summarized_text

def classify_text(text):
    new_text_features = vectorizer.transform([text])
    predictions = model_classification.predict(new_text_features.toarray())
    predicted_class = np.argmax(predictions, axis=1)
    predicted_probability = np.max(predictions, axis=1)
    return predicted_class, predicted_probability

# Streamlit-App
def main():
    
    # Führe eine Wartezeit von wait_time Sekunden durch
    time.sleep(20)

    # Texteingabe
    text = "Welcome to Syntex, to proceed further with the screenreader function, tap the big red botton on the left in the middle of your screen"
    #text_to_speech(text)

    st.title("SynTex")

    # Texteingabe
    input_text = st.text_area("Enter a text", "")
    
    # Dokument auswählen
    file = st.file_uploader("PLease select a doucment", type=["pdf", "docx", "txt"])


    # Add custom CSS style
    st.markdown(
        """
        <style>
        .red-button {
            background-color: red;
            color: white;
            padding: 0.5rem 2rem;
            font-size: 1.5rem;
            border-radius: 0.5rem;
            border: none;
            cursor: pointer;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Display the red button using custom CSS class
    button_clicked = st.markdown(
        """
        <button class="red-button">Screen Reader</button>
        """,
        unsafe_allow_html=True
    )
    #if button_clicked:
        #text_to_speech(all_texts)
    

    if file is not None:
        content = ""

        file_type = file.type
        if file_type == 'application/pdf':
            content = read_pdf(file)
        elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            content = read_docx(file)
        elif file_type == 'text/plain':
            content = read_txt(file)

        st.header("Inhalt des Dokuments")
        input_text = st.text_area("Ausgabe", value=content, height=500)
    
    # Checkboxen für Klassifikation und Zusammenfassung
    classification_enabled = st.checkbox("Activate classification")
    summarization_enabled = st.checkbox("Activate summary")
    
    # Schieberegler für die Kompressionsrate
    compression_rate = st.slider("Compression rate (%)", min_value=0, max_value=100, value=50, step=1)
    
    # Knopf zum Zusammenfassen und Klassifizieren
    if st.button("Execute"):
        if input_text:
            
            if classification_enabled:
                predicted_class, predicted_probability = classify_text(input_text)
                predicted_probability = round(predicted_probability[0]*100, 2)
                if predicted_class == 0:
                    predicted_class = "Scientific Paper"
                elif predicted_class == 1:
                    predicted_class = "News"
                elif predicted_class == 2:
                    predicted_class = "Review"
                else:
                    predicted_class = "Story"
                st.subheader("Classification")

                if predicted_probability >= 95:
                    st.write(f"The class calculated by the model is:  \"{predicted_class}\". The model is very confident with a probability of {predicted_probability}%.")
                    if summarization_enabled:
                        paraphrase_zusammenfassung, text_compression_rate = summarize_text(input_text, compression_rate,predicted_class)
                        st.subheader("Summary")
                        st.write("This is a summary of the input text with a compression rate of {}%:".format(text_compression_rate))
                        st.write(paraphrase_zusammenfassung)
                else:
                    st.write(f"Attention! The model is not entirely certain. The class calculated by the model is: \"{predicted_class}\". However, the model predicts this class with a probability of only {predicted_probability}%.")
        else:
            st.warning("Please enter a text first.")



if __name__ == '__main__':
    main()

