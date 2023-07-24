import streamlit as st
import plotly.graph_objects as go
from keras.models import load_model

import joblib
from joblib import load
import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
import time

from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events


# Laden der Imports für die Speech to Text Funktion
import speech_recognition as sr
from time import ctime # get time details
import time
import playsound
import os
import random
from gtts import gTTS

from python_scripts.transcript_creation import modell_speak, record_audio

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
vectorizer = load("classification/vectorizer_1.joblib")

## Laden des Modells für die Zusammenfassung
for model_name in ['NICFRU/bart-base-paraphrasing-science','NICFRU/bart-base-paraphrasing-news','NICFRU/bart-base-paraphrasing-story','NICFRU/bart-base-paraphrasing-review']:
    tokenizer_zusammenfassung = AutoTokenizer.from_pretrained(model_name)
    paraphrase_zusammenfassung = pipeline("text2text-generation", model=model_name)


##############################################

# Schriftgröße für Buttons anpassen
st.write('<style>div.row-widget button{font-size: 30px !important;}</style>', unsafe_allow_html=True)

# Schriftgröße für Text Area anpassen
st.write('<style>div.row-widget textarea{font-size: 35px !important;}</style>', unsafe_allow_html=True)

m = st.markdown("""
<style>
div.stButton > button:nth-of-type(2) {
    background-color: #ce1126;
    color: white;
    height: 3em;
    width: 12em;
    border-radius:10px;
    border:3px solid #000000;
    font-size:20px;
    font-weight: bold;
    margin: auto;
    display: block;
}

div.stButton > button:hover {
	background:linear-gradient(to bottom, #ce1126 5%, #ff5a5a 100%);
	background-color:#ce1126;
}
    /* 1st button */
    .element-container:nth-child(3) {
      left: 10px;
      top: -60px;
    }
div.stButton > button:active {
	position:relative;
	top:3px;
}

</style>""", unsafe_allow_html=True)
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


# Button zum Auslösen der Sprachausgabe


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
def style_button_row(clicked_button_ix, n_buttons):
    def get_button_indices(button_ix):
        return {
            'nth_child': button_ix,
            'nth_last_child': n_buttons - button_ix + 1
        }

    clicked_style = """
    div[data-testid*="stHorizontalBlock"] > div:nth-child(%(nth_child)s):nth-last-child(%(nth_last_child)s) button {
        border-color: rgb(255, 75, 75);
        color: rgb(255, 75, 75);
        box-shadow: rgba(255, 75, 75, 0.5) 0px 0px 0px 0.2rem;
        outline: currentcolor none medium;
    }
    """
    unclicked_style = """
    div[data-testid*="stHorizontalBlock"] > div:nth-child(%(nth_child)s):nth-last-child(%(nth_last_child)s) button {
        pointer-events: none;
        cursor: not-allowed;
        opacity: 0.65;
        filter: alpha(opacity=65);
        -webkit-box-shadow: none;
        box-shadow: none;
    }
    """
    style = ""
    for ix in range(n_buttons):
        ix += 1
        if ix == clicked_button_ix:
            style += clicked_style % get_button_indices(ix)
        else:
            style += unclicked_style % get_button_indices(ix)
    st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)
# Streamlit-App
def main():
    
    # Führe eine Wartezeit von wait_time Sekunden durch
    time.sleep(30)

    # Texteingabe
    
    # text_vorlesen = "Welcome to Syntex, to proceed further with the screenreader function, tap the big red botton on the left in the middle of your screen"
    # modell_speak(text_vorlesen)


    st.title("SynTex")

    # Texteingabe
    input_text = st.text_area("Enter a text", "")
    col1, col2 =  st.columns(2)
    with col1:
        if st.button("read text", on_click=style_button_row, kwargs={
        'clicked_button_ix': 1, 'n_buttons': 2}):
            if input_text == "":
                    st.warning("Please enter a text first")
                    modell_speak("Please enter a text first.")
            else:
                modell_speak(input_text)
    with col2:
        if st.button("Speech to Text", on_click=style_button_row, kwargs={
            'clicked_button_ix': 2, 'n_buttons': 2
        }):
            st.write("Click to start recording")



    # if st.button("read text"):
    #     if input_text == "":
    #         st.warning("Please enter a text first")
    #         modell_speak("Please enter a text first.")
    #     else:
    #         modell_speak(input_text)
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
    if st.button("Execute",key=2,type='secondary'):
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

#     m = st.markdown("""
# <style>
# div.stButton > button:nth-of-type(1) {
#     background-color: #ce1126;
#     color: white;
#     height: 3em;
#     width: 12em;
#     border-radius:10px;
#     border:3px solid #000000;
#     font-size:20px;
#     font-weight: bold;
#     margin: auto;
#     display: block;
# }

# div.stButton > button:hover {
# 	background:linear-gradient(to bottom, #ce1126 5%, #ff5a5a 100%);
# 	background-color:#ce1126;
# }
 
# div.stButton > button:active {
# 	position:relative;
# 	top:3px;
# }

# </style>""", unsafe_allow_html=True)
    if st.button("Screen reader",type='primary'):
        scren_text='Welcome to Syntex. Please input your text in the text area and press the read text button to hear the text. If you want to upload a document, please select the document type and press the execute button. If you want to activate the classification and the summary, please check the corresponding boxes. You can adjust the compression rate with the slider. To execute the classification and the summary, please press the execute button.'
        modell_speak(scren_text)
if __name__ == '__main__':
    
    main()

