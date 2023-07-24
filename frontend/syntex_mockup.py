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
model_classification = load_model("../models/classification/neuro_net_1.h5")
vectorizer = joblib.load("../models/classification/vectorizer_1.joblib")

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

engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Anpassen der Sprechgeschwindigkeit (optional)

def text_to_speech(text):
    global engine
    engine.say(text)
    engine.runAndWait()  

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

def summarize_text(text, compression_rate):
    # Hier erfolgt die Zusammenfassung des Textes unter Berücksichtigung der Kompressionsrate
    # Implementiere hier deine Logik zur Zusammenfassung des Textes
    if compression_rate <= 20:
        summarized_text = """Boehringer Ingelheim is a global,
        family-owned researching pharmaceutical company that focuses on researching, developing,
        producing and selling prescription drugs for humans and animals. Pharmaceutical compa-
        nies like Boehringer Ingelheim invest billions of euros and years of work into researching
        and developing; and sometimes without even finding a satisfying result. The development of
        a drug can cost 1.0 to 1.6 billion US-Dollars and can last for over 13 years; only for one age
        class. Because of that, data quality and trustworthy data in general play an important role for
        the company.
        This project paper simply explains how data quality can be verified using automatic genera-
        tion of test cases from graph databases and how a framework could look like that ensures high
        quality data. It defines necessary vocabulary and explains required concepts, languages and
        functionalities like RDF, OWL, SHACL, SPARQL, the Semantic Web, graph databases (like
        knowledge graphs), the Astrea-Tool and the RDFUnit Testing Suite. The final result of this
        project paper is a software concept, that links the Astrea-Tool and RDFUnit Testing Suite to
        enable automatic generation of data shapes, as well as test cases for those data shapes. This
        final software concept only needs data stored in RDF triples and the corresponding ontolo-
        gies to automatically create an inspection report, which clearly depicts errors or irregularities
        in any dataset.
        """
    elif compression_rate >= 80:
        summarized_text = """This project paper simply explains how data quality can be verified using automatic 
        generation of test cases from graph databases and how a framework could look like that ensures high quality data. This
        final software concept only needs data stored in RDF triples and the corresponding ontolo-
        gies to automatically create an inspection report, which clearly depicts errors or irregularities
        in any dataset."""
    else:
        summarized_text = """This project paper simply explains how data quality can be verified using automatic 
        generation of test cases from graph databases and how a framework could look like that ensures high quality data. It defines necessary vocabulary and explains required concepts, languages and
        functionalities like RDF, OWL, SHACL, SPARQL, the Semantic Web, graph databases (like
        knowledge graphs), the Astrea-Tool and the RDFUnit Testing Suite. The final result of this
        project paper is a software concept, that links the Astrea-Tool and RDFUnit Testing Suite to
        enable automatic generation of data shapes, as well as test cases for those data shapes. This
        final software concept only needs data stored in RDF triples and the corresponding ontolo-
        gies to automatically create an inspection report, which clearly depicts errors or irregularities
        in any dataset."""
    return summarized_text

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
    text_to_speech(text)

    st.title("SynTex")

    # Texteingabe
    input_text = st.text_area("Text eingeben", "")
    
    # Dokument auswählen
    file = st.file_uploader("Dokument auswählen", type=["pdf", "docx", "txt"])


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
    if button_clicked:
        text_to_speech(all_texts)
    

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
    classification_enabled = st.checkbox("Klassifikation aktivieren")
    summarization_enabled = st.checkbox("Zusammenfassung aktivieren")
    
    # Schieberegler für die Kompressionsrate
    compression_rate = st.slider("Kompressionsrate (%)", min_value=0, max_value=100, value=50, step=1)
    
    # Knopf zum Zusammenfassen und Klassifizieren
    if st.button("Ausführen"):
        if input_text:
            
            if classification_enabled:
                predicted_class, predicted_probability = classify_text(input_text)
                predicted_probability = round(predicted_probability[0]*100, 2)
                if predicted_class == 0:
                    predicted_class = "Wissenschaftliches Paper"
                elif predicted_class == 1:
                    predicted_class = "Nachrichtenartikel"
                elif predicted_class == 2:
                    predicted_class = "Review"
                else:
                    predicted_class = "Literarischer Text"
                st.subheader("Klassifikation")

                if predicted_probability >= 95:
                    st.write(f"Die von dem Modell errechnete Klasse ist: \"{predicted_class}\". Das Modell ist sich mit einer Wahrscheinlichkeit von {predicted_probability}%  sehr sicher.")
                    if summarization_enabled:
                        summarized_text = summarize_text(input_text, compression_rate)
                        st.subheader("Zusammenfassung")
                        st.write("Dies ist eine Zusammenfassung des eingegebenen Textes mit einer Kompressionsrate von {}%:".format(compression_rate))
                        st.write(summarized_text)
                else:
                    st.write(f"Achtung! Das Modell ist sich nicht ganz sicher. Die von dem Modell errechnete Klasse ist: \"{predicted_class}\". Das Modell sagt diese Klasse allerdings nur mit einer Wahrscheinlichkeit von {predicted_probability}% vorraus.")
        else:
            st.warning("Bitte geben Sie zuerst einen Text ein.")



if __name__ == '__main__':
    main()

