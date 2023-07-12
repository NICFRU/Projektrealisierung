import streamlit as st
import plotly.graph_objects as go
from keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import numpy as np
import streamlit as st
from PyPDF2 import PdfFileReader
from docx import Document


model = load_model("../models/classification/neuro_net_1.h5")
vectorizer = joblib.load("../models/classification/vectorizer_1.joblib")

def read_pdf(file):
    pdf = PdfFileReader(file)
    text = ""
    for page in range(pdf.numPages):
        text += pdf.getPage(page).extractText()
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
    predictions = model.predict(new_text_features.toarray())
    predicted_class = np.argmax(predictions, axis=1)
    predicted_probability = np.max(predictions, axis=1)
    return predicted_class, predicted_probability

# Streamlit-App
def main():
    st.title("SynTex")
    
    # Texteingabe
    input_text = st.text_area("Text eingeben", "")
    
    # Dokument auswählen
    file = st.file_uploader("Dokument auswählen", type=["pdf", "docx", "txt"])

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
        st.text_area("Ausgabe", value=content, height=500)
    
    # Checkboxen für Klassifikation und Zusammenfassung
    classification_enabled = st.checkbox("Klassifikation aktivieren")
    summarization_enabled = st.checkbox("Zusammenfassung aktivieren")
    
    # Schieberegler für die Kompressionsrate
    compression_rate = st.slider("Kompressionsrate (%)", min_value=0, max_value=100, value=50, step=1)
    
    # Knopf zum Zusammenfassen und Klassifizieren
    if st.button("Ausführen"):
        if input_text:
            if summarization_enabled:
                summarized_text = summarize_text(input_text, compression_rate)
                st.subheader("Zusammenfassung")
                st.write("Dies ist eine Zusammenfassung des eingegebenen Textes mit einer Kompressionsrate von {}%:".format(compression_rate))
                st.write(summarized_text)
            
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
                else:
                    st.write(f"Achtung! Das Modell ist sich nicht ganz sicher. Die von dem Modell errechnete Klasse ist: \"{predicted_class}\". Das Modell sagt diese Klasse allerdings nur mit einer Wahrscheinlichkeit von {predicted_probability}% vorraus.")
        else:
            st.warning("Bitte geben Sie zuerst einen Text ein.")



if __name__ == '__main__':
    main()
