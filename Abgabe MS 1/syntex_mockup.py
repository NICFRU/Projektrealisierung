import streamlit as st
import plotly.graph_objects as go


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
    # Hier erfolgt die Klassifizierung des Textes
    # Implementiere hier deine Logik zur Klassifizierung des Textes
    # Beispielklassifikation
    labels = ['Review', 'Abstract', 'Literature', 'News']
    values = [10, 80, 3, 2]
    return labels, values

# Streamlit-App
def main():
    st.title("SynTex")
    
    # Texteingabe
    input_text = st.text_area("Text eingeben", "")
    
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
                labels, values = classify_text(input_text)
                st.subheader("Klassifikation")
                fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
                st.plotly_chart(fig)
        else:
            st.warning("Bitte geben Sie zuerst einen Text ein.")

if __name__ == '__main__':
    main()
