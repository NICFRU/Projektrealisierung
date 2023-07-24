import streamlit as st

# Dark-Mode-Thema aktivieren
st.markdown(
    """
    <style>
    body {
        color: white;
        background-color: #2c3e50;
    }
    .logo-container {
        display: flex;
        align-items: flex-start;
    }
    .logo {
        max-width: 100px;
        margin: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Logo und Header anzeigen
st.markdown(
    """
    <div class="logo-container">
        <img src="syntex_logo.png" class="logo">
        <h1>SynTeX</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Eingabefeld für Dateiupload und String-Eingabe
uploaded_file = st.file_uploader("Datei hochladen")
input_string = st.text_area("Eingabe", height=200)

# Ladebalken anzeigen
progress_bar = st.progress(0)

# Kategorien und Prozentwerte
categories = {
    "Scientific": 92,
    "Story": 33,
    "News": 15,
    "Reviews": 15
}

# Skala anzeigen
for category, percentage in categories.items():
    st.write(f"{category}: {percentage}%")

# Textanzeige
if uploaded_file or input_string:
    if uploaded_file:
        with st.spinner("Datei wird hochgeladen..."):
            # Dateiverarbeitung hier durchführen
            # Beispiel: Fortschrittsanzeige aktualisieren
            for percent_complete in range(100):
                progress_bar.progress(percent_complete + 1)
            progress_bar.empty()
            st.write("Datei erfolgreich hochgeladen!")
    else:
        st.text_area("Ausgabe", height=200, value="Your summarized text will be here...")
