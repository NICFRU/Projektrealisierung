import streamlit as st

# Hintergrundfarbe und Schriftfarbe einstellen
st.markdown(
    """
    <style>
    body {
        color: white;
        background-color: darkblue;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# SynTeX Logo anzeigen
st.image("syntex_logo.png", use_column_width=True)

# Eingabefeld für Dateiupload und String-Eingabe
uploaded_file = st.file_uploader("Datei hochladen")
input_string = st.text_input("Eingabe")

# Ladebalken anzeigen
progress_bar = st.progress(0)

# Ausgabetextfeld
output_text = st.empty()

# Tastendruck-Handler
def on_keypress(key):
    # Schriftfarbe ändern
    if key == "r":
        st.markdown(
            """
            <style>
            body {
                color: red !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    elif key == "g":
        st.markdown(
            """
            <style>
            body {
                color: green !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    elif key == "b":
        st.markdown(
            """
            <style>
            body {
                color: blue !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

# Tastendruck-Handler registrieren
st.register_on_keydown_callback(on_keypress)

# Beispieltext anzeigen
st.write("Willkommen bei SynTeX!")

# Dateiupload-Handler
if uploaded_file:
    with st.spinner("Datei wird hochgeladen..."):
        # Dateiverarbeitung hier durchführen
        # Beispiel: Fortschrittsanzeige aktualisieren
        for percent_complete in range(100):
            progress_bar.progress(percent_complete + 1)
        progress_bar.empty()
        output_text.write("Datei erfolgreich hochgeladen!")

# String-Eingabe-Handler
if input_string:
    st.write("Eingabe:", input_string)
