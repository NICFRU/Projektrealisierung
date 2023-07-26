# SynTex

## Projektrealisierung
Dieses Projekt wurde realisiert von Niclas Cramer, Niklas Koch, Jasmina Pascanovic und Antoine Fuchs.

## Über das Projekt
Das SynTex Repository beinhaltet verschiedene Komponenten, darunter Modelle zur Klassifizierung und Zusammenfassung von Texten, Trainingsdaten sowie ein Frontend, das all diese Aspekte kombiniert. Wenn Sie mehr über die Funktionen erfahren möchten, lesen Sie die Anwenderdokumentation oder werfen Sie einen Blick auf den Code.

## Funktionen
Das Projekt bietet verschiedene Funktionen:
- **Textklassifikation**: Die Anwendung ist in der Lage, Texte als Nachrichtenartikel, Geschichten, wissenschaftliche Artikel oder Reviews zu klassifizieren.
- **Textzusammenfassung**: Abhängig von der Textklasse wird ein spezifisches Modell zur Textzusammenfassung angewendet:
   - Für wissenschaftliche Artikel wird das Modell 'NICFRU/bart-base-paraphrasing-science' genutzt.
   - Für Nachrichtenartikel wird das Modell 'NICFRU/bart-base-paraphrasing-news' genutzt.
   - Für Geschichten wird das Modell 'NICFRU/bart-base-paraphrasing-story' genutzt.
   - Für Reviews wird das Modell 'NICFRU/bart-base-paraphrasing-review' genutzt.
- **Sprache-zu-Text**: Die Anwendung ist auch in der Lage, gesprochene Sprache in Text umzuwandeln.
- **Datei-Upload**: Nutzer können Daten in den Formaten PDF, DOCX und TXT hochladen.

## Anwendung des Frontends
Folgen Sie diesen Schritten, um das Frontend zu verwenden: 

1. Installieren Sie die erforderliche Umgebung mit Conda, indem Sie folgenden Befehl verwenden (Bitte beachten Sie, dass die Installation der Umgebung momentan nur auf Apple-Geräten unterstützt wird):
```bash
conda env create -n projektrealisierung --file environment.yml
```
2. Aktivieren Sie die neu erstellte Umgebung:
```bash
conda activate projektrealisierung
```
3. Installieren Sie das 'spacy' Modul und das 'en-core-web-lg' Modell:
```bash
python -m spacy download en_core_web_lg
```
4. Wechseln Sie in den 'frontend'-Ordner:
```bash
cd frontend
```
5. Führen Sie die Frontend-Applikation aus:
```bash
streamlit run frontend.py
```
## Data Input Instructions

To use the application, you have two options for inputting your data:

### Option 1: Using a local file

1. Place your data file(s) in the 'Data' folder, which is located in the 'Frontend' directory of this application.

2. Within the application interface, you'll find a file selection tool. Use this to select your data file from the 'Data' folder.

### Option 2: Using an absolute link

1. If your data file is hosted elsewhere, you can directly use the absolute link to that file. Enter this link in the provided text field in the application interface.

2. Confirm your input by pressing 'Enter' or the corresponding confirmation button in the interface. Upon confirmation, a new text field will appear.

3. Copy the contents of the new text field and paste them back into the original text field (where you initially entered the link).

Please ensure that you follow these instructions carefully to avoid any issues in data processing. 

Viel Spaß beim Erforschen und Verwenden von SynTex!
