# Autor: Niclas Cramer, Antoine Fuchs
import numpy as np
import re
import networkx as nx
import spacy
from summa import keywords
from summa.summarizer import summarize

# Laden des Spacy-Modells

from nltk.tokenize import sent_tokenize
from transformers import  AutoTokenizer
import math
from tqdm import tqdm

from transformers import pipeline


def reduce_repetitions(text):
    # Wir benutzen reguläre Ausdrücke (regex), um Wiederholungen von Zeichen zu reduzieren.
    text = re.sub(r'\.{2,}', '.', text)  # Ersetzt zwei oder mehr Punkte durch einen Punkt.
    text = re.sub(r'\!{2,}', '!', text)  # Ersetzt zwei oder mehr Ausrufezeichen durch ein Ausrufezeichen.
    text = re.sub(r'\,{2,}', ',', text)  # Ersetzt zwei oder mehr Kommas durch ein Komma.
    text = re.sub(r'\;{2,}', ';', text)  # Ersetzt zwei oder mehr Semikolons durch ein Semikolon.
    return text  # Gibt den bereinigten Text zurück.

def textrank_extractive(text, compression_rate=0.5,split='\. '):
    # Hier verwenden wir Spacy, um den Text in Sätze zu zerlegen und zu tokenisieren.
    nlp = spacy.load("en_core_web_lg")
    doc = re.split(fr'(?<!\b\w\w){split}', reduce_repetitions(re.sub(' +', ' ', text.replace("\n", " ").replace('-',' ').replace('_',' ').replace("\'", "").replace("!", ".").replace("?", ".").replace(";", ""))))
    sentences = [sent for sent in doc if len(sent.replace("-", " ").split()) > 2]
    sentence_docs = [nlp(sentence) for sentence in sentences]

    # Hier verwenden wir TextRank, um wichtige Sätze zu extrahieren.
    num_sentences = max(1, int(len(sentences) * compression_rate))
    extracted_sentences = summarize(text, words=num_sentences, split=True)

    # Wir bauen eine Matrix auf, die die Ähnlichkeit zwischen den Sätzen misst.
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for i, doc_i in enumerate(sentence_docs):
        for j, doc_j in enumerate(sentence_docs):
            similarity = similarity_function(doc_i, doc_j)
            similarity_matrix[i, j] = similarity

    # Hier erstellen wir einen Graphen, in dem die Sätze die Knoten und die Ähnlichkeiten die Kanten sind.
    graph = nx.from_numpy_array(similarity_matrix)

    # Nun berechnen wir den TextRank-Score für jeden Satz.
    scores = nx.pagerank_numpy(graph)

    # Wir wählen die besten Sätze basierend auf ihren TextRank-Scores aus.
    top_sentences = sorted(scores, key=scores.get, reverse=True)[:num_sentences]

    # Die ausgewählten Sätze werden nach ihrer Position im Text sortiert.
    top_sentences = sorted(top_sentences)

    # Schließlich geben wir die extrahierten Schlüsselsätze zurück.
    extracted_sentences = [sentences[index] for index in top_sentences]
    return extracted_sentences

def similarity_function(doc1, doc2):
    # Wir berechnen die Cosinus-Ähnlichkeit zwischen den beiden Dokumenten.
    similarity = doc1.similarity(doc2)
    return similarity

def compression_ratio(text, summary):
    # Wir berechnen das Verhältnis der Anzahl der Wörter in der Zusammenfassung zur Anzahl der Wörter im Ausgangstext.
    num_words_text = len(text.split())
    num_words_summary = len(summary.split())
    ratio = num_words_summary / num_words_text
    return ratio

def compression(text, compression_rate,split='\. '):
    max_iterations = 20
    iterations = 0
    extracted = textrank_extractive(text, compression_rate,split)
    summary = '. '.join(extracted)
    compression_rate_renwed = compression_rate

    # Wenn das Kompressionsverhältnis kleiner ist als die gewünschte Rate, versuchen wir, das Kompressionsverhältnis zu erhöhen.
    while compression_ratio(text, summary) < compression_rate and iterations < max_iterations:
        iterations += 1
        compression_rate_renwed += 0.05
        if compression_rate_renwed > 1:
            compression_rate_renwed = 1
        extracted = textrank_extractive(text, compression_rate=compression_rate_renwed)
        summary = '. '.join(extracted)
    return summary  # Gibt die komprimierte Zusammenfassung zurück.



def token_count(text):
    # Wir zerlegen den Text in einzelne Worte (Token) und zählen diese.
    tokens = text.split()
    return len(tokens)  # Gibt die Anzahl der Wörter (Token) im Text zurück.

def adjust_length(text):
    # Zuerst zählen wir die Anzahl der Wörter (Token) im Text.
    length = token_count(text)
    
    # Wir passen die Mindest- und Maxmimallänge des Textes basierend auf der aktuellen Länge an.
    # Die genauen Werte sind hier variabel und hängen von der Länge des Textes ab.
    
    if length <20:
        # Wenn der Text weniger als 20 Wörter hat, erhöhen wir die Mindestlänge um 5% der aktuellen Länge.
        # Die maximale Länge wird dann auf das Doppelte der Mindestlänge gesetzt.
        min_length = length + int(length * 0.05)
        max_length = min_length +min_length
    elif length <50:
        # Wenn der Text zwischen 20 und 50 Wörtern hat, verwenden wir die gleichen Regeln wie zuvor, 
        # aber die maximale Länge wird auf das 1,5-fache der Mindestlänge gesetzt.
        min_length = length + int(length * 0.05)
        max_length = min_length +min_length* 0.5
    elif length <60:
        # Bei Texten zwischen 50 und 60 Wörtern erhöhen wir die Mindestlänge um 5% der aktuellen Länge.
        # Die maximale Länge wird dann auf das 1,4-fache der Mindestlänge gesetzt.
        min_length = length + int(length * 0.05)
        max_length = min_length +min_length* 0.4
    elif length < 80:
        # Bei Texten zwischen 60 und 80 Wörtern erhöhen wir die Mindestlänge um 5% der aktuellen Länge.
        # Die maximale Länge wird dann auf das 1,25-fache der Mindestlänge gesetzt.
        min_length = length + int(length * 0.05)
        max_length = min_length + min_length* 0.25
    elif length < 100:
        # Bei Texten zwischen 80 und 100 Wörtern erhöhen wir die Mindestlänge um 30% der aktuellen Länge.
        # Die maximale Länge wird dann auf die Mindestlänge plus 100 Wörter gesetzt.
        min_length = length + int(length * 0.3)
        max_length = min_length + 100
    else:
        # Bei Texten mit 100 Wörtern oder mehr setzen wir die Mindestlänge auf das nächste Vielfache von 70,
        # das größer als die aktuelle Länge geteilt durch 50 ist. 
        # Die maximale Länge wird dann auf die Mindestlänge plus 100 Wörter gesetzt.
        min_length = math.ceil(length / 50) * 70
        max_length = min_length + 100

    # Wir geben die berechneten minimalen und maximalen Längen zurück.
    return min_length, max_length

def batch_sent(sentenc,splitt=180,split='\. '):
    # Teile den eingegebenen Text in einzelne Sätze auf, wobei jeder Satz durch ". " getrennt ist.
    # Hierbei wird sichergestellt, dass das ". " nicht auf ein einzelnes Wort folgt.
    sentences = re.split(fr'(?<!\b\w\w){split}', sentenc.lower())

    # Initialisierung der Batches und der aktuellen Batch-Liste sowie der aktuellen Batch-Länge.
    batches = []
    batch = []
    batch_len = 0
    
    # Durchlaufen Sie jeden Satz in den Sätzen.
    for sentence in sentences:
        # Berechnen Sie die Anzahl der Tokens im Satz.
        sentence_len = len(tokenizer.tokenize(sentence))
        
        # Wenn die Hinzufügung des aktuellen Satzes die maximale Batch-Länge überschreitet...
        if sentence_len + batch_len > splitt:
            # ...und wenn der aktuelle Satz weniger als die maximale Batch-Länge hat...
            if sentence_len < splitt:  
                # ...füge die aktuelle Batch-Liste zu den Batches hinzu...
                batches.append(batch)
                # ...und beginne eine neue Batch-Liste mit dem aktuellen Satz.
                batch = [sentence]
                # Die aktuelle Batch-Länge wird auf die Länge des aktuellen Satzes gesetzt.
                batch_len = sentence_len
            # Sätze, die länger als die maximale Batch-Länge sind, werden übersprungen.
        else:
            # Wenn der aktuelle Satz zur aktuellen Batch-Liste hinzugefügt werden kann, ohne die maximale Batch-Länge zu überschreiten...
            # ...füge den Satz zur aktuellen Batch-Liste hinzu...
            batch.append(sentence)
            # ...und erhöhe die aktuelle Batch-Länge um die Länge des aktuellen Satzes.
            batch_len += sentence_len
    
    # Füge die letzte Batch-Liste zu den Batches hinzu.
    batches.append(batch)

    # Die Funktion gibt die erstellten Batches zurück.
    return batches

def text_rank_algo(dictionary, komp='compression', split='\\. ', random_T=True, column='text'):
    # Text aus dem Wörterbuch extrahieren und bearbeiten
    text = dictionary[column].replace("\n", " ")
    if random_T:
        random_value = dictionary[komp]
    else:
        if dictionary['reduction_multiplier'] < 0.8:
            random_value = dictionary['desired_compression_rate']
        elif dictionary['reduction_multiplier'] < 0.9:
            random_value = dictionary['reduction_multiplier']
        else:
            random_value = 1

    # Durchführen der Textkompression
    text_rank_text = compression(text.replace("\n\n", " "), random_value, split)
    compression_ratio_value = compression_ratio(text, compression(text, random_value, split))

    # Weitere Textbearbeitung
    text = re.sub(' +', ' ', text.replace("\n", " ").replace('-',' ').replace('_',' ').replace("\'", "").replace("!", ".").replace("?", ".").replace(";", ""))
    text_rank_text = re.sub(' +', ' ', text_rank_text.replace("\n", " ").replace('-',' ').replace('_',' ').replace("\'", "").replace("!", ".").replace("?", ".").replace(";", ""))

    # Hinzufügen von Ergebnissen zum Wörterbuch
    if random_T:
        dictionary['text'] = text
        dictionary['text_rank_text'] = text_rank_text
        dictionary['tokens_gesamt'] = len(text.split(' '))
        dictionary['token_text_rank'] = len(text_rank_text.split(' '))
        dictionary['desired_compression_rate'] = random_value
        dictionary['text_rank_compression_rate'] = compression_ratio_value
    else:
        dictionary['text_rank_text_2'] = text_rank_text
        dictionary['tokens_gesamt_2'] = len(text.split(' '))
        dictionary['token_text_rank_2'] = len(text_rank_text.split(' '))
        dictionary['desired_compression_rate_2'] = random_value
        dictionary['text_rank_compression_rate_2'] = compression_ratio_value

    # Rückgabe des aktualisierten Wörterbuchs
    return dictionary

def check_class_and_get_model_name(input_dict, class_key):
    # Aus dem Wörterbuch den Wert des gegebenen Schlüssels abrufen
    class_value = input_dict.get(class_key)

    # Überprüfen, ob der Wert existiert
    if class_value is None:
        raise ValueError(f"'{class_key}' nicht im Eingabedictionary gefunden")

    # Entsprechend dem Wert den Modellnamen zuweisen
    if class_value == 'Wissenschaftliches Paper':
        model_name = 'NICFRU/bart-base-paraphrasing-science'
    elif class_value == 'Nachrichtenartikel':
        model_name = 'NICFRU/bart-base-paraphrasing-news'
    elif class_value == 'Literarischer Text':
        model_name = 'NICFRU/bart-base-paraphrasing-story'
    elif class_value == 'Review':
        model_name = 'NICFRU/bart-base-paraphrasing-review'
    else:
        return False

    # Rückgabe des Modellnamens
    return model_name

def create_model(dictionary):
    # Nutzen Sie die check_class_and_get_model_name Funktion, um den Namen des Modells zu ermitteln
    model_name = check_class_and_get_model_name(dictionary, 'classification')
    
    # Definieren Sie die Tokenizer und Summarizer als globale Variablen, damit sie außerhalb der Funktion verwendet werden können
    global tokenizer, summarizer

    # Laden Sie den Tokenizer und das Modell von Huggingface
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    summarizer = pipeline("text2text-generation", model=model_name)


def paraphrase_of_text(dictionary, text_name='text', komp_name='reduction_multiplier', split='\. '):
    # Initialisierung der Listen
    text_gesamt_list = []
    batch_text_list = []

    # Auslesen der Text- und Kompressionsinformationen aus dem Wörterbuch
    text = dictionary[text_name]
    komp = dictionary[komp_name]

    # Iteration über die Batches des Texts
    for batch in tqdm(batch_sent(text, split=split), desc='Verarbeite Batches'):
        # Überprüfen Sie, ob der aktuelle Batch nicht leer ist
        if len(batch):
            # Zusammenfügen der Sätze in einem Batch
            batch_text = '. '.join(batch)
            batch_text += "."
            batch_text_list.append(batch_text)
            
            # Anpassung der Länge und Generierung der Zusammenfassung
            min_length_test, max_length_test = adjust_length(batch_text)
            ext_summary = summarizer(batch_text, max_length=int(round(max_length_test * komp, 0)), min_length=int(round(min_length_test * komp, 0)), length_penalty=100, num_beams=2)

            # Hinzufügen der generierten Zusammenfassung zur Gesamtliste
            text_gesamt_list.append(ext_summary[0]['generated_text'])

    # Erstellen der Gesamtzusammenfassung und Berechnung der endgültigen Kompressionsrate
    text_gesamt = '. '.join(text_gesamt_list)
    actual_compression_rate = len(text_gesamt.split(' ')) / len(text.split(' ')) * 100

    # Aktualisierung des Wörterbuchs mit den neuen Informationen
    dictionary['Zusammenfassung'] = text_gesamt
    dictionary['Endgueltige_Kompressionsrate'] = actual_compression_rate
    dictionary['länge Zusammenfassung'] = len(text_gesamt.split(' '))
    dictionary['länge Ausgangstext'] = len(text.split(' '))
    dictionary['batch_texts'] = batch_text_list
    dictionary['batch_output'] = text_gesamt_list

    return dictionary

def calculate_compression(input_dict, total_tokens_col, current_tokens_col, desired_compression_rate):
    # Berechne die aktuelle Kompressionsrate
    input_dict['current_compression_rate'] = input_dict[current_tokens_col] / input_dict[total_tokens_col]
    # Berechne die Differenz zur gewünschten Kompressionsrate
    input_dict['compression_difference'] = input_dict[desired_compression_rate] - input_dict['current_compression_rate']
    # Berechne den Reduktionsmultiplikator
    input_dict['reduction_multiplier'] = input_dict[desired_compression_rate] / input_dict['current_compression_rate']
    return input_dict

def execute_text_gen(dictionary, split='\. ', seed=10):
    # Kopieren Sie das Wörterbuch für Manipulationen
    dictionary_copy = dictionary.copy()

    # Führe den Text-Rank-Algorithmus aus
    dictionary_copy = text_rank_algo(dictionary_copy, split=split)

    # Berechne die Kompression
    dictionary_copy = calculate_compression(dictionary_copy, 'tokens_gesamt', 'token_text_rank', 'desired_compression_rate')

    # Erstelle das Modell
    create_model(dictionary_copy)

    # Paraphrasiere den Text
    dictionary_copy = paraphrase_of_text(dictionary_copy, text_name='text_rank_text', split=split)

    # Berechne die endgültige Kompressionsrate
    dictionary_copy['ent_com_rate'] = dictionary_copy['länge Zusammenfassung'] / dictionary_copy['tokens_gesamt']

    # Berechne erneut die Kompression
    dictionary_copy = calculate_compression(dictionary_copy, 'tokens_gesamt', 'länge Zusammenfassung', 'desired_compression_rate')

    # Führe erneut den Text-Rank-Algorithmus aus
    dictionary_copy = text_rank_algo(dictionary_copy, random_T=False, column='Zusammenfassung')

    # Berechne die endgültige Kompressionsrate erneut
    dictionary_copy['ent_com_rate'] = dictionary_copy['länge Zusammenfassung'] / dictionary_copy['tokens_gesamt']

    return dictionary_copy
