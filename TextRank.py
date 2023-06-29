import numpy as np
import networkx as nx
import spacy
from summa import keywords
from summa.summarizer import summarize

# Laden des Spacy-Modells
nlp = spacy.load("en_core_web_sm")

def textrank_extractive(text, compression_rate=0.5):
    # Tokenisierung
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]

    # Extrahiere Schlüsselwörter aus dem Text
    #keywords_list = keywords.keywords(text).split('\n')

    # Extrahiere Schlüsselsätze mit TextRank
    num_sentences = max(1, int(len(sentences) * compression_rate))
    extracted_sentences = summarize(text, words=num_sentences, split=True)

    # Erzeuge eine Matrix mit den BM25-Ähnlichkeiten zwischen den Sätzen
    bm25_matrix = np.zeros((len(sentences), len(sentences)))
    for i, sentence_i in enumerate(sentences):
        for j, sentence_j in enumerate(sentences):
            similarity = similarity_function(sentence_i, sentence_j)
            bm25_matrix[i, j] = similarity

    # Konstruiere einen Graphen mit den Sätzen als Knoten und den Ähnlichkeiten als Kanten
    graph = nx.from_numpy_array(bm25_matrix)

    # Berechne den TextRank-Score für jeden Satz
    scores = nx.pagerank(graph)

    # Wähle die besten Sätze basierend auf ihren TextRank-Scores aus
    top_sentences = sorted(scores, key=scores.get, reverse=True)[:num_sentences]

    # Sortiere die ausgewählten Sätze nach ihrer Position im Text
    top_sentences = sorted(top_sentences)

    # Gebe die extrahierten Schlüsselsätze zurück
    extracted_sentences = [sentences[index] for index in top_sentences]
    return extracted_sentences


def similarity_function(sentence1, sentence2):
    # Implementiere hier eine Funktion, um die Ähnlichkeit zwischen zwei Sätzen zu berechnen
    # Du kannst z.B. Spacy verwenden, um die Sätze zu vektorisieren und den Cosinus-Ähnlichkeitswert zu berechnen
    return 0.5  # Beispielwert


# Beispielaufruf
text = "Deutschland ([ˈdɔɪ̯t͡ʃlant] Audiodatei abspielen; Vollform des Staatennamens seit 1949: Bundesrepublik Deutschland) ist ein Bundesstaat in Mitteleuropa.[6] Er hat 16 Bundesländer und ist als freiheitlich-demokratischer und sozialer Rechtsstaat verfasst. Die 1949 gegründete Bundesrepublik Deutschland stellt die jüngste Ausprägung des 1871 erstmals begründeten deutschen Nationalstaates dar. Bundeshauptstadt und Regierungssitz ist Berlin. Deutschland grenzt an neun Staaten, es hat Anteil an der Nord- und Ostsee im Norden sowie dem Bodensee und den Alpen im Süden. Es liegt in der gemäßigten Klimazone und verfügt über 16 National- und mehr als 100 Naturparks. Das heutige Deutschland hat circa 84,4 Millionen Einwohner und zählt bei einer Fläche von 357.588 Quadratkilometern mit durchschnittlich 236 Einwohnern pro Quadratkilometer zu den dicht besiedelten Flächenstaaten. Die bevölkerungsreichste deutsche Stadt ist Berlin; weitere Metropolen mit mehr als einer Million Einwohnern sind Hamburg, München und Köln; der größte Ballungsraum ist das Ruhrgebiet. Frankfurt am Main ist als europäisches Finanzzentrum von globaler Bedeutung. Die Geburtenrate liegt bei 1,58 Kindern pro Frau (2021).[7] Auf dem Gebiet Deutschlands ist die Anwesenheit von Menschen vor 600.000 Jahren durch Funde des Homo heidelbergensis sowie zahlreiche prähistorischer Kunstwerke aus der späteren Altsteinzeit nachgewiesen. Während der Jungsteinzeit, um 5600 v. Chr., wanderten die ersten Bauern mitsamt Vieh und Saatgut aus dem Nahen Osten ein. Seit der Antike ist die lateinische Bezeichnung Germania für das Siedlungsgebiet der Germanen bekannt. Das ab dem 10. Jahrhundert bestehende römisch-deutsche Reich, das aus vielen Herrschaftsgebieten bestand, war wie der 1815 ins Leben gerufene Deutsche Bund und die liberale demokratische Bewegung Vorläufer des späteren deutschen Gesamtstaates, der 1871 als Deutsches Reich gegründet wurde. Die rasche Entwicklung vom Agrar- zum Industriestaat vollzog sich während der Gründerzeit in der zweiten Hälfte des 19. Jahrhunderts. Nach dem Ersten Weltkrieg wurde 1918 die Monarchie abgeschafft und die demokratische Weimarer Republik konstituiert. Ab 1933 führte die nationalsozialistische Diktatur zu politischer und rassistischer Verfolgung und gipfelte in der Ermordung von sechs Millionen Juden und Angehörigen anderer Minderheiten wie Sinti und Roma. Der vom NS-Staat 1939 begonnene Zweite Weltkrieg endete 1945 mit der Niederlage der Achsenmächte. Das von den Siegermächten besetzte Land wurde 1949 geteilt, nachdem bereits 1945 seine Ostgebiete teils unter polnische, teils sowjetische Verwaltungshoheit gestellt worden waren. Der Gründung der Bundesrepublik als demokratischer westdeutscher Teilstaat mit Westbindung am 23. Mai 1949 folgte die Gründung der sozialistischen DDR am 7. Oktober 1949 als ostdeutscher Teilstaat unter sowjetischer Hegemonie. Die innerdeutsche Grenze war nach dem Berliner Mauerbau (ab 13. August 1961) abgeriegelt. Nach der friedlichen Revolution in der DDR 1989 erfolgte die Lösung der deutschen Frage durch die Wiedervereinigung beider Landesteile am 3. Oktober 1990, womit auch die Außengrenzen Deutschlands als endgültig anerkannt wurden. Durch den Beitritt der fünf ostdeutschen Länder sowie die Wiedervereinigung von Ost- und West-Berlin zur heutigen Bundeshauptstadt zählt die Bundesrepublik Deutschland seit 1990 sechzehn Bundesländer. Deutschland ist Gründungsmitglied der Europäischen Union und ihrer Vorgänger (Römische Verträge 1957) sowie deren bevölkerungsreichstes Land. Mit 18 anderen EU-Mitgliedstaaten bildet es eine Währungsunion, die Eurozone. Es ist Mitglied der UN, der OECD, der OSZE, der NATO, der G7, der G20 und des Europarates. Die Vereinten Nationen unterhalten seit 1951 ihren deutschen Sitz in Bonn („UNO-Stadt“).[8] Die Bundesrepublik Deutschland gilt als einer der politisch einflussreichsten Staaten Europas und ist ein gesuchtes Partnerland auf globaler Ebene.[9] Gemessen am Bruttoinlandsprodukt ist das marktwirtschaftlich organisierte Deutschland die größte Volkswirtschaft Europas und die viertgrößte der Welt.[10] 2016 war es die drittgrößte Export- und Importnation.[11] Es ist eine Informations- und Wissensgesellschaft. Automatisierung, Digitalisierung und Disruption prägen die innovative deutsche Industrieentwicklung. Die Steigerung der Qualität des deutschen Bildungssystems und die nachhaltige Entwicklung des Landes gelten als zentrale Aufgaben der Standortpolitik. Gemäß dem Index der menschlichen Entwicklung zählt Deutschland zu den sehr hoch entwickelten Ländern.[12][13] Muttersprache der Bevölkerungsmehrheit ist Deutsch. Daneben gibt es Regional- und Minderheitensprachen und sowohl Deutsche als auch Migranten mit anderen Muttersprachen, von denen die bedeutendsten Türkisch und Russisch sind.[14] Bedeutendste Fremdsprache ist Englisch, das in allen Bundesländern in der Schule gelehrt wird. Die Kultur Deutschlands ist vielfältig und wird neben zahlreichen Traditionen, Institutionen und Veranstaltungen beispielsweise in der Auszeichnung als UNESCO-Welterbe in Deutschland, in Kulturdenkmälern und als immaterielles Kulturerbe erfasst und gewürdigt."

compression_rate = 0.05  # 50% Kompressionsrate
extracted = textrank_extractive(text, compression_rate)
for sentence in extracted:
    print(sentence)
