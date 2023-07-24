####### Installieren der benötigten Pakete #######
'''
pip uninstall speechrecognition           
pip uninstall pyaudio
pip uninstall playsound
pip uninstall gtts
pip uninstall PyObjC - für Mac
'''
####
import speech_recognition as sr
from time import ctime # get time details
import time
import playsound
import os
import random
from gtts import gTTS

def modell_speak(text):
    text = str(text)
    engine.say(text)
    engine.runAndWait()
r = sr.Recognizer()

def there_exists(terms,voice_data):
    for term in terms:
        if term.lower() in voice_data.lower():
            return True
def modell_speak(audio_string):
    tts = gTTS(text=audio_string, lang='en') # text to speech(voice)
    r = random.randint(1,20000000)
    audio_file = 'audio-' + str(r) + '.mp3'
    tts.save(audio_file) # save as mp3
    playsound.playsound(audio_file) # play the audio file
    print(audio_string) # print what app said
    os.remove(audio_file) # remove audio file


def record_audio(ask="",filename='transcription.txt'):
    with sr.Microphone() as source: # microphone as source
        # if ask:!
        #     print(ask)
        #audio = r.listen(source, 5, 5)  # listen for the audio via source
        voice_data = ''
        modell_speak("Starting transcription")
        modell_speak("You can speak now!")
        voice_data=r.listen(source)
        modell_speak("Done Listening")
        
        try:
            voice_data = r.recognize_google(voice_data)
            modell_speak("I repead")
            modell_speak(voice_data)
            modell_speak('If incorrect press the button again')  # convert audio to text
        except sr.UnknownValueError: # error: recognizer does not understand
            modell_speak('I did not get that')
        except sr.RequestError:
            modell_speak('Sorry, the service is down')
        #print(">>", voice_data.lower()) # print what user said

        # Den Text in eine Datei schreiben
        with open(filename, 'w') as file:
            file.write(voice_data)
        #print(voice_data)
        
        return voice_data



def modell_speak(audio_string):
    tts = gTTS(text=audio_string.lower(), lang='en') # text to speech(voice)
    r = random.randint(1,20000000)
    audio_file = 'audio-' + str(r) + '.mp3'
    tts.save(audio_file) # save as mp3
    playsound.playsound(audio_file) # play the audio file
    print(audio_string) # print what app said
    os.remove(audio_file) # remove audio file
    time.sleep(1)


if __name__ == "__main__":
    while 1:

        voice_data=record_audio()
        print('Text: ',voice_data)
        modell_speak("bye")
        exit()
        #respond(voice_data)