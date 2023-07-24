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


# def zum_klass(filename='klassifizieren'):
#     modell_speak('choose among compression classification or say both')
#     voice_data=record_audio(filename=filename)
#     pmove=voice_data.lower()
#     if there_exists(["compression","classification","both"],voice_data):
#         pmove=voice_data.lower()
#         modell_speak("You chose " + pmove)
#         modell_speak('If incorrect say restart options else audio finisched')
#         voice_data=record_audio(filename=filename)
#         if there_exists(["restart options"],voice_data):
#             zum_klass(filename=filename)
#         else:
#             modell_speak('audio finisched. Say quit if you want to end the process')
#             voice_data=record_audio()
#             respond(voice_data)
#     else:
#        modell_speak("You chose " + pmove)
#        modell_speak(pmove + " is not an possibility")
#        modell_speak('Options restarting or exit')
#        voice_data=record_audio(filename=filename)
#        if there_exists(["exit"],voice_data):
#         modell_speak('Exit operation')
#        else:
#         modell_speak('restart operation')
#         zum_klass(filename=filename)



    


def respond(voice_data):
    # 1: greeting
    # if 'what is your name' in voice_data:

    #     modell_speak('My name is Nico')

    # if there_exists(["what's the time","tell me the time","what time is it","what is the time"],voice_data):
    #     time = ctime().split(" ")[4].split(":")[0:2]
    #     if time[0] == "00":
    #         hours = '12'
    #     else:
    #         hours = time[0]
    #     minutes = time[1]
    #     time = hours + " hours and " + minutes + "minutes"
    #     modell_speak(time)

    # if there_exists(["game"],voice_data):
    #     modell_speak("choose among rock paper or scissor")
    #     voice_data = record_audio("choose among rock paper or scissor")
    #     print(voice_data)
    #     moves=["rock", "paper", "scissor"]
    
    #     cmove=random.choice(moves)
    #     pmove=voice_data.lower()
        

    #     modell_speak("The computer chose " + cmove)
    #     modell_speak("You chose " + pmove)
    #     #modell_speak("hi")
    #     if pmove==cmove:
    #         modell_speak("the match is draw")
    #     elif pmove== "rock" and cmove== "scissor":
    #         modell_speak("Player wins")
    #     elif pmove== "rock" and cmove== "paper":
    #         modell_speak("Computer wins")
    #     elif pmove== "paper" and cmove== "rock":
    #         modell_speak("Player wins")
    #     elif pmove== "paper" and cmove== "scissor":
    #         modell_speak("Computer wins")
    #     elif pmove== "scissor" and cmove== "paper":
    #         modell_speak("Player wins")
    #     elif pmove== "scissor" and cmove== "rock":
    #         modell_speak("Computer wins")

    if there_exists(["start transcription", "begin recording","recording","open text file","transcription","restart transcription"],voice_data):
        modell_speak("Starting transcription")
        voice_data=record_audio(ask="",filename='text.txt')
        modell_speak("I repead")
 
        modell_speak(voice_data)

        modell_speak('If incorrect say restart transcription else say exit')
        voice_data=record_audio()
        if there_exists(["restart","restart transcription"],voice_data):
            respond(voice_data)
        elif there_exists(["exit", "quit", "goodbye"],voice_data):
            modell_speak("bye")
            exit()
        else:
           modell_speak("bye")
           exit()

    if there_exists(["exit", "quit", "goodbye"],voice_data):
        modell_speak("bye")
        exit()
time.sleep(1)

modell_speak('How can I help you')

if __name__ == "__main__":
    while 1:

        voice_data=record_audio()
        print('Text: ',voice_data)
        modell_speak("bye")
        exit()
        #respond(voice_data)