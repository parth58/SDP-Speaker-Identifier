ProjectSDP

https://kalliope-project.github.io/

https://www.ranks.nl/stopwords/hindi

https://towardsdatascience.com/how-to-use-google-speech-to-text-api-to-transcribe-long-audio-files-1c886f4eb3e9

https://www.analyticsvidhya.com/blog/2019/07/learn-build-first-speech-to-text-model-python/

https://scipy-cookbook.readthedocs.io/items/ApplyFIRFilter.html

https://towardsdatascience.com/illustrated-guide-to-recurrent-neural-networks-79e5eb8049c9

https://towardsdatascience.com/skip-gram-nlp-context-words-prediction-algorithm-5bbf34f84e0c

https://towardsdatascience.com/introduction-to-word-embedding-and-word2vec-652d0c2060fa

https://towardsdatascience.com/music-genre-classification-with-python-c714d032f0d8

https://www.analyticsvidhya.com/blog/2017/08/audio-voice-processing-deep-learning/


https://python-speech-features.readthedocs.io/en/latest/

####AudioFile To Text

# import speech_recognition as sr
# import webbrowser as wb
#
# r1 = sr.Recognizer()
# r2 = sr.Recognizer()
# r3 = sr.Recognizer()
# r4 = sr.Recognizer()
#
# with sr.Microphone() as source:
#     print('[search edureka : search video]')
#     print('Speak Now')
#     audio = r3.listen(source)
#     get = r3.recognize_google(audio)
#     print("You spoked:", get)

    # if 'edureka' in r2.recognize_google(audio):
    #     r2 = sr.Recognizer()
    #     url = 'https://www.edureka.co/'
    #     with sr.Microphone() as source:
    #         print('Search your Query')
    #         audio = r2.listen(source)
    #
    #         try:
    #             get = r2.recognize_google(audio)
    #             print(get)
    #             wb.get().open_new(url + get)
    #         except sr.UnknownValueError:
    #             print('error')
    #         except sr.RequestError as e:
    #             print('failed'.format(e))
    #
    # if 'video' or 'youtube' in r1.recognize_google(audio):
    #     r1 = sr.Recognizer()
    #     url = 'https://www.youtube.com/results?search_query='
    #     with sr.Microphone() as source:
    #         print('Search your Query')
    #         audio = r1.listen(source)
    #
    #         try:
    #             get = r1.recognize_google(audio)
    #             print(get)
    #             wb.get().open_new(url + get)
    #         except sr.UnknownValueError:
    #             print('error')
    #         except sr.RequestError as e:
    #             print('failed'.format(e))
    #
    # if 'open' in r4.recognize_google(audio):
    #     r4 = sr.Recognizer()
    #     url = "www.google.com/?results=query"
    #     with sr.Microphone as source:
    #         print('Search your query')
    #         audio = r4.listen(source)
    #
    #         try:
    #             get = r4.recognize_google(audio)
    #             print(get)
    #             wb.get().open_new(url + get)
    #         except sr.UnknownValueError:
    #             print('error')
    #         except sr.RequestError as e:
    #             print('failed'.format(e))

#Au -> Te Py
import speech_recognition as sr
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
def silence_based_conversion(path="alice-medium.wav"):
    song = AudioSegment.from_wav(path)
    fh = open("recognized.txt", "w+")
    chunks = split_on_silence(song,min_silence_len=500,silence_thresh=-16)
    try:
        os.mkdir('audio_chunks')
    except(FileExistsError):
        pass
    os.chdir('audio_chunks')
    i = 0
    for chunk in chunks:
        chunk_silent = AudioSegment.silent(duration=10)
        audio_chunk = chunk_silent + chunk + chunk_silent
        print("saving chunk{0}.wav".format(i))
        audio_chunk.export("./chunk{0}.wav".format(i), bitrate='192k', format="wav")
        filename = 'chunk' + str(i) + '.wav'
        print("Processing chunk " + str(i))
        file = filename
        r = sr.Recognizer()
        with sr.AudioFile(file) as source:
            r.adjust_for_ambient_noise(source)
            audio_listened = r.listen(source)
        try:
            rec = r.recognize_google(audio_listened)
            fh.write(rec + ". ")
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print("Could not request results. check your internet connection")
        i += 1
    os.chdir('..')

if __name__ == '__main__':
    silence_based_conversion(path)






# import speech_recognition as sr
# print(sr.__version__)
# r = sr.Recognizer()
#
# file_audio = sr.AudioFile('All of My Days.mp3.wav')
#
# with file_audio as source:
#    audio_text = r.record(source)
#
# print(type(audio_text))
# print(r.recognize_google(audio_text))




##transcribe.py 
import speech_recognition as sr
from os import path
from pydub import AudioSegment

# convert mp3 file to wav
sound = AudioSegment.from_mp3("Attention.mp3.mp3")
sound.export("transcript.wav", format="wav")


# transcribe audio file
AUDIO_FILE = "transcript.wav"

# use the audio file as the audio source
r = sr.Recognizer()
with sr.AudioFile(AUDIO_FILE) as source:
        audio = r.record(source)  # read the entire audio file

        print("Transcription: " + r.recognize_google(audio))




#Gujarati stopwords complete


import io
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

file = open("<<Your Stopword file.txt>>","r+",encoding="utf-8")

file1=file.read()
stop_words=word_tokenize(file1)
# for r in stop_words:
# 	i=r
# 	print(i)
file1 = open("<<Your Sentence file.txt>",encoding="utf-8")
line = file1.read()# Use this to read file content as a stream:
words = line.split()
for r in words:
	if r in stop_words:
		appendFile = open('<<Filtered sentences file.txt>>','a',encoding="utf-8")
		appendFile.write(" "+r)
		appendFile.close()



#Miscellaneous

# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
#
# example_sent = "Inside Bill's Brain. Decoding Bill Gates."
#
# stop_words = set(stopwords.words('english'))
#
# word_tokens = word_tokenize(example_sent)
#
# filtered_sentence = [w for w in word_tokens if not w in stop_words]
#
# filtered_sentence = []
#
# for w in word_tokens:
# 	if w not in stop_words:
# 		filtered_sentence.append(w)
#
# print(word_tokens)
# print(filtered_sentence)

# import io
# import nltk
# # nltk.download('punkt')
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
#
# #word_tokenize accepts a string as an input, not a file.
# stop_words = set(stopwords.words('english'))
# file1 = open("word.txt")
# line = file1.read()# Use this to read file content as a stream:
# words = line.split()
# for r in words:
# 	if not r in stop_words:
# 		appendFile = open('wordfile.txt','a')
# 		appendFile.write(" "+r)
# 		appendFile.close()

import io
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

file = open("hindi.txt","r+",encoding="utf-8")

file1=file.read()
stop_words=word_tokenize(file1)
# for r in stop_words:
# 	i=r
# 	print(i)
file1 = open("word.txt",encoding="utf-8")
line = file1.read()# Use this to read file content as a stream:
words = line.split()
for r in words:
	if r in stop_words:
		appendFile = open('wordfile.txt','a',encoding="utf-8")
		appendFile.write(" "+r)
		appendFile.close()

# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
#
# my_sent = "मफ़ोन बज रहा है क्या?"
# tokens = word_tokenize(my_sent)
#
# filtered_sentence = [w for w in tokens if not w in stopwords.words()]
# print(tokens)
# print(filtered_sentence)


#recognition.py


# importing libraries
# import speech_recognition as sr
# import os
# from pydub import AudioSegment
# from pydub.silence import split_on_silence
#
#
# # a function that splits the audio file into chunks
# # and applies speech recognition
# def silence_based_conversion(path="Surat (1).wav"):
#    # open the audio file stored in
#    # the local system as a wav file.
#    song = AudioSegment.from_wav(path)
#
#    # open a file where we will concatenate
#    # and store the recognized text
#    fh = open("recognized.txt", "w+")
#
#    # split track where silence is 0.5 seconds
#    # or more and get chunks
#    chunks = split_on_silence(song,
#                              # must be silent for at least 0.5 seconds
#                              # or 500 ms. adjust this value based on user
#                              # requirement. if the speaker stays silent for
#                              # longer, increase this value. else, decrease it.
#                              min_silence_len=500,
#
#                              # consider it silent if quieter than -16 dBFS
#                              # adjust this per requirement
#                              silence_thresh=-16
#                              )
#
#    # create a directory to store the audio chunks.
#    try:
#       os.mkdir('audio_chunks')
#    except(FileExistsError):
#       pass
#
#    # move into the directory to
#    # store the audio files.
#    os.chdir('audio_chunks')
#
#    i = 0
#    # process each chunk
#    for chunk in chunks:
#
#       # Create 0.5 seconds silence chunk
#       chunk_silent = AudioSegment.silent(duration=10)
#
#       # add 0.5 sec silence to beginning and
#       # end of audio chunk. This is done so that
#       # it doesn't seem abruptly sliced.
#       audio_chunk = chunk_silent + chunk + chunk_silent
#
#       # export audio chunk and save it in
#       # the current directory.
#       print("saving chunk{0}.wav".format(i))
#       # specify the bitrate to be 192 k
#       audio_chunk.export("./chunk{0}.wav".format(i), bitrate='192k', format="wav")
#
#       # the name of the newly created chunk
#       filename = 'chunk' + str(i) + '.wav'
#
#       print("Processing chunk " + str(i))
#
#       # get the name of the newly created chunk
#       # in the AUDIO_FILE variable for later use.
#       file = filename
#
#       # create a speech recognition object
#       r = sr.Recognizer()
#
#       # recognize the chunk
#       with sr.AudioFile(file) as source:
#          # remove this if it is not working
#          # correctly.
#          r.adjust_for_ambient_noise(source)
#          audio_listened = r.listen(source)
#
#       try:
#          # try converting it to text
#          rec = r.recognize_google(audio_listened)
#          # write the output to the file.
#          fh.write(rec + ". ")
#
#       # catch any errors.
#       except sr.UnknownValueError:
#          print("Could not understand audio")
#
#       except sr.RequestError as e:
#          print("Could not request results. check your internet connection")
#
#       i += 1
#
#    os.chdir('..')
#
#
# if __name__ == '__main__':
#
#    # print('Enter the audio file path')
#    # path = input()
#    path = "Surat (1).wav"
#
#    silence_based_conversion(path)



# import speech_recognition as sr
# print(sr.__version__)
# r = sr.Recognizer()
#
# file_audio = sr.AudioFile('Recording30.wav')
#
# with file_audio as source:
#    audio_text = r.record(source)
#
# print(type(audio_text))
# print(r.recognize_google(audio_text))


#recognition1.py

import random
import time

import speech_recognition as sr


def recognize_speech_from_mic(recognizer, microphone):
    """Transcribe speech from recorded from `microphone`.

    Returns a dictionary with three keys:
    "success": a boolean indicating whether or not the API request was
               successful
    "error":   `None` if no error occured, otherwise a string containing
               an error message if the API could not be reached or
               speech was unrecognizable
    "transcription": `None` if speech could not be transcribed,
               otherwise a string containing the transcribed text
    """
    # check that recognizer and microphone arguments are appropriate type
    if not isinstance(recognizer, sr.Recognizer):
        raise TypeError("`recognizer` must be `Recognizer` instance")

    if not isinstance(microphone, sr.Microphone):
        raise TypeError("`microphone` must be `Microphone` instance")

    # adjust the recognizer sensitivity to ambient noise and record audio
    # from the microphone
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    # set up the response object
    response = {
        "success": True,
        "error": None,
        "transcription": None
    }

    # try recognizing the speech in the recording
    # if a RequestError or UnknownValueError exception is caught,
    #     update the response object accordingly
    try:
        response["transcription"] = recognizer.recognize_google(audio)
    except sr.RequestError:
        # API was unreachable or unresponsive
        response["success"] = False
        response["error"] = "API unavailable"
    except sr.UnknownValueError:
        # speech was unintelligible
        response["error"] = "Unable to recognize speech"

    return response


if __name__ == "__main__":
    # set the list of words, maxnumber of guesses, and prompt limit
    WORDS = ["apple", "banana", "grape", "orange", "mango", "lemon"]
    NUM_GUESSES = 3
    PROMPT_LIMIT = 5

    # create recognizer and mic instances
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    # get a random word from the list
    word = random.choice(WORDS)

    # format the instructions string
    instructions = (
        "I'm thinking of one of these words:\n"
        "{words}\n"
        "You have {n} tries to guess which one.\n"
    ).format(words=', '.join(WORDS), n=NUM_GUESSES)

    # show instructions and wait 3 seconds before starting the game
    print(instructions)
    time.sleep(3)

    for i in range(NUM_GUESSES):
        # get the guess from the user
        # if a transcription is returned, break out of the loop and
        #     continue
        # if no transcription returned and API request failed, break
        #     loop and continue
        # if API request succeeded but no transcription was returned,
        #     re-prompt the user to say their guess again. Do this up
        #     to PROMPT_LIMIT times
        for j in range(PROMPT_LIMIT):
            print('Guess {}. Speak!'.format(i+1))
            guess = recognize_speech_from_mic(recognizer, microphone)
            if guess["transcription"]:
                break
            if not guess["success"]:
                break
            print("I didn't catch that. What did you say?\n")

        # if there was an error, stop the game
        if guess["error"]:
            print("ERROR: {}".format(guess["error"]))
            break

        # show the user the transcription
        print("You said: {}".format(guess["transcription"]))

        # determine if guess is correct and if any attempts remain
        guess_is_correct = guess["transcription"].lower() == word.lower()
        user_has_more_attempts = i < NUM_GUESSES - 1

        # determine if the user has won the game
        # if not, repeat the loop if user has more attempts
        # if no attempts left, the user loses the game
        if guess_is_correct:
            print("Correct! You win!".format(word))
            break
        elif user_has_more_attempts:
            print("Incorrect. Try again.\n")
        else:
            print("Sorry, you lose!\nI was thinking of '{}'.".format(word))
            break


#recognition2.py

# import speech_recognition as sr
# import webbrowser as wb
#
# r1 = sr.Recognizer()
# r2 = sr.Recognizer()
# r3 = sr.Recognizer()
# r4 = sr.Recognizer()
#
# with sr.Microphone() as source:
#     print('[search edureka : search video]')
#     print('Speak Now')
#     audio = r3.listen(source)
#     get = r3.recognize_google(audio)
#     print("You spoked:", get)

    # if 'edureka' in r2.recognize_google(audio):
    #     r2 = sr.Recognizer()
    #     url = 'https://www.edureka.co/'
    #     with sr.Microphone() as source:
    #         print('Search your Query')
    #         audio = r2.listen(source)
    #
    #         try:
    #             get = r2.recognize_google(audio)
    #             print(get)
    #             wb.get().open_new(url + get)
    #         except sr.UnknownValueError:
    #             print('error')
    #         except sr.RequestError as e:
    #             print('failed'.format(e))
    #
    # if 'video' or 'youtube' in r1.recognize_google(audio):
    #     r1 = sr.Recognizer()
    #     url = 'https://www.youtube.com/results?search_query='
    #     with sr.Microphone() as source:
    #         print('Search your Query')
    #         audio = r1.listen(source)
    #
    #         try:
    #             get = r1.recognize_google(audio)
    #             print(get)
    #             wb.get().open_new(url + get)
    #         except sr.UnknownValueError:
    #             print('error')
    #         except sr.RequestError as e:
    #             print('failed'.format(e))
    #
    # if 'open' in r4.recognize_google(audio):
    #     r4 = sr.Recognizer()
    #     url = "www.google.com/results?search_query"
    #     with sr.Microphone as source:
    #         print('Search your query')
    #         audio = r4.listen(source)
    #
    #         try:
    #             get = r4.recognize_google(audio)
    #             print(get)
    #             wb.get().open_new(url + get)
    #         except sr.UnknownValueError:
    #             print('error')
    #         except sr.RequestError as e:
    #             print('failed'.format(e))

import speech_recognition as sr
r = sr.Recognizer()

with sr.AudioFile("Voice 011.wav") as source:
   ## print('Say Something')
    audio = r.record(source,duration=15)
    print('Recognized Speech!!')
    text=r.recognize_google(audio,language="gu-IN")
    print(text)
    fo = open("wordfile.txt","w",encoding="utf-8")
    fo.write(text)
    fo.close()
    fo1=open("wordfile.txt","r",encoding="utf-8")
    print("Recognized text from File")
    text=fo1.read()
    print(text)
    if "ચાલો" == "ચાલો":
     print("True")
    if "નથીણાં" in text:
     print("Present")
    else:
     print("Not Present")

    # print(r.recognize_google(audio))
    # get = r.recognize_google(audio)
    # print("You spoked:", get)

# filepath = "Recording1.wav"
# output_filepath = "~/Transcripts/"
# bucketname = "callsaudiofiles"
# from pydub import AudioSegment
# import io
# import os
# from google.cloud import speech
# from google.cloud.speech import enums
# from google.cloud.speech import types
# import wave
# from google.cloud import storage
#
# def mp3_to_wav(audio_file_name):
#     if audio_file_name.split('.')[1] == 'mp3':
#         sound = AudioSegment.from_mp3(audio_file_name)
#         audio_file_name = audio_file_name.split('.')[0] + '.wav'
#         sound.export(audio_file_name, format="wav")
#
# def stereo_to_mono(audio_file_name):
#     sound = AudioSegment.from_wav(audio_file_name)
#     sound = sound.set_channels(1)
#     sound.export(audio_file_name, format="wav")
#
# def frame_rate_channel(audio_file_name):
#     with wave.open(audio_file_name, "rb") as wave_file:
#         frame_rate = wave_file.getframerate()
#         channels = wave_file.getnchannels()
#         return frame_rate,channels
#
# def upload_blob(bucket_name, source_file_name, destination_blob_name):
#     """Uploads a file to the bucket."""
#     storage_client = storage.Client()
#     bucket = storage_client.get_bucket(bucket_name)
#     blob = bucket.blob(destination_blob_name)
#
#     blob.upload_from_filename(source_file_name)
#
# def delete_blob(bucket_name, blob_name):
#     """Deletes a blob from the bucket."""
#     storage_client = storage.Client()
#     bucket = storage_client.get_bucket(bucket_name)
#     blob = bucket.blob(blob_name)
#
#     blob.delete()
#
#
# def google_transcribe(audio_file_name):
#     file_name = filepath + audio_file_name
#     mp3_to_wav(file_name)
#
#
#     frame_rate, channels = frame_rate_channel(file_name)
#
#     if channels > 1:
#         stereo_to_mono(file_name)
#
#     bucket_name = bucketname
#     source_file_name = filepath + audio_file_name
#     destination_blob_name = audio_file_name
#
#     upload_blob(bucket_name, source_file_name, destination_blob_name)
#
#     gcs_uri = 'gs://' + bucketname + '/' + audio_file_name
#     transcript = ''
#
#     client = speech.SpeechClient()
#     audio = types.RecognitionAudio(uri=gcs_uri)
#
#     config = types.RecognitionConfig(
#         encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
#         sample_rate_hertz=frame_rate,
#         language_code='en-US')
#
#     # Detects speech in the audio file
#     operation = client.long_running_recognize(config, audio)
#     response = operation.result(timeout=10000)
#
#     for result in response.results:
#         transcript += result.alternatives[0].transcript
#
#     delete_blob(bucket_name, destination_blob_name)
#     return transcript




#Wav file to Speechand Removing stop words

import io
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import speech_recognition as sr
r = sr.Recognizer()

with sr.AudioFile("Voice 011.wav") as source:
   ## print('Say Something')
    audio = r.record(source,duration=15)
    print('Recognized speech from file!')
    text=r.recognize_google(audio,language="gu-IN")
    print(text)
    fo = open("word.txt","w",encoding="utf-8")
    fo.write(text)
    fo.close()

file = open("GujStopWords.txt","r+",encoding="utf-8")
file1=file.read()
stop_words=word_tokenize(file1)
# for r in stop_words:
# 	i=r
# 	print(i)
file1 = open("word.txt",encoding="utf-8")
line = file1.read()# Use this to read file content as a stream:
words = line.split()
for r in words:
	if not r in stop_words:
		appendFile = open('wordfile.txt','a',encoding="utf-8")
		appendFile.write(" "+r)
		appendFile.close()

for r in words:
	if  r in stop_words:
		appendFile = open('wordfile1.txt','a',encoding="utf-8")
		appendFile.write(" "+r)
		appendFile.close()

# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
#
# my_sent = "मफ़ोन बज रहा है क्या?"
# tokens = word_tokenize(my_sent)
#
# filtered_sentence = [w for w in tokens if not w in stopwords.words()]
# print(tokens)
# print(filtered_sentence)

