from django.shortcuts import render
from django.http import HttpResponse,HttpResponseRedirect
from .forms import UploadFileForm
from django.contrib.staticfiles.storage import staticfiles_storage
from .apps import SpeakerConfig


import os
import tensorflow
from tensorflow import keras
from keras.models import load_model
import pathlib
from pathlib import Path
import html

import librosa
import numpy as np
# Create your views here.

def index(request):
    print("check1")

    if request.method == 'POST':
        print("check2")

        form = UploadFileForm(request.POST, request.FILES)
        print("check3")
        # handle_uploaded_file(request.FILES['file'])
        if form.is_valid():
            print("check4")
            handle_uploaded_file(request.FILES['file'])
            return HttpResponseRedirect('/speaker')
    else:
        print("check5")
        form = UploadFileForm()
    return render(request, 'index.html', {'form': form})

def handle_uploaded_file(f):
   spk=getSpeaker(f)
   print("speker is")

   print(spk)


def results(request, question_id):
    response = "You're looking at the results of question %s."
    return HttpResponse(response % question_id)


def getSpeaker(f):
    print("1")

    test_X, sample_rate = librosa.load(f,res_type='kaiser_fast')

    mfccs = np.mean(librosa.feature.mfcc(y=test_X, sr=sample_rate, n_mfcc=40).T,axis=0)

    test_X = np.array(mfccs)
    test_X=test_X.reshape((1, 40))
    print(test_X)
    print("2")
    # url = staticfiles_storage.path('test_with_new_data.model')

    # model = load_model('test_with_new_data.model')
    print("3")
    MODEL_PATH = Path("./model/test_with_new_data.model")
    print(MODEL_PATH)
    model = load_model("./model/test_with_new_data") 

    predicted_label=SpeakerConfig.model.predict(test_X)
    print(predicted_label.argmax())

    labels = [' A. E. Maroney', ' Andrew NG', ' Anya', ' Arielle Lipshaw',
       ' Betty Chen', ' Bill Mosley', ' BookAngel7', ' Brendan Hodge',
       ' Brian von Dedenroth', ' Cata', ' Christie Nowak',
       ' David Mecionis', ' David Mix', ' Doug', ' E. Tavano', ' Hilara',
       ' Jean Bascom', ' Jeana Wei', ' Jennifer Wiginton',
       ' JenniferRutters', ' JenniferW', ' Jill Engle', ' John Rose',
       ' JudyGibson', ' Julie VW', ' JustinJYN', ' Kathy Caver',
       ' Lisa Meyers', ' M. Bertke', ' Malone', ' Mark Nelson',
       ' Mark Welch', ' Mary J', ' Michael Packard', ' Moromis',
       ' Nelly ()', ' Nicodemus', ' Peter Eastman', ' President Lethe',
       ' Ransom', ' Renata', ' Russ Clough', ' S R Colon',
       ' Scott Walter', ' Sharon Bautista', ' Simon Evers',
       ' Stephen Kinford', ' Steven Collins', ' Susan Hooks', ' Tonia',
       ' VOICEGUY', ' WangHaojie', ' Wayne Donovan', ' Wendy Belcher',
       ' Winston Tharp', ' aquielisunari', ' ashleyspence', ' badey',
       ' calystra', ' camelot2302', ' chocmuse', ' dexter', ' emmablob',
       ' fling93', ' iamartin', ' neelma', ' nprigoda', ' om123',
       ' ppezz', ' rohde', ' sid', ' spiritualbeing', ' thestorygirl',
       ' zinniz']

    print("4")


    return labels[predicted_label.argmax()]
    # return labels[1]