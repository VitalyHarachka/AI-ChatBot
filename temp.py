# -*- coding: utf-8 -*-
"""
Spyder Editor

some of the code i used is not mime i fount some on keras.io here is the link: https://keras.io/examples/lstm_seq2seq_restore/

This is a temporary script file.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic chatbot design --- for your own modifications

please uncomment nltk.download('punkt') and
 nltk.download('wordnet') when using the first-time
"""
#######################################################
# Initialise Wikipedia agent
#######################################################

import wikipediaapi
wiki_wiki = wikipediaapi.Wikipedia('en')
wikipediaapi.log.setLevel(level=wikipediaapi.logging.ERROR)

#######################################################
# Initialise similarity baced questions
#######################################################
import os
import io
import nltk
import numpy as np
import random
import re
import string
import cv2
from PIL import Image
import gym

from itertools import islice

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf
from tensorflow import keras

import tkinter as tk
from tkinter import filedialog
from keras.models import load_model
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Embedding, Reshape
from keras.optimizers import Adam

 

from keras.models import Model, load_model
from keras.layers import Input
import numpy as np
#from google.colab import files
#files.download('marvel_cnn.h5')
######################################################


file1 = open('heros.txt','r+',errors = 'ignore')
filer = file1.read()
filer = filer.lower()# converts all text to lowercase

    
folval = nltk.Valuation.fromstring(filer)

        
grammar_file = 'simple-sem.fcfg'
objectCounter = 0
######

ENV_NAME = "Taxi-v3"
env = gym.make(ENV_NAME)
#env.render()

action = env.action_space.n
states = env.observation_space.n



#######


file = open('chatbot.txt','r',errors = 'ignore')
fr = file.read()
fr = fr.lower()# converts all text to lowercase



nltk.download('punkt') # only for first-time use 
nltk.download('wordnet') # only for first-time use 

sent_tokens = nltk.sent_tokenize(fr)# converts to list of sentences 
word_tokens = nltk.word_tokenize(fr)# converts to list of words

link = nltk.stem.WordNetLemmatizer()  # links words with similar meaning to one word.

#WordNet is a semantically-oriented dictionary of English included in NLTK.
def LemTokens(tokens):
    return [link.lemmatize(token) for token in tokens]
remove_punct = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct)))


def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    
    if(req_tfidf==0):
        robo_response=robo_response+" "
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx + 1]
        return robo_response
#I am sorry! I don't understand you


def errorInPut():
        params[1] = params[1].replace(" ","")
        params[2] = params[2].replace(" ","")
        params[1] = re.sub('[^0-9a-zA-Z]+', '', params[1])
        params[2] = re.sub('[^0-9a-zA-Z]+', '', params[2])
        
def deepL(environment, states, actions):
    
    model = Sequential()
    model.add(Embedding(500, 10, input_length=1))
    model.add(Reshape((10,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(action, activation='linear'))
    #print(model.summary())
    
    model.load_weights("dqn_avengers_map.h5f")
    
    from rl.agents.dqn import DQNAgent
    from rl.policy import EpsGreedyQPolicy
    from rl.memory import SequentialMemory
    
    memory = SequentialMemory(limit=50000, window_length=1)
    
    policy = EpsGreedyQPolicy()
    
    dqn = DQNAgent(model=model, 
                   nb_actions=action,
                   memory=memory, 
                   nb_steps_warmup=500,
                   target_model_update=1e-2,
                   policy=policy)
    
    dqn.compile(Adam(lr=0.001), metrics=['mae'])
    
    dqn.test(env, nb_episodes=1, visualize=True, nb_max_episode_steps=99)
    
    
    
        
    

categories = ['black widow', 'captain america', 'doctor strange', 'hulk', 'ironman','loki', 'spider-man', 'thanos']

#######################################################
#  Initialise AIML agent
#######################################################

import aiml
# Create a Kernel object. No string encoding (all I/O is unicode)
kern = aiml.Kernel()
kern.setTextEncoding(None)
# Use the Kernel's bootstrap() method to initialize the Kernel. The
# optional learnFiles argument is a file (or list of files) to load.
# The optional commands argument is a command (or list of commands)
# to run after the files are loaded.
# The optional brainFile argument specifies a brain file to load.
kern.bootstrap(learnFiles="mybot-basic.xml")

#######################################################
# Welcome user
#######################################################
print("Welcome to the Marvel universe chat bot. Please feel free to ask questions about",
      "Marvel characters, a look into their history and there characteristics,fun facts and the movies that they have been featured in. ")
print(" ")
print("You can crate your own team of heroes whith the charachters below: ")
for i in islice(folval,0,14):
        print(i)
print(" ")
print("The teams you can put the charactes in are:  ")
for i in islice(folval,14,17):
        print(i)

#######################################################
# Main loop
#######################################################


    

while True:
    #get user input
    try:
        user_response = input("> ")
    except (KeyboardInterrupt, EOFError) as e:
        print("Bye!")
    if user_response =="image":       
        root = tk.Tk()
        root.withdraw()
        image = filedialog.askopenfilename()
        model = keras.models.load_model('marvel_cnn.h5')

        img = mpimg.imread(image)
        img = cv2.resize(img,(60, 60))
        img = np.array(img, dtype = 'float32')
        img/=255
        img = np.reshape(img,(1, 60, 60, 3))
        prediction = model.predict(img)
        #print(numpy.argmax(prediction))
        predResult = np.argmax(prediction[0])
        print(categories[predResult])
        
        
        
        
        
    #pre-process user input and determine response agent (if needed)
    responseAgent = 'aiml'
    #activate selected response agent
    if responseAgent == 'aiml':
        answer = kern.respond(user_response)
        params = answer[1:].split('$')
        
    #post-process the answer for commands
    if answer[0] == '#':
        params = answer[1:].split('$')
        
        cmd = int(params[0])
        if cmd == 0:
            print(params[1])
            break
        elif cmd == 1:
            wpage = wiki_wiki.page(params[1])
            if wpage.exists():
                print(wpage.summary)
                print("Learn more at", wpage.canonicalurl)
            #else:
                #print("Sorry, I don't know what that is.")
                
    

                

        elif cmd == 4: # I will put x in y
                errorInPut()
                if params[1] in folval:
                    if params[2] in folval:
                        o = 'o' + str(objectCounter)
                        objectCounter += 1
                        folval['o' + o] = o #insert constant
                        if len(folval[params[1]]) == 1: #clean up if necessary
                                if ('',) in folval[params[1]]:
                                        folval[params[1]].clear()
                        folval[params[1]].add((o,)) #insert type of hero information
                        if len(folval["be_in"]) == 1: #clean up if necessary
                                if ('',) in folval["be_in"]:
                                        folval["be_in"].clear() 
                        folval["be_in"].add((o, folval[params[2]])) #insert team
                    else:
                        print("I did not get that, please check the team's spelling and try again.")
                else:
                    print("I did not get that, please check the heroes name  spelling and try again.")
                        
        



        elif cmd == 5: # IS * IN *
            errorInPut()
            if params[1] in folval:
                if params[2] in folval:
                    g = nltk.Assignment(folval.domain)
                    m = nltk.Model(folval.domain, folval)
                    sent = 'some ' + params[1] + ' are_in ' + params[2]
                    results = nltk.evaluate_sents([sent], grammar_file, m, g)[0][0]
                    if results[2] == True:
                        print("Yes.")
                    else:
                        print("No.")
                else:
                    print("I did not get that, please check the team's spelling and try again.")
            else:
                print("I did not get that, please check the heroes name spelling and try again.")


                
        elif cmd == 6: # is x only in y
                errorInPut()
                if params[1] in folval:
                    if params[2] in folval:
                        g = nltk.Assignment(folval.domain)
                        m = nltk.Model(folval.domain, folval)
                        sent = 'all ' + params[1] + ' are_in ' + params[2]
                        results = nltk.evaluate_sents([sent], grammar_file, m, g)[0][0]
                        if results[2] == True:
                            print("Yes.")
                        else:
                            print("No.")
                    else:
                        print("I did not get that, please check the team's spelling and try again.")
                else:
                    print("I did not get that, please check the heroes name spelling and try again.")


                
        elif cmd == 7: # Which heroes are in ...
                params[1] = params[1].replace(" ","")
                params[1] = re.sub('[^0-9a-zA-Z]+', '', params[1])
                if params[1] in folval:
                        g = nltk.Assignment(folval.domain)
                        m = nltk.Model(folval.domain, folval)
                        e = nltk.Expression.fromstring("be_in(x," + params[1] + ")")
                        sat = m.satisfiers(e, "x", g)
                        if len(sat) == 0:
                                print("None.")
                        else:
                                #find satisfying objects in the valuation dictionary,
                                #and print their type names
                                sol = folval.values()
                                for so in sat:
                                        for k, v in folval.items():
                                                if len(v) > 0:
                                                        vl = list(v)
                                                        if len(vl[0]) == 1:
                                                                for i in vl:
                                                                        if i[0] == so:
                                                                                print(k)
                                                                                break
                else:
                        print("I did not get that, please check the team's spelling and try again.")
        
        elif cmd == 8:
            ENV_NAME = "Taxi-v3"
            env = gym.make(ENV_NAME)
            #env.render()

            action = env.action_space.n
            states = env.observation_space.n
            
            deepL(env, states, action)
            
        elif cmd == 9:
            userInput = input("Enter English Word: ")
            batch_size = 64  # Batch size for training.
            epochs = 100  # Number of epochs to train for.
            latent_dim = 256  # Latent dimensionality of the encoding space.
            num_samples = 10000  # Number of samples to train on.
            # Path to the data txt file on disk.
            data_path = 'C:\\Users\\Vitaly\\Documents\\ai\\deu.txt'
            
            # Vectorize the data.  We use the same approach as the training script.
            # NOTE: the data must be identical, in order for the character -> integer
            # mappings to be consistent.
            # We omit encoding target_texts since they are not needed.
            input_texts = []
            target_texts = []
            input_characters = set()
            target_characters = set()
            with open(data_path, 'r', encoding='utf-8') as f:
                lines = f.read().split('\n')
            for line in lines[: min(num_samples, len(lines) - 1)]:
                input_text, target_text = line.split('\t')
                # We use "tab" as the "start sequence" character
                # for the targets, and "\n" as "end sequence" character.
                target_text = '\t' + target_text + '\n'
                input_texts.append(input_text)
                target_texts.append(target_text)
                for char in input_text:
                    if char not in input_characters:
                        input_characters.add(char)
                for char in target_text:
                    if char not in target_characters:
                        target_characters.add(char)
            
            input_characters = sorted(list(input_characters))
            target_characters = sorted(list(target_characters))
            num_encoder_tokens = len(input_characters)
            num_decoder_tokens = len(target_characters)
            max_encoder_seq_length = max([len(txt) for txt in input_texts])
            max_decoder_seq_length = max([len(txt) for txt in target_texts])
            
#            print('Number of samples:', len(input_texts))
#            print('Number of unique input tokens:', num_encoder_tokens)
#            print('Number of unique output tokens:', num_decoder_tokens)
#            print('Max sequence length for inputs:', max_encoder_seq_length)
#            print('Max sequence length for outputs:', max_decoder_seq_length)
            
            input_token_index = dict(
                [(char, i) for i, char in enumerate(input_characters)])
            target_token_index = dict(
                [(char, i) for i, char in enumerate(target_characters)])
            
            encoder_input_data = np.zeros(
                (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
                dtype='float32')
            
            for i, input_text in enumerate(input_texts):
                for t, char in enumerate(input_text):
                    encoder_input_data[i, t, input_token_index[char]] = 1.


            
            # Restore the model and construct the encoder and decoder.
            model = load_model('s2s.h5')
            
            encoder_inputs = model.input[0]   # input_1
            encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output   # lstm_1
            encoder_states = [state_h_enc, state_c_enc]
            encoder_model = Model(encoder_inputs, encoder_states)
            
            decoder_inputs = model.input[1]   # input_2
            decoder_state_input_h = Input(shape=(latent_dim,), name='input_3')
            decoder_state_input_c = Input(shape=(latent_dim,), name='input_5')
            decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
            decoder_lstm = model.layers[3]
            decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
                decoder_inputs, initial_state=decoder_states_inputs)
            decoder_states = [state_h_dec, state_c_dec]
            decoder_dense = model.layers[4]
            decoder_outputs = decoder_dense(decoder_outputs)
            decoder_model = Model(
                [decoder_inputs] + decoder_states_inputs,
                [decoder_outputs] + decoder_states)
            
            # Reverse-lookup token index to decode sequences back to
            # something readable.
            reverse_input_char_index = dict(
                (i, char) for char, i in input_token_index.items())
            reverse_target_char_index = dict(
                (i, char) for char, i in target_token_index.items())
            
            
            # Decodes an input sequence.  Future work should support beam search.
            def decode_sequence(input_seq):
                # Encode the input as state vectors.
                states_value = encoder_model.predict(input_seq)
            
                # Generate empty target sequence of length 1.
                target_seq = np.zeros((1, 1, num_decoder_tokens))
                # Populate the first character of target sequence with the start character.
                target_seq[0, 0, target_token_index['\t']] = 1.
            
                # Sampling loop for a batch of sequences
                # (to simplify, here we assume a batch of size 1).
                stop_condition = False
                decoded_sentence = ''
                while not stop_condition:
                    output_tokens, h, c = decoder_model.predict(
                        [target_seq] + states_value)
            
                    # Sample a token
                    sampled_token_index = np.argmax(output_tokens[0, -1, :])
                    sampled_char = reverse_target_char_index[sampled_token_index]
                    decoded_sentence += sampled_char
            
                    # Exit condition: either hit max length
                    # or find stop character.
                    if (sampled_char == '\n' or
                       len(decoded_sentence) > max_decoder_seq_length):
                        stop_condition = True
            
                    # Update the target sequence (of length 1).
                    target_seq = np.zeros((1, 1, num_decoder_tokens))
                    target_seq[0, 0, sampled_token_index] = 1.
            
                    # Update states
                    states_value = [h, c]
            
                return decoded_sentence
            
            
            #for seq_index in range(100):
                # Take one sequence (part of the training set)
                # for trying out decoding.
               
               
                #input_seq = encoder_input_data[seq_index: seq_index + 1]
                #decoded_sentence = decode_sequence(input_seq)
                
                #if input_seq == userInput:
#                print('-')
#                print('Input sentence:', input_texts[seq_index])
                print('translation:', decoded_sentence)
                
        elif user_response != None:
            print(response(user_response))
            sent_tokens.remove(user_response)
            
        elif cmd == 99:
            print("I did not get that, please try again.")
    else:
        print(answer)

