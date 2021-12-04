# What does this file do?
#   this file is run before main and used to train the bot with the dataset in
#   the json file. Everytime dataset is modified, the bot must be retrain.
#   After running training.py. 3 files will be created:
#       words.pkl, classes.pkl, chatbot_Model.h5

import random
import json
import pickle
import numpy as np                              # Must install library via command line
import nltk                                     # Must install library via command line

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential  # Must install library via command line
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

# Open json file and load in the dataset into variable "data"
with open("intents.json") as file:
    data = json.load(file)

# create lists. Dataset from json file will be organized into these lists
words = []
classes = []
docs = []
ignore_characters = ['?', '!', '.', ',']

# parse through the data
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        # populate lists "words" and "docs"
        word_List = nltk.word_tokenize(pattern)
        words.extend(word_List)
        docs.append((word_List, intent["tag"]))

    # spopulate list "classes"
    if intent["tag"] not in classes:
        classes.append(intent["tag"])

# lematize and remove duplicates in 'words" and "classes" lists
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_characters]
words = sorted(set(words))
classes = sorted(set(classes))

# create files to dump "words" and "classes" lists
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))


# TRAINING (Deep Learning)
training = []
output_Empty = [0] * len(classes)

for document in docs:
    bag = []
    word_Pattern = document[0]
    word_Pattern = [lemmatizer.lemmatize(w.lower()) for w in word_Pattern]
    for w in words:
        if w in word_Pattern:
            bag.append(1)
        else:
            bag.append(0)

    output_Row = list(output_Empty)
    output_Row[classes.index(document[1])] = 1
    training.append([bag, output_Row])

# shuffle training data and converting it to array
random.shuffle(training)
training = np.array(training)

# spit training data into X and Y values
training_X = list(training[:, 0])
training_Y = list(training[:, 1])


# ---------------------------------------------------------------------------
# The AI (The neural network, copied from NeuralNine tutorial "Intelligent AI Chatbot")

model = Sequential()
model.add(Dense(128, input_shape=(len(training_X[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(training_Y[0]), activation='softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
myModel = model.fit(np.array(training_X), np.array(training_Y), epochs=200, batch_size=5, verbose=1)

model.save('chatbot_Model.h5', myModel)
print("Done")

# ---------------------------------------------------------------------------

# The AI (simple neural network)
