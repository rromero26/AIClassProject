# What Does This File Do?
#   this file is run before main and used to train the bot with the dataset in
#   the json file. Everytime dataset is modified, the bot must be re-trained. Only
#   need to run this file once to create the AI model. (rerun if dataset is modified)
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

# create lists. (Dataset from json file will be organized into these lists)
words = []              # will hold a complete list of keywords ('patterns' in json)
classes = []            # hold a list of all the tags in the json file
docs = []               # dictonary of tags and their coresponding list of keywords (tags/pattern)
ignore_characters = ['?', '!', '.', ',']

# parse through the data and save it into lists
for intent in data["intents"]:                      # parse every "tag" chunk
    for pattern in intent["patterns"]:              # grabs list of words in "patterns" of current "tag"
        word_List = nltk.word_tokenize(pattern)     # tokenize words
        words.extend(word_List)                     # add it to a complete list of words
        docs.append((word_List, intent["tag"]))     # assigns current list of words to the current "tag" (greetings = hello, hey, whats up, ...)

    # saves list of "tags" (classes)
    if intent["tag"] not in classes:                # (classes = greetings, bye, advising, ...)
        classes.append(intent["tag"])

# lematize (stem) "words" and remove duplicates in lists 'words" and "classes"
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_characters]
words = sorted(set(words))
classes = sorted(set(classes))

# create files to save "words" and "classes" lists
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

# CHECKPOINT:
#   So far, we have a complete List-of-root-Words that function like keywords.          ("words" list)
#   We also have a List-of-Tags which are the header of each chunk in the json file.    ("classes" list)
#   We also have a list that connect a Tag with it corresponding List-of-Keywords       ("doc" list)
# print("List of Tags: ", classes, "\n")
# print("Complete list of Words:", words, "\n")
# print("Docs dictonary: ", docs, "\n")
# exit()


# PREPARE DATA FOR BOT TRAINING:
#   Will be using a one-hot encoded strategy. We will create a binary list the same size
#   as the complete list of words, each bit representing a word in that list.
#   We have list of words and character but need numerical values to feed to
#   the neural network which trains the AI bot.
training = []
output_Empty = [0] * len(classes)          # inital binary list of classes (tags)

for document in docs:                       # create bag of words
    bag = []
    word_Pattern = document[0]              # document[0] = the list of key words of current tag
    word_Pattern = [lemmatizer.lemmatize(w.lower()) for w in word_Pattern]

    # compare each word in complete list of keywords to current tag's list of keywords
    for w in words:
        if w in word_Pattern:               # IF: it exist in current tag's list of words, append 1
            bag.append(1)
        else:                               # ELSE: does not exist in current tag's list of words, append 0
            bag.append(0)

    output_Row = list(output_Empty)
    output_Row[classes.index(document[1])] = 1
    training.append([bag, output_Row])
    # print("current Tag: ",document[1], " = ", output_Row, "\n")
    # print("Current Tag's bag-of-words: ",word_Pattern, " = ",  bag, "\n")

# CHECKPOINT:
#   Every Tag and it's corresponding Bag-of-Words have been turn to binary lists.
#   example:
#       Words = {hello, goodbye, morning, advising, schedule}
#           Greetings Bag    = {1, 0, 1, 0, 0}
#           Farewell Bag     = {0, 1, 0, 0, 0}
#           Advising bag     = {0, 0, 0, 1, 1}
#
#       Tags = {greetings, farewell, advising}
#           Greeting Tag      = {1, 0, 0}
#           Advising Tag      = {0, 0, 1}
#           Farewell Tag      = {0, 1, 0}
#
#   We then save the binary Tag and it corresponding binary Bag-of-Words into a
#   dictonary named 'training'
# print("Training Dictonary: ", training, "\n")
# exit()



# shuffle training data and converting it to array
random.shuffle(training)
training = np.array(training)

# spit training data into X (word bags) and Y (tags)
training_X = list(training[:, 0])
training_Y = list(training[:, 1])


# ---------------------------------------------------------------------------
# The AI (The neural network, copied from NeuralNine tutorial "Intelligent AI Chatbot")
#   todo: explain what going on

model = Sequential()
model.add(Dense(128, input_shape=(len(training_X[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(training_Y[0]), activation='softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
myModel = model.fit(np.array(training_X), np.array(training_Y), epochs=200, batch_size=5, verbose=1)

# the AI has been train, save training model into a file that can now be access
model.save('chatbot_Model.h5', myModel)
print("---Done---")
