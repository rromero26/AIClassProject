import random
import json
import numpy as np      # Must install library via command line
import tflearn
import tensorflow
import pickle
import nltk             # Must install library via command line
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Open json file and load in data into a variable
with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, classes, training, output = pickle.load(f)

except:
    # create dictionaries to separate data in
    words = []
    classes = []
    docs = []
    ignore_characters = ['?', '!', '.', ',']

    # Set all lists created above
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            # set "words" and "docs" lists
            w = nltk.word_tokenize(pattern)
            words.extend(w)
            docs.append((w, intent["tag"]))

        # set "classes" list
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

    # lematize and remove duplicates in 'words" and "classes" lists
    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_characters]
    words = sorted(list(set(words)))
    classes = sorted(list(set(classes)))


    # --------------------------------------------------------
    # CHECKPOINT:
    #   What we done: created and populated all the lists.
    print("documents: ", len(documents))
    print("classes: ", len(classes), classes)
    print("Unique words: ", len(words), words)
    # --------------------------------------------------------


    # TRAINING (Deep Learning)
    training = []
    output = []
    out_empty = [0] * len(classes)

    for document in docs:
        bag = []
        word_pattern = document[0]
        word_pattern = [lemmatizer.lemmatize(w.lower()) for w in word_pattern]
        for w in words:
            if w in word_pattern:
                bag.append(1)
            else:
                bag.append(0)

        output_row = list(out_empty)
        output_row[classes.index(document[1])] = 1
        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, classes, training, output), f)
# ---------------------------------------------------------------------------

# The AI (simple neural network)
tensorflow.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

# create/open training model
try:
    model.load("model.tflearn")

except:
    # Pass training data to our network
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words (s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = lemmatizer.lemmatize(w.lower()) for w in word_pattern

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)
