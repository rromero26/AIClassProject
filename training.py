import random
import json
import numpy as np
import nltk
import tflearn
import tensorflow
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

with open("intents.json") as file:
    data = json.load(file)

words = []
classes = []
docs = []
ignore_characters = ['?', '!']

# SETS ALL LISTS ABOVE
for intent in data["intents"]:
    # set "words" and "docs" lists
    for pattern in intent["patterns"]:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        docs.append((w, intent["tag"]))

    # set "classes" list
    if intent["tag"] not in classes:
        classes.append(intent["tag"])

# remove duplicates in 'words" and "classes"
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_characters]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# CHECKPOINT: Test
print("documents: ", len(documents))
print("classes: ", len(classes), classes)
print("Unique words: ", len(words), words)


# TRAINING (Deep Learning)
training = []
output = []
out_empty = [0] * len(classes)

for doc in docs:
    bag = []
    word_pattern = doc[0]
    word_pattern = [lemmatizer.lemmatize(w.lower()) for w in word_pattern]
    for w in words:
        if w in word_pattern:
            bag.append(1)
        else:
            bag.append(0)

    output_row = list(out_empty)
    output_row[classes.index(doc[1])] = 1
    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

# The AI (simple neural network)
tensorflow.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

# Pass training data to our network
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")
