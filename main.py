# AI Class Project: CSUF Portal Bot
# Peter Bergeon, Brian Edwards, Ryan Romero

# ----- Libraries here -----
import random
import json
import pickle
import numpy as np
import nltk
from training import *
from nltk.stem import WordNetLemmatizer
import tensorflow


# ----- Main -----
with open("intents.json") as file:
    data = json.load(file)

print("Booting up bot. (Type quit to stop)!")
while True:
        userString = input("You: ")
        if userString.lower() == "quit":
            break

        results = model.predict([bag_of_words(userString, words)])[0]
        result_index = np.argmax(results)
        tag = classes[result_index]

        if results[result_index] > 0.7:
            for tg in data["intents"]:
                if tg["tag"] == tag:
                    responses = tg['responses']

            print(random.choice(responses))
        else:
            print("Sorry, I didn' understand. Try asking again.")
