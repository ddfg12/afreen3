import nltk
import json
import pickle
import numpy as np
import random
ignore_words = ['?', '!',',','.', "'s", "'m"]
import tensorflow
from data_preprocessing import get_stem_words
model = tensorflow.keras.models.load_model('./chatbot_model.h5')
intents = json.loads(open('./intents.json').read())
words = pickle.load(open('./words.pkl','rb'))
classes = pickle.load(open('./classes.pkl','rb'))
def preprocess_user_input(user_input):
    i1=nltk.word_tokenize(user_input)
    i2=get_stem_words(i1,ignore_words)
    i2=sorted(list(set(i2)))


    bag = []
    bag_of_words = [] 
    for word_tags in words:

              
     if words in i2:              
            bag_of_words.append(1)
     else:
        bag_of_words.append(0)
    bag.append(bag_of_words)
    return np.array(bag)

def bot_class_prediction(user_input):
    inp = preprocess_user_input(user_input)
    prediction = model.predict(inp)
    predicted_class_label = np.argmax(prediction[0])
    return predicted_class_label

def bot_response(user_input):
    predicted_class_label = bot_class_prediction(user_input)
    predicted_class = classes[predicted_class_label]
    for intent in intents["intents"]:
        if intent["tag"]==predicted_class:
          bot_response=random.choice(intent["responses"])
          return bot_response
    
print("hi I am stala,how can I help you")
while True:
    user_input=input("type your message here")
    print("User Input: ", user_input)
    response = bot_response(user_input)
    print("Bot Response: ", response)