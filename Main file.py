import nltk
import json
import pickle
import numpy as np
import random
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model

# Reading the first 30
def read_first_30_lines(file_path):
    with open(file_path, 'r') as f:
        for i in range(30):
            line = f.readline()
            if not line:
                break
            print(line.strip())

file_path = 'TestTrainEcom.json'
read_first_30_lines(file_path)

# preprocessing the data
lemmatizer = WordNetLemmatizer()

intents = json.loads(open('TestTrainEcom.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Convert bags and output rows to NumPy arrays
bags = np.array([item[0] for item in training])
output_rows = np.array([item[1] for item in training])


# Training the model

model = Sequential()
model.add(Dense(128, input_shape=(len(bags[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(output_rows[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(bags, output_rows, epochs=200, batch_size=5, verbose=1)
model.save('chatboxm.h5', hist)
print("done")


# Plot training accuracy over epochs
plt.plot(hist.history['accuracy'], color='blue', linestyle='-')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', alpha=0.7)  # Add grid lines
plt.legend(['Accuracy'], loc='lower right')  # Add legend
plt.show()

# Plot training loss over epochs
plt.plot(hist.history['loss'], color='red', linestyle='-')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.7)  # Add grid lines
plt.legend(['Loss'], loc='upper right')  # Add legend
plt.show()

# This part of the code is responsible for processing human text
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('TestTrainEcom.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatboxmodel.h5')
def inputText(msg):

    def clean_up_sentence(sentence):
        sentence_word = nltk.word_tokenize(sentence)
        sentence_word = [lemmatizer.lemmatize(word) for word in sentence_word]
        return sentence_word

    def bag_of_words(sentence):
        sentence_word = clean_up_sentence(sentence)
        bag = [0] * len(words)
        for w in sentence_word:
            for i, word in enumerate(words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)

    def predict_class(sentence):
        bow = bag_of_words(sentence)
        res = model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25

        results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

        results.sort(key=lambda x: x[1], reverse= True)
        return_list = []
        for r in results:
            return_list.append({'intent':classes[r[0]],'probability': str(r[1])})
        return return_list

    def get_response(intents_list, intents_json):
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                  result = random.choice(i['responses'])
                  break
        return result

    while True:
        ints = predict_class(msg)
        res = get_response(ints, intents)
        return(res)


# Feel free to give inputs from here
print("Hi, the bot is up, please feel free to greet me about any products or, any queries !Stop to end my service.")

while True:
    inputtxt = input()
    if inputtxt == "!Stop":
        print("Stopping the program...")
        break
    else:
        outputtxt = inputText(str(inputtxt))
        print(outputtxt)
