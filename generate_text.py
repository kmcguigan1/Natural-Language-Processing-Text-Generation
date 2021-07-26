import requests
import numpy as np

import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Bidirectional, LSTM

IRISH_POETRY = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/irish-lyrics-eof.txt"

def request_data(dataset_url):
    # send the get request to the server
    response = requests.get(dataset_url)
    # convert the response to text format
    datastore = response.text
    return datastore

def read_data(dataset_url):
    # get the datastore itself
    datastore = request_data(dataset_url)
    # convert and split the data
    datastore = datastore.lower().split("\n")
    return datastore

def get_tokenizer(datastore):
    # create the tokenizer
    tokenizer = Tokenizer()
    # fit the tokenizer
    tokenizer.fit_on_texts(datastore)
    # get the total number of words, add one for the oov token
    total_words = len(tokenizer.word_index) + 1
    return tokenizer, total_words

def sentences_to_sequences(tokenizer, sentences):
    # initialize the input sequences array
    input_sequences = []
    # iterate over the lines in the sentences
    for line in sentences:
        # get the token list for this line
        token_list = tokenizer.texts_to_sequences([line])[0]
        # iterate i from 1 to the total length of the tokenized list
        for i in range(1, len(token_list)):
            # create a sequence from the start of the sequence until the i-th entry in the sequence
            n_gram_sequence = token_list[:i+1]
            # append this n_gram_sequence to the input_sequences list
            input_sequences.append(n_gram_sequence)
    # get the length of the longest input sequence in the list
    max_sequence_len = max([len(x) for x in input_sequences])
    return input_sequences, max_sequence_len

def pad_input_sequences(input_sequences, max_length):
    # pad the sequences in the input list
    padded_sequences = pad_sequences(input_sequences, maxlen=max_length, padding='pre')
    # convert this list to a numpy array
    padded_sequences = np.array(padded_sequences)
    return padded_sequences

def data_to_train_data(input_sequences, total_words):
    # get all the words leading up to the last word
    xs = input_sequences[:, :-1]
    # get the last word in every sequence
    labels = input_sequences[:, -1]
    # transform these labels into categories
    ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)
    return xs, ys

def create_model(total_words, max_sequence_len):
    model = Sequential()
    model.add(Embedding(total_words, 240, input_length=max_sequence_len - 1))
    model.add(Bidirectional(LSTM(150)))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model.predict

def train_model(model, xs, ys):
    # train the model
    hist = model.fit(xs, ys, epochs=100, verbose=1)
    return hist

def generate_text(model, tokenizer, max_sequence_len, next_words=20):
    seed_text = "I made a poetry machine"
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if(index == predicted):
                output_word = word
                break
        seed_text += " " + output_word
    return

seed_text = "I made a poetry machine"
next_words = 20

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict_classes(token_list, verbose=0)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if(index == predicted):
            output_word = word
            break
    seed_text += " " + output_word
print(seed_text)

def main():
    
    return

main()