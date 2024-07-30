from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten 
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from tensorflow.keras.layers import MaxPool2D
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Attention
from keras.layers import GlobalMaxPooling1D, Dropout, concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical



df = pd.read_csv('C:/Users/Vaseema/Downloads/twitter depression.csv')

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['post_text'])
sequences = tokenizer.texts_to_sequences(df['post_text'])
word_index = tokenizer.word_index


data = pad_sequences(sequences, maxlen=100)


encoder = LabelEncoder()
labels = encoder.fit_transform(df['label'])
labels = to_categorical(labels)


X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)



input = Input(shape=(100,), dtype='int32')


embedding_layer = Embedding(len(word_index) + 1, 128, input_length=100)(input)


conv_layer = Conv1D(128, 5, activation='relu')(embedding_layer)
conv_layer = MaxPooling1D(pool_size=4)(conv_layer)


lstm_layer = Bidirectional(LSTM(128, return_sequences=True))(conv_layer)
flatten_layer = Flatten()(lstm_layer)  


dense_layer = Dense(64, activation='relu')(flatten_layer)
dropout_layer = Dropout(0.5)(dense_layer)
output = Dense(2, activation='softmax')(dropout_layer)


model = Model(inputs=input, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.summary()


history = model.fit(X_train, y_train, epochs=2, batch_size=64, validation_data=(X_test, y_test))
print('model trained successfully')
