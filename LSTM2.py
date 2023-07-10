import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import ModelCheckpoint

# Load and preprocess data for train and validation
data_train_val = pd.read_csv('Data_SemE_P/LA/train_preprocessed.csv')
texts_train_val = data_train_val['Tweet'].values
labels_train_val = data_train_val['Stance'].values

# Load and preprocess data for test
#data_test = pd.read_csv('Data_SemE_P/LA/test_preprocessed.csv')
data_test = pd.read_csv('Data_SemE_P/LA/re_test_preprocessed.csv')

texts_test = data_test['Tweet'].values

# Split data into train, validation sets
texts_train, texts_val, labels_train, labels_val = train_test_split(
    texts_train_val, labels_train_val, test_size=0.2, random_state=42)

# Tokenize text and convert to sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts_train)
sequences_train = tokenizer.texts_to_sequences(texts_train)
sequences_val = tokenizer.texts_to_sequences(texts_val)
sequences_test = tokenizer.texts_to_sequences(texts_test)

# Padding sequences 
max_sequence_length = max(max(len(seq) for seq in sequences_train),
                          max(len(seq) for seq in sequences_val),
                          max(len(seq) for seq in sequences_test))
sequences_train = pad_sequences(sequences_train, maxlen=max_sequence_length)
sequences_val = pad_sequences(sequences_val, maxlen=max_sequence_length)
sequences_test = pad_sequences(sequences_test, maxlen=max_sequence_length)

# Encoding labels
label_encoder = LabelEncoder()
labels_train = label_encoder.fit_transform(labels_train)
labels_val = label_encoder.transform(labels_val)
labels_test = label_encoder.transform(data_test['Stance'].values)

# Creating the model
model = Sequential()
model.add(Embedding(len(tokenizer.word_index) + 1, 50, input_length=max_sequence_length))
model.add(Bidirectional(LSTM(80, return_sequences=False)))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Compiling the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Defining checkpoint/saving best model weights
checkpoint = ModelCheckpoint('weights/best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

# Training the model
history = model.fit(sequences_train, labels_train, batch_size=64, epochs=300,
                    validation_data=(sequences_val, labels_val), callbacks=[checkpoint], verbose=1)

# Finding the best parameters
best_epoch = np.argmax(history.history['val_accuracy']) + 1
best_val_accuracy = np.max(history.history['val_accuracy'])
best_params = f"Best Epoch: {best_epoch}, Validation Accuracy: {best_val_accuracy}"

# Printing best parameters
print(best_params)

# Loading best model weights
model.load_weights('weights/best_model.h5')

# Evaluating the model on the test set
test_loss, test_accuracy = model.evaluate(sequences_test, labels_test, verbose=0)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
