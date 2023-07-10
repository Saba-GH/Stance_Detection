import numpy as np
import pandas as pd
import spacy
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import ModelCheckpoint

# Load and preprocess data for train and validation
data_train_val = pd.read_csv('Data_SemE_P/LA/train_preprocessed.csv')
texts_train_val = data_train_val['Tweet'].values
labels_train_val = data_train_val['Stance'].values

# Load and preprocess data for test
data_test = pd.read_csv('Data_SemE_P/LA/test_preprocessed.csv')
#data_test = pd.read_csv('Data_SemE_P/LA/test_preprocessed.csv')
texts_test = data_test['Tweet'].values

# Load the pretrained word embeddings model
nlp = spacy.load("en_core_web_md")

# Convert words in the dataset into word vectors
word_vectors_train_val = np.zeros((len(texts_train_val), nlp.vocab.vectors_length))
for i, text in enumerate(texts_train_val):
    doc = nlp(text)
    word_vectors_train_val[i] = doc.vector

word_vectors_test = np.zeros((len(texts_test), nlp.vocab.vectors_length))
for i, text in enumerate(texts_test):
    doc = nlp(text)
    word_vectors_test[i] = doc.vector

# Split data into train, validation, and test sets
word_vectors_train, word_vectors_val, labels_train, labels_val = train_test_split(
    word_vectors_train_val, labels_train_val, test_size=0.2, random_state=42)

# Reshape the input to have three dimensions
word_vectors_train = np.expand_dims(word_vectors_train, axis=1)
word_vectors_val = np.expand_dims(word_vectors_val, axis=1)
word_vectors_test = np.expand_dims(word_vectors_test, axis=1)

# Create the model
model = Sequential()
model.add(Bidirectional(LSTM(80, return_sequences=False), input_shape=(1, nlp.vocab.vectors_length)))
model.add(Dense(len(np.unique(labels_train_val)), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define checkpoint to save best model weights
checkpoint = ModelCheckpoint('weights/best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

# Encode labels
label_encoder = LabelEncoder()
labels_train = label_encoder.fit_transform(labels_train)
labels_val = label_encoder.transform(labels_val)
labels_test = label_encoder.transform(data_test['Stance'].values)

# Train the model
history = model.fit(word_vectors_train, labels_train, batch_size=16, epochs=100,
                    validation_data=(word_vectors_val, labels_val), callbacks=[checkpoint], verbose=1)

# Find best parameters
best_epoch = np.argmax(history.history['val_accuracy']) + 1
best_val_accuracy = np.max(history.history['val_accuracy'])
best_params = f"Best Epoch: {best_epoch}, Validation Accuracy: {best_val_accuracy}"

# Print best parameters
print(best_params)

# Load best model weights
model.load_weights('weights/best_model.h5')

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(word_vectors_test, labels_test, verbose=0)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
