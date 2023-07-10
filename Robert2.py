import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import RobertaTokenizer, TFRobertaModel

# Loading data for training and validation
data_train_val = pd.read_csv('Data_SemE_P/LA/train_preprocessed.csv')
texts_train_val = data_train_val['Tweet'].values
labels_train_val = data_train_val['Stance'].values

# Loading data for test
data_test = pd.read_csv('Data_SemE_P/LA/test_preprocessed.csv')
#data_test = pd.read_csv('Data_SemE_P/LA/test_preprocessed.csv')
texts_test = data_test['Tweet'].values

# Splitting data into train, validation
texts_train, texts_val, labels_train, labels_val = train_test_split(
    texts_train_val, labels_train_val, test_size=0.2, random_state=42)

# Tokenize text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts_train)
sequences_train = tokenizer.texts_to_sequences(texts_train)
sequences_val = tokenizer.texts_to_sequences(texts_val)
sequences_test = tokenizer.texts_to_sequences(texts_test)

# Pad sequences to have the same length
max_sequence_length = max(max(len(seq) for seq in sequences_train),
                          max(len(seq) for seq in sequences_val),
                          max(len(seq) for seq in sequences_test))
sequences_train = pad_sequences(sequences_train, maxlen=max_sequence_length)
sequences_val = pad_sequences(sequences_val, maxlen=max_sequence_length)
sequences_test = pad_sequences(sequences_test, maxlen=max_sequence_length)

# Encode labels
label_encoder = LabelEncoder()
labels_train = label_encoder.fit_transform(labels_train)
labels_val = label_encoder.transform(labels_val)
labels_test = label_encoder.transform(data_test['Stance'].values)

# Load RoBERTa tokenizer and model
tokenizer_roberta = RobertaTokenizer.from_pretrained('roberta-base')
model_roberta = TFRobertaModel.from_pretrained('roberta-base')

# Tokenize and encode RoBERTa input
input_ids_train = []
attention_masks_train = []
input_ids_val = []
attention_masks_val = []
input_ids_test = []
attention_masks_test = []

for text_train in texts_train:
    encoded = tokenizer_roberta.encode_plus(text_train, add_special_tokens=True, truncation=True, padding='max_length',
                                            max_length=max_sequence_length, return_tensors='tf')
    input_ids_train.append(encoded['input_ids'][0])
    attention_masks_train.append(encoded['attention_mask'][0])

for text_val in texts_val:
    encoded = tokenizer_roberta.encode_plus(text_val, add_special_tokens=True, truncation=True, padding='max_length',
                                            max_length=max_sequence_length, return_tensors='tf')
    input_ids_val.append(encoded['input_ids'][0])
    attention_masks_val.append(encoded['attention_mask'][0])

for text_test in texts_test:
    encoded = tokenizer_roberta.encode_plus(text_test, add_special_tokens=True, truncation=True, padding='max_length',
                                            max_length=max_sequence_length, return_tensors='tf')
    input_ids_test.append(encoded['input_ids'][0])
    attention_masks_test.append(encoded['attention_mask'][0])

input_ids_train = np.array(input_ids_train)
attention_masks_train = np.array(attention_masks_train)
input_ids_val = np.array(input_ids_val)
attention_masks_val = np.array(attention_masks_val)
input_ids_test = np.array(input_ids_test)
attention_masks_test = np.array(attention_masks_test)

# Create the RoBERTa model
roberta_input_ids = Input(shape=(max_sequence_length,), dtype='int32')
roberta_attention_masks = Input(shape=(max_sequence_length,), dtype='int32')
roberta_output = model_roberta(roberta_input_ids, attention_mask=roberta_attention_masks)[0][:, 0, :]
roberta_dense = Dense(64, activation='relu')(roberta_output)

# Create the output layer
output = Dense(len(label_encoder.classes_), activation='softmax')(roberta_dense)

# Create the model
model = Model(inputs=[roberta_input_ids, roberta_attention_masks], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([input_ids_train, attention_masks_train], labels_train,
          batch_size=16, epochs=30, validation_data=([input_ids_val, attention_masks_val], labels_val), verbose=1)

# Evaluate the model on the test set
loss, accuracy = model.evaluate([input_ids_test, attention_masks_test], labels_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')
