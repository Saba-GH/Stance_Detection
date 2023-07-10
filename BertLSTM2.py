import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Bidirectional, GlobalMaxPooling1D, concatenate
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, TFBertModel

# Load and preprocess data for train and validation
data_train_val = pd.read_csv('Data_SemE_P/LA/train_preprocessed.csv')
texts_train_val = data_train_val['Tweet'].values
labels_train_val = data_train_val['Stance'].values

# Load and preprocess data for test
data_test = pd.read_csv('Data_SemE_P/LA/test_preprocessed.csv')
texts_test = data_test['Tweet'].values

# Split data into train and validation sets
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

# Load BERT tokenizer and model
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
model_bert = TFBertModel.from_pretrained('bert-base-uncased')

# Tokenize and encode BERT input for train and validation
input_ids_train = []
attention_masks_train = []
input_ids_val = []
attention_masks_val = []

for text_train in texts_train:
    encoded = tokenizer_bert.encode_plus(text_train, add_special_tokens=True, truncation=True, padding='max_length',
                                         max_length=max_sequence_length, return_tensors='tf')
    input_ids_train.append(encoded['input_ids'][0])
    attention_masks_train.append(encoded['attention_mask'][0])

for text_val in texts_val:
    encoded = tokenizer_bert.encode_plus(text_val, add_special_tokens=True, truncation=True, padding='max_length',
                                         max_length=max_sequence_length, return_tensors='tf')
    input_ids_val.append(encoded['input_ids'][0])
    attention_masks_val.append(encoded['attention_mask'][0])

input_ids_train = np.array(input_ids_train)
attention_masks_train = np.array(attention_masks_train)
input_ids_val = np.array(input_ids_val)
attention_masks_val = np.array(attention_masks_val)

# Tokenize and encode BERT input for test
input_ids_test = []
attention_masks_test = []

for text_test in texts_test:
    encoded = tokenizer_bert.encode_plus(text_test, add_special_tokens=True, truncation=True, padding='max_length',
                                         max_length=max_sequence_length, return_tensors='tf')
    input_ids_test.append(encoded['input_ids'][0])
    attention_masks_test.append(encoded['attention_mask'][0])

input_ids_test = np.array(input_ids_test)
attention_masks_test = np.array(attention_masks_test)

# Create the LSTM model
lstm_input = Input(shape=(max_sequence_length,))
lstm_embed = Embedding(len(tokenizer.word_index) + 1, 50)(lstm_input)
lstm_lstm = Bidirectional(LSTM(80, return_sequences=True))(lstm_embed)
lstm_pool = GlobalMaxPooling1D()(lstm_lstm)
lstm_dense = Dense(64, activation='relu')(lstm_pool)

# Create the BERT model
bert_input_ids = Input(shape=(max_sequence_length,), dtype='int32')
bert_attention_masks = Input(shape=(max_sequence_length,), dtype='int32')
bert_output = model_bert(bert_input_ids, attention_mask=bert_attention_masks)[1]
bert_dense = Dense(64, activation='relu')(bert_output)

# Concatenate LSTM and BERT layers
concat = concatenate([lstm_dense, bert_dense])
output = Dense(len(label_encoder.classes_), activation='softmax')(concat)

# Create the combined model
model = Model(inputs=[lstm_input, bert_input_ids, bert_attention_masks], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([sequences_train, input_ids_train, attention_masks_train], labels_train,
          batch_size=16, epochs=30, validation_data=([sequences_val, input_ids_val, attention_masks_val], labels_val), verbose=1)

# Evaluate the model on test set
test_loss, test_accuracy = model.evaluate([sequences_test, input_ids_test, attention_masks_test], labels_test)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
