import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, TFBertModel

# Loading data for Training/Validation
data_train_val = pd.read_csv('Data_SemE_P/LA/train_preprocessed.csv')
texts_train_val = data_train_val['Tweet'].values
labels_train_val = data_train_val['Stance'].values

# Test Set (2 different test datasets)
#data_test = pd.read_csv('Data_SemE_P/LA/test_preprocessed.csv')
data_test = pd.read_csv('Data_SemE_P/LA/re_test_preprocessed.csv')
texts_test = data_test['Tweet'].values
labels_test = data_test['Stance'].values

# Splitting data into train and validation sets
texts_train, texts_val, labels_train, labels_val = train_test_split(texts_train_val, labels_train_val, test_size=0.2, random_state=42)

# Load BERT tokenizer and model
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
model_bert = TFBertModel.from_pretrained('bert-base-uncased')

# Tokenize and encode BERT input for train and validation data
input_ids_train = []
attention_masks_train = []
input_ids_val = []
attention_masks_val = []
input_ids_test = []
attention_masks_test = []

max_sequence_length = 0

# Tokenize and encode train data
for text_train in texts_train:
    encoded = tokenizer_bert.encode_plus(text_train, add_special_tokens=True, truncation=True, padding='max_length',
                                         max_length=max_sequence_length, return_tensors='tf')
    input_ids_train.append(encoded['input_ids'][0])
    attention_masks_train.append(encoded['attention_mask'][0])
    max_sequence_length = max(max_sequence_length, len(encoded['input_ids'][0]))

# Tokenize and encode validation data
for text_val in texts_val:
    encoded = tokenizer_bert.encode_plus(text_val, add_special_tokens=True, truncation=True, padding='max_length',
                                         max_length=max_sequence_length, return_tensors='tf')
    input_ids_val.append(encoded['input_ids'][0])
    attention_masks_val.append(encoded['attention_mask'][0])

# Tokenize and encode test data
for text_test in texts_test:
    encoded = tokenizer_bert.encode_plus(text_test, add_special_tokens=True, truncation=True, padding='max_length',
                                         max_length=max_sequence_length, return_tensors='tf')
    input_ids_test.append(encoded['input_ids'][0])
    attention_masks_test.append(encoded['attention_mask'][0])

input_ids_train = np.array(input_ids_train)
attention_masks_train = np.array(attention_masks_train)
input_ids_val = np.array(input_ids_val)
attention_masks_val = np.array(attention_masks_val)
input_ids_test = np.array(input_ids_test)
attention_masks_test = np.array(attention_masks_test)

# Create the BERT model
bert_input_ids = Input(shape=(max_sequence_length,), dtype='int32')
bert_attention_masks = Input(shape=(max_sequence_length,), dtype='int32')
bert_output = model_bert(bert_input_ids, attention_mask=bert_attention_masks)[1]
bert_dense = Dense(64, activation='relu')(bert_output)

# Encode labels
label_encoder = LabelEncoder()
labels_train = label_encoder.fit_transform(labels_train)
labels_val = label_encoder.transform(labels_val)
labels_test = label_encoder.transform(labels_test)

# Create the output layer
output = Dense(len(label_encoder.classes_), activation='softmax')(bert_dense)

# Create the model
model = Model(inputs=[bert_input_ids, bert_attention_masks], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([input_ids_train, attention_masks_train], labels_train,
          batch_size=16, epochs=30, validation_data=([input_ids_val, attention_masks_val], labels_val), verbose=1)

# Evaluate the model on the test data
_, test_accuracy = model.evaluate([input_ids_test, attention_masks_test], labels_test, verbose=0)
print(f"Test Accuracy: {test_accuracy}")
