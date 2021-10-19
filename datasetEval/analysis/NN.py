from keras import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout

epochs = 3
filters = 250
kernel_size = 3
strides = 1
lstm_units = 100
hide_u = 512
dropout_level = 0.5
num_categories = 3

model = Sequential()
model.add(Embedding(vocab_size, embed_dims, weights=[embed_matrix], input_length=seq_len, trainable=False))
model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=strides))
model.add(MaxPooling1D())
model.add(LSTM(lstm_units))
model.add(Dense(hide_u, activation='relu'))
model.add(Dropout(dropout_level))
model.add(Dense(num_categories, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])
