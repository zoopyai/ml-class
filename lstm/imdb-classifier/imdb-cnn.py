#dbg1 - changes and notes from classnotes
#dbg1 - when there is a loss of more than 7 or 8 it means network has instability issue
#dbg1 - that means it's not going to work 7 or 8 means infinity.

#dbg1 - if acc is higher than val_acc than we are over fitting.
#dbg1 - first question is 'am i over fitting'

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, MaxPooling1D, GRU
from keras.layers import Embedding, LSTM
from keras.layers import Conv1D, Flatten
from keras.datasets import imdb
import wandb
from wandb.keras import WandbCallback
import imdb
import numpy as np
from keras.preprocessing import text

wandb.init()
config = wandb.config

# set parameters:
config.vocab_size = 5000  # dbg1 - original 1000 - increase vocabulary should increase learning
config.maxlen = 1000
config.batch_size = 32
config.embedding_dims = 50
config.filters = 250
config.kernel_size = 3
config.hidden_dims = 250
config.epochs = 10

(X_train, y_train), (X_test, y_test) = imdb.load_imdb()
print("Review", X_train[0])
print("Label", y_train[0])

tokenizer = text.Tokenizer(num_words=config.vocab_size)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

X_train = sequence.pad_sequences(X_train, maxlen=config.maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=config.maxlen)
print(X_train.shape)
print("After pre-processing", X_train[0])

model = Sequential()
model.add(Embedding(config.vocab_size,
                    config.embedding_dims,
                    input_length=config.maxlen))
model.add(Dropout(0.5))

# # dbg1- for text this is 1D, for images it was 2D
# model.add(Conv1D(config.filters,
#                  config.kernel_size,
#                  padding='valid',
#                  activation='relu'))

# # dbg1 - to reduce time - add additional layer with pooling on both
# model.add(MaxPooling1D(2))
# # dbg1 
# model.add(Conv1D(config.filters,
#                  config.kernel_size,
#                  padding='valid',
#                  activation='relu'))

# # dbg1 - if 
# # dbg1 - on class one option is to add LSTM
# # dbg1 - 10 takes too much time, 
# # to reduce time, we can shrink the batch size OR reduce look back
# model.add(MaxPooling1D(2))
# model.add(LSTM(10, activation="sigmoid"))
#dbg1 - another option is to use the GRU that is fast
model.add(GRU(10, activation="sigmoid"))



# model.add(Flatten())
model.add(Dense(config.hidden_dims, activation='relu'))
model.add(Dropout(0.25))  # original was 0.5
# dbg1 add more dropout
model.add(Dense(config.hidden_dims, activation='relu'))
model.add(Dropout(0.25))
# dbg1 - original

# dbg1 - original - model.add(Dense(config.hidden_dims, activation='relu'))
# dbg1 - original - model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=config.batch_size,
          epochs=config.epochs,
          validation_data=(X_test, y_test), callbacks=[WandbCallback()])
