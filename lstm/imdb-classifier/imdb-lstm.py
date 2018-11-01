#dbg1 - on class and notes
#dbg1 - Bidirectional LSTM was used in QA and it's best practice ... see class notes.

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, LSTM, Bidirectional
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
config.vocab_size = 1000
config.maxlen = 300
config.batch_size = 32
config.embedding_dims = 50
config.filters = 250
config.kernel_size = 3
config.hidden_dims = 100
config.epochs = 10

(X_train, y_train), (X_test, y_test) = imdb.load_imdb()

tokenizer = text.Tokenizer(num_words=config.vocab_size)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

X_train = sequence.pad_sequences(X_train, maxlen=config.maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=config.maxlen)

model = Sequential()
model.add(Embedding(config.vocab_size,
                    config.embedding_dims,
                    input_length=config.maxlen), 
                    trainable=False)  # dbg1 s== adding trainable = False


# model.add(Embedding(config.vocab_size,
#                     config.embedding_dims,
#                     input_length=config.maxlen))



#dbg2 - for bidirectional
# model.add(Bidirectional(LSTM(config.hidden_dims, activation="sigmoid")))


model.add(LSTM(config.hidden_dims, activation="sigmoid", return_sequences=True))
#dbg1 - added one more layer (deep LSTM - see class notes)
#dbg1 - the dimentions by default above LSTM is only giving it's final output
#dbg1 - all other dimentions ... what we wanted is two dimentional i.e., on all "I love this movie" (see pdf)
#dbg1 - you can see keras docs and see... we need return state
#dbg1 - 
model.add(LSTM(config.hidden_dims, activation="sigmoid"))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=config.batch_size,
          epochs=config.epochs,
          validation_data=(X_test, y_test), callbacks=[WandbCallback()])
