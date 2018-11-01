#dbg1 - class nov1 

# A very simple perceptron for classifying american sign language letters
import signdata
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.layers import Dense, Flatten, Dropout, BatchNormalization

from keras.utils import np_utils
import wandb
from wandb.keras import WandbCallback

# logging code
run = wandb.init()
config = run.config
config.loss = "categorical_crossentropy"
config.optimizer = "adam"
config.epochs = 10

# load data
(X_test, y_test) = signdata.load_test_data()
(X_train, y_train) = signdata.load_train_data()

#dbg1
# normalize data
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.


img_width = X_test.shape[1]
img_height = X_test.shape[2]

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_train.shape[1]

# you may want to normalize the data here..


#dbg2 - chris code
X_test = X_test.astype("float32") / 255.
X_train = X_train.astype("float32") / 255.
model=Sequential()
model.add(Reshape((28,28,1), input_shape=(28,28)))
model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.4))
model.add(Flatten(input_shape=(img_width, img_height)))
model.add(Dropout(0.4))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation="softmax"))
model.compile(loss=config.loss, optimizer=config.optimizer,
                metrics=['accuracy'])











# # create model
# model=Sequential()

# # dbg1 - this is 28x28x32 (32 feature map.  Conv size is 3x3, inputsize is 28x28x1).  this one is not 0 padded.
# # model.add(Conv2D(32,
# #     (3, 3),
# #     input_shape=(img_width, img_height, 1),
# #     activation='relu'))
# # model.add(MaxPooling2D(pool_size=(2, 2)))
# # # dbg1 - copy past above 3-4 lines to below... during class
# # # change see dbg1
# # model.add(Conv2D(32,
# #     (3, 3),
# #     input_shape=(img_width, img_height, 1),
# #     padding='same',  #dbg1 - added in class.  keras has a 0 padding to keep the input and output size same (by default conv-layer redue the output size)
# #     activation='relu'))
# # model.add(MaxPooling2D(pool_size=(2, 2)))
# # dbg1 - flatten evergtbing out
# model.add(Flatten())


# # model.add(Flatten(input_shape=(img_width, img_height)))
# model.add(Dense(100, activation="relu")) # dbg1 - this is 100 hidden node layer
# model.add(Dropout(0.25)) #dbg1 - add dropout
# model.add(Dense(100, activation="relu")) # dbg1 - this is 100 hidden node layer
# model.add(Dropout(0.25)) #dbg1 - add dropout
# model.add(Dense(num_classes, activation="softmax")) #dbg1 - add activation function
# model.add(Dropout(0.25))  #dbg1 - add activation function
# model.compile(loss=config.loss, optimizer=config.optimizer,
#                 metrics=['accuracy'])

# Fit the model
# model.fit(X_train, y_train, epochs=config.epochs, validation_data=(X_test, y_test), 
# 	callbacks=[WandbCallback(data_type="image", labels=signdata.letters)])


model.fit(X_train, y_train, validation_data=(X_test, y_test),
        epochs=config.epochs,
        callbacks=[WandbCallback(data_type="image", save_model=False)])

# # ----------------------- CNN.py
# # dbg1 - odsc class notes
# # 3x3+1 (1=bias term)s
# # 
# from keras.datasets import mnist
# from keras.models import Sequential
# # dbg1 - convolution layers (Max.. can also be Average), it's in keras docs.
# # https://keras.io/layers/convolutional/
# from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
# from keras.utils import np_utils
# from wandb.keras import WandbCallback
# import wandb

# run = wandb.init()
# config = run.config
# config.first_layer_convs = 32
# config.first_layer_conv_width = 3
# config.first_layer_conv_height = 3
# config.dropout = 0.5
# config.dense_layer_size = 128
# config.img_width = 28
# config.img_height = 28
# config.epochs = 10

# (X_train, y_train), (X_test, y_test) = mnist.load_data()

# X_train = X_train.astype('float32')
# X_train /= 255.
# X_test = X_test.astype('float32')
# X_test /= 255.

# #reshape input data
# # dbg1 - this is becasue our image is 60k, our conv. it's 3d (not 2d).  
# # dbg1 - conv. layer always expects the 3d.  essentially changing the state to 3d.
# X_train = X_train.reshape(X_train.shape[0], config.img_width, config.img_height, 1)
# X_test = X_test.reshape(X_test.shape[0], config.img_width, config.img_height, 1)

# # one hot encode outputs
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)
# num_classes = y_test.shape[1]
# labels=range(10)

# # build model
# model = Sequential()
# # dbg1 - this is 28x28x32 (32 feature map.  Conv size is 3x3, inputsize is 28x28x1).  this one is not 0 padded.
# model.add(Conv2D(32,
#     (config.first_layer_conv_width, config.first_layer_conv_height),
#     input_shape=(28, 28,1),
#     activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# # dbg1 - copy past above 3-4 lines to below... during class
# # change see dbg1
# model.add(Conv2D(32,
#     (config.first_layer_conv_width, config.first_layer_conv_height),
#     input_shape=(28, 28,1),
#     padding='same',  #dbg1 - added in class.  keras has a 0 padding to keep the input and output size same (by default conv-layer redue the output size)
#     activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# # dbg1 - flatten evergtbing out
# model.add(Flatten())
# # dbg1 - 128 pereptron with addition to relu.

# # dbg1 - we can add dropout
# model.add(Dropout(config.dropout))
# model.add(Dense(config.dense_layer_size, activation='relu'))
# model.add(Dropout(config.dropout))
# model.add(Dense(num_classes, activation='softmax'))

# model.compile(loss='categorical_crossentropy', optimizer='adam',
#                 metrics=['accuracy'])


# model.fit(X_train, y_train, validation_data=(X_test, y_test),
#         epochs=config.epochs,
#         callbacks=[WandbCallback(data_type="image", save_model=False)])
