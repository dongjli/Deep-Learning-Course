from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils

n_image = 28
n_labels = 10

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
n_image = 28
x_train = X_train.reshape(X_train.shape[0], n_image*n_image)
x_test = X_test.reshape(X_test.shape[0], n_image*n_image)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = np_utils.to_categorical(Y_train, n_labels)
y_test = np_utils.to_categorical(Y_test, n_labels)

##############  Single  Layer  #################
model = Sequential()
model.add(Dense(10, input_shape=(784,)))
model.add(Activation('softmax'))  
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=500, epochs=100,
          verbose=1, validation_data=(x_test, y_test))



###Layer: [(50, ReLU), (50, ReLU), (10, Linear)] ###
model = Sequential()
model.add(Dense(50, input_shape=(784,)))
model.add(Activation('relu')) 
                       
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax')) 

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=500, epochs=100,
          verbose=1, validation_data=(x_test, y_test))
