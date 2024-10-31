import random
import collections
import glob
import matplotlib.pyplot as plt

import librosa
from sklearn.preprocessing import StandardScaler

import tensorflow

import numpy as np
from numpy.matlib import repmat
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
import pandas as pd

import pretty_midi
#import fluidsynth

# dataset parameters
copies = 200       # number of times to copy the song
window_length = 12  # number of notes used to predict the next note
# neural network parameters
nn_nodes = 10      # number of nodes in the RNN
# training parameters

##How many times the set of equations would be solved together
##epochs = 200         # number of epochs used for training

epochs = 2
##How many data points are fed to the NN at the same time
batch_size = None   # size of each batch (None is default)

def dataset_from_song(song,copies,window_length):
    # repeat song "copies" times
    songs = repmat(song,1,copies)[0]
    # number of windows used
    num_windows = len(songs) - window_length

    x_train,y_train = [],[]
    for i in range(num_windows):
        # get a "window_length" number of notes
        x_train.append(songs[i:i+window_length])
        # get the note after the window
        y_train.append(songs[i+window_length])

    # convert to numpy arrays
    x_train = np.array(x_train,dtype='float32')
    x_train = np.expand_dims(x_train,axis=-1)
    y_train = np.array(y_train,dtype='float32')
    y_train = np.expand_dims(y_train,axis=-1)

    return x_train,y_train

##Importing a song
file_name = r"W. A. Mozart, Symphony No.38 in D major - A Far Cry.mp3"
audio, sr = librosa.load(file_name, sr=None)

song = audio [100000:110000]
# a scale
#song = np.array([72,74,76,77,79,81,83,84])
# generate a dataset from copies of the song
x_train,y_train = dataset_from_song(song,copies,window_length)

##Normalization
scaler = StandardScaler()
scaler2 = StandardScaler()
##x_train_scaled_data = scaler.fit_transform(x_train.reshape(len(x_train[:,0,0])),-1)
# Fit the scaler on the reshaped training data (fit)
x_train_reshaped = x_train.reshape(len(x_train[:, 0, 0]), window_length)
scaler.fit(x_train_reshaped)
scaler2.fit(y_train)
# Transform the training data (transform)
x_train_scaled_data = scaler.transform(x_train_reshaped)
y_train_scaled = scaler2.transform(y_train)
x_train_scaled_data = x_train_scaled_data.reshape(len(x_train[:,0,0]),window_length, 1)

# specify the architecture of the neural network
##Defining a NN in Tensor Flow
model = Sequential()
##Once defined, add the layers you want
model.add(SimpleRNN(nn_nodes,activation='relu'))
##Dense is a fully connected layer
##we pass 1 because we need 1 output(the result)
model.add(Dense(1,activation=None))

# setup the neural network
model.compile(
    loss='MeanSquaredError',
    optimizer='Adam',
    metrics=[])
# use this to save the best weights found so far
# AC: can we set this so that the loss has to be less than 0.001
callback = tf.keras.callbacks.EarlyStopping(
    monitor='loss',patience=10,restore_best_weights=True)


# train the neural network from data
history = model.fit(x_train_scaled_data,y_train_scaled,
                    batch_size=batch_size,epochs=epochs,
                    callbacks=[callback])
print("finished training:")
model.evaluate(x_train_scaled_data,y_train_scaled)

#predictions = model.predict(x_train_scaled_data).round()
predictions = model.predict(x_train_scaled_data)
correct = sum(predictions == y_train_scaled)
N = len(predictions)
print("train set accuracy: %.4f%% (%d/%d)" % (100*correct/N,correct,N))


plt.plot(predictions[:1000],"b",label="predictions")
plt.plot(y_train_scaled[:1000],"r",label="true")
plt.legend()
plt.savefig("Training.png")



# shift the scale up by 3 half-steps
test_song = song + 3
# reverse the scale
#test_song = song[-1::-1]
x_test,y_test = dataset_from_song(test_song,3,window_length)


# Transform the training data (transform)
x_test_reshaped = x_test.reshape(len(x_test[:, 0, 0]), window_length)
x_test_scaled_data = scaler.transform(x_test_reshaped)
y_test_scaled = scaler2.transform(y_test)
x_test_scaled_data = x_test_scaled_data.reshape(len(x_test[:,0,0]),window_length, 1)

##predictions = model.predict(x_test).round()

predictions = model.predict(x_test_scaled_data)
correct = sum(predictions == y_test_scaled)
N = len(predictions)
print(" test set accuracy: %.4f%% (%d/%d)" % (100*correct/N,correct,N))
plt.clf()
plt.plot(predictions[:1000],"b",label="predictions")
plt.plot(y_test_scaled[:1000],"r",label="true")
plt.legend()
plt.savefig("Testing.png")