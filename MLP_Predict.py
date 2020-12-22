import hyperopt
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
import pickle
import matplotlib.pyplot as pyplot

from hyperopt import Trials, STATUS_OK, tpe
import numpy as np
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD , Adam
from keras.optimizers import Adadelta, rmsprop
#from hyperas.distributions import choice, uniform
__author__ = 'Yanli Zhang-James'

from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras import regularizers
from keras.callbacks import EarlyStopping,CSVLogger
import numpy as np
import pandas as pd

import tensorflow as tf
from hyperopt import fmin, hp
import sys
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import os
os.chdir('/home/user/yanli/ENIGMA_CD/data/')
data_file = 'adhd_f100_train.csv'
valid_file = 'adhd_f100_valid.csv'
test_file = 'adhd_f100_test.csv'
all_file = 'adhd_f100_all.csv'
data = np.loadtxt(data_file, delimiter=",", skiprows=1)
valid_data = np.loadtxt(valid_file, delimiter=",", skiprows=1)
test_data = np.loadtxt(test_file, delimiter=",", skiprows=1)
all_data = np.loadtxt(all_file, delimiter=",", skiprows=1)

# SELECT FEATURES AND TARGET FROM DATA
x_train = data[:, 1:-1]
y_train = data[:, -1].reshape(-1,1)
print("training samples", x_train.shape, y_train.shape)
x_valid = valid_data[:, 1:-1]
y_valid = valid_data[:, -1].reshape(-1,1)
print("Validation samples", x_valid.shape, y_valid.shape)
x_test = test_data[:, 1:-1]
y_test = test_data[:, -1].reshape(-1,1)
print("Test Sample", x_test.shape, y_test.shape)

x_all = all_data[:, 1:-1]
y_all = all_data[:, -1].reshape(-1,1)
print("All Samples", x_all.shape, y_all.shape)
id_all= all_data[:, 0].reshape(-1, 1)

#scale features and targets
scaler = MinMaxScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_valid = scaler.transform(x_valid)
x_all = scaler.transform(x_all)

scaler_y=MinMaxScaler().fit(y_train)
y_train = scaler_y.transform(y_train)
y_test = scaler_y.transform(y_test)
y_valid = scaler_y.transform(y_valid)
y_all = scaler_y.transform(y_all)


input_dim = len(x_train[1])
output_dim = len(y_train[1])
print ("input", input_dim, "output", output_dim)

#Hyper-Parameters
layer1 = 179
layer2 = 479
layer3 = 287
dropout1 = 8919837252906864
dropout2 = .4717077179553653
dropout3 = .45801654819487025
batch_size = 240
optimizer = 'adadelta'


#MLP
input_img = Input(shape=(input_dim,))
deep = Dense(output_dim=layer1, activation='relu')(input_img)
deep = Dropout(dropout1)(deep)
deep = Dense(layer2, activation='relu')(deep)
deep = Dropout(dropout2)(deep)
deep = Dense(output_dim=layer3, activation='relu')(deep)
deep = Dropout(dropout3)(deep)
outlayer = Dense(output_dim=output_dim, activation='sigmoid')(deep)
model = Model(input_img, outlayer)

model.compile(loss=['binary_crossentropy'], metrics=['accuracy'],#, 'sparse_categorical_accuracy'
                        optimizer= optimizer)
print(model.metrics_names)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=50)
csvlogger = CSVLogger("HO_output.csv", separator=",", append = False)

history = model.fit(x_train, y_train,
                    nb_epoch=2000, #int(params['n_epochs']),
                    batch_size=batch_size,
                    shuffle=True,
                    verbose=2,
                    validation_data=(x_valid, y_valid),callbacks=[es,csvlogger])

#evaluate model
loss, acc= model.evaluate(x_valid, y_valid, verbose=0)
t_loss, t_acc= model.evaluate(x_train, y_train, verbose=0)
print("train loss:", t_loss, "valid loss", loss)
print("train acc:",t_acc, "valid acc", acc)

# plot loss during training
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['val_loss'], label='valid')
pyplot.plot(history.history['loss'], label='train')
pyplot.legend()
# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['val_accuracy'], label='valid')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.legend()
#pyplot.show()
pyplot.savefig('Child_F100AgeSexIQ_MLP_B.png')

#predictions
predict_all =model.predict(x_all)
print("prediction shape", predict_all.shape)
predict=np.hstack((id_all, y_all, predict_all))
print("output shape", predict.shape)
prediction=pd.DataFrame(predict)
columns=("ID", "ADHD", "prob_Child_F100AgeSexIQ_MLP_B")
prediction.to_csv("prob_Child_F100AgeSexIQ_MLP_B.csv", header=columns)





