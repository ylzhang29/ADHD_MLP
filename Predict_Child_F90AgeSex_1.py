import hyperopt
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
import pickle
import matplotlib.pyplot as pyplot

from hyperopt import Trials, STATUS_OK, tpe
from keras.layers.core import Dense, Activation

from keras.optimizers import Adadelta, rmsprop
#from hyperas.distributions import choice, uniform
__author__ = 'Yanli Zhang-James'

from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras import optimizers
from keras.optimizers import SGD , Adam, Adamax,Nadam, Adagrad
from keras.optimizers import Adadelta, RMSprop

from keras.callbacks import EarlyStopping,CSVLogger, callbacks
from keras.callbacks import TerminateOnNaN
tn = TerminateOnNaN()
from keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd

import tensorflow as tf
from hyperopt import fmin, hp
import sys
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

#Hyper-Parameters
layer1 = 432
dropout1 = .21693555416001645
batch_size = 12
optimizer = optimizers.Adagrad(lr=.0002152780586205303)

# 'act': 'relu', 'batch_size': 12.0, 'optimizer': <class 'keras.optimizers.'>}

#Dataset
os.chdir('/home/user/yanli/ENIGMA_CD/data')
data_file = 'F90adhd_train.csv'
valid_file = 'F90adhd_valid.csv'
test_file = 'F90adhd_test.csv'
all_file = 'F90adhd_all.csv'
#use pandas to load data if there are string or missing values
df_train=pd.read_csv(data_file)
df_valid=pd.read_csv(valid_file)
df_test=pd.read_csv(test_file)
df_all =pd.read_csv(all_file)
print('train/validation/test/all set shapes', df_train.shape, df_valid.shape, df_test.shape, df_all.shape)
df_train=df_train[(df_train['source']==0)&(df_train['age']<18)]
df_valid=df_valid[(df_valid['source']==0)&(df_valid['age']<18)]
df_test=df_test[(df_test['source']==0)&(df_test['age']<18)]
df_all=df_all
print('shapes-child only: train/valid/test/all', df_train.shape, df_valid.shape, df_test.shape, df_all.shape)
# turn pd df to array
data	  = df_train.drop([ 'asd','source', 'iq'], axis=1).values
valid_data =  df_valid.drop([ 'asd','source', 'iq'], axis=1).values
test_data =  df_test.drop([ 'asd','source', 'iq'], axis=1).values
all_data =  df_all.drop([ 'asd','source', 'iq'], axis=1).values
print('shapes-child only: train/valid/test/all', df_train.shape, df_valid.shape, df_test.shape, df_all.shape)


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

input_dim = len(x_train[1])
output_dim = len(y_train[1])
print ("input", input_dim, "output", output_dim)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)
csvlogger = CSVLogger("HO_output_test.csv", separator=",", append = False)
checkpoint = ModelCheckpoint("Child_F90AgeSex_MLP1.hdf5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=5)

# define model
input_img = Input(shape=(input_dim,))
deep = Dense(output_dim=layer1, activation='relu')(input_img)
deep = Dropout(dropout1)(deep)
#deep = Dense(layer2, activation='relu')(deep)
#deep = Dropout(dropout2)(deep)
#deep = Dense(output_dim=layer3, activation='relu')(deep)
#deep = Dropout(dropout3)(deep)
outlayer = Dense(output_dim=output_dim, activation='sigmoid')(deep)
model = Model(input_img, outlayer)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=5000, 
		verbose=2, batch_size=batch_size,shuffle=False, 
		validation_data=(x_valid, y_valid),
		callbacks=[es,csvlogger, checkpoint])
# evaluate the model
test_loss, test_acc = model.evaluate(x_valid, y_valid, verbose=0)
print("validation loss and acc", test_loss, test_acc)
print("history", history)
print("history.history", history.history)



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
pyplot.savefig('Training_curve_Child_F90AgeSex.png')

#predictions
predict_all =model.predict(x_all)
print("prediction shape", predict_all.shape)
predict=np.hstack((id_all, y_all, predict_all))
print("output shape", predict.shape)
prediction=pd.DataFrame(predict)
columns=("uniqueID", "ADHD", "prob_Child_F90AgeSex")
prediction.to_csv("prob_Child_F90AgeSex.csv", header=columns)





