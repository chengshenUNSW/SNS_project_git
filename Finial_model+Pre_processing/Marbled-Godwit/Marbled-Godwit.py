import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import normalize, to_categorical
import time
import datetime
from numpy.random import randn
from numpy.random import randint
import keras
from keras.callbacks import LearningRateScheduler
from sklearn.model_selection import GridSearchCV
from keras.constraints import maxnorm
import keras_tuner as kt
from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Dropout

###############################################################################################
# Input the data
data = pd.read_csv('marbledGodwit_USGS_ASC_argos.csv');

b=data.shape;   #shape
# print(b)

# filter out one animols
animal_ID_MAGO_7H = data[data['individual-local-identifier']=='MAGO_7H']
# animal_ID_MAGO_7H

# timestamp
animal_ID_MAGO_7H.timestamp = pd.to_datetime(animal_ID_MAGO_7H.timestamp)
# animal_ID_MAGO_7H.head()

# set index (datetime)
animal_ID_MAGO_7H = animal_ID_MAGO_7H.set_index('timestamp')
# animal_ID_MAGO_7H.head()

# sort the index
animal_ID_MAGO_7H.sort_index(ascending = True).head()

# the location of animal
animal_ID_MAGO_7H_pre_data = animal_ID_MAGO_7H[animal_ID_MAGO_7H.columns[2:4]]

a = animal_ID_MAGO_7H_pre_data.resample('D').mean()
animal_ID_MAGO_7H_data = a.resample('D').interpolate('linear')

###############################################################################################
pre_days =10
animal_ID_MAGO_7H_data['label_1'] = animal_ID_MAGO_7H_data['location_long'].shift(-pre_days)
animal_ID_MAGO_7H_data['label_2'] = animal_ID_MAGO_7H_data['location_lat'].shift(-pre_days)
# Standardlise
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
data_scalar = scalar.fit_transform(animal_ID_MAGO_7H_data.iloc[:,:-2])
# print(data_scalar)

###############################################################################################
# memory days
memory_days = 10

from collections import deque
deq = deque(maxlen=memory_days)

# processing queue--input features
queue_input = []
for i in data_scalar:
    deq.append(list(i))
    if len(deq)==memory_days:
        queue_input.append(list(deq))
# print(len(queue_input))

# processing the final misssing data ( cut the pre_days data )
queue_last = queue_input[-pre_days:]
queue_input = queue_input[:-pre_days]
# print(len(queue_input))

queue_output = animal_ID_MAGO_7H_data.iloc[:,2:4].values[memory_days-1:-pre_days]
# print(queue_output)

###############################################################################################
# transfer list to array
queue_input = np.array(queue_input)
# print(queue_input.shape)

queue_output = np.array(queue_output)
# print(queue_output.shape)

###############################################################################################
# split into traning and test datasets
from sklearn.model_selection import train_test_split
queue_input_train,queue_input_test,queue_output_train,queue_output_test = train_test_split(queue_input,queue_output,test_size=0.2)

###############################################################################################
# split into traning and test datasets
from sklearn.model_selection import train_test_split
queue_input_train,queue_input_test,queue_output_train,queue_output_test = train_test_split(queue_input,queue_output,shuffle=False,test_size=0.2)

from tensorflow import keras
from tensorflow.keras import layers
from kerastuner import HyperParameters

metric_str = 'val_mse'


def build_model(hp):
    # All HPs

    drop_out_rate = hp.Float('dropout_rate', 0, 0.5)

    hp_LSTM_input_units = hp.Int('LSTM_input_units', 32, 128, 32)
    hp_LSTM_input_act = hp.Choice('LSTM_input_activation', ['relu', 'tanh'])

    hp_LSTM_internal_layers = hp.Int('LSTM_num_layers', 0, 3, 1)
    # units and avtivation for internal LSTM layers are defined inside the loop

    hp_LSTM_last_units = hp.Int('LSTM_last_units', 32, 128, 32)
    hp_LSTM_last_act = hp.Choice('LSTM_last_activation', ['relu', 'tanh'])

    hp_Dense_layers = hp.Int('Dense_num_layers', 1, 3, 1)
    # units and avtivation for dense layers are defined inside the loop

    hp_optimizers = hp.Choice('optimizer', ['adam', 'rmsprop'])

    model = Sequential()
    model.add(layers.LSTM(units=hp_LSTM_input_units,
                          return_sequences=True,
                          input_shape=(memory_days, 2),
                          activation=hp_LSTM_input_act
                          ))

    for i in range(hp_LSTM_internal_layers):
        model.add(layers.LSTM(units=hp.Int(f"LSTM_internal_{i}_units", 32, 128, 32),
                              activation=hp.Choice(f"LSTM_internal_{i}_activation", ['relu', 'tanh']),
                              # f"activation_{i}"
                              return_sequences=True))
        model.add(Dropout(drop_out_rate))

    model.add(layers.LSTM(units=hp_LSTM_last_units,
                          activation=hp_LSTM_last_act))
    model.add(Dropout(drop_out_rate))

    for i in range(hp_Dense_layers):
        model.add(Dense(units=hp.Int(f"Dense_{i}_units", 32, 128, 32),
                        activation=hp.Choice(f"Dense_{i}_activation", ['relu', 'tanh', 'sigmoid'])))
        model.add(Dropout(drop_out_rate))
    model.add(layers.Dense(2))

    model.compile(optimizer=hp_optimizers,
                  loss='mse',
                  metrics=['mse'])
    return model

# configure the Tuner
from kerastuner import RandomSearch

tuner = RandomSearch(
    build_model,
    objective=metric_str,
    max_trials=500,
    executions_per_trial=3,
    directory='my_dir',
    overwrite=True,
    project_name='helloworld')

# set callback
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, verbose=1)

# start searching
tuner.search(queue_input_train,queue_output_train, epochs=50, validation_data=(queue_input_test,queue_output_test), callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
model = tuner.hypermodel.build(best_hps)
history = model.fit(queue_input_train,queue_output_train,epochs=50,validation_data=(queue_input_test,queue_output_test))

# perform prediction:
test_predict = model.predict(queue_input_test)
train_predict = model.predict(queue_input_train)

# plot
import matplotlib.pyplot as plt
data_time = animal_ID_MAGO_7H_data.index[-len(queue_output_test):]
plt.plot(data_time,queue_output_test,color='red')
plt.plot(data_time,test_predict,color='green')
#plt.plot(data_time,train_predict_1,color='black')
plt.show()

# save model for server use
import os
import joblib
queue_input_=np.row_stack([queue_input,queue_last])
dirs = 'result_shamoun'
if not os.path.exists(dirs):
    os.makedirs(dirs)
joblib.dump(model, dirs+'/hypermodel.pkl')
joblib.dump(animal_ID_MAGO_7H_data, dirs+'/results_data.txt')
joblib.dump(queue_input_, dirs+'/results_queue_.txt')
joblib.dump(memory_days, dirs+'/memory_days.txt')
joblib.dump(pre_days, dirs+'/pre_days.txt')

# save prediction results for plot
import os
import joblib
time_str = datetime.datetime.now().strftime("%B_%d_%I%M%p")
dirs_vars = 'predict_data_shamoun'+time_str
if not os.path.exists(dirs_vars):
    os.makedirs(dirs_vars)
joblib.dump(data_time, dirs_vars+'/datetime1.pkl')
joblib.dump(queue_output_test, dirs_vars+'/truth_data.txt')
joblib.dump(test_predict, dirs_vars+'/predict_data.txt')

# save parameters for local references
time_str = datetime.datetime.now().strftime("%B_%d_%I%M%p")
model_name = 'model_Shamoun_'+time_str+'.h5'
model.save(model_name)

# Evaluate accuracy scores
import math
from sklearn.metrics import mean_squared_error
train_score = math.sqrt(mean_squared_error(queue_output_train[0],train_predict[0,:]))
print('Train Score: %.2f RMSE' % (train_score))

test_score = math.sqrt(mean_squared_error(queue_output_test[0], test_predict[0,:]))
print('Test Score: %.2f RMSE' % (test_score))
