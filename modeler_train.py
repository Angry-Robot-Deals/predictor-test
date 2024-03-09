#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !pip uninstall keras_tuner
# !pip uninstall keras-nightly tf-nightly[and-cuda]
# !pip uninstall keras
# !pip uninstall tensorflow
# !pip uninstall autokeras
# !pip install tensorflow==2.15
# !pip install autokeras
# !pip install -y scikit-learn autokeras gputil psutil humanize


# In[ ]:


# load learning model libraries

import absl.logging
import os
absl.logging.set_verbosity('fatal')  # 'error' warnings 'fatal' mute all except ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # '0' show all, '1' mute INFO, '2' mute INFO and WARNING, '3' mute all except ERROR
# os.environ["KERAS_BACKEND"] = "tensorflow"

# Set CUDA_VISIBLE_DEVICES if we want to use CPU or GPU
# USE_GPU = True
# os.environ["CUDA_VISIBLE_DEVICES"] = 'None' if USE_GPU else '' # '' - for CPU, 'None' - for GPU

import warnings
warnings.filterwarnings("ignore")
import inspect
import logging
import traceback

import sys
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import gc  # garbage collector

import time
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta

import autokeras as ak
import tensorflow as tf
from tensorflow.keras import backend as K  # keras backend functions
from tensorflow.keras.utils import to_categorical  # convert to OHE
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

import psutil as ps  # library for retrieving information on running processes and system utilization
import humanize as hm  # library for turning a number into a fuzzy human-readable
import GPUtil as GPU  # access to GPU subsystem

from modeler import Modeler
from utils import CustomTimeseriesGenerator

# 'libriaries loaded'


# import tensorflow as tf
print('TensorFlow version:', tf.__version__)
import keras
print('Keras version:', keras.__version__)
import keras_tuner
print('Keras Tuner version:', keras_tuner.__version__)
# import autokeras
# print('autokeras  version:', autokeras.__version__)


symbol = "BNB-USDT"
folder_path = f'./data'
file_path = f'{folder_path}/dataset_{symbol}.parquet'

df = pd.read_parquet(file_path)

modeler = Modeler(symbol) # create object of Modeler
modeler.last_months=6 # get 1 month from the end of data

modeler.epochs=3 # задать количество эпох для autokeras
modeler.max_trials=30 # задать количество триалов для autokeras
modeler.lookback=384
modeler.predict_forward=5
modeler.batch_size=32
modeler.max_epochs=10

# modeler.epochs=1 # set number epochs for Autokeras
# modeler.max_trials=1 # set number max_trials for Autokeras
# modeler.load_csv(os.path.join('.', 'data/BTCUSD_5M.csv')) # load csv with absolute path
modeler.load_df(df) # load csv with absolute path

modeler.prepare_dataset() # prepare dataset from raw data (cut months, set types, locate targets to right part d columns etc.)
modeler.split_sets() # split dataset to train and val sets, because it is timeseries data with candles from financial market
modeler.save_test_set() # we do this because if we plan to use x_val.npy and y_val.npy and these files are useful as data for testing model.predict()


modeler.dataset_raw.info()


# start searching for the best architecture in neural networks. model will be saved in the property modeler.model
modeler.create_ak_model()


# save current model as <ticker>_model.h5
modeler.save_model()


# load model from the disk (there are two files: <ticker>_model.h5 and <ticker>_final_model.h5)
modeler.load_model() 


# load final model from the disk (there are two files: <ticker>_model.h5 and <ticker>_final_model.h5)
modeler.load_model_final() 


print("\nEvaluate model:")
modeler.evaluate_model() # evaluate model by validation set and save metrics in the property modeler.history 


print("\nModel history:")
try:
    print(modeler.history) # print last metrics we have got
except:
    print("No model history")


print("\nModel summary:")
try:
    modeler.model.summary() # print summary of the model. The model locates in modeler.model
except:
    print("No model summary")



# load test .npy files from the disk for debug or smth else
x = np.load('x_val.npy')
y = np.load('y_val.npy')

print("\nPredict:")
p = modeler.model_predict(x[0:10])
print("Predicted:", p)

# формула_обратного_предикта
# согласно разметке, нужно взять значение из самого последнего которое нам известно и пройти по тому как размечали, но в обратную сторону:
# значение_свечи = предыдущее_значение + (предыдущее_значение*предикт)

print("\nCorrelation:")
print(y[0], p[0])
modeler.calc_and_print_corr(y[0], p[0], len(p[0])) # print auto correlation of the predict and target sample with index 0


print("\nPerformance test:")
modeler.model_perf_test() # start performance test

print("\nTrain model:")
# start final training with the list of learning_rates. The model will be trained one by one and will use these lr list
# modeler.train_model([1e-3, 1e-4])

print("\nSave final model:")
# save current model as <ticker>_final_model.h5
# modeler.save_model_final()

print("\nDone.")