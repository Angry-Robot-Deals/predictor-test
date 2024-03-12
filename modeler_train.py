#!/usr/bin/env python
# coding: utf-8

import os
# os.environ["TF_USE_LEGACY_KERAS"] = "1"


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

absl.logging.set_verbosity('fatal')  # 'error' warnings 'fatal' mute all except ERROR
os.environ[
    'TF_CPP_MIN_LOG_LEVEL'] = '3'  # '0' show all, '1' mute INFO, '2' mute INFO and WARNING, '3' mute all except ERROR
# os.environ["KERAS_BACKEND"] = "tensorflow"

# Set CUDA_VISIBLE_DEVICES if we want to use CPU or GPU
# USE_GPU = True
# os.environ["CUDA_VISIBLE_DEVICES"] = 'None' if USE_GPU else '' # '' - for CPU, 'None' - for GPU

import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

import autokeras as ak
import tensorflow as tf

from modeler import Modeler

# 'libraries loaded'


# import tensorflow as tf
print('TensorFlow version:', tf.__version__)
print('AutoKeras version:', ak.__version__)
import keras

print('Keras version:', keras.__version__)
import keras_tuner

print('Keras Tuner version:', keras_tuner.__version__)
# import autokeras
# print('autokeras  version:', autokeras.__version__)


symbol = "BTC-USDT"
folder_path = f'./data'
file_path = f'{folder_path}/dataset_{symbol}.parquet'

df = pd.read_parquet(file_path)

modeler = Modeler(symbol)  # create object of Modeler
modeler.last_months = 24  # get 1 month from the end of data

modeler.epochs = 3  # задать количество эпох для autokeras
modeler.max_trials = 5  # задать количество триалов для autokeras
modeler.lookback = 384
modeler.predict_forward = 60
modeler.batch_size = 24
modeler.max_epochs = 100

modeler.target_headers = ['lpclose']
modeler.feature_headers = [
    # 'open', 'high', 'low', 'close', 'volume',
    'past_buy_profit', 'past_buy_dd', 'past_buy_time', 'past_buy_dd_time',
    'past_sell_profit', 'past_sell_dd', 'past_sell_time', 'past_sell_dd_time',
    'tday_year', 'tday_month', 'tday_week', 'tmonth_year', 'tweek_year', 'tsecond_day',
    'topen', 'thigh', 'tlow', 'tclose',
    'tvolume', 'lpvolume', 'lfvolume', 'lpvolumema', 'lfvolumema',
    'lpopen', 'lfopen', 'lphigh', 'lfhigh', 'lplow', 'lflow',
    'lppricema', 'lfpricema', 'lfclose', 'lpclose'
]

# modeler.epochs=1 # set number epochs for Autokeras
# modeler.max_trials=1 # set number max_trials for Autokeras
# modeler.load_csv(os.path.join('.', 'data/BTCUSD_5M.csv')) # load csv with absolute path
modeler.load_df(df)  # load csv with absolute path

modeler.prepare_dataset()  # prepare dataset from raw data (cut months, set types, locate targets to right part d columns etc.)
modeler.split_sets()  # split dataset to train and val sets, because it is timeseries data with candles from financial market
modeler.save_test_set()  # we do this because if we plan to use x_val.npy and y_val.npy and these files are useful as data for testing model.predict()

modeler.dataset_raw.info()

# modeler.create_conv1d_model()
# modeler.model.summary()
# modeler.train_model([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])

# start searching for the best architecture in neural networks. model will be saved in the property modeler.model
# modeler.create_ak_model()

# save current model as <ticker>_model.h5
# modeler.save_model()

# load model from the disk (there are two files: <ticker>_model.h5 and <ticker>_final_model.h5)
modeler.load_model()

# load final model from the disk (there are two files: <ticker>_model.h5 and <ticker>_final_model.h5)
modeler.load_model_final()

print("\nEvaluate model:")
modeler.evaluate_model()  # evaluate model by validation set and save metrics in the property modeler.history

print("\nModel history:")
try:
    print(modeler.history)  # print last metrics we have got
except:
    print("No model history")

print("\nModel summary:")
try:
    modeler.model.summary()  # print summary of the model. The model locates in modeler.model
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
modeler.calc_and_print_corr(y[0], p[0],
                            len(p[0]))  # print auto correlation of the predict and target sample with index 0

print("\nPerformance test:")
modeler.model_perf_test()  # start performance test

print("\nTrain model:")
# start final training with the list of learning_rates. The model will be trained one by one and will use these lr list
# modeler.train_model([1e-3, 1e-4])
modeler.train_model([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])

print("\nSave final model:")
# save current model as <ticker>_final_model.h5
modeler.save_model_final()

print("\nDone.")
