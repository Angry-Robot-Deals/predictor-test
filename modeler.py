# load libraries
import absl.logging
import os

absl.logging.set_verbosity('fatal')  # 'error' warnings 'fatal' mute all except ERROR
os.environ[
    'TF_CPP_MIN_LOG_LEVEL'] = '3'  # '0' show all, '1' mute INFO, '2' mute INFO and WARNING, '3' mute all except ERROR
# os.environ["KERAS_BACKEND"] = "tensorflow"

# Set CUDA_VISIBLE_DEVICES if we want to use CPU or GPU
# USE_GPU = True
# os.environ["CUDA_VISIBLE_DEVICES"] = 'None' if USE_GPU else '' # '' - for CPU, 'None' - for GPU

import warnings

warnings.filterwarnings("ignore")
import inspect
import logging
from logging.handlers import RotatingFileHandler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc  # garbage collector

import time
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta

import autokeras as ak
import tensorflow as tf

from tensorflow.keras import backend as K  # keras backend functions
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
# from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from utils import CustomTimeseriesGenerator

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Flatten, Activation, LeakyReLU
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall

import psutil as ps  # library for retrieving information on running processes and system utilization
import humanize as hm  # library for turning a number into a fuzzy human-readable
import GPUtil as GPU  # access to GPU subsystem


# Regression losses and metrics
def mse_loss(y_true, y_pred):
    # print('mse_loss:', y_true.shape, y_pred.shape)
    diff = y_true - y_pred
    squared_diff = K.square(diff)
    return K.mean(squared_diff, axis=-1)


def mape_all(y_true, y_pred):
    # print('mape_all:', y_true.shape, y_pred.shape)
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true), K.epsilon(), None))
    return K.mean(diff, axis=-1)


# base class
class Modeler:
    # hyperparameters
    lookback = 360  # how many candles we look in a past
    predict_from = 1  # first (which next candle we predict)
    predict_until = 1  # last candle to predict
    predict_forward = 48  # 24 number of steps to forecast
    #
    max_trials = 30  # 100 how many attempts does autokeras make
    val_size = 0.1  # validation set 10%
    batch_size = 24  # 24 size of batch
    epochs = 10  # number of epochs for autokeras
    max_epochs = 100  # this value will not be reached due to callbacks
    #
    last_months = 6  # 24  # how many last months in history we take
    num_targets = 1  # how many targets in the right part of data we get to cut and drop them into Y
    #
    # directories
    cur_dir = os.getcwd()  # current working directory
    script_name = 'modeler.py'  # script name
    models_dir = os.path.join(cur_dir, 'models')  # models' dir
    log_file_name = os.path.join(cur_dir, f'{os.path.splitext(script_name)[0]}.log')  # log file
    #
    # headers
    target_headers = ['lpclose']
    # categories_headers  = ['timestamp', 'cbuy_profit', 'cbuy_drawdown', 'cbuy_time', 'csell_profit', 'csell_drawdown',
    #                       'csell_time', 'csignal']
    # feature_headers     = ['lpclose']
    feature_headers = ['tday_year', 'tday_month', 'tday_week', 'tmonth_year', 'tweek_year', 'tsecond_day', 'topen',
                       'thigh', 'tlow', 'tclose', 'tvolume', 'lpopen', 'lfopen', 'lphigh', 'lfhigh', 'lplow', 'lflow',
                       'lpclose', 'lfclose', 'lpvolume', 'lfvolume', 'lppricema', 'lfpricema', 'lpvolumema',
                       'lfvolumema', ]

    def __init__(self, ticker: str = 'TICKER'):
        try:
            # datasets
            self.dataset_raw = None
            self.dataset_train = None
            self.dataset_val = None

            # models
            self.model = None  # current model
            self.history = None  # current history of training

            # ticker and models' filenames
            self.ticker = ticker
            self.model_name = os.path.join(self.models_dir, f'{self.ticker.lower()}_model.h5')
            self.model_final_name = os.path.join(self.models_dir, f'{self.ticker.lower()}_model_final.h5')

            # enable logging
            # logging.basicConfig(filename=self.log_file_name, level=logging.DEBUG,
            #                     format='%(asctime)s - %(levelname)s - %(message)s')
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.DEBUG)  # logging.INFO
            max_size_bytes = 10 * 1024 * 1024  # 10 Mbytes
            backup_count = 5  # count of backup files
            file_handler = RotatingFileHandler(self.log_file_name, maxBytes=max_size_bytes, backupCount=backup_count)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            # create directory for models
            if not os.path.exists(self.models_dir):
                os.makedirs(self.models_dir)
        except Exception as e:
            self.logdebug(f'ERROR {inspect.currentframe().f_code.co_name}: {e}')
        else:
            self.loginfo('modeler started')

    # print message
    def print(self, msg):
        print(f'({self.ticker.upper()}) {msg}')

    # logging info message
    def loginfo(self, msg):
        self.logger.info(f'({self.ticker.upper()}) {msg}')
        self.print(msg)

    # logging debug message
    def logdebug(self, msg):
        self.logger.debug(f'({self.ticker.upper()}) {msg}')
        self.print(msg)

    # clear memory
    @staticmethod
    def clear_memory():
        gc.collect()  # run garbage collector
        K.clear_session()  # clear memory from old models

    # get model memory usage
    def model_memory_usage(self, batch_size, model):
        shapes_mem_count = 0
        internal_model_mem_count = 0
        for l in model.layers:
            layer_type = l.__class__.__name__
            if layer_type == 'Model':
                internal_model_mem_count += self.model_memory_usage(batch_size, l)
            single_layer_mem = 1
            out_shape = l.output_shape
            if type(out_shape) is list:
                out_shape = out_shape[0]
            for s in out_shape:
                if s is None:
                    continue
                single_layer_mem *= s
            shapes_mem_count += single_layer_mem
        trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
        non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])
        number_size = 4.0
        if K.floatx() == 'float16':
            number_size = 2.0
        if K.floatx() == 'float64':
            number_size = 8.0
        total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
        gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
        return gbytes

    # GPU info
    @staticmethod
    def gpu_info():
        GPUs = GPU.getGPUs()  # get number of GPUs
        # XXX: only one GPU on Colab and isn’t guaranteed
        gpu = GPUs[0]
        process = ps.Process(os.getpid())
        print(f'Gen RAM Free: {hm.naturalsize(ps.virtual_memory().available)} | '
              f'Proc size: {hm.naturalsize(process.memory_info().rss)}')
        print('GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util: {2:3.0f}% | Total: {3:.0f}MB'.format(gpu.memoryFree,
                                                                                                      gpu.memoryUsed,
                                                                                                      gpu.memoryUtil * 100,
                                                                                                      gpu.memoryTotal))
        print(f'GPU Model: {gpu.name}')

    # correlation function
    @staticmethod
    def correlate(a, b):
        ma = a.mean()
        mb = b.mean()
        mab = (a * b).mean()
        sa = a.std()
        sb = b.std()
        val = 1
        if ((sa > 0) & (sb > 0)):
            val = (mab - ma * mb) / (sa * sb)
        return val

    # calculate autocorrelation of 2 arrays
    def calc_corr(self, y, pred, corrSteps=10):
        corr = []
        yLen = len(y)
        for i in range(corrSteps):
            corr.append(self.correlate(y[:yLen - i], pred[i:]))
        return np.array(corr)

    # test autocorrelation for valid
    def corr_is_valid(self, y, pred):
        try:
            r1, r2, r3 = False, False, False  # def values
            own_corr = self.calc_corr(y, y)  # correlation for itself
            pred_corr = self.calc_corr(y, pred)  # correlation to predict
            d = pred_corr - own_corr
            # check if pred corr line above target corr line
            # 3 levels of good autocorrelation test
            # 1 level - ok
            r1 = np.mean(d) > 0
            # 2 level - better
            r2 = r1 and d[int(len(d) * 0.33)] > 0 and d[int(len(d) * 0.5)] > 0 and d[int(len(d) * 0.66)] > 0
            # 3 level - excellent
            r3 = r2 and np.median(d) > np.mean(d)
        except Exception as e:
            self.logdebug(f'ERROR {inspect.currentframe().f_code.co_name}: {e}')
        else:
            self.logdebug(f'{inspect.currentframe().f_code.co_name} completed')
        finally:
            return r1, r2, r3

    # test autocorrelation for valid - ANY
    def corr_is_valid_any(self, y, pred):
        return np.any(self.corr_is_valid(y, pred))

    # show autocorrelations
    def print_corr(self, y, pred):
        try:
            plt.figure(figsize=(14, 7))
            plt.plot(pred, label='Prediction', marker='.')
            plt.plot(y, label='Target', marker='.')
            plt.xlabel('Steps')
            plt.ylabel('Value')
            plt.legend()
            plt.savefig(os.path.join(self.models_dir, f'{self.ticker.lower()}_corr.jpg'))
            plt.show()
        except Exception as e:
            self.logdebug(f'ERROR {inspect.currentframe().f_code.co_name}: {e}')
        else:
            self.logdebug(f'{inspect.currentframe().f_code.co_name} completed')

    # show demo ideal autocorrelations
    def print_ideal_corr(self):
        # ideal autocorrelations
        p = np.array([0.999, 0.9995, 0.9997, 0.99965, 0.99951, 0.9992, 0.99877, 0.9985, 0.9981, 0.9977, 0.9972])
        y = np.array([1., 0.99965, 0.9993, 0.99895, 0.9986, 0.99825, 0.9979, 0.99755, 0.9972, 0.99685, 0.9965])
        # y = np.linspace(p[0], p[-1], num=len(p))
        self.print_corr(y, p)

    # calc and print autocorrelation
    def calc_and_print_corr(self, y, pred, corrSteps=10):
        self.print_corr(self.calc_corr(y, y, corrSteps), self.calc_corr(y, pred, corrSteps))

    # load dataset from csv
    def load_csv(self, filename=None):
        try:
            self.dataset_raw = pd.read_csv(filename).reset_index(drop=True)  # os.path.join(self.cur_dir, filename)
        except Exception as e:
            self.dataset_raw = None
            self.logdebug(f'ERROR {inspect.currentframe().f_code.co_name}: {e}')
        else:
            self.logdebug(f'{inspect.currentframe().f_code.co_name} completed')

    # load dataset from DataFrame
    def load_df(self, df=None):
        try:
            self.dataset_raw = df.copy(deep=True).reset_index(drop=True)
        except Exception as e:
            self.dataset_raw = None
            self.logdebug(f'ERROR {inspect.currentframe().f_code.co_name}: {e}')
        else:
            self.logdebug(f'{inspect.currentframe().f_code.co_name} completed')

    # check dataset
    def prepare_dataset(self):
        try:
            # Drop rows which contain missing values
            self.dataset_raw = self.dataset_raw.dropna(axis=0).reset_index(drop=True)

            # Stripe dataset by last months
            last_dataset_date = datetime.fromtimestamp(
                int(self.dataset_raw.loc[self.dataset_raw.shape[0] - 1, 'timestamp']), tz=timezone.utc)
            start_date = last_dataset_date - relativedelta(months=self.last_months)  # datetime.now(tz=timezone.utc)
            start_timestamp = int(start_date.timestamp())
            self.dataset_raw = self.dataset_raw[self.dataset_raw['timestamp'] >= start_timestamp].reset_index(drop=True)

            # # Shift targets forward, convert to a TimeSeries
            # for col in target_headers:
            #     self.dataset_raw[col] = [0.] + self.dataset_raw[col].tolist()[:-1]
            # self.dataset_raw = self.dataset_raw[1:].reset_index(drop=True)

            # Convert types
            self.dataset_raw[self.feature_headers] = self.dataset_raw[self.feature_headers].astype('float64')
            # self.dataset_raw[self.categories_headers] = self.dataset_raw[self.categories_headers].astype('int')

            # Return only Features and Targets
            self.dataset_raw = self.dataset_raw[self.feature_headers]  # get just features
            for i, col in enumerate(self.target_headers):  # create new column copy for each target
                col_name = f'Y{i}'  # new column which we fill with target data
                self.dataset_raw[col_name] = self.dataset_raw[col]

        except Exception as e:
            self.logdebug(f'ERROR {inspect.currentframe().f_code.co_name}: {e}')
        else:
            self.logdebug(
                f'{inspect.currentframe().f_code.co_name} completed: dataset_raw.shape: {self.dataset_raw.shape}')

    # create sets from dataset (train and val)
    def split_sets(self):
        try:
            val_split = int(self.dataset_raw.shape[0] * (1 - self.val_size))  # last index for train_set
            self.dataset_train = self.dataset_raw[:val_split - 2 * self.lookback].reset_index(drop=True)  # train
            self.dataset_val = self.dataset_raw[val_split:].reset_index(
                drop=True)  # val  # self.dataset_raw = None # free memory from dataset_raw
        except Exception as e:
            self.logdebug(f'ERROR {inspect.currentframe().f_code.co_name}: {e}')
        else:
            self.logdebug(f'{inspect.currentframe().f_code.co_name} completed: '
                          f'dataset_train.shape: {self.dataset_train.shape} '
                          f'dataset_val.shape: {self.dataset_val.shape} ')

    # split dataset to XY for TimeSeriesForecaster
    def split_xy(self, df, squeeze=False, expand_dims_y=False):
        try:
            X, Y = None, None
            x, y = df.iloc[:, :-self.num_targets].values, df.iloc[:, -self.num_targets:].values  # split x,y by columns
            y = np.array([np.squeeze(y[i:i + self.predict_forward]) for i in
                          range(len(y) - self.predict_forward)])  # collect forecast candles in one candle
            x = x[:-self.predict_forward]  # cut X to be equal Y
            if squeeze:  # if we need remove all shapes with dim 1
                x = np.squeeze(x)
                y = np.squeeze(y)
            if expand_dims_y:  # if we need fix shape, adjust 1 dim
                y = np.expand_dims(y, 1)
            X, Y = x, y
        except Exception as e:
            self.logdebug(f'ERROR {inspect.currentframe().f_code.co_name}: {e}')
        else:
            self.logdebug(f'{inspect.currentframe().f_code.co_name} completed: X.shape: {X.shape} Y.shape: {Y.shape}')
        finally:
            return X, Y

    # Save model filename
    def save_model_filename(self, filename):
        try:
            self.model.save(filename, overwrite=True, include_optimizer=True, save_format="h5")
        except Exception as e:
            self.logdebug(f'ERROR {inspect.currentframe().f_code.co_name}: {e}')
        else:
            self.logdebug(f'{inspect.currentframe().f_code.co_name} completed: filename={filename}')

    # Load regression model filename
    def load_model_filename(self, filename):
        try:
            self.model = None  # def value

            # load model from .h5 format
            self.model = load_model(filename, compile=False)

            # compile the model with custom metrics
            self.model.compile(loss=mse_loss, metrics=[mape_all],  # + [f for f in mape_func_list[-self.num_targets:]],
                               # optimizer=Adam(lr=1e-4),
                               )
        except Exception as e:
            self.loginfo(f'ERROR {inspect.currentframe().f_code.co_name}: {e}')
        else:
            self.loginfo(f'{inspect.currentframe().f_code.co_name} completed: filename={filename}')

    # Save regression model
    def save_model(self):
        self.save_model_filename(self.model_name)
        self.save_model_filename(self.model_final_name)

    # Load regression model
    def load_model(self):
        self.load_model_filename(self.model_name)

    # Save regression model final
    def save_model_final(self):
        self.save_model_filename(self.model_final_name)

    # Load regression model final
    def load_model_final(self):
        self.load_model_filename(self.model_final_name)

    # Test regression model
    def evaluate_model(self):
        try:
            self.history = None  # def value

            # prepare X and Y for sets headers
            # x_train, y_train = self.split_xy(self.dataset_train, squeeze=False ,expand_dims_y=False)
            x_val, y_val = self.split_xy(self.dataset_val, squeeze=False, expand_dims_y=False)

            # create generator for test set
            gen_test = CustomTimeseriesGenerator(x=x_val, y=y_val, lookback=self.lookback,
                predict_from=self.predict_from, predict_until=self.predict_until, batch_size=self.batch_size,
                # x_val.shape[0]
                squeeze=True, )

            # compile the model with custom metrics
            self.model.compile(loss=mse_loss, metrics=[mape_all],  # + [f for f in mape_func_list[-self.num_targets:]],
                               # optimizer=Adam(lr=1e-4),
                               )
            # get metrics of the model
            self.history = self.model.evaluate(gen_test, verbose=0)

            self.print('The history is saved in self.history ')
        except Exception as e:
            self.logdebug(f'ERROR {inspect.currentframe().f_code.co_name}: {e}')
        else:
            self.logdebug(f'{inspect.currentframe().f_code.co_name} completed: '
                          f'model size = {self.model_memory_usage(1000, self.model)} Gbytes for batch=1000, '
                          f'history={self.history} ')

    # Find best Regression model with AutoKeras
    def create_ak_model(self):
        try:
            self.model, self.history = None, None  # def value

            # prepare X and Y for sets headers
            x_train, y_train = self.split_xy(self.dataset_train, squeeze=False, expand_dims_y=False)
            x_val, y_val = self.split_xy(self.dataset_val, squeeze=False, expand_dims_y=False)

            # create forecaster
            forecaster = ak.TimeseriesForecaster(lookback=self.lookback, predict_from=self.predict_from,
                predict_until=self.predict_until, max_trials=self.max_trials, objective="val_loss", loss=mse_loss,
                # 'mse' # 'mean_squared_error'
                metrics=[mape_all],  # + [f for f in mape_func_list[-self.num_targets:]],
                overwrite=True, directory=self.cur_dir, )

            # start autokeras
            forecaster.fit(x_train, y_train, validation_data=(x_val, y_val), verbose=1, batch_size=self.batch_size,
                           epochs=self.epochs)

            # get the best model
            self.model = forecaster.tuner.get_best_models(num_models=1)[0]
            # self.model.summary()

            # create generator for test set
            gen_test = CustomTimeseriesGenerator(x=x_val, y=y_val, lookback=self.lookback,
                predict_from=self.predict_from, predict_until=self.predict_until, batch_size=self.batch_size,
                # x_val.shape[0]
                squeeze=True, )

            # compile the model with custom metrics
            self.model.compile(loss=mse_loss, metrics=[mape_all],  # + [f for f in mape_func_list[-self.num_targets:]],
                               # optimizer=Adam(lr=1e-4),
                               )
            # get metrics of the model
            self.history = self.model.evaluate(gen_test, verbose=0)

            self.print('The model is saved in self.model, the history is saved in self.history ')
        except Exception as e:
            self.logdebug(f'ERROR {inspect.currentframe().f_code.co_name}: {e}')
        else:
            self.logdebug(f'{inspect.currentframe().f_code.co_name} completed: '
                          f'model size = {self.model_memory_usage(1000, self.model)} Gbytes for batch=1000, '
                          f'history={self.history} ')

    # Create new functional model Conv1D
    def create_conv1d_model(self):
        try:
            filters = 8  # initial filter count
            kernel = 3  # kernel_size
            dropout = 0  # dropout
            alpha = 0.2  # alpha
            momentum = 0.8  # momentum
            normalization = True  # normalization

            # conv1d block
            def conv1d(block_input, f, num_blocks=2, k=kernel, n=normalization, d=dropout, a=alpha, m=momentum):
                x = block_input
                for i in range(1, num_blocks + 1):
                    x = Conv1D(f, k, padding='same')(block_input)
                    x = LeakyReLU(alpha=a)(x)
                    if n: x = BatchNormalization(momentum=m)(x)
                x = MaxPooling1D()(x)
                if d: x = Dropout(d)(x)
                return x

            input1 = Input(shape=(self.lookback, len(self.feature_headers)))  # input layer
            x = input1
            x = conv1d(x, filters * 1, 8)  # block_1 256->128
            x = conv1d(x, filters * 2, 8)  # block_2 128->64
            x = conv1d(x, filters * 4, 8)  # block_3 64->32
            x = conv1d(x, filters * 8, 8)  # block_4 32->16
            x = conv1d(x, filters * 16, 8)  # block_5 16->8
            x = conv1d(x, filters * 32, 8)  # block_6 8->4
            x = conv1d(x, filters * 64, 8)  # block_7 4->2
            x = Flatten()(x)  # flatten layer
            x = Dense(512, activation='relu')(x)  # fc layer
            x = Dense(self.predict_forward, activation='tanh')(x)  # output fc layer

            self.model = Model(input1, x)  # model

            self.print('The model is saved in self.model ')
        except Exception as e:
            self.logdebug(f'ERROR {inspect.currentframe().f_code.co_name}: {e}')
        else:
            self.logdebug(f'{inspect.currentframe().f_code.co_name} completed: '
                          f'model size = {self.model_memory_usage(1000, self.model)} Gbytes for batch=1000 ')

    # predict function
    def model_predict(self, x):
        try:
            result = None  # def value
            start_time = time.time()
            result = self.model.predict(x, verbose=0)  # predict
            end_time = time.time()
        except Exception as e:
            self.loginfo(f'ERROR {inspect.currentframe().f_code.co_name}: {e}')
        else:
            self.loginfo(f'{inspect.currentframe().f_code.co_name} completed: '
                         f'{round(end_time - start_time, 3)} sec ')
        finally:
            return result  # return as numpy array

    # performance test
    def model_perf_test(self):
        try:
            # prepare X and Y for sets headers
            # x_train, y_train = self.split_xy(self.dataset_train, squeeze=False ,expand_dims_y=False)
            x_val, y_val = self.split_xy(self.dataset_val, squeeze=False, expand_dims_y=False)

            # create generator for test set
            gen_test = CustomTimeseriesGenerator(x=x_val, y=y_val, lookback=self.lookback,
                predict_from=self.predict_from, predict_until=self.predict_until, batch_size=x_val.shape[0],
                # self.batch_size
                squeeze=True, )

            # compile the model with custom metrics
            self.model.compile(loss=mse_loss, metrics=[mape_all],  # + [f for f in mape_func_list[-self.num_targets:]],
                               # optimizer=Adam(lr=1e-4),
                               )

            # measure time for predict
            xLen = gen_test[0][0].shape[0]
            start_time = time.time()
            pred = self.model.predict(gen_test[0][0], verbose=0)  # predict all val set
            end_time = time.time()
        except Exception as e:
            self.logdebug(f'ERROR {inspect.currentframe().f_code.co_name}: {e}')
        else:
            self.logdebug(f'{inspect.currentframe().f_code.co_name} completed: '
                          f'Performance = {round(xLen / (end_time - start_time), 3)} samples/sec, '
                          f'{round((end_time - start_time) / xLen, 3)} seconds/sample ')

    # save test set in .npy files
    def save_test_set(self):
        try:
            # prepare X and Y for sets headers
            # x_train, y_train = self.split_xy(self.dataset_train, squeeze=False ,expand_dims_y=False)
            x_val, y_val = self.split_xy(self.dataset_val, squeeze=False, expand_dims_y=False)

            # create generator for test set
            gen_test = CustomTimeseriesGenerator(x=x_val, y=y_val, lookback=self.lookback,
                predict_from=self.predict_from, predict_until=self.predict_until, batch_size=x_val.shape[0],
                # self.batch_size
                squeeze=True, )

            # save test dataset in numpy files
            np.save(os.path.join(self.cur_dir, 'x_val.npy'), gen_test[0][0])
            np.save(os.path.join(self.cur_dir, 'y_val.npy'), gen_test[0][1])
        except Exception as e:
            self.logdebug(f'ERROR {inspect.currentframe().f_code.co_name}: {e}')
        else:
            self.logdebug(f'{inspect.currentframe().f_code.co_name} completed: '
                          f'x_val.shape: {gen_test[0][0].shape} '
                          f'y_val.shape: {gen_test[0][1].shape} ')

    # train model
    def train_model(self, learning_rate_list=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]):
        try:
            self.history = None  # def value

            gc.collect()  # run garbage collector

            # callback - save best model at the end of epoch
            checkpoint = [ModelCheckpoint(self.model_final_name, monitor='val_loss',  # loss val_loss
                                          save_best_only=True)]

            # callback - stop training if metric does not increase
            early = EarlyStopping(monitor='val_loss',  # loss val_loss
                                  patience=5, mode='auto')

            # callback - reduce learning rate when metric does not increase
            lr_reduce = ReduceLROnPlateau(monitor='val_loss',  # loss val_loss
                patience=5,  # 4
                verbose=0, mode='auto')

            # prepare X and Y for sets headers
            x_train, y_train = self.split_xy(self.dataset_train, squeeze=False, expand_dims_y=False)
            x_val, y_val = self.split_xy(self.dataset_val, squeeze=False, expand_dims_y=False)

            # create generator for train set
            gen_train = CustomTimeseriesGenerator(x=x_train, y=y_train, lookback=self.lookback,
                predict_from=self.predict_from, predict_until=self.predict_until, batch_size=self.batch_size,
                squeeze=True, infinite=True, )

            # create generator for val set
            gen_val = CustomTimeseriesGenerator(x=x_val, y=y_val, lookback=self.lookback,
                predict_from=self.predict_from, predict_until=self.predict_until, batch_size=self.batch_size,
                squeeze=True, infinite=True, )

            train_steps = len(gen_train)  # num of train batches
            validation_steps = len(gen_val)  # num of val batches

            for lr in learning_rate_list:
                # compile the model with custom metrics and current learning_rate=lr
                self.model.compile(loss=mse_loss,  # 'mse'
                                   metrics=[mape_all],  # + [f for f in mape_func_list[-self.num_targets:]],
                                   optimizer=Adam(lr=lr), )

                # train model
                history = self.model.fit(gen_train, validation_data=gen_val, epochs=self.max_epochs,
                                         # batch_size=self.batch_size,
                                         steps_per_epoch=train_steps, validation_steps=validation_steps,
                                         callbacks=[checkpoint, early], verbose=1, )

                # load the final model was saved by callbacks
                self.load_model_final()

                # create generator for test set
                gen_test = CustomTimeseriesGenerator(x=x_val, y=y_val, lookback=self.lookback,
                    predict_from=self.predict_from, predict_until=self.predict_until, batch_size=self.batch_size,
                    squeeze=True, infinite=False,  # for evaluate we should stop generator at the end of data
                )

                # get metrics of the model
                self.history = self.model.evaluate(gen_test, verbose=0)

                self.print(
                    f'Trained model (lr={lr}) is saved in self.model and to disk, history is saved in self.history ')
        except Exception as e:
            self.logdebug(f'ERROR {inspect.currentframe().f_code.co_name}: {e}')
        else:
            self.logdebug(f'{inspect.currentframe().f_code.co_name} completed: '
                          f'learning_rate_list={learning_rate_list} '
                          f'history={self.history} ')
