import numpy as np
import math
from sklearn.utils import shuffle
import pickle

# save pickles
def save_obj(obj_to_save, filename):
  with open(filename, 'wb') as f:
    pickle.dump(obj_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

# load pickles
def load_obj(filename):
  with open(filename, 'rb') as f:
    return pickle.load(f)


# class CustomTimeseriesGenerator
class CustomTimeseriesGenerator:
    def __init__(self, x, y, lookback, predict_from, predict_until, batch_size, squeeze=True, infinite=False):
        self.x = x # x_train part of dataframe
        self.y = y # y_train part of dataframe
        self.lookback = lookback # size of X-part what we get
        self.predict_from = predict_from # first time point after X-part of data (default=1)
        self.predict_until = predict_until # next time point after X-part of data (default=1)
        self.batch_size = batch_size # batch size
        self.squeeze = squeeze # if we need remove all shapes with dim 1
        self.infinite = infinite # infinite generator, we will automatically return to begin after last batch
        self.idx = self.lookback+(self.predict_until-self.predict_from+1) # start position in data (lookback and forecast before idx)
    def __len__(self):
        return math.ceil( (len(self.x)-self.lookback-(self.predict_until-self.predict_from+1)) / self.batch_size ) # round up to whole count of batches
    def __getitem__(self, index):
        num_batches = self.__len__() # get whole count of batches in data
        if index>(num_batches-1): # index is out of range
            index = index - (index // num_batches)*num_batches # recalculate index
        # start position index
        idx = (index * self.batch_size +
                     (self.lookback+(self.predict_until-self.predict_from+1)))
        # if we have data
        if (idx<len(self.x)) and (len(self.x)==len(self.y)):
            batch_x, batch_y = [], []
            # collect next batch (lookback and forecast before idx)
            while (idx<len(self.x)) and (len(batch_x)<self.batch_size):
                bx = self.x[idx-self.lookback-(self.predict_until-self.predict_from+1): idx-(self.predict_until-self.predict_from+1), ...]
                by = self.y[idx-(self.predict_until-self.predict_from+1): idx, ...]
                if self.squeeze: # do we need squeeze dimensions
                    bx = np.squeeze(bx)
                    by = np.squeeze(by)
                batch_x.append(bx) # append to batch_x shape=(lookback, x_cols)
                batch_y.append(by) # append to batch_y shape=(y_cols,)
                idx += 1 # one step forward
            batch_x, batch_y = np.array(batch_x), np.array(batch_y) # convert to numpy array
            batch_x, batch_y = shuffle(batch_x, batch_y) # shuffle samples inside the batch
            return batch_x, batch_y # return next batch (x, y)
    def __iter__(self):
        return self
    def __next__(self):
        # if we have data
        if (self.idx<len(self.x)) and (len(self.x)==len(self.y)):
            batch_x, batch_y = [], []
            # collect next batch (lookback and forecast before idx)
            while (self.idx<len(self.x)) and (len(batch_x)<self.batch_size):
                bx = self.x[self.idx-self.lookback-(self.predict_until-self.predict_from+1): self.idx-(self.predict_until-self.predict_from+1), ...]
                by = self.y[self.idx-(self.predict_until-self.predict_from+1): self.idx, ...]
                if self.squeeze: # do we need squeeze dimensions
                    bx = np.squeeze(bx)
                    by = np.squeeze(by)
                batch_x.append(bx) # append to batch_x shape=(lookback, x_cols)
                batch_y.append(by) # append to batch_y shape=(y_cols,)
                self.idx += 1 # one step forward
            batch_x, batch_y = np.array(batch_x), np.array(batch_y) # convert to numpy array
            batch_x, batch_y = shuffle(batch_x, batch_y) # shuffle samples inside the batch
            return batch_x, batch_y # return next batch (x, y)
        else:
            # end of data
            if self.infinite:
                # set position to next batch from the very first (set to batch=1)
                self.idx = (1 * self.batch_size +
                     (self.lookback+(self.predict_until-self.predict_from+1)))
                return self.__getitem__(0) # return very first batch from data (batch=0)
            else:
                # just stop generator
                raise StopIteration


