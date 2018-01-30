import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras import optimizers
from keras.utils.np_utils import to_categorical
import pdb


def balance_batch(data, label):
    num = int((label==1).sum()*1.5)
    index_0 = np.random.choice(np.where(label==0)[0], num)
    batch = []
    batch_label = []
    for i in range(len(data)): 
        if i in index_0 or label[i]==1:
            batch.append(data[i])
            batch_label.append(label[i])
    return np.array(batch), np.array(batch_label)

class Classifier(object):
    
    def __init__(self):
        
        self.model = Sequential()
        self.model.add(Dense(units=64, activation='relu', input_dim=282))
        self.model.add(Dense(units=64, activation='relu', input_dim=50))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
    
    def fit(self, X, y):
        
        print('Incremental training, each time we sample a balanced dataset for training:')
        num_iterations = 15
        for i in range(num_iterations):
            batch_train, batch_label = balance_batch(X, y)
            self.model.train_on_batch(batch_train, batch_label)
    
    def predict(self, X):
        return (self.model.predict(X)>=0.49).astype(int)
    
    def predict_proba(self, X):
        prob_1 = self.model.predict_proba(X)
        prob_0 = 1 - self.model.predict_proba(X)
        return np.concatenate((prob_0, prob_1), axis=1)