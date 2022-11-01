# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 15:40:03 2022

@author: tghosh
"""
import h5py
import tensorflow as tf 

class DataLoader:
    def __init__(self, 
                 dataset):
        print('Loading dataset', dataset)
        if dataset != "mnist" and dataset != "usps":
            raise Exception("Unknown data source")
        self.dataset = dataset
        self.__load__()
        
    def __load__(self):
        if self.dataset == 'usps':
            self.__load__usps__()
        else:
            self.__load__mnist__()
            
    def __load__usps__(self):
        with h5py.File('./data/usps.h5', 'r') as hf:
            train = hf.get('train')
            X_tr = train.get('data')[:]
            y_tr = train.get('target')[:]
            test = hf.get('test')
            X_te = test.get('data')[:]
            y_te = test.get('target')[:]

        X_tr = tf.expand_dims(tf.reshape(X_tr, [-1, 16, 16]), axis = 3)        
        self.X_tr = tf.image.resize(X_tr, [28,28]) 
        
        X_te = tf.expand_dims(tf.reshape(X_te, [-1, 16, 16]), axis = 3)        
        self.X_te = tf.image.resize(X_te, [28,28]) 
        
        self.y_tr = y_tr
        self.y_te = y_te
        
    def __load__mnist__(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        y_train = tf.one_hot(y_train, 10)
        x_train = x_train.astype("float32") / 255
        x_train = tf.expand_dims(x_train, axis=3)
        
        y_test = tf.one_hot(y_test, 10)
        x_test = x_test.astype("float32") / 255 
        x_test = tf.expand_dims(x_test, axis = 3)
        
        self.X_tr = x_train
        self.y_tr =  y_train
        
        self.X_te = x_test
        self.y_te = y_test
        
    def getData(self, subset):
        if subset != "train" and subset != "test":
            raise Exception("Unknown data subset")
        if subset == "test":
            return (self.X_te, self.y_te)        
        else:
            return (self.X_tr, self.y_tr)        
         