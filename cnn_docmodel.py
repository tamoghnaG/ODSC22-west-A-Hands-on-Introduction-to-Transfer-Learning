# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 14:24:27 2022

@author: tghosh
"""

import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, Conv1D, GlobalMaxPool1D, Dropout


class DocumentEncoder(tf.keras.Model):
    def __init__(self,
                 vocab_size,
                 embedding_dim=100,
                 doc_embd_dim = 256,                 
                 training=False,
                 name="cnn_document_model",
                 **kwargs):
        super(DocumentEncoder, self).__init__(name=name, **kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.doc_embd_dim = doc_embd_dim
        self.training = training 

    
        self.embedding = Embedding(self.vocab_size,
                                   self.embedding_dim)
        #Bigram Conv
        self.cnn1 = Conv1D(filters=25,
                           kernel_size=2,
                           padding="valid",
                           activation="relu",
                           strides = 1)
        #trigram Conv
        self.cnn2 = Conv1D(filters=25,
                           kernel_size=3,
                           padding="valid",
                           activation="relu",
                           strides = 1)
        #4-gram Conv
        self.cnn3 = Conv1D(filters=25,
                           kernel_size=4,
                           padding="valid",
                           activation="relu",
                           strides = 1)
        self.dropout = Dropout(0.1)
        self.doc_embedding = Dense(self.doc_embd_dim, activation="relu")
        
        self.pool = GlobalMaxPool1D()
    
    def call(self, document):
        x = self.embedding(document) 
        bigrams = self.cnn1(x)        
        bigrams = self.pool(bigrams)        
        
        trigrams = self.cnn2(x)
        trigrams = self.pool(trigrams)
        
        quadgrams = self.cnn3(x)
        quadgrams = self.pool(quadgrams)
        
        all_ngrams = tf.concat([bigrams, trigrams, quadgrams], axis = 1)
        if self.training:
            all_ngrams = self.dropout(all_ngrams)
        x = self.doc_embedding(all_ngrams)
        return x 
    
class DocClassifier(tf.keras.Model):
    def __init__(self, 
                 vocab_size,
                 num_classes = 2,
                 embedding_dim=100,
                 doc_embd_dim = 256,
                 dropout_rate=0.1,
                 training=False,
                 name="sentiment_model",
                 **kwargs):
        super(DocClassifier, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.training = training
        self.doc_model = DocumentEncoder(vocab_size,
                 embedding_dim,
                 doc_embd_dim)
        self.dropout = Dropout(dropout_rate)
        if num_classes > 2:
            self.fc1 = Dense(num_classes, activation = 'softmax')
        else:
            self.fc1 = Dense(1, activation = 'sigmoid')
    
    def call(self, document):
        x = self.doc_model(document)
        if self.training:
            x = self.dropout(x)
        x = self.fc1(x)
        return x
'''    
#encoder = DocumentEncoder(20)
        
classifier = DocClassifier(20, 2)
X = tf.Variable([[14,  0,  9, 12,  8, 11,  5,  3, 19,  7,  3, 10,  7, 16, 15, 19,  8,
       10, 17,  5,  8,  5,  0,  1, 13,  2, 12, 18, 19, 13],[13,  2, 10, 10,  1,  6,  8,  5, 10, 12,  9,  1,  1, 13, 17, 14,  6,
        9, 18, 18, 17, 13, 17,  7, 11,  3, 13, 12, 14, 15]])

y = classifier(X)
'''