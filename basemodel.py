# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 17:23:54 2022

@author: tghosh
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
from dataloader import DataLoader

class LeNetEncoder(Model):
    
    def __init__(
        self,
        numInputChannels = 1,  
        dropoutPct = 0.2,
        training = True,
        name="LeNetEncoder",
        **kwargs
        ):
        super(LeNetEncoder, self).__init__(name=name, **kwargs)
        self.training = training 
        self.Conv1 = layers.Conv2D(20, kernel_size=(5,5), strides= 1,
                        padding= 'valid', activation= 'relu',                        
                        kernel_initializer= 'he_normal')
        
        self.Conv2 = layers.Conv2D(50, kernel_size=(5,5), strides= 1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal')
                
        self.pool1 = layers.MaxPooling2D(pool_size=(2,2), strides= (2,2),
                              padding= 'valid', data_format= None)
        self.pool2 = layers.MaxPooling2D(pool_size=(2,2), strides= (2,2),
                              padding= 'valid', data_format= None)
        
        self.fc1 = layers.Dense(500, activation= 'relu')
        self.dropout2d = layers.Dropout(dropoutPct)
        
        
    def call(self, inputs):
        x = self.Conv1(inputs)
        x = self.pool1(x)
        x = self.Conv2(x)
        if self.training:
            x = self.dropout2d(x)
        x = self.pool2(x)        
        x = layers.Flatten()(x)        
        x = self.fc1(x)         
        return x


class LeNetClassifier(Model):
    def __init__(self,
                 dropoutPct = 0.2,
                 num_classes = 10,
                 training = True,
                 name="LeNetClassifier",
                 **kwargs
                 ):
        super(LeNetClassifier, self).__init__(name=name, **kwargs)
        self.dropout = layers.Dropout(dropoutPct)
        self.fc2 = layers.Dense(num_classes, activation= 'softmax')
        self.training = training 
        
    def call(self, x):
        if self.training:
            x = self.dropout(x)        
        x = self.fc2(x)          
        return x


class LeNet(Model):
    def __init__(self, 
                 numInputChannels = 1,                 
                 num_classes = 10,
                 dropoutPct = 0.2,
                 name="LeNet",
                 training = True,
                 **kwargs):
        super(LeNet, self).__init__(name=name, **kwargs)
        self.encoder = LeNetEncoder(numInputChannels, dropoutPct, training)
        self.classifier = LeNetClassifier(dropoutPct, num_classes, training)
    
    def call(self, inputs):
        x = self.encoder(inputs)
        x = self.classifier(x)
        return x

class Discriminator(Model):
    def __init__(self,  
                 hidden_dims,
                 name="Discriminator",
                 training = True,
                 **kwargs):
        super(Discriminator, self).__init__(name=name, **kwargs)
        self.hidden_dims =  hidden_dims
        self.fc1 = layers.Dense(hidden_dims, activation= 'relu')
        self.fc2 = layers.Dense(hidden_dims, activation= 'relu')
        self.fc3 = layers.Dense(1, activation= 'sigmoid')
    
    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    
if __name__=="__main__":    
    model = LeNet()
    loader = DataLoader("mnist")
    
    loss_metric = tf.keras.metrics.Mean()
    m = tf.keras.metrics.Accuracy()
    
    (x_train, y_train) = loader.getData('train')
    (x_test, y_test) = loader.getData('test')
    
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.shuffle(buffer_size=1024).batch(64)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    ce_loss_fn = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    
    epochs = 10
    
    # Iterate over epochs.
    for epoch in range(epochs):
        print("Start of epoch %d" % (epoch,))
    
        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch) in enumerate(train_dataset):
            #x_batch_train = tf.expand_dims(x_batch_train, axis=3)
            with tf.GradientTape() as tape:
                pred = model(x_batch_train)
                
                loss = tf.reduce_mean(ce_loss_fn(y_batch, pred))
                m.update_state(tf.argmax(y_batch, 1),tf.argmax(pred, 1))
    
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
    
            loss_metric(loss)
    
            if step % 100 == 0:
                print("step %d: mean loss = %.4f acc=%.4f" % (step, loss_metric.result(),  m.result()))
        
        val_loss_metric = tf.keras.metrics.Mean()
        val_acc = tf.keras.metrics.Accuracy()
        
        for i, (x_batch_test, y_batch_test) in enumerate(test_dataset):
            #x_batch_test = tf.expand_dims(x_batch_test, axis=3)
            pred_val = model(x_batch_test)                
            val_loss = tf.reduce_mean(ce_loss_fn(y_batch_test, pred_val))
            val_loss_metric(val_loss)
            val_acc.update_state(tf.argmax(y_batch_test, 1),tf.argmax(pred_val, 1))
        print("Validation: Mean loss = %.4f acc=%.4f" % (val_loss_metric.result(),  val_acc.result()))                    
    model.save_weights('./models/mnist_lenet')            
    
    