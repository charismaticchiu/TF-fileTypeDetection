#!/bin python
import tensorflow as tf
import random
import numpy as np
import nltk
import os
import itertools
import h5py
import time
import sklearn as sk
from tensorflow.contrib.layers import dropout
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import train_test_split
import pandas as pd
from Autoencoder import Autoencoder
np.random.seed(5)

# ==========
#   MODEL
# ==========
class BaseConfiguration:

  def __init__(self):
    self.learning_rate = 0.01
    #self.training_iters = 40000
    self.batch_size = 100
    self.display_step = 1
    self.val_step = 30
    self.maxEpoch = 10
    #self.utterances = 10
    # Network Parameters
    
    self.n_hidden1 = 50 # hidden layer num of features
    self.n_hidden2 = 50 # hidden layer num of features
    self.n_hidden3 = 50 # hidden layer num of features
    self.n_classes = 2 # linear sequence or not
    #self.num_layers = 1
    self.keep_prob = 0.8
    #self.debug_sentences = True
    self.path = os.getcwd()

  def printConfiguration(self):
  
    # print configuration
    print '---------- Configuration: ----------'
    print 'learning_rate', self.learning_rate
    #print 'training_iters', self.training_iters
    print 'batch_size', self.batch_size
    
    # Network Parameters

    print 'n_hidden', self.n_hidden1 # hidden layer num of features
    print 'n_hidden2', self.n_hidden2 # hidden layer num of features
    print 'n_hidden3', self.n_hidden3 # hidden layer num of features
    
    print 'n_classes', self.n_classes # linear sequence or not   
    # print 'num_layers', self.num_layers
    print 'keep_prob (dropout = 1-keep_prob)', self.keep_prob
    print '------------------------------------'
    

    

def process_table(filename):
  #data = pd.read_table(filename, header = None)
  data = pd.read_csv(filename,delimiter=',',header = None)
  
  
    
  return failed, label
     
config = BaseConfiguration()
# Prepare DATA
filename1 = 'temp_data.csv'
= process_table(filename1)
data1 = np.hstack((men, io, iow, maxvmem))


data = np.vstack((data1,data2))
label = np.vstack((label1,label2))

print 'data shape', data.shape
print 'label shape', label.shape

# Normalization
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.1)
scaler = StandardScaler(with_mean = True, with_std = True).fit(X_train)
#print scaler.scale_
#print scaler.mean_
#print scaler.var_
X_train_transformed = scaler.transform(X_train)
X_test_transformed = scaler.transform(X_test)

u, c = np.unique(y_train, return_counts=True)
c = c.astype(np.float32)
print 'Prior (Baseline):', c[1]/(c[0]+c[1])

print "Pre-processing Finished!" 


# Define weights
weights = {
  'h1': tf.Variable(tf.random_normal([data.shape[1], config.n_hidden1])),
  'h2': tf.Variable(tf.random_normal([config.n_hidden1, config.n_hidden2])),
  'h3': tf.Variable(tf.random_normal([config.n_hidden2, config.n_hidden3])),
  'out': tf.Variable(tf.random_normal([config.n_hidden3, config.n_classes]))
}
biases = {
  'h1': tf.Variable(tf.random_normal([config.n_hidden1])),
  'h2': tf.Variable(tf.random_normal([config.n_hidden2])),
  'h3': tf.Variable(tf.random_normal([config.n_hidden3])),
  'out': tf.Variable(tf.random_normal([config.n_classes]))
}

# tf Graph input
x = tf.placeholder(tf.float32, [None, data.shape[1]])
y = tf.placeholder(tf.float32, [None, config.n_classes])

tf.layers.conv1d(x,)
hidden1 = tf.nn.relu(tf.matmul(x, weights['h1']) + biases['h1'])
drop_h1 = dropout(hidden1, keep_prob=config.keep_prob)
hidden2 = tf.nn.relu(tf.matmul(drop_h1, weights['h2']) + biases['h2'])
drop_h2 = dropout(hidden2, keep_prob=config.keep_prob)
hidden3 = tf.nn.relu(tf.matmul(drop_h2, weights['h3']) + biases['h3'])
#drop_h3 = dropout(hidden3, keep_prob=config.keep_prob)
pred = tf.nn.relu(tf.matmul(hidden3, weights['out']) + biases['out'])

predictions = tf.cast(tf.argmax(pred,1), tf.int64)
labels = tf.cast(tf.argmax(y,1), tf.int64)
y_p = tf.argmax(pred,1)
y_l = tf.argmax(y,1)

conf_mat = tf.contrib.metrics.confusion_matrix(labels, predictions)
#pred = tf.Print(pred, [sk.metrics.f1_score(labels,predictions, average = None)], message='f1 score', summarize = 100 )
#pred = tf.Print(pred, [pred.get_shape()], message='predictions', summarize=100)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=config.learning_rate).minimize(cost)
#optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate,epsilon=0.1).minimize(cost)
# Evaluate model
pred = tf.Print(pred, [predictions], message='prediction', summarize=10)
pred = tf.Print(pred, [labels], message='   label  ', summarize=10)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# Initializing the variables
init = tf.global_variables_initializer()


# Launch the graph
with tf.Session() as sess:

  config.printConfiguration()  
 
  tStart_total = time.time()
  sess.run(init)
  discount = 0.01
  epoch =1

  while epoch <= config.maxEpoch:
    step = 1
    while step * config.batch_size <= X_train_transformed.shape[0]:
      rnd_sample = np.random.randint(0, X_train_transformed.shape[0], config.batch_size) 
      
      batch, labels = X_train_transformed[rnd_sample,:], y_train[rnd_sample,:]
      
      sess.run(optimizer, feed_dict={x: batch, y: labels})
      
      if step % config.display_step == 0:
        
        acc = sess.run(accuracy, feed_dict={x: batch, y: labels})
        
        loss = sess.run(cost, feed_dict={x: batch, y: labels})
        
        if step == 1:
          val_moving_avg_acc = 0.0
          moving_avg_loss = loss
          moving_avg_acc = acc
        else:
          moving_avg_loss = (1.0-discount) * moving_avg_loss + discount * loss
          moving_avg_acc = (1.0-discount) * moving_avg_acc + discount * acc
          print "Iter " + str(step*config.batch_size) + ", Minibatch Loss= " + \
               "{:.6f}".format(loss) + ", Training Accuracy= " + \
               "{:.5f}".format(acc)  + ", Moving Average Loss= " + \
               "{:.6f}".format(moving_avg_loss) + ", Moving Average Acc= " + \
               "{:.5f}".format(moving_avg_acc)        

      if step % config.val_step == 0:
        val_acc = sess.run(accuracy, feed_dict={x: X_test, y: y_test})
        val_moving_avg_acc =  (1.0-discount) * val_moving_avg_acc + discount * val_acc 
        print "\nTesting Accuracy:", val_acc#, "Testing Moving Avg Accuracy:", val_moving_avg_acc
        print "Test Confmatrix: \n", \
                sess.run(conf_mat, feed_dict={x: X_test, y: y_test})
        val_pred, val_label = sess.run([y_p, y_l],feed_dict={x: X_test, y: y_test})

        report = sk.metrics.classification_report(val_label, val_pred)
        print report
        
      step += 1
    epoch +=1 
  tStop_total = time.time()
  print "Total Time Cost:", round(tStop_total - tStart_total,2), "s"
  print "Optimization Finished!"
