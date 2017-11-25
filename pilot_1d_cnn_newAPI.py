from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import pickle
from utilitiy import *
fileshape = 0

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode, params):
  """Model function for MLP."""
  
  config = params

  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  
  input_layer = tf.reshape(features["x"], [-1, features["x"].shape[1], 1, 1])
  print(input_layer.shape)
  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 256, 1, 1]
  # Output Tensor Shape: [batch_size, 252, 1, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[1, 5],
      padding="same",
      activation=tf.nn.relu)
  print(conv1.shape)
  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 252, 1, 32]
  # Output Tensor Shape: [batch_size, 126, 1, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 1], strides=[2,1])
  print(pool1.shape)
  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 126, 1, 32]
  # Output Tensor Shape: [batch_size, 122, 1, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[1, 5],
      padding="same",
      activation=tf.nn.relu)
  print(conv2.shape)
  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 122, 1, 64]
  # Output Tensor Shape: [batch_size, 61, 1, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 1], strides=[2,1])
  print(pool2.shape)
  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 61, 1, 64]
  # Output Tensor Shape: [batch_size, 61 * 1* 64]]
  pool2_flat = tf.reshape(pool2, [-1, 64 * 1 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 61 * 1* 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 93]
  logits = tf.layers.dense(inputs=dropout, units=config['nclasses'])

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=config['nclasses'])
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def my_input_fn(data_set):
  pass



def main(unused_argv):
  # Load training and eval data
  #unused_argv: a list of strings
  print ('unused_argv', unused_argv)
  filename = unused_argv[1]


  if os.path.isfile('ft_to_idx.npy') and os.path.isfile('nclasses.npy') and os.path.isfile('group_data.npy'):
    ft_to_idx = np.load('ft_to_idx.npy')
    ft_to_idx = ft_to_idx.item()
    nclasses = np.load('nclasses.npy')
    #f = open('group_data.pkl','r')
    #group_data = pickle.load('group_data.pkl')
    group_data = np.load('group_data.npy')
    group_data = group_data.item()

  else:
    ft_to_idx, nclasses, group_data = prepare_file(filename)
    np.save("ft_to_idx", ft_to_idx)
    np.save("nclasses", nclasses)
    #f = open('group_data.pkl','w')
    #pickle.dump(group_data, f)
    np.save("group_data", group_data)

  
  train_data, train_labels = np.zeros((1, group_data['application/pdf'].shape[1]-1)), np.zeros((1,))

  for i in range(1):
    tmp_data, tmp_labels = subsampled_batch(ft_to_idx, group_data, class_size=10)
    train_data = np.vstack((train_data, tmp_data))
    train_labels = np.hstack((train_labels, tmp_labels))

  
  np.delete(train_data,0,0)
  np.delete(train_labels,0,0)
  train_data = train_data.astype(np.float32)
  #train_data = tf.cast(train_data, tf.float32)
  #train_labels = tf.cast(train_labels, tf.int64)
  #print ('train type:', train_data.dtype)
  print (train_labels.dtype)
  #print ('train shape:', train_data.shape)
  

  eval_data, eval_labels = subsampled_batch(ft_to_idx, group_data, class_size=10)

  eval_data = eval_data.astype(np.float32) 
  #eval_data = tf.cast(eval_data, tf.float32)
  #eval_labels = tf.cast(eval_labels, tf.int64)

  config = {}
  config['nclasses'] = int(nclasses)
  config['model_dir'] = unused_argv[2]
  config['n_hidden1'] = int(unused_argv[3])
  config['n_hidden2'] = int(unused_argv[4])
  config['n_hidden3'] = int(unused_argv[5])
  

  # Create the Estimator
  cnn_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir=config['model_dir'], params=config)

  # Set up logging for predictions
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
  
  cnn_classifier.train(
      input_fn=train_input_fn,
      steps=20000,
      hooks=[logging_hook])
  
  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)
  
  eval_results = cnn_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()
